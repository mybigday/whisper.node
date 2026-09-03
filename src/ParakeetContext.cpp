#include "ParakeetContext.h"
#include "common.hpp"
#include "parakeet.h"
#include <thread>

// Helper class for async Parakeet transcription
class ParakeetTranscribeWorker : public Napi::AsyncWorker {
public:
    ParakeetTranscribeWorker(
        const Napi::Function& callback,
        ParakeetSessionPtr session,
        const std::vector<float>& audioData,
        const parakeet_full_params& params,
        std::shared_ptr<std::atomic<bool>> cancelFlag
    ) : AsyncWorker(callback), session_(session), audioData_(audioData), params_(params),
        cancelFlag_(cancelFlag) {}

protected:
    void Execute() override {
        if (!session_ || !session_->isValid()) {
            SetError("Invalid parakeet context");
            return;
        }

        // Handle empty audio data gracefully
        if (audioData_.empty()) {
            return;
        }

        // Lock the session to ensure thread safety
        std::lock_guard<std::mutex> lock(session_->mtx);

        if (!session_->ctx) {
            SetError("Parakeet context was destroyed");
            return;
        }

        if (cancelFlag_ && cancelFlag_->load()) {
            aborted_ = true;
            return;
        }

        // Wire the cancellation flag into the ggml abort callback so stop()
        // interrupts the computation instead of waiting for completion
        parakeet_full_params params_copy = params_;
        params_copy.abort_callback = [](void* user_data) -> bool {
            auto* flag = static_cast<std::atomic<bool>*>(user_data);
            return flag && flag->load();
        };
        params_copy.abort_callback_user_data = cancelFlag_.get();

        int result = parakeet_full(session_->ctx, params_copy, audioData_.data(), audioData_.size());

        aborted_ = cancelFlag_ && cancelFlag_->load();

        if (result != 0 && !aborted_) {
            SetError("Parakeet transcription failed");
            return;
        }
    }

    void OnOK() override {
        if (!session_ || !session_->isValid()) {
            Callback().Call({Napi::Error::New(Env(), "Context was destroyed").Value(), Env().Null()});
            return;
        }

        // Handle empty audio data case
        if (audioData_.empty()) {
            auto result = whisper_utils::createParakeetTranscribeResult(Env(), nullptr, aborted_);
            Callback().Call({Env().Null(), result});
            return;
        }

        std::lock_guard<std::mutex> lock(session_->mtx);
        auto result = whisper_utils::createParakeetTranscribeResult(Env(), session_->ctx, aborted_);

        Callback().Call({Env().Null(), result});
    }

    void OnError(const Napi::Error& error) override {
        // The completion callback expects (error, result); the base OnError
        // only passes one argument, which would leave the promise pending
        Callback().Call({error.Value(), Env().Undefined()});
    }

private:
    ParakeetSessionPtr session_;  // Hold shared pointer instead of raw pointer
    std::vector<float> audioData_;
    parakeet_full_params params_;
    std::shared_ptr<std::atomic<bool>> cancelFlag_;
    bool aborted_ = false;
};

// ParakeetContext implementation
ParakeetContext::ParakeetContext(const Napi::CallbackInfo& info) : Napi::ObjectWrap<ParakeetContext>(info) {
    Napi::Env env = info.Env();

    if (info.Length() < 1 || !info[0].IsObject()) {
        Napi::TypeError::New(env, "Expected options object").ThrowAsJavaScriptException();
        return;
    }

    auto options = info[0].As<Napi::Object>();
    std::string modelPath = whisper_utils::getString(options.Get("filePath"));
    bool useGpu = whisper_utils::getBool(options.Get("useGpu"), USE_GPU);

    if (modelPath.empty()) {
        Napi::TypeError::New(env, "Model path is required").ThrowAsJavaScriptException();
        return;
    }

    // Initialize parakeet context
    parakeet_context_params cparams = parakeet_context_default_params();
    cparams.use_gpu = useGpu;
    cparams.gpu_device = 0;

    parakeet_context* ctx = parakeet_init_from_file_with_params(modelPath.c_str(), cparams);
    if (!ctx) {
        Napi::Error::New(env, "Failed to initialize parakeet context").ThrowAsJavaScriptException();
        return;
    }

    _sess = std::make_shared<ParakeetSession>(modelPath, ctx);

    // Build metadata (persistent reference so it outlives the constructor scope)
    auto meta = Napi::Object::New(env);
    meta.Set("filePath", modelPath);
    meta.Set("useGpu", useGpu);
    _meta = Napi::Persistent(meta);
}

ParakeetContext::~ParakeetContext() {
    // Note: The worker holds a shared pointer to the session, so it stays
    // alive until any running transcription finishes
}

// Job tracking methods
int ParakeetContext::registerJob(std::shared_ptr<std::atomic<bool>> cancelFlag) {
    std::lock_guard<std::mutex> lock(_cancelMutex);
    int jobId = _nextJobId++;
    _cancelFlags[jobId] = cancelFlag;
    return jobId;
}

void ParakeetContext::unregisterJob(int jobId) {
    std::lock_guard<std::mutex> lock(_cancelMutex);
    _cancelFlags.erase(jobId);
}

void ParakeetContext::ToggleNativeLog(const Napi::CallbackInfo& info) {
    toggle_native_log(info);
}

Napi::Value ParakeetContext::ModelInfo(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 1) {
        Napi::TypeError::New(env, "Expected model path").ThrowAsJavaScriptException();
        return env.Null();
    }

    std::string path = whisper_utils::getString(info[0]);

    auto modelInfo = Napi::Object::New(env);
    modelInfo.Set("path", path);
    modelInfo.Set("type", "parakeet");

    return modelInfo;
}

void ParakeetContext::Init(Napi::Env env, Napi::Object& exports) {
    Napi::Function func = DefineClass(env, "ParakeetContext", {
        StaticMethod("toggleNativeLog", &ParakeetContext::ToggleNativeLog),
        StaticMethod("loadModelInfo", &ParakeetContext::ModelInfo),
        InstanceMethod("getModelInfo", &ParakeetContext::GetModelInfo),
        InstanceMethod("transcribeFile", &ParakeetContext::TranscribeFile),
        InstanceMethod("transcribe", &ParakeetContext::TranscribeFile),
        InstanceMethod("transcribeData", &ParakeetContext::TranscribeData),
        InstanceMethod("abortTranscribe", &ParakeetContext::AbortTranscribe),
        InstanceMethod("release", &ParakeetContext::Release),
    });

    exports.Set("ParakeetContext", func);
}

Napi::Value ParakeetContext::GetModelInfo(const Napi::CallbackInfo& info) {
    if (_meta.IsEmpty()) {
        return info.Env().Null();
    }
    return _meta.Value();
}

Napi::Value ParakeetContext::TranscribeFile(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 1) {
        Napi::TypeError::New(env, "Expected file path").ThrowAsJavaScriptException();
        return env.Null();
    }

    std::string filePath = whisper_utils::getString(info[0]);
    auto options = info.Length() >= 2 && info[1].IsObject() ?
        info[1].As<Napi::Object>() : Napi::Object::New(env);

    if (!_sess || !_sess->isValid()) {
        Napi::Error::New(env, "Invalid parakeet context").ThrowAsJavaScriptException();
        return env.Null();
    }

    // Create cancellation flag
    auto cancelFlag = std::make_shared<std::atomic<bool>>(false);
    int jobId = registerJob(cancelFlag);

    auto deferred = Napi::Promise::Deferred::New(env);

    try {
        // Load audio file
        std::vector<float> audioData = whisper_utils::loadAudioFile(filePath);

        // Create parameters
        parakeet_full_params params = whisper_utils::createParakeetParamsFromOptions(options);

        // Create async worker with cancellation support
        auto callback = Napi::Function::New(env, [deferred, this, jobId](const Napi::CallbackInfo& cbInfo) {
            // Clean up job tracking
            this->unregisterJob(jobId);

            if (cbInfo.Length() >= 2) {
                if (!cbInfo[0].IsNull()) {
                    deferred.Reject(cbInfo[0]);
                } else {
                    deferred.Resolve(cbInfo[1]);
                }
            }
        });

        auto worker = new ParakeetTranscribeWorker(callback, _sess, audioData, params, cancelFlag);
        worker->Queue();

    } catch (const std::exception& e) {
        unregisterJob(jobId);
        deferred.Reject(Napi::Error::New(env, e.what()).Value());
    }

    // Create the return object with stop and promise
    auto result = Napi::Object::New(env);

    // Create stop function
    auto stopFunction = Napi::Function::New(env, [this, jobId](const Napi::CallbackInfo& stopInfo) {
        Napi::Env env = stopInfo.Env();

        // Cancel the job directly
        {
            std::lock_guard<std::mutex> lock(this->_cancelMutex);
            auto it = this->_cancelFlags.find(jobId);
            if (it != this->_cancelFlags.end()) {
                it->second->store(true);
            }
        }

        auto deferred = Napi::Promise::Deferred::New(env);
        deferred.Resolve(env.Undefined());
        return deferred.Promise();
    });

    result.Set("stop", stopFunction);
    result.Set("promise", deferred.Promise());

    return result;
}

Napi::Value ParakeetContext::TranscribeData(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 1 || !info[0].IsArrayBuffer()) {
        Napi::TypeError::New(env, "Expected ArrayBuffer").ThrowAsJavaScriptException();
        return env.Null();
    }

    auto audioBuffer = info[0].As<Napi::ArrayBuffer>();
    auto options = info.Length() >= 2 && info[1].IsObject() ?
        info[1].As<Napi::Object>() : Napi::Object::New(env);

    if (!_sess || !_sess->isValid()) {
        Napi::Error::New(env, "Invalid parakeet context").ThrowAsJavaScriptException();
        return env.Null();
    }

    // Create cancellation flag
    auto cancelFlag = std::make_shared<std::atomic<bool>>(false);
    int jobId = registerJob(cancelFlag);

    auto deferred = Napi::Promise::Deferred::New(env);

    try {
        // Convert ArrayBuffer to float array
        std::vector<float> audioData = whisper_utils::convertAudioBufferToFloat(audioBuffer);

        // Create parameters
        parakeet_full_params params = whisper_utils::createParakeetParamsFromOptions(options);

        // Create async worker with cancellation support
        auto callback = Napi::Function::New(env, [deferred, this, jobId](const Napi::CallbackInfo& cbInfo) {
            // Clean up job tracking
            this->unregisterJob(jobId);

            if (cbInfo.Length() >= 2) {
                if (!cbInfo[0].IsNull()) {
                    deferred.Reject(cbInfo[0]);
                } else {
                    deferred.Resolve(cbInfo[1]);
                }
            }
        });

        auto worker = new ParakeetTranscribeWorker(callback, _sess, audioData, params, cancelFlag);
        worker->Queue();

    } catch (const std::exception& e) {
        unregisterJob(jobId);
        deferred.Reject(Napi::Error::New(env, e.what()).Value());
    }

    // Create the return object with stop and promise
    auto result = Napi::Object::New(env);

    // Create stop function
    auto stopFunction = Napi::Function::New(env, [this, jobId](const Napi::CallbackInfo& stopInfo) {
        Napi::Env env = stopInfo.Env();

        // Cancel the job directly
        {
            std::lock_guard<std::mutex> lock(this->_cancelMutex);
            auto it = this->_cancelFlags.find(jobId);
            if (it != this->_cancelFlags.end()) {
                it->second->store(true);
            }
        }

        auto deferred = Napi::Promise::Deferred::New(env);
        deferred.Resolve(env.Undefined());
        return deferred.Promise();
    });

    result.Set("stop", stopFunction);
    result.Set("promise", deferred.Promise());

    return result;
}

Napi::Value ParakeetContext::AbortTranscribe(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 1 || !info[0].IsNumber()) {
        Napi::TypeError::New(env, "Expected job ID").ThrowAsJavaScriptException();
        return env.Null();
    }

    int jobId = info[0].As<Napi::Number>().Int32Value();

    {
        std::lock_guard<std::mutex> lock(_cancelMutex);
        auto it = _cancelFlags.find(jobId);
        if (it != _cancelFlags.end()) {
            it->second->store(true);
        }
    }

    auto deferred = Napi::Promise::Deferred::New(env);
    deferred.Resolve(env.Undefined());
    return deferred.Promise();
}

Napi::Value ParakeetContext::Release(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    auto deferred = Napi::Promise::Deferred::New(env);

    // Cancel all running jobs
    {
        std::lock_guard<std::mutex> lock(_cancelMutex);
        for (auto& [jobId, cancelFlag] : _cancelFlags) {
            cancelFlag->store(true);
        }
        _cancelFlags.clear();
    }

    // The shared_ptr will ensure the context stays alive until any running worker finishes
    _sess.reset();
    deferred.Resolve(env.Undefined());

    return deferred.Promise();
}
