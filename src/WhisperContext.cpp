#include "WhisperContext.h"
#include "common.hpp"
#include "whisper.h"
#include "common-whisper.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <thread>
#include <future>

// Helper class for async transcription
class WhisperTranscribeWorker : public Napi::AsyncWorker {
public:
    WhisperTranscribeWorker(
        const Napi::Function& callback,
        WhisperSessionPtr session,
        const std::vector<float>& audioData,
        const whisper_full_params& params
    ) : AsyncWorker(callback), session_(session), audioData_(audioData), params_(params) {}

protected:
    void Execute() override {
        if (!session_ || !session_->isValid()) {
            SetError("Invalid whisper context");
            return;
        }

        // Handle empty audio data gracefully
        if (audioData_.empty()) {
            resultText_ = "";
            return;
        }

        // Lock the session to ensure thread safety
        std::lock_guard<std::mutex> lock(session_->mtx);
        
        if (!session_->ctx) {
            SetError("Whisper context was destroyed");
            return;
        }

        int result = whisper_full(session_->ctx, params_, audioData_.data(), audioData_.size());
        if (result != 0) {
            SetError("Transcription failed");
            return;
        }

        // Build result text
        int n_segments = whisper_full_n_segments(session_->ctx);
        std::stringstream ss;
        for (int i = 0; i < n_segments; i++) {
            ss << whisper_full_get_segment_text(session_->ctx, i);
        }
        resultText_ = ss.str();
    }

    void OnOK() override {
        if (!session_ || !session_->isValid()) {
            Callback().Call({Napi::Error::New(Env(), "Context was destroyed").Value(), Env().Null()});
            return;
        }
        
        // Handle empty audio data case
        if (audioData_.empty()) {
            auto result = whisper_utils::createTranscribeResult(Env(), nullptr, resultText_, false);
            Callback().Call({Env().Null(), result});
            return;
        }
        
        std::lock_guard<std::mutex> lock(session_->mtx);
        auto result = whisper_utils::createTranscribeResult(Env(), session_->ctx, resultText_, false);
        Callback().Call({Env().Null(), result});
    }

private:
    WhisperSessionPtr session_;  // Hold shared pointer instead of raw pointer
    std::vector<float> audioData_;
    whisper_full_params params_;
    std::string resultText_;
};

// Helper class for async VAD
class WhisperVadWorker : public Napi::AsyncWorker {
public:
    WhisperVadWorker(
        const Napi::Function& callback,
        WhisperVadSessionPtr session,
        const std::vector<float>& audioData,
        const whisper_vad_params& vadParams
    ) : AsyncWorker(callback), session_(session), audioData_(audioData), vadParams_(vadParams) {}

protected:
    void Execute() override {
        if (!session_ || !session_->isValid()) {
            SetError("Invalid VAD context");
            return;
        }

        if (audioData_.empty()) {
            SetError("Empty audio data");
            return;
        }

        // Lock the session to ensure thread safety
        std::lock_guard<std::mutex> lock(session_->mtx);
        
        if (!session_->ctx) {
            SetError("VAD context was destroyed");
            return;
        }

        // Use proper VAD detection
        hasSpeech_ = whisper_vad_detect_speech(session_->ctx, audioData_.data(), audioData_.size());

        // Calculate speech probability from VAD probabilities
        int n_probs = whisper_vad_n_probs(session_->ctx);
        float* probs = whisper_vad_probs(session_->ctx);
        
        if (n_probs > 0 && probs) {
            // Calculate average probability across all frames
            float prob_sum = 0.0f;
            for (int i = 0; i < n_probs; i++) {
                prob_sum += probs[i];
            }
            speechProbability_ = prob_sum / n_probs;
        } else {
            // Fallback: use simple binary probability based on detection
            speechProbability_ = hasSpeech_ ? 0.8f : 0.1f;
        }

        if (hasSpeech_) {
            // Get VAD segments using provided parameters
            whisper_vad_segments* segments = whisper_vad_segments_from_samples(
                session_->ctx, vadParams_, audioData_.data(), audioData_.size());
            
            if (segments) {
                int n_segments = whisper_vad_segments_n_segments(segments);
                
                for (int i = 0; i < n_segments; i++) {
                    float t0 = whisper_vad_segments_get_segment_t0(segments, i);
                    float t1 = whisper_vad_segments_get_segment_t1(segments, i);
                    
                    segments_.push_back({t0, t1});
                }
                
                whisper_vad_free_segments(segments);
            }
        }
    }

    void OnOK() override {
        Napi::Env env = Env();
        Napi::Array result = Napi::Array::New(env);
        
        // Create VadSegment[] - array of objects with t0 and t1 properties
        for (size_t i = 0; i < segments_.size(); i++) {
            Napi::Object segment = Napi::Object::New(env);
            segment.Set("t0", segments_[i].first);
            segment.Set("t1", segments_[i].second);
            result.Set(i, segment);
        }
        
        Callback().Call({Env().Null(), result});
    }

    void OnError(const Napi::Error& error) override {
        Callback().Call({error.Value(), Env().Undefined()});
    }

private:
    WhisperVadSessionPtr session_;  // Hold shared pointer instead of raw pointer
    std::vector<float> audioData_;
    whisper_vad_params vadParams_;
    bool hasSpeech_ = false;
    float speechProbability_ = 0.0f;
    std::vector<std::pair<float, float>> segments_;  // Changed to float for time values
};

// WhisperContext implementation
WhisperContext::WhisperContext(const Napi::CallbackInfo& info) : Napi::ObjectWrap<WhisperContext>(info) {
    Napi::Env env = info.Env();

    if (info.Length() < 1 || !info[0].IsObject()) {
        Napi::TypeError::New(env, "Expected options object").ThrowAsJavaScriptException();
        return;
    }

    auto options = info[0].As<Napi::Object>();
    std::string modelPath = whisper_utils::getString(options.Get("filePath"));
    bool useGpu = whisper_utils::getBool(options.Get("useGpu"), true);
    bool useFlashAttn = whisper_utils::getBool(options.Get("useFlashAttn"), false);

    if (modelPath.empty()) {
        Napi::TypeError::New(env, "Model path is required").ThrowAsJavaScriptException();
        return;
    }

    // Initialize whisper context
    whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = useGpu;
    cparams.flash_attn = useFlashAttn;

    whisper_context* ctx = whisper_init_from_file_with_params(modelPath.c_str(), cparams);
    if (!ctx) {
        Napi::Error::New(env, "Failed to initialize whisper context").ThrowAsJavaScriptException();
        return;
    }

    _sess = std::make_shared<WhisperSession>(modelPath, ctx);

    // Build metadata
    _meta = Napi::Object::New(env);
    _meta.Set("filePath", modelPath);
    _meta.Set("useGpu", useGpu);
    _meta.Set("useFlashAttn", useFlashAttn);
}

WhisperContext::~WhisperContext() {
    // Note: Don't delete _wip here as it's managed by Node.js async worker lifecycle
    // The worker will clean itself up when it completes
}

void WhisperContext::ToggleNativeLog(const Napi::CallbackInfo& info) {
    if (info.Length() < 1) return;

    bool enable = whisper_utils::getBool(info[0], false);
    g_log_enabled = enable;
}

Napi::Value WhisperContext::ModelInfo(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 1) {
        Napi::TypeError::New(env, "Expected model path").ThrowAsJavaScriptException();
        return env.Null();
    }

    std::string path = whisper_utils::getString(info[0]);

    auto modelInfo = Napi::Object::New(env);
    modelInfo.Set("path", path);
    modelInfo.Set("type", "whisper");

    return modelInfo;
}

void WhisperContext::Init(Napi::Env env, Napi::Object& exports) {
    Napi::Function func = DefineClass(env, "WhisperContext", {
        StaticMethod("toggleNativeLog", &WhisperContext::ToggleNativeLog),
        StaticMethod("loadModelInfo", &WhisperContext::ModelInfo),
        InstanceMethod("getModelInfo", &WhisperContext::GetModelInfo),
        InstanceMethod("transcribeFile", &WhisperContext::TranscribeFile),
        InstanceMethod("transcribe", &WhisperContext::TranscribeFile),
        InstanceMethod("transcribeData", &WhisperContext::TranscribeData),
        InstanceMethod("release", &WhisperContext::Release),
    });

    exports.Set("WhisperContext", func);
}


Napi::Value WhisperContext::GetModelInfo(const Napi::CallbackInfo& info) {
    return _meta;
}

Napi::Value WhisperContext::TranscribeFile(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 1) {
        Napi::TypeError::New(env, "Expected file path").ThrowAsJavaScriptException();
        return env.Null();
    }

    std::string filePath = whisper_utils::getString(info[0]);
    auto options = info.Length() >= 2 && info[1].IsObject() ?
        info[1].As<Napi::Object>() : Napi::Object::New(env);

    if (!_sess || !_sess->isValid()) {
        Napi::Error::New(env, "Invalid whisper context").ThrowAsJavaScriptException();
        return env.Null();
    }

    auto deferred = Napi::Promise::Deferred::New(env);

    try {
        // Load audio file
        std::vector<float> audioData = whisper_utils::loadAudioFile(filePath);

        // Create parameters
        whisper_full_params params = whisper_utils::createFullParamsFromOptions(options);

        // Create async worker - pass shared pointer to session instead of raw pointer
        auto callback = Napi::Function::New(env, [deferred](const Napi::CallbackInfo& cbInfo) {
            if (cbInfo.Length() >= 2) {
                if (!cbInfo[0].IsNull()) {
                    deferred.Reject(cbInfo[0]);
                } else {
                    deferred.Resolve(cbInfo[1]);
                }
            }
        });

        auto worker = new WhisperTranscribeWorker(callback, _sess, audioData, params);
        worker->Queue();

        // Don't store the worker pointer since it's managed by Node.js
        // and will clean itself up when it completes

    } catch (const std::exception& e) {
        deferred.Reject(Napi::Error::New(env, e.what()).Value());
    }

    return deferred.Promise();
}

Napi::Value WhisperContext::TranscribeData(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 1 || !info[0].IsArrayBuffer()) {
        Napi::TypeError::New(env, "Expected ArrayBuffer").ThrowAsJavaScriptException();
        return env.Null();
    }

    auto audioBuffer = info[0].As<Napi::ArrayBuffer>();
    auto options = info.Length() >= 2 && info[1].IsObject() ?
        info[1].As<Napi::Object>() : Napi::Object::New(env);

    if (!_sess || !_sess->isValid()) {
        Napi::Error::New(env, "Invalid whisper context").ThrowAsJavaScriptException();
        return env.Null();
    }

    auto deferred = Napi::Promise::Deferred::New(env);

    try {
        // Convert ArrayBuffer to float array
        std::vector<float> audioData = whisper_utils::convertAudioBufferToFloat(audioBuffer);

        // Create parameters
        whisper_full_params params = whisper_utils::createFullParamsFromOptions(options);

        // Create async worker - pass shared pointer to session instead of raw pointer
        auto callback = Napi::Function::New(env, [deferred](const Napi::CallbackInfo& cbInfo) {
            if (cbInfo.Length() >= 2) {
                if (!cbInfo[0].IsNull()) {
                    deferred.Reject(cbInfo[0]);
                } else {
                    deferred.Resolve(cbInfo[1]);
                }
            }
        });

        auto worker = new WhisperTranscribeWorker(callback, _sess, audioData, params);
        worker->Queue();

        // Don't store the worker pointer since it's managed by Node.js
        // and will clean itself up when it completes

    } catch (const std::exception& e) {
        deferred.Reject(Napi::Error::New(env, e.what()).Value());
    }

    return deferred.Promise();
}

Napi::Value WhisperContext::Release(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    auto deferred = Napi::Promise::Deferred::New(env);

    // The shared_ptr will ensure the context stays alive until any running worker finishes
    
    _sess.reset();
    deferred.Resolve(env.Undefined());

    return deferred.Promise();
}

// WhisperVadContext implementation
WhisperVadContext::WhisperVadContext(const Napi::CallbackInfo& info) : Napi::ObjectWrap<WhisperVadContext>(info) {
    Napi::Env env = info.Env();

    if (info.Length() < 1 || !info[0].IsObject()) {
        Napi::TypeError::New(env, "Expected options object").ThrowAsJavaScriptException();
        return;
    }

    auto options = info[0].As<Napi::Object>();
    std::string modelPath = whisper_utils::getString(options.Get("filePath"));
    bool useGpu = whisper_utils::getBool(options.Get("useGpu"), true);
    int nThreads = whisper_utils::getInt(options.Get("nThreads"), std::thread::hardware_concurrency());

    if (modelPath.empty()) {
        Napi::TypeError::New(env, "Model path is required").ThrowAsJavaScriptException();
        return;
    }

    // Initialize VAD context with proper parameters
    whisper_vad_context_params vparams = whisper_vad_default_context_params();
    vparams.use_gpu = useGpu;
    vparams.n_threads = nThreads;

    whisper_vad_context* ctx = whisper_vad_init_from_file_with_params(modelPath.c_str(), vparams);
    if (!ctx) {
        Napi::Error::New(env, "Failed to initialize whisper vad context").ThrowAsJavaScriptException();
        return;
    }
    _sess = std::make_shared<WhisperVadSession>(modelPath, ctx);

    // Build metadata
    _meta = Napi::Object::New(env);
    _meta.Set("filePath", modelPath);
    _meta.Set("useGpu", useGpu);
    _meta.Set("nThreads", nThreads);
}

WhisperVadContext::~WhisperVadContext() {
}

void WhisperVadContext::ToggleNativeLog(const Napi::CallbackInfo& info) {
    if (info.Length() < 1) return;

    bool enable = whisper_utils::getBool(info[0], false);
    g_log_enabled = enable;
}

Napi::Value WhisperVadContext::ModelInfo(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 1) {
        Napi::TypeError::New(env, "Expected model path").ThrowAsJavaScriptException();
        return env.Null();
    }

    std::string path = whisper_utils::getString(info[0]);

    auto modelInfo = Napi::Object::New(env);
    modelInfo.Set("path", path);
    modelInfo.Set("type", "whisper_vad");

    return modelInfo;
}

void WhisperVadContext::Init(Napi::Env env, Napi::Object& exports) {
    Napi::Function func = DefineClass(env, "WhisperVadContext", {
        StaticMethod("toggleNativeLog", &WhisperVadContext::ToggleNativeLog),
        StaticMethod("loadModelInfo", &WhisperVadContext::ModelInfo),
        InstanceMethod("getModelInfo", &WhisperVadContext::GetModelInfo),
        InstanceMethod("detectSpeechFile", &WhisperVadContext::DetectSpeechFile),
        InstanceMethod("detectSpeech", &WhisperVadContext::DetectSpeechFile),
        InstanceMethod("detectSpeechData", &WhisperVadContext::DetectSpeechData),
        InstanceMethod("release", &WhisperVadContext::Release),
    });

    exports.Set("WhisperVadContext", func);
}

Napi::Value WhisperVadContext::GetModelInfo(const Napi::CallbackInfo& info) {
    return _meta;
}

Napi::Value WhisperVadContext::DetectSpeechFile(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 1) {
        Napi::TypeError::New(env, "Expected file path").ThrowAsJavaScriptException();
        return env.Null();
    }

    std::string filePath = whisper_utils::getString(info[0]);
    auto options = info.Length() >= 2 && info[1].IsObject() ?
        info[1].As<Napi::Object>() : Napi::Object::New(env);

    auto deferred = Napi::Promise::Deferred::New(env);

    try {
        // Load audio file
        std::vector<float> audioData = whisper_utils::loadAudioFile(filePath);

        // Create VAD parameters
        whisper_vad_params vadParams = whisper_utils::createVadParamsFromOptions(options);

        // Create async worker - pass shared pointer to session
        auto callback = Napi::Function::New(env, [deferred](const Napi::CallbackInfo& cbInfo) {
            if (cbInfo.Length() >= 2) {
                if (!cbInfo[0].IsNull()) {
                    deferred.Reject(cbInfo[0]);
                } else {
                    deferred.Resolve(cbInfo[1]);
                }
            }
        });

        auto worker = new WhisperVadWorker(callback, _sess, audioData, vadParams);
        worker->Queue();

    } catch (const std::exception& e) {
        deferred.Reject(Napi::Error::New(env, e.what()).Value());
    }

    return deferred.Promise();
}

Napi::Value WhisperVadContext::DetectSpeechData(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 1 || !info[0].IsArrayBuffer()) {
        Napi::TypeError::New(env, "Expected ArrayBuffer").ThrowAsJavaScriptException();
        return env.Null();
    }

    auto audioBuffer = info[0].As<Napi::ArrayBuffer>();
    auto options = info.Length() >= 2 && info[1].IsObject() ?
        info[1].As<Napi::Object>() : Napi::Object::New(env);

    auto deferred = Napi::Promise::Deferred::New(env);

    try {
        // Convert ArrayBuffer to float array
        std::vector<float> audioData = whisper_utils::convertAudioBufferToFloat(audioBuffer);

        // Create VAD parameters
        whisper_vad_params vadParams = whisper_utils::createVadParamsFromOptions(options);

        // Create async worker - pass shared pointer to session
        auto callback = Napi::Function::New(env, [deferred](const Napi::CallbackInfo& cbInfo) {
            if (cbInfo.Length() >= 2) {
                if (!cbInfo[0].IsNull()) {
                    deferred.Reject(cbInfo[0]);
                } else {
                    deferred.Resolve(cbInfo[1]);
                }
            }
        });

        auto worker = new WhisperVadWorker(callback, _sess, audioData, vadParams);
        worker->Queue();

    } catch (const std::exception& e) {
        deferred.Reject(Napi::Error::New(env, e.what()).Value());
    }

    return deferred.Promise();
}

Napi::Value WhisperVadContext::Release(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    auto deferred = Napi::Promise::Deferred::New(env);

    _sess.reset();
    deferred.Resolve(env.Undefined());

    return deferred.Promise();
}
