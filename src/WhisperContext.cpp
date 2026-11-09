#include "WhisperContext.h"
#include "common.hpp"
#include "whisper.h"
#include "common-whisper.h"
#include "ggml-backend.h"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <fstream>
#include <sstream>
#include <thread>
#include <future>
#include <condition_variable>

// Callback context for progress and new segments
struct TranscribeCallbackContext {
    Napi::ThreadSafeFunction tsfnProgress;
    Napi::ThreadSafeFunction tsfnNewSegments;
    std::shared_ptr<std::atomic<bool>> cancelFlag;
    int totalNNew = 0;
    bool tdrzEnable = false;
    std::atomic<int> pendingCallbacks{0};
    std::mutex callbackMutex;
    std::condition_variable callbackCV;
};

// Helper class for async transcription
class WhisperTranscribeWorker : public Napi::AsyncWorker {
public:
    WhisperTranscribeWorker(
        const Napi::Function& callback,
        WhisperSessionPtr session,
        const std::vector<float>& audioData,
        const whisper_full_params& params,
        int nProcessors,
        std::shared_ptr<std::atomic<bool>> cancelFlag,
        Napi::ThreadSafeFunction tsfnProgress = Napi::ThreadSafeFunction(),
        Napi::ThreadSafeFunction tsfnNewSegments = Napi::ThreadSafeFunction(),
        bool hasProgress = false,
        bool hasNewSegments = false,
        std::string language = "",
        std::string prompt = ""
    ) : AsyncWorker(callback), session_(session), audioData_(audioData), params_(params), nProcessors_(nProcessors),
        cancelFlag_(cancelFlag), tsfnProgress_(tsfnProgress), tsfnNewSegments_(tsfnNewSegments),
        hasProgress_(hasProgress), hasNewSegments_(hasNewSegments), language_(language), prompt_(prompt) {
            // Apply language and prompt from stored strings to ensure valid lifetime
            if (!language_.empty()) {
                params_.language = language_.c_str();
            }
            if (!prompt_.empty()) {
                params_.initial_prompt = prompt_.c_str();
            }
        }

protected:
    void Execute() override {
        if (!session_ || !session_->isValid()) {
            SetError("Invalid whisper context");
            return;
        }

        // Check if cancelled before starting
        if (cancelFlag_ && cancelFlag_->load()) {
            aborted_ = true;
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

        // Check if cancelled before processing
        if (cancelFlag_ && cancelFlag_->load()) {
            aborted_ = true;
            return;
        }

        // Create a copy of params and set up callbacks if needed
        whisper_full_params params_copy = params_;
        auto callbackCtxPtr = std::make_shared<TranscribeCallbackContext>();
        TranscribeCallbackContext& callbackCtx = *callbackCtxPtr;
        callbackCtx.tsfnProgress = tsfnProgress_;
        callbackCtx.tsfnNewSegments = tsfnNewSegments_;
        callbackCtx.cancelFlag = cancelFlag_;
        callbackCtx.totalNNew = 0;
        callbackCtx.tdrzEnable = params_.tdrz_enable;

        // Wire the cancellation flag into the ggml abort callback so stop()
        // interrupts the computation instead of waiting for completion
        if (cancelFlag_) {
            params_copy.abort_callback = [](void* user_data) -> bool {
                auto* flag = static_cast<std::atomic<bool>*>(user_data);
                return flag && flag->load();
            };
            params_copy.abort_callback_user_data = cancelFlag_.get();
        }

        if (hasProgress_) {
            params_copy.progress_callback = [](struct whisper_context* /*ctx*/, struct whisper_state* /*state*/, int progress, void* user_data) {
                auto cbCtx = *static_cast<std::shared_ptr<TranscribeCallbackContext>*>(user_data);
                if (cbCtx->cancelFlag && cbCtx->cancelFlag->load()) {
                    return;
                }
                if (cbCtx->tsfnProgress) {
                    cbCtx->pendingCallbacks++;
                    auto status = cbCtx->tsfnProgress.NonBlockingCall([cbCtx, progress](Napi::Env env, Napi::Function jsCallback) {
                        jsCallback.Call({Napi::Number::New(env, progress)});
                        cbCtx->pendingCallbacks--;
                        cbCtx->callbackCV.notify_one();
                    });
                    if (status != napi_ok) {
                        cbCtx->pendingCallbacks--;
                    }
                }
            };
            params_copy.progress_callback_user_data = &callbackCtxPtr;
        }

        if (hasNewSegments_) {
            params_copy.new_segment_callback = [](struct whisper_context* ctx, struct whisper_state* /*state*/, int n_new, void* user_data) {
                auto cbCtx = *static_cast<std::shared_ptr<TranscribeCallbackContext>*>(user_data);
                if (cbCtx->cancelFlag && cbCtx->cancelFlag->load()) {
                    return;
                }
                cbCtx->totalNNew += n_new;

                if (cbCtx->tsfnNewSegments) {
                    // Capture values needed for the callback
                    int totalNNew = cbCtx->totalNNew;
                    bool tdrzEnable = cbCtx->tdrzEnable;

                    cbCtx->pendingCallbacks++;
                    auto status = cbCtx->tsfnNewSegments.NonBlockingCall([cbCtx, ctx, n_new, totalNNew, tdrzEnable](Napi::Env env, Napi::Function jsCallback) {
                        // the context may already be freed once the job was cancelled
                        if (cbCtx->cancelFlag && cbCtx->cancelFlag->load()) {
                            cbCtx->pendingCallbacks--;
                            cbCtx->callbackCV.notify_one();
                            return;
                        }

                        std::string text = "";
                        Napi::Array segments = Napi::Array::New(env);

                        for (int i = totalNNew - n_new; i < totalNNew; i++) {
                            const char* text_cur = whisper_full_get_segment_text(ctx, i);
                            std::string segment_text = text_cur;

                            if (tdrzEnable && whisper_full_get_segment_speaker_turn_next(ctx, i)) {
                                segment_text += " [SPEAKER_TURN]";
                            }

                            text += segment_text;

                            Napi::Object segment = Napi::Object::New(env);
                            segment.Set("text", segment_text);
                            segment.Set("t0", whisper_full_get_segment_t0(ctx, i) * 10);
                            segment.Set("t1", whisper_full_get_segment_t1(ctx, i) * 10);
                            segments.Set(static_cast<uint32_t>(i - (totalNNew - n_new)), segment);
                        }

                        Napi::Object result = Napi::Object::New(env);
                        result.Set("nNew", n_new);
                        result.Set("totalNNew", totalNNew);
                        result.Set("result", text);
                        result.Set("segments", segments);

                        jsCallback.Call({result});

                        cbCtx->pendingCallbacks--;
                        cbCtx->callbackCV.notify_one();
                    });
                    if (status != napi_ok) {
                        cbCtx->pendingCallbacks--;
                    }
                }
            };
            params_copy.new_segment_callback_user_data = &callbackCtxPtr;
        }

        int result = whisper_full_parallel(session_->ctx, params_copy, audioData_.data(), audioData_.size(), nProcessors_);

        // Wait for all pending callbacks to complete before returning, unless the job was cancelled:
        // the pending callback may be the one that cancelled us and it will not return before we do
        {
            std::unique_lock<std::mutex> lock(callbackCtx.callbackMutex);
            callbackCtx.callbackCV.wait(lock, [&callbackCtx] {
                return callbackCtx.pendingCallbacks.load() == 0 ||
                       (callbackCtx.cancelFlag && callbackCtx.cancelFlag->load());
            });
        }

        // A cancelled job resolves with isAborted=true and the segments decoded so far
        aborted_ = cancelFlag_ && cancelFlag_->load();

        if (result != 0 && !aborted_) {
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
            CleanupCallbacks();
            Callback().Call({Napi::Error::New(Env(), "Context was destroyed").Value(), Env().Null()});
            return;
        }

        // Handle empty audio data / cancelled-before-start cases
        if (audioData_.empty() || (aborted_ && resultText_.empty())) {
            CleanupCallbacks();
            auto result = whisper_utils::createTranscribeResult(Env(), nullptr, resultText_, aborted_);
            Callback().Call({Env().Null(), result});
            return;
        }

        std::lock_guard<std::mutex> lock(session_->mtx);
        auto result = whisper_utils::createTranscribeResult(Env(), session_->ctx, resultText_, aborted_);

        // Clean up callbacks BEFORE calling completion callback
        // This ensures BlockingCall callbacks complete before the promise resolves
        CleanupCallbacks();

        Callback().Call({Env().Null(), result});
    }

    void OnError(const Napi::Error& error) override {
        CleanupCallbacks();
        // The completion callback expects (error, result); the base OnError
        // only passes one argument, which would leave the promise pending
        Callback().Call({error.Value(), Env().Undefined()});
    }

    void CleanupCallbacks() {
        // Release thread-safe functions if they exist
        if (hasProgress_ && tsfnProgress_) {
            tsfnProgress_.Release();
        }
        if (hasNewSegments_ && tsfnNewSegments_) {
            tsfnNewSegments_.Release();
        }
    }

private:
    WhisperSessionPtr session_;  // Hold shared pointer instead of raw pointer
    std::vector<float> audioData_;
    whisper_full_params params_;
    int nProcessors_;
    std::string resultText_;
    std::shared_ptr<std::atomic<bool>> cancelFlag_;
    Napi::ThreadSafeFunction tsfnProgress_;
    Napi::ThreadSafeFunction tsfnNewSegments_;
    bool hasProgress_;
    bool hasNewSegments_;
    std::string language_;
    std::string prompt_;
    bool aborted_ = false;
};

// Ends a registered VAD call (WhisperVadSession::retain) while `mtx` is still
// held: declare it right after the lock. May free the session when release()
// is waiting for this call.
struct VadCallGuard {
    WhisperVadSession* session;
    explicit VadCallGuard(WhisperVadSession* s) : session(s) {}
    ~VadCallGuard() { session->releaseLocked(); }
};

// Offline VAD: speech probabilities for the whole buffer (a fresh vad.cpp
// stream, so the Silero state spans the input), then silero-vad's segmentation.
// Segment times are in 10 ms units, as with whisper.cpp's VAD before.
class WhisperVadWorker : public Napi::AsyncWorker {
public:
    WhisperVadWorker(
        const Napi::Function& callback,
        WhisperVadSessionPtr session,
        std::vector<float>&& audioData,
        const vad_segment_params& vadParams
    ) : AsyncWorker(callback), session_(session), audioData_(std::move(audioData)), vadParams_(vadParams),
        retained_(session && session->retain()) {}

protected:
    void Execute() override {
        if (!session_ || !retained_) {
            SetError("Invalid VAD context");
            return;
        }

        std::lock_guard<std::mutex> lock(session_->mtx);
        VadCallGuard guard(session_.get());
        session_->freeOrphansLocked();
        if (!session_->ctx) {
            SetError("VAD context was destroyed");
            return;
        }
        if (audioData_.empty()) {
            return;
        }

        const int hop = session_->hopSamples;
        const int nSamples = static_cast<int>(audioData_.size());
        std::vector<float> probs(static_cast<size_t>(nSamples / hop + 1));
        int nProbs = vad_detect(session_->ctx, audioData_.data(), nSamples, probs.data(), static_cast<int>(probs.size()));
        if (nProbs < 0) {
            SetError("VAD detection failed");
            return;
        }
        nProbs = std::min(nProbs, static_cast<int>(probs.size()));

        std::vector<vad_segment> segments(16);
        for (;;) {
            int n = vad_segments_from_probs(
                probs.data(), nProbs, 1, hop, session_->sampleRate, nSamples,
                vadParams_, segments.data(), static_cast<int>(segments.size()));
            const bool fits = n <= static_cast<int>(segments.size());
            segments.resize(static_cast<size_t>(n));
            if (fits) break;
        }

        const double toCentiseconds = 100.0 / session_->sampleRate;
        segments_.reserve(segments.size());
        for (const auto& segment : segments) {
            segments_.push_back({
                static_cast<float>(segment.start * toCentiseconds),
                static_cast<float>(segment.end * toCentiseconds),
            });
        }
    }

    void OnOK() override {
        Napi::Env env = Env();
        Napi::Array result = Napi::Array::New(env, segments_.size());
        for (size_t i = 0; i < segments_.size(); i++) {
            Napi::Object segment = Napi::Object::New(env);
            segment.Set("t0", segments_[i].first);
            segment.Set("t1", segments_[i].second);
            result.Set(i, segment);
        }
        Callback().Call({env.Null(), result});
    }

    void OnError(const Napi::Error& error) override {
        Callback().Call({error.Value(), Env().Undefined()});
    }

private:
    WhisperVadSessionPtr session_;
    std::vector<float> audioData_;
    vad_segment_params vadParams_;
    bool retained_;
    std::vector<std::pair<float, float>> segments_;
};

// WhisperVadContext.createStream()
class WhisperVadStreamCreateWorker : public Napi::AsyncWorker {
public:
    WhisperVadStreamCreateWorker(Napi::Env env, WhisperVadSessionPtr session)
        : AsyncWorker(env), session_(session), retained_(session && session->retain()),
          deferred_(Napi::Promise::Deferred::New(env)) {}

    Napi::Promise Promise() { return deferred_.Promise(); }

protected:
    void Execute() override {
        if (!session_ || !retained_) {
            SetError("Invalid VAD context");
            return;
        }
        std::lock_guard<std::mutex> lock(session_->mtx);
        VadCallGuard guard(session_.get());
        session_->freeOrphansLocked();
        if (!session_->ctx) {
            SetError("VAD context was destroyed");
            return;
        }
        state_ = std::make_shared<VadStreamState>();
        state_->stream = vad_stream_new(session_->ctx);
        if (!state_->stream) {
            SetError("Failed to create a VAD stream");
            return;
        }
        session_->streams.insert(state_);
    }

    void OnOK() override {
        deferred_.Resolve(WhisperVadStream::NewInstance(Env(), session_, state_));
    }

    void OnError(const Napi::Error& error) override { deferred_.Reject(error.Value()); }

private:
    WhisperVadSessionPtr session_;
    bool retained_;
    std::shared_ptr<VadStreamState> state_;
    Napi::Promise::Deferred deferred_;
};

// One call on a WhisperVadStream. Started by the stream's FIFO; starts the
// next call once settled.
class WhisperVadStreamWorker : public Napi::AsyncWorker {
public:
    enum class Op { Push, Flush, Reset, Release };

    WhisperVadStreamWorker(
        Napi::Env env,
        Napi::Object streamObject,
        WhisperVadSessionPtr session,
        std::shared_ptr<VadStreamState> state,
        Op op,
        std::vector<float>&& audio = {}
    ) : AsyncWorker(env),
        streamRef_(Napi::Persistent(streamObject)),
        session_(session),
        state_(state),
        op_(op),
        audio_(std::move(audio)),
        retained_(session && session->retain()),
        deferred_(Napi::Promise::Deferred::New(env)) {}

    Napi::Promise Promise() { return deferred_.Promise(); }

protected:
    void Execute() override {
        if (!session_ || !state_ || !retained_) {
            // Not retained: the context was released, and its streams with it
            if (op_ != Op::Release) SetError("VAD stream not found");
            return;
        }
        std::lock_guard<std::mutex> lock(session_->mtx);
        VadCallGuard guard(session_.get());
        session_->freeOrphansLocked();
        vad_stream* stream = state_->stream;
        if (!stream) {
            // Released, or freed with its context
            if (op_ != Op::Release) SetError("VAD stream not found");
            return;
        }
        switch (op_) {
            case Op::Push: {
                const int nSamples = static_cast<int>(audio_.size());
                const int maxProbs = std::max(vad_stream_n_pending(stream, nSamples), 0);
                probs_.resize(static_cast<size_t>(maxProbs));
                int n = vad_stream_push(stream, audio_.data(), nSamples, probs_.data(), maxProbs);
                if (n < 0) {
                    SetError("VAD stream push failed");
                    return;
                }
                probs_.resize(static_cast<size_t>(n));
                break;
            }
            case Op::Flush: {
                const int maxProbs = std::max(vad_stream_n_flush(stream), 0);
                probs_.resize(static_cast<size_t>(maxProbs));
                int n = vad_stream_flush(stream, probs_.data(), maxProbs);
                if (n < 0) {
                    SetError("VAD stream flush failed");
                    return;
                }
                probs_.resize(static_cast<size_t>(n));
                break;
            }
            case Op::Reset:
                vad_stream_reset(stream);
                break;
            case Op::Release:
                vad_stream_free(stream);
                state_->stream = nullptr;
                session_->streams.erase(state_);
                break;
        }
    }

    void OnOK() override {
        Napi::Env env = Env();
        if (op_ == Op::Push || op_ == Op::Flush) {
            Napi::Array result = Napi::Array::New(env, probs_.size());
            for (size_t i = 0; i < probs_.size(); i++) {
                result.Set(i, Napi::Number::New(env, probs_[i]));
            }
            deferred_.Resolve(result);
        } else {
            deferred_.Resolve(env.Undefined());
        }
        finish();
    }

    void OnError(const Napi::Error& error) override {
        deferred_.Reject(error.Value());
        finish();
    }

private:
    void finish() {
        WhisperVadStream::Unwrap(streamRef_.Value())->runNext();
        streamRef_.Reset();
    }

    Napi::ObjectReference streamRef_;  // keeps the stream object alive until settled
    WhisperVadSessionPtr session_;
    std::shared_ptr<VadStreamState> state_;
    Op op_;
    std::vector<float> audio_;
    bool retained_;
    std::vector<float> probs_;
    Napi::Promise::Deferred deferred_;
};

// WhisperContext implementation
static LiveContexts<WhisperContext> g_whisper_contexts;
static LiveContexts<WhisperVadContext> g_vad_contexts;

WhisperContext::WhisperContext(const Napi::CallbackInfo& info) : Napi::ObjectWrap<WhisperContext>(info) {
    Napi::Env env = info.Env();

    if (info.Length() < 1 || !info[0].IsObject()) {
        Napi::TypeError::New(env, "Expected options object").ThrowAsJavaScriptException();
        return;
    }

    auto options = info[0].As<Napi::Object>();
    std::string modelPath = whisper_utils::getString(options.Get("filePath"));
    bool useGpu = whisper_utils::getBool(options.Get("useGpu"), USE_GPU);
    bool useFlashAttn = whisper_utils::getBool(options.Get("useFlashAttn"), false);

    if (modelPath.empty()) {
        Napi::TypeError::New(env, "Model path is required").ThrowAsJavaScriptException();
        return;
    }

    // The Hexagon NPU (snapdragon variant) only runs the flash-attention graph:
    // whisper.rn's ggml-hexagon patch writes the KV caches with ggml_set_rows
    // there, while the non-FA graph uses an f32->f16 ggml_cpy that HTP cannot
    // execute (ggml-backend asserts at whisper_init_state). Always on when the
    // model is offloaded to HTP, matching whisper.rn.
    if (useGpu && !useFlashAttn && ggml_backend_reg_by_name("HTP") != nullptr) {
        useFlashAttn = true;
    }

    // Initialize whisper context
    whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = useGpu;
    cparams.gpu_device = 0;
    cparams.flash_attn = useFlashAttn;

    whisper_context* ctx = whisper_init_from_file_with_params(modelPath.c_str(), cparams);
    if (!ctx) {
        Napi::Error::New(env, "Failed to initialize whisper context").ThrowAsJavaScriptException();
        return;
    }

    _sess = std::make_shared<WhisperSession>(modelPath, ctx);

    // Build metadata (persistent reference so it outlives the constructor scope)
    auto meta = Napi::Object::New(env);
    meta.Set("filePath", modelPath);
    meta.Set("useGpu", useGpu);
    meta.Set("useFlashAttn", useFlashAttn);
    _meta = Napi::Persistent(meta);

    g_whisper_contexts.add(this);
}

WhisperContext::~WhisperContext() {
    g_whisper_contexts.remove(this);

    // Note: Don't delete _wip here as it's managed by Node.js async worker lifecycle
    // The worker will clean itself up when it completes
}

// Job tracking methods
int WhisperContext::registerJob(std::shared_ptr<std::atomic<bool>> cancelFlag) {
    std::lock_guard<std::mutex> lock(_cancelMutex);
    int jobId = _nextJobId++;
    _cancelFlags[jobId] = cancelFlag;
    return jobId;
}

void WhisperContext::unregisterJob(int jobId) {
    std::lock_guard<std::mutex> lock(_cancelMutex);
    _cancelFlags.erase(jobId);
}

bool WhisperContext::isJobCancelled(int jobId) {
    std::lock_guard<std::mutex> lock(_cancelMutex);
    auto it = _cancelFlags.find(jobId);
    return it != _cancelFlags.end() && it->second->load();
}

// Static JavaScript callback function for logging
static Napi::ThreadSafeFunction g_js_log_callback;

struct NativeLogMessage {
    std::string level;
    std::string text;
};

const char* whisper_log_level_to_string(ggml_log_level level) {
    switch (level) {
        case GGML_LOG_LEVEL_ERROR:
            return "error";
        case GGML_LOG_LEVEL_WARN:
            return "warn";
        case GGML_LOG_LEVEL_INFO:
            return "info";
        case GGML_LOG_LEVEL_DEBUG:
            return "debug";
        default:
            return "";
    }
}

// C++ wrapper function that calls the JavaScript callback
void whisper_log_callback_js(const char* level, const char* text) {
    if (g_js_log_callback) {
        auto* data = new NativeLogMessage{level ? level : "", text ? text : ""};
        auto status = g_js_log_callback.BlockingCall(data, [](Napi::Env env, Napi::Function jsCallback, NativeLogMessage* data) {
            jsCallback.Call({Napi::String::New(env, data->level), Napi::String::New(env, data->text)});
            delete data;
        });
        if (status != napi_ok) {
            delete data;
        }
    }
}

void whisper_native_log_callback(ggml_log_level level, const char* text, void*) {
    if (!g_log_enabled || !g_log_callback) {
        return;
    }

    g_log_callback(whisper_log_level_to_string(level), text ? text : "");
}

// Function to clean up JavaScript logging callback
void cleanup_js_log_callback() {
    if (g_js_log_callback) {
        g_js_log_callback.Release();
        g_js_log_callback = nullptr;
    }
}

// Shared implementation for the static toggleNativeLog methods
void toggle_native_log(const Napi::CallbackInfo& info) {
    if (info.Length() < 1) return;

    bool enable = whisper_utils::getBool(info[0], false);

    if (enable) {
        cleanup_js_log_callback();
        if (info.Length() >= 2 && info[1].IsFunction()) {
            auto callback = info[1].As<Napi::Function>();
            g_js_log_callback = Napi::ThreadSafeFunction::New(
                info.Env(),
                callback,
                "whisper_log_callback",
                0,
                1
            );
            g_log_callback = whisper_log_callback_js;
            whisper_log_set(whisper_native_log_callback, nullptr);
            parakeet_log_set(whisper_native_log_callback, nullptr);
        } else {
            g_log_callback = nullptr;
            whisper_log_set(nullptr, nullptr);
            parakeet_log_set(nullptr, nullptr);
        }
        g_log_enabled = true;
    } else {
        g_log_enabled = false;
        g_log_callback = nullptr;
        whisper_log_set(nullptr, nullptr);
        parakeet_log_set(nullptr, nullptr);
        cleanup_js_log_callback();
    }
}

void WhisperContext::ToggleNativeLog(const Napi::CallbackInfo& info) {
    toggle_native_log(info);
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
        InstanceMethod("abortTranscribe", &WhisperContext::AbortTranscribe),
        InstanceMethod("bench", &WhisperContext::Bench),
        InstanceMethod("release", &WhisperContext::Release),
        InstanceMethod("releaseSync", &WhisperContext::ReleaseSync),
        StaticMethod("releaseAllSync", &WhisperContext::ReleaseAllSync),
    });

    exports.Set("WhisperContext", func);
}


Napi::Value WhisperContext::GetModelInfo(const Napi::CallbackInfo& info) {
    if (_meta.IsEmpty()) {
        return info.Env().Null();
    }
    return _meta.Value();
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

    // Create cancellation flag
    auto cancelFlag = std::make_shared<std::atomic<bool>>(false);
    int jobId = registerJob(cancelFlag);

    auto deferred = Napi::Promise::Deferred::New(env);

    try {
        // Load audio file
        std::vector<float> audioData = whisper_utils::loadAudioFile(filePath);

        // Create parameters
        whisper_full_params params = whisper_utils::createFullParamsFromOptions(options);
        int nProcessors = whisper_utils::getNProcessorsFromOptions(options);

        // Check for onProgress callback
        Napi::ThreadSafeFunction tsfnProgress;
        bool hasProgress = false;
        auto onProgressValue = options.Get("onProgress");
        if (onProgressValue.IsFunction()) {
            hasProgress = true;
            tsfnProgress = Napi::ThreadSafeFunction::New(
                env,
                onProgressValue.As<Napi::Function>(),
                "WhisperProgressCallback",
                0,
                1
            );
        }

        // Check for onNewSegments callback
        Napi::ThreadSafeFunction tsfnNewSegments;
        bool hasNewSegments = false;
        auto onNewSegmentsValue = options.Get("onNewSegments");
        if (onNewSegmentsValue.IsFunction()) {
            hasNewSegments = true;
            tsfnNewSegments = Napi::ThreadSafeFunction::New(
                env,
                onNewSegmentsValue.As<Napi::Function>(),
                "WhisperNewSegmentsCallback",
                0,
                1
            );
        }

        // Create async worker with cancellation support and callbacks
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

        // Extract language and prompt for worker storage
        std::string language = whisper_utils::getString(options.Get("language"));
        std::string prompt = whisper_utils::getString(options.Get("prompt"));

        auto worker = new WhisperTranscribeWorker(callback, _sess, audioData, params, nProcessors, cancelFlag, tsfnProgress, tsfnNewSegments, hasProgress, hasNewSegments, language, prompt);
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

    // Create cancellation flag
    auto cancelFlag = std::make_shared<std::atomic<bool>>(false);
    int jobId = registerJob(cancelFlag);

    auto deferred = Napi::Promise::Deferred::New(env);

    try {
        // Convert ArrayBuffer to float array
        std::vector<float> audioData = whisper_utils::convertAudioBufferToFloat(audioBuffer);

        // Create parameters
        whisper_full_params params = whisper_utils::createFullParamsFromOptions(options);
        int nProcessors = whisper_utils::getNProcessorsFromOptions(options);

        // Check for onProgress callback
        Napi::ThreadSafeFunction tsfnProgress;
        bool hasProgress = false;
        auto onProgressValue = options.Get("onProgress");
        if (onProgressValue.IsFunction()) {
            hasProgress = true;
            tsfnProgress = Napi::ThreadSafeFunction::New(
                env,
                onProgressValue.As<Napi::Function>(),
                "WhisperProgressCallback",
                0,
                1
            );
        }

        // Check for onNewSegments callback
        Napi::ThreadSafeFunction tsfnNewSegments;
        bool hasNewSegments = false;
        auto onNewSegmentsValue = options.Get("onNewSegments");
        if (onNewSegmentsValue.IsFunction()) {
            hasNewSegments = true;
            tsfnNewSegments = Napi::ThreadSafeFunction::New(
                env,
                onNewSegmentsValue.As<Napi::Function>(),
                "WhisperNewSegmentsCallback",
                0,
                1
            );
        }

        // Create async worker with cancellation support and callbacks
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

        // Extract language and prompt for worker storage
        std::string language = whisper_utils::getString(options.Get("language"));
        std::string prompt = whisper_utils::getString(options.Get("prompt"));

        auto worker = new WhisperTranscribeWorker(callback, _sess, audioData, params, nProcessors, cancelFlag, tsfnProgress, tsfnNewSegments, hasProgress, hasNewSegments, language, prompt);
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

Napi::Value WhisperContext::AbortTranscribe(const Napi::CallbackInfo& info) {
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

Napi::Value WhisperContext::Bench(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    int n_threads = 1;
    if (info.Length() >= 1 && info[0].IsNumber()) {
        n_threads = info[0].As<Napi::Number>().Int32Value();
    }

    if (!_sess || !_sess->isValid()) {
        Napi::Error::New(env, "Invalid whisper context").ThrowAsJavaScriptException();
        return env.Null();
    }

    auto deferred = Napi::Promise::Deferred::New(env);

    // Lock the session for benchmarking
    std::lock_guard<std::mutex> lock(_sess->mtx);

    if (!_sess->ctx) {
        deferred.Reject(Napi::Error::New(env, "Whisper context was destroyed").Value());
        return deferred.Promise();
    }

    whisper_context* ctx = _sess->ctx;

    // Get model info for setting mel
    const int n_mels = whisper_model_n_mels(ctx);

    // Set empty mel spectrogram
    if (int ret = whisper_set_mel(ctx, nullptr, 0, n_mels)) {
        std::string error = "error: failed to set mel: " + std::to_string(ret);
        deferred.Reject(Napi::Error::New(env, error).Value());
        return deferred.Promise();
    }

    // Heat encoder
    if (int ret = whisper_encode(ctx, 0, n_threads) != 0) {
        std::string error = "error: failed to encode: " + std::to_string(ret);
        deferred.Reject(Napi::Error::New(env, error).Value());
        return deferred.Promise();
    }

    // Prepare tokens for decode
    whisper_token tokens[512];
    memset(tokens, 0, sizeof(tokens));

    // Prompt heat
    if (int ret = whisper_decode(ctx, tokens, 256, 0, n_threads) != 0) {
        std::string error = "error: failed to decode: " + std::to_string(ret);
        deferred.Reject(Napi::Error::New(env, error).Value());
        return deferred.Promise();
    }

    // Text-generation heat
    if (int ret = whisper_decode(ctx, tokens, 1, 256, n_threads) != 0) {
        std::string error = "error: failed to decode: " + std::to_string(ret);
        deferred.Reject(Napi::Error::New(env, error).Value());
        return deferred.Promise();
    }

    // Reset timings for actual benchmark
    whisper_reset_timings(ctx);

    // Actual encode run
    if (int ret = whisper_encode(ctx, 0, n_threads) != 0) {
        std::string error = "error: failed to encode: " + std::to_string(ret);
        deferred.Reject(Napi::Error::New(env, error).Value());
        return deferred.Promise();
    }

    // Text-generation (256 decode calls with 1 token each)
    for (int i = 0; i < 256; i++) {
        if (int ret = whisper_decode(ctx, tokens, 1, i, n_threads) != 0) {
            std::string error = "error: failed to decode: " + std::to_string(ret);
            deferred.Reject(Napi::Error::New(env, error).Value());
            return deferred.Promise();
        }
    }

    // Batched decoding (64 calls with 5 tokens each)
    for (int i = 0; i < 64; i++) {
        if (int ret = whisper_decode(ctx, tokens, 5, 0, n_threads) != 0) {
            std::string error = "error: failed to decode: " + std::to_string(ret);
            deferred.Reject(Napi::Error::New(env, error).Value());
            return deferred.Promise();
        }
    }

    // Prompt processing (16 calls with 256 tokens each)
    for (int i = 0; i < 16; i++) {
        if (int ret = whisper_decode(ctx, tokens, 256, 0, n_threads) != 0) {
            std::string error = "error: failed to decode: " + std::to_string(ret);
            deferred.Reject(Napi::Error::New(env, error).Value());
            return deferred.Promise();
        }
    }

    // Get timings
    const struct whisper_timings* timings = whisper_get_timings(ctx);

    // Get system info
    const char* system_info = whisper_print_system_info();

    // Build result object
    auto result = Napi::Object::New(env);
    result.Set("config", Napi::String::New(env, system_info ? system_info : ""));
    result.Set("nThreads", Napi::Number::New(env, n_threads));
    result.Set("encodeMs", Napi::Number::New(env, timings ? timings->encode_ms : 0.0f));
    result.Set("decodeMs", Napi::Number::New(env, timings ? timings->decode_ms : 0.0f));
    result.Set("batchdMs", Napi::Number::New(env, timings ? timings->batchd_ms : 0.0f));
    result.Set("promptMs", Napi::Number::New(env, timings ? timings->prompt_ms : 0.0f));

    deferred.Resolve(result);
    return deferred.Promise();
}

Napi::Value WhisperContext::Release(const Napi::CallbackInfo& info) {
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

void WhisperContext::releaseSync() {
    {
        std::lock_guard<std::mutex> lock(_cancelMutex);
        for (auto& [jobId, cancelFlag] : _cancelFlags) {
            cancelFlag->store(true);
        }
        _cancelFlags.clear();
    }

    // a running job holds the session mutex until it has stopped using the context
    if (_sess) {
        std::lock_guard<std::mutex> lock(_sess->mtx);
        if (_sess->ctx) {
            whisper_free(_sess->ctx);
            _sess->ctx = nullptr;
        }
    }
    _sess.reset();
}

// releaseSync(): void
// Synchronous variant of release() for process exit handlers, where no event loop is available
void WhisperContext::ReleaseSync(const Napi::CallbackInfo& info) {
    releaseSync();
}

// WhisperContext.releaseAllSync(): void
// Release every live context, used on process exit so that no backend buffers outlive the process teardown
void WhisperContext::ReleaseAllSync(const Napi::CallbackInfo& info) {
    for (auto* ctx : g_whisper_contexts.snapshot()) {
        ctx->releaseSync();
    }
}

// WhisperVadContext implementation (vad.cpp)
WhisperVadContext::WhisperVadContext(const Napi::CallbackInfo& info) : Napi::ObjectWrap<WhisperVadContext>(info) {
    Napi::Env env = info.Env();

    if (info.Length() < 1 || !info[0].IsObject()) {
        Napi::TypeError::New(env, "Expected options object").ThrowAsJavaScriptException();
        return;
    }

    auto options = info[0].As<Napi::Object>();
    std::string modelPath = whisper_utils::getString(options.Get("filePath"));
    // useGpu is ignored: VAD always runs on the CPU (vad.cpp's own kernels),
    // which is fastest for these small per-hop graphs. 1 thread is best too.
    int nThreads = whisper_utils::getInt(options.Get("nThreads"), 1);

    if (modelPath.empty()) {
        Napi::TypeError::New(env, "Model path is required").ThrowAsJavaScriptException();
        return;
    }

    vad_model_params params = vad_model_default_params();
    params.n_threads = nThreads > 0 ? nThreads : 1;

    vad_model* model = vad_model_load(modelPath.c_str(), params);
    if (!model) {
        Napi::Error::New(env, "Failed to load the VAD model").ThrowAsJavaScriptException();
        return;
    }
    // vad.cpp also loads turn-end and diarization models; only speech
    // probabilities at 16 kHz are wired up for now.
    if (vad_model_output(model) != VAD_OUTPUT_SPEECH || vad_model_n_outputs(model) != 1) {
        std::string arch = vad_model_arch(model);
        vad_model_free(model);
        Napi::Error::New(env, "Unsupported VAD model (" + arch + "): only speech detection models are supported")
            .ThrowAsJavaScriptException();
        return;
    }
    if (vad_model_sample_rate(model) != WHISPER_SAMPLE_RATE) {
        vad_model_free(model);
        Napi::Error::New(env, "Unsupported VAD model: expected a 16 kHz model").ThrowAsJavaScriptException();
        return;
    }
    _sess = std::make_shared<WhisperVadSession>(modelPath, model);
    _sampleRate = _sess->sampleRate;
    _hopSamples = _sess->hopSamples;
    _arch = _sess->arch;

    // Build metadata (persistent reference so it outlives the constructor scope)
    auto meta = Napi::Object::New(env);
    meta.Set("filePath", modelPath);
    meta.Set("useGpu", false);
    meta.Set("nThreads", params.n_threads);
    meta.Set("sampleRate", _sampleRate);
    meta.Set("hopSamples", _hopSamples);
    meta.Set("arch", _arch);
    _meta = Napi::Persistent(meta);

    g_vad_contexts.add(this);
}

WhisperVadContext::~WhisperVadContext() {
    g_vad_contexts.remove(this);
}

void WhisperVadContext::releaseSync() {
    if (_sess) {
        std::lock_guard<std::mutex> lock(_sess->mtx);
        _sess->freeLocked();
    }
    _sess.reset();
}

void WhisperVadContext::ReleaseSync(const Napi::CallbackInfo& info) {
    releaseSync();
}

void WhisperVadContext::ReleaseAllSync(const Napi::CallbackInfo& info) {
    for (auto* ctx : g_vad_contexts.snapshot()) {
        ctx->releaseSync();
    }
}

void WhisperVadContext::ToggleNativeLog(const Napi::CallbackInfo& info) {
    toggle_native_log(info);
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
        InstanceAccessor("sampleRate", &WhisperVadContext::GetSampleRate, nullptr),
        InstanceAccessor("hopSamples", &WhisperVadContext::GetHopSamples, nullptr),
        InstanceAccessor("arch", &WhisperVadContext::GetArch, nullptr),
        InstanceMethod("getModelInfo", &WhisperVadContext::GetModelInfo),
        InstanceMethod("detectSpeechFile", &WhisperVadContext::DetectSpeechFile),
        InstanceMethod("detectSpeech", &WhisperVadContext::DetectSpeechFile),
        InstanceMethod("detectSpeechData", &WhisperVadContext::DetectSpeechData),
        InstanceMethod("createStream", &WhisperVadContext::CreateStream),
        InstanceMethod("release", &WhisperVadContext::Release),
        InstanceMethod("releaseSync", &WhisperVadContext::ReleaseSync),
        StaticMethod("releaseAllSync", &WhisperVadContext::ReleaseAllSync),
    });

    exports.Set("WhisperVadContext", func);
}

Napi::Value WhisperVadContext::GetModelInfo(const Napi::CallbackInfo& info) {
    if (_meta.IsEmpty()) {
        return info.Env().Null();
    }
    return _meta.Value();
}

Napi::Value WhisperVadContext::GetSampleRate(const Napi::CallbackInfo& info) {
    return Napi::Number::New(info.Env(), _sampleRate);
}

Napi::Value WhisperVadContext::GetHopSamples(const Napi::CallbackInfo& info) {
    return Napi::Number::New(info.Env(), _hopSamples);
}

Napi::Value WhisperVadContext::GetArch(const Napi::CallbackInfo& info) {
    return Napi::String::New(info.Env(), _arch);
}

static Napi::Function createVadPromiseCallback(Napi::Env env, Napi::Promise::Deferred deferred) {
    return Napi::Function::New(env, [deferred](const Napi::CallbackInfo& cbInfo) {
        if (cbInfo.Length() >= 2) {
            if (!cbInfo[0].IsNull()) {
                deferred.Reject(cbInfo[0]);
            } else {
                deferred.Resolve(cbInfo[1]);
            }
        }
    });
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
        std::vector<float> audioData = whisper_utils::loadAudioFile(filePath);
        vad_segment_params vadParams = whisper_utils::createVadParamsFromOptions(options);
        auto worker = new WhisperVadWorker(createVadPromiseCallback(env, deferred), _sess, std::move(audioData), vadParams);
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
        std::vector<float> audioData = whisper_utils::convertAudioBufferToFloat(audioBuffer);
        vad_segment_params vadParams = whisper_utils::createVadParamsFromOptions(options);
        auto worker = new WhisperVadWorker(createVadPromiseCallback(env, deferred), _sess, std::move(audioData), vadParams);
        worker->Queue();
    } catch (const std::exception& e) {
        deferred.Reject(Napi::Error::New(env, e.what()).Value());
    }

    return deferred.Promise();
}

Napi::Value WhisperVadContext::CreateStream(const Napi::CallbackInfo& info) {
    auto worker = new WhisperVadStreamCreateWorker(info.Env(), _sess);
    auto promise = worker->Promise();
    worker->Queue();
    return promise;
}

Napi::Value WhisperVadContext::Release(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    // Frees the model and its streams now, or right after the calls already
    // made on them, not when the last holder (a stream object) drops it.
    // Calls made from now on fail.
    if (_sess) {
        _sess->requestRelease();
    }
    _sess.reset();
    auto deferred = Napi::Promise::Deferred::New(env);
    deferred.Resolve(env.Undefined());
    return deferred.Promise();
}

// WhisperVadStream implementation
struct WhisperVadAddonData {
    Napi::FunctionReference vadStreamConstructor;
};

void WhisperVadStream::Init(Napi::Env env, Napi::Object& exports) {
    Napi::Function func = DefineClass(env, "WhisperVadStream", {
        InstanceAccessor("sampleRate", &WhisperVadStream::GetSampleRate, nullptr),
        InstanceAccessor("hopSamples", &WhisperVadStream::GetHopSamples, nullptr),
        InstanceMethod("push", &WhisperVadStream::Push),
        InstanceMethod("flush", &WhisperVadStream::Flush),
        InstanceMethod("reset", &WhisperVadStream::Reset),
        InstanceMethod("release", &WhisperVadStream::Release),
    });

    auto* data = new WhisperVadAddonData();
    data->vadStreamConstructor = Napi::Persistent(func);
    env.SetInstanceData<WhisperVadAddonData>(data);
    // Not exported: streams come from WhisperVadContext.createStream()
}

Napi::Object WhisperVadStream::NewInstance(
    Napi::Env env,
    WhisperVadSessionPtr session,
    std::shared_ptr<VadStreamState> state) {
    auto* data = env.GetInstanceData<WhisperVadAddonData>();
    Napi::Object object = data->vadStreamConstructor.New({});
    auto* stream = WhisperVadStream::Unwrap(object);
    stream->_sess = session;
    stream->_state = state;
    stream->_sampleRate = session->sampleRate;
    stream->_hopSamples = session->hopSamples;
    return object;
}

WhisperVadStream::WhisperVadStream(const Napi::CallbackInfo& info) : Napi::ObjectWrap<WhisperVadStream>(info) {}

WhisperVadStream::~WhisperVadStream() {
    // Collected without release(). No call of this stream is in flight (each
    // one holds a reference to this object), but another call on the context
    // may hold the lock for long (a whole offline detection): never wait for
    // it on the JS thread, hand the stream to the next call to free instead.
    if (!_sess || !_state) {
        return;
    }
    std::unique_lock<std::mutex> lock(_sess->mtx, std::try_to_lock);
    if (lock.owns_lock()) {
        if (_state->stream) {
            vad_stream_free(_state->stream);
            _state->stream = nullptr;
        }
        _sess->streams.erase(_state);
        return;
    }
    std::lock_guard<std::mutex> orphansLock(_sess->orphansMtx);
    _sess->orphans.push_back(_state);
}

Napi::Value WhisperVadStream::GetSampleRate(const Napi::CallbackInfo& info) {
    return Napi::Number::New(info.Env(), _sampleRate);
}

Napi::Value WhisperVadStream::GetHopSamples(const Napi::CallbackInfo& info) {
    return Napi::Number::New(info.Env(), _hopSamples);
}

void WhisperVadStream::enqueue(Napi::AsyncWorker* worker) {
    if (_running) {
        _pending.push_back(worker);
        return;
    }
    _running = true;
    worker->Queue();
}

void WhisperVadStream::runNext() {
    if (_pending.empty()) {
        _running = false;
        return;
    }
    Napi::AsyncWorker* worker = _pending.front();
    _pending.pop_front();
    worker->Queue();
}

Napi::Value WhisperVadStream::Push(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    std::vector<float> audio;
    try {
        if (info.Length() >= 1 && info[0].IsArrayBuffer()) {
            auto buffer = info[0].As<Napi::ArrayBuffer>();
            audio = whisper_utils::convertAudioBufferToFloat(buffer);
        } else if (info.Length() >= 1 && info[0].IsTypedArray()) {
            auto typed = info[0].As<Napi::TypedArray>();
            const auto type = typed.TypedArrayType();
            const uint8_t* data = static_cast<const uint8_t*>(typed.ArrayBuffer().Data()) + typed.ByteOffset();
            if (type == napi_float32_array) {
                // Float samples in [-1, 1], as the WASM package accepts
                audio.resize(typed.ElementLength());
                memcpy(audio.data(), data, typed.ByteLength());
            } else if (type == napi_uint8_array || type == napi_int16_array) {
                // PCM16 bytes / samples
                auto bytes = Napi::ArrayBuffer::New(env, typed.ByteLength());
                memcpy(bytes.Data(), data, typed.ByteLength());
                audio = whisper_utils::convertAudioBufferToFloat(bytes);
            } else {
                Napi::TypeError::New(env, "Expected ArrayBuffer, Uint8Array, Int16Array or Float32Array").ThrowAsJavaScriptException();
                return env.Null();
            }
        } else {
            Napi::TypeError::New(env, "Expected ArrayBuffer, Uint8Array, Int16Array or Float32Array").ThrowAsJavaScriptException();
            return env.Null();
        }
    } catch (const std::exception& e) {
        auto deferred = Napi::Promise::Deferred::New(env);
        deferred.Reject(Napi::Error::New(env, e.what()).Value());
        return deferred.Promise();
    }

    // The audio is copied above, so the caller may reuse its buffer right away
    auto worker = new WhisperVadStreamWorker(
        env, info.This().As<Napi::Object>(), _sess, _state, WhisperVadStreamWorker::Op::Push, std::move(audio));
    auto promise = worker->Promise();
    enqueue(worker);
    return promise;
}

Napi::Value WhisperVadStream::Flush(const Napi::CallbackInfo& info) {
    auto worker = new WhisperVadStreamWorker(
        info.Env(), info.This().As<Napi::Object>(), _sess, _state, WhisperVadStreamWorker::Op::Flush);
    auto promise = worker->Promise();
    enqueue(worker);
    return promise;
}

Napi::Value WhisperVadStream::Reset(const Napi::CallbackInfo& info) {
    auto worker = new WhisperVadStreamWorker(
        info.Env(), info.This().As<Napi::Object>(), _sess, _state, WhisperVadStreamWorker::Op::Reset);
    auto promise = worker->Promise();
    enqueue(worker);
    return promise;
}

Napi::Value WhisperVadStream::Release(const Napi::CallbackInfo& info) {
    auto worker = new WhisperVadStreamWorker(
        info.Env(), info.This().As<Napi::Object>(), _sess, _state, WhisperVadStreamWorker::Op::Release);
    auto promise = worker->Promise();
    enqueue(worker);
    return promise;
}
