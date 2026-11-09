#pragma once

#include <napi.h>
#include <memory>
#include <string>
#include <vector>
#include <map>
#include <mutex>
#include <functional>
#include <set>

#include "whisper.h"
#include "parakeet.h"
#include "vad.h"

// Forward declarations
struct whisper_context;
struct whisper_full_params;
struct parakeet_context;

// Logging callback type
typedef void (*whisper_log_callback)(const char* level, const char* text);

// Common utilities
namespace whisper_utils {
    std::string getString(const Napi::Value& value, const std::string& defaultValue = "");
    int getInt(const Napi::Value& value, int defaultValue = 0);
    float getFloat(const Napi::Value& value, float defaultValue = 0.0f);
    bool getBool(const Napi::Value& value, bool defaultValue = false);

    whisper_full_params createFullParamsFromOptions(const Napi::Object& options);
    int getNProcessorsFromOptions(const Napi::Object& options);
    vad_segment_params createVadParamsFromOptions(const Napi::Object& options);
    parakeet_full_params createParakeetParamsFromOptions(const Napi::Object& options);
    Napi::Object createTranscribeResult(Napi::Env env, whisper_context* ctx, const std::string& text, bool aborted = false);
    Napi::Object createParakeetTranscribeResult(Napi::Env env, parakeet_context* ctx, bool aborted = false);

    std::vector<float> convertAudioBufferToFloat(Napi::ArrayBuffer& buffer);
    std::vector<float> loadAudioFile(const std::string& filePath);
}

// Live contexts of one type, so that all of them can be released synchronously on process exit
template <typename T>
class LiveContexts {
public:
    void add(T* ctx) { std::lock_guard<std::mutex> lock(mtx_); items_.insert(ctx); }
    void remove(T* ctx) { std::lock_guard<std::mutex> lock(mtx_); items_.erase(ctx); }
    std::vector<T*> snapshot() { std::lock_guard<std::mutex> lock(mtx_); return std::vector<T*>(items_.begin(), items_.end()); }

private:
    std::mutex mtx_;
    std::set<T*> items_;
};

// Session management
class WhisperSession {
public:
    std::string path;
    whisper_context* ctx;
    std::mutex mtx;

    WhisperSession(const std::string& modelPath, whisper_context* context);
    ~WhisperSession();

    bool isValid() const;
    void lock();
    void unlock();
};

using WhisperSessionPtr = std::shared_ptr<WhisperSession>;

// VAD Session management (vad.cpp model + the streams created on it)
//
// vad_stream objects belong to the model and must be freed before it. Every
// operation that touches the model or one of its streams holds `mtx`.
// release() never blocks: it frees right away when no call is in flight,
// otherwise the last call made before it frees (calls made after it fail).
struct VadStreamState {
    vad_stream* stream = nullptr;  // guarded by WhisperVadSession::mtx
};

class WhisperVadSession {
public:
    std::string path;
    vad_model* ctx;
    std::mutex mtx;
    int sampleRate = 0;
    int hopSamples = 0;
    std::string arch;
    std::set<std::shared_ptr<VadStreamState>> streams;  // guarded by mtx
    // Calls created and not finished yet, and whether release() was called
    std::mutex stateMtx;
    int inflight = 0;
    bool releasing = false;
    // Streams collected while `mtx` was busy; freed by the next locked call.
    std::mutex orphansMtx;
    std::vector<std::shared_ptr<VadStreamState>> orphans;

    WhisperVadSession(const std::string& modelPath, vad_model* model);
    ~WhisperVadSession();

    bool isValid() const;
    // Register a call (JS thread). False once release() was called.
    bool retain();
    // A registered call finished its work. Caller must hold mtx.
    void releaseLocked();
    // release(): free now if idle, else after the calls in flight (JS thread)
    void requestRelease();
    // Free streams collected while busy. Caller must hold mtx.
    void freeOrphansLocked();
    // Free the streams and the model. Caller must hold mtx.
    void freeLocked();
};

using WhisperVadSessionPtr = std::shared_ptr<WhisperVadSession>;

// Parakeet Session management
class ParakeetSession {
public:
    std::string path;
    parakeet_context* ctx;
    std::mutex mtx;

    ParakeetSession(const std::string& modelPath, parakeet_context* context);
    ~ParakeetSession();

    bool isValid() const;
    void lock();
    void unlock();
};

using ParakeetSessionPtr = std::shared_ptr<ParakeetSession>;

// Logging utilities
extern whisper_log_callback g_log_callback;
extern bool g_log_enabled;

void whisper_log_callback_default(const char* level, const char* text);
void setup_logging();
void cleanup_logging();
void cleanup_js_log_callback();  // Cleanup JavaScript logging callback
void toggle_native_log(const Napi::CallbackInfo& info);  // Shared static toggleNativeLog implementation
