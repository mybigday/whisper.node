#pragma once

#include <napi.h>
#include <memory>
#include <string>
#include <vector>
#include <map>
#include <mutex>
#include <functional>

#include "whisper.h"

// Forward declarations
struct whisper_context;
struct whisper_vad_context;
struct whisper_full_params;

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
    whisper_vad_params createVadParamsFromOptions(const Napi::Object& options);
    Napi::Object createTranscribeResult(Napi::Env env, whisper_context* ctx, const std::string& text, bool aborted = false);
    Napi::Object createVadResult(Napi::Env env, bool hasSpeech, float speechProbability, const std::vector<std::pair<int64_t, int64_t>>& segments);

    std::vector<float> convertAudioBufferToFloat(Napi::ArrayBuffer& buffer);
    std::vector<float> loadAudioFile(const std::string& filePath);
}

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

// VAD Session management
class WhisperVadSession {
public:
    std::string path;
    whisper_vad_context* ctx;
    std::mutex mtx;

    WhisperVadSession(const std::string& modelPath, whisper_vad_context* context);
    ~WhisperVadSession();

    bool isValid() const;
    void lock();
    void unlock();
};

using WhisperVadSessionPtr = std::shared_ptr<WhisperVadSession>;

// Logging utilities
extern whisper_log_callback g_log_callback;
extern bool g_log_enabled;

void whisper_log_callback_default(const char* level, const char* text);
void setup_logging();
void cleanup_logging();
void cleanup_js_log_callback();  // Cleanup JavaScript logging callback
