#include "common.hpp"
#include "whisper.h"
#include "common-whisper.h"
#include <iostream>
#include <cstdarg>
#include <string>
#include <vector>
#include <algorithm>
#include <thread>

// Global logging state
whisper_log_callback g_log_callback = nullptr;
bool g_log_enabled = false;

// Default logging callback
void whisper_log_callback_default(const char* level, const char* text) {
    std::cout << "[whisper.node " << level << "] " << text << std::endl;
}

// Setup logging
void setup_logging() {
    if (!g_log_callback) {
        g_log_callback = whisper_log_callback_default;
    }
}

// Cleanup logging
void cleanup_logging() {
    g_log_callback = nullptr;
    g_log_enabled = false;
    cleanup_js_log_callback();  // Clean up JavaScript callback
}

// WhisperSession implementation
WhisperSession::WhisperSession(const std::string& modelPath, whisper_context* context)
    : path(modelPath), ctx(context) {
}

WhisperSession::~WhisperSession() {
    if (ctx) {
        whisper_free(ctx);
        ctx = nullptr;
    }
}

bool WhisperSession::isValid() const {
    return ctx != nullptr;
}

void WhisperSession::lock() {
    mtx.lock();
}

void WhisperSession::unlock() {
    mtx.unlock();
}

// WhisperVadSession implementation
WhisperVadSession::WhisperVadSession(const std::string& modelPath, whisper_vad_context* context)
    : path(modelPath), ctx(context) {
}

WhisperVadSession::~WhisperVadSession() {
    if (ctx) {
        whisper_vad_free(ctx);
        ctx = nullptr;
    }
}

bool WhisperVadSession::isValid() const {
    return ctx != nullptr;
}

void WhisperVadSession::lock() {
    mtx.lock();
}

void WhisperVadSession::unlock() {
    mtx.unlock();
}

// Utility functions
namespace whisper_utils {

std::string getString(const Napi::Value& value, const std::string& defaultValue) {
    if (value.IsString()) {
        return value.As<Napi::String>().Utf8Value();
    }
    return defaultValue;
}

int getInt(const Napi::Value& value, int defaultValue) {
    if (value.IsNumber()) {
        return value.As<Napi::Number>().Int32Value();
    }
    return defaultValue;
}

float getFloat(const Napi::Value& value, float defaultValue) {
    if (value.IsNumber()) {
        return value.As<Napi::Number>().FloatValue();
    }
    return defaultValue;
}

bool getBool(const Napi::Value& value, bool defaultValue) {
    if (value.IsBoolean()) {
        return value.As<Napi::Boolean>().Value();
    }
    return defaultValue;
}

whisper_full_params createFullParamsFromOptions(const Napi::Object& options) {
    whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    const int hardware = static_cast<int>(std::thread::hardware_concurrency());
    params.n_threads = std::max(1, std::min(8, hardware == 0 ? 1 : hardware));

    auto propNames = options.GetPropertyNames();
    for (uint32_t i = 0; i < propNames.Length(); i++) {
        auto key = propNames.Get(i).As<Napi::String>().Utf8Value();
        auto value = options.Get(key);

        if (key == "translate" && value.IsBoolean()) {
            params.translate = value.As<Napi::Boolean>().Value();
        } else if (key == "maxThreads" && value.IsNumber()) {
            params.n_threads = value.As<Napi::Number>().Int32Value();
        } else if (key == "maxContext" && value.IsNumber()) {
            params.n_max_text_ctx = value.As<Napi::Number>().Int32Value();
        } else if (key == "maxLen" && value.IsNumber()) {
            params.max_len = value.As<Napi::Number>().Int32Value();
        } else if (key == "tokenTimestamps" && value.IsBoolean()) {
            params.token_timestamps = value.As<Napi::Boolean>().Value();
        } else if (key == "tdrzEnable" && value.IsBoolean()) {
            params.tdrz_enable = value.As<Napi::Boolean>().Value();
        } else if (key == "wordThold" && value.IsNumber()) {
            params.thold_pt = value.As<Napi::Number>().FloatValue();
        } else if (key == "offset" && value.IsNumber()) {
            params.offset_ms = value.As<Napi::Number>().Int32Value();
        } else if (key == "duration" && value.IsNumber()) {
            params.duration_ms = value.As<Napi::Number>().Int32Value();
        } else if (key == "temperature" && value.IsNumber()) {
            params.temperature = value.As<Napi::Number>().FloatValue();
        } else if (key == "temperatureInc" && value.IsNumber()) {
            params.temperature_inc = value.As<Napi::Number>().FloatValue();
        } else if (key == "beamSize" && value.IsNumber()) {
            params.beam_search.beam_size = value.As<Napi::Number>().Int32Value();
        } else if (key == "bestOf" && value.IsNumber()) {
            params.greedy.best_of = value.As<Napi::Number>().Int32Value();
        }
        // Note: language and prompt are handled in WhisperContext.cpp to ensure string lifetime
    }

    return params;
}

int getNProcessorsFromOptions(const Napi::Object& options) {
    auto value = options.Get("nProcessors");
    if (value.IsNumber()) {
        return value.As<Napi::Number>().Int32Value();
    }
    // Default to 1 processor (no parallelization)
    return 1;
}

whisper_vad_params createVadParamsFromOptions(const Napi::Object& options) {
    whisper_vad_params params = whisper_vad_default_params();

    auto propNames = options.GetPropertyNames();
    for (uint32_t i = 0; i < propNames.Length(); i++) {
        auto key = propNames.Get(i).As<Napi::String>().Utf8Value();
        auto value = options.Get(key);

        if (key == "threshold" && value.IsNumber()) {
            params.threshold = value.As<Napi::Number>().FloatValue();
        } else if (key == "minSpeechDurationMs" && value.IsNumber()) {
            params.min_speech_duration_ms = value.As<Napi::Number>().Int32Value();
        } else if (key == "minSilenceDurationMs" && value.IsNumber()) {
            params.min_silence_duration_ms = value.As<Napi::Number>().Int32Value();
        } else if (key == "maxSpeechDurationS" && value.IsNumber()) {
            params.max_speech_duration_s = value.As<Napi::Number>().FloatValue();
        } else if (key == "speechPadMs" && value.IsNumber()) {
            params.speech_pad_ms = value.As<Napi::Number>().Int32Value();
        } else if (key == "samplesOverlap" && value.IsNumber()) {
            params.samples_overlap = value.As<Napi::Number>().FloatValue();
        }
    }

    return params;
}

Napi::Object createTranscribeResult(Napi::Env env, whisper_context* ctx, const std::string& text, bool aborted) {
    auto result = Napi::Object::New(env);
    result.Set("result", text);
    result.Set("isAborted", aborted);

    if (ctx) {
        auto languageId = whisper_full_lang_id(ctx);
        result.Set("language", whisper_lang_str(languageId));
    }

    // Create segments array
    auto segments = Napi::Array::New(env);
    if (ctx && !aborted) {
        int n_segments = whisper_full_n_segments(ctx);
        for (int i = 0; i < n_segments; i++) {
            auto segment = Napi::Object::New(env);
            segment.Set("text", whisper_full_get_segment_text(ctx, i));
            segment.Set("t0", whisper_full_get_segment_t0(ctx, i) * 10); // Convert to milliseconds
            segment.Set("t1", whisper_full_get_segment_t1(ctx, i) * 10); // Convert to milliseconds
            segments.Set(i, segment);
        }
    }
    result.Set("segments", segments);

    return result;
}

Napi::Object createVadResult(Napi::Env env, bool hasSpeech, float speechProbability, const std::vector<std::pair<int64_t, int64_t>>& segments) {
    auto result = Napi::Object::New(env);
    result.Set("is_speech", hasSpeech);
    result.Set("speech_probability", speechProbability);

    auto speechTimestamps = Napi::Array::New(env);
    for (size_t i = 0; i < segments.size(); i++) {
        auto segment = Napi::Object::New(env);
        segment.Set("start", segments[i].first);
        segment.Set("end", segments[i].second);
        speechTimestamps.Set(i, segment);
    }
    result.Set("speech_timestamps", speechTimestamps);

    return result;
}

std::vector<float> convertAudioBufferToFloat(Napi::ArrayBuffer& buffer) {
    uint8_t* data = static_cast<uint8_t*>(buffer.Data());
    size_t size = buffer.ByteLength();

    // Assume 16-bit PCM audio data
    if (size % 2 != 0) {
        throw std::runtime_error("Audio buffer size must be even for 16-bit PCM");
    }

    size_t numSamples = size / 2;
    std::vector<float> audioData(numSamples);

    // Convert 16-bit PCM to float32
    int16_t* pcmData = reinterpret_cast<int16_t*>(data);
    for (size_t i = 0; i < numSamples; i++) {
        audioData[i] = static_cast<float>(pcmData[i]) / 32768.0f;
    }

    return audioData;
}

std::vector<float> loadAudioFile(const std::string& filePath) {
    std::vector<float> pcmf32;
    std::vector<std::vector<float>> pcmf32s;
    
    bool success = read_audio_data(filePath, pcmf32, pcmf32s, false);
    if (!success) {
        throw std::runtime_error("Failed to load audio file: " + filePath);
    }
    
    return pcmf32;
}

} // namespace whisper_utils
