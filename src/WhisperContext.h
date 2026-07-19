#pragma once

#include "common.hpp"
#include <atomic>
#include <unordered_map>

#ifdef NO_GPU_SUPPORT
#define USE_GPU false
#else
#define USE_GPU true
#endif

class WhisperContext : public Napi::ObjectWrap<WhisperContext> {
public:
    WhisperContext(const Napi::CallbackInfo& info);
    ~WhisperContext();

    // Static methods
    static void ToggleNativeLog(const Napi::CallbackInfo& info);
    static Napi::Value ModelInfo(const Napi::CallbackInfo& info);
    static void Init(Napi::Env env, Napi::Object& exports);

private:
    // Instance methods
    Napi::Value GetSystemInfo(const Napi::CallbackInfo& info);
    Napi::Value GetModelInfo(const Napi::CallbackInfo& info);
    Napi::Value TranscribeFile(const Napi::CallbackInfo& info);
    Napi::Value TranscribeData(const Napi::CallbackInfo& info);
    Napi::Value AbortTranscribe(const Napi::CallbackInfo& info);
    Napi::Value Bench(const Napi::CallbackInfo& info);
    Napi::Value Release(const Napi::CallbackInfo& info);

    // Internal data
    std::string _info;
    Napi::ObjectReference _meta;
    WhisperSessionPtr _sess = nullptr;

    // Job tracking for cancellation
    std::atomic<int> _nextJobId{1};
    std::unordered_map<int, std::shared_ptr<std::atomic<bool>>> _cancelFlags;
    std::mutex _cancelMutex;

public:
    // Public method to register a job for cancellation
    int registerJob(std::shared_ptr<std::atomic<bool>> cancelFlag);
    void unregisterJob(int jobId);
    bool isJobCancelled(int jobId);
};

class WhisperVadContext : public Napi::ObjectWrap<WhisperVadContext> {
public:
    WhisperVadContext(const Napi::CallbackInfo& info);
    ~WhisperVadContext();

    // Static methods
    static void ToggleNativeLog(const Napi::CallbackInfo& info);
    static Napi::Value ModelInfo(const Napi::CallbackInfo& info);
    static void Init(Napi::Env env, Napi::Object& exports);

private:
    // Instance methods
    Napi::Value GetSystemInfo(const Napi::CallbackInfo& info);
    Napi::Value GetModelInfo(const Napi::CallbackInfo& info);
    Napi::Value DetectSpeechFile(const Napi::CallbackInfo& info);
    Napi::Value DetectSpeechData(const Napi::CallbackInfo& info);
    Napi::Value Release(const Napi::CallbackInfo& info);

    // Internal data
    std::string _info;
    Napi::ObjectReference _meta;
    WhisperVadSessionPtr _sess = nullptr;
};
