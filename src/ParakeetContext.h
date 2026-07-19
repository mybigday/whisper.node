#pragma once

#include "common.hpp"
#include <atomic>
#include <unordered_map>

#ifndef USE_GPU
#ifdef NO_GPU_SUPPORT
#define USE_GPU false
#else
#define USE_GPU true
#endif
#endif

class ParakeetContext : public Napi::ObjectWrap<ParakeetContext> {
public:
    ParakeetContext(const Napi::CallbackInfo& info);
    ~ParakeetContext();

    // Static methods
    static void ToggleNativeLog(const Napi::CallbackInfo& info);
    static Napi::Value ModelInfo(const Napi::CallbackInfo& info);
    static void Init(Napi::Env env, Napi::Object& exports);

private:
    // Instance methods
    Napi::Value GetModelInfo(const Napi::CallbackInfo& info);
    Napi::Value TranscribeFile(const Napi::CallbackInfo& info);
    Napi::Value TranscribeData(const Napi::CallbackInfo& info);
    Napi::Value AbortTranscribe(const Napi::CallbackInfo& info);
    Napi::Value Release(const Napi::CallbackInfo& info);

    // Internal data
    Napi::ObjectReference _meta;
    ParakeetSessionPtr _sess = nullptr;

    // Job tracking for cancellation
    std::atomic<int> _nextJobId{1};
    std::unordered_map<int, std::shared_ptr<std::atomic<bool>>> _cancelFlags;
    std::mutex _cancelMutex;

public:
    // Public method to register a job for cancellation
    int registerJob(std::shared_ptr<std::atomic<bool>> cancelFlag);
    void unregisterJob(int jobId);
};
