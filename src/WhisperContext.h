#pragma once

#include "common.hpp"
#include <atomic>
#include <deque>
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
    static void ReleaseAllSync(const Napi::CallbackInfo& info);
    static void Init(Napi::Env env, Napi::Object& exports);

    void releaseSync();

private:
    // Instance methods
    Napi::Value GetSystemInfo(const Napi::CallbackInfo& info);
    Napi::Value GetModelInfo(const Napi::CallbackInfo& info);
    void ReleaseSync(const Napi::CallbackInfo& info);
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
    static void ReleaseAllSync(const Napi::CallbackInfo& info);
    static void Init(Napi::Env env, Napi::Object& exports);

    void releaseSync();

private:
    // Instance methods
    Napi::Value GetModelInfo(const Napi::CallbackInfo& info);
    Napi::Value GetSampleRate(const Napi::CallbackInfo& info);
    Napi::Value GetHopSamples(const Napi::CallbackInfo& info);
    Napi::Value GetArch(const Napi::CallbackInfo& info);
    void ReleaseSync(const Napi::CallbackInfo& info);
    Napi::Value DetectSpeechFile(const Napi::CallbackInfo& info);
    Napi::Value DetectSpeechData(const Napi::CallbackInfo& info);
    Napi::Value CreateStream(const Napi::CallbackInfo& info);
    Napi::Value Release(const Napi::CallbackInfo& info);

    // Internal data
    Napi::ObjectReference _meta;
    WhisperVadSessionPtr _sess = nullptr;
    int _sampleRate = 0;
    int _hopSamples = 0;
    std::string _arch;
};

// Stateful VAD over live audio, created by WhisperVadContext.createStream().
// Calls on one stream run in call order: each one waits for the previous to
// settle (a FIFO of not yet queued workers, only touched on the JS thread).
class WhisperVadStream : public Napi::ObjectWrap<WhisperVadStream> {
public:
    WhisperVadStream(const Napi::CallbackInfo& info);
    ~WhisperVadStream();

    static void Init(Napi::Env env, Napi::Object& exports);
    static Napi::Object NewInstance(
        Napi::Env env,
        WhisperVadSessionPtr session,
        std::shared_ptr<VadStreamState> state);

    // Called from a finished stream worker (JS thread) to start the next call
    void runNext();

private:
    Napi::Value GetSampleRate(const Napi::CallbackInfo& info);
    Napi::Value GetHopSamples(const Napi::CallbackInfo& info);
    Napi::Value Push(const Napi::CallbackInfo& info);
    Napi::Value Flush(const Napi::CallbackInfo& info);
    Napi::Value Reset(const Napi::CallbackInfo& info);
    Napi::Value Release(const Napi::CallbackInfo& info);

    void enqueue(Napi::AsyncWorker* worker);

    WhisperVadSessionPtr _sess = nullptr;
    std::shared_ptr<VadStreamState> _state;
    int _sampleRate = 0;
    int _hopSamples = 0;
    std::deque<Napi::AsyncWorker*> _pending;
    bool _running = false;
};
