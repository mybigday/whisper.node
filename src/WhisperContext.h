#pragma once

#include "common.hpp"

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
    Napi::Value Release(const Napi::CallbackInfo& info);

    // Internal data
    std::string _info;
    Napi::Object _meta;
    WhisperSessionPtr _sess = nullptr;
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
    Napi::Object _meta;
    WhisperVadSessionPtr _sess = nullptr;
};
