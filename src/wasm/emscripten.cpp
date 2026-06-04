#include "whisper.h"

#include <emscripten/bind.h>
#include <emscripten/val.h>

#include <algorithm>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

using emscripten::val;

constexpr double kWasmMaximumMemoryBytes = 2000.0 * 1024.0 * 1024.0;
constexpr int kMaxWasmThreads = 8;

struct WhisperSession {
    std::string path;
    whisper_context * ctx = nullptr;
    std::mutex mutex;

    WhisperSession(std::string model_path, whisper_context * context)
        : path(std::move(model_path)), ctx(context) {}

    ~WhisperSession() {
        if (ctx) {
            whisper_free(ctx);
            ctx = nullptr;
        }
    }
};

struct VadSession {
    std::string path;
    whisper_vad_context * ctx = nullptr;
    std::mutex mutex;

    VadSession(std::string model_path, whisper_vad_context * context)
        : path(std::move(model_path)), ctx(context) {}

    ~VadSession() {
        if (ctx) {
            whisper_vad_free(ctx);
            ctx = nullptr;
        }
    }
};

std::mutex g_sessions_mutex;
int g_next_session_id = 1;
std::map<int, std::unique_ptr<WhisperSession>> g_whisper_sessions;
std::map<int, std::unique_ptr<VadSession>> g_vad_sessions;

val ok() {
    val result = val::object();
    result.set("ok", true);
    return result;
}

val error_result(const std::string & message) {
    val result = val::object();
    result.set("ok", false);
    result.set("error", message);
    return result;
}

bool has_property(const val & object, const char * name) {
    if (object.isUndefined() || object.isNull()) {
        return false;
    }

    val value = object[name];
    return !value.isUndefined() && !value.isNull();
}

bool get_bool(const val & object, const char * name, bool default_value) {
    if (!has_property(object, name)) {
        return default_value;
    }
    return object[name].as<bool>();
}

int get_int(const val & object, const char * name, int default_value) {
    if (!has_property(object, name)) {
        return default_value;
    }
    return object[name].as<int>();
}

float get_float(const val & object, const char * name, float default_value) {
    if (!has_property(object, name)) {
        return default_value;
    }
    return object[name].as<float>();
}

std::string get_string(const val & object, const char * name, const std::string & default_value = "") {
    if (!has_property(object, name)) {
        return default_value;
    }
    return object[name].as<std::string>();
}

int default_thread_count() {
#ifdef __EMSCRIPTEN_PTHREADS__
    const unsigned int hardware = std::thread::hardware_concurrency();
    return std::max(1, std::min(kMaxWasmThreads, static_cast<int>(hardware == 0 ? 1 : hardware)));
#else
    return 1;
#endif
}

int clamp_thread_count(int n_threads) {
#ifdef __EMSCRIPTEN_PTHREADS__
    return std::max(1, std::min(kMaxWasmThreads, n_threads));
#else
    (void) n_threads;
    return 1;
#endif
}

std::vector<float> copy_float32_array(const val & audio) {
    const int n_samples = audio["length"].as<int>();
    std::vector<float> pcmf32(n_samples);

    if (n_samples == 0) {
        return pcmf32;
    }

    val heap = val::module_property("HEAPU8");
    val memory = heap["buffer"];
    val memory_view = audio["constructor"].new_(
        memory,
        reinterpret_cast<uintptr_t>(pcmf32.data()),
        n_samples
    );
    memory_view.call<void>("set", audio);

    return pcmf32;
}

WhisperSession * get_whisper_session(int id) {
    std::lock_guard<std::mutex> lock(g_sessions_mutex);
    auto it = g_whisper_sessions.find(id);
    if (it == g_whisper_sessions.end()) {
        return nullptr;
    }
    return it->second.get();
}

VadSession * get_vad_session(int id) {
    std::lock_guard<std::mutex> lock(g_sessions_mutex);
    auto it = g_vad_sessions.find(id);
    if (it == g_vad_sessions.end()) {
        return nullptr;
    }
    return it->second.get();
}

whisper_full_params create_full_params(const val & options) {
    whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);

    params.print_realtime = false;
    params.print_progress = false;
    params.print_timestamps = false;
    params.print_special = false;
    params.n_threads = get_int(options, "maxThreads", default_thread_count());
    params.translate = get_bool(options, "translate", params.translate);
    params.n_max_text_ctx = get_int(options, "maxContext", params.n_max_text_ctx);
    params.max_len = get_int(options, "maxLen", params.max_len);
    params.token_timestamps = get_bool(options, "tokenTimestamps", params.token_timestamps);
    params.tdrz_enable = get_bool(options, "tdrzEnable", params.tdrz_enable);
    params.thold_pt = get_float(options, "wordThold", params.thold_pt);
    params.offset_ms = get_int(options, "offset", params.offset_ms);
    params.duration_ms = get_int(options, "duration", params.duration_ms);
    params.temperature = get_float(options, "temperature", params.temperature);
    params.temperature_inc = get_float(options, "temperatureInc", params.temperature_inc);
    params.beam_search.beam_size = get_int(options, "beamSize", params.beam_search.beam_size);
    params.greedy.best_of = get_int(options, "bestOf", params.greedy.best_of);

    params.n_threads = clamp_thread_count(params.n_threads);

    return params;
}

whisper_vad_params create_vad_params(const val & options) {
    whisper_vad_params params = whisper_vad_default_params();

    params.threshold = get_float(options, "threshold", params.threshold);
    params.min_speech_duration_ms = get_int(options, "minSpeechDurationMs", params.min_speech_duration_ms);
    params.min_silence_duration_ms = get_int(options, "minSilenceDurationMs", params.min_silence_duration_ms);
    params.max_speech_duration_s = get_float(options, "maxSpeechDurationS", params.max_speech_duration_s);
    params.speech_pad_ms = get_int(options, "speechPadMs", params.speech_pad_ms);
    params.samples_overlap = get_float(options, "samplesOverlap", params.samples_overlap);

    return params;
}

val create_segment(whisper_context * ctx, int index, bool tdrz_enable) {
    const char * text_cur = whisper_full_get_segment_text(ctx, index);
    std::string text = text_cur ? text_cur : "";

    if (tdrz_enable && whisper_full_get_segment_speaker_turn_next(ctx, index)) {
        text += " [SPEAKER_TURN]";
    }

    val segment = val::object();
    segment.set("text", text);
    segment.set("t0", whisper_full_get_segment_t0(ctx, index) * 10);
    segment.set("t1", whisper_full_get_segment_t1(ctx, index) * 10);
    return segment;
}

val create_transcribe_result(whisper_context * ctx, const std::string & text, bool aborted, bool tdrz_enable) {
    val result = ok();
    result.set("result", text);
    result.set("isAborted", aborted);

    if (ctx) {
        const int language_id = whisper_full_lang_id(ctx);
        result.set("language", whisper_lang_str(language_id));
    }

    val segments = val::array();
    if (ctx && !aborted) {
        const int n_segments = whisper_full_n_segments(ctx);
        for (int i = 0; i < n_segments; ++i) {
            segments.call<void>("push", create_segment(ctx, i, tdrz_enable));
        }
    }
    result.set("segments", segments);

    return result;
}

struct TranscribeCallbackContext {
    val on_progress;
    val on_new_segments;
    bool has_progress = false;
    bool has_new_segments = false;
    bool tdrz_enable = false;
    int total_n_new = 0;

    TranscribeCallbackContext()
        : on_progress(val::undefined()), on_new_segments(val::undefined()) {}
};

void progress_callback(whisper_context *, whisper_state *, int progress, void * user_data) {
    auto * context = static_cast<TranscribeCallbackContext *>(user_data);
    if (!context || !context->has_progress) {
        return;
    }

    context->on_progress(progress);
}

void new_segment_callback(whisper_context * ctx, whisper_state *, int n_new, void * user_data) {
    auto * context = static_cast<TranscribeCallbackContext *>(user_data);
    if (!context || !context->has_new_segments) {
        return;
    }

    context->total_n_new += n_new;

    val segments = val::array();
    std::string text;
    const int first = context->total_n_new - n_new;
    for (int i = first; i < context->total_n_new; ++i) {
        const char * text_cur = whisper_full_get_segment_text(ctx, i);
        std::string segment_text = text_cur ? text_cur : "";

        if (context->tdrz_enable && whisper_full_get_segment_speaker_turn_next(ctx, i)) {
            segment_text += " [SPEAKER_TURN]";
        }

        text += segment_text;

        val segment = val::object();
        segment.set("text", segment_text);
        segment.set("t0", whisper_full_get_segment_t0(ctx, i) * 10);
        segment.set("t1", whisper_full_get_segment_t1(ctx, i) * 10);
        segments.call<void>("push", segment);
    }

    val result = val::object();
    result.set("nNew", n_new);
    result.set("totalNNew", context->total_n_new);
    result.set("result", text);
    result.set("segments", segments);

    context->on_new_segments(result);
}

val init_whisper(const std::string & model_path, bool use_gpu, bool use_flash_attn) {
    if (model_path.empty()) {
        return error_result("Model path is required");
    }

    whisper_context_params params = whisper_context_default_params();
    params.use_gpu = use_gpu;
    params.gpu_device = 0;
    params.flash_attn = use_flash_attn;

    whisper_context * ctx = whisper_init_from_file_with_params(model_path.c_str(), params);
    if (!ctx) {
        return error_result("Failed to initialize whisper context");
    }

    int id = 0;
    {
        std::lock_guard<std::mutex> lock(g_sessions_mutex);
        id = g_next_session_id++;
        g_whisper_sessions[id] = std::make_unique<WhisperSession>(model_path, ctx);
    }

    val result = ok();
    result.set("id", id);
    result.set("filePath", model_path);
    result.set("useGpu", use_gpu);
    result.set("useFlashAttn", use_flash_attn);
    return result;
}

val init_vad(const std::string & model_path, bool use_gpu, int n_threads) {
    if (model_path.empty()) {
        return error_result("Model path is required");
    }

    whisper_vad_context_params params = whisper_vad_default_context_params();
    params.use_gpu = use_gpu;
    params.gpu_device = 0;
    params.n_threads = clamp_thread_count(n_threads);

    whisper_vad_context * ctx = whisper_vad_init_from_file_with_params(model_path.c_str(), params);
    if (!ctx) {
        return error_result("Failed to initialize whisper vad context");
    }

    int id = 0;
    {
        std::lock_guard<std::mutex> lock(g_sessions_mutex);
        id = g_next_session_id++;
        g_vad_sessions[id] = std::make_unique<VadSession>(model_path, ctx);
    }

    val result = ok();
    result.set("id", id);
    result.set("filePath", model_path);
    result.set("useGpu", use_gpu);
    result.set("nThreads", params.n_threads);
    return result;
}

void free_whisper(int id) {
    std::lock_guard<std::mutex> lock(g_sessions_mutex);
    g_whisper_sessions.erase(id);
}

void free_vad(int id) {
    std::lock_guard<std::mutex> lock(g_sessions_mutex);
    g_vad_sessions.erase(id);
}

val transcribe(int id, const val & audio, const val & options) {
    WhisperSession * session = get_whisper_session(id);
    if (!session || !session->ctx) {
        return error_result("Invalid whisper context");
    }

    std::vector<float> pcmf32 = copy_float32_array(audio);
    if (pcmf32.empty()) {
        return create_transcribe_result(nullptr, "", false, false);
    }

    std::lock_guard<std::mutex> lock(session->mutex);
    if (!session->ctx) {
        return error_result("Whisper context was destroyed");
    }

    whisper_full_params params = create_full_params(options);
    const int n_processors = std::max(1, get_int(options, "nProcessors", 1));
    std::string language = get_string(options, "language");
    std::string prompt = get_string(options, "prompt");

    if (!language.empty()) {
        params.language = language.c_str();
    }
    if (!prompt.empty()) {
        params.initial_prompt = prompt.c_str();
    }

    TranscribeCallbackContext callback_context;
    callback_context.has_progress = has_property(options, "onProgress");
    callback_context.has_new_segments = has_property(options, "onNewSegments");
    callback_context.tdrz_enable = params.tdrz_enable;

    if (callback_context.has_progress) {
        callback_context.on_progress = options["onProgress"];
        params.progress_callback = progress_callback;
        params.progress_callback_user_data = &callback_context;
    }

    if (callback_context.has_new_segments) {
        callback_context.on_new_segments = options["onNewSegments"];
        params.new_segment_callback = new_segment_callback;
        params.new_segment_callback_user_data = &callback_context;
    }

    whisper_reset_timings(session->ctx);
    const int result = whisper_full_parallel(
        session->ctx,
        params,
        pcmf32.data(),
        static_cast<int>(pcmf32.size()),
        n_processors
    );

    if (result != 0) {
        return error_result("Transcription failed: " + std::to_string(result));
    }

    std::stringstream text;
    const int n_segments = whisper_full_n_segments(session->ctx);
    for (int i = 0; i < n_segments; ++i) {
        const char * segment_text = whisper_full_get_segment_text(session->ctx, i);
        if (segment_text) {
            text << segment_text;
        }
    }

    return create_transcribe_result(session->ctx, text.str(), false, params.tdrz_enable);
}

val detect_speech(int id, const val & audio, const val & options) {
    VadSession * session = get_vad_session(id);
    if (!session || !session->ctx) {
        return error_result("Invalid VAD context");
    }

    std::vector<float> pcmf32 = copy_float32_array(audio);
    if (pcmf32.empty()) {
        return error_result("Empty audio data");
    }

    std::lock_guard<std::mutex> lock(session->mutex);
    if (!session->ctx) {
        return error_result("VAD context was destroyed");
    }

    whisper_vad_params params = create_vad_params(options);
    const bool has_speech = whisper_vad_detect_speech(session->ctx, pcmf32.data(), static_cast<int>(pcmf32.size()));

    val result = ok();
    val segments = val::array();

    if (has_speech) {
        whisper_vad_segments * vad_segments = whisper_vad_segments_from_samples(
            session->ctx,
            params,
            pcmf32.data(),
            static_cast<int>(pcmf32.size())
        );

        if (vad_segments) {
            const int n_segments = whisper_vad_segments_n_segments(vad_segments);
            for (int i = 0; i < n_segments; ++i) {
                val segment = val::object();
                segment.set("t0", whisper_vad_segments_get_segment_t0(vad_segments, i));
                segment.set("t1", whisper_vad_segments_get_segment_t1(vad_segments, i));
                segments.call<void>("push", segment);
            }
            whisper_vad_free_segments(vad_segments);
        }
    }

    result.set("segments", segments);
    return result;
}

val bench(int id, int n_threads) {
    WhisperSession * session = get_whisper_session(id);
    if (!session || !session->ctx) {
        return error_result("Invalid whisper context");
    }

    std::lock_guard<std::mutex> lock(session->mutex);
    if (!session->ctx) {
        return error_result("Whisper context was destroyed");
    }

    whisper_context * ctx = session->ctx;
    n_threads = clamp_thread_count(n_threads);

    const int n_mels = whisper_model_n_mels(ctx);
    int ret = whisper_set_mel(ctx, nullptr, 0, n_mels);
    if (ret != 0) {
        return error_result("Failed to set mel: " + std::to_string(ret));
    }

    ret = whisper_encode(ctx, 0, n_threads);
    if (ret != 0) {
        return error_result("Failed to encode: " + std::to_string(ret));
    }

    whisper_token tokens[512];
    std::memset(tokens, 0, sizeof(tokens));

    ret = whisper_decode(ctx, tokens, 256, 0, n_threads);
    if (ret != 0) {
        return error_result("Failed to decode prompt: " + std::to_string(ret));
    }

    ret = whisper_decode(ctx, tokens, 1, 256, n_threads);
    if (ret != 0) {
        return error_result("Failed to decode token: " + std::to_string(ret));
    }

    whisper_reset_timings(ctx);

    ret = whisper_encode(ctx, 0, n_threads);
    if (ret != 0) {
        return error_result("Failed to benchmark encode: " + std::to_string(ret));
    }

    for (int i = 0; i < 256; ++i) {
        ret = whisper_decode(ctx, tokens, 1, i, n_threads);
        if (ret != 0) {
            return error_result("Failed to benchmark decode: " + std::to_string(ret));
        }
    }

    for (int i = 0; i < 64; ++i) {
        ret = whisper_decode(ctx, tokens, 5, 0, n_threads);
        if (ret != 0) {
            return error_result("Failed to benchmark batch decode: " + std::to_string(ret));
        }
    }

    for (int i = 0; i < 16; ++i) {
        ret = whisper_decode(ctx, tokens, 256, 0, n_threads);
        if (ret != 0) {
            return error_result("Failed to benchmark prompt: " + std::to_string(ret));
        }
    }

    const whisper_timings * timings = whisper_get_timings(ctx);
    val result = ok();
    result.set("config", whisper_print_system_info());
    result.set("nThreads", n_threads);
    result.set("encodeMs", timings ? timings->encode_ms : 0.0f);
    result.set("decodeMs", timings ? timings->decode_ms : 0.0f);
    result.set("batchdMs", timings ? timings->batchd_ms : 0.0f);
    result.set("promptMs", timings ? timings->prompt_ms : 0.0f);
    return result;
}

val system_info() {
    val result = ok();
    result.set("systemInfo", whisper_print_system_info());
    return result;
}

bool webgpu_enabled() {
#ifdef GGML_USE_WEBGPU
    return true;
#else
    return false;
#endif
}

double wasm_maximum_memory_bytes() {
    return kWasmMaximumMemoryBytes;
}

} // namespace

EMSCRIPTEN_BINDINGS(whisper_node_wasm) {
    emscripten::function("__wasm_init_whisper", &init_whisper);
    emscripten::function("__wasm_free_whisper", &free_whisper);
    emscripten::function("__wasm_transcribe", &transcribe);
    emscripten::function("__wasm_bench", &bench);

    emscripten::function("__wasm_init_vad", &init_vad);
    emscripten::function("__wasm_free_vad", &free_vad);
    emscripten::function("__wasm_detect_speech", &detect_speech);

    emscripten::function("__wasm_system_info", &system_info);
    emscripten::function("__wasm_webgpu_enabled", &webgpu_enabled);
    emscripten::function("__wasm_maximum_memory_bytes", &wasm_maximum_memory_bytes);
}
