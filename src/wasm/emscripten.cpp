#include "whisper.h"
#include "parakeet.h"
#include "vad.h"

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

// vad.cpp model and the streams created on it (freed before the model)
struct VadSession {
    std::string path;
    vad_model * ctx = nullptr;
    std::mutex mutex;
    int sample_rate = 0;
    int hop_samples = 0;
    std::string arch;
    int next_stream_id = 1;
    std::map<int, vad_stream *> streams;

    VadSession(std::string model_path, vad_model * model)
        : path(std::move(model_path)), ctx(model) {
        sample_rate = vad_model_sample_rate(ctx);
        hop_samples = vad_model_hop_samples(ctx);
        arch = vad_model_arch(ctx);
    }

    ~VadSession() {
        for (auto & entry : streams) {
            vad_stream_free(entry.second);
        }
        streams.clear();
        if (ctx) {
            vad_model_free(ctx);
            ctx = nullptr;
        }
    }
};

struct ParakeetSession {
    std::string path;
    parakeet_context * ctx = nullptr;
    std::mutex mutex;

    ParakeetSession(std::string model_path, parakeet_context * context)
        : path(std::move(model_path)), ctx(context) {}

    ~ParakeetSession() {
        if (ctx) {
            parakeet_free(ctx);
            ctx = nullptr;
        }
    }
};

std::mutex g_sessions_mutex;
int g_next_session_id = 1;
std::map<int, std::unique_ptr<WhisperSession>> g_whisper_sessions;
std::map<int, std::unique_ptr<VadSession>> g_vad_sessions;
std::map<int, std::unique_ptr<ParakeetSession>> g_parakeet_sessions;

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

ParakeetSession * get_parakeet_session(int id) {
    std::lock_guard<std::mutex> lock(g_sessions_mutex);
    auto it = g_parakeet_sessions.find(id);
    if (it == g_parakeet_sessions.end()) {
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

vad_segment_params create_vad_params(const val & options) {
    vad_segment_params params = vad_segment_default_params();

    params.threshold = get_float(options, "threshold", params.threshold);
    params.neg_threshold = get_float(options, "negThreshold", params.neg_threshold);
    params.min_speech_ms = get_int(options, "minSpeechDurationMs", params.min_speech_ms);
    params.min_silence_ms = get_int(options, "minSilenceDurationMs", params.min_silence_ms);
    params.max_speech_s = get_float(options, "maxSpeechDurationS", params.max_speech_s);
    params.speech_pad_ms = get_int(options, "speechPadMs", params.speech_pad_ms);
    // samplesOverlap (whisper.cpp's windowed VAD) has no counterpart in vad.cpp

    return params;
}

parakeet_full_params create_parakeet_full_params(const val & options) {
    parakeet_full_params params = parakeet_full_default_params(PARAKEET_SAMPLING_GREEDY);

    params.n_threads = get_int(options, "maxThreads", default_thread_count());
    params.audio_ctx = get_int(options, "audioCtx", params.audio_ctx);
    params.no_context = true;

    params.n_threads = clamp_thread_count(params.n_threads);

    return params;
}

val create_parakeet_transcribe_result(parakeet_context * ctx, bool aborted) {
    val result = ok();
    result.set("isAborted", aborted);
    result.set("language", "");

    std::string text;
    val segments = val::array();
    if (ctx) {
        const int n_segments = parakeet_full_n_segments(ctx);
        for (int i = 0; i < n_segments; ++i) {
            const char * segment_text = parakeet_full_get_segment_text(ctx, i);
            std::string segment = segment_text ? segment_text : "";
            text += segment;

            val segment_obj = val::object();
            segment_obj.set("text", segment);
            segment_obj.set("t0", static_cast<double>(parakeet_full_get_segment_t0(ctx, i) * 10));
            segment_obj.set("t1", static_cast<double>(parakeet_full_get_segment_t1(ctx, i) * 10));
            segments.call<void>("push", segment_obj);
        }
    }
    result.set("result", text);
    result.set("segments", segments);

    return result;
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

val init_vad(const std::string & model_path, bool /* use_gpu: VAD always runs on the CPU */, int n_threads) {
    if (model_path.empty()) {
        return error_result("Model path is required");
    }

    vad_model_params params = vad_model_default_params();
    params.n_threads = clamp_thread_count(n_threads > 0 ? n_threads : 1);

    vad_model * model = vad_model_load(model_path.c_str(), params);
    if (!model) {
        return error_result("Failed to load the VAD model");
    }
    if (vad_model_output(model) != VAD_OUTPUT_SPEECH || vad_model_n_outputs(model) != 1) {
        std::string arch = vad_model_arch(model);
        vad_model_free(model);
        return error_result("Unsupported VAD model (" + arch + "): only speech detection models are supported");
    }
    if (vad_model_sample_rate(model) != WHISPER_SAMPLE_RATE) {
        vad_model_free(model);
        return error_result("Unsupported VAD model: expected a 16 kHz model");
    }

    int id = 0;
    VadSession * session = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_sessions_mutex);
        id = g_next_session_id++;
        auto owned = std::make_unique<VadSession>(model_path, model);
        session = owned.get();
        g_vad_sessions[id] = std::move(owned);
    }

    val result = ok();
    result.set("id", id);
    result.set("filePath", model_path);
    result.set("useGpu", false);
    result.set("nThreads", params.n_threads);
    result.set("sampleRate", session->sample_rate);
    result.set("hopSamples", session->hop_samples);
    result.set("arch", session->arch);
    return result;
}

val init_parakeet(const std::string & model_path, bool use_gpu) {
    if (model_path.empty()) {
        return error_result("Model path is required");
    }

    parakeet_context_params params = parakeet_context_default_params();
    params.use_gpu = use_gpu;
    params.gpu_device = 0;

    parakeet_context * ctx = parakeet_init_from_file_with_params(model_path.c_str(), params);
    if (!ctx) {
        return error_result("Failed to initialize parakeet context");
    }

    int id = 0;
    {
        std::lock_guard<std::mutex> lock(g_sessions_mutex);
        id = g_next_session_id++;
        g_parakeet_sessions[id] = std::make_unique<ParakeetSession>(model_path, ctx);
    }

    val result = ok();
    result.set("id", id);
    result.set("filePath", model_path);
    result.set("useGpu", use_gpu);
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

void free_parakeet(int id) {
    std::lock_guard<std::mutex> lock(g_sessions_mutex);
    g_parakeet_sessions.erase(id);
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

val transcribe_parakeet(int id, const val & audio, const val & options) {
    ParakeetSession * session = get_parakeet_session(id);
    if (!session || !session->ctx) {
        return error_result("Invalid parakeet context");
    }

    std::vector<float> pcmf32 = copy_float32_array(audio);
    if (pcmf32.empty()) {
        return create_parakeet_transcribe_result(nullptr, false);
    }

    std::lock_guard<std::mutex> lock(session->mutex);
    if (!session->ctx) {
        return error_result("Parakeet context was destroyed");
    }

    parakeet_full_params params = create_parakeet_full_params(options);

    const int result = parakeet_full(
        session->ctx,
        params,
        pcmf32.data(),
        static_cast<int>(pcmf32.size())
    );

    if (result != 0) {
        return error_result("Parakeet transcription failed: " + std::to_string(result));
    }

    return create_parakeet_transcribe_result(session->ctx, false);
}

// Offline detection: probabilities for the whole buffer, then silero-vad's
// segmentation. Segment times are in 10 ms units, as before.
val detect_speech(int id, const val & audio, const val & options) {
    VadSession * session = get_vad_session(id);
    if (!session || !session->ctx) {
        return error_result("Invalid VAD context");
    }

    std::vector<float> pcmf32 = copy_float32_array(audio);

    std::lock_guard<std::mutex> lock(session->mutex);
    if (!session->ctx) {
        return error_result("VAD context was destroyed");
    }

    val result = ok();
    val segments = val::array();
    result.set("segments", segments);
    if (pcmf32.empty()) {
        return result;
    }

    const vad_segment_params params = create_vad_params(options);
    const int hop = session->hop_samples;
    const int n_samples = static_cast<int>(pcmf32.size());
    std::vector<float> probs(static_cast<size_t>(n_samples / hop + 1));
    int n_probs = vad_detect(session->ctx, pcmf32.data(), n_samples, probs.data(), static_cast<int>(probs.size()));
    if (n_probs < 0) {
        return error_result("VAD detection failed");
    }
    n_probs = std::min(n_probs, static_cast<int>(probs.size()));

    std::vector<vad_segment> found(16);
    for (;;) {
        int n = vad_segments_from_probs(
            probs.data(), n_probs, 1, hop, session->sample_rate, n_samples,
            params, found.data(), static_cast<int>(found.size()));
        const bool fits = n <= static_cast<int>(found.size());
        found.resize(static_cast<size_t>(n));
        if (fits) break;
    }

    const double to_centiseconds = 100.0 / session->sample_rate;
    for (const auto & item : found) {
        val segment = val::object();
        segment.set("t0", static_cast<float>(item.start * to_centiseconds));
        segment.set("t1", static_cast<float>(item.end * to_centiseconds));
        segments.call<void>("push", segment);
    }
    return result;
}

val probs_result(const std::vector<float> & probs) {
    val result = ok();
    val array = val::array();
    for (float prob : probs) {
        array.call<void>("push", prob);
    }
    result.set("probs", array);
    return result;
}

val vad_stream_create(int id) {
    VadSession * session = get_vad_session(id);
    if (!session || !session->ctx) {
        return error_result("Invalid VAD context");
    }
    std::lock_guard<std::mutex> lock(session->mutex);
    vad_stream * stream = vad_stream_new(session->ctx);
    if (!stream) {
        return error_result("Failed to create a VAD stream");
    }
    const int stream_id = session->next_stream_id++;
    session->streams[stream_id] = stream;
    val result = ok();
    result.set("streamId", stream_id);
    return result;
}

// Runs `op(stream)` on a stream of a VAD context, under the context lock
template <typename Op>
val with_vad_stream(int id, int stream_id, Op op) {
    VadSession * session = get_vad_session(id);
    if (!session) {
        return error_result("VAD stream not found");
    }
    std::lock_guard<std::mutex> lock(session->mutex);
    auto it = session->streams.find(stream_id);
    if (it == session->streams.end()) {
        return error_result("VAD stream not found");
    }
    return op(it->second);
}

val vad_stream_push_audio(int id, int stream_id, const val & audio) {
    std::vector<float> pcmf32 = copy_float32_array(audio);
    return with_vad_stream(id, stream_id, [&](vad_stream * stream) {
        const int n_samples = static_cast<int>(pcmf32.size());
        std::vector<float> probs(static_cast<size_t>(std::max(vad_stream_n_pending(stream, n_samples), 0)));
        int n = vad_stream_push(stream, pcmf32.data(), n_samples, probs.data(), static_cast<int>(probs.size()));
        if (n < 0) {
            return error_result("VAD stream push failed");
        }
        probs.resize(static_cast<size_t>(n));
        return probs_result(probs);
    });
}

val vad_stream_flush_audio(int id, int stream_id) {
    return with_vad_stream(id, stream_id, [](vad_stream * stream) {
        std::vector<float> probs(static_cast<size_t>(std::max(vad_stream_n_flush(stream), 0)));
        int n = vad_stream_flush(stream, probs.data(), static_cast<int>(probs.size()));
        if (n < 0) {
            return error_result("VAD stream flush failed");
        }
        probs.resize(static_cast<size_t>(n));
        return probs_result(probs);
    });
}

val vad_stream_reset_state(int id, int stream_id) {
    return with_vad_stream(id, stream_id, [](vad_stream * stream) {
        vad_stream_reset(stream);
        return ok();
    });
}

void vad_stream_release(int id, int stream_id) {
    VadSession * session = get_vad_session(id);
    if (!session) {
        return;  // released with its context
    }
    std::lock_guard<std::mutex> lock(session->mutex);
    auto it = session->streams.find(stream_id);
    if (it != session->streams.end()) {
        vad_stream_free(it->second);
        session->streams.erase(it);
    }
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
    emscripten::function("__wasm_vad_stream_create", &vad_stream_create);
    emscripten::function("__wasm_vad_stream_push", &vad_stream_push_audio);
    emscripten::function("__wasm_vad_stream_flush", &vad_stream_flush_audio);
    emscripten::function("__wasm_vad_stream_reset", &vad_stream_reset_state);
    emscripten::function("__wasm_vad_stream_release", &vad_stream_release);

    emscripten::function("__wasm_init_parakeet", &init_parakeet);
    emscripten::function("__wasm_free_parakeet", &free_parakeet);
    emscripten::function("__wasm_transcribe_parakeet", &transcribe_parakeet);

    emscripten::function("__wasm_system_info", &system_info);
    emscripten::function("__wasm_webgpu_enabled", &webgpu_enabled);
    emscripten::function("__wasm_maximum_memory_bytes", &wasm_maximum_memory_bytes);
}
