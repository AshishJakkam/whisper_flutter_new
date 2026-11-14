#include "whisper_flutter_new.h"
#include "whisper.cpp/whisper.h"

#define DR_WAV_IMPLEMENTATION
#include "whisper.cpp/examples/dr_wav.h"

#include <cstdio>
#include <memory>
// OMI_WHISPER_RESOURCE_GUARDS_V1
#include <string>
#include <thread>
#include <mutex>
#include <vector>
#include <cmath>
#include <iostream>
#include <stdio.h>
#include "json/json.hpp"

using json = nlohmann::json;

char *jsonToChar(json jsonData) noexcept
{
    std::string result = jsonData.dump();
    char *ch = new char[result.size() + 1];
    strcpy(ch, result.c_str());
    return ch;
}

struct whisper_params
{
    int32_t seed = -1; // RNG seed, not used currently
    int32_t n_threads = std::min(4, (int32_t)std::thread::hardware_concurrency());

    int32_t n_processors = 1;
    int32_t offset_t_ms = 0;
    int32_t offset_n = 0;
    int32_t duration_ms = 0;
    int32_t max_context = -1;
    int32_t max_len = 0;
    int32_t best_of = 5;
    int32_t beam_size = -1;

    float word_thold = 0.01f;
    float entropy_thold = 2.40f;
    float logprob_thold = -1.00f;

    bool verbose = false;
    bool print_special_tokens = false;
    bool speed_up = false;
    bool translate = false;
    bool diarize = false;
    bool no_fallback = false;
    bool output_txt = false;
    bool output_vtt = false;
    bool output_srt = false;
    bool output_wts = false;
    bool output_csv = false;
    bool print_special = false;
    bool print_colors = false;
    bool print_progress = false;
    bool no_timestamps = false;
    bool split_on_word = false;

    std::string language = "auto";
    std::string prompt;
    std::string model = "models/ggml-tiny.bin";
    std::string audio = "samples/jfk.wav";
    std::vector<std::string> fname_inp = {};
    std::vector<std::string> fname_outp = {};
};

struct whisper_print_user_data
{
    const whisper_params *params;

    const std::vector<std::vector<float>> *pcmf32s;
};

// Reads a 16 kHz, 16-bit mono/stereo WAV into mono float samples.
static bool loadWav16k(const std::string &path, std::vector<float> &pcmf32, std::string &error)
{
    drwav wav;
    if (!drwav_init_file(&wav, path.c_str(), NULL))
    {
        error = "OMI_STT_PERMANENT_INPUT: failed to open WAV file";
        return false;
    }
    // RAII owns the decoder across every error return below.
    auto wav_cleanup = [](drwav *handle) { drwav_uninit(handle); };
    std::unique_ptr<drwav, decltype(wav_cleanup)> wav_guard(&wav, wav_cleanup);

    if (wav.channels != 1 && wav.channels != 2)
    {
        error = "OMI_STT_PERMANENT_INPUT: must be mono or stereo";
        return false;
    }
    if (wav.sampleRate != WHISPER_SAMPLE_RATE)
    {
        error = "OMI_STT_PERMANENT_INPUT: WAV file must be 16 kHz";
        return false;
    }
    if (wav.bitsPerSample != 16)
    {
        error = "OMI_STT_PERMANENT_INPUT: WAV file must be 16 bit";
        return false;
    }

    int n = wav.totalPCMFrameCount;
    std::vector<int16_t> pcm16;
    pcm16.resize(n * wav.channels);
    drwav_read_pcm_frames_s16(&wav, n, pcm16.data());

    pcmf32.resize(n);
    if (wav.channels == 1)
    {
        for (int i = 0; i < n; i++)
        {
            pcmf32[i] = float(pcm16[i]) / 32768.0f;
        }
    }
    else
    {
        for (int i = 0; i < n; i++)
        {
            pcmf32[i] = float(pcm16[2 * i] + pcm16[2 * i + 1]) / 65536.0f;
        }
    }
    return true;
}

// Parses the wire fields both entry points accept.
static whisper_params parseParams(json jsonBody)
{
    whisper_params params;
    params.n_threads = jsonBody["threads"];
    params.verbose = jsonBody["is_verbose"];
    params.translate = jsonBody["is_translate"];
    params.language = jsonBody["language"];
    params.print_special_tokens = jsonBody["is_special_tokens"];
    params.no_timestamps = jsonBody["is_no_timestamps"];
    params.model = jsonBody["model"];
    params.audio = jsonBody["audio"];
    params.split_on_word = jsonBody["split_on_word"];
    return params;
}

// One inference pass over an already-loaded model. The caller owns the context's lifetime.
static json runInference(struct whisper_context *ctx, const whisper_params &params)
{
    json jsonResult;
    jsonResult["@type"] = "transcribe";

    std::vector<float> pcmf32;
    std::string error;
    if (!loadWav16k(params.audio, pcmf32, error))
    {
        jsonResult["@type"] = "error";
        jsonResult["message"] = error;
        return jsonResult;
    }

    std::string language = params.language;
    bool translate = params.translate;
    if (language == "" || language == "auto")
    {
        language = "auto";
        translate = false;
    }

    whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    wparams.print_realtime = false;
    wparams.print_progress = false;
    wparams.print_timestamps = !params.no_timestamps;
    wparams.translate = translate;
    wparams.language = language.c_str();
    wparams.n_threads = params.n_threads;
    wparams.split_on_word = params.split_on_word;

    if (params.split_on_word)
    {
        wparams.max_len = 1;
        wparams.token_timestamps = true;
    }

    if (whisper_full(ctx, wparams, pcmf32.data(), pcmf32.size()) != 0)
    {
        jsonResult["@type"] = "error";
        jsonResult["message"] = "OMI_STT_RETRYABLE: failed to process audio";
        return jsonResult;
    }

    std::string text_result = "";
    const int n_segments = whisper_full_n_segments(ctx);
    std::vector<json> segmentsJson = {};

    for (int i = 0; i < n_segments; ++i)
    {
        const char *text = whisper_full_get_segment_text(ctx, i);
        text_result += std::string(text);
        if (!params.no_timestamps)
        {
            json jsonSegment;
            jsonSegment["from_ts"] = whisper_full_get_segment_t0(ctx, i);
            jsonSegment["to_ts"] = whisper_full_get_segment_t1(ctx, i);
            jsonSegment["text"] = text;
            segmentsJson.push_back(jsonSegment);
        }
    }

    if (!params.no_timestamps)
    {
        jsonResult["segments"] = segmentsJson;
    }
    jsonResult["text"] = text_result;
    return jsonResult;
}

// A model kept loaded for a live transcription session.
//
// Initialising whisper reads the whole model from flash. Measured in the app that was ~12 s of
// every single call, which is the only reason a 5 s window took over 13 s and the live transcript
// fell further behind. With the context kept, the same phone runs several times faster than
// realtime, so the per-window cost becomes the audio length rather than the model size.
struct whisper_session
{
    struct whisper_context *ctx = nullptr;
    std::string model;
    // The live socket and the batch runner both open a session over the same model. Without a
    // count, the first caller to close freed the context under the other one, and every later
    // window failed as if the model were missing.
    int refs = 0;
};

static whisper_session g_session;
static std::mutex g_session_mutex;

json transcribe(json jsonBody) noexcept
{
    whisper_params params = parseParams(jsonBody);
    json jsonResult;
    jsonResult["@type"] = "transcribe";

    if (params.language != "" && params.language != "auto" && whisper_lang_id(params.language.c_str()) == -1)
    {
        jsonResult["@type"] = "error";
        jsonResult["message"] = "OMI_STT_PERMANENT_INPUT: unknown language = " + params.language;
        return jsonResult;
    }

    // The per-call path owns its context for the length of one inference.
    std::unique_ptr<whisper_context, decltype(&whisper_free)> context(
        whisper_init_from_file(params.model.c_str()), &whisper_free);
    if (context.get() == nullptr)
    {
        jsonResult["@type"] = "error";
        jsonResult["message"] = "OMI_STT_MODEL_UNAVAILABLE: model could not be loaded";
        return jsonResult;
    }

    return runInference(context.get(), params);
}

extern "C"
{
    // Session API: keeps one whisper_context alive across calls. Actions are
    // sessionOpen {model}, getTextFromWavFile (same fields as request), sessionClose.
    FUNCTION_ATTRIBUTE
    char *session_request(char *body)
    {
        try
        {
            json jsonBody = json::parse(body);
            json jsonResult;
            const std::string type = jsonBody["@type"];

            std::lock_guard<std::mutex> lock(g_session_mutex);

            if (type == "sessionOpen")
            {
                const std::string model = jsonBody["model"];
                if (g_session.ctx != nullptr && g_session.model != model && g_session.refs <= 1)
                {
                    whisper_free(g_session.ctx);
                    g_session.ctx = nullptr;
                    g_session.refs = 0;
                }
                if (g_session.ctx == nullptr)
                {
                    g_session.ctx = whisper_init_from_file(model.c_str());
                    g_session.model = model;
                    g_session.refs = 0;
                }
                if (g_session.ctx != nullptr)
                {
                    g_session.refs += 1;
                }
                jsonResult["@type"] = "sessionOpen";
                jsonResult["ok"] = g_session.ctx != nullptr;
                if (g_session.ctx == nullptr)
                {
                    jsonResult["message"] = "OMI_STT_MODEL_UNAVAILABLE: model could not be loaded";
                }
                return jsonToChar(jsonResult);
            }
            if (type == "sessionClose")
            {
                if (g_session.refs > 0)
                {
                    g_session.refs -= 1;
                }
                if (g_session.refs == 0)
                {
                    if (g_session.ctx != nullptr)
                    {
                        whisper_free(g_session.ctx);
                    }
                    g_session.ctx = nullptr;
                    g_session.model.clear();
                }
                jsonResult["@type"] = "sessionClose";
                jsonResult["ok"] = true;
                return jsonToChar(jsonResult);
            }
            if (type == "getTextFromWavFile")
            {
                if (g_session.ctx == nullptr)
                {
                    jsonResult["@type"] = "error";
                    jsonResult["message"] = "OMI_STT_MODEL_UNAVAILABLE: session not open";
                    return jsonToChar(jsonResult);
                }
                whisper_params params = parseParams(jsonBody);
                return jsonToChar(runInference(g_session.ctx, params));
            }

            jsonResult["@type"] = "error";
            jsonResult["message"] = "unknown session request";
            return jsonToChar(jsonResult);
        }
        catch (const std::exception &e)
        {
            json jsonResult;
            jsonResult["@type"] = "error";
            jsonResult["message"] = e.what();
            return jsonToChar(jsonResult);
        }
        catch (...)
        {
            json jsonResult;
            jsonResult["@type"] = "error";
            jsonResult["message"] = "unknown error";
            return jsonToChar(jsonResult);
        }
    }
    FUNCTION_ATTRIBUTE
    char *request(char *body)
    {
        try
        {
            json jsonBody = json::parse(body);
            json jsonResult;

            if (jsonBody["@type"] == "getTextFromWavFile")
            {
                try
                {
                    return jsonToChar(transcribe(jsonBody));
                }
                catch (const std::exception &e)
                {
                    jsonResult["@type"] = "error";
                    jsonResult["message"] = e.what();
                    return jsonToChar(jsonResult);
                }
            }
            if (jsonBody["@type"] == "getVersion")
            {
                jsonResult["@type"] = "version";
                jsonResult["message"] = "lib version: v1.0.1";
                return jsonToChar(jsonResult);
            }

            jsonResult["@type"] = "error";
            jsonResult["message"] = "method not found";
            return jsonToChar(jsonResult);
        }
        catch (const std::exception &e)
        {
            json jsonResult;
            jsonResult["@type"] = "error";
            jsonResult["message"] = e.what();
            return jsonToChar(jsonResult);
        }
    }
}