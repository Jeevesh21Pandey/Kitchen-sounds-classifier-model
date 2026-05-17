#include <iostream>
#include <vector>
#include <array>
#include <string>
#include <thread>
#include <atomic>
#include <mutex>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <csignal>
#include <sstream>
#include <iomanip>

#include <alsa/asoundlib.h>
#include <onnxruntime_cxx_api.h>
#include "librosa.h"
#include <mqtt/async_client.h>

// =========================================================
// CONFIGURATION
// =========================================================

constexpr int   SAMPLE_RATE      = 48000;
constexpr float TARGET_DURATION  = 3.0f;     
constexpr int   FFT_SIZE         = 1200;     
constexpr int   HOP_SIZE         = 480;      
constexpr int   N_MELS           = 128;
constexpr int   N_FRAMES         = 300;      
constexpr float DB_MIN           = -80.0f;
constexpr float DB_MAX           =   0.0f;
constexpr int   RING_SECONDS     = 12;
constexpr int   RING_SIZE        = SAMPLE_RATE * RING_SECONDS;
constexpr int   ALSA_CHUNK_FRAMES = 1024;
constexpr int   INMP441_SHIFT    = 8;
constexpr float INMP441_SCALE    = static_cast<float>(1 << 23);
constexpr float WHISTLE_THRESHOLD = 0.80f;
constexpr int   DEBOUNCE_ON_CONFIRM  = 1;
constexpr int   DEBOUNCE_OFF_CONFIRM = 0;   
constexpr int   CLASS_AMBIENT  = 0;
constexpr int   CLASS_FRYING   = 1;
constexpr int   CLASS_WHISTLE  = 2;

constexpr const char* MQTT_SERVER_URI = "tcp://127.0.0.1:1883";
constexpr const char* MQTT_CLIENT_ID  = "kitchen_classifier_pi4";

// =========================================================
// SIGNAL HANDLING & RING BUFFER
// =========================================================

static volatile std::sig_atomic_t g_running = 1;
void handle_signal(int) { g_running = 0; }

struct RingBuffer {
    std::array<float, RING_SIZE> data{};
    std::atomic<size_t> write_pos{0};

    void push(const float* samples, size_t count) {
        size_t pos = write_pos.load(std::memory_order_relaxed);
        for (size_t i = 0; i < count; ++i)
            data[(pos + i) % RING_SIZE] = samples[i];
        write_pos.store(pos + count, std::memory_order_release);
    }

    bool snapshot(std::vector<float>& dst, size_t n) {
        size_t pos = write_pos.load(std::memory_order_acquire);
        if (pos < n) return false;
        dst.resize(n);
        size_t start = pos - n;
        for (size_t i = 0; i < n; ++i)
            dst[i] = data[(start + i) % RING_SIZE];
        return true;
    }
};

// =========================================================
// ALSA & FEATURE WRAPPERS
// =========================================================

class AlsaCapture {
public:
    explicit AlsaCapture(const std::string& device) {
        int rc = snd_pcm_open(&handle_, device.c_str(), SND_PCM_STREAM_CAPTURE, 0);
        if (rc < 0) throw std::runtime_error("Failed to open ALSA: " + device);
        snd_pcm_hw_params_t* params;
        snd_pcm_hw_params_alloca(&params);
        snd_pcm_hw_params_any(handle_, params);
        snd_pcm_hw_params_set_access(handle_, params, SND_PCM_ACCESS_RW_INTERLEAVED);
        snd_pcm_hw_params_set_format(handle_, params, SND_PCM_FORMAT_S32_LE);
        unsigned int rate = SAMPLE_RATE;
        snd_pcm_hw_params_set_rate_near(handle_, params, &rate, nullptr);
        snd_pcm_hw_params_set_channels(handle_, params, 1);
        snd_pcm_hw_params(handle_, params);
        snd_pcm_prepare(handle_);
        raw_buf_.resize(ALSA_CHUNK_FRAMES);
        float_buf_.resize(ALSA_CHUNK_FRAMES);
    }
    ~AlsaCapture() { if (handle_) snd_pcm_close(handle_); }
    int read(const float** out) {
        int rc = snd_pcm_readi(handle_, raw_buf_.data(), ALSA_CHUNK_FRAMES);
        if (rc == -EPIPE) { snd_pcm_prepare(handle_); return 0; }
        if (rc < 0) return rc;
        for (int i = 0; i < rc; ++i)
            float_buf_[i] = static_cast<float>(raw_buf_[i] >> INMP441_SHIFT) / INMP441_SCALE;
        *out = float_buf_.data();
        return rc;
    }
private:
    snd_pcm_t* handle_{nullptr};
    std::vector<int32_t> raw_buf_;
    std::vector<float> float_buf_;
};

static std::vector<float> build_features(const std::vector<float>& audio) {
    auto mel = librosa::Feature::melspectrogram(const_cast<std::vector<float>&>(audio), SAMPLE_RATE, FFT_SIZE, HOP_SIZE, "hann", false, "reflect", 2.0f, N_MELS, 0, SAMPLE_RATE / 2);
    std::vector<float> feat(static_cast<size_t>(N_MELS) * N_FRAMES, 0.0f);
    int frames = std::min(static_cast<int>(mel.size()), N_FRAMES);
    for (int t = 0; t < frames; ++t) {
        int mels = std::min(static_cast<int>(mel[t].size()), N_MELS);
        for (int m = 0; m < mels; ++m) {
            float power_db = 10.0f * std::log10(std::max(mel[t][m], 1e-10f));
            float norm = std::clamp((power_db - DB_MIN) / (DB_MAX - DB_MIN), 0.0f, 1.0f);
            feat[m * N_FRAMES + t] = norm;
        }
    }
    return feat;
}

class OnnxModel {
public:
    OnnxModel(const std::string& path, const std::string& log_tag) : env_(ORT_LOGGING_LEVEL_WARNING, log_tag.c_str()), session_(nullptr) {
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(2);
        session_ = Ort::Session(env_, path.c_str(), opts);
        auto alloc = Ort::AllocatorWithDefaultOptions{};
        input_name_str_  = session_.GetInputNameAllocated(0, alloc).get();
        output_name_str_ = session_.GetOutputNameAllocated(0, alloc).get();
        input_name_ = input_name_str_.c_str();
        output_name_ = output_name_str_.c_str();
        num_classes_ = static_cast<int>(session_.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape().back());
    }
    std::vector<float> infer(const std::vector<float>& feat) {
        std::array<int64_t, 4> shape = { 1, N_MELS, N_FRAMES, 1 };
        auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        auto tensor = Ort::Value::CreateTensor<float>(mem, const_cast<float*>(feat.data()), feat.size(), shape.data(), shape.size());
        auto outputs = session_.Run(Ort::RunOptions{nullptr}, &input_name_, &tensor, 1, &output_name_, 1);
        float* raw = outputs[0].GetTensorMutableData<float>();
        return std::vector<float>(raw, raw + outputs[0].GetTensorTypeAndShapeInfo().GetElementCount());
    }
private:
    Ort::Env env_; Ort::Session session_; std::string input_name_str_, output_name_str_;
    const char *input_name_, *output_name_; int num_classes_;
};

// =========================================================
// SAFE MQTT PUBLISH
// =========================================================

static void publish_safe(mqtt::async_client& client, const std::string& topic, const std::string& payload) {
    if (!client.is_connected()) return;
    try {
        client.publish(topic, payload.c_str(), payload.size(), 1, false);
    } catch (...) {}
}

static std::string scores_to_json(float p_a, float p_f, float p_w) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(4) << "{\"ambient\":" << p_a << ",\"frying\":" << p_f << ",\"whistle\":" << p_w << "}";
    return ss.str();
}

// =========================================================
// MAIN
// =========================================================

int main(int argc, char* argv[]) {
    std::signal(SIGINT, handle_signal);
    std::string device = "hw:3,0";
    std::string model_path = "kitchen_sound_classifier.onnx";
    if (argc > 1) device = argv[1];
    if (argc > 2) model_path = argv[2];

    RingBuffer ring;
    AlsaCapture cap(device);
    OnnxModel model(model_path, "kitchen");
    mqtt::async_client mqtt_client(MQTT_SERVER_URI, MQTT_CLIENT_ID);

    try {
        auto opts = mqtt::connect_options_builder().clean_session(true).finalize();
        mqtt_client.connect(opts)->wait();
    } catch (...) { std::cerr << "[MQTT] Initial connect failed\n"; }

    std::thread capture_thread([&]() {
        while (g_running) {
            const float* s = nullptr;
            int n = cap.read(&s);
            if (n > 0) ring.push(s, static_cast<size_t>(n));
        }
    });

    std::thread inference_thread([&]() {
        std::vector<float> audio;
        const size_t need = static_cast<size_t>(SAMPLE_RATE * TARGET_DURATION);
        bool confirmed_whistle = false;
        int on_streak = 0, off_streak = 0;

        while (g_running) {
            std::this_thread::sleep_for(std::chrono::seconds(1));

            // Reconnect if needed
            if (!mqtt_client.is_connected()) {
                try { mqtt_client.connect(); } catch(...) {}
            }

            if (!ring.snapshot(audio, need)) continue;

            try {
                auto feat = build_features(audio);
                auto probs = model.infer(feat);
                
                float p_a = probs[CLASS_AMBIENT], p_f = probs[CLASS_FRYING], p_w = probs[CLASS_WHISTLE];
                bool raw_whistle = p_w > WHISTLE_THRESHOLD;

                // Debounce
                int event = 0;
                if (raw_whistle) {
                    off_streak = 0;
                    if (!confirmed_whistle && ++on_streak >= DEBOUNCE_ON_CONFIRM) {
                        confirmed_whistle = true; on_streak = 0; event = 1;
                    }
                } else {
                    on_streak = 0;
                    if (confirmed_whistle && ++off_streak >= DEBOUNCE_OFF_CONFIRM) {
                        confirmed_whistle = false; off_streak = 0; event = -1;
                    }
                }

                std::cout << "[scores] amb=" << p_a << " whi=" << p_w << " | " << (confirmed_whistle ? "WHISTLE" : "IDLE") << std::endl;

                publish_safe(mqtt_client, "kitchen/whistle", confirmed_whistle ? "1" : "0");
                publish_safe(mqtt_client, "kitchen/status", confirmed_whistle ? "whistle" : "none");
                if (event == 1) publish_safe(mqtt_client, "kitchen/event", "whistle_started");
                if (event == -1) publish_safe(mqtt_client, "kitchen/event", "whistle_stopped");
                publish_safe(mqtt_client, "kitchen/scores", scores_to_json(p_a, p_f, p_w));

            } catch (const std::exception& e) { std::cerr << "Error: " << e.what() << std::endl; }
        }
    });

    capture_thread.join();
    inference_thread.join();
    return 0;
}
