// reviewscope_infer: run the exported classifier with the ONNX Runtime C++ API.
//
// Reads the fixture written by export_onnx.py (token ids + attention mask),
// runs it at batch 1 and as one full batch, checks the logits against the
// PyTorch-derived expected.bin, and prints one JSON line with latency and
// parity. Exits 2 if parity fails, so CI can gate on it.
//
// Tokenisation stays in Python on purpose: this binary measures the model, not
// a hand-written WordPiece port.

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Fixture {
    int64_t rows = 0;
    int64_t seq = 0;
    std::vector<int64_t> ids;
    std::vector<int64_t> mask;
};

Fixture read_tokens(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    Fixture fx;
    f.read(reinterpret_cast<char*>(&fx.rows), sizeof(int64_t));
    f.read(reinterpret_cast<char*>(&fx.seq), sizeof(int64_t));
    const auto n = static_cast<size_t>(fx.rows * fx.seq);
    fx.ids.resize(n);
    fx.mask.resize(n);
    f.read(reinterpret_cast<char*>(fx.ids.data()), n * sizeof(int64_t));
    f.read(reinterpret_cast<char*>(fx.mask.data()), n * sizeof(int64_t));
    if (!f) throw std::runtime_error("truncated fixture " + path);
    return fx;
}

std::vector<float> read_floats(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot open " + path);
    const auto bytes = static_cast<size_t>(f.tellg());
    std::vector<float> out(bytes / sizeof(float));
    f.seekg(0);
    f.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(bytes));
    return out;
}

double percentile(std::vector<double> v, double q) {
    std::sort(v.begin(), v.end());
    return v[static_cast<size_t>(q * static_cast<double>(v.size() - 1))];
}

class Classifier {
public:
    Classifier(Ort::Env& env, const std::string& model_path, int threads) {
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(threads);
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
#ifdef _WIN32
        const std::wstring wide(model_path.begin(), model_path.end());
        session_ = Ort::Session(env, wide.c_str(), opts);
#else
        session_ = Ort::Session(env, model_path.c_str(), opts);
#endif
    }

    // Returns logits, row-major [rows, 2].
    std::vector<float> run(const int64_t* ids, const int64_t* mask, int64_t rows, int64_t seq) {
        const std::array<int64_t, 2> shape{rows, seq};
        const auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        const size_t n = static_cast<size_t>(rows * seq);
        std::array<Ort::Value, 2> inputs{
            Ort::Value::CreateTensor<int64_t>(mem, const_cast<int64_t*>(ids), n, shape.data(), shape.size()),
            Ort::Value::CreateTensor<int64_t>(mem, const_cast<int64_t*>(mask), n, shape.data(), shape.size()),
        };
        const char* in_names[] = {"input_ids", "attention_mask"};
        const char* out_names[] = {"logits"};
        auto out = session_.Run(Ort::RunOptions{nullptr}, in_names, inputs.data(), 2, out_names, 1);
        const float* data = out[0].GetTensorData<float>();
        const auto count = out[0].GetTensorTypeAndShapeInfo().GetElementCount();
        return {data, data + count};
    }

private:
    Ort::Session session_{nullptr};
};

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: reviewscope_infer <model.onnx> <tokens.bin> [expected.bin] [repeats] [threads]\n";
        return 1;
    }
    const std::string model = argv[1];
    const std::string tokens = argv[2];
    const std::string expected_path = argc > 3 ? argv[3] : "";
    const int repeats = argc > 4 ? std::atoi(argv[4]) : 10;
    const int threads = argc > 5 ? std::atoi(argv[5]) : 4;

    try {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "reviewscope");
        Classifier clf(env, model, threads);
        const Fixture fx = read_tokens(tokens);

        // Warm-up, then batch-1 latency over every row, `repeats` times.
        clf.run(fx.ids.data(), fx.mask.data(), 1, fx.seq);
        std::vector<double> single_ms;
        for (int r = 0; r < repeats; ++r) {
            for (int64_t i = 0; i < fx.rows; ++i) {
                const auto t0 = std::chrono::steady_clock::now();
                clf.run(fx.ids.data() + i * fx.seq, fx.mask.data() + i * fx.seq, 1, fx.seq);
                single_ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
            }
        }

        std::vector<double> batch_ms;
        std::vector<float> logits;
        for (int r = 0; r < repeats; ++r) {
            const auto t0 = std::chrono::steady_clock::now();
            logits = clf.run(fx.ids.data(), fx.mask.data(), fx.rows, fx.seq);
            batch_ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        }

        double max_diff = -1.0;
        if (!expected_path.empty()) {
            const auto expected = read_floats(expected_path);
            if (expected.size() != logits.size()) throw std::runtime_error("expected.bin size mismatch");
            max_diff = 0.0;
            for (size_t i = 0; i < logits.size(); ++i) max_diff = std::max(max_diff, static_cast<double>(std::fabs(logits[i] - expected[i])));
        }

        std::cout << "{\"rows\": " << fx.rows << ", \"seq\": " << fx.seq << ", \"threads\": " << threads
                  << ", \"batch1_p50_ms\": " << percentile(single_ms, 0.5)
                  << ", \"batch1_p95_ms\": " << percentile(single_ms, 0.95)
                  << ", \"batch" << fx.rows << "_p50_ms\": " << percentile(batch_ms, 0.5)
                  << ", \"max_abs_diff\": " << max_diff << "}\n";
        if (max_diff > 1e-3) {
            std::cerr << "parity check failed: max abs diff " << max_diff << "\n";
            return 2;
        }
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
