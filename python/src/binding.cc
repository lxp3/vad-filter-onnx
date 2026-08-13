#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "utils/resample.h"
#include "vad-config.h"
#include "vad-filter-onnx-cxx-api.h"

namespace py = pybind11;
using namespace VadFilterOnnx;

namespace {

class PyLinearResampler {
  public:
    PyLinearResampler(int32_t input_sample_rate, int32_t output_sample_rate)
        : input_sample_rate_(input_sample_rate), output_sample_rate_(output_sample_rate) {
        float min_freq = static_cast<float>(std::min(input_sample_rate, output_sample_rate));
        float cutoff = 0.99f * 0.5f * min_freq;
        int32_t lowpass_filter_width = 6;

        resampler_ = std::make_unique<sherpa_onnx::LinearResample>(
            input_sample_rate, output_sample_rate, cutoff, lowpass_filter_width);
    }

    void Reset() { resampler_->Reset(); }

    int32_t GetInputSampleRate() const { return input_sample_rate_; }

    int32_t GetOutputSampleRate() const { return output_sample_rate_; }

    py::array_t<float> Process(py::object input, bool flush = false) {
        std::vector<float> input_float = ConvertInputToFloat(std::move(input));
        return ResampleFloat(input_float, flush);
    }

  private:
    static std::vector<float> ConvertPcm16BytesToFloat(const py::bytes &input) {
        std::string raw = input;
        py::ssize_t valid_size =
            raw.size() - (raw.size() % static_cast<py::ssize_t>(sizeof(int16_t)));

        py::ssize_t sample_count = valid_size / static_cast<py::ssize_t>(sizeof(int16_t));
        const auto *data = reinterpret_cast<const int16_t *>(raw.data());
        std::vector<float> output(sample_count);
        for (py::ssize_t i = 0; i != sample_count; ++i) {
            output[i] = static_cast<float>(data[i]) / 32768.0f;
        }

        return output;
    }

    static std::vector<float> ConvertInputToFloat(py::object input) {
        if (py::isinstance<py::bytes>(input)) {
            return ConvertPcm16BytesToFloat(input.cast<py::bytes>());
        }

        if (py::isinstance<py::bytearray>(input)) {
            py::bytes data = py::reinterpret_borrow<py::bytearray>(input);
            return ConvertPcm16BytesToFloat(data);
        }

        if (py::isinstance<py::array>(input)) {
            py::array array = py::reinterpret_borrow<py::array>(input);
            py::buffer_info buf = array.request();
            if (buf.ndim != 1) {
                throw std::runtime_error("Input numpy array must be 1D");
            }

            const std::string int16_format = py::format_descriptor<int16_t>::format();
            const std::string float_format = py::format_descriptor<float>::format();
            if (buf.format == int16_format) {
                auto typed = py::array_t<int16_t, py::array::c_style | py::array::forcecast>::ensure(
                    input);
                if (!typed) {
                    throw std::runtime_error("Failed to convert numpy array to int16");
                }

                py::buffer_info typed_buf = typed.request();
                auto *data = static_cast<const int16_t *>(typed_buf.ptr);
                std::vector<float> output(typed_buf.size);
                for (py::ssize_t i = 0; i != typed_buf.size; ++i) {
                    output[i] = static_cast<float>(data[i]) / 32768.0f;
                }
                return output;
            }

            if (buf.format == float_format) {
                auto typed = py::array_t<float, py::array::c_style | py::array::forcecast>::ensure(
                    input);
                if (!typed) {
                    throw std::runtime_error("Failed to convert numpy array to float32");
                }

                py::buffer_info typed_buf = typed.request();
                auto *data = static_cast<const float *>(typed_buf.ptr);
                return std::vector<float>(data, data + typed_buf.size);
            }

            throw std::runtime_error("Unsupported numpy dtype. Expected int16 or float32.");
        }

        throw std::runtime_error("Unsupported input type. Expected bytes, bytearray, or numpy.ndarray.");
    }

    py::array_t<float> ResampleFloat(const std::vector<float> &input, bool flush) {
        std::vector<float> output;
        {
            py::gil_scoped_release release;
            resampler_->Resample(input.data(), static_cast<int32_t>(input.size()), flush, &output);
        }

        py::array_t<float> result(output.size());
        py::buffer_info buf = result.request();
        auto *dst = static_cast<float *>(buf.ptr);
        std::copy(output.begin(), output.end(), dst);
        return result;
    }

  private:
    int32_t input_sample_rate_;
    int32_t output_sample_rate_;
    std::unique_ptr<sherpa_onnx::LinearResample> resampler_;
};

} // namespace

PYBIND11_MODULE(vad_filter_onnx, m) {
    m.doc() = "Python bindings for vad-filter-onnx";

    py::enum_<VadType>(m, "VadType", "VAD model types")
        .value("WebrtcVad", VadType::WebrtcVad)
        .value("SileroVadV4", VadType::SileroVadV4)
        .value("SileroVadV5", VadType::SileroVadV5)
        .value("FsmnVad", VadType::FsmnVad)
        .value("TenVad", VadType::TenVad)
        .value("FireRedVad", VadType::FireRedVad)
        .value("NemoMarbleNetVad", VadType::NemoMarbleNetVad)
        .value("None", VadType::None)
        .export_values();

    py::class_<VadSegment>(m, "VadSegment", "Represents a detected speech segment")
        .def(py::init<int, int, int, int, int>(), py::arg("idx") = -1, py::arg("start") = -1,
             py::arg("end") = -1, py::arg("start_ms") = -1, py::arg("end_ms") = -1)
        .def_readwrite("idx", &VadSegment::idx, "Segment index")
        .def_readwrite("start", &VadSegment::start, "Start sample index")
        .def_readwrite("end", &VadSegment::end, "End sample index")
        .def_readwrite("start_ms", &VadSegment::start_ms, "Start time in milliseconds")
        .def_readwrite("end_ms", &VadSegment::end_ms, "End time in milliseconds")
        .def("__repr__", [](const VadSegment &s) {
            return "<VadSegment idx=" + std::to_string(s.idx) +
                   " start_ms=" + std::to_string(s.start_ms) +
                   " end_ms=" + std::to_string(s.end_ms) + ">";
        });

    py::class_<VadConfig>(m, "VadConfig", "Configuration for VAD filtering")
        .def(py::init<>())
        .def_readwrite("threshold", &VadConfig::threshold, "Detection threshold (default: 0.4)")
        .def_readwrite("sample_rate", &VadConfig::sample_rate, "Audio sample rate (default: 16000)")
        .def_readwrite("speech_window_size_ms", &VadConfig::speech_window_size_ms,
                       "Window size for speech detection in ms (default: 300)")
        .def_readwrite("speech_window_threshold_ms", &VadConfig::speech_window_threshold_ms,
                       "Speech duration threshold within window in ms (default: 250)")
        .def_readwrite("silence_window_size_ms", &VadConfig::silence_window_size_ms,
                       "Window size for silence detection in ms (default: 600)")
        .def_readwrite("silence_window_threshold_ms", &VadConfig::silence_window_threshold_ms,
                       "Silence duration threshold within window in ms (default: 500)")
        .def_readwrite("max_speech_ms", &VadConfig::max_speech_ms,
                       "Maximum speech segment duration in ms (default: 10000)")
        .def_readwrite("left_padding_ms", &VadConfig::left_padding_ms,
                       "Padding added to start of speech in ms (default: 100)")
        .def_readwrite("right_padding_ms", &VadConfig::right_padding_ms,
                       "Padding added to end of speech in ms (default: 100)");

    py::class_<AutoVadModel>(m, "AutoVadModel", "High-level VAD model API")
        .def(py::init([](const std::string &path, int num_threads, int device_id) {
                 py::gil_scoped_release release;
                 return AutoVadModel::create(path, num_threads, device_id);
             }),
             py::arg("path"), py::arg("num_threads") = 1, py::arg("device_id") = -1,
             "Create a model handle by loading an ONNX model from the given path.")
        .def_static(
            "create",
            [](const std::string &path, int num_threads, int device_id) {
                py::gil_scoped_release release;
                return AutoVadModel::create(path, num_threads, device_id);
            },
            py::arg("path"), py::arg("num_threads") = 1, py::arg("device_id") = -1,
            "Create a model handle by loading an ONNX model from the given path (Legacy static method).")
        .def(
            "init",
            [](AutoVadModel &self, const VadConfig &config) {
                py::gil_scoped_release release;
                return self.init(config);
            },
            py::arg("config"),
            "Initialize a model instance for inference with the given configuration.")
        .def(
            "decode",
            [](AutoVadModel &self, py::array_t<float> data, bool input_finished) {
                py::buffer_info buf = data.request();
                if (buf.ndim != 1) {
                    throw std::runtime_error("Input data must be a 1D array");
                }
                py::gil_scoped_release release;
                return self.decode(static_cast<float *>(buf.ptr), static_cast<int>(buf.size),
                                   input_finished);
            },
            py::arg("data"), py::arg("input_finished") = false,
            "Process audio data and return detected segments.")
        .def(
            "setup_config",
            [](AutoVadModel &self, const VadConfig &config) {
                py::gil_scoped_release release;
                self.setup_config(config);
            },
            py::arg("config"),
            "Update VAD configuration and reset the model internal state.")
        .def(
            "get_config",
            [](const AutoVadModel &self) {
                py::gil_scoped_release release;
                return self.get_config();
            },
            "Return the current VAD configuration.")
        .def(
            "reset",
            [](AutoVadModel &self) {
                py::gil_scoped_release release;
                self.reset();
            },
            "Reset the model internal state.")
        .def(
            "flush",
            [](AutoVadModel &self) {
                py::gil_scoped_release release;
                return self.flush();
            },
            "Flush remaining audio and return the final segment if any.");

    py::class_<PyLinearResampler>(m, "LinearResampler",
                                  "Stateful streaming linear resampler for realtime audio.")
        .def(py::init<int32_t, int32_t>(), py::arg("input_sample_rate"),
             py::arg("output_sample_rate"))
        .def_property_readonly("input_sample_rate", &PyLinearResampler::GetInputSampleRate)
        .def_property_readonly("output_sample_rate", &PyLinearResampler::GetOutputSampleRate)
        .def("reset", &PyLinearResampler::Reset,
             "Reset internal streaming state and discard buffered remainder.")
        .def("process", &PyLinearResampler::Process, py::arg("input"), py::arg("flush") = false,
             "Resample one chunk of audio.\n\n"
             "Supported input types:\n"
             "- bytes / bytearray: little-endian PCM int16\n"
             "- numpy.ndarray[int16]\n"
             "- numpy.ndarray[float32]\n\n"
             "Returns a 1D numpy.float32 array.");

    m.def(
        "get_ort_available_providers",
        []() {
            py::gil_scoped_release release;
            return get_ort_available_providers();
        },
        "Get list of available ONNX Runtime execution providers.");
}
