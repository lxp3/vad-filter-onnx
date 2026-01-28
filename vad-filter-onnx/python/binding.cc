#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "vad-config.h"
#include "vad-filter-onnx-cxx-api.h"

namespace py = pybind11;
using namespace VadFilterOnnx;

PYBIND11_MODULE(vad_filter_onnx, m) {
    m.doc() = "Python bindings for vad-filter-onnx";

    py::enum_<VadType>(m, "VadType", "VAD model types")
        .value("WebrtcVad", VadType::WebrtcVad)
        .value("SileroVadV4", VadType::SileroVadV4)
        .value("SileroVadV5", VadType::SileroVadV5)
        .value("FsmnVad", VadType::FsmnVad)
        .value("TenVad", VadType::TenVad)
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
        .def_static("create", &AutoVadModel::create, py::arg("path"), py::arg("num_threads") = 1,
                    py::arg("device_id") = -1,
                    "Create a model handle by loading an ONNX model from the given path.")
        .def("init", &AutoVadModel::init, py::arg("config"),
             "Initialize a model instance for inference with the given configuration.")
        .def(
            "decode",
            [](AutoVadModel &self, py::array_t<float> data, bool input_finished) {
                py::buffer_info buf = data.request();
                if (buf.ndim != 1) {
                    throw std::runtime_error("Input data must be a 1D array");
                }
                return self.decode(static_cast<float *>(buf.ptr), static_cast<int>(buf.size),
                                   input_finished);
            },
            py::arg("data"), py::arg("input_finished"),
            "Process audio data and return detected segments.")
        .def("reset", &AutoVadModel::reset, "Reset the model internal state.")
        .def("flush", &AutoVadModel::flush,
             "Flush remaining audio and return the final segment if any.");

    m.def("get_ort_available_providers", &get_ort_available_providers,
          "Get list of available ONNX Runtime execution providers.");
}
