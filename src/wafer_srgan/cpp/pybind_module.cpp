#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "edge_ops.h"

namespace py = pybind11;

static cv::Mat numpy_to_mat(const py::array_t<uint8_t>& arr) {
    py::buffer_info buf = arr.request();
    int rows = buf.shape[0];
    int cols = buf.shape[1];
    int channels = buf.ndim == 3 ? buf.shape[2] : 1;
    int type = channels == 1 ? CV_8UC1 : CV_8UC3;
    return cv::Mat(rows, cols, type, buf.ptr).clone();
}

static py::array_t<uint8_t> mat_to_numpy(const cv::Mat& mat) {
    if (mat.channels() == 1) {
        return py::array_t<uint8_t>({mat.rows, mat.cols}, mat.data);
    } else {
        return py::array_t<uint8_t>({mat.rows, mat.cols, mat.channels()}, mat.data);
    }
}

PYBIND11_MODULE(_edge_ops, m) {
    m.doc() = "OpenCV C++ edge post-processing operators for wafer SRGAN engine";

    py::class_<wafer_srgan::EdgeConfig>(m, "EdgeConfig")
        .def(py::init<>())
        .def_readwrite("canny_threshold1", &wafer_srgan::EdgeConfig::canny_threshold1)
        .def_readwrite("canny_threshold2", &wafer_srgan::EdgeConfig::canny_threshold2)
        .def_readwrite("sobel_ksize", &wafer_srgan::EdgeConfig::sobel_ksize)
        .def_readwrite("morph_kernel_size", &wafer_srgan::EdgeConfig::morph_kernel_size)
        .def_readwrite("morph_iterations", &wafer_srgan::EdgeConfig::morph_iterations);

    m.def("detect_edges_canny", [](const py::array_t<uint8_t>& image, const wafer_srgan::EdgeConfig& cfg) {
        return mat_to_numpy(wafer_srgan::detect_edges_canny(numpy_to_mat(image), cfg));
    }, py::arg("image"), py::arg("cfg"));

    m.def("detect_edges_sobel", [](const py::array_t<uint8_t>& image, const wafer_srgan::EdgeConfig& cfg) {
        return mat_to_numpy(wafer_srgan::detect_edges_sobel(numpy_to_mat(image), cfg));
    }, py::arg("image"), py::arg("cfg"));

    m.def("detect_defect_edges", [](const py::array_t<uint8_t>& image, const wafer_srgan::EdgeConfig& cfg, const std::string& method) {
        return mat_to_numpy(wafer_srgan::detect_defect_edges(numpy_to_mat(image), cfg, method));
    }, py::arg("image"), py::arg("cfg"), py::arg("method") = "canny");

    m.def("overlay_edges", [](const py::array_t<uint8_t>& image, const py::array_t<uint8_t>& edges, int r, int g, int b) {
        return mat_to_numpy(wafer_srgan::overlay_edges(numpy_to_mat(image), numpy_to_mat(edges), r, g, b));
    }, py::arg("image"), py::arg("edges"), py::arg("r") = 0, py::arg("g") = 255, py::arg("b") = 0);
}
