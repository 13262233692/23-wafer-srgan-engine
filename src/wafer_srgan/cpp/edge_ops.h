#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

namespace wafer_srgan {

struct EdgeConfig {
    double canny_threshold1 = 50.0;
    double canny_threshold2 = 150.0;
    int sobel_ksize = 3;
    int morph_kernel_size = 3;
    int morph_iterations = 2;
};

cv::Mat detect_edges_canny(const cv::Mat& image, const EdgeConfig& cfg);

cv::Mat detect_edges_sobel(const cv::Mat& image, const EdgeConfig& cfg);

cv::Mat morphological_close(const cv::Mat& mask, const EdgeConfig& cfg);

cv::Mat morphological_open(const cv::Mat& mask, const EdgeConfig& cfg);

cv::Mat detect_defect_edges(const cv::Mat& image, const EdgeConfig& cfg, const std::string& method);

cv::Mat overlay_edges(const cv::Mat& image, const cv::Mat& edges, int r, int g, int b);

std::vector<cv::Mat> batch_detect_defects(const std::vector<cv::Mat>& images, const EdgeConfig& cfg, const std::string& method);

}
