#include "edge_ops.h"

namespace wafer_srgan {

cv::Mat detect_edges_canny(const cv::Mat& image, const EdgeConfig& cfg) {
    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_RGB2GRAY);
    } else {
        gray = image.clone();
    }

    if (gray.depth() != CV_8U) {
        gray.convertTo(gray, CV_8U, 1.0);
    }

    cv::Mat edges;
    cv::Canny(gray, edges, cfg.canny_threshold1, cfg.canny_threshold2);
    return edges;
}

cv::Mat detect_edges_sobel(const cv::Mat& image, const EdgeConfig& cfg) {
    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_RGB2GRAY);
    } else {
        gray = image.clone();
    }

    if (gray.depth() != CV_8U) {
        gray.convertTo(gray, CV_8U, 1.0);
    }

    cv::Mat sobel_x, sobel_y;
    cv::Sobel(gray, sobel_x, CV_64F, 1, 0, cfg.sobel_ksize);
    cv::Sobel(gray, sobel_y, CV_64F, 0, 1, cfg.sobel_ksize);

    cv::Mat magnitude;
    cv::magnitude(sobel_x, sobel_y, magnitude);

    double min_val, max_val;
    cv::minMaxLoc(magnitude, &min_val, &max_val);
    if (max_val > 0) {
        magnitude = magnitude / max_val * 255.0;
    }

    cv::Mat result;
    magnitude.convertTo(result, CV_8U);
    return result;
}

cv::Mat morphological_close(const cv::Mat& mask, const EdgeConfig& cfg) {
    cv::Mat input = mask.clone();
    if (input.depth() != CV_8U) {
        input.convertTo(input, CV_8U, 1.0);
    }

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT,
        cv::Size(cfg.morph_kernel_size, cfg.morph_kernel_size));
    cv::Mat result;
    cv::morphologyEx(input, result, cv::MORPH_CLOSE, kernel,
                     cv::Point(-1, -1), cfg.morph_iterations);
    return result;
}

cv::Mat morphological_open(const cv::Mat& mask, const EdgeConfig& cfg) {
    cv::Mat input = mask.clone();
    if (input.depth() != CV_8U) {
        input.convertTo(input, CV_8U, 1.0);
    }

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT,
        cv::Size(cfg.morph_kernel_size, cfg.morph_kernel_size));
    cv::Mat result;
    cv::morphologyEx(input, result, cv::MORPH_OPEN, kernel,
                     cv::Point(-1, -1), cfg.morph_iterations);
    return result;
}

cv::Mat detect_defect_edges(const cv::Mat& image, const EdgeConfig& cfg, const std::string& method) {
    cv::Mat edges;
    if (method == "sobel") {
        edges = detect_edges_sobel(image, cfg);
    } else {
        edges = detect_edges_canny(image, cfg);
    }

    edges = morphological_close(edges, cfg);
    edges = morphological_open(edges, cfg);
    return edges;
}

cv::Mat overlay_edges(const cv::Mat& image, const cv::Mat& edges, int r, int g, int b) {
    cv::Mat overlay;
    if (image.channels() == 1) {
        cv::cvtColor(image, overlay, cv::COLOR_GRAY2RGB);
    } else {
        overlay = image.clone();
    }

    for (int y = 0; y < edges.rows; ++y) {
        const uchar* edge_row = edges.ptr<uchar>(y);
        cv::Vec3b* overlay_row = overlay.ptr<cv::Vec3b>(y);
        for (int x = 0; x < edges.cols; ++x) {
            if (edge_row[x] > 0) {
                overlay_row[x] = cv::Vec3b(b, g, r);
            }
        }
    }
    return overlay;
}

std::vector<cv::Mat> batch_detect_defects(const std::vector<cv::Mat>& images,
                                           const EdgeConfig& cfg,
                                           const std::string& method) {
    std::vector<cv::Mat> results;
    results.reserve(images.size());
    for (const auto& img : images) {
        results.push_back(detect_defect_edges(img, cfg, method));
    }
    return results;
}

}
