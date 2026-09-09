#include "core/performance.h"
#include <sys/resource.h>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <unistd.h>

#define MAX_METRICS_HISTORY 1000

PerformanceMetrics::PerformanceMetrics()
    : preprocessingTime(0),
      inferenceTime(0),
      postprocessingTime(0),
      totalTime(0) {
    reset();
}

PerformanceMetrics::~PerformanceMetrics() {
}

void PerformanceMetrics::startPreprocessing() {
    preprocessingStart = std::chrono::high_resolution_clock::now();
}

void PerformanceMetrics::startInference() {
    inferenceStart = std::chrono::high_resolution_clock::now();
}

void PerformanceMetrics::startPostprocessing() {
    postprocessingStart = std::chrono::high_resolution_clock::now();
}

void PerformanceMetrics::startTotal() {
    totalStart = std::chrono::high_resolution_clock::now();
}

void PerformanceMetrics::stopPreprocessing() {
    auto end = std::chrono::high_resolution_clock::now();
    preprocessingTime = std::chrono::duration<double, std::milli>(end - preprocessingStart).count();
}

void PerformanceMetrics::stopInference() {
    auto end = std::chrono::high_resolution_clock::now();
    inferenceTime = std::chrono::duration<double, std::milli>(end - inferenceStart).count();
}

void PerformanceMetrics::stopPostprocessing() {
    auto end = std::chrono::high_resolution_clock::now();
    postprocessingTime = std::chrono::duration<double, std::milli>(end - postprocessingStart).count();
}

void PerformanceMetrics::stopTotal() {
    auto end = std::chrono::high_resolution_clock::now();
    totalTime = std::chrono::duration<double, std::milli>(end - totalStart).count();
}

double PerformanceMetrics::getInferenceTime() const {
    return inferenceTime;
}

double PerformanceMetrics::getPreprocessingTime() const {
    return preprocessingTime;
}

double PerformanceMetrics::getPostprocessingTime() const {
    return postprocessingTime;
}

double PerformanceMetrics::getTotalTime() const {
    return totalTime;
}

void PerformanceMetrics::reset() {
    preprocessingTime = 0;
    inferenceTime = 0;
    postprocessingTime = 0;
    totalTime = 0;
    metrics_history_.clear();
    total_measurements_ = 0;
}

void PerformanceMetrics::addMeasurement() {
    PerformanceMetricsData data;
    data.preprocessingTime = preprocessingTime;
    data.inferenceTime = inferenceTime;
    data.postprocessingTime = postprocessingTime;
    data.totalTime = totalTime;
    metrics_history_.push_back(data);
    if (metrics_history_.size() > MAX_METRICS_HISTORY) {
        metrics_history_.erase(metrics_history_.begin());
    }
    total_measurements_++;
}

double PerformanceMetrics::getAveragePreprocessingTime() const {
    if (metrics_history_.empty()) {
        return 0.0;
    }
    double sum = 0.0;
    for (const auto& m : metrics_history_) {
        sum += m.preprocessingTime;
    }
    return sum / metrics_history_.size();
}

double PerformanceMetrics::getAverageInferenceTime() const {
    if (metrics_history_.empty()) {
        return 0.0;
    }
    double sum = 0.0;
    for (const auto& m : metrics_history_) {
        sum += m.inferenceTime;
    }
    return sum / metrics_history_.size();
}

double PerformanceMetrics::getAveragePostprocessingTime() const {
    if (metrics_history_.empty()) {
        return 0.0;
    }
    double sum = 0.0;
    for (const auto& m : metrics_history_) {
        sum += m.postprocessingTime;
    }
    return sum / metrics_history_.size();
}

double PerformanceMetrics::getAverageTotalTime() const {
    if (metrics_history_.empty()) {
        return 0.0;
    }
    double sum = 0.0;
    for (const auto& m : metrics_history_) {
        sum += m.totalTime;
    }
    return sum / metrics_history_.size();
}
