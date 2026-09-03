#ifndef PERFORMANCE_H
#define PERFORMANCE_H

#include <chrono>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

struct PerformanceMetricsData {
    double preprocessingTime = 0;
    double inferenceTime = 0;
    double postprocessingTime = 0;
    double totalTime = 0;
};

class PerformanceMetrics {
public:
    PerformanceMetrics();
    ~PerformanceMetrics();

    // Start timing for different stages
    void startPreprocessing();
    void startInference();
    void startPostprocessing();
    void startTotal();

    // Stop timing for different stages
    void stopPreprocessing();
    void stopInference();
    void stopPostprocessing();
    void stopTotal();

    // Get timing results in milliseconds
    double getInferenceTime() const;
    double getPreprocessingTime() const;
    double getPostprocessingTime() const;
    double getTotalTime() const;

    // Reset all metrics
    void reset();

    // Add a measurement to history
    void addMeasurement();

    // Get average metrics (累积耗时 / 总测量次数，不受 history 截断影响)
    double getAveragePreprocessingTime() const;
    double getAverageInferenceTime() const;
    double getAveragePostprocessingTime() const;
    double getAverageTotalTime() const;

    // Get metrics history for detailed analysis
    const std::vector<PerformanceMetricsData>& getMetricsHistory() const { return metrics_history_; }
    int getTotalMeasurements() const { return (int)metrics_history_.size(); }

private:
    // Timing variables
    std::chrono::high_resolution_clock::time_point preprocessingStart;
    std::chrono::high_resolution_clock::time_point inferenceStart;
    std::chrono::high_resolution_clock::time_point postprocessingStart;
    std::chrono::high_resolution_clock::time_point totalStart;

    double preprocessingTime;
    double inferenceTime;
    double postprocessingTime;
    double totalTime;

    // History of measurements
    std::vector<PerformanceMetricsData> metrics_history_;
    // 总测量次数（独立于 history 截断，用于计算准确平均）
    int total_measurements_ = 0;
};

#endif // PERFORMANCE_H
