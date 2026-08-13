#ifndef OPS_MLIR_RUNTIME_PROFILER_H
#define OPS_MLIR_RUNTIME_PROFILER_H

#include <chrono>
#include <string>
#include <unordered_map>
#include <vector>

namespace ops_mlir {

struct KernelTiming {
    std::string kernel_name;
    double milliseconds;
    // Bytes read/written by the kernel's dataset args, following the OPS
    // convention: extent-of-iteration-range * elem_size, counted once for
    // OPS_READ/OPS_WRITE and twice for OPS_RW/OPS_INC (data is both fetched
    // and stored back).
    double bytes = 0.0;
};

struct KernelStats {
    std::string kernel_name;
    int count = 0;
    double total_ms = 0.0;
    double total_bytes = 0.0;
    double avg_ms() const { return count ? total_ms / count: 0.0; };
    // Aggregate bandwidth in GB/s (bytes / 1024^3 per second), matching how
    // the OPS library reports "Bandwidth(GB/s)" per kernel.
    double bandwidth_gbps() const {
        return total_ms > 0.0
                   ? total_bytes / (total_ms / 1000.0) / (1024.0 * 1024.0 * 1024.0)
                   : 0.0;
    }
};

class KernelProfiler {
public:
    using TimePoint = std::chrono::steady_clock::time_point;

    [[nodiscard]] TimePoint start() const { return std::chrono::steady_clock::now(); }

    void end(const std::string &kernelName, TimePoint startTime, double bytes = 0.0) {
        double ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - startTime)
                        .count();
        record(kernelName, ms, bytes);
    }

    void record(const std::string &kernelName, double ms, double bytes = 0.0) {
        records_.push_back({kernelName, ms, bytes});
    }

    const std::vector<KernelTiming> &records() const { return records_; }

    std::vector<KernelStats> aggregate() const;

    void report() const;
    void writeCsv(const std::string &path) const;

private:
    std::vector<KernelTiming> records_;
};

} // namespace ops_mlir

#endif // OPS_MLIR_RUNTIME_PROFILER_H
