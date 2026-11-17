// Native benchmark of the solver core, without Python in the loop.
//
// Builds layered random DAGs (the same family as benchmarks/_common.py, but
// not the same random stream), then times edge insertion, compilation, the
// unconstrained DAG solve and a one-resource RCSP solve.
//
//   cmake -S . -B build/bench -DCELERISROUTE_BUILD_PYTHON=OFF -DCELERISROUTE_BUILD_BENCHMARKS=ON
//   cmake --build build/bench
//   ./build/bench/celerisroute_bench [seed]

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <random>
#include <span>
#include <string>
#include <vector>

#include "celerisroute/graph.hpp"

namespace {

using namespace celerisroute;
using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

double median(std::vector<double> samples) {
    std::sort(samples.begin(), samples.end());
    const std::size_t middle = samples.size() / 2;
    return samples.size() % 2 ? samples[middle] : 0.5 * (samples[middle - 1] + samples[middle]);
}

struct Instance {
    std::uint64_t num_vertices = 0;
    std::vector<std::int64_t> source, target;
    std::vector<double> cost, resource;
};

// Source 0 -> `layers` layers of `width` vertices -> sink. Integer costs in
// [-5, 20]; the resource is in [1, 10] and larger on cheap edges. Only raw
// mt19937_64 output is used, so instances are identical on every platform.
Instance layered_dag(std::uint64_t layers, std::uint64_t width, std::uint64_t degree,
                     std::uint64_t seed) {
    std::mt19937_64 random(seed);
    Instance instance;
    instance.num_vertices = layers * width + 2;
    const auto add = [&](std::uint64_t from, std::uint64_t to) {
        const double cost = static_cast<double>(random() % 26) - 5.0;
        const double noise = static_cast<double>(random() % 5) - 2.0;
        const double scaled = std::round((20.0 - cost) / 25.0 * 9.0 + noise);
        instance.source.push_back(static_cast<std::int64_t>(from));
        instance.target.push_back(static_cast<std::int64_t>(to));
        instance.cost.push_back(cost);
        instance.resource.push_back(std::clamp(scaled, 1.0, 10.0));
    };
    for (std::uint64_t i = 0; i < width; ++i) add(0, 1 + i);
    for (std::uint64_t layer = 0; layer + 1 < layers; ++layer) {
        const std::uint64_t base = 1 + layer * width;
        for (std::uint64_t i = 0; i < width; ++i) {
            for (std::uint64_t k = 0; k < degree; ++k) add(base + i, base + width + random() % width);
        }
    }
    for (std::uint64_t i = 0; i < width; ++i) {
        add(1 + (layers - 1) * width + i, instance.num_vertices - 1);
    }
    return instance;
}

template <class Function>
double median_seconds(Function&& function, double budget = 0.5, int minimum = 3, int maximum = 50) {
    std::vector<double> samples;
    double total = 0.0;
    while (static_cast<int>(samples.size()) < minimum ||
           (total < budget && static_cast<int>(samples.size()) < maximum)) {
        const auto start = Clock::now();
        function();
        samples.push_back(seconds_since(start));
        total += samples.back();
    }
    return median(samples);
}

void run(std::uint64_t layers, std::uint64_t width, std::uint64_t degree, bool constrained,
         std::uint64_t seed) {
    const Instance instance = layered_dag(layers, width, degree, seed);
    const auto source = static_cast<VertexId>(0);
    const auto sink = static_cast<VertexId>(instance.num_vertices - 1);
    const std::vector<std::string> names{"r0"};
    const std::vector<std::span<const double>> columns{instance.resource};

    auto start = Clock::now();
    GraphBuilder builder(instance.num_vertices);
    builder.add_edges(instance.source, instance.target, instance.cost, names, columns);
    const double build = seconds_since(start);

    start = Clock::now();
    const std::shared_ptr<const CompiledGraph> free = builder.compile({});
    const double compile = seconds_since(start);

    const RcspOptions options;
    Solution cheapest = free->solve(source, sink, Algorithm::Dag, options);  // warm-up
    const double dag = median_seconds(
        [&] { cheapest = free->solve(source, sink, Algorithm::Dag, options); });

    std::printf("%9llu %9zu %7llu %6llu %3llu | %9.3f %9.3f %9.3f |",
                static_cast<unsigned long long>(instance.num_vertices), instance.cost.size(),
                static_cast<unsigned long long>(layers), static_cast<unsigned long long>(width),
                static_cast<unsigned long long>(degree), build * 1e3, compile * 1e3, dag * 1e3);
    if (!constrained) {
        std::printf(" %10s %10s %10s\n", "-", "-", "-");
        return;
    }

    // Bound a quarter of the way from the least possible consumption to the
    // consumption of the cheapest path, as benchmark_rcsp.py does.
    const double least =
        free->with_costs(instance.resource)->solve(source, sink, Algorithm::Dag, options).cost;
    const double used = cheapest.resources[0];
    const std::vector<ResourceSpec> declared{
        {"r0", -kInf, std::floor(least + 0.25 * (used - least)), 0.0}};
    const std::shared_ptr<const CompiledGraph> bounded = builder.compile(declared);
    Solution solution = bounded->solve(source, sink, Algorithm::Rcsp, options);  // warm-up
    const double rcsp = median_seconds(
        [&] { solution = bounded->solve(source, sink, Algorithm::Rcsp, options); }, 1.0, 3, 30);
    if (solution.status != Status::Optimal || solution.cost < cheapest.cost ||
        solution.resources[0] > declared[0].upper) {
        std::fprintf(stderr, "inconsistent RCSP answer\n");
        std::exit(1);
    }
    std::printf(" %10.3f %10llu %10.1f\n", rcsp * 1e3,
                static_cast<unsigned long long>(solution.stats.labels_created), solution.cost);
}

}  // namespace

int main(int argc, char** argv) {
    const std::uint64_t seed = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 20240607;
    std::printf("seed %llu; times in milliseconds (dag and rcsp: median of repeated solves)\n",
                static_cast<unsigned long long>(seed));
    std::printf("%9s %9s %7s %6s %3s | %9s %9s %9s | %10s %10s %10s\n", "vertices", "edges",
                "layers", "width", "deg", "build", "compile", "dag", "rcsp", "labels", "rcsp cost");
    run(10, 100, 4, true, seed);
    run(30, 33, 4, true, seed);
    run(100, 100, 8, false, seed);
    run(10, 1'000, 4, true, seed);
    run(1'000, 100, 8, false, seed);
    run(10, 10'000, 4, true, seed);
    run(10'000, 100, 8, false, seed);
    run(1'000, 1'000, 2, false, seed);
    return 0;
}
