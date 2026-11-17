// Common value types shared by the celerispath native core.
#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace celerispath {

using VertexId = std::uint32_t;
using EdgeId = std::uint32_t;
// Position of an edge inside a compressed adjacency (CSR) layout.
using Slot = std::uint32_t;

inline constexpr std::uint32_t kInvalid = std::numeric_limits<std::uint32_t>::max();
inline constexpr double kInf = std::numeric_limits<double>::infinity();

enum class Status : int {
    Optimal = 0,      // a minimum-cost (feasible) path was found
    Unreachable = 1,  // no directed path from source to target exists
    Infeasible = 2,   // paths exist, but none satisfies the resource bounds
    Interrupted = 3,  // stopped by a StopToken before an answer was established
};

enum class Algorithm : int {
    Dag = 0,   // boost::dag_shortest_paths
    Rcsp = 1,  // boost::r_c_shortest_paths
};

// Compressed out-adjacency of a directed graph. Slots of one vertex are
// contiguous and keep edge insertion order, so iteration is deterministic.
struct Csr {
    std::uint32_t num_vertices = 0;
    std::vector<std::uint32_t> offsets;  // num_vertices + 1
    std::vector<VertexId> slot_source;
    std::vector<VertexId> slot_target;
    std::vector<EdgeId> slot_edge;       // user-facing edge id of each slot
    // Topological order and its inverse; both empty when the graph is cyclic.
    std::vector<VertexId> topo_order;
    std::vector<std::uint32_t> topo_rank;

    std::uint32_t num_slots() const { return static_cast<std::uint32_t>(slot_target.size()); }
};

// Label-tracked resources in slot order. Version 1 supports additive,
// monotone (nonnegative increment) resources only: R_new = R_old + delta and
// a label is feasible iff lower <= R <= upper. Because increments are
// nonnegative and lower <= initial, a smaller accumulated value is never
// worse, which is what the dominance rule in rcsp_solver.cpp relies on. Any
// future resource kind for which that does not hold needs its own extension
// and dominance functors rather than a new field here.
struct RcspData {
    std::size_t num_resources = 0;
    std::vector<double> slot_delta;  // num_resources * num_slots, interleaved per slot
    std::vector<double> lower;
    std::vector<double> upper;
    std::vector<double> initial;
};

struct RcspOptions {
    // Slack added to every dominance comparison. 0 keeps dominance exact;
    // any positive value may discard labels that are needed for optimality.
    double dominance_tolerance = 0.0;
};

// Cooperative interruption of a solve: a shared cancellation flag, a deadline,
// or both. A default-constructed token never asks to stop.
struct StopToken {
    using Clock = std::chrono::steady_clock;

    const std::atomic<bool>* cancelled = nullptr;
    Clock::time_point deadline = Clock::time_point::max();

    bool stop_requested() const {
        if (cancelled != nullptr && cancelled->load(std::memory_order_relaxed)) return true;
        return deadline != Clock::time_point::max() && Clock::now() >= deadline;
    }
};

struct SolveStats {
    double solve_seconds = 0.0;
    std::uint64_t labels_created = 0;     // feasible labels, including the source label
    std::uint64_t labels_extended = 0;    // labels popped undominated and extended
    std::uint64_t labels_dominated = 0;   // labels discarded by dominance before extension
    std::uint64_t labels_infeasible = 0;  // extensions violating a bound or with no feasible completion
    std::uint64_t max_active_labels = 0;  // peak number of unprocessed labels
    std::uint64_t pareto_labels = 0;      // nondominated labels at the target
};

// A path expressed in slots of the Csr it was computed on, in travel order.
struct PathResult {
    Status status = Status::Unreachable;
    double cost = kInf;
    std::vector<Slot> slots;
};

}  // namespace celerispath
