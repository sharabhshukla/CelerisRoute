// Fleet pricing: many pricing subproblems ("vessels") on one shared graph.
//
// A PricingEngine owns a pruned copy of the graph for its source and target.
// That is the right trade when there are a few subproblems, and the wrong one
// for a fleet: a hundred vessels that sail the same network would hold a
// hundred copies of it and refresh a hundred reduced-cost arrays per
// column-generation iteration.
//
// A PricingFleet keeps exactly one topology, one resource table and one
// reduced-cost array, all shared and read-only while vessels are priced, and
// describes each vessel by a small profile that refers to them:
//
//   reduced_v(e) = cost(e) + offset_v(e) - sum_k pi[k] * (A[k, e] + B_v[k, e])
//
// The shared part cost(e) - sum_k pi[k] * A[k, e] is computed once per dual
// update. offset_v and B_v are sparse per-vessel overlays, applied to a
// worker-local copy of the shared array for the duration of one solve and
// undone afterwards, so no vessel-by-edge matrix is ever materialised.
#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <vector>

#include "celerispath/dag_solver.hpp"
#include "celerispath/graph.hpp"
#include "celerispath/pricing_engine.hpp"
#include "celerispath/rcsp_solver.hpp"
#include "celerispath/types.hpp"

namespace celerispath {

enum class Scheduler : int {
    Threads = 0,  // std::thread workers claiming vessels from an atomic counter
    Tbb = 1,      // oneTBB task arena with work stealing
};

enum class Partitioner : int {
    Auto = 0,    // tbb::auto_partitioner: chunks that split further when stolen
    Simple = 1,  // tbb::simple_partitioner: one task per vessel
    Static = 2,  // tbb::static_partitioner: even contiguous shares, no stealing
};

// True when the library was built with oneTBB.
bool tbb_available();
// Version of the oneTBB runtime in use, or "" without oneTBB.
std::string tbb_runtime_version();

// Master-constraint coefficients shared by every vessel, by edge id. Either
// the general sparse form, or a fixed number of (row, value) pairs per edge.
struct DualCoefficients {
    std::size_t num_rows = 0;
    // width == 0: compressed sparse columns.
    SparseColumns sparse;
    // width > 0: edge e owns row[e * width + j], value[e * width + j] for
    // j < width; a negative row marks an unused pair.
    std::size_t width = 0;
    std::span<const std::int64_t> row;
    std::span<const double> value;
};

struct FleetOptions {
    Algorithm algorithm = Algorithm::Dag;
    RcspOptions rcsp;
    Scheduler scheduler = Scheduler::Threads;
    Partitioner partitioner = Partitioner::Auto;
    unsigned threads = 0;  // 0: one per hardware thread
    // false: reduce the shared costs once per dual update (the default).
    // true: every vessel reduces the whole cost array itself; kept as the
    // baseline that the shared design is measured against.
    bool per_vessel_update = false;
    // Completion bounds (resource_to_go) are cached per distinct target while
    // they fit in this many bytes; beyond it they are recomputed per solve.
    std::size_t bound_cache_bytes = std::size_t{1} << 30;
};

// A batch of vessels to register. Per-vessel lists are in compressed row
// form: vessel i owns entries [ptr[i], ptr[i + 1]) of the matching arrays.
struct FleetVessels {
    std::span<const std::int64_t> source;
    std::span<const std::int64_t> target;
    std::span<const double> fixed_cost;
    // Upper bounds of the tracked resources, row-major (count x tracked);
    // empty to use the bounds of the graph for every vessel.
    std::span<const double> limits;
    // Route-level master coefficients (e.g. a convexity row).
    std::span<const std::int64_t> fixed_ptr;
    std::span<const std::int64_t> fixed_row;
    std::span<const double> fixed_value;
    // Edge-cost offsets; +infinity removes the edge for the vessel.
    std::span<const std::int64_t> offset_ptr;
    std::span<const std::int64_t> offset_edge;
    std::span<const double> offset_value;
    // Vessel-specific master coefficients B_v as (edge, row, value) triplets.
    std::span<const std::int64_t> coef_ptr;
    std::span<const std::int64_t> coef_edge;
    std::span<const std::int64_t> coef_row;
    std::span<const double> coef_value;
};

// Outcome of one fleet solve, as flat arrays. Vessel i owns columns
// [column_ptr[i], column_ptr[i + 1]); column c owns edges
// [edge_ptr[c], edge_ptr[c + 1]) and coefficients [coef_ptr[c], coef_ptr[c + 1]).
struct FleetResult {
    // Per vessel. A status of Interrupted means the vessel was not priced to
    // completion (cancelled, out of time, or failed) and proves nothing.
    std::vector<std::int8_t> status;
    std::vector<double> best_reduced_cost;
    std::vector<double> vessel_seconds;
    std::vector<std::string> errors;  // empty where the vessel did not fail
    std::vector<std::int64_t> column_ptr;

    // Per column, vessels in registration order, cheapest reduced cost first.
    std::vector<std::int64_t> column_vessel;
    std::vector<double> original_cost;
    std::vector<double> reduced_cost;
    std::vector<double> resources;  // columns x resource columns, row-major
    std::vector<std::int64_t> edge_ptr;
    std::vector<std::int64_t> edges;
    std::vector<std::int64_t> coef_ptr;
    std::vector<std::int64_t> coef_row;  // ascending within a column
    std::vector<double> coef_value;

    bool cancelled = false;        // cancel() was observed
    bool timed_out = false;        // the time limit expired
    std::size_t num_failed = 0;    // vessels with a nonempty error
    std::size_t num_interrupted = 0;
    unsigned threads = 0;          // worker slots available to the solve
    double update_seconds = 0.0;   // dual update, when fused into the call
    double solve_seconds = 0.0;    // wall time of the parallel pricing phase
    double assemble_seconds = 0.0; // merging per-vessel outcomes
    SolveStats labels;             // summed over vessels (max for max_active_labels)
};

// Bytes held by a fleet, by owner.
struct FleetMemory {
    std::size_t shared_graph = 0;   // topology, costs and resource tables
    std::size_t shared_pricing = 0; // coefficients, reduced costs, edge map
    std::size_t vessels = 0;        // profiles, limits, overlays
    std::size_t bound_cache = 0;    // cached completion bounds
    std::size_t workspaces = 0;     // per-worker scratch memory
    std::size_t results = 0;        // per-vessel result slots
};

class PricingFleet {
public:
    PricingFleet(std::shared_ptr<const CompiledGraph> graph, const DualCoefficients& coefficients,
                 const FleetOptions& options);
    ~PricingFleet();
    PricingFleet(const PricingFleet&) = delete;
    PricingFleet& operator=(const PricingFleet&) = delete;

    // Appends vessels and returns the index of the first one. Either every
    // vessel of the batch is added or (on a validation error) none is.
    std::size_t add_vessels(const FleetVessels& vessels);

    // Builds everything that does not depend on the duals: reachability and
    // completion bounds per vessel, the worker arena and its workspaces.
    // Idempotent; called implicitly by the first solve after a change.
    void prepare();

    // Stores the duals and, in shared mode, refreshes the shared reduced
    // costs. One call per column-generation iteration, whatever the fleet size.
    void update_duals(std::span<const double> duals);

    // Prices every vessel with the current duals. Thread-safe; concurrent
    // calls on one fleet are serialised. With negative_only, only columns
    // with reduced cost < -tolerance are returned. time_limit is in seconds;
    // +infinity means none, and 0 interrupts every vessel.
    FleetResult solve(std::uint32_t max_paths, bool negative_only, double tolerance,
                      double time_limit);

    // update_duals followed by solve, under one lock.
    FleetResult price(std::span<const double> duals, std::uint32_t max_paths, bool negative_only,
                      double tolerance, double time_limit);

    // Asks the solve in progress to stop; a no-op when none is running. Safe
    // to call from any thread.
    void cancel();

    const CompiledGraph& graph() const { return *graph_; }
    Algorithm algorithm() const { return options_.algorithm; }
    Scheduler scheduler() const { return options_.scheduler; }
    std::size_t num_constraints() const { return num_constraints_; }
    std::size_t num_vessels() const;
    unsigned threads() const;  // worker slots; resolved by prepare()
    bool prepared() const;
    bool has_duals() const;
    std::vector<std::uint8_t> reachable() const;  // per vessel, after prepare()
    FleetMemory memory() const;

    // Test hook: the given vessel throws inside its worker on every solve
    // (kInvalid switches it off). Exercises error isolation.
    void debug_fail_vessel(std::uint32_t vessel);

private:
    struct Vessel;
    struct Bound;
    struct Workspace;
    struct Outcome;
    struct SolveParams;
    class Executor;

    void load_coefficients(const DualCoefficients& coefficients);
    unsigned resolved_threads() const;
    void ensure_executor();
    void prepare_locked();
    void update_locked(std::span<const double> duals);
    FleetResult solve_locked(std::uint32_t max_paths, bool negative_only, double tolerance,
                             double time_limit, double update_seconds);
    void reduce(Slot begin, Slot end, double* out) const;
    void price_vessel(std::size_t index, Workspace& workspace, const SolveParams& params,
                      Outcome& outcome) const;
    void append_column(const Vessel& vessel, const PathResult& path, double fixed_reduced,
                       Workspace& workspace, Outcome& outcome) const;
    FleetResult assemble() const;

    std::shared_ptr<const CompiledGraph> graph_;
    FleetOptions options_;
    std::size_t num_constraints_ = 0;

    // ---- shared, immutable after construction --------------------------
    std::vector<Slot> edge_slot_;  // slot of each edge id
    // Shared coefficients by slot. Sparse form: coef_ptr_ has num_slots + 1
    // entries. Indexed form: width_ pairs per slot, unused pairs point at the
    // extra zero that follows the duals in duals_.
    std::size_t width_ = 0;
    std::vector<std::uint32_t> coef_ptr_;
    std::vector<std::uint32_t> coef_row_;
    std::vector<double> coef_value_;

    // ---- vessels: appended by add_vessels, read-only during a solve -----
    std::vector<Vessel> vessels_;
    std::vector<double> limits_;            // vessels x tracked (rcsp only, else empty)
    std::vector<std::uint32_t> fixed_row_;
    std::vector<double> fixed_value_;
    std::vector<Slot> overlay_slot_;        // ascending within a vessel
    std::vector<double> overlay_offset_;
    std::vector<std::uint32_t> overlay_coef_ptr_;  // overlay entries + 1
    std::vector<std::uint32_t> overlay_coef_row_;
    std::vector<double> overlay_coef_value_;

    // ---- built by prepare() ---------------------------------------------
    bool prepared_ = false;
    std::size_t prepared_vessels_ = 0;  // vessels whose reachability is known
    std::vector<Bound> bounds_;         // resource_to_go per distinct target
    std::unique_ptr<Executor> executor_;
    std::vector<Workspace> workspaces_;  // one per worker slot
    std::vector<Outcome> outcomes_;      // one per vessel, reused between solves

    // ---- dual-dependent state, written only by update_duals -------------
    bool has_duals_ = false;
    std::uint64_t epoch_ = 0;             // bumped whenever shared_reduced_ changes
    std::vector<double> duals_;           // num_constraints_ + 1; the last entry is 0
    std::vector<double> shared_reduced_;  // by slot

    mutable std::mutex mutex_;  // guards everything above
    std::atomic<bool> running_{false};
    std::atomic<bool> cancel_requested_{false};
    std::atomic<std::uint32_t> debug_fail_{kInvalid};
};

}  // namespace celerispath
