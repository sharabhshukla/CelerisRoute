// Column-generation pricing: repeated shortest-path solves on a fixed graph
// whose edge costs are reduced by changing dual prices.
#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <vector>

#include "celerispath/dag_solver.hpp"
#include "celerispath/graph.hpp"
#include "celerispath/rcsp_solver.hpp"
#include "celerispath/types.hpp"

namespace celerispath {

// Master-constraint coefficients A (num_rows x num_edges) in compressed
// sparse column form: column e lists edge e's contribution to each row.
struct SparseColumns {
    std::size_t num_rows = 0;
    std::span<const std::int64_t> indptr;   // num_edges + 1
    std::span<const std::int64_t> indices;  // row of each entry
    std::span<const double> data;
};

struct PricingColumn {
    std::vector<EdgeId> edges;  // travel order
    double original_cost = 0.0;
    double reduced_cost = 0.0;
    std::vector<double> resources;        // final value per resource column
    std::vector<std::int64_t> coef_index;  // ascending master-constraint rows
    std::vector<double> coef_value;
};

struct PricingStats {
    double update_seconds = 0.0;       // reduced-cost refresh
    double solve_seconds = 0.0;        // shortest-path algorithm
    double reconstruct_seconds = 0.0;  // building columns from paths
    SolveStats labels;                 // RCSP label counters (zero for "dag")
};

struct PricingResult {
    // Status of the pricing subproblem itself, before `negative_only` filters.
    Status status = Status::Unreachable;
    // Smallest reduced cost over all feasible paths; +inf when there is none.
    double best_reduced_cost = kInf;
    std::vector<PricingColumn> columns;  // nondecreasing reduced cost
    PricingStats stats;
};

// Prices paths between one fixed source and target.
//
// Construction keeps only the vertices and edges that lie on some
// source-target path, renumbers them in topological order and stores the
// matching columns of A next to them. A solve then touches nothing but that
// subgraph: it rewrites the reduced-cost array and runs the solver.
class PricingEngine {
public:
    PricingEngine(std::shared_ptr<const CompiledGraph> graph, VertexId source, VertexId target,
                  Algorithm algorithm, const SparseColumns& coefficients, double fixed_cost,
                  std::span<const std::int64_t> fixed_index, std::span<const double> fixed_value,
                  const RcspOptions& options);

    // reduced(e) = cost(e) - sum_k duals[k] * A[k, e]; a path's reduced cost
    // adds fixed_cost - sum_k duals[k] * fixed[k]. With negative_only, only
    // columns with reduced cost < -tolerance are returned.
    //
    // max_paths > 1 is exact for Algorithm::Dag and rejected for
    // Algorithm::Rcsp. Calls on one engine are serialised by a mutex.
    PricingResult solve(std::span<const double> duals, std::uint32_t max_paths,
                        bool negative_only, double tolerance);

    const CompiledGraph& graph() const { return *graph_; }
    Algorithm algorithm() const { return algorithm_; }
    VertexId source() const { return source_; }
    VertexId target() const { return target_; }
    std::size_t num_constraints() const { return num_constraints_; }
    bool target_reachable() const { return reachable_; }
    std::uint32_t active_vertices() const { return csr_.num_vertices; }
    std::uint32_t active_edges() const { return csr_.num_slots(); }

private:
    void build_subgraph();
    void load_coefficients(const SparseColumns& coefficients);
    PricingColumn make_column(const PathResult& path, double fixed_reduced);

    std::shared_ptr<const CompiledGraph> graph_;
    Algorithm algorithm_;
    RcspOptions options_;
    VertexId source_;
    VertexId target_;

    bool reachable_ = false;
    Csr csr_;                        // pruned subgraph; vertex id == topological rank
    VertexId sub_source_ = 0;
    VertexId sub_target_ = 0;
    RcspData rcsp_data_;             // tracked resources by pruned slot
    std::vector<double> to_go_;      // resource_to_go of the pruned subgraph; dual-independent
    std::vector<double> base_cost_;  // original cost by pruned slot

    std::size_t num_constraints_ = 0;
    std::vector<std::size_t> coef_ptr_;  // pruned slots + 1
    std::vector<std::uint32_t> coef_row_;
    std::vector<double> coef_value_;
    double fixed_cost_ = 0.0;
    std::vector<std::uint32_t> fixed_row_;
    std::vector<double> fixed_value_;

    // Mutable per-solve state, guarded by mutex_.
    std::mutex mutex_;
    std::vector<double> reduced_cost_;
    DagWorkspace dag_workspace_;
    KBestWorkspace k_best_workspace_;
    RcspWorkspace rcsp_workspace_;
    std::vector<double> accumulator_;  // dense scratch over master rows
    std::vector<std::uint8_t> touched_flag_;
    std::vector<std::uint32_t> touched_;
};

struct BatchOutcome {
    std::vector<PricingResult> results;
    std::vector<std::string> errors;  // empty string where the engine succeeded
};

// Solves every engine once, spreading engines over up to `threads` native
// threads (0 = hardware concurrency, never more than engines.size()).
// `duals` holds either one vector shared by all engines or one per engine.
// results[i] always belongs to engines[i], independent of scheduling. An
// exception in one engine is captured in errors[i] and does not stop others.
BatchOutcome solve_batch(std::span<PricingEngine* const> engines,
                         std::span<const std::span<const double>> duals, std::uint32_t max_paths,
                         bool negative_only, double tolerance, unsigned threads);

}  // namespace celerispath
