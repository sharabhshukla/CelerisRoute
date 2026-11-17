// Resource-constrained shortest paths on a directed acyclic graph.
#pragma once

#include <vector>

#include "celerispath/label_pool.hpp"
#include "celerispath/types.hpp"

namespace celerispath {

// Label memory is kept between solves so repeated pricing does not return to
// the system allocator for every label.
struct RcspWorkspace {
    LabelPool label_pool;
    std::vector<double> to_go;  // scratch for callers that compute resource_to_go per solve
    std::vector<double> limit;  // pruning thresholds of the running solve
};

// For every vertex ranked between `source` and `target`, the least amount of
// each tracked resource that any path from the vertex to `target` consumes
// (+infinity when the target cannot be reached from it). `to_go` is resized to
// num_resources * num_vertices and indexed [vertex * num_resources + i];
// entries of vertices outside that rank range are left untouched.
//
// The minima are taken per resource, over possibly different paths, so they
// are lower bounds on what any single completion needs. One backward sweep,
// O((V + E) * num_resources) restricted to the rank range.
void resource_to_go(const Csr& csr, const RcspData& resources, VertexId source, VertexId target,
                    std::vector<double>& to_go);

// Minimum-cost source-target path whose accumulated resources stay within
// [lower, upper] after every edge, via boost::r_c_shortest_paths.
//
// Requirements (validated when the graph is compiled, not here):
//   * `csr` is acyclic and carries its topological ranks;
//   * every resource increment in `resources.slot_delta` is nonnegative.
// Under these conditions the result is exact for options.dominance_tolerance
// == 0, including with negative edge costs.
//
// `to_go` is the output of resource_to_go for the same source and target. It
// is used to discard, at creation, labels that cannot be completed to the
// target within the upper bounds: a label at v with consumption r is dropped
// when r + to_go[v] exceeds the bound. Such a label has no feasible
// completion, so dropping it never changes the optimum. The comparison leaves
// a relative margin of 1e-9 so that rounding (the bound is summed from the
// target, a path from the source) can never discard a path that the exact
// per-edge feasibility test accepts.
//
// `upper`, when given, replaces resources.upper for this solve (one value per
// tracked resource; +infinity leaves a resource unbounded). It lets many
// subproblems with different limits share one RcspData.
//
// An edge whose slot_cost is +infinity is treated as absent, which is how a
// caller removes edges for one solve without touching the shared topology.
//
// `stop`, when given, is polled every few hundred labels; once it asks to
// stop, the solve returns Status::Interrupted without an answer.
//
// Returns Status::Infeasible when no feasible path reaches the target; the
// caller decides whether that is because the target is unreachable at all.
// Label counts can grow exponentially with the graph size.
PathResult solve_rcsp(const Csr& csr, const RcspData& resources, const double* slot_cost,
                      VertexId source, VertexId target, const RcspOptions& options,
                      RcspWorkspace& workspace, SolveStats& stats, const double* to_go,
                      const double* upper = nullptr, const StopToken* stop = nullptr);

}  // namespace celerispath
