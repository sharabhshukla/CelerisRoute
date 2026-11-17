// Unconstrained shortest paths on a directed acyclic graph.
#pragma once

#include <cstdint>
#include <vector>

#include <boost/graph/properties.hpp>

#include "celerispath/types.hpp"

namespace celerispath {

// Scratch buffers reused across solves; sized on first use.
struct DagWorkspace {
    std::vector<double> distance;
    std::vector<boost::default_color_type> color;
    std::vector<Slot> pred_slot;
};

// Minimum-cost source-target path via boost::dag_shortest_paths. Edge costs
// may be negative. Returns Status::Unreachable when no path exists. An edge
// whose cost is +infinity is never relaxed, so it behaves as if it were
// absent (the same holds for solve_dag_k_best).
//
// Boost derives its own topological order with a depth-first visit from the
// source on every call, so one solve costs O(V + E) and the order cached in
// `csr` is not consumed here.
PathResult solve_dag(const Csr& csr, const double* slot_cost, VertexId source, VertexId target,
                     DagWorkspace& workspace);

struct KBestWorkspace {
    std::vector<double> distance;          // num_vertices * k, ascending per vertex
    std::vector<Slot> pred_slot;           // edge used to reach the label
    std::vector<std::uint32_t> pred_rank;  // which label of the predecessor it extends
    std::vector<std::uint32_t> count;      // labels currently stored per vertex
};

// The k cheapest distinct source-target paths in nondecreasing cost order
// (fewer if fewer exist). This is celerispath's own dynamic program over the
// cached topological order; Boost has no k-shortest-path routine for DAGs.
// Runs in O(V + E * k^2) time and O(V * k) memory.
std::vector<PathResult> solve_dag_k_best(const Csr& csr, const double* slot_cost, VertexId source,
                                         VertexId target, std::uint32_t k,
                                         KBestWorkspace& workspace);

}  // namespace celerispath
