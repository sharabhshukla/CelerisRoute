#include "celerispath/dag_solver.hpp"

#include <algorithm>
#include <functional>

#include <boost/graph/dag_shortest_paths.hpp>
#include <boost/property_map/property_map.hpp>

#include "celerispath/bgl_adaptor.hpp"

namespace celerispath {

namespace {

// Boost's predecessor map stores vertices, which cannot tell parallel edges
// apart. Recording the relaxed edge itself keeps edge ids exact.
struct RecordPredecessorEdge : boost::default_dijkstra_visitor {
    explicit RecordPredecessorEdge(Slot* pred_slot) : pred_slot_(pred_slot) {}

    void edge_relaxed(EdgeRef e, const BglGraph& g) const { pred_slot_[target(e, g)] = e.slot; }

private:
    Slot* pred_slot_;
};

}  // namespace

PathResult solve_dag(const Csr& csr, const double* slot_cost, VertexId source, VertexId target,
                     DagWorkspace& workspace) {
    const std::uint32_t n = csr.num_vertices;
    workspace.distance.resize(n);
    workspace.pred_slot.resize(n);
    workspace.color.assign(n, boost::white_color);

    const BglGraph graph{&csr};
    const VertexIndexMap index;
    boost::dag_shortest_paths(
        graph, source, boost::make_iterator_property_map(workspace.distance.begin(), index),
        SlotCostMap{slot_cost}, boost::make_iterator_property_map(workspace.color.begin(), index),
        boost::dummy_property_map(), RecordPredecessorEdge(workspace.pred_slot.data()),
        std::less<double>(), boost::closed_plus<double>(kInf), kInf, 0.0);

    PathResult result;
    if (workspace.distance[target] == kInf) return result;  // Status::Unreachable
    result.status = Status::Optimal;
    result.cost = workspace.distance[target];
    // Every vertex on the chain was relaxed during this call, so no stale
    // entry of pred_slot from an earlier solve can be read.
    for (VertexId v = target; v != source;) {
        const Slot slot = workspace.pred_slot[v];
        result.slots.push_back(slot);
        v = csr.slot_source[slot];
    }
    std::reverse(result.slots.begin(), result.slots.end());
    return result;
}

std::vector<PathResult> solve_dag_k_best(const Csr& csr, const double* slot_cost, VertexId source,
                                         VertexId target, std::uint32_t k,
                                         KBestWorkspace& workspace) {
    std::vector<PathResult> results;
    if (k == 0) return results;
    const std::size_t n = csr.num_vertices;
    const std::uint32_t source_rank = csr.topo_rank[source];
    const std::uint32_t target_rank = csr.topo_rank[target];
    if (source_rank > target_rank) return results;

    workspace.count.assign(n, 0);
    workspace.distance.resize(n * k);
    workspace.pred_slot.resize(n * k);
    workspace.pred_rank.resize(n * k);
    double* const distance = workspace.distance.data();
    Slot* const pred_slot = workspace.pred_slot.data();
    std::uint32_t* const pred_rank = workspace.pred_rank.data();
    std::uint32_t* const count = workspace.count.data();

    count[source] = 1;
    distance[static_cast<std::size_t>(source) * k] = 0.0;
    pred_slot[static_cast<std::size_t>(source) * k] = kInvalid;

    // Vertices are finalised in topological order: when u is expanded, its
    // label list already holds the k cheapest paths from the source to u.
    for (std::uint32_t rank = source_rank; rank < target_rank; ++rank) {
        const VertexId u = csr.topo_order[rank];
        const std::uint32_t labels_u = count[u];
        if (labels_u == 0) continue;
        const double* const distance_u = distance + static_cast<std::size_t>(u) * k;
        for (Slot slot = csr.offsets[u]; slot < csr.offsets[u + 1]; ++slot) {
            const VertexId v = csr.slot_target[slot];
            if (csr.topo_rank[v] > target_rank) continue;
            const double weight = slot_cost[slot];
            if (weight == kInf) continue;  // edge removed for this solve
            double* const distance_v = distance + static_cast<std::size_t>(v) * k;
            Slot* const slot_v = pred_slot + static_cast<std::size_t>(v) * k;
            std::uint32_t* const rank_v = pred_rank + static_cast<std::size_t>(v) * k;
            for (std::uint32_t i = 0; i < labels_u; ++i) {
                const double candidate = distance_u[i] + weight;
                std::uint32_t size = count[v];
                // Candidates along one edge are nondecreasing in i, so once
                // one fails to enter a full list the rest fail too. Ties keep
                // the earlier label, which makes the output deterministic.
                if (size == k && !(candidate < distance_v[k - 1])) break;
                std::uint32_t position = size < k ? size : k - 1;
                while (position > 0 && candidate < distance_v[position - 1]) {
                    distance_v[position] = distance_v[position - 1];
                    slot_v[position] = slot_v[position - 1];
                    rank_v[position] = rank_v[position - 1];
                    --position;
                }
                distance_v[position] = candidate;
                slot_v[position] = slot;
                rank_v[position] = i;
                if (size < k) count[v] = size + 1;
            }
        }
    }

    results.resize(count[target]);
    for (std::uint32_t i = 0; i < count[target]; ++i) {
        PathResult& result = results[i];
        result.status = Status::Optimal;
        result.cost = distance[static_cast<std::size_t>(target) * k + i];
        VertexId v = target;
        std::uint32_t label = i;
        while (true) {
            const std::size_t at = static_cast<std::size_t>(v) * k + label;
            const Slot slot = pred_slot[at];
            if (slot == kInvalid) break;
            result.slots.push_back(slot);
            label = pred_rank[at];
            v = csr.slot_source[slot];
        }
        std::reverse(result.slots.begin(), result.slots.end());
    }
    return results;
}

}  // namespace celerispath
