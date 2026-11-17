#include "celerispath/rcsp_solver.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/r_c_shortest_paths.hpp>

#include "celerispath/bgl_adaptor.hpp"

namespace celerispath {

namespace {

// Resource container handed to Boost: objective cost, the tracked resources,
// and the topological rank of the vertex the label resides at.
//
// The rank is not a resource. It exists because Boost orders its queue of
// unprocessed labels with operator< on this container alone; ranking by
// topological position first means every label of a vertex exists before the
// first of them is extended. Dominance is then applied to complete label
// sets, and negative edge costs cannot cause a label to be extended and later
// found dominated.
template <class Store>
struct Label {
    double cost = 0.0;
    std::uint32_t rank = 0;
    Store resource{};
};

template <class Store>
bool operator<(const Label<Store>& a, const Label<Store>& b) {
    if (a.rank != b.rank) return a.rank < b.rank;
    return a.cost < b.cost;
}

template <class Store>
bool operator==(const Label<Store>& a, const Label<Store>& b) {
    return a.rank == b.rank && a.cost == b.cost && a.resource == b.resource;
}

// Resource extension function: R_new = R_old + delta(edge); feasible iff
// lower <= R_new <= upper for every tracked resource. A feasible label is
// still rejected when it provably cannot be completed: its vertex lies beyond
// the target in topological order, or R_new plus the least consumption on the
// way to the target exceeds `limit` (upper plus a rounding margin).
template <class Store>
struct Extend {
    const Csr* csr;
    const double* slot_cost;
    const RcspData* data;
    const double* to_go;
    const double* upper;
    const double* limit;
    std::uint32_t target_rank;

    bool operator()(const BglGraph&, Label<Store>& next, const Label<Store>& previous,
                    EdgeRef e) const {
        const double edge_cost = slot_cost[e.slot];
        if (edge_cost == kInf) return false;  // edge removed for this solve
        // Boost constructs `next` as a copy of `previous`, so a dynamically
        // sized store already has the right length.
        const std::size_t count = next.resource.size();
        const double* delta = data->slot_delta.data() + static_cast<std::size_t>(e.slot) * count;
        const double* lower = data->lower.data();
        const VertexId head = csr->slot_target[e.slot];
        next.cost = previous.cost + edge_cost;
        next.rank = csr->topo_rank[head];
        if (next.rank > target_rank) return false;
        const double* rest = to_go + static_cast<std::size_t>(head) * count;
        bool feasible = true;
        for (std::size_t i = 0; i < count; ++i) {
            const double value = previous.resource[i] + delta[i];
            next.resource[i] = value;
            feasible &= (value >= lower[i]) & (value <= upper[i]) & (value + rest[i] <= limit[i]);
        }
        return feasible;
    }
};

// a dominates b iff a is no worse in cost and in every resource. Boost calls
// this in both directions and discards one label when both hold, so of two
// equal labels exactly one representative survives.
//
// "Smaller is no worse" is valid only for the monotone cumulative resources
// of RcspData; see the note there before reusing this for other resources.
template <class Store>
struct Dominates {
    double tolerance;

    bool operator()(const Label<Store>& a, const Label<Store>& b) const {
        if (a.cost > b.cost + tolerance) return false;
        const std::size_t count = a.resource.size();
        for (std::size_t i = 0; i < count; ++i) {
            if (a.resource[i] > b.resource[i] + tolerance) return false;
        }
        return true;
    }
};

struct Visitor : boost::default_r_c_shortest_paths_visitor {
    // Loop iterations between two polls of the stop token.
    static constexpr std::uint32_t kStopInterval = 256;

    SolveStats* stats;
    std::uint32_t target_rank;
    const StopToken* stop = nullptr;
    bool* interrupted = nullptr;  // Boost copies the visitor, so the flag lives outside
    std::uint32_t countdown = kStopInterval;

    template <class L, class G>
    void on_label_feasible(const L&, const G&) {
        ++stats->labels_created;
    }
    template <class L, class G>
    void on_label_not_feasible(const L&, const G&) {
        ++stats->labels_infeasible;
    }
    template <class L, class G>
    void on_label_dominated(const L&, const G&) {
        ++stats->labels_dominated;
    }
    template <class L, class G>
    void on_label_not_dominated(const L&, const G&) {
        ++stats->labels_extended;
    }
    // Labels leave the queue in topological order, so once the cheapest
    // remaining label lies beyond the target no label can reach it any more.
    template <class Queue, class G>
    bool on_enter_loop(const Queue& queue, const G&) {
        if (stop != nullptr && --countdown == 0) {
            countdown = kStopInterval;
            if (stop->stop_requested()) {
                *interrupted = true;
                return false;
            }
        }
        stats->max_active_labels =
            std::max<std::uint64_t>(stats->max_active_labels, queue.size());
        return queue.top()->cumulated_resource_consumption.rank <= target_rank;
    }
};

// boost::r_c_shortest_paths forwards to an implementation function in
// boost::detail through an unqualified call, so that function is only found
// when argument-dependent lookup reaches boost::detail. Boost's own graph
// classes get there through their edge descriptors; a graph type defined
// outside Boost does not. Parameterising the (stateless) label allocator on
// such a descriptor adds the namespace without touching any Boost internals.
using BoostLookupAnchor = boost::graph_traits<boost::adjacency_list<>>::edge_descriptor;

template <class Store>
void set_initial(Store& store, const RcspData& data) {
    std::copy(data.initial.begin(), data.initial.end(), store.begin());
}

void set_initial(std::vector<double>& store, const RcspData& data) { store = data.initial; }

template <class Store>
PathResult run(const Csr& csr, const RcspData& data, const double* slot_cost, VertexId source,
               VertexId target, const RcspOptions& options, RcspWorkspace& workspace,
               SolveStats& stats, const double* to_go, const double* upper,
               const StopToken* stop) {
    if (upper == nullptr) upper = data.upper.data();
    // Forward and backward sums of the same nonnegative increments differ by
    // a relative error far below 1e-9 for any path that fits in memory.
    workspace.limit.resize(data.num_resources);
    for (std::size_t i = 0; i < data.num_resources; ++i) {
        if (upper[i] == kInf) {
            workspace.limit[i] = kInf;
            continue;
        }
        const double scale = std::max(std::abs(upper[i]), std::abs(data.initial[i]));
        workspace.limit[i] = upper[i] + 1e-9 * scale;
    }
    // Boost never tests the source label, so an overriding bound below the
    // initial value has to be caught here.
    for (std::size_t i = 0; i < data.num_resources; ++i) {
        if (data.initial[i] > upper[i]) {
            PathResult none;
            none.status = Status::Infeasible;
            return none;
        }
    }

    Label<Store> initial;
    initial.rank = csr.topo_rank[source];
    set_initial(initial.resource, data);

    stats.labels_created = 1;  // the source label
    Visitor visitor;
    visitor.stats = &stats;
    visitor.target_rank = csr.topo_rank[target];
    bool interrupted = false;
    visitor.stop = stop;
    visitor.interrupted = &interrupted;

    std::vector<std::vector<EdgeRef>> pareto_paths;
    std::vector<Label<Store>> pareto_labels;
    {
        const LabelPoolScope scope(&workspace.label_pool);
        boost::r_c_shortest_paths(BglGraph{&csr}, VertexIndexMap(), SlotIndexMap(), source,
                                  target, pareto_paths, pareto_labels, initial,
                                  Extend<Store>{&csr, slot_cost, &data, to_go, upper,
                                                workspace.limit.data(), visitor.target_rank},
                                  Dominates<Store>{options.dominance_tolerance},
                                  PoolAllocator<BoostLookupAnchor>(), visitor);
    }

    PathResult result;
    if (interrupted) {
        // The labels found so far prove nothing about the optimum.
        result.status = Status::Interrupted;
        return result;
    }
    stats.pareto_labels = pareto_labels.size();
    result.status = Status::Infeasible;
    if (pareto_labels.empty()) return result;

    // Boost returns the whole Pareto frontier at the target, in no cost
    // order. The single-path answer is its cheapest member (first on ties).
    std::size_t best = 0;
    for (std::size_t i = 1; i < pareto_labels.size(); ++i) {
        if (pareto_labels[i].cost < pareto_labels[best].cost) best = i;
    }
    result.status = Status::Optimal;
    result.cost = pareto_labels[best].cost;
    // Boost lists path edges from the target back to the source.
    result.slots.reserve(pareto_paths[best].size());
    for (auto it = pareto_paths[best].rbegin(); it != pareto_paths[best].rend(); ++it) {
        result.slots.push_back(it->slot);
    }
    return result;
}

}  // namespace

void resource_to_go(const Csr& csr, const RcspData& resources, VertexId source, VertexId target,
                    std::vector<double>& to_go) {
    const std::size_t count = resources.num_resources;
    to_go.resize(static_cast<std::size_t>(csr.num_vertices) * count);
    const std::uint32_t first = csr.topo_rank[source];
    const std::uint32_t last = csr.topo_rank[target];
    if (count == 0 || first > last) return;
    for (std::uint32_t rank = last + 1; rank-- > first;) {
        const VertexId v = csr.topo_order[rank];
        double* out = to_go.data() + static_cast<std::size_t>(v) * count;
        if (v == target) {
            std::fill(out, out + count, 0.0);
            continue;
        }
        std::fill(out, out + count, kInf);
        for (Slot slot = csr.offsets[v]; slot < csr.offsets[v + 1]; ++slot) {
            const VertexId head = csr.slot_target[slot];
            if (csr.topo_rank[head] > last) continue;
            const double* rest = to_go.data() + static_cast<std::size_t>(head) * count;
            const double* delta =
                resources.slot_delta.data() + static_cast<std::size_t>(slot) * count;
            for (std::size_t i = 0; i < count; ++i) out[i] = std::min(out[i], delta[i] + rest[i]);
        }
    }
}

PathResult solve_rcsp(const Csr& csr, const RcspData& resources, const double* slot_cost,
                      VertexId source, VertexId target, const RcspOptions& options,
                      RcspWorkspace& workspace, SolveStats& stats, const double* to_go,
                      const double* upper, const StopToken* stop) {
    // Fixed-size stores keep the common cases free of per-label heap
    // allocations and let the compiler unroll the resource loops.
    switch (resources.num_resources) {
        case 0:
            return run<std::array<double, 0>>(csr, resources, slot_cost, source, target, options,
                                              workspace, stats, to_go, upper, stop);
        case 1:
            return run<std::array<double, 1>>(csr, resources, slot_cost, source, target, options,
                                              workspace, stats, to_go, upper, stop);
        case 2:
            return run<std::array<double, 2>>(csr, resources, slot_cost, source, target, options,
                                              workspace, stats, to_go, upper, stop);
        case 3:
            return run<std::array<double, 3>>(csr, resources, slot_cost, source, target, options,
                                              workspace, stats, to_go, upper, stop);
        case 4:
            return run<std::array<double, 4>>(csr, resources, slot_cost, source, target, options,
                                              workspace, stats, to_go, upper, stop);
        default:
            return run<std::vector<double>>(csr, resources, slot_cost, source, target, options,
                                            workspace, stats, to_go, upper, stop);
    }
}

}  // namespace celerispath
