// Adapts celerispath::Csr to the Boost Graph Library concepts required by
// boost::dag_shortest_paths and boost::r_c_shortest_paths (IncidenceGraph and
// VertexListGraph), so both solvers traverse the same contiguous arrays
// without copying them into a Boost container.
#pragma once

#include <cstdint>
#include <utility>

#include <boost/graph/graph_traits.hpp>
#include <boost/iterator/counting_iterator.hpp>
#include <boost/iterator/transform_iterator.hpp>
#include <boost/property_map/property_map.hpp>

#include "celerispath/types.hpp"

namespace celerispath {

// Edge descriptor: a distinct type (rather than a bare integer) so it can
// never be confused with a vertex descriptor in overload resolution.
struct EdgeRef {
    Slot slot = kInvalid;
    friend bool operator==(EdgeRef a, EdgeRef b) { return a.slot == b.slot; }
    friend bool operator!=(EdgeRef a, EdgeRef b) { return a.slot != b.slot; }
};

struct SlotToEdgeRef {
    using result_type = EdgeRef;
    EdgeRef operator()(Slot slot) const { return EdgeRef{slot}; }
};

struct BglGraph {
    const Csr* csr = nullptr;
};

struct BglTraversalTag : boost::incidence_graph_tag, boost::vertex_list_graph_tag {};

// Readable property map: edge descriptor -> per-slot cost.
struct SlotCostMap {
    const double* cost = nullptr;
};
inline double get(const SlotCostMap& map, EdgeRef e) { return map.cost[e.slot]; }

// Readable property map: edge descriptor -> edge index.
struct SlotIndexMap {};
inline std::size_t get(const SlotIndexMap&, EdgeRef e) { return e.slot; }

using VertexIndexMap = boost::typed_identity_property_map<VertexId>;

}  // namespace celerispath

namespace boost {

template <>
struct graph_traits<celerispath::BglGraph> {
    using vertex_descriptor = celerispath::VertexId;
    using edge_descriptor = celerispath::EdgeRef;
    using directed_category = directed_tag;
    using edge_parallel_category = allow_parallel_edge_tag;
    using traversal_category = celerispath::BglTraversalTag;
    using vertices_size_type = std::uint32_t;
    using edges_size_type = std::uint32_t;
    using degree_size_type = std::uint32_t;
    using vertex_iterator = counting_iterator<celerispath::VertexId>;
    using out_edge_iterator =
        transform_iterator<celerispath::SlotToEdgeRef, counting_iterator<celerispath::Slot>>;

    static vertex_descriptor null_vertex() { return celerispath::kInvalid; }
};

template <>
struct property_traits<celerispath::SlotCostMap> {
    using value_type = double;
    using reference = double;
    using key_type = celerispath::EdgeRef;
    using category = readable_property_map_tag;
};

template <>
struct property_traits<celerispath::SlotIndexMap> {
    using value_type = std::size_t;
    using reference = std::size_t;
    using key_type = celerispath::EdgeRef;
    using category = readable_property_map_tag;
};

}  // namespace boost

namespace celerispath {

using BglTraits = boost::graph_traits<BglGraph>;

inline std::uint32_t num_vertices(const BglGraph& g) { return g.csr->num_vertices; }
inline std::uint32_t num_edges(const BglGraph& g) { return g.csr->num_slots(); }

inline std::pair<BglTraits::vertex_iterator, BglTraits::vertex_iterator> vertices(
    const BglGraph& g) {
    return {BglTraits::vertex_iterator(0), BglTraits::vertex_iterator(g.csr->num_vertices)};
}

inline std::pair<BglTraits::out_edge_iterator, BglTraits::out_edge_iterator> out_edges(
    VertexId v, const BglGraph& g) {
    using Counter = boost::counting_iterator<Slot>;
    return {BglTraits::out_edge_iterator(Counter(g.csr->offsets[v]), SlotToEdgeRef{}),
            BglTraits::out_edge_iterator(Counter(g.csr->offsets[v + 1]), SlotToEdgeRef{})};
}

inline std::uint32_t out_degree(VertexId v, const BglGraph& g) {
    return g.csr->offsets[v + 1] - g.csr->offsets[v];
}

inline VertexId source(EdgeRef e, const BglGraph& g) { return g.csr->slot_source[e.slot]; }
inline VertexId target(EdgeRef e, const BglGraph& g) { return g.csr->slot_target[e.slot]; }

}  // namespace celerispath
