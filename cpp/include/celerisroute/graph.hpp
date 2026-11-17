// Mutable graph builder and the immutable compiled graph shared by all solvers.
#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "celerisroute/types.hpp"

namespace celerisroute {

// Bounds of a declared resource. Edge increments live in the builder columns.
struct ResourceSpec {
    std::string name;
    double lower = -kInf;
    double upper = kInf;
    double initial = 0.0;
};

// Frozen topology. Shared (never copied) between cost variants of a graph.
struct Topology {
    std::uint32_t num_vertices = 0;
    std::vector<VertexId> edge_source;  // by edge id
    std::vector<VertexId> edge_target;  // by edge id
    Csr csr;                            // csr.slot_edge maps slots to edge ids
    bool acyclic = false;
    // When cyclic: one directed cycle v0 -> v1 -> ... -> v0 (v0 not repeated).
    std::vector<VertexId> cycle;

    std::uint32_t num_edges() const { return static_cast<std::uint32_t>(edge_source.size()); }
};

// Every edge-resource column of a graph, declared or not. Columns that are
// declared with a finite upper bound are "tracked": they are carried in RCSP
// labels. All other columns are only summed along returned paths.
struct ResourceTable {
    std::vector<std::string> names;
    std::vector<std::vector<double>> edge_delta;  // [column][edge id]
    std::vector<double> lower;
    std::vector<double> upper;
    std::vector<double> initial;
    std::vector<std::uint8_t> declared;
    std::vector<std::uint32_t> tracked;  // column index of each tracked resource
    RcspData tracked_data;               // tracked columns in slot order

    std::size_t num_columns() const { return names.size(); }
};

struct Solution {
    Status status = Status::Unreachable;
    double cost = kInf;
    std::vector<EdgeId> edges;      // travel order
    std::vector<double> resources;  // final value per resource column
    SolveStats stats;
};

class CompiledGraph;

class GraphBuilder {
public:
    explicit GraphBuilder(std::uint64_t num_vertices);

    std::uint32_t num_vertices() const { return num_vertices_; }
    std::uint32_t num_edges() const { return static_cast<std::uint32_t>(source_.size()); }
    const std::vector<std::string>& resource_names() const { return resource_names_; }

    // Appends one edge and returns its id. Throws std::invalid_argument on
    // out-of-range endpoints or nonfinite values, leaving the builder unchanged.
    EdgeId add_edge(std::int64_t source, std::int64_t target, double cost,
                    std::span<const std::pair<std::string, double>> resources);

    // Appends source.size() edges and returns the id of the first one. Either
    // every edge is appended or (on a validation error) none is.
    EdgeId add_edges(std::span<const std::int64_t> source, std::span<const std::int64_t> target,
                     std::span<const double> cost, std::span<const std::string> resource_names,
                     std::span<const std::span<const double>> resource_values);

    // Snapshots the current edges into an immutable compiled graph. The
    // builder stays usable; later edits never affect an existing snapshot.
    std::shared_ptr<const CompiledGraph> compile(std::span<const ResourceSpec> declared) const;

private:
    std::size_t resource_column(const std::string& name);
    void check_capacity(std::size_t additional) const;

    std::uint32_t num_vertices_;
    std::vector<VertexId> source_;
    std::vector<VertexId> target_;
    std::vector<double> cost_;
    std::vector<std::string> resource_names_;
    std::vector<std::vector<double>> resource_columns_;  // each of size num_edges()
};

struct SolveWorkspace;  // defined in graph.cpp

class CompiledGraph {
public:
    CompiledGraph(std::shared_ptr<const Topology> topology,
                  std::shared_ptr<const ResourceTable> resources, std::vector<double> edge_cost);
    ~CompiledGraph();
    CompiledGraph(const CompiledGraph&) = delete;
    CompiledGraph& operator=(const CompiledGraph&) = delete;

    const Topology& topology() const { return *topology_; }
    const ResourceTable& resources() const { return *resources_; }
    std::span<const double> edge_cost() const { return edge_cost_; }
    std::span<const double> slot_cost() const { return slot_cost_; }

    // A graph with identical (shared) topology and resources but new costs.
    std::shared_ptr<const CompiledGraph> with_costs(std::span<const double> edge_cost) const;

    // Single source-target solve on the whole graph. Thread-safe: concurrent
    // calls draw separate workspaces from an internal pool.
    Solution solve(VertexId source, VertexId target, Algorithm algorithm,
                   const RcspOptions& options) const;

    // Final value of every resource column along `edges` (initial + increments,
    // accumulated in travel order).
    std::vector<double> path_resources(std::span<const EdgeId> edges) const;

private:
    std::shared_ptr<const Topology> topology_;
    std::shared_ptr<const ResourceTable> resources_;
    std::vector<double> edge_cost_;
    std::vector<double> slot_cost_;

    mutable std::mutex workspace_mutex_;
    mutable std::vector<std::unique_ptr<SolveWorkspace>> workspaces_;
};

// True when `target` can be reached from `source` in an acyclic Csr.
bool is_reachable(const Csr& csr, VertexId source, VertexId target);

}  // namespace celerisroute
