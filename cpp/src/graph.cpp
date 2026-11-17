#include "celerispath/graph.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <unordered_map>

#include "celerispath/dag_solver.hpp"
#include "celerispath/rcsp_solver.hpp"

namespace celerispath {

namespace {

std::string edge_context(std::size_t first_new_edge, std::size_t offset) {
    return "edge " + std::to_string(first_new_edge + offset) + " (position " +
           std::to_string(offset) + " of this call)";
}

void check_vertex(std::int64_t v, std::uint32_t num_vertices, const char* role,
                  const std::string& context) {
    if (v < 0 || v >= static_cast<std::int64_t>(num_vertices)) {
        throw std::invalid_argument(context + ": " + role + " vertex id " + std::to_string(v) +
                                    " is outside [0, " + std::to_string(num_vertices) + ")");
    }
}

void check_finite(double value, const std::string& what, const std::string& context) {
    if (!std::isfinite(value)) {
        throw std::invalid_argument(context + ": " + what + " must be finite, got " +
                                    std::to_string(value));
    }
}

void build_csr(Topology& topo) {
    const std::uint32_t n = topo.num_vertices;
    const std::uint32_t m = topo.num_edges();
    Csr& csr = topo.csr;
    csr.num_vertices = n;
    csr.offsets.assign(static_cast<std::size_t>(n) + 1, 0);
    for (std::uint32_t e = 0; e < m; ++e) ++csr.offsets[topo.edge_source[e] + 1];
    for (std::uint32_t v = 0; v < n; ++v) csr.offsets[v + 1] += csr.offsets[v];

    csr.slot_source.resize(m);
    csr.slot_target.resize(m);
    csr.slot_edge.resize(m);
    // Stable counting sort by source: slots of a vertex keep insertion order.
    std::vector<std::uint32_t> cursor(csr.offsets.begin(), csr.offsets.end() - 1);
    for (std::uint32_t e = 0; e < m; ++e) {
        const Slot slot = cursor[topo.edge_source[e]]++;
        csr.slot_source[slot] = topo.edge_source[e];
        csr.slot_target[slot] = topo.edge_target[e];
        csr.slot_edge[slot] = e;
    }
}

// Kahn's algorithm. On success fills csr.topo_order / csr.topo_rank; on a
// cyclic graph leaves them empty and stores one witness cycle.
void compute_topological_order(Topology& topo) {
    Csr& csr = topo.csr;
    const std::uint32_t n = csr.num_vertices;
    std::vector<std::uint32_t> indegree(n, 0);
    for (VertexId target : csr.slot_target) ++indegree[target];

    std::vector<VertexId> order;
    order.reserve(n);
    for (VertexId v = 0; v < n; ++v) {
        if (indegree[v] == 0) order.push_back(v);
    }
    for (std::size_t head = 0; head < order.size(); ++head) {
        const VertexId u = order[head];
        for (Slot slot = csr.offsets[u]; slot < csr.offsets[u + 1]; ++slot) {
            const VertexId v = csr.slot_target[slot];
            if (--indegree[v] == 0) order.push_back(v);
        }
    }

    if (order.size() == n) {
        topo.acyclic = true;
        csr.topo_rank.resize(n);
        for (std::uint32_t rank = 0; rank < n; ++rank) csr.topo_rank[order[rank]] = rank;
        csr.topo_order = std::move(order);
        return;
    }

    // Every vertex Kahn could not remove still has an in-edge from another
    // such vertex, so walking those in-edges backwards must close a cycle.
    topo.acyclic = false;
    std::vector<VertexId> predecessor(n, kInvalid);
    VertexId start = kInvalid;
    for (Slot slot = 0; slot < csr.num_slots(); ++slot) {
        const VertexId u = csr.slot_source[slot];
        const VertexId v = csr.slot_target[slot];
        if (indegree[u] > 0 && indegree[v] > 0) {
            predecessor[v] = u;
            if (start == kInvalid) start = v;
        }
    }
    std::vector<std::uint32_t> step(n, kInvalid);
    std::vector<VertexId> walk;
    VertexId v = start;
    while (step[v] == kInvalid) {
        step[v] = static_cast<std::uint32_t>(walk.size());
        walk.push_back(v);
        v = predecessor[v];
    }
    topo.cycle.assign(walk.begin() + step[v], walk.end());
    std::reverse(topo.cycle.begin(), topo.cycle.end());
}

std::shared_ptr<const ResourceTable> build_resource_table(
    const Topology& topo, const std::vector<std::string>& column_names,
    const std::vector<std::vector<double>>& columns, std::span<const ResourceSpec> declared) {
    auto table = std::make_shared<ResourceTable>();
    table->names = column_names;
    table->edge_delta = columns;
    const std::uint32_t m = topo.num_edges();

    std::unordered_map<std::string, std::size_t> column_of;
    for (std::size_t c = 0; c < column_names.size(); ++c) column_of.emplace(column_names[c], c);

    std::vector<const ResourceSpec*> spec_of(column_names.size(), nullptr);
    for (const ResourceSpec& spec : declared) {
        const std::string label = "resource '" + spec.name + "'";
        if (std::isnan(spec.lower) || std::isnan(spec.upper) || !std::isfinite(spec.initial)) {
            throw std::invalid_argument(label + ": bounds must not be NaN and the initial value "
                                                "must be finite");
        }
        if (!(spec.lower <= spec.initial && spec.initial <= spec.upper)) {
            throw std::invalid_argument(label + ": initial value " + std::to_string(spec.initial) +
                                        " must lie within [lower, upper] = [" +
                                        std::to_string(spec.lower) + ", " +
                                        std::to_string(spec.upper) + "]");
        }
        auto [it, inserted] = column_of.emplace(spec.name, table->names.size());
        if (inserted) {
            // Declared but never used on an edge: an all-zero column.
            table->names.push_back(spec.name);
            table->edge_delta.emplace_back(m, 0.0);
            spec_of.push_back(&spec);
        } else {
            if (spec_of[it->second] != nullptr) {
                throw std::invalid_argument(label + " is declared more than once");
            }
            spec_of[it->second] = &spec;
        }
    }

    const std::size_t num_columns = table->names.size();
    table->lower.assign(num_columns, -kInf);
    table->upper.assign(num_columns, kInf);
    table->initial.assign(num_columns, 0.0);
    table->declared.assign(num_columns, 0);
    for (std::size_t c = 0; c < num_columns; ++c) {
        const ResourceSpec* spec = spec_of[c];
        if (spec == nullptr) continue;
        // Cumulative resources are monotone: the dominance rule and the
        // "lower bound never binds" argument both require delta >= 0.
        const std::vector<double>& delta = table->edge_delta[c];
        for (std::uint32_t e = 0; e < m; ++e) {
            if (delta[e] < 0.0) {
                throw std::invalid_argument(
                    "resource '" + spec->name + "': edge " + std::to_string(e) +
                    " has a negative increment (" + std::to_string(delta[e]) +
                    "); cumulative resources require nonnegative increments. "
                    "Replenishing or otherwise nonmonotone resources are not supported.");
            }
        }
        table->lower[c] = spec->lower;
        table->upper[c] = spec->upper;
        table->initial[c] = spec->initial;
        table->declared[c] = 1;
        if (std::isfinite(spec->upper)) table->tracked.push_back(static_cast<std::uint32_t>(c));
    }

    RcspData& data = table->tracked_data;
    const std::size_t k = table->tracked.size();
    data.num_resources = k;
    data.slot_delta.resize(k * static_cast<std::size_t>(m));
    for (std::size_t j = 0; j < k; ++j) {
        const std::uint32_t c = table->tracked[j];
        data.lower.push_back(table->lower[c]);
        data.upper.push_back(table->upper[c]);
        data.initial.push_back(table->initial[c]);
        const std::vector<double>& delta = table->edge_delta[c];
        for (Slot slot = 0; slot < m; ++slot) {
            data.slot_delta[static_cast<std::size_t>(slot) * k + j] = delta[topo.csr.slot_edge[slot]];
        }
    }
    return table;
}

}  // namespace

// ---------------------------------------------------------------- GraphBuilder

GraphBuilder::GraphBuilder(std::uint64_t num_vertices) {
    if (num_vertices == 0) throw std::invalid_argument("a graph needs at least one vertex");
    if (num_vertices >= kInvalid) {
        throw std::invalid_argument("graphs are limited to " + std::to_string(kInvalid - 1) +
                                    " vertices, got " + std::to_string(num_vertices));
    }
    num_vertices_ = static_cast<std::uint32_t>(num_vertices);
}

void GraphBuilder::check_capacity(std::size_t additional) const {
    if (source_.size() + additional >= kInvalid) {
        throw std::invalid_argument("graphs are limited to " + std::to_string(kInvalid - 1) +
                                    " edges");
    }
}

std::size_t GraphBuilder::resource_column(const std::string& name) {
    const auto it = std::find(resource_names_.begin(), resource_names_.end(), name);
    if (it != resource_names_.end()) return static_cast<std::size_t>(it - resource_names_.begin());
    resource_names_.push_back(name);
    resource_columns_.emplace_back(source_.size(), 0.0);
    return resource_names_.size() - 1;
}

EdgeId GraphBuilder::add_edge(std::int64_t source, std::int64_t target, double cost,
                              std::span<const std::pair<std::string, double>> resources) {
    const std::string context = "edge " + std::to_string(source_.size());
    check_capacity(1);
    check_vertex(source, num_vertices_, "source", context);
    check_vertex(target, num_vertices_, "target", context);
    check_finite(cost, "cost", context);
    for (const auto& [name, value] : resources) {
        check_finite(value, "resource '" + name + "'", context);
    }

    for (const auto& [name, value] : resources) resource_column(name);
    const EdgeId id = num_edges();
    source_.push_back(static_cast<VertexId>(source));
    target_.push_back(static_cast<VertexId>(target));
    cost_.push_back(cost);
    for (auto& column : resource_columns_) column.push_back(0.0);
    for (const auto& [name, value] : resources) resource_columns_[resource_column(name)][id] = value;
    return id;
}

EdgeId GraphBuilder::add_edges(std::span<const std::int64_t> source,
                               std::span<const std::int64_t> target, std::span<const double> cost,
                               std::span<const std::string> resource_names,
                               std::span<const std::span<const double>> resource_values) {
    const std::size_t count = source.size();
    const std::size_t first = source_.size();
    if (target.size() != count || cost.size() != count) {
        throw std::invalid_argument("source, target and cost must have the same length");
    }
    if (resource_names.size() != resource_values.size()) {
        throw std::invalid_argument("resource names and value arrays do not match");
    }
    check_capacity(count);
    for (std::size_t i = 0; i < count; ++i) {
        if (source[i] < 0 || source[i] >= static_cast<std::int64_t>(num_vertices_)) {
            check_vertex(source[i], num_vertices_, "source", edge_context(first, i));
        }
        if (target[i] < 0 || target[i] >= static_cast<std::int64_t>(num_vertices_)) {
            check_vertex(target[i], num_vertices_, "target", edge_context(first, i));
        }
        if (!std::isfinite(cost[i])) check_finite(cost[i], "cost", edge_context(first, i));
    }
    for (std::size_t r = 0; r < resource_names.size(); ++r) {
        if (resource_values[r].size() != count) {
            throw std::invalid_argument("resource '" + resource_names[r] + "' has " +
                                        std::to_string(resource_values[r].size()) +
                                        " values but " + std::to_string(count) +
                                        " edges are being added");
        }
        for (std::size_t i = 0; i < count; ++i) {
            if (!std::isfinite(resource_values[r][i])) {
                check_finite(resource_values[r][i], "resource '" + resource_names[r] + "'",
                             edge_context(first, i));
            }
        }
        for (std::size_t q = 0; q < r; ++q) {
            if (resource_names[q] == resource_names[r]) {
                throw std::invalid_argument("resource '" + resource_names[r] +
                                            "' is given more than once");
            }
        }
    }

    for (const std::string& name : resource_names) resource_column(name);
    source_.reserve(first + count);
    target_.reserve(first + count);
    for (std::size_t i = 0; i < count; ++i) {
        source_.push_back(static_cast<VertexId>(source[i]));
        target_.push_back(static_cast<VertexId>(target[i]));
    }
    cost_.insert(cost_.end(), cost.begin(), cost.end());
    for (auto& column : resource_columns_) column.resize(first + count, 0.0);
    for (std::size_t r = 0; r < resource_names.size(); ++r) {
        std::vector<double>& column = resource_columns_[resource_column(resource_names[r])];
        std::copy(resource_values[r].begin(), resource_values[r].end(), column.begin() + first);
    }
    return static_cast<EdgeId>(first);
}

std::shared_ptr<const CompiledGraph> GraphBuilder::compile(
    std::span<const ResourceSpec> declared) const {
    auto topo = std::make_shared<Topology>();
    topo->num_vertices = num_vertices_;
    topo->edge_source = source_;
    topo->edge_target = target_;
    build_csr(*topo);
    compute_topological_order(*topo);
    auto table = build_resource_table(*topo, resource_names_, resource_columns_, declared);
    return std::make_shared<const CompiledGraph>(std::move(topo), std::move(table), cost_);
}

// --------------------------------------------------------------- CompiledGraph

struct SolveWorkspace {
    DagWorkspace dag;
    RcspWorkspace rcsp;
};

CompiledGraph::CompiledGraph(std::shared_ptr<const Topology> topology,
                             std::shared_ptr<const ResourceTable> resources,
                             std::vector<double> edge_cost)
    : topology_(std::move(topology)),
      resources_(std::move(resources)),
      edge_cost_(std::move(edge_cost)) {
    const Csr& csr = topology_->csr;
    slot_cost_.resize(csr.num_slots());
    for (Slot slot = 0; slot < csr.num_slots(); ++slot) {
        slot_cost_[slot] = edge_cost_[csr.slot_edge[slot]];
    }
}

CompiledGraph::~CompiledGraph() = default;

std::shared_ptr<const CompiledGraph> CompiledGraph::with_costs(
    std::span<const double> edge_cost) const {
    if (edge_cost.size() != topology_->num_edges()) {
        throw std::invalid_argument("expected " + std::to_string(topology_->num_edges()) +
                                    " edge costs, got " + std::to_string(edge_cost.size()));
    }
    for (std::size_t e = 0; e < edge_cost.size(); ++e) {
        if (!std::isfinite(edge_cost[e])) {
            check_finite(edge_cost[e], "cost", "edge " + std::to_string(e));
        }
    }
    return std::make_shared<const CompiledGraph>(
        topology_, resources_, std::vector<double>(edge_cost.begin(), edge_cost.end()));
}

std::vector<double> CompiledGraph::path_resources(std::span<const EdgeId> edges) const {
    const ResourceTable& table = *resources_;
    std::vector<double> totals(table.initial);
    for (std::size_t c = 0; c < totals.size(); ++c) {
        const std::vector<double>& delta = table.edge_delta[c];
        double value = totals[c];
        for (EdgeId e : edges) value += delta[e];
        totals[c] = value;
    }
    return totals;
}

Solution CompiledGraph::solve(VertexId source, VertexId target, Algorithm algorithm,
                              const RcspOptions& options) const {
    const Topology& topo = *topology_;
    const Csr& csr = topo.csr;
    if (!topo.acyclic) {
        throw std::invalid_argument("the graph contains a directed cycle; only acyclic graphs "
                                    "can be solved");
    }
    if (source >= topo.num_vertices || target >= topo.num_vertices) {
        throw std::invalid_argument("source or target vertex id is out of range");
    }

    std::unique_ptr<SolveWorkspace> workspace;
    {
        std::lock_guard<std::mutex> lock(workspace_mutex_);
        if (!workspaces_.empty()) {
            workspace = std::move(workspaces_.back());
            workspaces_.pop_back();
        }
    }
    if (!workspace) workspace = std::make_unique<SolveWorkspace>();
    // Return the workspace to the pool even if the solve throws.
    struct Return {
        const CompiledGraph* graph;
        std::unique_ptr<SolveWorkspace>* workspace;
        ~Return() {
            std::lock_guard<std::mutex> lock(graph->workspace_mutex_);
            graph->workspaces_.push_back(std::move(*workspace));
        }
    } give_back{this, &workspace};

    const auto start = std::chrono::steady_clock::now();
    Solution solution;
    PathResult path;
    if (algorithm == Algorithm::Dag) {
        path = solve_dag(csr, slot_cost_.data(), source, target, workspace->dag);
    } else {
        resource_to_go(csr, resources_->tracked_data, source, target, workspace->rcsp.to_go);
        path = solve_rcsp(csr, resources_->tracked_data, slot_cost_.data(), source, target,
                          options, workspace->rcsp, solution.stats,
                          workspace->rcsp.to_go.data());
        if (path.status != Status::Optimal && !is_reachable(csr, source, target)) {
            path.status = Status::Unreachable;
        }
    }

    solution.status = path.status;
    if (path.status == Status::Optimal) {
        solution.cost = path.cost;
        solution.edges.reserve(path.slots.size());
        for (Slot slot : path.slots) solution.edges.push_back(csr.slot_edge[slot]);
        solution.resources = path_resources(solution.edges);
    }
    solution.stats.solve_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return solution;
}

bool is_reachable(const Csr& csr, VertexId source, VertexId target) {
    if (source == target) return true;
    const std::uint32_t target_rank = csr.topo_rank[target];
    if (csr.topo_rank[source] > target_rank) return false;
    std::vector<std::uint8_t> seen(csr.num_vertices, 0);
    std::vector<VertexId> stack{source};
    seen[source] = 1;
    while (!stack.empty()) {
        const VertexId u = stack.back();
        stack.pop_back();
        for (Slot slot = csr.offsets[u]; slot < csr.offsets[u + 1]; ++slot) {
            const VertexId v = csr.slot_target[slot];
            if (v == target) return true;
            if (!seen[v] && csr.topo_rank[v] < target_rank) {
                seen[v] = 1;
                stack.push_back(v);
            }
        }
    }
    return false;
}

}  // namespace celerispath
