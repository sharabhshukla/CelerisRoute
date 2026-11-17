#include "celerispath/pricing_engine.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <system_error>
#include <thread>

namespace celerispath {

namespace {

using Clock = std::chrono::steady_clock;

double seconds_between(Clock::time_point from, Clock::time_point to) {
    return std::chrono::duration<double>(to - from).count();
}

}  // namespace

PricingEngine::PricingEngine(std::shared_ptr<const CompiledGraph> graph, VertexId source,
                             VertexId target, Algorithm algorithm,
                             const SparseColumns& coefficients, double fixed_cost,
                             std::span<const std::int64_t> fixed_index,
                             std::span<const double> fixed_value, const RcspOptions& options)
    : graph_(std::move(graph)),
      algorithm_(algorithm),
      options_(options),
      source_(source),
      target_(target),
      num_constraints_(coefficients.num_rows),
      fixed_cost_(fixed_cost) {
    const Topology& topo = graph_->topology();
    if (!topo.acyclic) {
        throw std::invalid_argument("the graph contains a directed cycle; pricing requires an "
                                    "acyclic graph");
    }
    if (source >= topo.num_vertices || target >= topo.num_vertices) {
        throw std::invalid_argument("source or target vertex id is out of range");
    }
    if (!std::isfinite(fixed_cost)) throw std::invalid_argument("fixed_cost must be finite");
    if (num_constraints_ >= kInvalid) {
        throw std::invalid_argument("too many master constraints");
    }
    if (fixed_index.size() != fixed_value.size()) {
        throw std::invalid_argument("fixed coefficient indices and values differ in length");
    }
    for (std::size_t i = 0; i < fixed_index.size(); ++i) {
        if (fixed_index[i] < 0 || fixed_index[i] >= static_cast<std::int64_t>(num_constraints_)) {
            throw std::invalid_argument("fixed coefficient row " + std::to_string(fixed_index[i]) +
                                        " is outside [0, " + std::to_string(num_constraints_) +
                                        ")");
        }
        if (!std::isfinite(fixed_value[i])) {
            throw std::invalid_argument("fixed coefficients must be finite");
        }
        fixed_row_.push_back(static_cast<std::uint32_t>(fixed_index[i]));
        fixed_value_.push_back(fixed_value[i]);
    }

    build_subgraph();
    load_coefficients(coefficients);

    reduced_cost_.resize(csr_.num_slots());
    accumulator_.assign(num_constraints_, 0.0);
    touched_flag_.assign(num_constraints_, 0);
}

void PricingEngine::build_subgraph() {
    const Csr& full = graph_->topology().csr;
    const RcspData& full_data = graph_->resources().tracked_data;
    const std::size_t num_resources = full_data.num_resources;
    rcsp_data_.num_resources = num_resources;
    rcsp_data_.lower = full_data.lower;
    rcsp_data_.upper = full_data.upper;
    rcsp_data_.initial = full_data.initial;

    const std::uint32_t n = full.num_vertices;
    const std::uint32_t source_rank = full.topo_rank[source_];
    const std::uint32_t target_rank = full.topo_rank[target_];

    // keep[v]: v lies on some source-target path. Only vertices ranked
    // between the two endpoints can qualify, so both sweeps stay in that range.
    std::vector<std::uint8_t> forward(n, 0);
    std::vector<std::uint8_t> keep(n, 0);
    if (source_rank <= target_rank) {
        forward[source_] = 1;
        for (std::uint32_t rank = source_rank; rank < target_rank; ++rank) {
            const VertexId u = full.topo_order[rank];
            if (!forward[u]) continue;
            for (Slot slot = full.offsets[u]; slot < full.offsets[u + 1]; ++slot) {
                forward[full.slot_target[slot]] = 1;
            }
        }
        keep[target_] = forward[target_];
        for (std::uint32_t rank = target_rank; rank-- > source_rank;) {
            const VertexId u = full.topo_order[rank];
            if (!forward[u]) continue;
            for (Slot slot = full.offsets[u]; slot < full.offsets[u + 1]; ++slot) {
                if (keep[full.slot_target[slot]]) {
                    keep[u] = 1;
                    break;
                }
            }
        }
    }
    reachable_ = keep[target_] != 0;

    csr_.offsets.assign(1, 0);
    if (!reachable_) {
        coef_ptr_.assign(1, 0);
        return;
    }

    // Number kept vertices by topological rank, so in the subgraph the vertex
    // id is the rank and a linear scan over ids is a topological sweep.
    std::vector<VertexId> new_id(n, kInvalid);
    std::vector<VertexId> kept;
    for (std::uint32_t rank = source_rank; rank <= target_rank; ++rank) {
        const VertexId v = full.topo_order[rank];
        if (keep[v]) {
            new_id[v] = static_cast<VertexId>(kept.size());
            kept.push_back(v);
        }
    }

    const std::span<const double> full_cost = graph_->slot_cost();
    csr_.num_vertices = static_cast<std::uint32_t>(kept.size());
    for (const VertexId u : kept) {
        for (Slot slot = full.offsets[u]; slot < full.offsets[u + 1]; ++slot) {
            const VertexId v = full.slot_target[slot];
            if (!keep[v]) continue;
            csr_.slot_source.push_back(new_id[u]);
            csr_.slot_target.push_back(new_id[v]);
            csr_.slot_edge.push_back(full.slot_edge[slot]);
            base_cost_.push_back(full_cost[slot]);
            const double* delta = full_data.slot_delta.data() +
                                  static_cast<std::size_t>(slot) * num_resources;
            rcsp_data_.slot_delta.insert(rcsp_data_.slot_delta.end(), delta,
                                         delta + num_resources);
        }
        csr_.offsets.push_back(csr_.num_slots());
    }
    csr_.topo_order.resize(csr_.num_vertices);
    std::iota(csr_.topo_order.begin(), csr_.topo_order.end(), VertexId{0});
    csr_.topo_rank = csr_.topo_order;
    sub_source_ = new_id[source_];
    sub_target_ = new_id[target_];
    resource_to_go(csr_, rcsp_data_, sub_source_, sub_target_, to_go_);
}

void PricingEngine::load_coefficients(const SparseColumns& a) {
    const std::size_t num_edges = graph_->topology().num_edges();
    if (a.indptr.size() != num_edges + 1) {
        throw std::invalid_argument(
            "dual_matrix must have one column per edge: expected " + std::to_string(num_edges) +
            " columns, got " + std::to_string(a.indptr.empty() ? 0 : a.indptr.size() - 1));
    }
    if (a.indices.size() != a.data.size() || a.indptr.front() != 0 ||
        a.indptr.back() != static_cast<std::int64_t>(a.data.size())) {
        throw std::invalid_argument("dual_matrix has inconsistent sparse index arrays");
    }
    for (std::size_t e = 0; e < num_edges; ++e) {
        if (a.indptr[e] > a.indptr[e + 1]) {
            throw std::invalid_argument("dual_matrix has inconsistent sparse index arrays");
        }
    }
    for (std::size_t j = 0; j < a.data.size(); ++j) {
        if (a.indices[j] < 0 || a.indices[j] >= static_cast<std::int64_t>(a.num_rows)) {
            throw std::invalid_argument("dual_matrix has a row index outside its shape");
        }
        if (!std::isfinite(a.data[j])) {
            throw std::invalid_argument("dual_matrix contains a nonfinite coefficient");
        }
    }

    // Keep only the columns of edges in the subgraph, laid out in slot order
    // so the reduced-cost refresh reads them front to back.
    coef_ptr_.assign(1, 0);
    coef_ptr_.reserve(static_cast<std::size_t>(csr_.num_slots()) + 1);
    for (const EdgeId e : csr_.slot_edge) {
        for (std::int64_t j = a.indptr[e]; j < a.indptr[e + 1]; ++j) {
            coef_row_.push_back(static_cast<std::uint32_t>(a.indices[j]));
            coef_value_.push_back(a.data[j]);
        }
        coef_ptr_.push_back(coef_row_.size());
    }
}

PricingColumn PricingEngine::make_column(const PathResult& path, double fixed_reduced) {
    PricingColumn column;
    column.edges.reserve(path.slots.size());
    double original = fixed_cost_;
    for (const Slot slot : path.slots) {
        column.edges.push_back(csr_.slot_edge[slot]);
        original += base_cost_[slot];
        for (std::size_t j = coef_ptr_[slot]; j < coef_ptr_[slot + 1]; ++j) {
            const std::uint32_t row = coef_row_[j];
            if (!touched_flag_[row]) {
                touched_flag_[row] = 1;
                touched_.push_back(row);
            }
            accumulator_[row] += coef_value_[j];
        }
    }
    for (std::size_t i = 0; i < fixed_row_.size(); ++i) {
        const std::uint32_t row = fixed_row_[i];
        if (!touched_flag_[row]) {
            touched_flag_[row] = 1;
            touched_.push_back(row);
        }
        accumulator_[row] += fixed_value_[i];
    }
    std::sort(touched_.begin(), touched_.end());
    for (const std::uint32_t row : touched_) {
        if (accumulator_[row] != 0.0) {
            column.coef_index.push_back(row);
            column.coef_value.push_back(accumulator_[row]);
        }
        accumulator_[row] = 0.0;
        touched_flag_[row] = 0;
    }
    touched_.clear();

    column.original_cost = original;
    // path.cost is the solver's own sum of reduced edge costs, so the ordering
    // of returned columns matches the ordering the solver established.
    column.reduced_cost = fixed_reduced + path.cost;
    column.resources = graph_->path_resources(column.edges);
    return column;
}

PricingResult PricingEngine::solve(std::span<const double> duals, std::uint32_t max_paths,
                                   bool negative_only, double tolerance) {
    if (duals.size() != num_constraints_) {
        throw std::invalid_argument("expected " + std::to_string(num_constraints_) +
                                    " dual values (one per master constraint), got " +
                                    std::to_string(duals.size()));
    }
    for (std::size_t k = 0; k < duals.size(); ++k) {
        if (!std::isfinite(duals[k])) {
            throw std::invalid_argument("dual value " + std::to_string(k) + " is not finite");
        }
    }
    if (max_paths == 0) throw std::invalid_argument("max_paths must be at least 1");
    if (algorithm_ == Algorithm::Rcsp && max_paths > 1) {
        throw std::invalid_argument("max_paths > 1 is not supported by the rcsp backend");
    }
    if (!(tolerance >= 0.0)) throw std::invalid_argument("tolerance must be nonnegative");

    const std::lock_guard<std::mutex> lock(mutex_);
    PricingResult result;
    if (!reachable_) return result;  // Status::Unreachable

    const auto t0 = Clock::now();
    const double* const pi = duals.data();
    const std::uint32_t num_slots = csr_.num_slots();
    for (Slot slot = 0; slot < num_slots; ++slot) {
        double value = base_cost_[slot];
        for (std::size_t j = coef_ptr_[slot]; j < coef_ptr_[slot + 1]; ++j) {
            value -= coef_value_[j] * pi[coef_row_[j]];
        }
        reduced_cost_[slot] = value;
    }
    double fixed_reduced = fixed_cost_;
    for (std::size_t i = 0; i < fixed_row_.size(); ++i) {
        fixed_reduced -= fixed_value_[i] * pi[fixed_row_[i]];
    }

    const auto t1 = Clock::now();
    std::vector<PathResult> paths;
    if (algorithm_ == Algorithm::Rcsp) {
        PathResult path = solve_rcsp(csr_, rcsp_data_, reduced_cost_.data(), sub_source_,
                                     sub_target_, options_, rcsp_workspace_, result.stats.labels,
                                     to_go_.data());
        if (path.status == Status::Optimal) paths.push_back(std::move(path));
    } else if (max_paths == 1) {
        PathResult path =
            solve_dag(csr_, reduced_cost_.data(), sub_source_, sub_target_, dag_workspace_);
        if (path.status == Status::Optimal) paths.push_back(std::move(path));
    } else {
        paths = solve_dag_k_best(csr_, reduced_cost_.data(), sub_source_, sub_target_, max_paths,
                                 k_best_workspace_);
    }

    const auto t2 = Clock::now();
    // The target is reachable, so an empty result can only mean that the
    // resource bounds exclude every path.
    result.status = paths.empty() ? Status::Infeasible : Status::Optimal;
    if (!paths.empty()) result.best_reduced_cost = fixed_reduced + paths.front().cost;
    for (const PathResult& path : paths) {
        if (negative_only && !(fixed_reduced + path.cost < -tolerance)) break;
        result.columns.push_back(make_column(path, fixed_reduced));
    }

    const auto t3 = Clock::now();
    result.stats.update_seconds = seconds_between(t0, t1);
    result.stats.solve_seconds = seconds_between(t1, t2);
    result.stats.reconstruct_seconds = seconds_between(t2, t3);
    result.stats.labels.solve_seconds = result.stats.solve_seconds;
    return result;
}

BatchOutcome solve_batch(std::span<PricingEngine* const> engines,
                         std::span<const std::span<const double>> duals, std::uint32_t max_paths,
                         bool negative_only, double tolerance, unsigned threads) {
    const std::size_t count = engines.size();
    if (duals.size() != 1 && duals.size() != count) {
        throw std::invalid_argument("expected one dual vector, or one per engine");
    }
    BatchOutcome outcome;
    outcome.results.resize(count);
    outcome.errors.resize(count);
    if (count == 0) return outcome;

    if (threads == 0) threads = std::max(1u, std::thread::hardware_concurrency());
    threads = static_cast<unsigned>(std::min<std::size_t>(threads, count));

    // Each worker claims the next unsolved engine. Results are written to the
    // engine's own slot, so the outcome does not depend on scheduling.
    std::atomic<std::size_t> next{0};
    const auto work = [&]() {
        for (;;) {
            const std::size_t i = next.fetch_add(1, std::memory_order_relaxed);
            if (i >= count) return;
            try {
                const std::span<const double> pi = duals.size() == 1 ? duals[0] : duals[i];
                outcome.results[i] = engines[i]->solve(pi, max_paths, negative_only, tolerance);
            } catch (const std::exception& error) {
                outcome.errors[i] = error.what()[0] != '\0' ? error.what() : "unknown error";
            } catch (...) {
                outcome.errors[i] = "unknown error";
            }
        }
    };

    std::vector<std::thread> pool;
    pool.reserve(threads - 1);
    for (unsigned i = 1; i < threads; ++i) {
        try {
            pool.emplace_back(work);
        } catch (const std::system_error&) {
            break;  // run with the threads that did start
        }
    }
    work();
    for (std::thread& thread : pool) thread.join();
    return outcome;
}

}  // namespace celerispath
