#include "celerisroute/pricing_fleet.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <exception>
#include <functional>
#include <limits>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <unordered_map>
#include <utility>

#if CELERISROUTE_HAS_TBB
#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/partitioner.h>
#include <oneapi/tbb/task_arena.h>
#include <oneapi/tbb/task_group.h>
#include <oneapi/tbb/version.h>
#endif

namespace celerisroute {

namespace {

using Clock = std::chrono::steady_clock;

double seconds_between(Clock::time_point from, Clock::time_point to) {
    return std::chrono::duration<double>(to - from).count();
}

// Below this many slots a dual update is cheaper than waking the workers.
constexpr std::uint32_t kParallelUpdateSlots = std::uint32_t{1} << 17;
// Slots per task of a parallel dual update.
constexpr std::uint32_t kUpdateGrain = std::uint32_t{1} << 15;
// Time limits beyond this many seconds are treated as no limit.
constexpr double kMaxTimeLimit = 1e9;

void check_rows(std::span<const std::int64_t> ptr, std::size_t count, std::size_t entries,
                const char* what) {
    const std::string message = std::string(what) + " have inconsistent row pointers";
    if (ptr.size() != count + 1 || ptr.front() != 0 ||
        ptr.back() != static_cast<std::int64_t>(entries)) {
        throw std::invalid_argument(message);
    }
    for (std::size_t i = 0; i < count; ++i) {
        if (ptr[i] > ptr[i + 1]) throw std::invalid_argument(message);
    }
}

template <class T>
std::size_t bytes_of(const std::vector<T>& values) {
    return values.capacity() * sizeof(T);
}

std::string describe(const std::exception& error) {
    return error.what()[0] != '\0' ? error.what() : "unknown error";
}

// Reachability of `target` from `source` when the slots flagged in `removed`
// (if any) do not exist. `seen` and `stack` are scratch.
bool reachable_without(const Csr& csr, VertexId source, VertexId target,
                       const std::uint8_t* removed, std::vector<std::uint8_t>& seen,
                       std::vector<VertexId>& stack) {
    if (source == target) return true;
    const std::uint32_t target_rank = csr.topo_rank[target];
    if (csr.topo_rank[source] > target_rank) return false;
    seen.assign(csr.num_vertices, 0);
    stack.assign(1, source);
    seen[source] = 1;
    while (!stack.empty()) {
        const VertexId u = stack.back();
        stack.pop_back();
        for (Slot slot = csr.offsets[u]; slot < csr.offsets[u + 1]; ++slot) {
            if (removed != nullptr && removed[slot]) continue;
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

// One overlay entry while a vessel is being registered. row == kInvalid marks
// a cost offset, anything else a master coefficient.
struct OverlayEntry {
    Slot slot;
    std::uint32_t row;
    double value;
};

}  // namespace

bool tbb_available() {
#if CELERISROUTE_HAS_TBB
    return true;
#else
    return false;
#endif
}

std::string tbb_runtime_version() {
#if CELERISROUTE_HAS_TBB
    return TBB_runtime_version();
#else
    return "";
#endif
}

// A vessel is a handful of indices into the shared and per-fleet arrays.
struct PricingFleet::Vessel {
    VertexId source = 0;
    VertexId target = 0;
    double fixed_cost = 0.0;
    std::uint32_t fixed_begin = 0;
    std::uint32_t fixed_end = 0;
    std::uint32_t overlay_begin = 0;
    std::uint32_t overlay_end = 0;
    std::uint32_t bound = kInvalid;  // index into bounds_; kInvalid: compute per solve
    bool removes_edges = false;      // some overlay offset is +infinity
    bool reachable = false;          // target reachable in the vessel's own network
};

// Completion bounds towards one target, shared by every vessel that ends there.
struct PricingFleet::Bound {
    VertexId target = 0;
    std::uint32_t first_rank = 0;  // valid for vertices ranked from here up to the target
    std::vector<double> to_go;
};

// Scratch memory of one worker slot. A slot is occupied by one thread at a
// time, so a workspace is never shared while a solve runs.
struct alignas(64) PricingFleet::Workspace {
    DagWorkspace dag;
    KBestWorkspace k_best;
    RcspWorkspace rcsp;
    // Copy of the shared reduced costs that vessel overlays are written into
    // and removed from again; valid while `epoch` matches the fleet's.
    std::vector<double> cost;
    std::uint64_t epoch = 0;
    std::vector<double> accumulator;  // dense scratch over master rows
    std::vector<std::uint8_t> touched_flag;
    std::vector<std::uint32_t> touched;

    void add(std::uint32_t row, double value) {
        if (!touched_flag[row]) {
            touched.push_back(row);
            touched_flag[row] = 1;
        }
        accumulator[row] += value;
    }

    void clear_accumulator() noexcept {
        for (const std::uint32_t row : touched) {
            accumulator[row] = 0.0;
            touched_flag[row] = 0;
        }
        touched.clear();
    }

    std::size_t bytes() const {
        return bytes_of(dag.distance) + bytes_of(dag.color) + bytes_of(dag.pred_slot) +
               bytes_of(k_best.distance) + bytes_of(k_best.pred_slot) +
               bytes_of(k_best.pred_rank) + bytes_of(k_best.count) + bytes_of(rcsp.to_go) +
               bytes_of(rcsp.limit) + rcsp.label_pool.bytes_reserved() + bytes_of(cost) +
               bytes_of(accumulator) + bytes_of(touched_flag) + bytes_of(touched);
    }
};

// Result slot of one vessel. Slots are written by whichever worker prices the
// vessel and read only after the parallel phase, so the mapping from vessel
// to result does not depend on scheduling. Buffers keep their capacity
// between solves.
struct alignas(64) PricingFleet::Outcome {
    Status status = Status::Interrupted;
    double best = kInf;
    double seconds = 0.0;
    SolveStats stats;
    std::string error;
    std::vector<EdgeId> edges;
    std::vector<std::uint32_t> edge_end;  // per column
    std::vector<double> original;
    std::vector<double> reduced;
    std::vector<double> resources;
    std::vector<std::uint32_t> coef_row;
    std::vector<double> coef_value;
    std::vector<std::uint32_t> coef_end;  // per column

    void reset() noexcept {
        status = Status::Interrupted;
        best = kInf;
        seconds = 0.0;
        stats = SolveStats{};
        error.clear();
        edges.clear();
        edge_end.clear();
        original.clear();
        reduced.clear();
        resources.clear();
        coef_row.clear();
        coef_value.clear();
        coef_end.clear();
    }

    std::size_t num_columns() const { return reduced.size(); }

    std::size_t bytes() const {
        return bytes_of(edges) + bytes_of(edge_end) + bytes_of(original) + bytes_of(reduced) +
               bytes_of(resources) + bytes_of(coef_row) + bytes_of(coef_value) +
               bytes_of(coef_end);
    }
};

struct PricingFleet::SolveParams {
    std::uint32_t max_paths = 1;
    bool negative_only = true;
    double tolerance = 0.0;
    StopToken stop;
};

// Runs a body once per index on a fixed number of worker slots. A slot is the
// identity a body may use to pick thread-private scratch memory: no two
// bodies run on the same slot at the same time, provided that the body
// itself starts no nested parallel work.
class PricingFleet::Executor {
public:
    // body(slot, index) returns false to abandon the indices not yet started.
    using Body = std::function<bool(unsigned slot, std::size_t index)>;

    Executor(Scheduler scheduler, Partitioner partitioner, unsigned threads)
        : partitioner_(partitioner), slots_(threads) {
#if CELERISROUTE_HAS_TBB
        if (scheduler == Scheduler::Tbb && threads > 1) {
            // One slot is reserved for the calling thread, which takes part
            // in the work; the other threads - 1 are oneTBB workers that stay
            // alive between runs.
            arena_ = std::make_unique<tbb::task_arena>(static_cast<int>(threads));
            arena_->initialize();
            slots_ = static_cast<unsigned>(arena_->max_concurrency());
        }
#else
        (void)scheduler;
#endif
    }

    unsigned slots() const { return slots_; }

    // Whether workers outlive a run, which makes short runs worthwhile.
    bool persistent() const {
#if CELERISROUTE_HAS_TBB
        return arena_ != nullptr;
#else
        return false;
#endif
    }

    // Returns when every started body has finished. An exception that
    // escapes a body is rethrown here, on the calling thread.
    void run(std::size_t count, const Body& body) {
        if (count == 0) return;
        if (slots_ == 1 || count == 1) {
            for (std::size_t i = 0; i < count; ++i) {
                if (!body(0, i)) return;
            }
            return;
        }
#if CELERISROUTE_HAS_TBB
        if (arena_ != nullptr) {
            run_tbb(count, body);
            return;
        }
#endif
        run_threads(count, body);
    }

private:
#if CELERISROUTE_HAS_TBB
    void run_tbb(std::size_t count, const Body& body) {
        // Isolated, so cancelling this run never reaches a task group of the
        // caller (and the other way round).
        tbb::task_group_context context(tbb::task_group_context::isolated);
        const unsigned slots = slots_;
        const auto chunk = [&](const tbb::blocked_range<std::size_t>& range) {
            const int index = tbb::this_task_arena::current_thread_index();
            if (index < 0 || static_cast<unsigned>(index) >= slots) {
                throw std::logic_error("oneTBB reported a worker slot outside the arena");
            }
            for (std::size_t i = range.begin(); i != range.end(); ++i) {
                if (context.is_group_execution_cancelled()) return;
                if (!body(static_cast<unsigned>(index), i)) {
                    context.cancel_group_execution();
                    return;
                }
            }
        };
        // Grain size 1: a range may be split down to single indices, so an
        // idle worker can always steal the unstarted half of a busy one.
        const tbb::blocked_range<std::size_t> range(0, count, 1);
        arena_->execute([&] {
            switch (partitioner_) {
                case Partitioner::Simple:
                    tbb::parallel_for(range, chunk, tbb::simple_partitioner(), context);
                    break;
                case Partitioner::Static:
                    tbb::parallel_for(range, chunk, tbb::static_partitioner(), context);
                    break;
                case Partitioner::Auto:
                    tbb::parallel_for(range, chunk, tbb::auto_partitioner(), context);
                    break;
            }
        });
    }
#endif

    // Fallback without oneTBB: threads that live for one run and claim
    // indices from a shared counter.
    void run_threads(std::size_t count, const Body& body) {
        std::atomic<std::size_t> next{0};
        std::atomic<bool> stop{false};
        std::mutex failure_mutex;
        std::exception_ptr failure;
        const auto work = [&](unsigned slot) noexcept {
            try {
                while (!stop.load(std::memory_order_relaxed)) {
                    const std::size_t i = next.fetch_add(1, std::memory_order_relaxed);
                    if (i >= count) return;
                    if (!body(slot, i)) {
                        stop.store(true, std::memory_order_relaxed);
                        return;
                    }
                }
            } catch (...) {
                stop.store(true, std::memory_order_relaxed);
                const std::lock_guard<std::mutex> lock(failure_mutex);
                if (!failure) failure = std::current_exception();
            }
        };

        const unsigned workers = static_cast<unsigned>(std::min<std::size_t>(slots_, count));
        std::vector<std::thread> pool;
        pool.reserve(workers - 1);
        for (unsigned slot = 1; slot < workers; ++slot) {
            try {
                pool.emplace_back(work, slot);
            } catch (const std::system_error&) {
                break;  // run with the threads that did start
            }
        }
        work(0);
        for (std::thread& thread : pool) thread.join();
        if (failure) std::rethrow_exception(failure);
    }

    [[maybe_unused]] Partitioner partitioner_;  // only read by the oneTBB scheduler
    unsigned slots_;
#if CELERISROUTE_HAS_TBB
    std::unique_ptr<tbb::task_arena> arena_;
#endif
};

// ------------------------------------------------------------- construction

PricingFleet::PricingFleet(std::shared_ptr<const CompiledGraph> graph,
                           const DualCoefficients& coefficients, const FleetOptions& options)
    : graph_(std::move(graph)), options_(options), num_constraints_(coefficients.num_rows) {
    const Topology& topo = graph_->topology();
    if (!topo.acyclic) {
        throw std::invalid_argument("the graph contains a directed cycle; pricing requires an "
                                    "acyclic graph");
    }
    if (num_constraints_ >= kInvalid) {
        throw std::invalid_argument("too many master constraints");
    }
    if (!(options_.rcsp.dominance_tolerance >= 0.0)) {
        throw std::invalid_argument("dominance_tolerance must be nonnegative");
    }
    if (options_.scheduler == Scheduler::Tbb && !tbb_available()) {
        throw std::invalid_argument("this build of celerisroute has no oneTBB support; use the "
                                    "\"threads\" scheduler or rebuild with oneTBB installed");
    }

    const Csr& csr = topo.csr;
    edge_slot_.resize(topo.num_edges());
    for (Slot slot = 0; slot < csr.num_slots(); ++slot) edge_slot_[csr.slot_edge[slot]] = slot;
    load_coefficients(coefficients);
    overlay_coef_ptr_.assign(1, 0);
    duals_.assign(num_constraints_ + 1, 0.0);
}

PricingFleet::~PricingFleet() = default;

void PricingFleet::load_coefficients(const DualCoefficients& c) {
    const Csr& csr = graph_->topology().csr;
    const std::size_t num_edges = graph_->topology().num_edges();
    const std::int64_t num_rows = static_cast<std::int64_t>(num_constraints_);

    if (c.width > 0) {
        if (c.row.size() != c.value.size() || c.row.size() != num_edges * c.width) {
            throw std::invalid_argument("indexed dual coefficients need " +
                                        std::to_string(c.width) + " (row, value) pairs for "
                                        "each of the " + std::to_string(num_edges) + " edges");
        }
        width_ = c.width;
        coef_row_.resize(num_edges * width_);
        coef_value_.resize(num_edges * width_);
        for (Slot slot = 0; slot < csr.num_slots(); ++slot) {
            const std::size_t from = static_cast<std::size_t>(csr.slot_edge[slot]) * width_;
            const std::size_t to = static_cast<std::size_t>(slot) * width_;
            for (std::size_t j = 0; j < width_; ++j) {
                const std::int64_t row = c.row[from + j];
                if (row < 0) {
                    // Unused pair: reads the zero stored after the duals, so
                    // the update needs no branch.
                    coef_row_[to + j] = static_cast<std::uint32_t>(num_constraints_);
                    coef_value_[to + j] = 0.0;
                    continue;
                }
                if (row >= num_rows) {
                    throw std::invalid_argument("dual mapping has a row index outside [0, " +
                                                std::to_string(num_constraints_) + ")");
                }
                if (!std::isfinite(c.value[from + j])) {
                    throw std::invalid_argument("dual mapping contains a nonfinite coefficient");
                }
                coef_row_[to + j] = static_cast<std::uint32_t>(row);
                coef_value_[to + j] = c.value[from + j];
            }
        }
        return;
    }

    const SparseColumns& a = c.sparse;
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
    if (a.data.size() >= kInvalid) {
        throw std::invalid_argument("dual_matrix has too many nonzero entries");
    }
    for (std::size_t j = 0; j < a.data.size(); ++j) {
        if (a.indices[j] < 0 || a.indices[j] >= num_rows) {
            throw std::invalid_argument("dual_matrix has a row index outside its shape");
        }
        if (!std::isfinite(a.data[j])) {
            throw std::invalid_argument("dual_matrix contains a nonfinite coefficient");
        }
    }

    // Laid out in slot order, so the reduced-cost refresh reads the costs and
    // the coefficients front to back.
    coef_ptr_.reserve(static_cast<std::size_t>(csr.num_slots()) + 1);
    coef_ptr_.push_back(0);
    coef_row_.reserve(a.data.size());
    coef_value_.reserve(a.data.size());
    for (const EdgeId e : csr.slot_edge) {
        for (std::int64_t j = a.indptr[e]; j < a.indptr[e + 1]; ++j) {
            coef_row_.push_back(static_cast<std::uint32_t>(a.indices[j]));
            coef_value_.push_back(a.data[j]);
        }
        coef_ptr_.push_back(static_cast<std::uint32_t>(coef_row_.size()));
    }
}

// ------------------------------------------------------------------ vessels

std::size_t PricingFleet::add_vessels(const FleetVessels& v) {
    const Topology& topo = graph_->topology();
    const RcspData& data = graph_->resources().tracked_data;
    const std::size_t count = v.source.size();
    const std::size_t tracked = data.num_resources;
    const std::int64_t num_vertices = topo.num_vertices;
    const std::int64_t num_edges = topo.num_edges();
    const std::int64_t num_rows = static_cast<std::int64_t>(num_constraints_);
    const bool store_limits = options_.algorithm == Algorithm::Rcsp && tracked > 0;

    if (v.target.size() != count || v.fixed_cost.size() != count) {
        throw std::invalid_argument("source, target and fixed_cost differ in length");
    }
    if (!v.limits.empty()) {
        if (options_.algorithm != Algorithm::Rcsp) {
            throw std::invalid_argument("vessel resource limits require the rcsp algorithm");
        }
        if (v.limits.size() != count * tracked) {
            throw std::invalid_argument("resource limits need one value per vessel and tracked "
                                        "resource: expected " + std::to_string(count * tracked) +
                                        ", got " + std::to_string(v.limits.size()));
        }
    }
    if (v.fixed_row.size() != v.fixed_value.size() ||
        v.offset_edge.size() != v.offset_value.size() ||
        v.coef_edge.size() != v.coef_row.size() || v.coef_edge.size() != v.coef_value.size()) {
        throw std::invalid_argument("vessel index and value arrays differ in length");
    }
    check_rows(v.fixed_ptr, count, v.fixed_row.size(), "fixed coefficients");
    check_rows(v.offset_ptr, count, v.offset_edge.size(), "cost offsets");
    check_rows(v.coef_ptr, count, v.coef_edge.size(), "vessel edge coefficients");

    const std::lock_guard<std::mutex> lock(mutex_);
    const std::size_t first = vessels_.size();
    const std::size_t fixed_base = fixed_row_.size();
    const std::size_t overlay_base = overlay_slot_.size();
    const std::size_t coef_base = overlay_coef_row_.size();

    // Everything is validated and built on the side first, so a rejected
    // batch leaves the fleet exactly as it was.
    std::vector<Vessel> vessels;
    vessels.reserve(count);
    std::vector<double> limits;
    std::vector<std::uint32_t> fixed_row;
    std::vector<double> fixed_value;
    std::vector<Slot> overlay_slot;
    std::vector<double> overlay_offset;
    std::vector<std::size_t> overlay_coef_end;
    std::vector<std::uint32_t> overlay_coef_row;
    std::vector<double> overlay_coef_value;
    std::vector<OverlayEntry> entries;
    if (store_limits) limits.reserve(count * tracked);

    for (std::size_t i = 0; i < count; ++i) {
        const std::string who = "vessel " + std::to_string(i);
        if (v.source[i] < 0 || v.source[i] >= num_vertices || v.target[i] < 0 ||
            v.target[i] >= num_vertices) {
            throw std::invalid_argument(who + ": source or target vertex id is out of range");
        }
        if (!std::isfinite(v.fixed_cost[i])) {
            throw std::invalid_argument(who + ": fixed_cost must be finite");
        }
        Vessel vessel;
        vessel.source = static_cast<VertexId>(v.source[i]);
        vessel.target = static_cast<VertexId>(v.target[i]);
        vessel.fixed_cost = v.fixed_cost[i];

        if (store_limits) {
            for (std::size_t r = 0; r < tracked; ++r) {
                double limit = data.upper[r];
                if (!v.limits.empty()) {
                    limit = v.limits[i * tracked + r];
                    if (std::isnan(limit) || limit == -kInf) {
                        throw std::invalid_argument(who + ": resource limits must be numbers "
                                                          "(+infinity for no limit)");
                    }
                }
                limits.push_back(limit);
            }
        }

        vessel.fixed_begin = static_cast<std::uint32_t>(fixed_base + fixed_row.size());
        for (std::int64_t j = v.fixed_ptr[i]; j < v.fixed_ptr[i + 1]; ++j) {
            if (v.fixed_row[j] < 0 || v.fixed_row[j] >= num_rows) {
                throw std::invalid_argument(who + ": fixed coefficient row " +
                                            std::to_string(v.fixed_row[j]) + " is outside [0, " +
                                            std::to_string(num_constraints_) + ")");
            }
            if (!std::isfinite(v.fixed_value[j])) {
                throw std::invalid_argument(who + ": fixed coefficients must be finite");
            }
            fixed_row.push_back(static_cast<std::uint32_t>(v.fixed_row[j]));
            fixed_value.push_back(v.fixed_value[j]);
        }
        vessel.fixed_end = static_cast<std::uint32_t>(fixed_base + fixed_row.size());

        entries.clear();
        for (std::int64_t j = v.offset_ptr[i]; j < v.offset_ptr[i + 1]; ++j) {
            if (v.offset_edge[j] < 0 || v.offset_edge[j] >= num_edges) {
                throw std::invalid_argument(who + ": cost offset refers to edge " +
                                            std::to_string(v.offset_edge[j]) +
                                            ", which does not exist");
            }
            const double value = v.offset_value[j];
            if (std::isnan(value) || value == -kInf) {
                throw std::invalid_argument(who + ": cost offsets must be finite, or +infinity "
                                                  "to remove an edge");
            }
            entries.push_back({edge_slot_[v.offset_edge[j]], kInvalid, value});
        }
        for (std::int64_t j = v.coef_ptr[i]; j < v.coef_ptr[i + 1]; ++j) {
            if (v.coef_edge[j] < 0 || v.coef_edge[j] >= num_edges) {
                throw std::invalid_argument(who + ": edge coefficient refers to edge " +
                                            std::to_string(v.coef_edge[j]) +
                                            ", which does not exist");
            }
            if (v.coef_row[j] < 0 || v.coef_row[j] >= num_rows) {
                throw std::invalid_argument(who + ": edge coefficient row " +
                                            std::to_string(v.coef_row[j]) + " is outside [0, " +
                                            std::to_string(num_constraints_) + ")");
            }
            if (!std::isfinite(v.coef_value[j])) {
                throw std::invalid_argument(who + ": edge coefficients must be finite");
            }
            entries.push_back({edge_slot_[v.coef_edge[j]],
                               static_cast<std::uint32_t>(v.coef_row[j]), v.coef_value[j]});
        }
        // One overlay entry per touched slot, slots ascending (looked up by
        // binary search), coefficient rows ascending within a slot.
        // Duplicates add up.
        std::stable_sort(entries.begin(), entries.end(),
                         [](const OverlayEntry& a, const OverlayEntry& b) {
                             return a.slot != b.slot ? a.slot < b.slot : a.row < b.row;
                         });
        vessel.overlay_begin = static_cast<std::uint32_t>(overlay_base + overlay_slot.size());
        for (std::size_t j = 0; j < entries.size();) {
            const Slot slot = entries[j].slot;
            double offset = 0.0;
            for (; j < entries.size() && entries[j].slot == slot; ++j) {
                const OverlayEntry& entry = entries[j];
                if (entry.row == kInvalid) {
                    offset += entry.value;
                } else if (j > 0 && entries[j - 1].slot == slot &&
                           entries[j - 1].row == entry.row) {
                    overlay_coef_value.back() += entry.value;
                } else {
                    overlay_coef_row.push_back(entry.row);
                    overlay_coef_value.push_back(entry.value);
                }
            }
            if (offset == kInf) vessel.removes_edges = true;
            overlay_slot.push_back(slot);
            overlay_offset.push_back(offset);
            overlay_coef_end.push_back(coef_base + overlay_coef_row.size());
        }
        vessel.overlay_end = static_cast<std::uint32_t>(overlay_base + overlay_slot.size());
        vessels.push_back(vessel);
    }

    if (first + count >= kInvalid || fixed_base + fixed_row.size() >= kInvalid ||
        overlay_base + overlay_slot.size() >= kInvalid ||
        coef_base + overlay_coef_row.size() >= kInvalid) {
        throw std::invalid_argument("the fleet has too many vessels or vessel-specific entries");
    }

    // Reserve first: once the appends start, none of them can fail.
    vessels_.reserve(first + count);
    limits_.reserve(limits_.size() + limits.size());
    fixed_row_.reserve(fixed_base + fixed_row.size());
    fixed_value_.reserve(fixed_base + fixed_value.size());
    overlay_slot_.reserve(overlay_base + overlay_slot.size());
    overlay_offset_.reserve(overlay_base + overlay_offset.size());
    overlay_coef_ptr_.reserve(overlay_coef_ptr_.size() + overlay_coef_end.size());
    overlay_coef_row_.reserve(coef_base + overlay_coef_row.size());
    overlay_coef_value_.reserve(coef_base + overlay_coef_value.size());

    vessels_.insert(vessels_.end(), vessels.begin(), vessels.end());
    limits_.insert(limits_.end(), limits.begin(), limits.end());
    fixed_row_.insert(fixed_row_.end(), fixed_row.begin(), fixed_row.end());
    fixed_value_.insert(fixed_value_.end(), fixed_value.begin(), fixed_value.end());
    overlay_slot_.insert(overlay_slot_.end(), overlay_slot.begin(), overlay_slot.end());
    overlay_offset_.insert(overlay_offset_.end(), overlay_offset.begin(), overlay_offset.end());
    for (const std::size_t end : overlay_coef_end) {
        overlay_coef_ptr_.push_back(static_cast<std::uint32_t>(end));
    }
    overlay_coef_row_.insert(overlay_coef_row_.end(), overlay_coef_row.begin(),
                             overlay_coef_row.end());
    overlay_coef_value_.insert(overlay_coef_value_.end(), overlay_coef_value.begin(),
                               overlay_coef_value.end());
    prepared_ = false;
    return first;
}

// ------------------------------------------------------------------ prepare

unsigned PricingFleet::resolved_threads() const {
    if (options_.threads != 0) return options_.threads;
    return std::max(1u, std::thread::hardware_concurrency());
}

void PricingFleet::ensure_executor() {
    if (executor_ == nullptr) {
        executor_ = std::make_unique<Executor>(options_.scheduler, options_.partitioner,
                                               resolved_threads());
    }
}

void PricingFleet::prepare() {
    const std::lock_guard<std::mutex> lock(mutex_);
    prepare_locked();
}

void PricingFleet::prepare_locked() {
    if (prepared_) return;
    ensure_executor();
    const Csr& csr = graph_->topology().csr;
    const std::size_t count = vessels_.size();

    // Reachability in each vessel's own network: the shared graph minus the
    // edges the vessel removes. Vessels that remove nothing share the answer
    // of their (source, target) pair.
    {
        std::vector<std::uint8_t> seen;
        std::vector<VertexId> stack;
        std::vector<std::uint8_t> removed;
        std::unordered_map<std::uint64_t, bool> known;
        for (std::size_t i = prepared_vessels_; i < count; ++i) {
            Vessel& vessel = vessels_[i];
            if (vessel.removes_edges) {
                removed.resize(csr.num_slots(), 0);
                for (std::uint32_t j = vessel.overlay_begin; j < vessel.overlay_end; ++j) {
                    if (overlay_offset_[j] == kInf) removed[overlay_slot_[j]] = 1;
                }
                vessel.reachable = reachable_without(csr, vessel.source, vessel.target,
                                                     removed.data(), seen, stack);
                for (std::uint32_t j = vessel.overlay_begin; j < vessel.overlay_end; ++j) {
                    removed[overlay_slot_[j]] = 0;
                }
            } else {
                const std::uint64_t key =
                    (static_cast<std::uint64_t>(vessel.source) << 32) | vessel.target;
                const auto [it, inserted] = known.try_emplace(key, false);
                if (inserted) {
                    it->second = reachable_without(csr, vessel.source, vessel.target, nullptr,
                                                   seen, stack);
                }
                vessel.reachable = it->second;
            }
        }
        prepared_vessels_ = count;
    }

    // Completion bounds depend on the target only (and are valid from the
    // lowest-ranked source that uses them), so vessels with a common target
    // share one table. They are computed on the full graph: removing edges
    // for a vessel can only raise its true minima, so they stay lower bounds.
    const RcspData& data = graph_->resources().tracked_data;
    if (options_.algorithm == Algorithm::Rcsp && data.num_resources > 0) {
        const std::size_t bytes_each =
            static_cast<std::size_t>(csr.num_vertices) * data.num_resources * sizeof(double);
        const std::size_t capacity = options_.bound_cache_bytes / bytes_each;
        std::unordered_map<VertexId, std::uint32_t> group_of;
        std::vector<Bound> wanted;
        for (Vessel& vessel : vessels_) {
            vessel.bound = kInvalid;
            if (!vessel.reachable) continue;
            const std::uint32_t rank = csr.topo_rank[vessel.source];
            auto it = group_of.find(vessel.target);
            if (it == group_of.end()) {
                if (wanted.size() >= capacity) continue;  // over budget: bound per solve
                it = group_of.emplace(vessel.target, static_cast<std::uint32_t>(wanted.size()))
                         .first;
                wanted.push_back(Bound{vessel.target, rank, {}});
            } else {
                Bound& bound = wanted[it->second];
                bound.first_rank = std::min(bound.first_rank, rank);
            }
            vessel.bound = it->second;
        }

        // Keep tables from an earlier prepare() that still cover their group.
        std::unordered_map<VertexId, std::size_t> previous;
        for (std::size_t k = 0; k < bounds_.size(); ++k) previous.emplace(bounds_[k].target, k);
        std::vector<std::uint32_t> missing;
        for (std::uint32_t g = 0; g < wanted.size(); ++g) {
            const auto it = previous.find(wanted[g].target);
            if (it != previous.end() && !bounds_[it->second].to_go.empty() &&
                bounds_[it->second].first_rank <= wanted[g].first_rank) {
                wanted[g].first_rank = bounds_[it->second].first_rank;
                wanted[g].to_go = std::move(bounds_[it->second].to_go);
            } else {
                missing.push_back(g);
            }
        }
        executor_->run(missing.size(), [&](unsigned, std::size_t k) {
            Bound& bound = wanted[missing[k]];
            resource_to_go(csr, data, csr.topo_order[bound.first_rank], bound.target,
                           bound.to_go);
            return true;
        });
        bounds_ = std::move(wanted);
    }

    if (workspaces_.size() != executor_->slots()) {
        workspaces_ = std::vector<Workspace>(executor_->slots());
    }
    outcomes_.resize(count);
    prepared_ = true;
}

// -------------------------------------------------------------- dual update

void PricingFleet::reduce(Slot begin, Slot end, double* out) const {
    const double* const base = graph_->slot_cost().data();
    const double* const pi = duals_.data();
    const double* const value = coef_value_.data();
    const std::uint32_t* const row = coef_row_.data();
    if (width_ == 0) {
        const std::uint32_t* const ptr = coef_ptr_.data();
        for (Slot slot = begin; slot < end; ++slot) {
            double reduced = base[slot];
            for (std::uint32_t j = ptr[slot]; j < ptr[slot + 1]; ++j) {
                reduced -= value[j] * pi[row[j]];
            }
            out[slot] = reduced;
        }
    } else if (width_ == 1) {
        for (Slot slot = begin; slot < end; ++slot) {
            out[slot] = base[slot] - value[slot] * pi[row[slot]];
        }
    } else {
        const std::size_t width = width_;
        for (Slot slot = begin; slot < end; ++slot) {
            const std::size_t at = static_cast<std::size_t>(slot) * width;
            double reduced = base[slot];
            for (std::size_t j = 0; j < width; ++j) reduced -= value[at + j] * pi[row[at + j]];
            out[slot] = reduced;
        }
    }
}

void PricingFleet::update_duals(std::span<const double> duals) {
    const std::lock_guard<std::mutex> lock(mutex_);
    update_locked(duals);
}

void PricingFleet::update_locked(std::span<const double> duals) {
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
    std::copy(duals.begin(), duals.end(), duals_.begin());
    duals_[num_constraints_] = 0.0;

    if (!options_.per_vessel_update) {
        // The one reduced-cost array every vessel reads.
        const std::uint32_t num_slots = graph_->topology().csr.num_slots();
        shared_reduced_.resize(num_slots);
        double* const out = shared_reduced_.data();
        ensure_executor();
        if (num_slots >= kParallelUpdateSlots && executor_->persistent()) {
            const std::size_t chunks = (num_slots + kUpdateGrain - 1) / kUpdateGrain;
            executor_->run(chunks, [&](unsigned, std::size_t chunk) {
                const Slot begin = static_cast<Slot>(chunk * kUpdateGrain);
                const Slot end = static_cast<Slot>(
                    std::min<std::size_t>(num_slots, (chunk + 1) * std::size_t{kUpdateGrain}));
                reduce(begin, end, out);
                return true;
            });
        } else {
            reduce(0, num_slots, out);
        }
    }
    ++epoch_;
    has_duals_ = true;
}

// -------------------------------------------------------------------- solve

namespace {

void check_solve_arguments(Algorithm algorithm, std::uint32_t max_paths, double tolerance,
                           double time_limit) {
    if (max_paths == 0) throw std::invalid_argument("max_paths must be at least 1");
    if (algorithm == Algorithm::Rcsp && max_paths > 1) {
        throw std::invalid_argument("max_paths > 1 is not supported by the rcsp backend");
    }
    if (!(tolerance >= 0.0)) throw std::invalid_argument("tolerance must be nonnegative");
    if (!(time_limit >= 0.0)) throw std::invalid_argument("time_limit must be nonnegative");
}

}  // namespace

FleetResult PricingFleet::solve(std::uint32_t max_paths, bool negative_only, double tolerance,
                                double time_limit) {
    check_solve_arguments(options_.algorithm, max_paths, tolerance, time_limit);
    const std::lock_guard<std::mutex> lock(mutex_);
    return solve_locked(max_paths, negative_only, tolerance, time_limit, 0.0);
}

FleetResult PricingFleet::price(std::span<const double> duals, std::uint32_t max_paths,
                                bool negative_only, double tolerance, double time_limit) {
    check_solve_arguments(options_.algorithm, max_paths, tolerance, time_limit);
    const std::lock_guard<std::mutex> lock(mutex_);
    const auto start = Clock::now();
    update_locked(duals);
    const double update_seconds = seconds_between(start, Clock::now());
    return solve_locked(max_paths, negative_only, tolerance, time_limit, update_seconds);
}

FleetResult PricingFleet::solve_locked(std::uint32_t max_paths, bool negative_only,
                                       double tolerance, double time_limit,
                                       double update_seconds) {
    if (!has_duals_) {
        throw std::invalid_argument("the fleet has no dual values yet: call update_duals() "
                                    "before solve(), or use price()");
    }
    prepare_locked();

    const auto start = Clock::now();
    SolveParams params;
    params.max_paths = max_paths;
    params.negative_only = negative_only;
    params.tolerance = tolerance;
    params.stop.cancelled = &cancel_requested_;
    const bool limited = time_limit < kMaxTimeLimit;
    if (limited) {
        params.stop.deadline =
            start + std::chrono::duration_cast<Clock::duration>(
                        std::chrono::duration<double>(time_limit));
    }

    // Every slot starts as "not priced". A vessel that is never reached,
    // because the solve was stopped first, keeps that status.
    for (Outcome& outcome : outcomes_) outcome.reset();

    // A cancel() counts from here on; one that arrived earlier was aimed at
    // a solve that has already returned.
    cancel_requested_.store(false);
    struct Running {
        std::atomic<bool>& flag;
        explicit Running(std::atomic<bool>& f) : flag(f) { flag.store(true); }
        ~Running() { flag.store(false); }
    };
    const Running running(running_);

    executor_->run(vessels_.size(), [&](unsigned slot, std::size_t index) {
        if (params.stop.stop_requested()) return false;
        Workspace& workspace = workspaces_[slot];
        Outcome& outcome = outcomes_[index];
        // A failing vessel must not take the others down with it: its error
        // is recorded in its own slot and the worker moves on.
        try {
            price_vessel(index, workspace, params, outcome);
        } catch (const std::exception& error) {
            outcome.reset();
            workspace.clear_accumulator();
            outcome.error = describe(error);
        } catch (...) {
            outcome.reset();
            workspace.clear_accumulator();
            outcome.error = "unknown error";
        }
        return true;
    });
    const auto solved = Clock::now();

    FleetResult result = assemble();
    result.cancelled = cancel_requested_.load();
    result.timed_out = limited && !result.cancelled &&
                       result.num_interrupted > result.num_failed &&
                       solved >= params.stop.deadline;
    result.threads = executor_->slots();
    result.update_seconds = update_seconds;
    result.solve_seconds = seconds_between(start, solved);
    result.assemble_seconds = seconds_between(solved, Clock::now());
    return result;
}

void PricingFleet::price_vessel(std::size_t index, Workspace& workspace,
                                const SolveParams& params, Outcome& outcome) const {
    const auto start = Clock::now();
    const Vessel& vessel = vessels_[index];
    if (debug_fail_.load(std::memory_order_relaxed) == index) {
        throw std::runtime_error("injected failure in vessel " + std::to_string(index));
    }
    if (!vessel.reachable) {
        outcome.status = Status::Unreachable;
        outcome.seconds = seconds_between(start, Clock::now());
        return;
    }

    const Csr& csr = graph_->topology().csr;
    const std::uint32_t num_slots = csr.num_slots();
    const double* const pi = duals_.data();
    double fixed_reduced = vessel.fixed_cost;
    for (std::uint32_t j = vessel.fixed_begin; j < vessel.fixed_end; ++j) {
        fixed_reduced -= fixed_value_[j] * pi[fixed_row_[j]];
    }

    // The vessel's reduced costs. Without an overlay they are the shared
    // array itself. With one, the worker patches its private copy of the
    // shared array at the overlay's slots and puts the shared values back
    // when the vessel is done, so the copy is made once per dual update and
    // not once per vessel.
    struct Restore {
        double* cost = nullptr;
        const double* shared = nullptr;
        const Slot* begin = nullptr;
        const Slot* end = nullptr;
        ~Restore() {
            if (cost == nullptr) return;
            for (const Slot* slot = begin; slot != end; ++slot) cost[*slot] = shared[*slot];
        }
    } restore;

    const double* cost = shared_reduced_.data();
    const bool has_overlay = vessel.overlay_begin != vessel.overlay_end;
    if (options_.per_vessel_update || has_overlay) {
        if (options_.per_vessel_update) {
            workspace.cost.resize(num_slots);
            reduce(0, num_slots, workspace.cost.data());
        } else {
            if (workspace.epoch != epoch_ || workspace.cost.size() != num_slots) {
                workspace.epoch = 0;
                workspace.cost.assign(shared_reduced_.begin(), shared_reduced_.end());
                workspace.epoch = epoch_;
            }
            restore.cost = workspace.cost.data();
            restore.shared = shared_reduced_.data();
            restore.begin = overlay_slot_.data() + vessel.overlay_begin;
            restore.end = overlay_slot_.data() + vessel.overlay_end;
        }
        double* const own = workspace.cost.data();
        for (std::uint32_t j = vessel.overlay_begin; j < vessel.overlay_end; ++j) {
            // An offset of +infinity makes the sum +infinity, which the
            // solvers read as "this edge does not exist".
            double value = own[overlay_slot_[j]] + overlay_offset_[j];
            for (std::uint32_t k = overlay_coef_ptr_[j]; k < overlay_coef_ptr_[j + 1]; ++k) {
                value -= overlay_coef_value_[k] * pi[overlay_coef_row_[k]];
            }
            own[overlay_slot_[j]] = value;
        }
        cost = own;
    }

    PathResult single;
    std::vector<PathResult> many;
    if (options_.algorithm == Algorithm::Rcsp) {
        const RcspData& data = graph_->resources().tracked_data;
        const double* to_go = nullptr;
        if (vessel.bound != kInvalid) {
            to_go = bounds_[vessel.bound].to_go.data();
        } else {
            resource_to_go(csr, data, vessel.source, vessel.target, workspace.rcsp.to_go);
            to_go = workspace.rcsp.to_go.data();
        }
        const double* const upper =
            limits_.empty() ? nullptr : limits_.data() + index * data.num_resources;
        single = solve_rcsp(csr, data, cost, vessel.source, vessel.target, options_.rcsp,
                            workspace.rcsp, outcome.stats, to_go, upper, &params.stop);
        if (single.status == Status::Interrupted) {
            outcome.seconds = seconds_between(start, Clock::now());
            return;  // the slot keeps Status::Interrupted
        }
    } else if (params.max_paths == 1) {
        single = solve_dag(csr, cost, vessel.source, vessel.target, workspace.dag);
    } else {
        many = solve_dag_k_best(csr, cost, vessel.source, vessel.target, params.max_paths,
                                workspace.k_best);
    }

    const std::size_t found =
        params.max_paths == 1 ? (single.status == Status::Optimal ? 1 : 0) : many.size();
    if (found == 0) {
        // The target is reachable, so with resources an empty answer means
        // that the bounds exclude every path.
        outcome.status = options_.algorithm == Algorithm::Rcsp ? Status::Infeasible
                                                               : Status::Unreachable;
    } else {
        if (workspace.accumulator.size() != num_constraints_) {
            workspace.accumulator.assign(num_constraints_, 0.0);
            workspace.touched_flag.assign(num_constraints_, 0);
            workspace.touched.clear();
        }
        outcome.status = Status::Optimal;
        outcome.best = fixed_reduced + (params.max_paths == 1 ? single.cost : many.front().cost);
        for (std::size_t p = 0; p < found; ++p) {
            const PathResult& path = params.max_paths == 1 ? single : many[p];
            if (params.negative_only && !(fixed_reduced + path.cost < -params.tolerance)) break;
            append_column(vessel, path, fixed_reduced, workspace, outcome);
        }
    }
    outcome.stats.solve_seconds = outcome.seconds = seconds_between(start, Clock::now());
}

void PricingFleet::append_column(const Vessel& vessel, const PathResult& path,
                                 double fixed_reduced, Workspace& workspace,
                                 Outcome& outcome) const {
    const Csr& csr = graph_->topology().csr;
    const ResourceTable& table = graph_->resources();
    const double* const base = graph_->slot_cost().data();
    const Slot* const overlay_begin = overlay_slot_.data() + vessel.overlay_begin;
    const Slot* const overlay_end = overlay_slot_.data() + vessel.overlay_end;
    const std::uint32_t unused_row = static_cast<std::uint32_t>(num_constraints_);

    const std::size_t first_edge = outcome.edges.size();
    double original = vessel.fixed_cost;
    for (const Slot slot : path.slots) {
        outcome.edges.push_back(csr.slot_edge[slot]);
        original += base[slot];
        if (width_ == 0) {
            for (std::uint32_t j = coef_ptr_[slot]; j < coef_ptr_[slot + 1]; ++j) {
                workspace.add(coef_row_[j], coef_value_[j]);
            }
        } else {
            const std::size_t at = static_cast<std::size_t>(slot) * width_;
            for (std::size_t j = at; j < at + width_; ++j) {
                if (coef_row_[j] != unused_row) workspace.add(coef_row_[j], coef_value_[j]);
            }
        }
        if (overlay_begin == overlay_end) continue;
        const Slot* const hit = std::lower_bound(overlay_begin, overlay_end, slot);
        if (hit == overlay_end || *hit != slot) continue;
        const std::size_t j = static_cast<std::size_t>(hit - overlay_slot_.data());
        original += overlay_offset_[j];
        for (std::uint32_t k = overlay_coef_ptr_[j]; k < overlay_coef_ptr_[j + 1]; ++k) {
            workspace.add(overlay_coef_row_[k], overlay_coef_value_[k]);
        }
    }
    for (std::uint32_t j = vessel.fixed_begin; j < vessel.fixed_end; ++j) {
        workspace.add(fixed_row_[j], fixed_value_[j]);
    }
    std::sort(workspace.touched.begin(), workspace.touched.end());
    for (const std::uint32_t row : workspace.touched) {
        if (workspace.accumulator[row] != 0.0) {
            outcome.coef_row.push_back(row);
            outcome.coef_value.push_back(workspace.accumulator[row]);
        }
    }
    workspace.clear_accumulator();

    // Final value of every resource column, accumulated in travel order like
    // CompiledGraph::path_resources.
    for (std::size_t c = 0; c < table.num_columns(); ++c) {
        const double* const delta = table.edge_delta[c].data();
        double value = table.initial[c];
        for (std::size_t j = first_edge; j < outcome.edges.size(); ++j) {
            value += delta[outcome.edges[j]];
        }
        outcome.resources.push_back(value);
    }

    outcome.edge_end.push_back(static_cast<std::uint32_t>(outcome.edges.size()));
    outcome.coef_end.push_back(static_cast<std::uint32_t>(outcome.coef_row.size()));
    outcome.original.push_back(original);
    // path.cost is the solver's own sum of reduced edge costs, so the order
    // of a vessel's columns matches the order the solver established.
    outcome.reduced.push_back(fixed_reduced + path.cost);
}

FleetResult PricingFleet::assemble() const {
    const std::size_t count = vessels_.size();
    FleetResult result;
    result.status.resize(count);
    result.best_reduced_cost.resize(count);
    result.vessel_seconds.resize(count);
    result.errors.resize(count);
    result.column_ptr.assign(count + 1, 0);

    std::size_t columns = 0;
    std::size_t edges = 0;
    std::size_t coefficients = 0;
    for (const Outcome& outcome : outcomes_) {
        columns += outcome.num_columns();
        edges += outcome.edges.size();
        coefficients += outcome.coef_row.size();
    }
    result.column_vessel.reserve(columns);
    result.original_cost.reserve(columns);
    result.reduced_cost.reserve(columns);
    result.resources.reserve(columns * graph_->resources().num_columns());
    result.edge_ptr.reserve(columns + 1);
    result.edges.reserve(edges);
    result.coef_ptr.reserve(columns + 1);
    result.coef_row.reserve(coefficients);
    result.coef_value.reserve(coefficients);
    result.edge_ptr.push_back(0);
    result.coef_ptr.push_back(0);

    for (std::size_t i = 0; i < count; ++i) {
        const Outcome& outcome = outcomes_[i];
        result.status[i] = static_cast<std::int8_t>(outcome.status);
        // An unpriced vessel has no known best reduced cost. NaN compares
        // false with everything, so it can never pass a convergence test.
        result.best_reduced_cost[i] =
            outcome.status == Status::Interrupted ? std::numeric_limits<double>::quiet_NaN()
                                                  : outcome.best;
        result.vessel_seconds[i] = outcome.seconds;
        if (outcome.status == Status::Interrupted) ++result.num_interrupted;
        if (!outcome.error.empty()) {
            result.errors[i] = outcome.error;
            ++result.num_failed;
        }

        const std::size_t edge_base = result.edges.size();
        const std::size_t coef_base = result.coef_row.size();
        result.edges.insert(result.edges.end(), outcome.edges.begin(), outcome.edges.end());
        result.coef_row.insert(result.coef_row.end(), outcome.coef_row.begin(),
                               outcome.coef_row.end());
        result.coef_value.insert(result.coef_value.end(), outcome.coef_value.begin(),
                                 outcome.coef_value.end());
        result.original_cost.insert(result.original_cost.end(), outcome.original.begin(),
                                    outcome.original.end());
        result.reduced_cost.insert(result.reduced_cost.end(), outcome.reduced.begin(),
                                   outcome.reduced.end());
        result.resources.insert(result.resources.end(), outcome.resources.begin(),
                                outcome.resources.end());
        for (std::size_t c = 0; c < outcome.num_columns(); ++c) {
            result.column_vessel.push_back(static_cast<std::int64_t>(i));
            result.edge_ptr.push_back(static_cast<std::int64_t>(edge_base + outcome.edge_end[c]));
            result.coef_ptr.push_back(static_cast<std::int64_t>(coef_base + outcome.coef_end[c]));
        }
        result.column_ptr[i + 1] = static_cast<std::int64_t>(result.reduced_cost.size());

        const SolveStats& stats = outcome.stats;
        result.labels.solve_seconds += outcome.seconds;
        result.labels.labels_created += stats.labels_created;
        result.labels.labels_extended += stats.labels_extended;
        result.labels.labels_dominated += stats.labels_dominated;
        result.labels.labels_infeasible += stats.labels_infeasible;
        result.labels.pareto_labels += stats.pareto_labels;
        result.labels.max_active_labels =
            std::max(result.labels.max_active_labels, stats.max_active_labels);
    }
    return result;
}

// ---------------------------------------------------------------- accessors

void PricingFleet::cancel() {
    if (running_.load()) cancel_requested_.store(true);
}

void PricingFleet::debug_fail_vessel(std::uint32_t vessel) { debug_fail_.store(vessel); }

std::size_t PricingFleet::num_vessels() const {
    const std::lock_guard<std::mutex> lock(mutex_);
    return vessels_.size();
}

unsigned PricingFleet::threads() const {
    const std::lock_guard<std::mutex> lock(mutex_);
    return executor_ != nullptr ? executor_->slots() : resolved_threads();
}

bool PricingFleet::prepared() const {
    const std::lock_guard<std::mutex> lock(mutex_);
    return prepared_;
}

bool PricingFleet::has_duals() const {
    const std::lock_guard<std::mutex> lock(mutex_);
    return has_duals_;
}

std::vector<std::uint8_t> PricingFleet::reachable() const {
    const std::lock_guard<std::mutex> lock(mutex_);
    if (!prepared_) throw std::invalid_argument("call prepare() first");
    std::vector<std::uint8_t> flags(vessels_.size());
    for (std::size_t i = 0; i < flags.size(); ++i) flags[i] = vessels_[i].reachable ? 1 : 0;
    return flags;
}

FleetMemory PricingFleet::memory() const {
    const std::lock_guard<std::mutex> lock(mutex_);
    const Topology& topo = graph_->topology();
    const Csr& csr = topo.csr;
    const ResourceTable& table = graph_->resources();
    FleetMemory memory;
    memory.shared_graph =
        bytes_of(topo.edge_source) + bytes_of(topo.edge_target) + bytes_of(csr.offsets) +
        bytes_of(csr.slot_source) + bytes_of(csr.slot_target) + bytes_of(csr.slot_edge) +
        bytes_of(csr.topo_order) + bytes_of(csr.topo_rank) + graph_->edge_cost().size_bytes() +
        graph_->slot_cost().size_bytes() + bytes_of(table.tracked_data.slot_delta);
    for (const std::vector<double>& column : table.edge_delta) {
        memory.shared_graph += bytes_of(column);
    }
    memory.shared_pricing = bytes_of(edge_slot_) + bytes_of(coef_ptr_) + bytes_of(coef_row_) +
                            bytes_of(coef_value_) + bytes_of(shared_reduced_) + bytes_of(duals_);
    memory.vessels = bytes_of(vessels_) + bytes_of(limits_) + bytes_of(fixed_row_) +
                     bytes_of(fixed_value_) + bytes_of(overlay_slot_) +
                     bytes_of(overlay_offset_) + bytes_of(overlay_coef_ptr_) +
                     bytes_of(overlay_coef_row_) + bytes_of(overlay_coef_value_);
    for (const Bound& bound : bounds_) memory.bound_cache += bytes_of(bound.to_go);
    for (const Workspace& workspace : workspaces_) memory.workspaces += workspace.bytes();
    for (const Outcome& outcome : outcomes_) memory.results += outcome.bytes();
    return memory;
}

}  // namespace celerisroute
