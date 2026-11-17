// nanobind module celerispath._core: a thin, private bridge between the Python
// package and the native core. Argument normalisation and the public object
// model live in Python; this file only moves contiguous buffers across.
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "celerispath/graph.hpp"
#include "celerispath/pricing_engine.hpp"
#include "celerispath/pricing_fleet.hpp"

namespace nb = nanobind;
using namespace nb::literals;
using namespace celerispath;

namespace {

template <class T>
using InArray = nb::ndarray<const T, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
template <class T>
using OutArray = nb::ndarray<nb::numpy, T, nb::ndim<1>>;

template <class T>
std::span<const T> as_span(const InArray<T>& array) {
    return {array.data(), array.shape(0)};
}

// Hands a vector to NumPy without copying; the capsule frees it.
template <class T>
nb::object to_numpy(std::vector<T>&& values) {
    auto* heap = new std::vector<T>(std::move(values));
    nb::capsule owner(heap, [](void* p) noexcept { delete static_cast<std::vector<T>*>(p); });
    return nb::cast(OutArray<T>(heap->data(), {heap->size()}, owner));
}

template <class Range>
nb::object ids_to_numpy(const Range& ids) {
    return to_numpy(std::vector<std::int64_t>(ids.begin(), ids.end()));
}

VertexId checked_vertex(std::int64_t v, std::uint32_t num_vertices, const char* role) {
    if (v < 0 || v >= static_cast<std::int64_t>(num_vertices)) {
        throw std::invalid_argument(std::string(role) + " vertex id " + std::to_string(v) +
                                    " is outside [0, " + std::to_string(num_vertices) + ")");
    }
    return static_cast<VertexId>(v);
}

Algorithm checked_algorithm(int value) {
    if (value != 0 && value != 1) throw std::invalid_argument("unknown algorithm code");
    return static_cast<Algorithm>(value);
}

nb::tuple stats_tuple(const SolveStats& s) {
    return nb::make_tuple(s.solve_seconds, s.labels_created, s.labels_extended,
                          s.labels_dominated, s.labels_infeasible, s.max_active_labels,
                          s.pareto_labels);
}

// The builder is the only mutable native object. Its mutex is always taken
// after the GIL has been released (or while holding it for O(1) work) and is
// never held while acquiring the GIL, so the two cannot deadlock.
struct PyBuilder {
    explicit PyBuilder(std::uint64_t num_vertices) : builder(num_vertices) {}
    GraphBuilder builder;
    std::mutex mutex;
};

struct PyNetwork {
    std::shared_ptr<const CompiledGraph> graph;
};

struct PyPricing {
    std::shared_ptr<PricingEngine> engine;
};

struct PyFleet {
    std::shared_ptr<PricingFleet> fleet;
};

// Moves a fleet result into NumPy arrays without copying them. Called with
// the GIL held, after the native call has returned.
nb::dict fleet_result_dict(FleetResult&& r) {
    nb::dict out;
    out["status"] = to_numpy(std::move(r.status));
    out["best_reduced_cost"] = to_numpy(std::move(r.best_reduced_cost));
    out["vessel_seconds"] = to_numpy(std::move(r.vessel_seconds));
    out["column_ptr"] = to_numpy(std::move(r.column_ptr));
    out["column_vessel"] = to_numpy(std::move(r.column_vessel));
    out["original_cost"] = to_numpy(std::move(r.original_cost));
    out["reduced_cost"] = to_numpy(std::move(r.reduced_cost));
    out["resources"] = to_numpy(std::move(r.resources));
    out["edge_ptr"] = to_numpy(std::move(r.edge_ptr));
    out["edges"] = to_numpy(std::move(r.edges));
    out["coef_ptr"] = to_numpy(std::move(r.coef_ptr));
    out["coef_row"] = to_numpy(std::move(r.coef_row));
    out["coef_value"] = to_numpy(std::move(r.coef_value));
    // Only built when a vessel failed, which is the exception.
    if (r.num_failed > 0) {
        out["errors"] = nb::cast(r.errors);
    } else {
        out["errors"] = nb::none();
    }
    out["cancelled"] = r.cancelled;
    out["timed_out"] = r.timed_out;
    out["num_interrupted"] = r.num_interrupted;
    out["threads"] = r.threads;
    out["timings"] = nb::make_tuple(r.update_seconds, r.solve_seconds, r.assemble_seconds);
    out["labels"] = stats_tuple(r.labels);
    return out;
}

nb::tuple pricing_result_tuple(PricingResult&& result) {
    nb::list columns;
    for (PricingColumn& column : result.columns) {
        columns.append(nb::make_tuple(ids_to_numpy(column.edges), column.original_cost,
                                      column.reduced_cost, to_numpy(std::move(column.resources)),
                                      to_numpy(std::move(column.coef_index)),
                                      to_numpy(std::move(column.coef_value))));
    }
    const PricingStats& s = result.stats;
    return nb::make_tuple(static_cast<int>(result.status), result.best_reduced_cost, columns,
                          nb::make_tuple(s.update_seconds, s.solve_seconds, s.reconstruct_seconds),
                          stats_tuple(s.labels));
}

}  // namespace

NB_MODULE(_core, m) {
    m.doc() = "Private native core of celerispath. Use the celerispath package instead.";

    nb::class_<PyBuilder>(m, "GraphBuilder")
        .def(nb::init<std::uint64_t>(), "num_vertices"_a)
        .def_prop_ro("num_vertices",
                     [](PyBuilder& self) {
                         const std::lock_guard<std::mutex> lock(self.mutex);
                         return self.builder.num_vertices();
                     })
        .def_prop_ro("num_edges",
                     [](PyBuilder& self) {
                         const std::lock_guard<std::mutex> lock(self.mutex);
                         return self.builder.num_edges();
                     })
        .def("resource_names",
             [](PyBuilder& self) {
                 const std::lock_guard<std::mutex> lock(self.mutex);
                 return self.builder.resource_names();
             })
        .def(
            "add_edge",
            [](PyBuilder& self, std::int64_t source, std::int64_t target, double cost,
               const std::vector<std::string>& names, const std::vector<double>& values) {
                if (names.size() != values.size()) {
                    throw std::invalid_argument("resource names and values do not match");
                }
                std::vector<std::pair<std::string, double>> resources;
                resources.reserve(names.size());
                for (std::size_t i = 0; i < names.size(); ++i) {
                    resources.emplace_back(names[i], values[i]);
                }
                const std::lock_guard<std::mutex> lock(self.mutex);
                return self.builder.add_edge(source, target, cost, resources);
            },
            "source"_a, "target"_a, "cost"_a, "names"_a, "values"_a)
        .def(
            "add_edges",
            [](PyBuilder& self, const InArray<std::int64_t>& source,
               const InArray<std::int64_t>& target, const InArray<double>& cost,
               const std::vector<std::string>& names,
               const std::vector<InArray<double>>& columns) {
                std::vector<std::span<const double>> spans;
                spans.reserve(columns.size());
                for (const InArray<double>& column : columns) spans.push_back(as_span(column));
                const nb::gil_scoped_release release;
                const std::lock_guard<std::mutex> lock(self.mutex);
                return self.builder.add_edges(as_span(source), as_span(target), as_span(cost),
                                              names, spans);
            },
            "source"_a.noconvert(), "target"_a.noconvert(), "cost"_a.noconvert(), "names"_a,
            "columns"_a)
        .def(
            "compile",
            [](PyBuilder& self, const std::vector<std::string>& names,
               const std::vector<double>& lower, const std::vector<double>& upper,
               const std::vector<double>& initial) {
                if (lower.size() != names.size() || upper.size() != names.size() ||
                    initial.size() != names.size()) {
                    throw std::invalid_argument("resource declaration arrays differ in length");
                }
                std::vector<ResourceSpec> specs(names.size());
                for (std::size_t i = 0; i < names.size(); ++i) {
                    specs[i] = ResourceSpec{names[i], lower[i], upper[i], initial[i]};
                }
                const nb::gil_scoped_release release;
                const std::lock_guard<std::mutex> lock(self.mutex);
                return PyNetwork{self.builder.compile(specs)};
            },
            "names"_a, "lower"_a, "upper"_a, "initial"_a);

    nb::class_<PyNetwork>(m, "CompiledGraph")
        .def_prop_ro("num_vertices",
                     [](const PyNetwork& self) { return self.graph->topology().num_vertices; })
        .def_prop_ro("num_edges",
                     [](const PyNetwork& self) { return self.graph->topology().num_edges(); })
        .def_prop_ro("is_acyclic",
                     [](const PyNetwork& self) { return self.graph->topology().acyclic; })
        .def("topological_order",
             [](const PyNetwork& self) {
                 return ids_to_numpy(self.graph->topology().csr.topo_order);
             })
        .def("cycle",
             [](const PyNetwork& self) { return ids_to_numpy(self.graph->topology().cycle); })
        .def("edge_source",
             [](const PyNetwork& self) { return ids_to_numpy(self.graph->topology().edge_source); })
        .def("edge_target",
             [](const PyNetwork& self) { return ids_to_numpy(self.graph->topology().edge_target); })
        .def("edge_cost",
             [](const PyNetwork& self) {
                 const std::span<const double> cost = self.graph->edge_cost();
                 return to_numpy(std::vector<double>(cost.begin(), cost.end()));
             })
        .def("resource_names",
             [](const PyNetwork& self) { return self.graph->resources().names; })
        // One (name, lower, upper, initial, declared, tracked) tuple per column.
        .def("resource_info",
             [](const PyNetwork& self) {
                 const ResourceTable& table = self.graph->resources();
                 std::vector<std::uint8_t> tracked(table.num_columns(), 0);
                 for (const std::uint32_t column : table.tracked) tracked[column] = 1;
                 nb::list info;
                 for (std::size_t c = 0; c < table.num_columns(); ++c) {
                     info.append(nb::make_tuple(table.names[c], table.lower[c], table.upper[c],
                                                table.initial[c], table.declared[c] != 0,
                                                tracked[c] != 0));
                 }
                 return info;
             })
        .def(
            "edge_resource",
            [](const PyNetwork& self, std::size_t column) {
                const ResourceTable& table = self.graph->resources();
                if (column >= table.num_columns()) {
                    throw std::invalid_argument("resource column is out of range");
                }
                return to_numpy(std::vector<double>(table.edge_delta[column]));
            },
            "column"_a)
        .def(
            "with_costs",
            [](const PyNetwork& self, const InArray<double>& cost) {
                return PyNetwork{self.graph->with_costs(as_span(cost))};
            },
            "cost"_a.noconvert())
        // -> (status, cost, edges, resources, statistics)
        .def(
            "solve",
            [](const PyNetwork& self, std::int64_t source, std::int64_t target, int algorithm,
               double dominance_tolerance) {
                const std::uint32_t n = self.graph->topology().num_vertices;
                const VertexId s = checked_vertex(source, n, "source");
                const VertexId t = checked_vertex(target, n, "target");
                const Algorithm algo = checked_algorithm(algorithm);
                if (!(dominance_tolerance >= 0.0)) {
                    throw std::invalid_argument("dominance_tolerance must be nonnegative");
                }
                RcspOptions options;
                options.dominance_tolerance = dominance_tolerance;
                Solution solution;
                {
                    const nb::gil_scoped_release release;
                    solution = self.graph->solve(s, t, algo, options);
                }
                return nb::make_tuple(static_cast<int>(solution.status), solution.cost,
                                      ids_to_numpy(solution.edges),
                                      to_numpy(std::move(solution.resources)),
                                      stats_tuple(solution.stats));
            },
            "source"_a, "target"_a, "algorithm"_a, "dominance_tolerance"_a);

    nb::class_<PyPricing>(m, "PricingEngine")
        .def(
            "__init__",
            [](PyPricing* self, const PyNetwork& network, std::int64_t source, std::int64_t target,
               int algorithm, std::size_t num_rows, const InArray<std::int64_t>& indptr,
               const InArray<std::int64_t>& indices, const InArray<double>& data,
               double fixed_cost, const InArray<std::int64_t>& fixed_index,
               const InArray<double>& fixed_value, double dominance_tolerance) {
                const std::uint32_t n = network.graph->topology().num_vertices;
                const VertexId s = checked_vertex(source, n, "source");
                const VertexId t = checked_vertex(target, n, "target");
                const Algorithm algo = checked_algorithm(algorithm);
                if (!(dominance_tolerance >= 0.0)) {
                    throw std::invalid_argument("dominance_tolerance must be nonnegative");
                }
                RcspOptions options;
                options.dominance_tolerance = dominance_tolerance;
                const SparseColumns a{num_rows, as_span(indptr), as_span(indices), as_span(data)};
                std::shared_ptr<PricingEngine> engine;
                {
                    const nb::gil_scoped_release release;
                    engine = std::make_shared<PricingEngine>(network.graph, s, t, algo, a,
                                                             fixed_cost, as_span(fixed_index),
                                                             as_span(fixed_value), options);
                }
                new (self) PyPricing{std::move(engine)};
            },
            "network"_a, "source"_a, "target"_a, "algorithm"_a, "num_rows"_a,
            "indptr"_a.noconvert(), "indices"_a.noconvert(), "data"_a.noconvert(), "fixed_cost"_a,
            "fixed_index"_a.noconvert(), "fixed_value"_a.noconvert(), "dominance_tolerance"_a)
        .def_prop_ro("num_constraints",
                     [](const PyPricing& self) { return self.engine->num_constraints(); })
        .def_prop_ro("target_reachable",
                     [](const PyPricing& self) { return self.engine->target_reachable(); })
        .def_prop_ro("active_vertices",
                     [](const PyPricing& self) { return self.engine->active_vertices(); })
        .def_prop_ro("active_edges",
                     [](const PyPricing& self) { return self.engine->active_edges(); })
        // -> (status, best_reduced_cost, columns, timings, label statistics)
        .def(
            "solve",
            [](const PyPricing& self, const InArray<double>& duals, std::uint32_t max_paths,
               bool negative_only, double tolerance) {
                // Copied so the solve cannot observe a concurrent in-place edit.
                const std::vector<double> pi(duals.data(), duals.data() + duals.shape(0));
                PricingResult result;
                {
                    const nb::gil_scoped_release release;
                    result = self.engine->solve(pi, max_paths, negative_only, tolerance);
                }
                return pricing_result_tuple(std::move(result));
            },
            "duals"_a.noconvert(), "max_paths"_a, "negative_only"_a, "tolerance"_a);

    m.def("tbb_available", &tbb_available);
    m.def("tbb_version", &tbb_runtime_version);

    // Every method that may wait for the fleet's mutex or run native work
    // releases the GIL first, so a thread blocked on the fleet never keeps
    // the interpreter from running, and no Python object is touched while
    // vessels are being priced.
    nb::class_<PyFleet>(m, "PricingFleet")
        .def(
            "__init__",
            [](PyFleet* self, const PyNetwork& network, int algorithm, std::size_t num_rows,
               const InArray<std::int64_t>& indptr, const InArray<std::int64_t>& indices,
               const InArray<double>& data, std::size_t width, const InArray<std::int64_t>& row,
               const InArray<double>& value, int scheduler, int partitioner, unsigned threads,
               bool per_vessel_update, double dominance_tolerance,
               std::size_t bound_cache_bytes) {
                if (scheduler != 0 && scheduler != 1) {
                    throw std::invalid_argument("unknown scheduler code");
                }
                if (partitioner < 0 || partitioner > 2) {
                    throw std::invalid_argument("unknown partitioner code");
                }
                FleetOptions options;
                options.algorithm = checked_algorithm(algorithm);
                options.rcsp.dominance_tolerance = dominance_tolerance;
                options.scheduler = static_cast<Scheduler>(scheduler);
                options.partitioner = static_cast<Partitioner>(partitioner);
                options.threads = threads;
                options.per_vessel_update = per_vessel_update;
                options.bound_cache_bytes = bound_cache_bytes;
                DualCoefficients coefficients;
                coefficients.num_rows = num_rows;
                coefficients.sparse =
                    SparseColumns{num_rows, as_span(indptr), as_span(indices), as_span(data)};
                coefficients.width = width;
                coefficients.row = as_span(row);
                coefficients.value = as_span(value);
                std::shared_ptr<PricingFleet> fleet;
                {
                    const nb::gil_scoped_release release;
                    fleet = std::make_shared<PricingFleet>(network.graph, coefficients, options);
                }
                new (self) PyFleet{std::move(fleet)};
            },
            "network"_a, "algorithm"_a, "num_rows"_a, "indptr"_a.noconvert(),
            "indices"_a.noconvert(), "data"_a.noconvert(), "width"_a, "row"_a.noconvert(),
            "value"_a.noconvert(), "scheduler"_a, "partitioner"_a, "threads"_a,
            "per_vessel_update"_a, "dominance_tolerance"_a, "bound_cache_bytes"_a)
        .def(
            "add_vessels",
            [](const PyFleet& self, const InArray<std::int64_t>& source,
               const InArray<std::int64_t>& target, const InArray<double>& fixed_cost,
               const InArray<double>& limits, const InArray<std::int64_t>& fixed_ptr,
               const InArray<std::int64_t>& fixed_row, const InArray<double>& fixed_value,
               const InArray<std::int64_t>& offset_ptr, const InArray<std::int64_t>& offset_edge,
               const InArray<double>& offset_value, const InArray<std::int64_t>& coef_ptr,
               const InArray<std::int64_t>& coef_edge, const InArray<std::int64_t>& coef_row,
               const InArray<double>& coef_value) {
                FleetVessels vessels;
                vessels.source = as_span(source);
                vessels.target = as_span(target);
                vessels.fixed_cost = as_span(fixed_cost);
                vessels.limits = as_span(limits);
                vessels.fixed_ptr = as_span(fixed_ptr);
                vessels.fixed_row = as_span(fixed_row);
                vessels.fixed_value = as_span(fixed_value);
                vessels.offset_ptr = as_span(offset_ptr);
                vessels.offset_edge = as_span(offset_edge);
                vessels.offset_value = as_span(offset_value);
                vessels.coef_ptr = as_span(coef_ptr);
                vessels.coef_edge = as_span(coef_edge);
                vessels.coef_row = as_span(coef_row);
                vessels.coef_value = as_span(coef_value);
                const nb::gil_scoped_release release;
                return self.fleet->add_vessels(vessels);
            },
            "source"_a.noconvert(), "target"_a.noconvert(), "fixed_cost"_a.noconvert(),
            "limits"_a.noconvert(), "fixed_ptr"_a.noconvert(), "fixed_row"_a.noconvert(),
            "fixed_value"_a.noconvert(), "offset_ptr"_a.noconvert(),
            "offset_edge"_a.noconvert(), "offset_value"_a.noconvert(), "coef_ptr"_a.noconvert(),
            "coef_edge"_a.noconvert(), "coef_row"_a.noconvert(), "coef_value"_a.noconvert())
        .def("prepare",
             [](const PyFleet& self) {
                 const nb::gil_scoped_release release;
                 self.fleet->prepare();
             })
        .def(
            "update_duals",
            [](const PyFleet& self, const InArray<double>& duals) {
                // Copied so the update cannot observe a concurrent in-place edit.
                const std::vector<double> pi(duals.data(), duals.data() + duals.shape(0));
                const nb::gil_scoped_release release;
                self.fleet->update_duals(pi);
            },
            "duals"_a.noconvert())
        .def(
            "solve",
            [](const PyFleet& self, std::uint32_t max_paths, bool negative_only, double tolerance,
               double time_limit) {
                FleetResult result;
                {
                    const nb::gil_scoped_release release;
                    result = self.fleet->solve(max_paths, negative_only, tolerance, time_limit);
                }
                return fleet_result_dict(std::move(result));
            },
            "max_paths"_a, "negative_only"_a, "tolerance"_a, "time_limit"_a)
        // Dual update and pricing of the whole fleet in one native call.
        .def(
            "price",
            [](const PyFleet& self, const InArray<double>& duals, std::uint32_t max_paths,
               bool negative_only, double tolerance, double time_limit) {
                const std::vector<double> pi(duals.data(), duals.data() + duals.shape(0));
                FleetResult result;
                {
                    const nb::gil_scoped_release release;
                    result = self.fleet->price(pi, max_paths, negative_only, tolerance,
                                               time_limit);
                }
                return fleet_result_dict(std::move(result));
            },
            "duals"_a.noconvert(), "max_paths"_a, "negative_only"_a, "tolerance"_a,
            "time_limit"_a)
        // Lock-free, so it gets through while another thread is inside solve().
        .def("cancel", [](const PyFleet& self) { self.fleet->cancel(); })
        .def_prop_ro("num_constraints",
                     [](const PyFleet& self) { return self.fleet->num_constraints(); })
        .def_prop_ro("num_vessels",
                     [](const PyFleet& self) {
                         const nb::gil_scoped_release release;
                         return self.fleet->num_vessels();
                     })
        .def_prop_ro("threads",
                     [](const PyFleet& self) {
                         const nb::gil_scoped_release release;
                         return self.fleet->threads();
                     })
        .def_prop_ro("prepared",
                     [](const PyFleet& self) {
                         const nb::gil_scoped_release release;
                         return self.fleet->prepared();
                     })
        .def_prop_ro("has_duals",
                     [](const PyFleet& self) {
                         const nb::gil_scoped_release release;
                         return self.fleet->has_duals();
                     })
        .def("reachable",
             [](const PyFleet& self) {
                 std::vector<std::uint8_t> flags;
                 {
                     const nb::gil_scoped_release release;
                     flags = self.fleet->reachable();
                 }
                 return to_numpy(std::move(flags));
             })
        // -> bytes of (shared graph, shared pricing data, vessels, bound
        //    cache, worker workspaces, result slots)
        .def("memory",
             [](const PyFleet& self) {
                 FleetMemory memory;
                 {
                     const nb::gil_scoped_release release;
                     memory = self.fleet->memory();
                 }
                 return nb::make_tuple(memory.shared_graph, memory.shared_pricing,
                                       memory.vessels, memory.bound_cache, memory.workspaces,
                                       memory.results);
             })
        .def(
            "debug_fail_vessel",
            [](const PyFleet& self, std::int64_t vessel) {
                self.fleet->debug_fail_vessel(vessel < 0 ? kInvalid
                                                         : static_cast<std::uint32_t>(vessel));
            },
            "vessel"_a);

    // -> (results, errors); results[i] is None where errors[i] is nonempty.
    m.def(
        "solve_batch",
        [](const std::vector<PyPricing*>& engines, const std::vector<InArray<double>>& duals,
           std::uint32_t max_paths, bool negative_only, double tolerance, unsigned threads) {
            std::vector<PricingEngine*> native;
            native.reserve(engines.size());
            for (const PyPricing* engine : engines) native.push_back(engine->engine.get());
            std::vector<std::vector<double>> copies;
            copies.reserve(duals.size());
            for (const InArray<double>& pi : duals) {
                copies.emplace_back(pi.data(), pi.data() + pi.shape(0));
            }
            const std::vector<std::span<const double>> spans(copies.begin(), copies.end());
            BatchOutcome outcome;
            {
                const nb::gil_scoped_release release;
                outcome = solve_batch(native, spans, max_paths, negative_only, tolerance, threads);
            }
            nb::list results;
            for (std::size_t i = 0; i < outcome.results.size(); ++i) {
                if (outcome.errors[i].empty()) {
                    results.append(pricing_result_tuple(std::move(outcome.results[i])));
                } else {
                    results.append(nb::none());
                }
            }
            return nb::make_tuple(results, outcome.errors);
        },
        "engines"_a, "duals"_a, "max_paths"_a, "negative_only"_a, "tolerance"_a, "threads"_a);
}
