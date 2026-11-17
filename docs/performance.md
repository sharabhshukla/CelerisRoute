# Performance

!!! abstract "In short"
    * A shortest path on a DAG with 100,000 vertices and 800,000 edges takes
      about 4 ms; a million vertices and 8 million edges take about 50 ms.
    * Re-pricing with new duals costs a sparse matrix product plus the solve.
      Python adds at most 20 microseconds.
    * A `PricingFleet` needs 6 to 12 times less memory than one graph per
      vessel and is ready 9 to 14 times sooner. Its DAG iterations are about
      1.4 times faster; its RCSP iterations are not faster.
    * Resource-constrained solves are as fast as their label count allows,
      and that count can explode. Read the RCSP table before you rely on it.

Every number on this page comes from a result file in
[`benchmarks/results/`](https://github.com/sharabhshukla/CelerisRoute/tree/main/benchmarks/results),
measured on
an Apple M4 (4 performance + 6 efficiency cores, 16 GB), macOS 27, Apple Clang
21, Boost 1.92, free-threaded CPython 3.14, Release build, seed 20240607. The
machine was a desktop in use (an IDE and a browser were running), and repeated
runs differed by up to 2x on individual rows, so read these as orders of
magnitude. Instances are layered random DAGs with integer costs in [-5, 20].

## Shortest path on a DAG

Median of repeated `network.solve` calls. All 22 instances match the NumPy
reference.

| vertices | edges | layers x width | compile | solve (Python-facing) |
|---:|---:|---:|---:|---:|
| 1,002 | 7,400 | 10 x 100 | 45 us | 14 us |
| 10,002 | 79,400 | 100 x 100 | 0.46 ms | 0.26 ms |
| 100,002 | 799,400 | 1,000 x 100 | 5.0 ms | 4.2 ms |
| 1,000,002 | 2,000,000 | 1,000 x 1,000 | 14 ms | 21 ms |
| 1,000,002 | 7,999,400 | 10,000 x 100 | 50 ms | 44 ms |
| 1,000,002 | 7,999,940 | 100,000 x 10 | 98 ms | 53 ms |

Python overhead per solve is about 3 us; throughput is roughly 100-300 million
edges per second at 10k vertices and above.

## Resource-constrained shortest path

Out-degree 4. `tight` places the bound between the least possible
consumption (0) and the consumption of the unconstrained optimum (1).
One-resource costs match an independent NumPy dynamic program.

| vertices | layers | resources | tight | labels created | solve |
|---:|---:|---:|---:|---:|---:|
| 1,002 | 10 | 1 | 0.25 | 23,509 | 3.4 ms |
| 1,002 | 10 | 2 | 0.75 | 110,125 | 42 ms |
| 992 | 30 | 1 | 0.75 | 240,918 | 77 ms |
| 992 | 30 | 2 | 0.50 | 1,483,104 | 4.3 s |
| 10,002 | 10 | 1 | 0.75 | 577,541 | 0.24 s |
| 10,002 | 10 | 2 | 0.25 | 15,958 | 2.8 ms |
| 10,002 | 10 | 2 | 0.75 | 1,602,906 | 1.4 s |
| 9,992 | 30 | 1 | 0.75 | 2,491,532 | 1.7 s |
| 100,002 | 10 | 1 | 0.25 | 1,809,837 | 1.3 s |
| 100,002 | 10 | 1 | 0.75 | 5,643,119 | 6.8 s |
| 100,002 | 10 | 2 | 0.25 | 64,109 | 21 ms |
| 100,002 | 10 | 2 | 0.75 | 12,928,445 | 19.5 s |

RCSP time follows the label count, not the graph size, and the label count
grows with path length and with loose bounds. These random instances, where
cheap edges consume more, are close to a worst case. Cost per label rises from
about 0.15 us to 3 us as the labels per vertex grow, because Boost compares
labels of a vertex pairwise.

## Repeated pricing

One engine, 200 master constraints, 2 matrix nonzeros per edge, new random
duals every solve. Medians.

| algorithm | vertices | edges | engine build | dual update | native solve | reconstruction | total from Python |
|---|---:|---:|---:|---:|---:|---:|---:|
| dag | 10,002 | 79,400 | 3.9 ms | 79 us | 0.34 ms | 3.4 us | 0.43 ms |
| dag | 100,002 | 794,000 | 23 ms | 0.78 ms | 4.6 ms | 9.0 us | 5.4 ms |
| dag | 1,000,002 | 3,998,000 | 146 ms | 3.9 ms | 35.7 ms | 65 us | 39.7 ms |
| rcsp | 1,002 | 3,800 | 0.19 ms | 4.0 us | 2.96 ms | 0.9 us | 2.97 ms |
| rcsp | 10,002 | 38,000 | 1.2 ms | 41 us | 45.0 ms | 2.9 us | 45.0 ms |

Graph construction and compilation for the same instances took 0.04-18 ms and
0.04-35 ms. The time seen from Python exceeds the native total by at most 20 us.

## Batch pricing

Median wall time. Serial is a Python loop over `engine.solve`. All batch
results equal the serial ones.

| engines | serial | 1 thread | 2 | 4 | 8 | 10 |
|---|---:|---:|---:|---:|---:|---:|
| 10 dag (10k vertices each) | 5.1 ms | 5.0 ms | 2.7 ms | 1.8 ms | 1.2 ms | 1.2 ms |
| 50 dag | 26.1 ms | 25.8 ms | 13.4 ms | 8.1 ms | 5.7 ms | 5.3 ms |
| 100 dag | 52.1 ms | 51.6 ms | 26.9 ms | 15.3 ms | 10.9 ms | 9.9 ms |
| 10 rcsp (1k vertices each) | 24.1 ms | 24.0 ms | 12.4 ms | 7.7 ms | 5.4 ms | 5.5 ms |
| 50 rcsp | 125 ms | 125 ms | 64 ms | 34 ms | 25 ms | 23 ms |
| 100 rcsp | 418 ms | 357 ms | 196 ms | 98 ms | 62 ms | 56 ms |

Speedup is about 2x on 2 threads, 2.9-4.3x on 4 and 4.2-7.5x on all 10 cores. Only
four of the ten cores are performance cores, which is why scaling flattens
after 4 threads. The last row was the least stable: over three runs its serial
time was 258, 418 and 566 ms.

## Fleet pricing

`benchmarks/benchmark_fleet.py`, seed 0, oneTBB 2023.2, measured with only a
browser open (load average 2.0 at the start). 125 vessels on a time-expanded
network of 125 ports: a vertex is (day, port, cargo, phase), arcs are port
operations, waiting and sailing to neighbouring ports, sailing burns fuel.
The master has one row per (port, day, product) and one convexity row per
vessel. Every vessel has its own start port, bunker capacity (25-40 % of the
horizon in sailing days), three ports it may not enter, discounted fees at
five ports, a charter cost and a convexity row.

| horizon | vertices | edges | master rows | overlay entries, 125 vessels |
|---|---:|---:|---:|---:|
| 30 days | 22,501 | 87,584 | 7,625 | 204,369 |
| 60 days | 45,001 | 182,258 | 15,125 | 430,953 |

Two designs solve the same subproblems: one `PricingFleet` on the shared
network, and one network plus one `PricingEngine` per vessel behind a
`BatchPricingEngine` ("graph per vessel"). `price` is one complete iteration
as Python sees it: dual update, scheduling, all solves, route reconstruction
and the Python-facing result. `+ columns` turns the result into column objects
and a sparse coefficient matrix. `setup` builds the networks and the solver
once. Peak RSS is that of a fresh process per row and includes the interpreter,
NumPy, SciPy and the instance arrays (72 MB and 85 MB on their own). Medians
of 5 iterations with new duals each, after one warm-up iteration, 10 threads:

| horizon | algorithm | design | price | + columns | total | setup | peak RSS |
|---|---|---|---:|---:|---:|---:|---:|
| 30 days | rcsp | fleet, oneTBB, `indexed` | 22.9 ms | 173 us | 23.1 ms | 29.7 ms | 124 MB |
| 30 days | rcsp | fleet, oneTBB, `shared_sparse` | 23.1 ms | 177 us | 23.3 ms | 27.9 ms | 124 MB |
| 30 days | rcsp | fleet, oneTBB, `per_vessel` | 24.1 ms | 171 us | 24.3 ms | 28.2 ms | 122 MB |
| 30 days | rcsp | fleet, threads, `indexed` | 22.9 ms | 178 us | 23.1 ms | 27.9 ms | 122 MB |
| 30 days | rcsp | graph per vessel | 24.7 ms | 842 us | 25.6 ms | 302 ms | 975 MB |
| 30 days | dag | fleet, oneTBB, `indexed` | 3.8 ms | 151 us | 4.0 ms | 27.8 ms | 104 MB |
| 30 days | dag | fleet, oneTBB, `shared_sparse` | 3.8 ms | 152 us | 3.9 ms | 28.3 ms | 102 MB |
| 30 days | dag | fleet, oneTBB, `per_vessel` | 4.7 ms | 162 us | 4.8 ms | 28.1 ms | 101 MB |
| 30 days | dag | fleet, threads, `indexed` | 3.8 ms | 167 us | 3.9 ms | 27.8 ms | 101 MB |
| 30 days | dag | graph per vessel | 4.6 ms | 980 us | 5.5 ms | 260 ms | 834 MB |
| 60 days | rcsp | fleet, oneTBB, `indexed` | 1.32 s | 230 us | 1.32 s | 52.9 ms | 418 MB |
| 60 days | rcsp | fleet, oneTBB, `shared_sparse` | 1.31 s | 213 us | 1.31 s | 51.3 ms | 435 MB |
| 60 days | rcsp | fleet, oneTBB, `per_vessel` | 1.33 s | 233 us | 1.34 s | 50.1 ms | 397 MB |
| 60 days | rcsp | fleet, threads, `indexed` | 1.39 s | 226 us | 1.39 s | 49.4 ms | 444 MB |
| 60 days | rcsp | graph per vessel | 1.19 s | 1.1 ms | 1.19 s | 737 ms | 2,616 MB |
| 60 days | dag | fleet, oneTBB, `indexed` | 14.1 ms | 215 us | 14.3 ms | 49.0 ms | 159 MB |
| 60 days | dag | fleet, oneTBB, `shared_sparse` | 14.9 ms | 219 us | 15.1 ms | 49.9 ms | 160 MB |
| 60 days | dag | fleet, oneTBB, `per_vessel` | 16.1 ms | 220 us | 16.4 ms | 48.7 ms | 158 MB |
| 60 days | dag | fleet, threads, `indexed` | 14.0 ms | 207 us | 14.2 ms | 48.6 ms | 159 MB |
| 60 days | dag | graph per vessel | 19.7 ms | 1.0 ms | 20.7 ms | 589 ms | 1,970 MB |

Every row returned the same statuses and best reduced costs: bit-identical
among the fleet configurations, and within 1e-9 relative (1e-7 absolute)
between the fleet and the per-vessel graphs, which add the same numbers in
another order.

### What the numbers say

* **Memory and setup are where sharing pays.** The fleet needs 6 to 12 times
  less memory (124 MB against 975 MB, 418 MB against 2.6 GB) and is ready 9
  to 14 times sooner. Its native memory for 30 days is 4.8 MB of shared graph,
  2.2 MB of shared pricing data, 3.3 MB for all 125 vessel profiles and
  26 MB of worker workspaces. For 60-day RCSP the workspaces hold 334 MB of
  labels, which is most of that row's peak.
* **RCSP iterations are not faster.** The label work dominates (1.1 million
  labels per iteration at 30 days, 25 million at 60), and it is the same work
  in both designs. At 30 days the fleet was 7 % and 10 % faster in two runs
  on a quiet machine; at 60 days the per-vessel graphs were 4 % and 10 %
  faster.
* **DAG iterations are about 1.4 times faster** with the fleet (4.0 ms against
  5.5 ms, 14.3 ms against 20.7 ms).
* **Shared reduced-cost preprocessing** saves 13-17 % of a DAG iteration
  against `per_vessel` and 1-5 % of an RCSP iteration, which is inside the
  noise. `indexed` and `shared_sparse` are equally fast here.
* **oneTBB and plain threads are equally fast** on this instance; the largest
  gap, 5 % at 60-day RCSP, is inside the noise.
* **Result transfer is cheap.** Building 125 column objects and the
  coefficient matrix takes 0.15-0.23 ms from a fleet result and 0.84-1.1 ms
  from 125 engine results.

### Stages of one iteration

oneTBB, 10 threads. `Python` is the time `price` takes beyond the native
stages.

| horizon | algorithm | cost update | dual update | parallel solve | assemble | Python | columns |
|---|---|---|---:|---:|---:|---:|---:|
| 30 days | rcsp | `indexed` | 51 us | 22.8 ms | 16 us | 77 us | 173 us |
| 30 days | rcsp | `per_vessel` | 3 us | 24.0 ms | 16 us | 70 us | 171 us |
| 30 days | dag | `indexed` | 37 us | 3.7 ms | 11 us | 32 us | 151 us |
| 30 days | dag | `per_vessel` | 3 us | 4.6 ms | 11 us | 34 us | 162 us |
| 60 days | rcsp | `indexed` | 115 us | 1.32 s | 24 us | 90 us | 230 us |
| 60 days | rcsp | `per_vessel` | 6 us | 1.33 s | 24 us | 90 us | 233 us |
| 60 days | dag | `indexed` | 104 us | 13.9 ms | 23 us | 77 us | 215 us |
| 60 days | dag | `per_vessel` | 6 us | 16.0 ms | 21 us | 79 us | 220 us |

With `per_vessel` the dual update only stores the duals; every vessel then
reduces the whole cost array inside the parallel solve.

### Thread count

rcsp, 125 vessels; total time / peak RSS.

| threads | 30 days: fleet | graph per vessel | 60 days: fleet | graph per vessel |
|---:|---:|---:|---:|---:|
| 1 | 129 ms / 96 MB | 133 ms / 1,001 MB | 4.75 s / 157 MB | 4.75 s / 2,224 MB |
| 2 | 67.7 ms / 97 MB | 69.0 ms / 1,002 MB | 2.76 s / 175 MB | 2.64 s / 2,377 MB |
| 4 | 34.0 ms / 100 MB | 36.4 ms / 1,003 MB | 1.65 s / 226 MB | 1.56 s / 2,504 MB |
| 8 | 25.7 ms / 118 MB | 28.0 ms / 1,007 MB | 1.32 s / 344 MB | 1.23 s / 2,702 MB |
| 10 | 23.1 ms / 124 MB | 25.6 ms / 975 MB | 1.32 s / 418 MB | 1.19 s / 2,616 MB |

The fleet speeds up 1.9x, 3.8x, 5.0x and 5.6x on 2, 4, 8 and 10 threads at 30
days, and 1.7x, 2.9x, 3.6x and 3.6x at 60 days. Four of the ten cores are
performance cores. The 60-day case falls short of linear already on 2 threads;
a profile at 10 threads shows about three quarters of the samples inside the
RCSP label loop and under a tenth in the allocator, so the limit is the label
work itself, which both designs share. Fleet memory grows with the thread
count because every worker keeps its own label workspace.

### Vessel count

rcsp, 10 threads; total time / peak RSS.

| vessels | 30 days: fleet | vessels/s | graph per vessel | 60 days: fleet | vessels/s | graph per vessel |
|---:|---:|---:|---:|---:|---:|---:|
| 25 | 5.5 ms / 108 MB | 4,553 | 5.6 ms / 260 MB | 289 ms / 330 MB | 86 | 265 ms / 793 MB |
| 125 | 23.1 ms / 124 MB | 5,415 | 25.6 ms / 975 MB | 1.32 s / 418 MB | 94 | 1.19 s / 2,616 MB |
| 500 | 93.5 ms / 195 MB | 5,348 | not run | 4.41 s / 502 MB | 114 | not run |

The per-vessel design was not run at 500 vessels: at the 7 MB (30 days) and
20 MB (60 days) per vessel it needs above, it would take roughly 3.5 GB and
10 GB on a 16 GB machine. Two complete runs on a quiet machine agreed within
about 10 % on every row quoted here; the tables are from the second.

## Reproducing

```bash
python benchmarks/benchmark_dag.py        # about 10 s
python benchmarks/benchmark_rcsp.py       # about 2.5 min; --quick: 35 s
python benchmarks/benchmark_pricing.py    # about 25 s
python benchmarks/benchmark_fleet.py      # about 3.5 min, peaks at 2.7 GB; --quick: 5 s
```

Each script takes `--seed`, `--quick` and `--output`, validates its answers
against an independent NumPy dynamic program where one exists, and writes the
configuration, the environment and all raw timings to
`benchmarks/results/<name>.json`.

A native harness times the C++ core without Python:

```bash
cmake -S . -B build/bench -DCELERISROUTE_BUILD_PYTHON=OFF -DCELERISROUTE_BUILD_BENCHMARKS=ON
cmake --build build/bench
./build/bench/celerisroute_bench
```
