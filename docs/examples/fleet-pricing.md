# Fleet pricing

[Column generation](column-generation.md) builds one pricing engine per
vessel. This example plans twelve vessels that are all a little different:
each has its own home port, fuel tank and hire, and the largest ones cannot
enter Mumbai.

A `PricingFleet` stores the network once and each vessel as a small profile,
and prices all of them in one native call.

```bash
python examples/fleet_pricing.py
```

## The script

```python title="examples/fleet_pricing.py"
--8<-- "examples/fleet_pricing.py"
```

Things to notice:

* **One `add_vessels` call describes the whole fleet**, with one value per
  vessel for every argument: sources, targets, tanks, hire, convexity rows and
  forbidden edges.
* **The result is ready for the master.** `result.coefficient_matrix` and
  `result.original_costs` are appended as they are; no loop over columns.
* **`result.converged` is the stopping test.** It is true only when every
  vessel was priced exactly and none has an improving route.
* Every column knows its vessel through `route.vessel` and
  `route.vessel_index`.

## Output

```text
Network(num_vertices=240, num_edges=844, acyclic=True, resources=['fuel'])
PricingFleet(vessels=12, algorithm='rcsp', scheduler='tbb', threads=10, cost_update='shared_sparse', num_constraints=31)

round  columns  master cost  new routes  pricing
    1       19      7300.00          12    0.81 ms
    2       31      6399.00          12    0.35 ms
    3       43      5545.00          12    0.60 ms
    4       55      5348.50          12    0.24 ms
    5       67      5196.50          10    0.24 ms
    6       77      4953.00          10    0.20 ms
    7       87      4749.00           6    0.16 ms
    8       93      4689.00           6    0.26 ms
    9       99      4646.00           6    0.24 ms
   10      105      4646.00           6    0.26 ms
   11      111      4646.00           6    0.17 ms
   12      117      4646.00           2    0.19 ms
   13      119      4637.00           2    0.20 ms
   14      121      4637.00           0    0.20 ms

LP optimum: 4637.00 with 102 generated routes
Integer plan from these routes: 4637.00

nobody delivers to Rotterdam in days 19-21: penalty 500
nobody delivers to Rotterdam in days 22-24: penalty 500
nobody delivers to Rotterdam in days 25-27: penalty 500
vessel 03 from Dubai     cost  353, 24 of 26 days at sea, delivers to Rotterdam (day 17)
vessel 10 from Singapore cost  233, 14 of 14 days at sea, delivers to Singapore (day 25)
vessel 12 from Dubai     cost  353, 24 of 26 days at sea, delivers to Rotterdam (day 15)
vessel 04 from Singapore cost  233, 14 of 14 days at sea, delivers to Singapore (day 19)
vessel 08 from Mumbai    cost  268, 12 of 20 days at sea, delivers to Mumbai (day 10), Mumbai (day 18)
vessel 02 from Mumbai    cost  340, 20 of 20 days at sea, delivers to Singapore (day 15), Mumbai (day 27)
vessel 09 from Dubai     cost  263, 14 of 26 days at sea, delivers to Singapore (day 10)
vessel 07 from Dubai     cost  253, 12 of 14 days at sea, delivers to Mumbai (day 6), Mumbai (day 15)
vessel 11 from Mumbai    cost  340, 20 of 20 days at sea, delivers to Singapore (day 12), Mumbai (day 24)
vessel 05 from Mumbai    cost  268, 12 of 20 days at sea, delivers to Mumbai (day 9), Mumbai (day 21)
vessel 01 from Dubai     cost  233, 14 of 14 days at sea, delivers to Singapore (day 22)
```

The pricing column is the wall time of one `fleet.price` call for all twelve
vessels and varies from run to run. The LP optimum of 4637 is the same on
every machine; the number of rounds and routes on the way there depends on
which optimal duals the LP solver returns. The scheduler reads `tbb` on a build with
oneTBB and `threads` on the published wheels.

See [Fleet pricing](../guide/fleet.md) in the guide for everything a vessel
profile can hold.

[:octicons-mark-github-16: View on GitHub](https://github.com/sharabhshukla/CelerisRoute/blob/main/examples/fleet_pricing.py){ .md-button }
