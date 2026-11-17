# Column generation

A complete column-generation loop in about 150 lines. Three vessels have to
serve five delivery contracts, and a contract that nobody serves costs a
penalty. The master problem chooses routes; CelerisRoute invents them.

Each vessel gets a `PricingEngine`, and a `BatchPricingEngine` prices all of
them in one call. The master is solved with SciPy's HiGHS interface, so the
example needs nothing beyond the library's own dependencies.

```bash
python examples/column_generation.py
```

## The script

```python title="examples/column_generation.py"
--8<-- "examples/column_generation.py"
```

Things to notice:

* **The dual matrix is built while the network is built.** When a delivery
  edge is added, its edge id goes into the row of the contract it serves.
* **The convexity row is a fixed coefficient.** Every route of vessel `v` has
  a -1 in that vessel's row, passed as `fixed_coefficients`.
* **The loop is short.** Solve the master, read the duals, price, append the
  new columns. It stops when no vessel has a route with negative reduced cost.
* **SciPy's sign convention.** `linprog` wants `<=` rows, so the model's
  `>=` rows are negated on the way in and their marginals on the way out. See
  [Dual signs](../guide/pricing.md#dual-signs).
* The final integer plan is a small `milp` over the generated columns. That
  is a heuristic in general; here it reaches the LP bound, which proves it
  optimal.

## Output

```text
round  columns  master cost  new routes
    1        5      1900.00           3
    2        8      1485.00           3
    3       11      1070.00           2
    4       13      1061.00           2
    5       15      1061.00           0

LP optimum: 1061.00 with 10 generated routes
Integer plan: 1061.00

Nobody delivers to Mumbai in days 4-8: penalty 300
Calypso: cost 193, 14 days at sea
  day 12  sail Singapore -> Dubai (empty), arrive on day 19
  day 19  load in Dubai
  day 20  sail Dubai -> Singapore (full), arrive on day 27
  day 27  deliver in Singapore
Borealis: cost 285, 20 days at sea
  day  2  load in Dubai
  day  3  sail Dubai -> Singapore (full), arrive on day 10
  day 10  deliver in Singapore
  day 11  sail Singapore -> Dubai (empty), arrive on day 18
  day 18  load in Dubai
  day 19  sail Dubai -> Mumbai (full), arrive on day 22
  day 22  deliver in Mumbai
  day 27  sail Mumbai -> Dubai (empty), arrive on day 30
Aurora: cost 283, 24 days at sea
  day  4  load in Dubai
  day  5  sail Dubai -> Rotterdam (full), arrive on day 17
  day 17  deliver in Rotterdam
  day 18  sail Rotterdam -> Dubai (empty), arrive on day 30
```

The master cost falls as better routes arrive and the last round finds none,
which certifies the LP optimum. The integer plan has the same value.

The LP optimum of 1061 is the same on every machine. The rounds in between
can differ with the SciPy version, because an LP usually has several optimal
dual solutions and the solver may return any of them.

[:octicons-mark-github-16: View on GitHub](https://github.com/sharabhshukla/CelerisRoute/blob/main/examples/column_generation.py){ .md-button }
