# Examples

Five scripts, each a plain top-to-bottom program you can read in one sitting.
They build on each other, so reading them in order is a good way to learn the
library.

<div class="grid cards" markdown>

-   :material-truck-fast-outline:{ .lg .middle } **[Shortest path](shortest-path.md)**

    ---

    A truck from Lisbon to Berlin. States, edges, a negative cost, an
    unreachable pair and a cycle. 50 lines.

-   :material-timer-sand:{ .lg .middle } **[Resource limits](resource-limits.md)**

    ---

    The same trip with a cap on driving hours and toll legs. The cheapest
    route is no longer allowed. 55 lines.

-   :material-ferry:{ .lg .middle } **[Vessel routing](vessel-routing.md)**

    ---

    A trading vessel on a `(day, port, cargo)` network with a fuel tank, and a
    printed itinerary. 80 lines.

-   :material-table-column-plus-after:{ .lg .middle } **[Column generation](column-generation.md)**

    ---

    Three vessels, five contracts, a master LP in SciPy and a pricing engine
    per vessel. 150 lines.

-   :material-sail-boat:{ .lg .middle } **[Fleet pricing](fleet-pricing.md)**

    ---

    Twelve different vessels on one shared network, priced in one call.
    150 lines.

</div>

## Running them

The examples need only CelerisPath itself; SciPy, which solves the master
problems, is already a dependency.

```bash
git clone https://github.com/sharabhshukla/CelerisPath
cd CelerisPath
python examples/shortest_path.py
```

Every page shows the script and the output it prints.
