# RDIQKD SDP Tools

Julia tools for computing Receiver-Device-Independent QKD key-rate bounds with the min-entropy (`H_min`) and Brown–Fawzi–Fawzi (`BFF`) SDP constructions.

## Files

- `RDIQKD Builder.jl`: constructs and solves the H_min and BFF SDPs.
- `RDIQKD Tools.jl`: theta optimization, eta scans, heatmaps, plotting, comparisons, CSV export and progress printing.
- `Run RDIQKD.jl`: ready-to-run examples for all main functions.

## Installation

Julia 1.11 is recommended. Install the core packages and at least one solver:

```julia
using Pkg
Pkg.add(["JuMP", "Plots", "FastGaussQuadrature"])
Pkg.add("MosekTools")  # requires a Mosek installation and licence
# or: Pkg.add("SCS")
```

Set `RUN_SOLVER = :mosek` or `:scs` near the top of `Run RDIQKD.jl`.

## Quick start

From the repository directory, start Julia and run:

```julia
include("Run RDIQKD.jl")

point = run_rate_at_example(
    method=:hmin,
    n=3,
    theta=0.7,
    eta=0.8,
)
```

For a per-node BFF evaluation:

```julia
bff_point = run_rate_at_example(
    method=:bff,
    bff_mode=:per_node,
    bff_n_nodes=4,
    n=2,
    theta=0.8,
    eta=0.9,
)
```

## Scans and plots

```julia
hmin_scan = run_scan_rates_example(
    method=:hmin,
    ns=2:5,
    eta_points=20,
    csv_path="results/hmin.csv",
)

heatmap_result = run_rate_heatmap_example(
    method=:hmin,
    n=3,
    eta_points=15,
    theta_points=25,
)
```

Compare two saved scans with:

```julia
run_compare_rates_example(
    "results/hmin.csv",
    "results/bff.csv",
    labels=("H-min", "BFF"),
)
```

Scan CSV files contain only `n, eta, theta_opt, rate`. Heatmap CSV files contain only `eta, theta, rate`.
