# RDIQKD SDP Tools

Julia code for computing Receiver-Device-Independent QKD key-rate lower bounds with the min-entropy (`H_min`) and Brown–Fawzi–Fawzi (`BFF`) semidefinite-programming constructions.

The repository contains the SDP builder, higher-level scanning and plotting tools, and a runner with ready-to-use examples. The builder itself is kept separate from the tools so that scans and figures can be changed without modifying the SDP construction.

## Repository structure

- `RDIQKD Builder.jl` — constructs and solves the H_min, combined BFF, and per-node BFF SDPs.
- `RDIQKD Tools.jl` — evaluates individual rates, optimizes theta, scans eta values, creates heatmaps and plots, compares methods, and reads or writes CSV files.
- `Run RDIQKD.jl` — simple wrapper functions showing how to call both the builder and the tools.

## Installation

Julia 1.11 is recommended. Install the main dependencies and at least one SDP solver:

```julia
using Pkg
Pkg.add(["JuMP", "Plots", "FastGaussQuadrature"])
Pkg.add("MosekTools")  # requires Mosek and a valid licence
# Alternatively:
# Pkg.add("SCS")
```

The builder also supports solvers from `SDPAFamily.jl`. Select the solver near the top of `Run RDIQKD.jl`:

```julia
RUN_SOLVER = :mosek
# Alternatives include :scs, :sdpa_gmp, :sdpa_qd and :sdpa_dd.
```

## Quick start

Place the three `.jl` files in the same directory. Start Julia from that directory and run:

```julia
include("Run RDIQKD.jl")
```

The file only loads the functions; it does not automatically start a large scan.

Evaluate one H_min point:

```julia
point = run_rate_at_example(
    method=:hmin,
    n=3,
    theta=0.7,
    eta=0.8,
)
```

Evaluate one per-node BFF point:

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

The BFF mode can be changed from `:per_node` to `:combined`.

## Optimizing and scanning rates

Optimize theta for one value of eta:

```julia
optimum = run_optimize_theta_example(
    method=:hmin,
    n=3,
    eta=0.8,
)
```

Scan several values of `n` and `eta`, optimizing theta at every point:

```julia
hmin_scan = run_scan_rates_example(
    method=:hmin,
    ns=2:5,
    eta_points=20,
    csv_path="results/hmin.csv",
)
```

Create an eta-theta heatmap:

```julia
heatmap_result = run_rate_heatmap_example(
    method=:hmin,
    n=3,
    eta_points=15,
    theta_points=25,
    csv_path="results/hmin_heatmap.csv",
)
```

Progress is printed on one continuously updated line while the SDPs are solved.

## Plotting and comparisons

Plot a saved scan:

```julia
run_plot_rates_example(
    "results/hmin.csv",
    output_path="results/hmin_rates.pdf",
)
```

Compare H_min and BFF scans:

```julia
run_compare_rates_example(
    "results/hmin.csv",
    "results/bff.csv",
    labels=("H-min", "BFF"),
    output_path="results/comparison.pdf",
)
```

Scan CSV files contain only:

```text
n,eta,theta_opt,rate
```

Heatmap CSV files contain only:

```text
eta,theta,rate
```

## Custom settings

The main tool functions accept the preparation, noise, word set, solver, field, and BFF-node settings as keyword arguments. For example:

```julia
point = RDITools.rate_at(
    method=:hmin,
    preparation=:phase,
    n=3,
    theta=0.7,
    eta=0.8,
    lambda=0.01,
    words=["Id", "B", "E", "BE"],
    solver=RUN_SOLVER,
)
```



