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

For large scans, first test a single point with the intended word set and solver. BFF calculations, especially combined-node SDPs, can require substantially more memory and solving time than H_min calculations.


````markdown
## Using custom preparation states

Each qubit preparation has the form

\[
|\psi_x\rangle
=
\alpha_x |0\rangle
+
\beta_x |1\rangle.
\]

There is one value of `alpha` and one value of `beta` for each prepared state.

For example, when `n=3`, Alice has three preparations:

\[
\begin{aligned}
|\psi_0\rangle &= \alpha_0|0\rangle+\beta_0|1\rangle,\\
|\psi_1\rangle &= \alpha_1|0\rangle+\beta_1|1\rangle,\\
|\psi_2\rangle &= \alpha_2|0\rangle+\beta_2|1\rangle.
\end{aligned}
\]

Therefore, three values of `alpha` and three values of `beta` are required:

```julia
alphas = [
    1.0,
    cos(0.5),
    cos(1.0),
]

betas = [
    0.0,
    sin(0.5),
    sin(1.0),
]
```

These coefficients can be passed directly to the builder:

```julia
result = RDI.build_hmin_sdp(
    alphas=alphas,
    betas=betas,
    eta=0.9,
    lambda=0.01,
    word_blocks=["Id", "B", "E", "BE"],
    solver=RUN_SOLVER,
)
```

The builder normalizes each state automatically.

Complex coefficients are also accepted:

```julia
alphas = ComplexF64[
    1.0,
    1 / sqrt(2),
    1 / sqrt(2),
]

betas = ComplexF64[
    0.0,
    1 / sqrt(2),
    im / sqrt(2),
]
```

## Using an amplitude matrix

The states can also be supplied as a `2 × n` amplitude matrix.

Each column represents one preparation:

```julia
amplitudes = ComplexF64[
    1.0   1/sqrt(2)   1/sqrt(2)
    0.0   1/sqrt(2)   im/sqrt(2)
]
```

This matrix contains three states because it has three columns.

Use it with:

```julia
result = RDI.build_hmin_sdp(
    amplitudes=amplitudes,
    eta=0.9,
    lambda=0.01,
    word_blocks=["Id", "B", "E", "BE"],
    solver=RUN_SOLVER,
)
```

For `n` qubit preparations, the required matrix size is:

```julia
size(amplitudes) == (2, n)
```

## Built-in preparations

For the built-in phase preparation, use:

```julia
amplitudes = RDI.phase_amplitudes(n, theta)
```

The phase-encoded states are

\[
|\psi_x\rangle
=
\cos\left(\frac{\theta}{2}\right)|0\rangle
+
e^{-2\pi i x/n}
\sin\left(\frac{\theta}{2}\right)|1\rangle.
\]

For the built-in real X-Z preparation, use:

```julia
amplitudes = RDI.realxz_amplitudes(n, theta)
```

## Custom preparations with the tools file

The tools file accepts:

- `preparation=:phase`;
- `preparation=:real`;
- a custom function of `(n, theta)`.

For example:

```julia
function my_preparation(n, theta)
    n == 3 || error("This example preparation requires n=3")

    alphas = [
        1.0,
        cos(theta / 2),
        cos(theta),
    ]

    betas = [
        0.0,
        sin(theta / 2),
        sin(theta),
    ]

    return RDI.amplitudes_from_alphas_betas(
        alphas,
        betas;
        normalize=true,
    )
end
```

Use the custom preparation in one rate calculation:

```julia
point = RDITools.rate_at(
    method=:hmin,
    preparation=my_preparation,
    n=3,
    theta=0.7,
    eta=0.8,
    lambda=0.01,
)
```

The same preparation can be used in a scan:

```julia
scan = RDITools.scan_rates(
    method=:hmin,
    preparation=my_preparation,
    ns=[3],
    eta_points=10,
)
```

The preparation function must return exactly `n` qubit states.

## Using custom observed statistics

The solver can accept a custom matrix of observed statistics.

The entries are

\[
P0[x,y]=p(b=0\mid x,y),
\]

where:

- the row index `x` identifies Alice's preparation;
- the column index `y` identifies Bob's input.

For `n=3`, the matrix must have size `3 × 3`:

```julia
P0_custom = [
    0.01  0.20  0.21
    0.19  0.01  0.22
    0.20  0.21  0.01
]
```

Use the custom statistics with the H_min builder:

```julia
result = RDI.build_hmin_sdp(
    amplitudes=amplitudes,
    P0=P0_custom,
    eta=0.9,
    lambda=0.01,
    word_blocks=["Id", "B", "E", "BE"],
    solver=RUN_SOLVER,
)
```

Use them with the combined BFF builder:

```julia
result = RDI.build_bff_sdp(
    amplitudes=amplitudes,
    P0=P0_custom,
    eta=0.9,
    lambda=0.01,
    word_blocks=[
        "Id",
        "B",
        "Z",
        "Zdag",
        "BZ",
        "BZdag",
    ],
    bff_n_nodes=4,
    solver=RUN_SOLVER,
)
```

Use them with the per-node BFF builder:

```julia
result = RDI.build_bff_per_node_sdp(
    amplitudes=amplitudes,
    P0=P0_custom,
    eta=0.9,
    lambda=0.01,
    bff_n_nodes=4,
    solver=RUN_SOLVER,
)
```

## Custom statistics with the tools file

Additional builder arguments can be passed through `RDITools.rate_at`:

```julia
point = RDITools.rate_at(
    method=:hmin,
    preparation=my_preparation,
    n=3,
    theta=0.7,
    eta=0.9,
    lambda=0.01,
    P0=P0_custom,
)
```

They can also be passed through the runner:

```julia
point = run_rate_at_example(
    method=:hmin,
    preparation=my_preparation,
    n=3,
    theta=0.7,
    eta=0.9,
    lambda=0.01,
    P0=P0_custom,
)
```

The statistics and preparation coefficients are separate inputs.

In this example:

- `my_preparation` defines the states;
- `P0_custom` defines the observed probabilities;
- changing `P0_custom` does not change `alpha` or `beta`.

## Generating statistics with a function

Instead of providing a fixed matrix, a function can generate the statistics from the current preparation matrix:

```julia
function my_statistics(amplitudes)
    n = size(amplitudes, 2)
    P0 = zeros(Float64, n, n)

    for x in 1:n
        for y in 1:n
            if x == y
                P0[x, y] = 0.01
            else
                P0[x, y] = 0.20
            end
        end
    end

    return P0
end
```

Use it with:

```julia
result = RDI.build_hmin_sdp(
    amplitudes=amplitudes,
    statistics_function=my_statistics,
    eta=0.9,
    lambda=0.01,
    solver=RUN_SOLVER,
)
```

It can also be passed through the tools file:

```julia
point = RDITools.rate_at(
    method=:hmin,
    preparation=my_preparation,
    n=3,
    theta=0.7,
    eta=0.9,
    lambda=0.01,
    statistics_function=my_statistics,
)
```

The builder uses the following priority:

1. use `P0` when it is provided;
2. otherwise use `statistics_function`;
3. otherwise calculate the default statistics from the preparation, `eta`, and `lambda`.

## Providing custom states and statistics together

Custom preparation coefficients and custom statistics can be supplied in the same call:

```julia
alphas = [
    1.0,
    cos(0.5),
    cos(1.0),
]

betas = [
    0.0,
    sin(0.5),
    sin(1.0),
]

P0_custom = [
    0.01  0.20  0.21
    0.19  0.01  0.22
    0.20  0.21  0.01
]

result = RDI.build_hmin_sdp(
    alphas=alphas,
    betas=betas,
    P0=P0_custom,
    eta=0.9,
    lambda=0.01,
    word_blocks=["Id", "B", "E", "BE"],
    solver=RUN_SOLVER,
)
```

Here:

- `alphas` and `betas` define the three preparation states;
- `P0_custom` defines the observed statistics;
- changing `P0_custom` does not change the preparation coefficients.

## Important consistency requirement

In the current builder, Bob's qubit operators are fixed internally to the noisy exclusion model

\[
B_y
=
\eta\left[
(1-\lambda)
\left(
I-|\psi_y\rangle\langle\psi_y|
\right)
+
\frac{\lambda}{2}I
\right].
\]

The builder also imposes the statistics constraint

\[
\langle\psi_x|B_y|\psi_x\rangle
=
P0[x,y].
\]

The supplied statistics must therefore be consistent with:

- the preparation states;
- `eta`;
- `lambda`;
- the internally fixed Bob operators.

If these inputs are inconsistent, the SDP will normally return an infeasible status.

The current builder therefore accepts custom statistics, but they cannot be completely arbitrary.

Supporting arbitrary experimental statistics would require changing the builder so that Bob's complete matrix blocks are not fixed internally.

## Success probability and QBER

By default, the success probability and QBER are calculated from `P0`.

They can also be supplied manually:

```julia
result = RDI.build_hmin_sdp(
    amplitudes=amplitudes,
    P0=P0_custom,
    psucc=0.12,
    qber=0.02,
    eta=0.9,
    lambda=0.01,
    solver=RUN_SOLVER,
)
```

It is normally safer to let the builder calculate these quantities from the supplied statistics unless a different protocol convention is intentionally being used.

## Custom statistics in theta scans

Care is required when optimizing or scanning over `theta`.

During a theta scan, the preparation states change for every tested value of `theta`. A fixed `P0_custom` matrix does not change and may become inconsistent with the new preparation.

For theta scans, use one of the following:

- the default statistics generated from each preparation;
- a `statistics_function` that generates statistics for the current preparation;
- fixed experimental statistics only for one fixed preparation.

For a single experimental data set, a direct call to `RDITools.rate_at` or directly to the builder is normally more appropriate than optimizing `theta`.

## Basic input checks

Before running the solver, verify:

```julia
length(alphas) == n
length(betas) == n
size(P0_custom) == (n, n)
```

For an amplitude matrix, verify:

```julia
size(amplitudes) == (2, n)
```

Each column of the amplitude matrix represents one qubit preparation.
````

