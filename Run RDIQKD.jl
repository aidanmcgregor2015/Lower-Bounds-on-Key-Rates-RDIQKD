function find_rdiqkd_file(preferred_name::AbstractString, prefix::AbstractString)
    preferred_path = joinpath(@__DIR__, preferred_name)
    isfile(preferred_path) && return preferred_path

    matches = sort(filter(
        name -> startswith(name, prefix) && endswith(lowercase(name), ".jl"),
        readdir(@__DIR__),
    ))
    isempty(matches) && error(
        "Could not find $(preferred_name). Put it in the same folder as this runner.",
    )
    return joinpath(@__DIR__, first(matches))
end

builder_file = find_rdiqkd_file("RDIQKD Builder.jl", "RDIQKD Builder")
tools_file   = find_rdiqkd_file("RDIQKD Tools.jl", "RDIQKD Tools")

if !isdefined(Main, :RDIQKDSDPBuilderGeneralWordsPerNodeV2MarginalBFF)
    include(builder_file)
end
if !isdefined(Main, :RDIQKDTools)
    include(tools_file)
end

RDI = Main.RDIQKDSDPBuilderGeneralWordsPerNodeV2MarginalBFF
RDITools = Main.RDIQKDTools

println("Loaded builder version: ", RDI.BUILDER_FIXED_VERSION)
println("Builder file: ", builder_file)
println("Tools file:   ", tools_file)

# Change this once here to use another solver in the examples.
RUN_SOLVER = :mosek  # alternatives supported by the builder: :scs, :sdpa_gmp, ...

# =============================================================================
# Preparation helper
# =============================================================================

"Phase preparation accepted by both the builder and the tools file."
function simple_phase_preparation(n::Int, theta::Real)
    return RDI.phase_amplitudes(n, theta)
end

# =============================================================================
# Direct builder examples (builder unchanged)
# =============================================================================

function run_hmin_builder_example(;
    n::Int=2,
    theta::Real=1.0,
    eta::Real=0.9,
    lambda::Real=0.01,
    solver::Symbol=RUN_SOLVER,
    words=["Id", "B", "E", "BE"],
    silent::Bool=true,
)
    result = RDI.build_hmin_sdp(
        amplitudes=simple_phase_preparation(n, theta),
        eta=eta,
        lambda=lambda,
        word_blocks=words,
        solver=solver,
        silent=silent,
    )

    println("\n=== Direct builder: H_min ===")
    RDITools.print_result(result)
    return result
end

function run_bff_builder_example(;
    n::Int=2,
    theta::Real=1.0,
    eta::Real=0.9,
    lambda::Real=0.01,
    solver::Symbol=RUN_SOLVER,
    bff_n_nodes::Int=4,
    words=["Id", "B", "Z", "Zdag", "BZ", "BZdag"],
    silent::Bool=true,
)
    result = RDI.build_bff_sdp(
        amplitudes=simple_phase_preparation(n, theta),
        eta=eta,
        lambda=lambda,
        word_blocks=words,
        bff_n_nodes=bff_n_nodes,
        solver=solver,
        silent=silent,
    )

    println("\n=== Direct builder: combined BFF ===")
    RDITools.print_result(result)
    return result
end

function run_bff_per_node_builder_example(;
    n::Int=2,
    theta::Real=1.0,
    eta::Real=0.9,
    lambda::Real=0.01,
    solver::Symbol=RUN_SOLVER,
    bff_n_nodes::Int=4,
    words=["Id", "B", "Z", "Zdag", "BZ", "BZdag"],
    silent::Bool=true,
)
    result = RDI.build_bff_per_node_sdp(
        amplitudes=simple_phase_preparation(n, theta),
        eta=eta,
        lambda=lambda,
        word_blocks=words,
        bff_n_nodes=bff_n_nodes,
        solver=solver,
        silent=silent,
    )

    println("\n=== Direct builder: per-node BFF ===")
    RDITools.print_result(result)
    return result
end

# =============================================================================
# Tools examples
# =============================================================================

"Solve one H_min or BFF point with RDIQKDTools.rate_at."
function run_rate_at_example(;
    method::Symbol=:hmin,
    n::Int=2,
    theta::Real=1.0,
    eta::Real=0.9,
    lambda::Real=0.01,
    preparation=:phase,
    words=nothing,
    solver::Symbol=RUN_SOLVER,
    field::Symbol=:auto,
    bff_mode::Symbol=:per_node,
    bff_n_nodes::Int=4,
    certified::Bool=true,
    silent::Bool=true,
    builder_kwargs...,
)
    point = RDITools.rate_at(
        method=method,
        n=n,
        theta=theta,
        eta=eta,
        lambda=lambda,
        preparation=preparation,
        words=words,
        solver=solver,
        field=field,
        bff_mode=bff_mode,
        bff_n_nodes=bff_n_nodes,
        certified=certified,
        silent=silent,
        builder_kwargs...,
    )

    println("\n=== Tools: one rate point ===")
    println("method   = ", point.method)
    println("BFF mode = ", point.bff_mode)
    println("n         = ", point.n)
    println("eta       = ", point.eta)
    println("theta     = ", point.theta)
    println("rate      = ", point.rate)
    println("raw rate  = ", point.raw_rate)
    println("status    = ", point.status)
    RDITools.print_result(point.result; prefix="  ")
    return point
end

"Optimize theta for one fixed (n, eta) pair."
function run_optimize_theta_example(;
    method::Symbol=:hmin,
    n::Int=2,
    eta::Real=0.9,
    lambda::Real=0.01,
    preparation=:phase,
    words=nothing,
    solver::Symbol=RUN_SOLVER,
    field::Symbol=:auto,
    bff_mode::Symbol=:per_node,
    bff_n_nodes::Int=4,
    theta_range=(1.0e-3, pi / 2 - 1.0e-3),
    coarse_points::Int=15,
    refine_points::Int=11,
    golden_steps::Int=6,
    progress::Bool=true,
    silent::Bool=true,
    builder_kwargs...,
)
    optimum = RDITools.optimize_theta(
        method=method,
        n=n,
        eta=eta,
        lambda=lambda,
        preparation=preparation,
        words=words,
        solver=solver,
        field=field,
        bff_mode=bff_mode,
        bff_n_nodes=bff_n_nodes,
        theta_range=theta_range,
        coarse_points=coarse_points,
        refine_points=refine_points,
        golden_steps=golden_steps,
        progress=progress,
        silent=silent,
        builder_kwargs...,
    )

    println("\n=== Tools: optimized theta ===")
    println("theta*      = ", optimum.theta_opt)
    println("rate        = ", optimum.rate)
    println("raw rate    = ", optimum.raw_rate)
    println("status      = ", optimum.status)
    println("evaluations = ", optimum.evaluations)
    return optimum
end

"Scan eta values and optimize theta at every point."
function run_scan_rates_example(;
    method::Symbol=:hmin,
    ns=2:4,
    etas=nothing,
    eta_points::Int=6,
    eta_pad::Real=5.0e-3,
    lambda::Real=0.01,
    preparation=:phase,
    words=nothing,
    solver::Symbol=RUN_SOLVER,
    field::Symbol=:auto,
    bff_mode::Symbol=:per_node,
    bff_n_nodes::Int=4,
    coarse_points::Int=11,
    refine_points::Int=9,
    golden_steps::Int=4,
    progress::Bool=true,
    silent::Bool=true,
    csv_path::Union{Nothing,AbstractString}=joinpath(@__DIR__, "results", "rate_scan.csv"),
    label::Union{Nothing,AbstractString}=nothing,
    builder_kwargs...,
)
    scan = RDITools.scan_rates(
        method=method,
        ns=ns,
        etas=etas,
        eta_points=eta_points,
        eta_pad=eta_pad,
        lambda=lambda,
        preparation=preparation,
        words=words,
        solver=solver,
        field=field,
        bff_mode=bff_mode,
        bff_n_nodes=bff_n_nodes,
        coarse_points=coarse_points,
        refine_points=refine_points,
        golden_steps=golden_steps,
        progress=progress,
        silent=silent,
        csv_path=csv_path,
        label=label,
        builder_kwargs...,
    )

    println("\n=== Tools: rate scan ===")
    println("optimized points = ", length(scan.rate))
    csv_path !== nothing && println("CSV              = ", csv_path)
    return scan
end

"Build an eta-theta heatmap and save only eta, theta, rate in its CSV."
function run_rate_heatmap_example(;
    method::Symbol=:hmin,
    n::Int=2,
    etas=nothing,
    thetas=nothing,
    eta_points::Int=6,
    theta_points::Int=9,
    lambda::Real=0.01,
    preparation=:phase,
    words=nothing,
    solver::Symbol=RUN_SOLVER,
    field::Symbol=:auto,
    bff_mode::Symbol=:per_node,
    bff_n_nodes::Int=4,
    normalize_columns::Bool=false,
    log_color::Bool=true,
    progress::Bool=true,
    silent::Bool=true,
    show_plot::Bool=true,
    csv_path::Union{Nothing,AbstractString}=joinpath(@__DIR__, "results", "rate_heatmap.csv"),
    plot_path::Union{Nothing,AbstractString}=joinpath(@__DIR__, "results", "rate_heatmap.pdf"),
    builder_kwargs...,
)
    heatmap_result = RDITools.rate_heatmap(
        method=method,
        n=n,
        etas=etas,
        thetas=thetas,
        eta_points=eta_points,
        theta_points=theta_points,
        lambda=lambda,
        preparation=preparation,
        words=words,
        solver=solver,
        field=field,
        bff_mode=bff_mode,
        bff_n_nodes=bff_n_nodes,
        normalize_columns=normalize_columns,
        log_color=log_color,
        progress=progress,
        silent=silent,
        show_plot=show_plot,
        csv_path=csv_path,
        plot_path=plot_path,
        builder_kwargs...,
    )

    println("\n=== Tools: heatmap ===")
    println("grid size = ", size(heatmap_result.data.rate))
    println("CSV       = ", heatmap_result.csv_path)
    println("plot      = ", heatmap_result.plot_path)
    return heatmap_result
end

"Plot several n values from either a RateScan or a scan CSV path."
function run_plot_rates_example(
    scan_or_csv;
    ns=nothing,
    output_path::Union{Nothing,AbstractString}=joinpath(@__DIR__, "results", "rates.pdf"),
    show_plot::Bool=true,
    title=nothing,
)
    return RDITools.plot_rates(
        scan_or_csv;
        ns=ns,
        output_path=output_path,
        show_plot=show_plot,
        title=title,
    )
end

"Compare two RateScan objects or two scan CSV files."
function run_compare_rates_example(
    first_scan_or_csv,
    second_scan_or_csv;
    labels=("H-min", "BFF"),
    ns=nothing,
    output_path::Union{Nothing,AbstractString}=joinpath(@__DIR__, "results", "rate_comparison.pdf"),
    show_plot::Bool=true,
)
    return RDITools.compare_rates(
        first_scan_or_csv,
        second_scan_or_csv;
        labels=labels,
        ns=ns,
        output_path=output_path,
        show_plot=show_plot,
    )
end

"Read a saved scan CSV back into a RateScan object."
function run_read_scan_csv_example(
    path::AbstractString=joinpath(@__DIR__, "results", "rate_scan.csv");
    method::Symbol=:unknown,
    label=nothing,
)
    scan = RDITools.read_scan_csv(path; method=method, label=label)
    println("Loaded ", length(scan.rate), " scan rows from ", path)
    return scan
end

"Explicitly save a RateScan with exactly n, eta, theta_opt, rate."
function run_save_scan_csv_example(
    scan::RDITools.RateScan,
    path::AbstractString=joinpath(@__DIR__, "results", "saved_rate_scan.csv"),
)
    saved_path = RDITools.save_scan_csv(scan, path)
    println("Saved scan CSV to ", saved_path)
    return saved_path
end

"Explicitly save a RateHeatmap with exactly eta, theta, rate."
function run_save_heatmap_csv_example(
    heatmap_data::RDITools.RateHeatmap,
    path::AbstractString=joinpath(@__DIR__, "results", "saved_rate_heatmap.csv"),
)
    saved_path = RDITools.save_heatmap_csv(heatmap_data, path)
    println("Saved heatmap CSV to ", saved_path)
    return saved_path
end

function print_available_calls()
    println("""
\nMain callable examples now available:

  run_rate_at_example(...)                  # one H_min/BFF point
  run_optimize_theta_example(...)           # best theta for one n and eta
  run_scan_rates_example(...)               # optimized eta scan
  run_rate_heatmap_example(...)             # full eta-theta heatmap
  run_plot_rates_example(scan_or_csv, ...)   # plot several n values
  run_compare_rates_example(a, b, ...)       # compare two rate families
  run_read_scan_csv_example(path, ...)       # reload a scan CSV
  run_save_scan_csv_example(scan, path)      # save n,eta,theta_opt,rate
  run_save_heatmap_csv_example(data, path)   # save eta,theta,rate

Direct builder examples are also available:

  run_hmin_builder_example(...)
  run_bff_builder_example(...)
  run_bff_per_node_builder_example(...)

Examples:

  point = run_rate_at_example(method=:hmin, n=3, theta=0.7, eta=0.8)

  bff_point = run_rate_at_example(
      method=:bff,
      bff_mode=:per_node,
      bff_n_nodes=4,
      n=2,
      theta=0.8,
      eta=0.9,
  )

  hmin_scan = run_scan_rates_example(
      method=:hmin,
      ns=2:5,
      eta_points=20,
      csv_path="results/hmin.csv",
  )

  bff_scan = run_scan_rates_example(
      method=:bff,
      bff_mode=:per_node,
      ns=2:3,
      eta_points=10,
      csv_path="results/bff.csv",
  )

  run_compare_rates_example(
      "results/hmin.csv",
      "results/bff.csv",
      labels=("H-min", "BFF"),
  )
""")
    return nothing
end

print_available_calls()

# Nothing computationally expensive is run automatically. Call one of the
# functions printed above after including this file.
