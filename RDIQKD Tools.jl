module RDIQKDTools

using Printf
using DelimitedFiles
using Plots

# The tools file can be included on its own, but the normal order is:
#   include("RDIQKD Builder.jl")
#   include("RDIQKD Tools.jl")
#
# The fallback search also accepts copied/downloaded names such as
# "RDIQKD Builder(2).jl" without requiring any change to the builder itself.
if !isdefined(Main, :RDIQKDSDPBuilderGeneralWordsPerNodeV2MarginalBFF)
    preferred_builder_files = [
        joinpath(@__DIR__, "RDIQKD Builder.jl"),
        joinpath(@__DIR__, "RDIQKD Builder(2).jl"),
    ]
    builder_file = findfirst(isfile, preferred_builder_files)
    if builder_file === nothing
        matching_files = sort(filter(
            name -> startswith(name, "RDIQKD Builder") && endswith(lowercase(name), ".jl"),
            readdir(@__DIR__),
        ))
        isempty(matching_files) && error(
            "Could not find the builder file. Put RDIQKD Builder.jl in the same folder as RDIQKD Tools.jl.",
        )
        Base.include(Main, joinpath(@__DIR__, first(matching_files)))
    else
        Base.include(Main, preferred_builder_files[builder_file])
    end
end

const RDI = Main.RDIQKDSDPBuilderGeneralWordsPerNodeV2MarginalBFF
const RATE_FLOOR = 1.0e-12

export RateScan,
       RateHeatmap,
       rate_at,
       optimize_theta,
       scan_rates,
       rate_heatmap,
       plot_rates,
       compare_rates,
       save_scan_csv,
       save_heatmap_csv,
       read_scan_csv,
       print_result

# =============================================================================
# Data returned by the scan and heatmap functions
# =============================================================================

struct RateScan
    n::Vector{Int}
    eta::Vector{Float64}
    theta_opt::Vector{Float64}
    rate::Vector{Float64}
    method::Symbol
    label::String
end

struct RateHeatmap
    n::Int
    eta::Vector{Float64}
    theta::Vector{Float64}
    rate::Matrix{Float64}       # rows = theta, columns = eta
    theta_opt::Vector{Float64}
    rate_opt::Vector{Float64}
    method::Symbol
    label::String
end

# =============================================================================
# Small internal helpers
# =============================================================================

function _as_int_vector(ns)
    ns isa Integer && return [Int(ns)]
    out = Int.(collect(ns))
    isempty(out) && error("ns must contain at least one value")
    return out
end

function _as_float_vector(xs, name::AbstractString)
    out = Float64.(collect(xs))
    isempty(out) && error("$name must contain at least one value")
    all(isfinite, out) || error("$name contains a non-finite value")
    return out
end

function _default_label(method::Symbol, bff_mode::Symbol)
    method == :hmin && return "H-min"
    method == :bff && bff_mode == :per_node && return "BFF per node"
    method == :bff && return "BFF combined"
    return String(method)
end

function _states(preparation, n::Int, theta::Real)
    if preparation isa Symbol
        preparation == :phase && return RDI.phase_amplitudes(n, theta)
        preparation in (:real, :realxz) && return RDI.realxz_amplitudes(n, theta)
        error("Unknown preparation=$preparation. Use :phase, :real, or a function (n, theta) -> amplitudes.")
    elseif preparation isa Function
        return preparation(n, theta)
    else
        error("preparation must be :phase, :real, or a function")
    end
end

function _finite_number(x)
    x === nothing && return NaN
    try
        y = Float64(x)
        return isfinite(y) ? y : NaN
    catch
        return NaN
    end
end

function _rate_from_result(res; certified::Bool=true, clip_rate::Bool=true)
    value = NaN
    bound = NaN

    if res isa RDI.RDISDPResult
        value = _finite_number(get(res.metadata, :rate_value, NaN))
        bound = _finite_number(get(res.metadata, :rate_bound, NaN))
    elseif res isa RDI.BFFPerNodeResult
        value = _finite_number(res.rate_value)
        bound = _finite_number(res.rate_bound)
    else
        error("Unsupported SDP result type $(typeof(res))")
    end

    raw = certified && isfinite(bound) ? bound : value
    if !isfinite(raw)
        raw = certified ? value : bound
    end
    rate = clip_rate && isfinite(raw) ? max(raw, 0.0) : raw
    return rate, raw
end

function _status_string(res)
    if res isa RDI.RDISDPResult
        return string(get(res.metadata, :termination_status, :UNKNOWN))
    elseif res isa RDI.BFFPerNodeResult
        statuses = get(res.metadata, :termination_statuses, Any[])
        isempty(statuses) && return "UNKNOWN"
        strings = unique(string.(statuses))
        return length(strings) == 1 ? strings[1] : join(strings, "/")
    end
    return "UNKNOWN"
end

function _method_tag(method::Symbol, bff_mode::Symbol)
    method == :hmin && return "HMIN"
    bff_mode == :per_node && return "BFF-NODES"
    return "BFF"
end

mutable struct _ProgressLine
    enabled::Bool
    started::Float64
    previous_length::Int
end

_ProgressLine(enabled::Bool) = _ProgressLine(enabled, time(), 0)

function _duration(seconds::Real)
    s = max(0, round(Int, seconds))
    h, rem1 = divrem(s, 3600)
    m, sec = divrem(rem1, 60)
    h > 0 && return @sprintf("%d:%02d:%02d", h, m, sec)
    return @sprintf("%02d:%02d", m, sec)
end

function _update!(progress::_ProgressLine, message::AbstractString)
    progress.enabled || return
    line = string(message, " | elapsed ", _duration(time() - progress.started))
    padding = max(progress.previous_length - length(line), 0)
    print('\r', line, repeat(" ", padding))
    flush(stdout)
    progress.previous_length = length(line)
    return
end

function _finish!(progress::_ProgressLine, message::AbstractString)
    progress.enabled || return
    _update!(progress, message)
    println()
    progress.previous_length = 0
    return
end

function _ensure_parent(path::AbstractString)
    dir = dirname(path)
    !isempty(dir) && dir != "." && mkpath(dir)
    return path
end

function _safe_savefig(fig, path::Union{Nothing,AbstractString})
    path === nothing && return nothing
    _ensure_parent(path)
    try
        savefig(fig, path)
        return path
    catch err
        fallback = endswith(lowercase(path), ".pdf") ? replace(path, r"\.pdf$"i => ".png") : string(path, ".png")
        @warn "Could not save the requested plot format; saving PNG instead" requested=path fallback=fallback exception=(err, catch_backtrace())
        savefig(fig, fallback)
        return fallback
    end
end

# =============================================================================
# One SDP evaluation
# =============================================================================

"""
    rate_at(; method=:hmin, n, theta, eta, ...)

Solve one RDIQKD point and return the selected key rate together with the full
SDP result. `method` is `:hmin` or `:bff`. For BFF, `bff_mode` is
`:per_node` (default) or `:combined`.
"""
function rate_at(;
    method::Symbol=:hmin,
    n::Int,
    theta::Real,
    eta::Real,
    lambda::Real=0.0,
    preparation=:phase,
    words=nothing,
    solver::Symbol=:mosek,
    field::Symbol=:auto,
    silent::Bool=true,
    certified::Bool=true,
    clip_rate::Bool=true,
    bff_mode::Symbol=:per_node,
    bff_n_nodes::Int=4,
    progress_callback=nothing,
    builder_kwargs...,
)
    method in (:hmin, :bff) || error("method must be :hmin or :bff")
    bff_mode in (:per_node, :combined) || error("bff_mode must be :per_node or :combined")

    amps = _states(preparation, n, theta)
    common = (
        amplitudes=amps,
        eta=eta,
        lambda=lambda,
        word_blocks=words,
        solver=solver,
        field=field,
        silent=silent,
    )

    if method == :hmin
        progress_callback !== nothing && progress_callback(:before, 1, 1)
        result = RDI.build_hmin_sdp(; common..., builder_kwargs...)
        progress_callback !== nothing && progress_callback(:after, 1, 1)
    elseif bff_mode == :combined
        progress_callback !== nothing && progress_callback(:before, 1, 1)
        result = RDI.build_bff_sdp(;
            common...,
            bff_n_nodes=bff_n_nodes,
            builder_kwargs...,
        )
        progress_callback !== nothing && progress_callback(:after, 1, 1)
    else
        # Build first, then solve node by node so the progress line can show
        # exactly which independent BFF SDP is running.
        result = RDI.build_bff_per_node_sdp_models(;
            common...,
            bff_n_nodes=bff_n_nodes,
            builder_kwargs...,
        )
        total_nodes = length(result.node_results)
        for (node_index, node_result) in enumerate(result.node_results)
            progress_callback !== nothing && progress_callback(:before, node_index, total_nodes)
            RDI.solve_rdi_sdp!(node_result)
            progress_callback !== nothing && progress_callback(:after, node_index, total_nodes)
        end
        RDI.solve_bff_per_node_sdp!(result)  # aggregates already-solved nodes without solving twice
    end

    rate, raw_rate = _rate_from_result(result; certified=certified, clip_rate=clip_rate)
    return (
        rate=rate,
        raw_rate=raw_rate,
        result=result,
        status=_status_string(result),
        method=method,
        bff_mode=bff_mode,
        n=n,
        eta=Float64(eta),
        theta=Float64(theta),
    )
end

# =============================================================================
# Theta optimization for one eta
# =============================================================================

function _best_finite(values)
    best_index = 0
    best_value = -Inf
    for (i, value) in enumerate(values)
        if isfinite(value) && value > best_value
            best_index = i
            best_value = value
        end
    end
    return best_index, best_value
end

function _optimize_theta_core(;
    n::Int,
    eta::Real,
    method::Symbol,
    theta_range::Tuple{<:Real,<:Real},
    coarse_points::Int,
    refine_points::Int,
    refine_width::Real,
    golden_steps::Int,
    theta_tol::Real,
    lambda::Real,
    preparation,
    words,
    solver::Symbol,
    field::Symbol,
    silent::Bool,
    certified::Bool,
    clip_rate::Bool,
    bff_mode::Symbol,
    bff_n_nodes::Int,
    strict::Bool,
    progress::_ProgressLine,
    case_index::Int,
    case_total::Int,
    builder_kwargs...,
)
    lower = Float64(theta_range[1])
    upper = Float64(theta_range[2])
    lower < upper || error("theta_range must satisfy lower < upper")
    coarse_points >= 3 || error("coarse_points must be at least 3")
    refine_points >= 3 || error("refine_points must be at least 3")
    golden_steps >= 0 || error("golden_steps must be non-negative")

    tag = _method_tag(method, bff_mode)
    cache = Dict{Float64,Any}()
    solve_count = Ref(0)
    current_best = Ref(-Inf)

    function evaluate(theta::Real, stage::AbstractString, stage_index::Int, stage_total::Int)
        theta_value = clamp(Float64(theta), lower, upper)
        key = round(theta_value; digits=14)
        haskey(cache, key) && return cache[key]

        solve_count[] += 1
        node_text = Ref("")

        function callback(moment, node_index, node_total)
            if node_total > 1
                node_text[] = @sprintf(" | node %d/%d", node_index, node_total)
            else
                node_text[] = ""
            end
            action = moment == :before ? "solving" : "done"
            best_text = isfinite(current_best[]) ? @sprintf("%.4e", current_best[]) : "--"
            _update!(progress,
                @sprintf("[%s] case %d/%d | n=%d eta=%.6f | %s %d/%d | theta=%.7f%s | %s | best=%s",
                    tag, case_index, case_total, n, Float64(eta), stage, stage_index, stage_total,
                    theta_value, node_text[], action, best_text))
        end

        record = nothing
        try
            point = rate_at(;
                method=method,
                n=n,
                theta=theta_value,
                eta=eta,
                lambda=lambda,
                preparation=preparation,
                words=words,
                solver=solver,
                field=field,
                silent=silent,
                certified=certified,
                clip_rate=clip_rate,
                bff_mode=bff_mode,
                bff_n_nodes=bff_n_nodes,
                progress_callback=callback,
                builder_kwargs...,
            )
            record = merge(point, (error=nothing,))
            isfinite(point.rate) && (current_best[] = max(current_best[], point.rate))
        catch err
            strict && rethrow(err)
            record = (
                rate=NaN,
                raw_rate=NaN,
                result=nothing,
                status="ERROR",
                method=method,
                bff_mode=bff_mode,
                n=n,
                eta=Float64(eta),
                theta=theta_value,
                error=sprint(showerror, err),
            )
            _update!(progress,
                @sprintf("[%s] case %d/%d | n=%d eta=%.6f | %s %d/%d | theta=%.7f | ERROR",
                    tag, case_index, case_total, n, Float64(eta), stage, stage_index, stage_total, theta_value))
        end

        cache[key] = record
        return record
    end

    coarse_thetas = collect(LinRange(lower, upper, coarse_points))
    coarse_records = [evaluate(theta, "coarse", i, coarse_points) for (i, theta) in enumerate(coarse_thetas)]
    coarse_rates = [record.rate for record in coarse_records]
    coarse_best_index, _ = _best_finite(coarse_rates)

    if coarse_best_index == 0
        return (
            theta_opt=NaN,
            rate=NaN,
            raw_rate=NaN,
            result=nothing,
            status="NO_FINITE_POINT",
            evaluations=solve_count[],
            errors=[record.error for record in values(cache) if record.error !== nothing],
        )
    end

    coarse_step = (upper - lower) / (coarse_points - 1)
    coarse_best_theta = coarse_thetas[coarse_best_index]
    refine_half_width = Float64(refine_width) * coarse_step
    refine_lower = clamp(coarse_best_theta - refine_half_width, lower, upper)
    refine_upper = clamp(coarse_best_theta + refine_half_width, lower, upper)
    if refine_lower == refine_upper
        refine_lower, refine_upper = lower, upper
    end

    refine_thetas = collect(LinRange(refine_lower, refine_upper, refine_points))
    refine_records = [evaluate(theta, "refine", i, refine_points) for (i, theta) in enumerate(refine_thetas)]
    refine_rates = [record.rate for record in refine_records]
    refine_best_index, _ = _best_finite(refine_rates)

    if refine_best_index > 0 && golden_steps > 0
        left_index = max(refine_best_index - 1, 1)
        right_index = min(refine_best_index + 1, length(refine_thetas))
        a = refine_thetas[left_index]
        b = refine_thetas[right_index]
        if a < b
            phi = (sqrt(5.0) - 1.0) / 2.0
            c = b - phi * (b - a)
            d = a + phi * (b - a)
            rc = evaluate(c, "golden", 1, golden_steps + 2)
            rd = evaluate(d, "golden", 2, golden_steps + 2)

            for iteration in 1:golden_steps
                (b - a) <= Float64(theta_tol) && break
                if (!isfinite(rd.rate)) || (isfinite(rc.rate) && rc.rate >= rd.rate)
                    b, d, rd = d, c, rc
                    c = b - phi * (b - a)
                    rc = evaluate(c, "golden", iteration + 2, golden_steps + 2)
                else
                    a, c, rc = c, d, rd
                    d = a + phi * (b - a)
                    rd = evaluate(d, "golden", iteration + 2, golden_steps + 2)
                end
            end
            evaluate((a + b) / 2.0, "final", 1, 1)
        end
    end

    records = collect(values(cache))
    rates = [record.rate for record in records]
    best_index, _ = _best_finite(rates)
    best_index == 0 && error("No finite rate was found")
    best = records[best_index]

    return (
        theta_opt=best.theta,
        rate=best.rate,
        raw_rate=best.raw_rate,
        result=best.result,
        status=best.status,
        evaluations=solve_count[],
        errors=[record.error for record in records if record.error !== nothing],
    )
end

"""
    optimize_theta(; n, eta, method=:hmin, ...)

Find the best theta for one `(n, eta)` point using a coarse grid, a local
refined grid, and a short golden-section refinement.
"""
function optimize_theta(;
    n::Int,
    eta::Real,
    method::Symbol=:hmin,
    theta_range::Tuple{<:Real,<:Real}=(1.0e-3, pi / 2 - 1.0e-3),
    coarse_points::Int=31,
    refine_points::Int=21,
    refine_width::Real=1.5,
    golden_steps::Int=12,
    theta_tol::Real=1.0e-4,
    lambda::Real=0.0,
    preparation=:phase,
    words=nothing,
    solver::Symbol=:mosek,
    field::Symbol=:auto,
    silent::Bool=true,
    certified::Bool=true,
    clip_rate::Bool=true,
    bff_mode::Symbol=:per_node,
    bff_n_nodes::Int=4,
    strict::Bool=false,
    progress::Bool=true,
    builder_kwargs...,
)
    progress_line = _ProgressLine(progress)
    result = _optimize_theta_core(;
        n=n,
        eta=eta,
        method=method,
        theta_range=theta_range,
        coarse_points=coarse_points,
        refine_points=refine_points,
        refine_width=refine_width,
        golden_steps=golden_steps,
        theta_tol=theta_tol,
        lambda=lambda,
        preparation=preparation,
        words=words,
        solver=solver,
        field=field,
        silent=silent,
        certified=certified,
        clip_rate=clip_rate,
        bff_mode=bff_mode,
        bff_n_nodes=bff_n_nodes,
        strict=strict,
        progress=progress_line,
        case_index=1,
        case_total=1,
        builder_kwargs...,
    )
    _finish!(progress_line,
        @sprintf("[%s] complete | n=%d eta=%.6f | theta*=%.8f | rate=%.6e | %s",
            _method_tag(method, bff_mode), n, Float64(eta), result.theta_opt, result.rate, result.status))
    return result
end

# =============================================================================
# Eta scan with theta optimization
# =============================================================================

function _etas_for_n(n::Int, etas, eta_points::Int, eta_pad::Real)
    if etas === nothing
        eta_points >= 1 || error("eta_points must be positive")
        eta_min = 1.0 / n + Float64(eta_pad)
        eta_min < 1.0 || error("1/n + eta_pad must be below 1")
        return collect(LinRange(1.0, eta_min, eta_points))
    end
    return _as_float_vector(etas, "etas")
end

"Save exactly the columns: n, eta, theta_opt, rate."
function save_scan_csv(data::RateScan, path::AbstractString)
    _ensure_parent(path)
    open(path, "w") do io
        println(io, "n,eta,theta_opt,rate")
        for i in eachindex(data.n)
            @printf(io, "%d,%.17g,%.17g,%.17g\n",
                data.n[i], data.eta[i], data.theta_opt[i], data.rate[i])
        end
    end
    return path
end

"""
    scan_rates(; ns=2:5, etas=nothing, method=:hmin, csv_path=nothing, ...)

For each `n` and `eta`, optimize theta and store only
`n, eta, theta_opt, rate` in the optional CSV file. If `etas=nothing`, each n
is scanned from eta=1 down to `1/n + eta_pad`.
"""
function scan_rates(;
    ns=2:5,
    etas=nothing,
    eta_points::Int=25,
    eta_pad::Real=5.0e-3,
    method::Symbol=:hmin,
    theta_range::Tuple{<:Real,<:Real}=(1.0e-3, pi / 2 - 1.0e-3),
    coarse_points::Int=31,
    refine_points::Int=21,
    refine_width::Real=1.5,
    golden_steps::Int=12,
    theta_tol::Real=1.0e-4,
    lambda::Real=0.0,
    preparation=:phase,
    words=nothing,
    solver::Symbol=:mosek,
    field::Symbol=:auto,
    silent::Bool=true,
    certified::Bool=true,
    clip_rate::Bool=true,
    bff_mode::Symbol=:per_node,
    bff_n_nodes::Int=4,
    strict::Bool=false,
    progress::Bool=true,
    csv_path::Union{Nothing,AbstractString}=nothing,
    label::Union{Nothing,AbstractString}=nothing,
    builder_kwargs...,
)
    n_values = _as_int_vector(ns)
    eta_lists = Dict(n => _etas_for_n(n, etas, eta_points, eta_pad) for n in n_values)
    total_cases = sum(length(v) for v in Base.values(eta_lists))
    progress_line = _ProgressLine(progress)

    n_out = Int[]
    eta_out = Float64[]
    theta_out = Float64[]
    rate_out = Float64[]
    scan_label = label === nothing ? _default_label(method, bff_mode) : String(label)

    case_index = 0
    for n in n_values
        for eta_value in eta_lists[n]
            case_index += 1
            optimum = _optimize_theta_core(;
                n=n,
                eta=eta_value,
                method=method,
                theta_range=theta_range,
                coarse_points=coarse_points,
                refine_points=refine_points,
                refine_width=refine_width,
                golden_steps=golden_steps,
                theta_tol=theta_tol,
                lambda=lambda,
                preparation=preparation,
                words=words,
                solver=solver,
                field=field,
                silent=silent,
                certified=certified,
                clip_rate=clip_rate,
                bff_mode=bff_mode,
                bff_n_nodes=bff_n_nodes,
                strict=strict,
                progress=progress_line,
                case_index=case_index,
                case_total=total_cases,
                builder_kwargs...,
            )

            push!(n_out, n)
            push!(eta_out, eta_value)
            push!(theta_out, optimum.theta_opt)
            push!(rate_out, optimum.rate)

            partial = RateScan(n_out, eta_out, theta_out, rate_out, method, scan_label)
            csv_path !== nothing && save_scan_csv(partial, csv_path)

            _update!(progress_line,
                @sprintf("[%s] case %d/%d complete | n=%d eta=%.6f | theta*=%.8f | rate=%.6e | %s",
                    _method_tag(method, bff_mode), case_index, total_cases, n, eta_value,
                    optimum.theta_opt, optimum.rate, optimum.status))
        end
    end

    data = RateScan(n_out, eta_out, theta_out, rate_out, method, scan_label)
    _finish!(progress_line,
        @sprintf("[%s] scan complete | %d optimized points%s",
            _method_tag(method, bff_mode), length(rate_out),
            csv_path === nothing ? "" : " | saved to $(csv_path)"))
    return data
end

# =============================================================================
# Heatmap
# =============================================================================

"Save exactly the columns: eta, theta, rate."
function save_heatmap_csv(data::RateHeatmap, path::AbstractString)
    _ensure_parent(path)
    open(path, "w") do io
        println(io, "eta,theta,rate")
        for j in eachindex(data.eta), i in eachindex(data.theta)
            @printf(io, "%.17g,%.17g,%.17g\n",
                data.eta[j], data.theta[i], data.rate[i, j])
        end
    end
    return path
end

function _column_normalized(rate::Matrix{Float64})
    output = copy(rate)
    for j in axes(output, 2)
        valid = [i for i in axes(output, 1) if isfinite(output[i, j])]
        isempty(valid) && continue
        minimum_value = minimum(output[i, j] for i in valid)
        maximum_value = maximum(output[i, j] for i in valid)
        span = maximum_value - minimum_value
        if span > 0
            for i in valid
                output[i, j] = (output[i, j] - minimum_value) / span
            end
        else
            for i in valid
                output[i, j] = 0.0
            end
        end
    end
    return output
end

"""
    rate_heatmap(; n, etas, thetas, method=:hmin, ...)

Evaluate the complete eta-theta grid. The optional CSV contains exactly
`eta, theta, rate`. The red points on the figure mark the best theta for each
eta column.
"""
function rate_heatmap(;
    n::Int,
    etas=nothing,
    thetas=nothing,
    eta_min::Real=1.0 / n + 5.0e-3,
    eta_max::Real=1.0,
    eta_points::Int=30,
    theta_min::Real=1.0e-3,
    theta_max::Real=pi / 2 - 1.0e-3,
    theta_points::Int=50,
    method::Symbol=:hmin,
    lambda::Real=0.0,
    preparation=:phase,
    words=nothing,
    solver::Symbol=:mosek,
    field::Symbol=:auto,
    silent::Bool=true,
    certified::Bool=true,
    clip_rate::Bool=true,
    bff_mode::Symbol=:per_node,
    bff_n_nodes::Int=4,
    strict::Bool=false,
    progress::Bool=true,
    normalize_columns::Bool=false,
    log_color::Bool=true,
    csv_path::Union{Nothing,AbstractString}=nothing,
    plot_path::Union{Nothing,AbstractString}=nothing,
    show_plot::Bool=true,
    title::Union{Nothing,AbstractString}=nothing,
    label::Union{Nothing,AbstractString}=nothing,
    builder_kwargs...,
)
    eta_values = etas === nothing ? collect(LinRange(Float64(eta_max), Float64(eta_min), eta_points)) : _as_float_vector(etas, "etas")
    theta_values = thetas === nothing ? collect(LinRange(Float64(theta_min), Float64(theta_max), theta_points)) : _as_float_vector(thetas, "thetas")

    rates = fill(NaN, length(theta_values), length(eta_values))
    theta_opt = fill(NaN, length(eta_values))
    rate_opt = fill(NaN, length(eta_values))
    total = length(theta_values) * length(eta_values)
    completed = 0
    progress_line = _ProgressLine(progress)
    tag = _method_tag(method, bff_mode)

    for (j, eta_value) in enumerate(eta_values)
        best_rate = -Inf
        best_theta = NaN
        for (i, theta_value) in enumerate(theta_values)
            completed += 1

            function callback(moment, node_index, node_total)
                node_text = node_total > 1 ? @sprintf(" | node %d/%d", node_index, node_total) : ""
                action = moment == :before ? "solving" : "done"
                _update!(progress_line,
                    @sprintf("[%s] heatmap %d/%d | n=%d eta=%.6f theta=%.7f%s | %s",
                        tag, completed, total, n, eta_value, theta_value, node_text, action))
            end

            point = nothing
            try
                point = rate_at(;
                    method=method,
                    n=n,
                    theta=theta_value,
                    eta=eta_value,
                    lambda=lambda,
                    preparation=preparation,
                    words=words,
                    solver=solver,
                    field=field,
                    silent=silent,
                    certified=certified,
                    clip_rate=clip_rate,
                    bff_mode=bff_mode,
                    bff_n_nodes=bff_n_nodes,
                    progress_callback=callback,
                    builder_kwargs...,
                )
                rates[i, j] = point.rate
            catch err
                strict && rethrow(err)
                rates[i, j] = NaN
                _update!(progress_line,
                    @sprintf("[%s] heatmap %d/%d | n=%d eta=%.6f theta=%.7f | ERROR",
                        tag, completed, total, n, eta_value, theta_value))
            end

            if isfinite(rates[i, j]) && rates[i, j] > best_rate
                best_rate = rates[i, j]
                best_theta = theta_value
            end
        end
        theta_opt[j] = best_theta
        rate_opt[j] = isfinite(best_rate) ? best_rate : NaN

        partial = RateHeatmap(
            n, eta_values, theta_values, rates, theta_opt, rate_opt, method,
            label === nothing ? _default_label(method, bff_mode) : String(label),
        )
        csv_path !== nothing && save_heatmap_csv(partial, csv_path)
    end

    heat_label = label === nothing ? _default_label(method, bff_mode) : String(label)
    data = RateHeatmap(n, eta_values, theta_values, rates, theta_opt, rate_opt, method, heat_label)

    plot_values = normalize_columns ? _column_normalized(rates) : copy(rates)
    if log_color && !normalize_columns
        for index in eachindex(plot_values)
            if !isfinite(plot_values[index])
                plot_values[index] = NaN
            elseif plot_values[index] <= 0
                plot_values[index] = RATE_FLOOR
            end
        end
    end

    plot_title = title === nothing ? "$(heat_label) key-rate heatmap, n=$n" : String(title)
    color_title = normalize_columns ? "R (column normalized)" : "R"

    fig = heatmap(
        eta_values,
        theta_values,
        plot_values;
        xlabel="η",
        ylabel="θ",
        title=plot_title,
        colorbar_title=color_title,
        xflip=true,
        framestyle=:box,
        grid=false,
        color=:viridis,
        size=(1050, 750),
        dpi=300,
        titlefontsize=18,
        guidefontsize=17,
        tickfontsize=14,
        zscale=(log_color && !normalize_columns) ? :log10 : :identity,
    )
    scatter!(
        fig,
        eta_values,
        theta_opt;
        marker=:circle,
        markersize=4.5,
        markerstrokewidth=0,
        color=:red,
        label=false,
    )

    saved_plot = _safe_savefig(fig, plot_path)
    show_plot && display(fig)
    _finish!(progress_line,
        @sprintf("[%s] heatmap complete | %d SDP points%s",
            tag, total, csv_path === nothing ? "" : " | saved to $(csv_path)"))

    return (data=data, plot=fig, csv_path=csv_path, plot_path=saved_plot)
end

# =============================================================================
# CSV loading
# =============================================================================

function _normal_header(value)
    return lowercase(replace(strip(string(value)), r"[^a-z0-9ηθ]" => ""))
end

function read_scan_csv(path::AbstractString; method::Symbol=:unknown, label::Union{Nothing,AbstractString}=nothing)
    lines = filter(line -> !isempty(strip(line)), readlines(path))
    length(lines) >= 2 || error("No data rows found in $path")

    delimiter = occursin(',', lines[1]) ? ',' : occursin('\t', lines[1]) ? '\t' : nothing
    split_row(line) = delimiter === nothing ? split(strip(line)) : strip.(split(strip(line), delimiter))

    headers = _normal_header.(split_row(lines[1]))
    function column(names...)
        index = findfirst(header -> header in names, headers)
        index === nothing && error("Missing one of columns $(names) in $path. Headers are $(headers)")
        return index
    end

    n_column = column("n")
    eta_column = column("eta", "η")
    theta_column = column("thetaopt", "theta", "θopt", "θ")
    rate_column = column("rate", "r")

    n = Int[]
    eta = Float64[]
    theta = Float64[]
    rate = Float64[]
    for (row_number, line) in enumerate(lines[2:end])
        columns = split_row(line)
        length(columns) >= maximum((n_column, eta_column, theta_column, rate_column)) ||
            error("Row $row_number in $path has too few columns")
        push!(n, round(Int, parse(Float64, columns[n_column])))
        push!(eta, parse(Float64, columns[eta_column]))
        push!(theta, parse(Float64, columns[theta_column]))
        push!(rate, parse(Float64, columns[rate_column]))
    end

    data_label = label === nothing ? splitext(basename(path))[1] : String(label)
    return RateScan(n, eta, theta, rate, method, data_label)
end

# =============================================================================
# Plot style shared by the two rate plotting functions
# =============================================================================

function _rate_plot_base(;
    title::AbstractString,
    eta_min::Real,
    eta_max::Real,
    ylims::Tuple{<:Real,<:Real}=(5.0e-9, 1.2),
    legend=:bottomleft,
)
    yticks_values = [10.0^(-k) for k in 0:8]
    yticks_labels = ["10⁰", "10⁻¹", "10⁻²", "10⁻³", "10⁻⁴", "10⁻⁵", "10⁻⁶", "10⁻⁷", "10⁻⁸"]

    lower_x = max(0.0, Float64(eta_min) - 0.02)
    upper_x = min(1.03, Float64(eta_max) + 0.03)

    fig = plot(
        xlabel="η",
        ylabel="R",
        title=title,
        yaxis=:log10,
        yticks=(yticks_values, yticks_labels),
        xlims=(lower_x, upper_x),
        ylims=ylims,
        xflip=true,
        grid=true,
        minorgrid=true,
        framestyle=:box,
        legend=legend,
        size=(1050, 750),
        dpi=300,
        titlefontsize=18,
        guidefontsize=17,
        tickfontsize=14,
        legendfontsize=13,
        left_margin=10Plots.mm,
        right_margin=8Plots.mm,
        bottom_margin=9Plots.mm,
        top_margin=8Plots.mm,
    )

    standard_ticks = [1.0, 0.8, 0.6, 0.5, 1 / 3, 0.25, 0.2, 1 / 6, 0.125]
    tick_labels = ["1", "0.8", "0.6", "0.5", "0.333", "0.25", "0.2", "0.167", "0.125"]
    keep = [i for i in eachindex(standard_ticks) if lower_x <= standard_ticks[i] <= upper_x]
    !isempty(keep) && xticks!(fig, standard_ticks[keep], tick_labels[keep])
    return fig
end

function _positive_series(data::RateScan, n::Int)
    indices = [i for i in eachindex(data.n) if data.n[i] == n && isfinite(data.eta[i]) && isfinite(data.rate[i]) && data.rate[i] > 0]
    order = sortperm(data.eta[indices]; rev=true)
    selected = indices[order]
    return data.eta[selected], data.rate[selected]
end

function _as_scan(data; method::Symbol=:unknown, label=nothing)
    data isa RateScan && return data
    data isa AbstractString && return read_scan_csv(data; method=method, label=label)
    error("Expected a RateScan or a scan CSV path")
end

"Plot one rate family for several n values using the Better-RDIQKD plot style."
function plot_rates(data_input;
    ns=nothing,
    output_path::Union{Nothing,AbstractString}=nothing,
    show_plot::Bool=true,
    title::Union{Nothing,AbstractString}=nothing,
    marker=:circle,
    connect::Bool=false,
    ylims::Tuple{<:Real,<:Real}=(5.0e-9, 1.2),
)
    data = _as_scan(data_input)
    n_values = ns === nothing ? sort(unique(data.n)) : _as_int_vector(ns)
    finite_eta = data.eta[isfinite.(data.eta)]
    isempty(finite_eta) && error("No finite eta values to plot")
    plot_title = title === nothing ? "$(data.label) key rates" : String(title)
    fig = _rate_plot_base(title=plot_title, eta_min=minimum(finite_eta), eta_max=maximum(finite_eta), ylims=ylims)

    for (color_index, n) in enumerate(n_values)
        eta, rate = _positive_series(data, n)
        isempty(rate) && continue
        plot!(
            fig,
            eta,
            rate;
            seriestype=connect ? :path : :scatter,
            marker=marker,
            markersize=6.5,
            markerstrokewidth=0,
            linewidth=connect ? 2.5 : 0,
            color=color_index,
            label="n=$n",
        )
    end

    saved_path = _safe_savefig(fig, output_path)
    show_plot && display(fig)
    return (plot=fig, output_path=saved_path)
end

"Compare two rate families; marker shape identifies the family and colour identifies n."
function compare_rates(first_input, second_input;
    labels::Tuple{<:AbstractString,<:AbstractString}=("Rate 1", "Rate 2"),
    markers::Tuple=(:circle, :star5),
    ns=nothing,
    output_path::Union{Nothing,AbstractString}=nothing,
    show_plot::Bool=true,
    title::AbstractString="Comparison of RDIQKD key rates",
    ylims::Tuple{<:Real,<:Real}=(5.0e-9, 1.2),
)
    first = _as_scan(first_input; label=labels[1])
    second = _as_scan(second_input; label=labels[2])
    n_values = ns === nothing ? sort(union(unique(first.n), unique(second.n))) : _as_int_vector(ns)
    all_eta = vcat(first.eta[isfinite.(first.eta)], second.eta[isfinite.(second.eta)])
    isempty(all_eta) && error("No finite eta values to plot")
    fig = _rate_plot_base(title=title, eta_min=minimum(all_eta), eta_max=maximum(all_eta), ylims=ylims)

    # Marker legend entries.
    for family_index in 1:2
        plot!(
            fig,
            [NaN],
            [NaN];
            seriestype=:scatter,
            marker=markers[family_index],
            markersize=family_index == 1 ? 7 : 9,
            markerstrokewidth=0,
            color=:black,
            label=labels[family_index],
        )
    end

    # Colour legend entries.
    for (color_index, n) in enumerate(n_values)
        plot!(
            fig,
            [NaN],
            [NaN];
            seriestype=:scatter,
            marker=:square,
            markersize=7,
            markerstrokewidth=0,
            color=color_index,
            label="n=$n",
        )
    end

    for (color_index, n) in enumerate(n_values)
        for (family_index, data) in enumerate((first, second))
            eta, rate = _positive_series(data, n)
            isempty(rate) && continue
            plot!(
                fig,
                eta,
                rate;
                seriestype=:scatter,
                marker=markers[family_index],
                markersize=family_index == 1 ? 6.5 : 9,
                markerstrokewidth=0,
                color=color_index,
                label=false,
            )
        end
    end

    saved_path = _safe_savefig(fig, output_path)
    show_plot && display(fig)
    return (plot=fig, output_path=saved_path)
end

# =============================================================================
# Human-readable result printing
# =============================================================================

function print_result(res; prefix::AbstractString="")
    if res isa RDI.RDISDPResult
        println(prefix, "status          = ", get(res.metadata, :termination_status, :UNKNOWN))
        println(prefix, "moment dimension= ", get(res.metadata, :moment_dim_solver, missing))
        println(prefix, "number of words = ", length(res.words))
        println(prefix, "p_succ          = ", res.psucc)
        println(prefix, "qber            = ", res.qber)
        println(prefix, "entropy value   = ", get(res.metadata, :entropy_value, NaN))
        println(prefix, "entropy bound   = ", get(res.metadata, :entropy_bound, NaN))
        println(prefix, "rate value      = ", get(res.metadata, :rate_value, NaN))
        println(prefix, "rate bound      = ", get(res.metadata, :rate_bound, NaN))
        haskey(res.metadata, :pguess_value) && println(prefix, "pguess value    = ", res.metadata[:pguess_value])
        haskey(res.metadata, :pguess_bound) && println(prefix, "pguess bound    = ", res.metadata[:pguess_bound])
    elseif res isa RDI.BFFPerNodeResult
        println(prefix, "statuses        = ", get(res.metadata, :termination_statuses, missing))
        println(prefix, "number of nodes = ", length(res.node_results))
        println(prefix, "moment dimensions= ", get(res.metadata, :moment_dim_solver_by_node, missing))
        println(prefix, "words per node  = ", get(res.metadata, :number_of_words_by_node, missing))
        println(prefix, "p_succ          = ", res.psucc)
        println(prefix, "qber            = ", res.qber)
        println(prefix, "entropy value   = ", res.entropy_value)
        println(prefix, "entropy bound   = ", res.entropy_bound)
        println(prefix, "rate value      = ", res.rate_value)
        println(prefix, "rate bound      = ", res.rate_bound)
    else
        error("Unsupported result type $(typeof(res))")
    end
    return res
end

end # module RDIQKDTools
