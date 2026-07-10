
using LinearAlgebra
using SparseArrays
using JuMP
import MathOptInterface as MOI

const BUILDER_FIXED_VERSION = "2026-07-10_always_fixed_bob_blocks_full"

# Optional solvers.  Only the selected solver has to be installed.
const _HAS_MOSEK = Ref(false)
const _HAS_SCS = Ref(false)
const _HAS_SDPAFAMILY = Ref(false)

try
    @eval import MosekTools
    _HAS_MOSEK[] = true
catch
end
try
    @eval import SCS
    _HAS_SCS[] = true
catch
end
try
    @eval import SDPAFamily
    _HAS_SDPAFAMILY[] = true
catch
end

export BUILDER_FIXED_VERSION,
       RDISDPResult,
       BFFPerNodeResult,
       build_rdi_sdp,
       build_rdi_sdp_model,
       build_hmin_sdp,
       build_hmin_sdp_model,
       build_bff_sdp,
       build_bff_sdp_model,
       build_bff_node_sdp,
       build_bff_node_sdp_model,
       build_bff_per_node_sdp,
       build_bff_per_node_sdp_models,
       solve_bff_per_node_sdp!,
       solve_rdi_sdp!,
       all_pairs0,
       amplitudes_from_alphas_betas,
       phase_amplitudes,
       realxz_amplitudes,
       ideal_exclusion_B_matrices,
       statistics_from_amplitudes,
       state_coefficient_matrix,
       can_use_real_moment,
       default_classical_weight,
       success_and_qber_from_P0,
       qber_from_P0,
       binary_entropy2,
       key_rate_from_entropy,
       key_rate_from_pguess,
       gauss_radau_bff_nodes_weights

# =============================================================================
# Result container
# =============================================================================

Base.@kwdef mutable struct RDISDPResult
    model::JuMP.Model
    Γ::Any
    moment_variable::Any
    construction::Symbol
    field::Symbol
    n::Int
    d::Int
    amplitudes::Matrix{ComplexF64}
    P0::Matrix{Float64}
    word_blocks::Vector{String}
    words::Vector{String}
    word_factors::Dict{String,Vector{String}}
    word_index::Dict{String,Int}
    index_map::Dict{Tuple{Int,String},Int}
    pairs::Vector{Tuple{Int,Int}}
    psucc::Float64
    qber::Union{Nothing,Float64} = nothing
    objective_kind::Symbol = :unknown
    objective_constant::Float64 = 0.0
    bff_branch_mode::Union{Nothing,Symbol} = nothing
    bff_nodes::Vector{Float64} = Float64[]
    bff_weights::Vector{Float64} = Float64[]
    metadata::Dict{Symbol,Any} = Dict{Symbol,Any}()
end

Base.@kwdef mutable struct BFFPerNodeResult
    node_results::Vector{RDISDPResult}
    bff_nodes::Vector{Float64}
    bff_weights::Vector{Float64}
    psucc::Float64 = NaN
    qber::Union{Nothing,Float64} = nothing
    entropy_value::Float64 = NaN
    entropy_bound::Float64 = NaN
    rate_value::Union{Nothing,Float64} = nothing
    rate_bound::Union{Nothing,Float64} = nothing
    metadata::Dict{Symbol,Any} = Dict{Symbol,Any}()
end

# =============================================================================
# Preparation and statistics helpers
# =============================================================================

function all_pairs0(n::Int)
    n >= 2 || error("n must be at least 2")
    pairs = Tuple{Int,Int}[]
    for i in 0:n-2, j in i+1:n-1
        push!(pairs, (i,j))
    end
    return pairs
end

function amplitudes_from_alphas_betas(alphas::AbstractVector, betas::AbstractVector; normalize::Bool=true)
    length(alphas) == length(betas) || error("alphas and betas must have the same length")
    n = length(alphas)
    A = Matrix{ComplexF64}(undef, 2, n)
    for x in 1:n
        A[1,x] = ComplexF64(alphas[x])
        A[2,x] = ComplexF64(betas[x])
    end
    if normalize
        for x in 1:n
            nr = norm(A[:,x])
            nr > 0 || error("state $x has zero norm")
            A[:,x] ./= nr
        end
    end
    return A
end

"Phase encoding: |psi_x> = cos(theta/2)|0> + exp(-2π i x/n) sin(theta/2)|1>, x=0,...,n-1."
function phase_amplitudes(n::Int, θ::Real; phase_sign::Real=-1)
    c = cos(float(θ)/2)
    s = sin(float(θ)/2)
    α = ComplexF64[c for _ in 1:n]
    β = ComplexF64[exp(im * float(phase_sign) * 2π * (x-1) / n) * s for x in 1:n]
    return amplitudes_from_alphas_betas(α, β; normalize=true)
end

"Real X-Z encoding: |psi_x> = cos(delta_x)|0> + sin(delta_x)|1>."
function realxz_amplitudes(n::Int, θ::Real; deltas=nothing)
    δs = deltas === nothing ? collect(LinRange(0.0, float(θ), n)) : Float64.(collect(deltas))
    length(δs) == n || error("deltas must have length n")
    α = [cos(δs[x]) for x in 1:n]
    β = [sin(δs[x]) for x in 1:n]
    return amplitudes_from_alphas_betas(α, β; normalize=true)
end

function _parse_amplitudes(; amplitudes=nothing, alphas=nothing, betas=nothing,
                             preparation_function=nothing, n::Union{Nothing,Int}=nothing,
                             normalize::Bool=true)
    data = amplitudes
    if preparation_function !== nothing
        n === nothing && error("n must be provided when using preparation_function")
        data = preparation_function(n)
    end

    if data !== nothing
        if data isa NamedTuple
            if haskey(data, :amplitudes)
                return _parse_amplitudes(amplitudes=data.amplitudes; normalize=normalize)
            elseif haskey(data, :alphas) && haskey(data, :betas)
                return amplitudes_from_alphas_betas(data.alphas, data.betas; normalize=normalize)
            else
                error("preparation NamedTuple must contain either :amplitudes or both :alphas and :betas")
            end
        elseif data isa Tuple && length(data) == 2
            return amplitudes_from_alphas_betas(data[1], data[2]; normalize=normalize)
        elseif data isa AbstractMatrix
            A = Matrix{ComplexF64}(data)
            if size(A,1) == 2
                # already 2 × n
            elseif size(A,2) == 2
                A = transpose(A)
                A = Matrix{ComplexF64}(A)
            else
                error("amplitudes must be a 2×n matrix, an n×2 matrix, a tuple (alphas, betas), or a NamedTuple")
            end
            if normalize
                for x in 1:size(A,2)
                    nr = norm(A[:,x])
                    nr > 0 || error("state $x has zero norm")
                    A[:,x] ./= nr
                end
            end
            return A
        else
            error("Unsupported amplitude input")
        end
    end

    if alphas !== nothing && betas !== nothing
        return amplitudes_from_alphas_betas(alphas, betas; normalize=normalize)
    end
    error("Provide amplitudes, or alphas and betas, or preparation_function with n")
end

"Coefficient matrix C_x[a,b] = conj(c_x[a]) c_x[b] used in <psi_x|M|psi_x>."
function state_coefficient_matrix(amplitudes::AbstractMatrix, x::Int)
    c = amplitudes[:,x]
    d = length(c)
    C = Matrix{ComplexF64}(undef, d, d)
    for a in 1:d, b in 1:d
        C[a,b] = conj(c[a]) * c[b]
    end
    return C
end

"Noisy ideal exclusion matrices B_y = eta*((1-lambda)*(I-|psi_y><psi_y|) + lambda*I/2)."
function ideal_exclusion_B_matrices(amplitudes::AbstractMatrix; η::Real=1.0, λ::Real=0.0)
    A = Matrix{ComplexF64}(amplitudes)
    d, n = size(A)
    d == 2 || error("This builder currently expects qubit amplitudes, i.e. a 2×n matrix")
    I2 = Matrix{ComplexF64}(I, 2, 2)
    out = Vector{Matrix{ComplexF64}}(undef, n)
    ηf = float(η)
    λf = float(λ)
    for y in 1:n
        ψ = A[:,y]
        proj = ψ * ψ' # |psi_y><psi_y|, entries c_a conj(c_b)
        out[y] = ηf * ((1 - λf) * (I2 - proj) + (λf/2) * I2)
    end
    return out
end

"Compute P0[x,y] = <psi_x|B_y|psi_x>."
function statistics_from_amplitudes(amplitudes::AbstractMatrix; η::Real=1.0, λ::Real=0.0,
                                    B_matrices=nothing)
    A = Matrix{ComplexF64}(amplitudes)
    d, n = size(A)
    B = B_matrices === nothing ? ideal_exclusion_B_matrices(A; η=η, λ=λ) : B_matrices
    length(B) == n || error("B_matrices must have length n")
    P0 = zeros(Float64, n, n)
    for x in 1:n, y in 1:n
        size(B[y]) == (d,d) || error("B_matrices[$y] has wrong size")
        ψ = A[:,x]
        P0[x,y] = real(dot(ψ, B[y] * ψ))
    end
    return P0
end

function default_classical_weight(n::Int, r_index::Int, k::Int, y::Int)
    return 1.0 / (n^2 * (n-1))
end

function success_and_qber_from_P0(n::Int, P0::AbstractMatrix; classical_weight::Function=default_classical_weight)
    size(P0) == (n,n) || error("P0 must be n×n")
    pairs = all_pairs0(n)
    ps = 0.0
    pe = 0.0
    for (ridx,(r0,r1)) in enumerate(pairs)
        for k0 in 0:1
            x0 = k0 == 0 ? r0 : r1
            for y0 in (r0,r1)
                p = classical_weight(n, ridx, k0, y0) * float(P0[x0+1,y0+1])
                ps += p
                if y0 == x0
                    pe += p
                end
            end
        end
    end
    qber = ps > 0 ? clamp(pe / ps, 0.0, 0.5) : NaN
    return ps, qber
end

qber_from_P0(n::Int, P0::AbstractMatrix; classical_weight::Function=default_classical_weight) =
    success_and_qber_from_P0(n, P0; classical_weight=classical_weight)[2]

function binary_entropy2(q::Real)
    qf = clamp(float(q), 0.0, 1.0)
    (qf == 0.0 || qf == 1.0) && return 0.0
    return -qf*log2(qf) - (1-qf)*log2(1-qf)
end

key_rate_from_entropy(psucc::Real, Hbound::Real, qber::Real) = float(psucc) * (float(Hbound) - binary_entropy2(qber))
key_rate_from_pguess(psucc::Real, pguess::Real, qber::Real) = key_rate_from_entropy(psucc, -log2(clamp(float(pguess), eps(Float64), 1.0)), qber)

_is_real_array(A; tol=1e-12) = maximum(abs, imag.(ComplexF64.(A))) <= tol

function can_use_real_moment(amplitudes::AbstractMatrix; extra_matrices=Any[], tol::Real=1e-12)
    _is_real_array(amplitudes; tol=tol) || return false
    for M in extra_matrices
        _is_real_array(M; tol=tol) || return false
    end
    return true
end

# =============================================================================
# Word machinery
# =============================================================================

B_label(y::Int) = "B[y=$(y)]"
E_label(r::Int) = "E[r=$(r)]"
BE_label(y::Int,r::Int) = "BE[y=$(y),r=$(r)]"
BB_label(y::Int,y2::Int) = "BB[y=$(y),y2=$(y2)]"
EE_label(r::Int,r2::Int) = "EE[r=$(r),r2=$(r2)]"

_z_suffix(node::Int, r::Int, k::Int) = "i=$(node),r=$(r),k=$(k)"
Z_label(node::Int,r::Int,k::Int) = "Z[$(_z_suffix(node,r,k))]"
Zdag_label(node::Int,r::Int,k::Int) = "Zdag[$(_z_suffix(node,r,k))]"
ZdagZ_label(node::Int,r::Int,k::Int) = "ZdagZ[$(_z_suffix(node,r,k))]"
BZ_label(y::Int,node::Int,r::Int,k::Int) = "BZ[y=$(y),$(_z_suffix(node,r,k))]"
BZdag_label(y::Int,node::Int,r::Int,k::Int) = "BZdag[y=$(y),$(_z_suffix(node,r,k))]"
BZdagZ_label(y::Int,node::Int,r::Int,k::Int) = "BZdagZ[y=$(y),$(_z_suffix(node,r,k))]"

_isB(f::String) = startswith(f, "B:")
_isE(f::String) = startswith(f, "E:")
_isZ(f::String) = startswith(f, "Z:") || startswith(f, "Zdag:")

function _adjoint_atom(f::String)
    if startswith(f,"Zdag:")
        return replace(f, "Zdag:" => "Z:", count=1)
    elseif startswith(f,"Z:")
        return replace(f, "Z:" => "Zdag:", count=1)
    else
        return f # B and E are Hermitian projectors
    end
end

_adjoint_factors(factors::Vector{String}) = [_adjoint_atom(f) for f in reverse(factors)]

function _canonical_factors(factors::Vector{String}; commute_B_E::Bool=true, commute_B_Z::Bool=true)
    # Only commute Bob factors through E/Z factors.  We do NOT commute different
    # Bob inputs with each other, different Eve inputs with each other, or Z with
    # Zdag.  In particular, Zdag*Z and Z*Zdag remain distinct.
    fs = copy(factors)
    changed = true
    while changed
        changed = false
        for i in 1:max(length(fs)-1, 0)
            left, right = fs[i], fs[i+1]
            if _isB(right) && ((commute_B_E && _isE(left)) || (commute_B_Z && _isZ(left)))
                fs[i], fs[i+1] = fs[i+1], fs[i]
                changed = true
            end
        end
    end

    # Projector idempotency only for adjacent identical B or E atoms.
    out = String[]
    for f in fs
        if !isempty(out) && out[end] == f && (_isB(f) || _isE(f))
            continue
        end
        push!(out, f)
    end
    return out
end

function _product_key(factors::Vector{String}; commute_B_E::Bool=true, commute_B_Z::Bool=true)
    fs = _canonical_factors(factors; commute_B_E=commute_B_E, commute_B_Z=commute_B_Z)
    isempty(fs) && return "Id"
    return join(fs, "*")
end

function _register_word!(words::Vector{String}, factors::Dict{String,Vector{String}}, label::String, fs::Vector{String})
    if !haskey(factors, label)
        push!(words, label)
        factors[label] = fs
    end
    return label
end


function _atom_kind(f::String)
    startswith(f, "B:") && return :B
    startswith(f, "E:") && return :E
    startswith(f, "Zdag:") && return :Zdag
    startswith(f, "Z:") && return :Z
    error("Unknown atom factor '$f'")
end

_atom_payload(f::String) = split(f, ":", limit=2)[2]
_parse_atom_int(f::String) = parse(Int, _atom_payload(f))

function _parse_z_indices(f::String)
    s = _atom_payload(f)
    m = match(r"i=(\d+),r=(\d+),k=(\d+)", s)
    m === nothing && error("Could not parse Z atom '$f'")
    return (parse(Int, m.captures[1]), parse(Int, m.captures[2]), parse(Int, m.captures[3]))
end

function _word_label_from_factors(fs_in::Vector{String})
    fs = _canonical_factors(fs_in; commute_B_E=true, commute_B_Z=true)
    isempty(fs) && return "Id"

    kinds = [_atom_kind(f) for f in fs]

    if length(fs) == 1
        if kinds[1] == :B
            return B_label(_parse_atom_int(fs[1]))
        elseif kinds[1] == :E
            return E_label(_parse_atom_int(fs[1]))
        elseif kinds[1] == :Z
            i,r,k = _parse_z_indices(fs[1])
            return Z_label(i,r,k)
        elseif kinds[1] == :Zdag
            i,r,k = _parse_z_indices(fs[1])
            return Zdag_label(i,r,k)
        end
    elseif length(fs) == 2
        if kinds == [:B, :E]
            return BE_label(_parse_atom_int(fs[1]), _parse_atom_int(fs[2]))
        elseif kinds == [:B, :B]
            return BB_label(_parse_atom_int(fs[1]), _parse_atom_int(fs[2]))
        elseif kinds == [:E, :E]
            return EE_label(_parse_atom_int(fs[1]), _parse_atom_int(fs[2]))
        elseif kinds == [:B, :Z]
            y = _parse_atom_int(fs[1])
            i,r,k = _parse_z_indices(fs[2])
            return BZ_label(y,i,r,k)
        elseif kinds == [:B, :Zdag]
            y = _parse_atom_int(fs[1])
            i,r,k = _parse_z_indices(fs[2])
            return BZdag_label(y,i,r,k)
        elseif kinds == [:Zdag, :Z]
            z1 = _parse_z_indices(fs[1])
            z2 = _parse_z_indices(fs[2])
            if z1 == z2
                i,r,k = z1
                return ZdagZ_label(i,r,k)
            end
        end
    elseif length(fs) == 3
        if kinds == [:B, :Zdag, :Z]
            z1 = _parse_z_indices(fs[2])
            z2 = _parse_z_indices(fs[3])
            if z1 == z2
                y = _parse_atom_int(fs[1])
                i,r,k = z1
                return BZdagZ_label(y,i,r,k)
            end
        end
    end

    # Fallback label for genuinely general words, e.g. BBE/BEE/BBEE.
    # The factor list itself is stored separately in word_factors, so this label
    # only has to be unique and readable.
    return "W[" * join(fs, "*") * "]"
end

function _parse_general_word_pattern(block::AbstractString)
    b = replace(strip(block), " " => "")
    isempty(b) && error("Empty word block")
    b == "Id" && return Symbol[]

    raw_tokens = String[]
    if occursin("*", b)
        raw_tokens = split(b, "*")
        any(isempty, raw_tokens) && error("Malformed word block '$block'")
    else
        pos = firstindex(b)
        while pos <= lastindex(b)
            if startswith(b[pos:end], "Zdag")
                push!(raw_tokens, "Zdag")
                pos = nextind(b, pos, 4)
            else
                c = b[pos]
                if c == 'B' || c == 'E' || c == 'Z'
                    push!(raw_tokens, string(c))
                    pos = nextind(b, pos)
                elseif c == 'D'
                    # Convenience alias: D means Zdag in compact strings.
                    push!(raw_tokens, "Zdag")
                    pos = nextind(b, pos)
                else
                    error("Could not parse general word block '$block'. Use tokens B, E, Z, Zdag, optionally separated by '*'.")
                end
            end
        end
    end

    tokens = Symbol[]
    for t in raw_tokens
        if t == "B"
            push!(tokens, :B)
        elseif t == "E"
            push!(tokens, :E)
        elseif t == "Z"
            push!(tokens, :Z)
        elseif t in ("Zdag", "D")
            push!(tokens, :Zdag)
        elseif t == "Id"
            # Id inside a product is ignored.
        else
            error("Unsupported token '$t' in word block '$block'. Allowed tokens: B, E, Z, Zdag.")
        end
    end
    return tokens
end

function _choices_for_token(tok::Symbol, n::Int, pairs::Vector{Tuple{Int,Int}};
                            construction::Symbol, nodes::Vector{Float64})
    nr = length(pairs)
    if tok == :B
        return [["B:$y"] for y in 1:n]
    elseif tok == :E
        return [["E:$r"] for r in 1:nr]
    elseif tok == :Z || tok == :Zdag
        construction == :bff || error("The token $(tok) is only valid for construction=:bff")
        isempty(nodes) && error("BFF token $(tok) requires nonempty quadrature nodes")
        prefix = tok == :Z ? "Z" : "Zdag"
        return [["$prefix:$(_z_suffix(inode,r,k))"] for inode in eachindex(nodes), r in 1:nr, k in 1:2]
    else
        error("Unsupported token $(tok)")
    end
end

function _register_general_word_pattern!(words::Vector{String}, factors::Dict{String,Vector{String}},
                                         block::AbstractString, n::Int, pairs::Vector{Tuple{Int,Int}};
                                         construction::Symbol, nodes::Vector{Float64})
    tokens = _parse_general_word_pattern(block)
    isempty(tokens) && return

    choices_by_token = [_choices_for_token(tok, n, pairs; construction=construction, nodes=nodes) for tok in tokens]

    function rec(pos::Int, acc::Vector{String})
        if pos > length(choices_by_token)
            canon = _canonical_factors(acc; commute_B_E=true, commute_B_Z=true)

            # If a repeated projector collapsed the word, the corresponding case
            # is already represented by a lower-order word.  We therefore skip it
            # instead of silently adding, e.g., B_y B_y E_r as BE.
            length(canon) < length(acc) && return

            label = _word_label_from_factors(canon)
            _register_word!(words, factors, label, canon)
            return
        end
        for choice in choices_by_token[pos]
            rec(pos + 1, vcat(acc, choice))
        end
    end

    rec(1, String[])
    return
end

function _expand_word_blocks(word_blocks::AbstractVector{<:AbstractString}, n::Int, pairs::Vector{Tuple{Int,Int}};
                             construction::Symbol, nodes::Vector{Float64})
    words = String[]
    factors = Dict{String,Vector{String}}()

    # The identity is always needed for normalization and first-row entries.
    _register_word!(words, factors, "Id", String[])

    nr = length(pairs)
    for raw in word_blocks
        block = String(strip(String(raw)))
        block == "Id" && continue

        if block == "B"
            for y in 1:n
                _register_word!(words, factors, B_label(y), ["B:$y"])
            end
        elseif block == "E"
            for r in 1:nr
                _register_word!(words, factors, E_label(r), ["E:$r"])
            end
        elseif block == "BE"
            for y in 1:n, r in 1:nr
                _register_word!(words, factors, BE_label(y,r), ["B:$y", "E:$r"])
            end
        elseif block == "BB"
            for y in 1:n, y2 in 1:n
                y == y2 && continue
                _register_word!(words, factors, BB_label(y,y2), ["B:$y", "B:$y2"])
            end
        elseif block == "EE"
            for r in 1:nr, r2 in 1:nr
                r == r2 && continue
                _register_word!(words, factors, EE_label(r,r2), ["E:$r", "E:$r2"])
            end
        elseif block in ("Z", "Zdag", "ZdagZ", "BZ", "BZdag", "BZdagZ")
            construction == :bff || error("Word block $block is only valid for construction=:bff")
            isempty(nodes) && error("BFF word block $block requires nonempty quadrature nodes")
            for inode in eachindex(nodes), r in 1:nr, k in 1:2
                zs = _z_suffix(inode, r, k)
                z = "Z:$zs"
                zd = "Zdag:$zs"
                if block == "Z"
                    _register_word!(words, factors, Z_label(inode,r,k), [z])
                elseif block == "Zdag"
                    _register_word!(words, factors, Zdag_label(inode,r,k), [zd])
                elseif block == "ZdagZ"
                    _register_word!(words, factors, ZdagZ_label(inode,r,k), [zd, z])
                elseif block == "BZ"
                    for y in 1:n
                        _register_word!(words, factors, BZ_label(y,inode,r,k), ["B:$y", z])
                    end
                elseif block == "BZdag"
                    for y in 1:n
                        _register_word!(words, factors, BZdag_label(y,inode,r,k), ["B:$y", zd])
                    end
                elseif block == "BZdagZ"
                    for y in 1:n
                        _register_word!(words, factors, BZdagZ_label(y,inode,r,k), ["B:$y", zd, z])
                    end
                end
            end
        else
            # General fallback.  Examples:
            #   "B*B*E" or "BBE"      -> B_y1 B_y2 E_r, with y1 != y2
            #   "B*E*E" or "BEE"      -> B_y E_r1 E_r2, with r1 != r2
            #   "B*B*E*E" or "BBEE"  -> B_y1 B_y2 E_r1 E_r2
            #   "E*B"                 -> canonicalized to B*E
            _register_general_word_pattern!(words, factors, block, n, pairs;
                                            construction=construction, nodes=nodes)
        end
    end

    return words, factors
end

# =============================================================================
# Sparse picker matrices and real-block complex lift
# =============================================================================

function real_block(A::SparseMatrixCSC{ComplexF64,Int})
    m, n = size(A)
    I0, J0, V = findnz(A)
    VR, VI = real.(V), imag.(V)
    I2 = [I0; I0; m .+ I0; m .+ I0]
    J2 = [J0; n .+ J0; J0; n .+ J0]
    V2 = [VR; -VI; VI; VR]
    return sparse(I2, J2, V2, 2m, 2n)
end
real_block(A::AbstractMatrix{ComplexF64}) = real_block(sparse(A))

function F_re_ij(N::Int, i::Int, j::Int)
    S = spzeros(ComplexF64, N, N)
    if i == j
        S[i,i] = 1.0
    else
        S[i,j] = 0.5
        S[j,i] = 0.5
    end
    return S
end

function F_im_ij(N::Int, i::Int, j::Int)
    if i == j
        return spzeros(ComplexF64, N, N)
    end
    S = spzeros(ComplexF64, N, N)
    S[i,j] = -0.5im
    S[j,i] =  0.5im
    return S
end

function F_sym_ij(N::Int, i::Int, j::Int)
    S = spzeros(Float64, N, N)
    if i == j
        S[i,i] = 1.0
    else
        S[i,j] = 0.5
        S[j,i] = 0.5
    end
    return S
end

function _add_complex_weighted_entry!(F::SparseMatrixCSC{ComplexF64,Int}, N::Int, i::Int, j::Int, c::ComplexF64)
    iszero(c) && return
    # Represents Re(c * Γ[i,j]).
    F .+= real(c) * F_re_ij(N,i,j) - imag(c) * F_im_ij(N,i,j)
    return
end

function _add_complex_weighted_im_entry!(F::SparseMatrixCSC{ComplexF64,Int}, N::Int, i::Int, j::Int, c::ComplexF64)
    iszero(c) && return
    # Represents Im(c * Γ[i,j]) = Re((-im*c) * Γ[i,j]).
    _add_complex_weighted_entry!(F, N, i, j, -1im*c)
    return
end

# =============================================================================
# Solver and quadrature
# =============================================================================

function _optimizer_factory(solver::Symbol; silent::Bool=true, sdpa_presolve::Bool=true, sdpa_params=nothing)
    if solver == :mosek
        _HAS_MOSEK[] || error("MosekTools.jl is not installed/loaded. Choose solver=:scs or install MosekTools.")
        return MosekTools.Optimizer
    elseif solver == :scs
        _HAS_SCS[] || error("SCS.jl is not installed/loaded. Install SCS.jl or choose another solver.")
        return SCS.Optimizer
    elseif solver in (:sdpa_gmp, :sdpa_qd, :sdpa_dd, :sdpa)
        _HAS_SDPAFAMILY[] || error("SDPAFamily.jl is not installed/loaded. Install it with `Pkg.add(\"SDPAFamily\")`.")
        variant = solver == :sdpa_gmp ? :sdpa_gmp : solver == :sdpa_qd ? :sdpa_qd : solver == :sdpa_dd ? :sdpa_dd : :sdpa
        if sdpa_params === nothing
            return () -> SDPAFamily.Optimizer{BigFloat}(variant=variant, silent=silent, presolve=sdpa_presolve)
        else
            return () -> SDPAFamily.Optimizer{BigFloat}(variant=variant, silent=silent, presolve=sdpa_presolve, params=sdpa_params)
        end
    else
        error("Unknown solver $(solver). Supported: :mosek, :scs, :sdpa_gmp, :sdpa_qd, :sdpa_dd, :sdpa.")
    end
end

function _gauss_legendre_unit_nodes_weights(n_nodes::Int)
    # Pure-Julia fallback on [0,1].  This is not Radau, but it keeps the BFF
    # code runnable without FastGaussQuadrature.jl.  Install
    # FastGaussQuadrature.jl for the intended Gauss-Radau rule.
    n_nodes >= 1 || error("n_nodes must be at least 1")
    if n_nodes == 1
        return [0.5], [1.0]
    end
    diag = zeros(Float64, n_nodes)
    off = [i / sqrt(4.0*i^2 - 1.0) for i in 1:n_nodes-1]
    E = eigen(SymTridiagonal(diag, off))
    x = E.values
    w = 2.0 .* abs2.(E.vectors[1, :])
    t = (x .+ 1.0) ./ 2.0
    wt = w ./ 2.0
    perm = sortperm(t)
    return collect(t[perm]), collect(wt[perm])
end

function gauss_radau_bff_nodes_weights(n_nodes::Int)
    n_nodes >= 1 || error("n_nodes must be at least 1")
    try
        @eval import FastGaussQuadrature
        # Intended rule: request one extra Radau node and drop the endpoint, so
        # the number of active BFF nodes is exactly n_nodes.
        x, w = FastGaussQuadrature.gaussradau(n_nodes + 1, 0.0, 0.0)
        t = (1.0 .- x) ./ 2.0
        wt = w ./ 2.0
        nodes = collect(t[2:end])
        weights = collect(wt[2:end])
        length(nodes) == n_nodes || error("Internal Gauss-Radau node construction failed")
        return nodes, weights
    catch err
        @warn "FastGaussQuadrature Gauss-Radau nodes unavailable; using pure-Julia Gauss-Legendre fallback. Install FastGaussQuadrature.jl for production BFF runs." exception=(err, catch_backtrace())
        return _gauss_legendre_unit_nodes_weights(n_nodes)
    end
end

function _resolve_bff_nodes_weights(; bff_n_nodes::Int=4, bff_nodes=nothing, bff_weights=nothing)
    if bff_nodes === nothing || bff_weights === nothing
        return gauss_radau_bff_nodes_weights(bff_n_nodes)
    else
        nodes = Float64.(collect(bff_nodes))
        weights = Float64.(collect(bff_weights))
        length(nodes) == length(weights) || error("bff_nodes and bff_weights must have the same length")
        all(0.0 .< nodes .< 1.0) || @warn "Some BFF nodes are outside (0,1); check your quadrature input." nodes=nodes
        return nodes, weights
    end
end

# =============================================================================
# Main build and solve
# =============================================================================

function build_rdi_sdp_model(; amplitudes=nothing,
                               alphas=nothing,
                               betas=nothing,
                               preparation_function=nothing,
                               n::Union{Nothing,Int}=nothing,
                               normalize_states::Bool=true,
                               P0=nothing,
                               statistics_function=nothing,
                               η::Real=1.0,
                               λ::Real=0.0,
                               eta::Union{Nothing,Real}=nothing,
                               lambda::Union{Nothing,Real}=nothing,
                               construction::Symbol=:hmin,
                               word_blocks=nothing,
                               field::Symbol=:auto,
                               solver::Symbol=:mosek,
                               silent::Bool=true,
                               qber::Union{Nothing,Real}=nothing,
                               psucc::Union{Nothing,Real}=nothing,
                               classical_weight::Function=default_classical_weight,
                               bff_n_nodes::Int=4,
                               bff_nodes::Union{Nothing,AbstractVector}=nothing,
                               bff_weights::Union{Nothing,AbstractVector}=nothing,
                               bff_constant::Union{Nothing,Real}=nothing,
                               bff_add_norm_bounds::Bool=true,
                               bff_add_product_bounds::Bool=true,
                               deduplicate_product_constraints::Bool=true,
                               sdpa_presolve::Bool=true,
                               sdpa_params=nothing,
                               metadata=Dict{Symbol,Any}())
    construction in (:hmin, :bff) || error("construction must be :hmin or :bff")
    metadata = Dict{Symbol,Any}(metadata)

    # Accept both Greek and ASCII spellings.  If both are supplied, the ASCII
    # aliases deliberately override because they are usually what the caller
    # intended in ordinary Julia scripts.
    eta !== nothing && (η = eta)
    lambda !== nothing && (λ = lambda)

    A = _parse_amplitudes(amplitudes=amplitudes, alphas=alphas, betas=betas,
                          preparation_function=preparation_function, n=n,
                          normalize=normalize_states)
    d, nstates = size(A)
    d == 2 || error("This compact builder currently expects qubit amplitudes, i.e. a 2×n matrix")

    # Observed statistics are the constraints.  They may be provided manually,
    # returned by a function, or generated from the ideal/noisy exclusion model.
    P = nothing
    if P0 !== nothing
        P = Matrix{Float64}(P0)
    elseif statistics_function !== nothing
        P = Matrix{Float64}(statistics_function(A))
    else
        P = statistics_from_amplitudes(A; η=η, λ=λ)
    end
    size(P) == (nstates,nstates) || error("P0 must be n×n with n=size(amplitudes,2)")

    # Bob's full qubit blocks are always fixed for every construction.
    Bfixed = ideal_exclusion_B_matrices(A; η=η, λ=λ)

    if field == :auto
        field = can_use_real_moment(A; extra_matrices=Any[Bfixed...]) ? :real : :complex
    end
    field in (:real, :complex) || error("field must be :auto, :real, or :complex")

    pairs = all_pairs0(nstates)
    nr = length(pairs)

    nodes = Float64[]
    weights = Float64[]
    cm = 0.0
    branch_mode = nothing
    if construction == :bff
        nodes, weights = _resolve_bff_nodes_weights(bff_n_nodes=bff_n_nodes, bff_nodes=bff_nodes, bff_weights=bff_weights)
        cm = bff_constant === nothing ? sum(weights[i] / (nodes[i] * log(2)) for i in eachindex(nodes)) : float(bff_constant)
        branch_mode = :rk
    end

    if word_blocks === nothing
        word_blocks = construction == :hmin ? ["Id", "B", "E", "BE"] : ["Id", "B", "Z", "Zdag", "BZ", "BZdag"]
    end
    blocks = String.(collect(word_blocks))
    words, word_factors = _expand_word_blocks(blocks, nstates, pairs; construction=construction, nodes=nodes)
    word_index = Dict(w => i for (i,w) in enumerate(words))

    # Required words for objectives/constraints.
    haskey(word_index, B_label(1)) || error("word_blocks must include \"B\" because B words are needed for the statistics constraints")
    if construction == :hmin
        haskey(word_index, E_label(1)) || error("H_min requires word_blocks to include \"E\"")
    else
        haskey(word_index, Z_label(1,1,1)) || error("BFF requires word_blocks to include \"Z\"")
        haskey(word_index, BZ_label(1,1,1,1)) || error("BFF requires word_blocks to include \"BZ\" so that <BZ> and <Z^dag B Z> are represented correctly")
        haskey(word_index, Zdag_label(1,1,1)) || error("BFF marginal objective requires word_blocks to include \"Zdag\"")
        haskey(word_index, BZdag_label(1,1,1,1)) || error("BFF marginal objective requires word_blocks to include \"BZdag\"")
    end

    ns = length(words)
    N = d * ns
    IA(a::Int, sidx::Int) = (a-1)*ns + sidx
    IA_word(a::Int, w::String) = IA(a, word_index[w])
    index_map = Dict{Tuple{Int,String},Int}()
    for a in 1:d, w in words
        index_map[(a,w)] = IA_word(a,w)
    end

    opt = _optimizer_factory(solver; silent=silent, sdpa_presolve=sdpa_presolve, sdpa_params=sdpa_params)
    model = Model(opt)
    silent && set_silent(model)

    if field == :real
        @variable(model, Γvar[1:N,1:N], Symmetric)
        @constraint(model, Γvar in PSDCone())
    else
        @variable(model, Γvar[1:(2*N),1:(2*N)], Symmetric)
        @constraint(model, Γvar in PSDCone())
    end
    moment_variable = Γvar

    constraint_count = Ref(0)
    inequality_count = Ref(0)
    skipped_zero_constraints = Ref(0)
    tie_seen = Set{Tuple{Int,Int,Int,Int}}()

    function expr_from_F(F)
        if field == :real
            return dot(F, Γvar)
        else
            return dot(0.5 * real_block(F), Γvar)
        end
    end

    function add_eq_F!(F, rhs::Real)
        if nnz(F) == 0
            abs(float(rhs)) < 1e-12 || error("Zero LHS equality with nonzero RHS $rhs")
            skipped_zero_constraints[] += 1
            return
        end
        @constraint(model, expr_from_F(F) == float(rhs))
        constraint_count[] += 1
    end

    function add_le_F!(F, rhs::Real)
        nnz(F) == 0 && return
        @constraint(model, expr_from_F(F) <= float(rhs))
        inequality_count[] += 1
    end

    function add_ge_F!(F, rhs::Real)
        nnz(F) == 0 && return
        @constraint(model, expr_from_F(F) >= float(rhs))
        inequality_count[] += 1
    end

    function add_entry_eq!(i::Int, j::Int, value)
        z = ComplexF64(value)
        if field == :real
            abs(imag(z)) <= 1e-10 || error("Trying to impose imaginary entry in a real moment matrix: $z")
            add_eq_F!(F_sym_ij(N,i,j), real(z))
        else
            add_eq_F!(F_re_ij(N,i,j), real(z))
            add_eq_F!(F_im_ij(N,i,j), imag(z))
        end
    end

    function add_tie!(i::Int,j::Int,k::Int,l::Int)
        key = (i,j,k,l)
        if deduplicate_product_constraints && key in tie_seen
            return
        end
        push!(tie_seen, key)
        if field == :real
            add_eq_F!(F_sym_ij(N,i,j) - F_sym_ij(N,k,l), 0.0)
        else
            add_eq_F!(F_re_ij(N,i,j) - F_re_ij(N,k,l), 0.0)
            add_eq_F!(F_im_ij(N,i,j) - F_im_ij(N,k,l), 0.0)
        end
    end

    function weighted_F_re(x::Int, wL::String, wR::String)
        sL = word_index[wL]
        sR = word_index[wR]
        C = state_coefficient_matrix(A, x) # conj(c_a)c_b
        if field == :real
            F = spzeros(Float64, N, N)
            for a in 1:d, b in 1:d
                c = ComplexF64(C[a,b])
                abs(imag(c)) > 1e-10 && error("Complex preparation coefficient in real mode")
                iszero(real(c)) && continue
                F .+= real(c) * F_sym_ij(N, IA(a,sL), IA(b,sR))
            end
            return F
        else
            F = spzeros(ComplexF64, N, N)
            for a in 1:d, b in 1:d
                c = ComplexF64(C[a,b])
                _add_complex_weighted_entry!(F, N, IA(a,sL), IA(b,sR), c)
            end
            return F
        end
    end

    function weighted_F_im(x::Int, wL::String, wR::String)
        field == :real && return spzeros(Float64, N, N)
        sL = word_index[wL]
        sR = word_index[wR]
        C = state_coefficient_matrix(A, x)
        F = spzeros(ComplexF64, N, N)
        for a in 1:d, b in 1:d
            c = ComplexF64(C[a,b])
            _add_complex_weighted_im_entry!(F, N, IA(a,sL), IA(b,sR), c)
        end
        return F
    end

    weighted_expr(x::Int, wL::String, wR::String) = expr_from_F(weighted_F_re(x,wL,wR))

    # Identity block <a|I|b> = delta_ab.
    Id = "Id"
    for a in 1:d, b in 1:d
        add_entry_eq!(IA_word(a,Id), IA_word(b,Id), a == b ? 1.0 : 0.0)
    end

    # Statistics constraints: <psi_x|B_y|psi_x> = P0[x,y].
    # These are the observed statistics constraints.
    for x in 1:nstates, y in 1:nstates
        add_eq_F!(weighted_F_re(x, Id, B_label(y)), P[x,y])
    end

    # Fix all four computational-basis entries of every Bob effect.
    # This is unconditional, so H_min, combined BFF, and per-node BFF all use
    # exactly the same fixed Bob blocks.
    for y in 1:nstates
        By = Bfixed[y]
        wy = B_label(y)
        for a in 1:d, b in 1:d
            add_entry_eq!(IA_word(a,Id), IA_word(b,wy), By[a,b])
            add_entry_eq!(IA_word(a,wy), IA_word(b,Id), conj(By[b,a]))
        end
    end

    # Generic word-logic constraints.  If two entries correspond to the same
    # canonical product S_i^dag S_j, tie them in every computational block.
    pair_rep = Dict{String,Tuple{Int,Int}}()
    for i in 1:ns, j in 1:ns
        wi, wj = words[i], words[j]
        fi = _adjoint_factors(word_factors[wi])
        fj = word_factors[wj]
        key = _product_key(vcat(fi, fj); commute_B_E=true, commute_B_Z=true)
        if !haskey(pair_rep, key)
            pair_rep[key] = (i,j)
        else
            i0,j0 = pair_rep[key]
            for a in 1:d, b in 1:d
                add_tie!(IA(a,i), IA(b,j), IA(a,i0), IA(b,j0))
            end
        end
    end

    ps_calc, qber_calc = success_and_qber_from_P0(nstates, P; classical_weight=classical_weight)
    ps = psucc === nothing ? ps_calc : float(psucc)
    qber_used = qber === nothing ? qber_calc : float(qber)
    ps > 0 || error("p_succ <= 0; no successful rounds")
    if psucc !== nothing && abs(ps - ps_calc) > 1e-7
        @warn "Provided psucc differs from psucc computed from P0" provided=ps computed=ps_calc
    end
    if qber !== nothing && isfinite(qber_calc) && abs(qber_used - qber_calc) > 1e-7
        @warn "Provided qber differs from qber computed from P0" provided=qber_used computed=qber_calc
    end

    objective_kind = construction == :hmin ? :guessing_numerator : :bff_entropy_bound
    obj = JuMP.AffExpr(0.0)
    obj_constant = 0.0

    if construction == :hmin
        # Maximize p(e=k, succ).  Conditional p_guess is objective/psucc.
        for (ridx,(r0,r1)) in enumerate(pairs)
            E0 = E_label(ridx) # E means Eve guesses k=0; k=1 is I-E.
            for y0 in (r0,r1)
                y = y0 + 1
                w0 = classical_weight(nstates, ridx, 0, y0)
                obj += w0 * weighted_expr(r0+1, B_label(y), E0)

                w1 = classical_weight(nstates, ridx, 1, y0)
                obj += w1 * weighted_expr(r1+1, B_label(y), Id)
                obj += -w1 * weighted_expr(r1+1, B_label(y), E0)
            end
        end
        @objective(model, Max, obj)
    else
        obj_constant = cm
        obj += cm
        base_norm = 1.0 / ps
        for inode in eachindex(nodes)
            t = nodes[inode]
            τ = weights[inode] / (t * log(2))
            κ = 1.5 * max(1.0/t, 1.0/(1.0-t))
            κ2 = κ^2

            for (ridx,(r0,r1)) in enumerate(pairs)
                for k0 in 0:1
                    x0 = k0 == 0 ? r0 : r1
                    k1 = k0 + 1
                    Z = Z_label(inode,ridx,k1)
                    Zdag = Zdag_label(inode,ridx,k1)
                    # Key-state terms: these live on the successful branch for the
                    # same key k, represented by B_y-localized moments.
                    for y0 in (r0,r1)
                        y = y0 + 1
                        BZ = BZ_label(y,inode,ridx,k1)
                        BZdag = BZdag_label(y,inode,ridx,k1)
                        scale = τ * base_norm * classical_weight(nstates, ridx, k0, y0)

                        # Linear terms written explicitly as <B_y Z> + <B_y Z^dag>.
                        # With correct moment Hermiticity this equals
                        # Gamma[Id,BZ] + Gamma[BZ,Id], but the explicit BZdag form
                        # is easier to audit.
                        obj += scale * weighted_expr(x0+1, Id, BZ)
                        obj += scale * weighted_expr(x0+1, Id, BZdag)

                        # (1-t) term on the same key branch:
                        #   <Z^dag B_y Z> = Gamma[Z,BZ].
                        obj += scale * (1.0 - t) * weighted_expr(x0+1, Z, BZ)
                    end

                    # Marginal conditioning term: this is the important correction.
                    # For each fixed Z_{i,r,k}, the t term is evaluated on the
                    # successful marginal over the conditioning system, so it sums
                    # over ell and y in r:
                    #   t * sum_{ell,y} p(r,ell,y) <Z B_y Z^dag>_{rho_{r_ell}}
                    # represented as Gamma[Zdag, B_y Zdag].
                    for ell0 in 0:1
                        xell = ell0 == 0 ? r0 : r1
                        for yy0 in (r0,r1)
                            yy = yy0 + 1
                            BZdag = BZdag_label(yy,inode,ridx,k1)
                            scale_marg = τ * base_norm * classical_weight(nstates, ridx, ell0, yy0)
                            obj += scale_marg * t * weighted_expr(xell+1, Zdag, BZdag)
                        end
                    end
                end
            end

            if bff_add_norm_bounds
                for x in 1:nstates, ridx in 1:nr, k1 in 1:2
                    Z = Z_label(inode,ridx,k1)
                    Zdag = Zdag_label(inode,ridx,k1)

                    # Bound both <Z^dag Z> and <Z Z^dag>.  The corrected
                    # objective uses both orientations through the successful
                    # localizing terms.
                    add_le_F!(weighted_F_re(x, Z, Z) - κ2 * weighted_F_re(x, Id, Id), 0.0)
                    add_le_F!(weighted_F_re(x, Zdag, Zdag) - κ2 * weighted_F_re(x, Id, Id), 0.0)

                    # These diagonal/localizing terms should be real.
                    if field == :complex
                        add_eq_F!(weighted_F_im(x, Z, Z), 0.0)
                        add_eq_F!(weighted_F_im(x, Zdag, Zdag), 0.0)
                    end
                end
            end

            if bff_add_product_bounds
                for x in 1:nstates, y in 1:nstates, ridx in 1:nr, k1 in 1:2
                    pxy = float(P[x,y])
                    Z = Z_label(inode,ridx,k1)
                    Zdag = Zdag_label(inode,ridx,k1)
                    BZ = BZ_label(y,inode,ridx,k1)
                    BZdag = BZdag_label(y,inode,ridx,k1)

                    # Bound Re <B_y Z> and Re <B_y Z^dag>.
                    add_ge_F!(weighted_F_re(x, Id, BZ), -κ * pxy)
                    add_le_F!(weighted_F_re(x, Id, BZ),  κ * pxy)
                    add_ge_F!(weighted_F_re(x, Id, BZdag), -κ * pxy)
                    add_le_F!(weighted_F_re(x, Id, BZdag),  κ * pxy)

                    # Bound <Z^dag B_y Z>, represented as Gamma[Z,BZ], and
                    # <Z B_y Z^dag>, represented as Gamma[Zdag,BZdag].
                    add_ge_F!(weighted_F_re(x, Z, BZ), -κ2 * pxy)
                    add_le_F!(weighted_F_re(x, Z, BZ),  κ2 * pxy)
                    add_ge_F!(weighted_F_re(x, Zdag, BZdag), -κ2 * pxy)
                    add_le_F!(weighted_F_re(x, Zdag, BZdag),  κ2 * pxy)
                    if field == :complex
                        add_eq_F!(weighted_F_im(x, Z, BZ), 0.0)
                        add_eq_F!(weighted_F_im(x, Zdag, BZdag), 0.0)
                    end
                end
            end
        end
        @objective(model, Min, obj)
    end

    result = RDISDPResult(model=model,
                          Γ=Γvar,
                          moment_variable=moment_variable,
                          construction=construction,
                          field=field,
                          n=nstates,
                          d=d,
                          amplitudes=A,
                          P0=P,
                          word_blocks=blocks,
                          words=words,
                          word_factors=word_factors,
                          word_index=word_index,
                          index_map=index_map,
                          pairs=pairs,
                          psucc=ps,
                          qber=qber_used,
                          objective_kind=objective_kind,
                          objective_constant=obj_constant,
                          bff_branch_mode=branch_mode,
                          bff_nodes=nodes,
                          bff_weights=weights,
                          metadata=merge(metadata, Dict(
                              :ns => ns,
                              :moment_dim_complex => N,
                              :moment_dim_solver => field == :real ? N : 2*N,
                              :constraint_count => constraint_count[],
                              :inequality_count => inequality_count[],
                              :skipped_zero_constraints => skipped_zero_constraints[],
                              :product_classes => length(pair_rep),
                              :solver => solver,
                              :statistics_source => P0 !== nothing ? :manual_P0 : statistics_function !== nothing ? :statistics_function : :ideal_exclusion_from_amplitudes,
                              :fix_bob_blocks => true,
                              :bob_blocks_mode => :always_fixed,
                              :psucc_from_P0 => ps_calc,
                              :qber_from_P0 => qber_calc,
                              :qber_used => qber_used,
                              :eta => float(η),
                              :lambda => float(λ),
                              :has_mosek => _HAS_MOSEK[],
                              :has_scs => _HAS_SCS[],
                              :has_sdpafamily => _HAS_SDPAFAMILY[]
                          )))
    return result
end

function _as_symbol_any_dict(x)
    x === nothing && return Dict{Symbol,Any}()
    if x isa AbstractDict
        return Dict{Symbol,Any}(Symbol(k) => v for (k,v) in x)
    else
        error("metadata must be a dictionary-like object with Symbol/string keys")
    end
end

function _canonicalize_builder_kwargs(kwargs; fixed::Dict{Symbol,Any}=Dict{Symbol,Any}())
    d = Dict{Symbol,Any}()
    for (k,v) in kwargs
        kk = k === :eta ? :η : k === :lambda ? :λ : Symbol(k)
        if kk === :metadata
            d[kk] = _as_symbol_any_dict(v)
        else
            d[kk] = v
        end
    end
    for (k,v) in fixed
        d[k] = k === :metadata ? _as_symbol_any_dict(v) : v
    end
    return (; d...)
end

"Build and solve directly.  Use build_rdi_sdp_model if you only want to inspect the SDP."
function build_rdi_sdp(; kwargs...)
    kw = _canonicalize_builder_kwargs(kwargs)
    res = build_rdi_sdp_model(; kw...)
    solve_rdi_sdp!(res)
    return res
end

"Build only the H_min SDP model."
function build_hmin_sdp_model(; word_blocks=nothing, kwargs...)
    wb = word_blocks === nothing ? ["Id", "B", "E", "BE"] : word_blocks
    kw = _canonicalize_builder_kwargs(kwargs; fixed=Dict(:construction=>:hmin, :word_blocks=>wb))
    return build_rdi_sdp_model(; kw...)
end

"Build and solve the H_min SDP."
function build_hmin_sdp(; kwargs...)
    res = build_hmin_sdp_model(; kwargs...)
    solve_rdi_sdp!(res)
    return res
end

"Build only the BFF/von-Neumann entropy SDP model."
function build_bff_sdp_model(; word_blocks=nothing, kwargs...)
    wb = word_blocks === nothing ? ["Id", "B", "Z", "Zdag", "BZ", "BZdag"] : word_blocks
    kw = _canonicalize_builder_kwargs(kwargs; fixed=Dict(:construction=>:bff, :word_blocks=>wb))
    return build_rdi_sdp_model(; kw...)
end

"Build and solve the BFF/von-Neumann entropy SDP."
function build_bff_sdp(; kwargs...)
    res = build_bff_sdp_model(; kwargs...)
    solve_rdi_sdp!(res)
    return res
end

"Build only the BFF SDP corresponding to one quadrature node."
function build_bff_node_sdp_model(; node::Real, weight::Real, word_blocks=nothing, bff_constant::Union{Nothing,Real}=nothing, kwargs...)
    wb = word_blocks === nothing ? ["Id", "B", "Z", "Zdag", "BZ", "BZdag"] : word_blocks
    t = float(node)
    w = float(weight)
    c = bff_constant === nothing ? w / (t * log(2)) : float(bff_constant)
    return build_bff_sdp_model(; kwargs..., word_blocks=wb, bff_nodes=[t], bff_weights=[w], bff_constant=c)
end

"Build and solve the BFF SDP corresponding to one quadrature node."
function build_bff_node_sdp(; kwargs...)
    res = build_bff_node_sdp_model(; kwargs...)
    solve_rdi_sdp!(res)
    return res
end

function _bff_node_constant_share(total_constant, nodes::Vector{Float64}, weights::Vector{Float64}, i::Int)
    default_constants = [weights[j] / (nodes[j] * log(2)) for j in eachindex(nodes)]
    total_default = sum(default_constants)
    if total_constant === nothing
        return default_constants[i]
    elseif total_default == 0.0
        return float(total_constant) / length(nodes)
    else
        return float(total_constant) * default_constants[i] / total_default
    end
end

"Build one independent BFF SDP model per quadrature node, without solving them."
function build_bff_per_node_sdp_models(; word_blocks=nothing,
                                        bff_n_nodes::Int=4,
                                        bff_nodes=nothing,
                                        bff_weights=nothing,
                                        bff_constant::Union{Nothing,Real}=nothing,
                                        kwargs...)
    wb = word_blocks === nothing ? ["Id", "B", "Z", "Zdag", "BZ", "BZdag"] : word_blocks
    nodes, weights = _resolve_bff_nodes_weights(bff_n_nodes=bff_n_nodes, bff_nodes=bff_nodes, bff_weights=bff_weights)
    node_results = RDISDPResult[]
    for i in eachindex(nodes)
        ci = _bff_node_constant_share(bff_constant, nodes, weights, i)
        res_i = build_bff_node_sdp_model(; kwargs...,
            node=nodes[i],
            weight=weights[i],
            bff_constant=ci,
            word_blocks=wb,
            metadata=Dict{Symbol,Any}(:bff_per_node => true, :bff_node_local_index => i, :bff_node => nodes[i], :bff_weight => weights[i], :bff_node_constant => ci))
        push!(node_results, res_i)
    end
    ps = isempty(node_results) ? NaN : node_results[1].psucc
    qb = isempty(node_results) ? nothing : node_results[1].qber
    return BFFPerNodeResult(node_results=node_results,
                            bff_nodes=nodes,
                            bff_weights=weights,
                            psucc=ps,
                            qber=qb,
                            metadata=Dict{Symbol,Any}(:mode => :bff_per_node_models,
                                          :n_nodes => length(nodes),
                                          :solved => false,
                                          :node_constants => [_bff_node_constant_share(bff_constant, nodes, weights, i) for i in eachindex(nodes)]))
end

function _finite_sum_or_nan(vals)
    all(isfinite, vals) ? sum(vals) : NaN
end

"Solve every node SDP in a BFFPerNodeResult and aggregate the entropy/rate."
function solve_bff_per_node_sdp!(res::BFFPerNodeResult)
    for node_res in res.node_results
        if !haskey(node_res.metadata, :termination_status)
            solve_rdi_sdp!(node_res)
        end
    end

    obj_vals = [get(r.metadata, :objective_value, NaN) for r in res.node_results]
    obj_bounds = [get(r.metadata, :objective_bound, NaN) for r in res.node_results]
    ent_vals = [get(r.metadata, :entropy_value, NaN) for r in res.node_results]
    ent_bounds = [get(r.metadata, :entropy_bound, NaN) for r in res.node_results]

    # For BFF each node objective already includes its own constant contribution,
    # so the per-node entropy bound is the sum of the per-node optima.
    Hval_raw = _finite_sum_or_nan(obj_vals)
    Hbound_raw = _finite_sum_or_nan(obj_bounds)
    Hval = isfinite(Hval_raw) ? clamp(Hval_raw, 0.0, 1.0) : NaN
    Hbound = isfinite(Hbound_raw) ? clamp(Hbound_raw, 0.0, 1.0) : NaN

    res.entropy_value = Hval
    res.entropy_bound = Hbound
    res.psucc = isempty(res.node_results) ? NaN : res.node_results[1].psucc
    res.qber = isempty(res.node_results) ? nothing : res.node_results[1].qber
    if res.qber !== nothing
        res.rate_value = key_rate_from_entropy(res.psucc, Hval, res.qber)
        res.rate_bound = key_rate_from_entropy(res.psucc, Hbound, res.qber)
    end

    res.metadata[:mode] = :bff_per_node_solved
    res.metadata[:solved] = true
    res.metadata[:objective_values_by_node] = obj_vals
    res.metadata[:objective_bounds_by_node] = obj_bounds
    res.metadata[:entropy_values_by_node] = ent_vals
    res.metadata[:entropy_bounds_by_node] = ent_bounds
    res.metadata[:termination_statuses] = [get(r.metadata, :termination_status, missing) for r in res.node_results]
    res.metadata[:primal_statuses] = [get(r.metadata, :primal_status, missing) for r in res.node_results]
    res.metadata[:dual_statuses] = [get(r.metadata, :dual_status, missing) for r in res.node_results]
    res.metadata[:moment_dim_solver_by_node] = [get(r.metadata, :moment_dim_solver, missing) for r in res.node_results]
    res.metadata[:number_of_words_by_node] = [length(r.words) for r in res.node_results]
    res.metadata[:entropy_value_raw] = Hval_raw
    res.metadata[:entropy_bound_raw] = Hbound_raw
    res.metadata[:rate_value] = res.rate_value
    res.metadata[:rate_bound] = res.rate_bound
    res.metadata[:entropy_value] = res.entropy_value
    res.metadata[:entropy_bound] = res.entropy_bound
    res.metadata[:psucc] = res.psucc
    res.metadata[:qber] = res.qber
    return res
end

"Build and solve one independent BFF SDP per quadrature node, then sum the node objectives."
function build_bff_per_node_sdp(; kwargs...)
    res = build_bff_per_node_sdp_models(; kwargs...)
    solve_bff_per_node_sdp!(res)
    return res
end

function solve_rdi_sdp!(res::RDISDPResult)
    optimize!(res.model)
    status = termination_status(res.model)
    pstatus = primal_status(res.model)
    dstatus = dual_status(res.model)
    obj_val = try objective_value(res.model) catch; NaN end
    obj_bound = try objective_bound(res.model) catch
        try MOI.get(res.model, MOI.ObjectiveBound()) catch; NaN end
    end

    if res.construction == :hmin
        pg_val = isfinite(obj_val) ? clamp(obj_val / res.psucc, 0.0, 1.0) : NaN
        pg_bound = isfinite(obj_bound) ? clamp(obj_bound / res.psucc, 0.0, 1.0) : NaN
        H_val = isfinite(pg_val) && pg_val > 0 ? -log2(pg_val) : NaN
        H_bound = isfinite(pg_bound) && pg_bound > 0 ? -log2(pg_bound) : NaN
        res.metadata[:pguess_value] = pg_val
        res.metadata[:pguess_bound] = pg_bound
        res.metadata[:entropy_value] = H_val
        res.metadata[:entropy_bound] = H_bound
        if res.qber !== nothing
            res.metadata[:rate_value] = key_rate_from_entropy(res.psucc, H_val, res.qber)
            res.metadata[:rate_bound] = key_rate_from_entropy(res.psucc, H_bound, res.qber)
        end
    else
        # The BFF objective is an entropy lower-bound expression after the
        # Gauss-Radau approximation.  We keep the raw value and clamp only for
        # rate reporting.
        H_val = isfinite(obj_val) ? clamp(obj_val, 0.0, 1.0) : NaN
        H_bound = isfinite(obj_bound) ? clamp(obj_bound, 0.0, 1.0) : NaN
        res.metadata[:entropy_value] = H_val
        res.metadata[:entropy_bound] = H_bound
        if res.qber !== nothing
            res.metadata[:rate_value] = key_rate_from_entropy(res.psucc, H_val, res.qber)
            res.metadata[:rate_bound] = key_rate_from_entropy(res.psucc, H_bound, res.qber)
        end
    end

    res.metadata[:termination_status] = status
    res.metadata[:primal_status] = pstatus
    res.metadata[:dual_status] = dstatus
    res.metadata[:objective_value] = obj_val
    res.metadata[:objective_bound] = obj_bound
    res.metadata[:solve_time] = try MOI.get(res.model, MOI.SolveTimeSec()) catch; NaN end
    return res
end

end # module RDIQKDSDPBuilderGeneralWordsPerNodeV2MarginalBFF
