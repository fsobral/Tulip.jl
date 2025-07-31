"""
    compute_step!(ipm, params)

Compute next IP iterate for the QNC formulation.

# Arguments
- `ipm`: The QNC optimizer model
- `params`: Optimization parameters
"""
function compute_step!(qnc::QNC{T, Tv}, params::IPMOptions{T}) where{T, Tv<:AbstractVector{T}}

  # Names
  dat = qnc.dat
  pt = qnc.pt
  res = qnc.res

  m, n, p = pt.m, pt.n, pt.p

  A = dat.A
  b = dat.b
  c = dat.c

  # Compute scaling
  θl = (pt.zl ./ pt.xl) .* dat.lflag
  θu = (pt.zu ./ pt.xu) .* dat.uflag
  θinv = θl .+ θu

  # Update regularizations
  qnc.regP ./= 10
  qnc.regD ./= 10
  clamp!(qnc.regP, sqrt(eps(T)), one(T))
  clamp!(qnc.regD, sqrt(eps(T)), one(T))

  # Update factorization
  nbump = 0
  while nbump <= 3
    try
      @timeit qnc.timer "Factorization" KKT.update!(qnc.kkt, θinv, qnc.regP, qnc.regD)
      break
    catch err
      isa(err, PosDefException) || isa(err, ZeroPivotException) || rethrow(err)

      # Increase regularization
      qnc.regD .*= 100
      qnc.regP .*= 100
      nbump += 1
      @warn "Increase regularizations to $(qnc.regP[1])"
    end
  end
  # TODO: throw a custom error for numerical issues
  nbump < 3 || throw(PosDefException(0))  # factorization could not be saved

  # II. Compute search direction
  Δ  = qnc.Δ
  Δc = qnc.Δc

  # Affine-scaling direction and associated step size

  @timeit qnc.timer "Predictor" compute_predictor!(qnc::QNC)
  qnc.αp, qnc.αd = max_step_length_pd(qnc.pt, qnc.Δ) 

  # TODO: if step size is large enough, skip corrector

  # Corrector
  
  @timeit qnc.timer "Corrector" Quasi_Newton_Corrector!(qnc, params)
  
  # Aqui, Δc já é a soma do passo de Newton e dos passos de Broyden multiplicada pelos seus tamanhos de passo.
  
  copyto!(Δ.x,  Δc.x)
  copyto!(Δ.xl, Δc.xl)
  copyto!(Δ.xu, Δc.xu)
  copyto!(Δ.y,  Δc.y)
  copyto!(Δ.zl, Δc.zl)
  copyto!(Δ.zu, Δc.zu)

#  # Extra centrality corrections
#  ncor = 0
#  ncor_max = params.CorrectionLimit
#
#  # Zero out the Newton RHS. This only needs to be done once.
#  # TODO: not needed if no additional corrections
#  rmul!(qnc.ξp, zero(T))
#  rmul!(qnc.ξl, zero(T))
#  rmul!(qnc.ξu, zero(T))
#  rmul!(qnc.ξd, zero(T))
#
#  @timeit qnc.timer "Extra corr" while false#ncor < ncor_max
#    compute_extra_correction!(qnc)
#
#    # TODO: function to compute step size given Δ and Δc
#    # This would avoid copying data around
#    αp_c, αd_c = max_step_length_pd(qnc.pt, qnc.Δc)
#
#    if αp_c >= 1.01 * qnc.αp && αd_c >= 1.01 * qnc.αd
#      qnc.αp = αp_c
#      qnc.αd = αd_c
#
#      # Δ ⟵ Δc
#      copyto!(Δ.x, Δc.x)
#      copyto!(Δ.xl, Δc.xl)
#      copyto!(Δ.xu, Δc.xu)
#      copyto!(Δ.y, Δc.y)
#      copyto!(Δ.zl, Δc.zl)
#      copyto!(Δ.zu, Δc.zu)
#
#      ncor += 1
#    else
#      # Not enough improvement: abort
#      break
#    end
#  end

  # Update current iterate
  #    qnc.αp *= params.StepDampFactor
  #    qnc.αd *= params.StepDampFactor
  pt.x  .+=  Δ.x   
  pt.xl .+=  Δ.xl  
  pt.xu .+=  Δ.xu  
  pt.y  .+=  Δ.y   
  pt.zl .+=  Δ.zl  
  pt.zu .+=  Δ.zu  
  update_mu!(pt)

  return nothing
end


"""
    solve_newton_system!(Δ, qnc, ξp, ξd, ξu, ξg, ξxs, ξwz, ξtk)

Solve the Newton system
```math
\\begin{bmatrix}
    A & & & R_{d} & & \\\\
    I & -I & & & & \\\\
    I & & I & & & \\\\
    -R_{p} & & & A^{T} & I & -I \\\\
    & Z_{l} & & & X_{l}\\\\
    & & Z_{u} & & & X_{u}\\\\
\\end{bmatrix}
\\begin{bmatrix}
    Δ x\\\\
    Δ x_{l}\\\\
    Δ x_{u}\\\\
    Δ y\\\\
    Δ z_{l} \\\\
    Δ z_{u}
\\end{bmatrix}
=
\\begin{bmatrix}
    ξ_p\\\\
    ξ_l\\\\
    ξ_u\\\\
    ξ_d\\\\
    ξ_{xz}^{l}\\\\
    ξ_{xz}^{u}
\\end{bmatrix}
```

# Arguments
- `Δ`: Search direction, modified
- `qnc`: The MPC optimizer
- `hx, hy, hz, h0`: Terms obtained in the preliminary augmented system solve
- `ξp, ξd, ξu, ξg, ξxs, ξwz, ξtk`: Right-hand side vectors
"""
function solve_newton_system!(Δ::Point{T, Tv},
    qnc::QNC{T, Tv},
    # Right-hand side
    ξp::Tv, ξl::Tv, ξu::Tv, ξd::Tv, ξxzl::Tv, ξxzu::Tv
  ) where{T, Tv<:AbstractVector{T}}

  pt = qnc.pt
  dat = qnc.dat

  # I. Solve augmented system
  @timeit qnc.timer "ξd_"  begin
    ξd_ = copy(ξd)
    @. ξd_ += -((ξxzl + pt.zl .* ξl) ./ pt.xl) .* dat.lflag + ((ξxzu - pt.zu .* ξu) ./ pt.xu) .* dat.uflag
  end
  @timeit qnc.timer "KKT" KKT.solve!(Δ.x, Δ.y, qnc.kkt, ξp, ξd_)

  # II. Recover Δxl, Δxu
  @timeit qnc.timer "Δxl" begin
    @. Δ.xl = (-ξl + Δ.x) * dat.lflag
  end
  @timeit qnc.timer "Δxu" begin
    @. Δ.xu = ( ξu - Δ.x) * dat.uflag
  end

  # III. Recover Δzl, Δzu
  @timeit qnc.timer "Δzl" @. Δ.zl = ((ξxzl - pt.zl .* Δ.xl) ./ pt.xl) .* dat.lflag
  @timeit qnc.timer "Δzu" @. Δ.zu = ((ξxzu - pt.zu .* Δ.xu) ./ pt.xu) .* dat.uflag

  # IV. Set Δτ, Δκ to zero
  Δ.τ = zero(T)
  Δ.κ = zero(T)

  # Check Newton residuals
  # @printf "Newton residuals:\n"
  # @printf "|rp|   = %16.8e\n" norm(dat.A * Δ.x - ξp, Inf)
  # @printf "|rl|   = %16.8e\n" norm((Δ.x - Δ.xl) .* dat.lflag - ξl, Inf)
  # @printf "|ru|   = %16.8e\n" norm((Δ.x + Δ.xu) .* dat.uflag - ξu, Inf)
  # @printf "|rd|   = %16.8e\n" norm(dat.A'Δ.y + Δ.zl - Δ.zu - ξd, Inf)
  # @printf "|rxzl| = %16.8e\n" norm(pt.zl .* Δ.xl + pt.xl .* Δ.zl - ξxzl, Inf)
  # @printf "|rxzu| = %16.8e\n" norm(pt.zu .* Δ.xu + pt.xu .* Δ.zu - ξxzu, Inf)

  return nothing
end

"""
    max_step_length_pd(pt, δ)

Compute maximum primal-dual step length.
"""
function max_step_length_pd(pt::Point{T, Tv}, δ::Point{T, Tv}) where{T, Tv<:AbstractVector{T}} # Eu modifiquei essa funcao
  axl = max_step_length(pt.xl, δ.xl)
  axu = max_step_length(pt.xu, δ.xu)
  azl = max_step_length(pt.zl, δ.zl)
  azu = max_step_length(pt.zu, δ.zu)

  αp = min(one(T), axl, axu)
  αd = min(one(T), azl, azu)
  alpha = min(αp, αd) 
#  αp = αd = alpha # Comentar para usar alphas diferentes                       

  return αp, αd
end

"""
    compute_predictor!(qnc::MPC) -> Nothing
"""
function compute_predictor!(qnc::QNC)

  # Newton RHS
  copyto!(qnc.ξp, qnc.res.rp)
  copyto!(qnc.ξl, qnc.res.rl)
  copyto!(qnc.ξu, qnc.res.ru)
  copyto!(qnc.ξd, qnc.res.rd)
  @. qnc.ξxzl = -(qnc.pt.xl .* qnc.pt.zl) .* qnc.dat.lflag
  @. qnc.ξxzu = -(qnc.pt.xu .* qnc.pt.zu) .* qnc.dat.uflag

  # Compute affine-scaling direction
  @timeit qnc.timer "Newton" solve_newton_system!(qnc.Δ, qnc,
                                                  qnc.ξp, qnc.ξl, qnc.ξu, qnc.ξd, qnc.ξxzl, qnc.ξxzu
                                                 )

  # TODO: check Newton system residuals, perform iterative refinement if needed
  return nothing
end

# ATENÇÃO: a função abaixo não é a original do Tulip. Para ver a original, veja o step.jl do MPC.

#"""
#    compute_corrector!(qnc::MPC) -> Nothing
#"""
#function compute_corrector!(qnc::QNC{T, Tv}, σ, alpha, xlb, xub, zlb, zub) where{T, Tv<:AbstractVector{T}} # Com essa mudança, essa função calcula a primeira direção de Broyden (Jw d = - F_{\sigma \mu}). Preciso apenas tomar o cuidado de calcular os resíduos (primal, dual, gap) de forma correta (no ponto após o passo de Newton)
#
#  # WARNING: Eu acho que o deslocamento que estou fazendo no ponto pode estar alterando a matriz jacobiana, então apenas recalcular os resíduos não vai ser suficiente. 
#
#
#  dat = qnc.dat
#  pt = qnc.pt
#  Δ = qnc.Δ
#  Δc = qnc.Δc
#
#
#  # Step length for affine-scaling direction
#  #  αp_aff, αd_aff = qnc.αp, qnc.αd
#  #  μₐ = (
#  #        dot((@. ((pt.xl + αp_aff * Δ.xl) * dat.lflag)), pt.zl .+ αd_aff .* Δ.zl)
#  #        + dot((@. ((pt.xu + αp_aff * Δ.xu) * dat.uflag)), pt.zu .+ αd_aff .* Δ.zu)
#  #       ) / pt.p
#  #  σ = clamp((μₐ / pt.μ)^3, sqrt(eps(T)), one(T) - sqrt(eps(T)))
#
#  # Newton RHS
#  # compute_predictor! was called ⟹ ξp, ξl, ξu, ξd are already set
#  @. qnc.ξxzl = (σ * pt.μ .- xlb .* zlb) .* dat.lflag
#  @. qnc.ξxzu = (σ * pt.μ .- xub .* zub) .* dat.uflag
#
#  # Compute corrector
#  @timeit qnc.timer "Newton" solve_newton_system!(qnc.Δc, qnc, qnc.ξp, qnc.ξl, qnc.ξu, qnc.ξd, qnc.ξxzl, qnc.ξxzu)
#
#  # TODO: check Newton system residuals, perform iterative refinement if needed
#  return nothing
#end

#"""
#    compute_extra_correction!(qnc) -> Nothing
#"""
#function compute_extra_correction!(qnc::MPC{T, Tv};
#    δ::T = T(3 // 10),
#    γ::T = T(1 // 10),
#  ) where{T, Tv<:AbstractVector{T}}
#  pt = qnc.pt
#  Δ  = qnc.Δ
#  Δc = qnc.Δc
#  dat = qnc.dat
#
#  # Tentative step sizes and centrality parameter
#  αp, αd = qnc.αp, qnc.αd
#  αp_ = min(αp + δ, one(T))
#  αd_ = min(αd + δ, one(T))
#
#  g  = dot(pt.xl, pt.zl) + dot(pt.xu, pt.zu)
#  gₐ = dot((@. ((pt.xl + qnc.αp * Δ.xl) * dat.lflag)), pt.zl .+ qnc.αd .* Δ.zl) +
#  dot((@. ((pt.xu + qnc.αp * Δ.xu) * dat.uflag)), pt.zu .+ qnc.αd .* Δ.zu)
#  μ = (gₐ / g) * (gₐ / g) * (gₐ / pt.p)
#
#  # Newton RHS
#  # ξp, ξl, ξu, ξd are already at zero
#  @timeit qnc.timer "target" begin
#    compute_target!(qnc.ξxzl, pt.xl, Δ.xl, pt.zl, Δ.zl, αp_, αd_, γ, μ)
#    compute_target!(qnc.ξxzu, pt.xu, Δ.xu, pt.zu, Δ.zu, αp_, αd_, γ, μ)
#  end
#
#  @timeit qnc.timer "Newton" solve_newton_system!(Δc, qnc,
#                                                  qnc.ξp, qnc.ξl, qnc.ξu, qnc.ξd, qnc.ξxzl, qnc.ξxzu
#                                                 )
#
#  # Δc ⟵ Δp + Δc
#  axpy!(one(T), Δ.x, Δc.x)
#  axpy!(one(T), Δ.xl, Δc.xl)
#  axpy!(one(T), Δ.xu, Δc.xu)
#  axpy!(one(T), Δ.y, Δc.y)
#  axpy!(one(T), Δ.zl, Δc.zl)
#  axpy!(one(T), Δ.zu, Δc.zu)
#
#  # TODO: check Newton residuals
#  return nothing
#end
#
#"""
#    compute_target!(t, x, z, γ, μ)
#
#Compute centrality target.
#"""
#function compute_target!(
#    t::Vector{T},
#    x::Vector{T},
#    δx::Vector{T},
#    z::Vector{T},
#    δz::Vector{T},
#    αp::T,
#    αd::T,
#    γ::T,
#    μ::T
#  ) where{T}
#
#  n = length(t)
#
#  tmin = μ * γ
#  tmax = μ / γ
#
#  @inbounds for j in 1:n
#    v = (x[j] + αp * δx[j]) * (z[j] + αd * δz[j])
#    if v < tmin
#      t[j] = tmin - v
#    elseif v > tmax
#      t[j] = tmax - v
#    else
#      t[j] = zero(T)
#    end
#  end
#
#  return nothing
#end
