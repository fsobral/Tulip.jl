"""
    compute_step!(ipm, params)

Compute next IP iterate for the QNC formulation.

# Arguments
- `ipm`: The QNC optimizer model
- `params`: Optimization parameters
"""
function compute_step!(mpc::QNC{T, Tv}, params::IPMOptions{T}) where{T, Tv<:AbstractVector{T}}

  # Names
  dat = mpc.dat
  pt = mpc.pt
  res = mpc.res

  m, n, p = pt.m, pt.n, pt.p

  A = dat.A
  b = dat.b
  c = dat.c

  # Compute scaling
  θl = (pt.zl ./ pt.xl) .* dat.lflag
  θu = (pt.zu ./ pt.xu) .* dat.uflag
  θinv = θl .+ θu

  # Update regularizations
  mpc.regP ./= 10
  mpc.regD ./= 10
  clamp!(mpc.regP, sqrt(eps(T)), one(T))
  clamp!(mpc.regD, sqrt(eps(T)), one(T))

  # Update factorization
  nbump = 0
  while nbump <= 3
    try
      @timeit mpc.timer "Factorization" KKT.update!(mpc.kkt, θinv, mpc.regP, mpc.regD)
      break
    catch err
      isa(err, PosDefException) || isa(err, ZeroPivotException) || rethrow(err)

      # Increase regularization
      mpc.regD .*= 100
      mpc.regP .*= 100
      nbump += 1
      @warn "Increase regularizations to $(mpc.regP[1])"
    end
  end
  # TODO: throw a custom error for numerical issues
  nbump < 3 || throw(PosDefException(0))  # factorization could not be saved

  # II. Compute search direction
  Δ  = mpc.Δ
  Δc = mpc.Δc

  # Affine-scaling direction and associated step size
  # Pedro: aqui foi modificado para adotar um alpha unico para ambos os problemas primal e dual

  @timeit mpc.timer "Predictor" compute_predictor!(mpc::QNC)
  mpc.αp, mpc.αd = max_step_length_pd(mpc.pt, mpc.Δ)

  # Reconstruindo z e Δz
#  z = zeros(n)
#  dz = zeros(n)
#  for i=1:n
# #   z[i] = c[i] - dot(A[:, i], pt.y)
#    dz[i] = - dot(A[:, i], Δ.y) # calcule -A'*Δ.y coordenada a coordenada
#  end
#  z = mpc.pt.z

#  mpc.αp, mpc.αd = max_step_length_pd2(mpc.pt, Δ, z, dz) # Espaço para otimizar!
  #mpc.αp, mpc.αd = max_step_length_pd(mpc.pt, mpc.Δ)
  #    if mpc.αp <= mpc.αd
  #        mpc.αd = mpc.αp
  #    else
  #        mpc.αp = mpc.αd
  #    end

  # TODO: if step size is large enough, skip corrector

  # Corrector
#  corretor_qn = true
#  cp_x, cp_y, cp_z = copy(pt.x), copy(pt.y), copy(pt.z) 
  #@timeit mpc.timer "Corrector" all_tests_failed_old = Quasi_Newton_Corrector!(mpc, params)
  @timeit mpc.timer "Corrector" Quasi_Newton_Corrector!(mpc, params)
#  if false#all_tests_failed == true
#    pt.x, pt.y, pt.z = cp_x, cp_y, cp_z
#    #mpc.αp, mpc.αd = max_step_length_pd(mpc.pt, mpc.Δ)
#    compute_corrector!(mpc::MPC)
#    mpc.αp, mpc.αd = max_step_length_pd(mpc.pt, mpc.Δc)
#    corretor_qn = false
#  end

  # TODO: the following is not needed if there are no additional corrections
  copyto!(Δ.x, Δc.x)
  copyto!(Δ.xl, Δc.xl)
  copyto!(Δ.xu, Δc.xu)
  copyto!(Δ.y, Δc.y)
  copyto!(Δ.zl, Δc.zl)
  copyto!(Δ.zu, Δc.zu)

  # Extra centrality corrections
  ncor = 0
  ncor_max = params.CorrectionLimit

  # Zero out the Newton RHS. This only needs to be done once.
  # TODO: not needed if no additional corrections
  rmul!(mpc.ξp, zero(T))
  rmul!(mpc.ξl, zero(T))
  rmul!(mpc.ξu, zero(T))
  rmul!(mpc.ξd, zero(T))

  @timeit mpc.timer "Extra corr" while false#ncor < ncor_max
    compute_extra_correction!(mpc)

    # TODO: function to compute step size given Δ and Δc
    # This would avoid copying data around
    αp_c, αd_c = max_step_length_pd(mpc.pt, mpc.Δc)

    if αp_c >= 1.01 * mpc.αp && αd_c >= 1.01 * mpc.αd
      mpc.αp = αp_c
      mpc.αd = αd_c

      # Δ ⟵ Δc
      copyto!(Δ.x, Δc.x)
      copyto!(Δ.xl, Δc.xl)
      copyto!(Δ.xu, Δc.xu)
      copyto!(Δ.y, Δc.y)
      copyto!(Δ.zl, Δc.zl)
      copyto!(Δ.zu, Δc.zu)

      ncor += 1
    else
      # Not enough improvement: abort
      break
    end
  end

#    if corretor_qn == false
#      # Update current iterate
#      mpc.αp *= params.StepDampFactor
#      mpc.αd *= params.StepDampFactor
#      pt.x  .+= mpc.αp .* Δ.x
#      pt.xl .+= mpc.αp .* Δ.xl
#      pt.xu .+= mpc.αp .* Δ.xu
#      pt.y  .+= mpc.αd .* Δ.y
#      pt.zl .+= mpc.αd .* Δ.zl
#      pt.zu .+= mpc.αd .* Δ.zu
#      update_mu!(pt)
#    else
#      #nothing
#  #    pt.x  .+= Δ.x
#  #    pt.xl .+= Δ.xl
#  #    pt.xu .+= Δ.xu
#  #    pt.y  .+= Δ.y
#  #    pt.zl .+= Δ.zl
#  #    pt.zu .+= Δ.zu
#  #    update_mu!(pt)
#  end

    # Update current iterate
#    mpc.αp *= params.StepDampFactor
#    mpc.αd *= params.StepDampFactor
    pt.x  .+=  Δ.x   # Modificado, pois esta já é a direção resultante ( alpha * direcao de Newton + correcão de Broyden) 
    pt.xl .+=  Δ.xl  #
    pt.xu .+=  Δ.xu  #
    pt.y  .+=  Δ.y   #
    pt.zl .+=  Δ.zl  #
    pt.zu .+=  Δ.zu  #
    update_mu!(pt)

  return nothing
end


"""
    solve_newton_system!(Δ, mpc, ξp, ξd, ξu, ξg, ξxs, ξwz, ξtk)

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
- `mpc`: The MPC optimizer
- `hx, hy, hz, h0`: Terms obtained in the preliminary augmented system solve
- `ξp, ξd, ξu, ξg, ξxs, ξwz, ξtk`: Right-hand side vectors
"""
function solve_newton_system!(Δ::Point{T, Tv},
    mpc::QNC{T, Tv},
    # Right-hand side
    ξp::Tv, ξl::Tv, ξu::Tv, ξd::Tv, ξxzl::Tv, ξxzu::Tv
  ) where{T, Tv<:AbstractVector{T}}

  pt = mpc.pt
  dat = mpc.dat

  # I. Solve augmented system
  @timeit mpc.timer "ξd_"  begin
    ξd_ = copy(ξd)
    @. ξd_ += -((ξxzl + pt.zl .* ξl) ./ pt.xl) .* dat.lflag + ((ξxzu - pt.zu .* ξu) ./ pt.xu) .* dat.uflag
  end
  @timeit mpc.timer "KKT" KKT.solve!(Δ.x, Δ.y, mpc.kkt, ξp, ξd_)

  # II. Recover Δxl, Δxu
  @timeit mpc.timer "Δxl" begin
    @. Δ.xl = (-ξl + Δ.x) * dat.lflag
  end
  @timeit mpc.timer "Δxu" begin
    @. Δ.xu = ( ξu - Δ.x) * dat.uflag
  end

  # III. Recover Δzl, Δzu
  @timeit mpc.timer "Δzl" @. Δ.zl = ((ξxzl - pt.zl .* Δ.xl) ./ pt.xl) .* dat.lflag
  @timeit mpc.timer "Δzu" @. Δ.zu = ((ξxzu - pt.zu .* Δ.xu) ./ pt.xu) .* dat.uflag

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
  αd = min(one(T), azl, azu, αp) # Modificado para usar o mesmo alpha para ambos x e y.
  αp = αd                        #
  # O código original do Tulip não testa se é maior que zero. Talvez tenha que acrescentar isso aqui.

  return αp, αd
end

function max_step_length_pd2(pt, delta, p_z, d_z)
  #    axl = max_step_length(pt.xl, δ.xl)
  #    axu = max_step_length(pt.xu, δ.xu)
  #    azl = max_step_length(pt.zl, δ.zl)
  #    azu = max_step_length(pt.zu, δ.zu)

  alpha0 = 1.0
  for i=1:length(pt.x)
    if delta.x[i] < 0
      temp = - pt.x[i]/delta.x[i]
      if temp < alpha0
        alpha0 = temp
      end
    end
    if d_z[i] < 0
      temp = - p_z[i]/d_z[i]
      if temp < alpha0
        alpha0 = temp
      end
    end
    if alpha0 < 0
      alpha0 = 0
      break
    end

  end

  return alpha0, alpha0
end


"""
    compute_predictor!(mpc::MPC) -> Nothing
"""
function compute_predictor!(mpc::QNC)

  # Newton RHS
  copyto!(mpc.ξp, mpc.res.rp)
  copyto!(mpc.ξl, mpc.res.rl)
  copyto!(mpc.ξu, mpc.res.ru)
  copyto!(mpc.ξd, mpc.res.rd)
  @. mpc.ξxzl = -(mpc.pt.xl .* mpc.pt.zl) .* mpc.dat.lflag
  @. mpc.ξxzu = -(mpc.pt.xu .* mpc.pt.zu) .* mpc.dat.uflag

  # Compute affine-scaling direction
  @timeit mpc.timer "Newton" solve_newton_system!(mpc.Δ, mpc,
                                                  mpc.ξp, mpc.ξl, mpc.ξu, mpc.ξd, mpc.ξxzl, mpc.ξxzu
                                                 )

  # TODO: check Newton system residuals, perform iterative refinement if needed
  return nothing
end

"""
    compute_corrector!(mpc::MPC) -> Nothing
"""
function compute_corrector!(mpc::QNC{T, Tv}, σ, alpha, xlb, xub, zlb, zub) where{T, Tv<:AbstractVector{T}} # Com essa mudança, essa função calcula a primeira direção de Broyden (Jw d = - F_{\sigma \mu}). Preciso apenas tomar o cuidado de calcular os resíduos (primal, dual, gap) de forma correta (no ponto após o passo de Newton)

  # WARNING: Eu acho que o deslocamento que estou fazendo no ponto pode estar alterando a matriz jacobiana, então apenas recalcular os resíduos não vai ser suficiente. 


  dat = mpc.dat
  pt = mpc.pt
  Δ = mpc.Δ
  Δc = mpc.Δc


  # Step length for affine-scaling direction
#  αp_aff, αd_aff = mpc.αp, mpc.αd
#  μₐ = (
#        dot((@. ((pt.xl + αp_aff * Δ.xl) * dat.lflag)), pt.zl .+ αd_aff .* Δ.zl)
#        + dot((@. ((pt.xu + αp_aff * Δ.xu) * dat.uflag)), pt.zu .+ αd_aff .* Δ.zu)
#       ) / pt.p
#  σ = clamp((μₐ / pt.μ)^3, sqrt(eps(T)), one(T) - sqrt(eps(T)))

  # Newton RHS
  # compute_predictor! was called ⟹ ξp, ξl, ξu, ξd are already set
  @. mpc.ξxzl = (σ * pt.μ .- xlb .* zlb) .* dat.lflag
  @. mpc.ξxzu = (σ * pt.μ .- xub .* zub) .* dat.uflag

  # Compute corrector
  @timeit mpc.timer "Newton" solve_newton_system!(mpc.Δc, mpc, mpc.ξp, mpc.ξl, mpc.ξu, mpc.ξd, mpc.ξxzl, mpc.ξxzu)

  # TODO: check Newton system residuals, perform iterative refinement if needed
  return nothing
end

"""
    compute_extra_correction!(mpc) -> Nothing
"""
function compute_extra_correction!(mpc::MPC{T, Tv};
    δ::T = T(3 // 10),
    γ::T = T(1 // 10),
  ) where{T, Tv<:AbstractVector{T}}
  pt = mpc.pt
  Δ  = mpc.Δ
  Δc = mpc.Δc
  dat = mpc.dat

  # Tentative step sizes and centrality parameter
  αp, αd = mpc.αp, mpc.αd
  αp_ = min(αp + δ, one(T))
  αd_ = min(αd + δ, one(T))

  g  = dot(pt.xl, pt.zl) + dot(pt.xu, pt.zu)
  gₐ = dot((@. ((pt.xl + mpc.αp * Δ.xl) * dat.lflag)), pt.zl .+ mpc.αd .* Δ.zl) +
  dot((@. ((pt.xu + mpc.αp * Δ.xu) * dat.uflag)), pt.zu .+ mpc.αd .* Δ.zu)
  μ = (gₐ / g) * (gₐ / g) * (gₐ / pt.p)

  # Newton RHS
  # ξp, ξl, ξu, ξd are already at zero
  @timeit mpc.timer "target" begin
    compute_target!(mpc.ξxzl, pt.xl, Δ.xl, pt.zl, Δ.zl, αp_, αd_, γ, μ)
    compute_target!(mpc.ξxzu, pt.xu, Δ.xu, pt.zu, Δ.zu, αp_, αd_, γ, μ)
  end

  @timeit mpc.timer "Newton" solve_newton_system!(Δc, mpc,
                                                  mpc.ξp, mpc.ξl, mpc.ξu, mpc.ξd, mpc.ξxzl, mpc.ξxzu
                                                 )

  # Δc ⟵ Δp + Δc
  axpy!(one(T), Δ.x, Δc.x)
  axpy!(one(T), Δ.xl, Δc.xl)
  axpy!(one(T), Δ.xu, Δc.xu)
  axpy!(one(T), Δ.y, Δc.y)
  axpy!(one(T), Δ.zl, Δc.zl)
  axpy!(one(T), Δ.zu, Δc.zu)

  # TODO: check Newton residuals
  return nothing
end

"""
    compute_target!(t, x, z, γ, μ)

Compute centrality target.
"""
function compute_target!(
    t::Vector{T},
    x::Vector{T},
    δx::Vector{T},
    z::Vector{T},
    δz::Vector{T},
    αp::T,
    αd::T,
    γ::T,
    μ::T
  ) where{T}

  n = length(t)

  tmin = μ * γ
  tmax = μ / γ

  @inbounds for j in 1:n
    v = (x[j] + αp * δx[j]) * (z[j] + αd * δz[j])
    if v < tmin
      t[j] = tmin - v
    elseif v > tmax
      t[j] = tmax - v
    else
      t[j] = zero(T)
    end
  end

  return nothing
end
