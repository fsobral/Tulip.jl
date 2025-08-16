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

  # Aqui, Δc já é a soma do passo de Newton e dos passos de Broyden
  # multiplicada pelos seus tamanhos de passo.

  copyto!(Δ.x,  Δc.x)
  copyto!(Δ.xl, Δc.xl)
  copyto!(Δ.xu, Δc.xu)
  copyto!(Δ.y,  Δc.y)
  copyto!(Δ.zl, Δc.zl)
  copyto!(Δ.zu, Δc.zu)

  # Update current iterate
   
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
function max_step_length_pd(pt::Point{T, Tv}, δ::Point{T, Tv}) where{T, Tv<:AbstractVector{T}} 
  axl = max_step_length(pt.xl, δ.xl)
  axu = max_step_length(pt.xu, δ.xu)
  azl = max_step_length(pt.zl, δ.zl)
  azu = max_step_length(pt.zu, δ.zu)

  αp = min(one(T), axl, axu)
  αd = min(one(T), azl, azu)

  # Descomentar a linha abaixo irá fazer com que o algoritmo utilize um alpha
  # único tanto para o deslocamento primal quanto dual. Use isso para replicar
  # o comportamento do algoritmo teórico.
  #αp = αd = min(αp, αd)

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

