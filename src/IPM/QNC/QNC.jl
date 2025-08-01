"""
    QNC

Implements Quasi-Newton correctors.
"""
mutable struct QNC{T, Tv, Tb, Ta, Tk} <: AbstractIPMOptimizer{T}

  # Problem data, in standard form
  dat::IPMData{T, Tv, Tb, Ta}

  # =================
  #   Book-keeping
  # =================
  niter::Int                        # Number of IPM iterations
  solver_status::TerminationStatus  # Optimization status
  primal_status::SolutionStatus     # Primal solution status
  dual_status::SolutionStatus       # Dual   solution status

  primal_objective::T  # Primal bound: c'x
  dual_objective::T    # Dual bound: b'y + l' zl - u'zu

  timer::TimerOutput

  n_tent_broyden::Int # numero de tentativas do broyden
  nitb::Int           # numero total de iterações de Broyden
  n_corr_alt::Int     # numero total de utilizações do método alternativo
  n_corr_jac::Int     # numero total de correções da jacobiana

  #=====================
  Working memory
  =====================#
  pt::Point{T, Tv}       # Current primal-dual iterate
  pt_cp::Point{T, Tv}       # Current primal-dual iterate copy
  res::Residuals{T, Tv}  # Residuals at current iterate

  Δ::Point{T, Tv}   # Predictor
  Δc::Point{T, Tv}  # Corrector

  # Step sizes
  αp::T
  αd::T

  # Newton system RHS
  ξp::Tv
  ξl::Tv
  ξu::Tv
  ξd::Tv
  ξxzl::Tv
  ξxzu::Tv

  # KKT solver
  kkt::Tk
  regP::Tv  # Primal regularization
  regD::Tv  # Dual regularization

  function QNC(
      dat::IPMData{T, Tv, Tb, Ta}, kkt_options::KKTOptions{T}
    ) where{T, Tv<:AbstractVector{T}, Tb<:AbstractVector{Bool}, Ta<:AbstractMatrix{T}}

    m, n = dat.nrow, dat.ncol
    p = sum(dat.lflag) + sum(dat.uflag)

    # Working memory
    pt  = Point{T, Tv}(m, n, p, hflag=false)
    pt_cp  = Point{T, Tv}(m, n, p, hflag=false)
    res = Residuals(
                    tzeros(Tv, m), tzeros(Tv, n), tzeros(Tv, n),
                    tzeros(Tv, n), zero(T),
                    zero(T), zero(T), zero(T), zero(T), zero(T)
                   )
    Δ  = Point{T, Tv}(m, n, p, hflag=false)
    Δc = Point{T, Tv}(m, n, p, hflag=false)

    # Newton RHS
    ξp = tzeros(Tv, m)
    ξl = tzeros(Tv, n)
    ξu = tzeros(Tv, n)
    ξd = tzeros(Tv, n)
    ξxzl = tzeros(Tv, n)
    ξxzu = tzeros(Tv, n)

    # Initial regularizations
    regP = tones(Tv, n)
    regD = tones(Tv, m)

    kkt = KKT.setup(dat.A, kkt_options.System, kkt_options.Backend)
    Tk = typeof(kkt)

    return new{T, Tv, Tb, Ta, Tk}(dat,
                                  0, Trm_Unknown, Sln_Unknown, Sln_Unknown,
                                  T(Inf), T(-Inf),
                                  TimerOutput(),
                                  0, 0, 0, 0,
                                  pt, pt_cp, res, Δ, Δc, zero(T), zero(T),
                                  ξp, ξl, ξu, ξd, ξxzl, ξxzu,
                                  kkt, regP, regD
                                 )
  end

end

include("step.jl")

"""
    compute_residuals!(::QNC)

In-place computation of primal-dual residuals at point `pt`.
"""
function compute_residuals!(qnc::QNC{T}) where{T}

  pt, res = qnc.pt, qnc.res
  dat = qnc.dat

  # Primal residual
  # rp = b - A*x
  res.rp .= dat.b
  mul!(res.rp, dat.A, pt.x, -one(T), one(T))

  # Lower-bound residual
  # rl_j = l_j - (x_j - xl_j)  if l_j ∈ R
  #      = 0                   if l_j = -∞
  @. res.rl = ((dat.l + pt.xl) - pt.x) * dat.lflag

  # Upper-bound residual
  # ru_j = u_j - (x_j + xu_j)  if u_j ∈ R
  #      = 0                   if u_j = +∞
  @. res.ru = (dat.u - (pt.x + pt.xu)) * dat.uflag

  # Dual residual
  # rd = c - (A'y + zl - zu)
  res.rd .= dat.c
  mul!(res.rd, transpose(dat.A), pt.y, -one(T), one(T))
  @. res.rd += pt.zu .* dat.uflag - pt.zl .* dat.lflag

  # Residuals norm
  res.rp_nrm = norm(res.rp, Inf)
  res.rl_nrm = norm(res.rl, Inf)
  res.ru_nrm = norm(res.ru, Inf)
  res.rd_nrm = norm(res.rd, Inf)

  # Compute primal and dual bounds
  qnc.primal_objective = dot(dat.c, pt.x) + dat.c0
  qnc.dual_objective   = (
                          dot(dat.b, pt.y)
                          + dot(dat.l .* dat.lflag, pt.zl)
                          - dot(dat.u .* dat.uflag, pt.zu)
                         ) + dat.c0

  return nothing
end


"""
    update_solver_status!()

Update status and return true if solver should stop.
"""
function update_solver_status!(qnc::QNC{T}, ϵp::T, ϵd::T, ϵg::T, ϵi::T) where{T}
  qnc.solver_status = Trm_Unknown

  pt, res = qnc.pt, qnc.res
  dat = qnc.dat

  ρp = max(
           res.rp_nrm / (one(T) + norm(dat.b, Inf)),
           res.rl_nrm / (one(T) + norm(dat.l .* dat.lflag, Inf)),
           res.ru_nrm / (one(T) + norm(dat.u .* dat.uflag, Inf))
          )
  ρd = res.rd_nrm / (one(T) + norm(dat.c, Inf))
  ρg = abs(qnc.primal_objective - qnc.dual_objective) / (one(T) + abs(qnc.primal_objective))

  #params.OutputLevel > 0 && println("ρd = $(ρd)\nρp = $(ρp)\nρg = $(ρg)")
  #params.OutputLevel > 0 && println("αd = $(qnc.αd)\nαp = $(qnc.αp)")
  #params.OutputLevel > 0 && println("μ = $(qnc.pt.μ)")

  # Check for feasibility
  if ρp <= ϵp
    qnc.primal_status = Sln_FeasiblePoint
  else
    qnc.primal_status = Sln_Unknown
  end

  if ρd <= ϵd
    qnc.dual_status = Sln_FeasiblePoint
  else
    qnc.dual_status = Sln_Unknown
  end

  # Check for optimal solution
  if ρp <= ϵp && ρd <= ϵd && ρg <= ϵg
    qnc.primal_status = Sln_Optimal
    qnc.dual_status   = Sln_Optimal
    qnc.solver_status = Trm_Optimal
    return nothing
  end

  # TODO: Primal/Dual infeasibility detection
  # Check for infeasibility certificates
  if max(
         norm(dat.A * pt.x, Inf),
         norm((pt.x .- pt.xl) .* dat.lflag, Inf),
         norm((pt.x .+ pt.xu) .* dat.uflag, Inf)
        ) * (norm(dat.c, Inf) / max(1, norm(dat.b, Inf))) < - ϵi * dot(dat.c, pt.x)
    # Dual infeasible, i.e., primal unbounded
    qnc.primal_status = Sln_InfeasibilityCertificate
    qnc.solver_status = Trm_DualInfeasible
    return nothing
  end

  δ = dat.A' * pt.y .+ (pt.zl .* dat.lflag) .- (pt.zu .* dat.uflag)
  if norm(δ, Inf) * max(
                        norm(dat.l .* dat.lflag, Inf),
                        norm(dat.u .* dat.uflag, Inf),
                        norm(dat.b, Inf)
                       ) / (max(one(T), norm(dat.c, Inf)))  < (dot(dat.b, pt.y) + dot(dat.l .* dat.lflag, pt.zl)- dot(dat.u .* dat.uflag, pt.zu)) * ϵi
    # Primal infeasible
    qnc.dual_status = Sln_InfeasibilityCertificate
    qnc.solver_status = Trm_PrimalInfeasible
    return nothing
  end

  return nothing
end

"""
    optimize!

"""
function ipm_optimize!(qnc::QNC{T}, params::IPMOptions{T}) where{T}
  # TODO: pre-check whether model needs to be re-optimized.
  # This should happen outside of this function
  dat = qnc.dat

  # Initialization
  TimerOutputs.reset_timer!(qnc.timer)
  tstart = time()
  qnc.niter = 0
  qnc.nitb = 0
  qnc.n_corr_alt = 0
  qnc.n_corr_jac = 0
  qnc.n_tent_broyden = 0

  # Print information about the problem
  if params.OutputLevel > 0
    @printf "\nOptimizer info (MPC)\n"
    @printf "Constraints  : %d\n" dat.nrow
    @printf "Variables    : %d\n" dat.ncol
    bmin, bmax = extrema(dat.b)
    @printf "RHS          : [%+.2e, %+.2e]\n" bmin bmax
    lmin, lmax = extrema(dat.l .* dat.lflag)
    @printf "Lower bounds : [%+.2e, %+.2e]\n" lmin lmax
    lmin, lmax = extrema(dat.u .* dat.uflag)
    @printf "Upper bounds : [%+.2e, %+.2e]\n" lmin lmax


    @printf "\nLinear solver options\n"
    @printf "  %-12s : %s\n" "Arithmetic" KKT.arithmetic(qnc.kkt)
    @printf "  %-12s : %s\n" "Backend" KKT.backend(qnc.kkt)
    @printf "  %-12s : %s\n" "System" KKT.linear_system(qnc.kkt)
  end

  # IPM LOG
  if params.OutputLevel > 0
    @printf "\n%4s  %14s  %14s  %8s %8s %8s  %7s  %4s\n" "Itn" "PObj" "DObj" "PFeas" "DFeas" "GFeas" "Mu" "Time"
  end

  # Set starting point
  @timeit qnc.timer "Initial point" compute_starting_point(qnc)

  # Main loop
  # Iteration 0 corresponds to the starting point.
  # Therefore, there is no numerical factorization before the first log is printed.
  # If the maximum number of iterations is set to 0, the only computation that occurs
  # is computing the residuals at the initial point.
  @timeit qnc.timer "Main loop" while(true)

    # I.A - Compute residuals at current iterate
    @timeit qnc.timer "Residuals" compute_residuals!(qnc)

    update_mu!(qnc.pt)

    # I.B - Log
    # TODO: Put this in a logging function
    ttot = time() - tstart
    if params.OutputLevel > 0
      # Display log
      @printf "%4d" qnc.niter

      # Objectives
      ϵ = dat.objsense ? one(T) : -one(T)
      @printf "  %+14.7e" ϵ * qnc.primal_objective
      @printf "  %+14.7e" ϵ * qnc.dual_objective

      # Residuals
      @printf "  %8.2e" max(qnc.res.rp_nrm, qnc.res.rl_nrm, qnc.res.ru_nrm)
      @printf " %8.2e" qnc.res.rd_nrm
      @printf " %8s" "--"

      # Mu
      @printf "  %7.1e" qnc.pt.μ

      # Time
      @printf "  %.2f" ttot

      print("\n")
    end

    # TODO: check convergence status
    # TODO: first call an `compute_convergence status`,
    #   followed by a check on the solver status to determine whether to stop
    # In particular, user limits should be checked last (if an optimal solution is found,
    # we want to report optimal, not user limits)

    @timeit qnc.timer "update status" update_solver_status!(qnc,
                                                            params.TolerancePFeas,
                                                            params.ToleranceDFeas,
                                                            params.ToleranceRGap,
                                                            params.ToleranceIFeas
                                                           )

    if (
        qnc.solver_status == Trm_Optimal
        || qnc.solver_status == Trm_PrimalInfeasible
        || qnc.solver_status == Trm_DualInfeasible
       )
      break
    elseif qnc.niter >= params.IterationsLimit
      qnc.solver_status = Trm_IterationLimit
      break
    elseif ttot >= params.TimeLimit
      qnc.solver_status = Trm_TimeLimit
      break
    end

    # TODO: step
    # For now, include the factorization in the step function
    # Q: should we use more arguments here?
    try
      @timeit qnc.timer "Step" compute_step!(qnc, params)

      params.OutputLevel > 0 && println("Nº total de tentativas de broyden : ", qnc.n_tent_broyden)
      params.OutputLevel > 0 && println("Nº total de iterações de Broyden  : ", qnc.nitb)
      params.OutputLevel > 0 && println("Nº total de correções alternativas: ", qnc.n_corr_alt)
      params.OutputLevel > 0 && println("Nº total de correções da jacobiana: ", qnc.n_corr_jac)
    catch err

      if isa(err, PosDefException) || isa(err, SingularException)
        # Numerical trouble while computing the factorization
        qnc.solver_status = Trm_NumericalProblem

      elseif isa(err, OutOfMemoryError)
        # Out of memory
        qnc.solver_status = Trm_MemoryLimit

      elseif isa(err, InterruptException)
        qnc.solver_status = Trm_Unknown
      else
        # Unknown error: rethrow
        rethrow(err)
      end

      rethrow(err)
      break
    end
    qnc.niter += 1
  end

  params.OutputLevel > 0 && println("Nº de iterações (algoritmo principal): ", qnc.niter)

  # TODO: print message based on termination status
  params.OutputLevel > 0 && println("Solver exited with status $((qnc.solver_status))")

  return nothing
end

function compute_starting_point(qnc::QNC{T}) where{T}

  pt = qnc.pt
  dat = qnc.dat
  m, n, p = pt.m, pt.n, pt.p

  KKT.update!(qnc.kkt, zeros(T, n), ones(T, n), T(1e-6) .* ones(T, m))

  # Get initial iterate
  KKT.solve!(zeros(T, n), pt.y, qnc.kkt, false .* qnc.dat.b, qnc.dat.c)  # For y
  KKT.solve!(pt.x, zeros(T, m), qnc.kkt, qnc.dat.b, false .* qnc.dat.c)  # For x

  # I. Recover positive primal-dual coordinates
  δx = one(T) + max(
                    zero(T),
                    (-3 // 2) * minimum((pt.x .- dat.l) .* dat.lflag),
                    (-3 // 2) * minimum((dat.u .- pt.x) .* dat.uflag)
                   )
  @. pt.xl  = ((pt.x - dat.l) + δx) * dat.lflag
  @. pt.xu  = ((dat.u - pt.x) + δx) * dat.uflag

  z = dat.c - dat.A' * pt.y
  #=
  We set zl, zu such that `z = zl - zu`

  lⱼ |  uⱼ |    zˡⱼ |     zᵘⱼ |
  ----+-----+--------+---------+
  yes | yes | ¹/₂ zⱼ | ⁻¹/₂ zⱼ |
  yes |  no |     zⱼ |      0  |
  no | yes |     0  |     -zⱼ |
  no |  no |     0  |      0  |
  ----+-----+--------+---------+
  =#
  @. pt.zl = ( z / (dat.lflag + dat.uflag)) * dat.lflag
  @. pt.zu = (-z / (dat.lflag + dat.uflag)) * dat.uflag

  δz = one(T) + max(zero(T), (-3 // 2) * minimum(pt.zl), (-3 // 2) * minimum(pt.zu))
  pt.zl[dat.lflag] .+= δz
  pt.zu[dat.uflag] .+= δz

  qnc.pt.τ   = one(T)
  qnc.pt.κ   = zero(T)

  # II. Balance complementarity products
  μ = dot(pt.xl, pt.zl) + dot(pt.xu, pt.zu)
  dx = μ / ( 2 * (sum(pt.zl) + sum(pt.zu)))
  dz = μ / ( 2 * (sum(pt.xl) + sum(pt.xu)))

  pt.xl[dat.lflag] .+= dx
  pt.xu[dat.uflag] .+= dx
  pt.zl[dat.lflag] .+= dz
  pt.zu[dat.uflag] .+= dz

  # Update centrality parameter
  update_mu!(qnc.pt)

  return nothing
end

