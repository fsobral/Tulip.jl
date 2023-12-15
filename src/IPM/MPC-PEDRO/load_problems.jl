using LinearAlgebra
using SparseArrays
using QPSReader: readqps, fetch_netlib
using Printf
using Logging

CONV_PARAMETERS = Dict(
    "25FV47"   => (),
    "ADLITTLE" => (),
    "AFIRO"    => (), # Esse é viável (resolveu)
    "AGG"      => (),
    "AGG2"     => (),
    "AGG3"     => (),
    "BANDM"    => (), # Esse é viável (resolveu)
    "BEACONFD" => (),
    "BLEND"    => (), # Aparentemente deu algum erro na leitura desse problema
    "BNL1"     => (), #
    "BNL2"     => (),
    "BRANDY"   => (), # Singular Exception na fatoração LU (no proprio load_problems.jl)
    "D2Q06C"   => (), #
    "DEGEN2"   => (),
    "DEGEN3"   => (),
    "E226"     => (),
    "FFFFF800" => (),
    "ISRAEL"   => (),
    "LOTFI"    => (), #
    "MAROS-R7" => (), # É viável (resolveu)
    "QAP8"     => (), # É viável, mas não resolveu (Result index of attribute MathOptInterface.ObjectiveValue(1) out of bounds. There are currently 0 solution(s) in the model.)
    "QAP12"    => (), # É viável (resolveu)
    "QAP15"    => (), # É viável, mas faltou memória (o processo foi morto automaticamente)
    "SC105"    => (),
    "SC205"    => (),
    "SC50A"    => (),
    "SC50B"    => (),
    "SCAGR25"  => (), # Esse é viável (resolveu)
    "SCAGR7"   => (), # Esse é viável (resolveu)
    "SCFXM1"   => (),
    "SCFXM2"   => (),
    "SCFXM3"   => (), 
    "SCORPION" => (),
    "SCRS8"    => (), # Esse é viável (resolveu)
    "SCSD1"    => (), # Esse é viável (resolveu)
    "SCSD6"    => (), # Esse é viável (resolveu)
    "SCSD8"    => (), # Esse é viável (resolveu)
    "SCTAP1"   => (), # Esse é viável (resolveu)
    "SCTAP2"   => (), # Esse é viável (resolveu)
    "SCTAP3"   => (), # Esse é viável (resolveu)
    "SHARE1B"  => (), # Esse é viável (resolveu)
    "SHARE2B"  => (), # Esse é viável (resolveu)
    "SHIP04L"  => (),
    "SHIP04S"  => (),
    "SHIP08L"  => (),
    "SHIP08S"  => (),
    "SHIP12L"  => (),
    "SHIP12S"  => (),
    "STOCFOR1" => (), # Esse é viável (resolveu)
    "STOCFOR2" => (), # Esse é viável (resolveu)
    "STOCFOR3" => (), # Esse é viável (resolveu)
    "TRUSS"    => (),
    "WOOD1P"   => (),#
    "WOODW"    => () #
)

mutable struct IPMProjProb

    m :: Integer
    n :: Integer
    x :: AbstractVector{<:Number}
    y :: AbstractVector{<:Number}
    s :: AbstractVector{<:Number}
    A :: AbstractMatrix{<:Number}
    b :: AbstractVector{<:Number}
    c :: AbstractVector{<:Number}

end

IPMProjProb(p::IPMProjProb) = IPMProjProb(p.m, p.n, copy(p.x), copy(p.y), copy(p.s),
                                          copy(p.A), copy(p.b), copy(p.c))

function load_problem(pname, μ=1000.0)

    m, n, A, b, c = load_mps(pname)

    x, y, s = make_feasible(A, b, c, μ; CONV_PARAMETERS[pname]...)

    return IPMProjProb(m, n, x, y, s, A, b, c)

end

function load_mps(pname)

    nlpath = fetch_netlib()

    qps = with_logger(Logging.NullLogger()) do
        readqps("$(nlpath)/$(pname).SIF")
    end
    
    m = qps.ncon
    n = qps.nvar

    nn = nm = 0
    
    for i = 1:m
        if qps.lcon[i] != qps.ucon[i]
            (qps.lcon[i] > -Inf) && (nn += 1)
            (qps.ucon[i] <  Inf) && (nn += 1)
            (qps.lcon[i] > -Inf) && (qps.ucon[i] <  Inf) && (nm += 1)
        end
    end

    for i = 1:n
        isinf(qps.lvar[i]) && error("Do not deal with lower-unbounded vars.")
        # if qps.lvar[i] != 0.0
        #     nn += 1
        #     nm += 1
        # end
        if qps.uvar[i] <  Inf
            nn += 1
            nm += 1
        end
    end

    # Allocate new b, nnzr, nnzc, nnzv

    b = zeros(Float64, m + nm)

    M = sparse(qps.arows, qps.acols, qps.avals, m + nm, n + nn)
    
    # Loop again to create additional data

    ii = m + 1
    jj = n + 1
    for i = 1:m
        b[i] = qps.lcon[i]
        if qps.lcon[i] != qps.ucon[i]
            if qps.lcon[i] > -Inf
                M[i, jj] = -1.0
                b[i]     = qps.lcon[i]
                jj      += 1
            end
            if (qps.lcon[i] > -Inf) && (qps.ucon[i] < Inf)
                for j in findall(!iszero, M[i, :])
                    M[ii, j] = M[i, j]
                end
                M[ii, jj] = 1.0
                b[ii]     = qps.ucon[i]
                ii       += 1
                jj       += 1
            elseif qps.ucon[i] < Inf
                M[i, jj] = 1.0
                b[i]     = qps.ucon[i]
                jj      += 1
            end
        end
    end

    for j = 1:n
        # If the LHS is not zero (and not -Inf), then shift to zero,
        # which changes the RHS of the system and the RHS bound of
        # x[j]
        if qps.lvar[j] != 0.0
            l = qps.lvar[j]
            for i in findall(!iszero, M[:, j])
                b[i] -= M[i, j] * l
            end
            qps.uvar[j] -= l
        end
        # If the RHS is not Inf, then add a new equality constraint
        if qps.uvar[j] < Inf
            M[ii, j] = M[ii, jj] = 1.0
            ii += 1
            jj += 1
        end
    end

    # Final check
    
    ((ii - 1 - m != nm) || (jj - 1 - n != nn)) &&
        error("Wrong values ($(ii - 1 - m) != $nm) or ($(jj - 1 - n) != $nn).")
    
    # Return data
    
    return m + nm, n + nn, M, b, [qps.c; zeros(nn)]

end

function create_symm_mat(A, x, s; reg=sqrt(eps(Float64)))

    m, n = size(A)

    ((length(x) != n) || (length(s) != n)) && throw(DimensionMismatch())
    
    # TODO: see the system for upper bounds too
    return Symmetric([spdiagm(x .* s .- reg) spzeros(n, m);
                      (A .* (x'))            (reg * I)     ], :L)
    
end

"""

This function uses the suggestion of Megiddo, N. (1989). Pathways to
the optimal set in linear programming. In Progress in mathematical
programming (pp. 131-158), which is better explained in Lustig,
I. J. (1990). Feasibility issues in a primal-dual interior-point
method for linear programming. Mathematical Programming, 49(1),
145-162.

This ensures that a feasible starting point of this new `m+1 x n+2`
problem can be in the form

    [x0; 1; b_{m+1} - (A' * y0 - s0 + c[1:n])^T * x0],
    [y0; -1.0],
    [z0; c_{n+1} - (b[1:m] - A * x0)^T * y0; 1.0] 

"""
function make_feasible(A, b, c, x0, y0, s0; Mp = 1.0e5, Md = 1.0e5)

    m, n = size(A)

    newrow = A' * y0
    @. newrow +=  s0 .- c

    newcol = b .- A * x0

    newA = spzeros(m + 1, n + 2)
    newA[1:m, 1:n] = A
    newA[1:m, n + 1] = newcol
    newA[m + 1, 1:n] = newrow
    newA[m + 1, n + 2] = 1.0

    # The max(...) here is to ensure that the new variables are able
    # to start positive when the trivial initial feasible point is
    # used
    bigmb = max(Md, dot(newrow, x0) + 1.0)
    bigmc = max(Mp, dot(newcol, y0) + 1.0)

    return m + 1, n + 2, newA, [b; bigmb], [c; bigmc; 0.0],
           [x0; 1; bigmb - dot(newrow, x0)],
           [y0; -1],
           [s0; bigmc - dot(newcol, y0); 1]

end

# TODO: This could be better done by making the explicit calculations
# using vector `e`.
make_feasible(A, b, c; kwargs...) = return make_feasible(
    A, b, c, 10*ones(Float64, length(c)),
    10*ones(Float64, length(b)), 10*ones(Float64, length(c)); kwargs...)

"""

    make_feasible!(A, b, c, x, y, s, μ; MAXIT=10, MAXIIT=10, output=false,
                   ϵ=sqrt(eps(Float64)), reg=sqrt(eps(Float64)))
    make_feasible(A, b, c, μ; kwargs...)

This is a simple (and bad) implementation of a Newton method to find a
point near the central path associated with `μ`. It requires a
starting point which is positive for x and s. If called without `(x,
y, s)` arguments, uses the vector of ones for them.

The optional parameter `reg` controls the regularity terms, to avoid
singular augmented systems, in the same sense of Altman and Gondzio,
OMS, 1999. The parameter `ϵ` controls the stopping criterium to be
considered feasible.

"""
function make_feasible!(A, b, c, x, y, s, μ; MAXIT=100, MAXIIT=20,
                        output=false, ϵ=sqrt(eps(Float64)), reg=sqrt(eps(Float64)), ϵls=1.5)

    m, n = size(A)

    rhs = zeros(n + m)
    ds = zeros(n)

    resd = @view rhs[1:n]
    resp = @view rhs[n + 1:m+n]
    #resμ = @view rhs[m+n+1:end]

    regp = regd = reg

    # Handling function to evaluate the RHS
    F(x, y, s, rp, rd) = begin

        @. rd = c - s
        mul!(rd, A', y, -1.0, 1.0)

        @. rp = b
        mul!(rp, A, x, -1.0, 1.0)

    end

    F(x, y, s, resp, resd)
    nd = norm(resd, Inf)
    np = norm(resp, Inf)
    n2d = norm(resd)
    n2p = norm(resp)

    # Change the RHS to use augmented systems
    @. ds = (μ - x * s)
    resd .-=  ds ./ x

    it = 1

    J = [ spdiagm(- s ./ x .- regp) A';
          spzeros(m, n)             (regd * I)]

    newx = zeros(n)
    newy = zeros(m)
    news = zeros(n)

    while (it <= MAXIT) && ((np > ϵ) || (nd > ϵ))

        d = Symmetric(J, :U) \ rhs
        dx = @view(d[1:n])
        dy = @view(d[n+1:m+n])
        @. ds = (ds - s * dx) / x

        alphap, alphad = compute_stepsize(x, s, dx, ds)

        # Update x, y and s
        @. newx = x + alphap * dx
        @. newy = y + alphad * dy
        @. news = s + alphad * ds

        F(newx, newy, news, resp, resd)
        newn2p = norm(resp)
        newn2d = norm(resd)

        t = 1.0
        iit = 1
        while (iit <= MAXIIT) && ((newn2p > ϵls * n2p) || (newn2d > ϵls * n2d))
            t *= 0.5
            @. newx = x + t * alphap * dx
            @. newy = y + t * alphad * dy
            @. news = s + t * alphad * ds
            #
            F(newx, newy, news, resp, resd)
            newn2p = norm(resp)
            newn2d = norm(resd)

            iit += 1
        end

        if iit > MAXIIT

            F(x, y, s, resp, resd)

            regp = min(1.0e-2, regp * 2)
            regd = min(1.0e-2, regd * 2)

        else

            regp = max(eps(Float64), regp / 100)
            regd = max(eps(Float64), regd / 100)

            @. x = newx
            @. y = newy
            @. s = news
            n2p = newn2p
            n2d = newn2d

        end

        # Update J
        for i = 1:n
            J[i, i] = - s[i] / x[i] - regp
        end
        for i = n + 1:n + m
            J[i, i] = regd
        end

        nd = norm(resd, Inf)
        np = norm(resp, Inf)

        ds .= (μ .- x .* s)
        resd .-=  ds ./ x

        # Symmetric neighbourhood check
        snlb, snub = estimate_gamma(x, s)

        it += 1

        (output) && @printf("%3d %8.1e %8.1e %8.1e %8.1e %8.1e %8.1e %8.1e\n", it, np, nd, alphap, alphad, t, snlb, snub)

    end

    ((np > ϵ) || (nd > ϵ) || any(x .<= 0) || any(s .<= 0)) && (@printf("Warning: not feasible.\n"))

    return x, y, s

end

make_feasible(p::IPMProjProb, μ; kwargs...) = make_feasible(p.A, p.b, p.c, μ; kwargs...)

make_feasible(A, b, c, μ; kwargs...) = make_feasible!(A, b, c,
    ones(length(c)), ones(length(b)), ones(length(c)), μ; kwargs...)

"Compute ``max{ \\|A (x + Δx) - b\\|_\\infty, \\|A^T (y + Δy) + s + Δs - c\\|_\\infty }``"
pd_feasibility(p::IPMProjProb, Δx, Δy, Δs) = pd_feasibility(p.A, p.b, p.c,
                                                            p.x + Δx, p.y + Δy, p.s + Δs)

"Compute ``max{ \\|A x - b\\|_\\infty, \\|A^T y + s - c\\|_\\infty }``"
pd_feasibility(p::IPMProjProb) = pd_feasibility(p.A, p.b, p.c, p.x, p.y, p.s)

pd_feasibility(A, b, c, x, y, s) = max(norm(A * x - b, Inf), norm(A' * y + s - c, Inf))

"Compute x^T s / n"
gap(p::IPMProjProb) = dot(p.x, p.s) / p.n

estimate_gamma(p::IPMProjProb) = estimate_gamma(p.x, p.s)

function estimate_gamma(x, s)

    n = length(x)
    
    μ = dot(x, s) / n

    return minimum((x .* s) ./ μ), maximum((x .* s) ./ μ)

end

function compute_stepsize(x, s, dx, ds; ssfrac=0.95)

        alphap = alphad = 1.0

        nI = findall(t -> t < 0, dx)
        !isempty(nI) && (alphap = minimum(- x[i] / dx[i] for i in nI))

        nI = findall(t -> t < 0, ds)
        !isempty(nI) && (alphad = minimum(- s[i] / ds[i] for i in nI))

        alphap = ssfrac * min(1.0, alphap)
        alphad = ssfrac * min(1.0, alphad)

    return alphap, alphad

end