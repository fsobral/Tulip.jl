using LinearAlgebra
import LinearAlgebra: ldiv!

mutable struct GoodBroyden{S}

    lu
    u :: Vector{Vector{S}}
    sb :: Vector{Vector{S}}
    rho :: Vector{S}
    size :: Int

    function GoodBroyden(lu, max_size=10)
        return new{Float64}(lu, Vector{Vector{Float64}}(undef, max_size), Vector{Vector{Float64}}(undef, max_size), Vector{Float64}(undef, max_size), 0)
    end
end

"""
    ldiv!(A::GoodBroyden, b, x)

Solves the linear system \$A x = b\$ where \$A\$ is given by the Good Broyden update.
"""
LinearAlgebra.ldiv!(A::GoodBroyden, b) = begin

    mpc = A.lu

    m, n, p = mpc.pt.m, mpc.pt.n, mpc.pt.p

    bd, bl, bu, bp, bzl, bzu = get_variables_tulip(b, m, n)

    tmpb = Point{Float64, Vector{Float64}}(m, n, p; hflag = false)

    # Resolve o caso base
    # ldiv!(A.lu, b)
    solve_newton_system!(tmpb, mpc,
        Vector(bp), Vector(bl), Vector(bu), Vector(bd), Vector(bzl), Vector(bzu)
    )

    bd  .= tmpb.x
    bl  .= tmpb.xl
    bu  .= tmpb.xu
    bp  .= tmpb.y
    bzl .= tmpb.zl
    bzu .= tmpb.zu
    for i = 1:A.size[]
        u = A.u[i]
        sb = A.sb[i]
        rho = A.rho[i]
        @. b = b + (dot(sb, b) / rho) * u
    end

    nothing

end

"""
Updates the Good Broyden approximation. The efficient way
"""
function update!(B::GoodBroyden, s, b)

    size = B.size + 1
    B.u[size] = copy(b)
    ldiv!(B, B.u[size])
    # Todo: melhorar s - B.u[size]
    B.rho[size] = dot(s, s - B.u[size]) 
    B.sb[size] = copy(s)
    B.size    = size

end

"""
Updates the Good Broyden approximation.
"""
function old_update!(B::GoodBroyden, s, y)

    size = B.size + 1
    # Compute H y
    @. B.u[size] = copy(y)
    ldiv!(B, B.u[size])
    # Compute s^T H y
    B.rho[size] = dot(s, B.u[size]) 
    # Store s
    B.sb[size] = copy(s)
    # Store u
    @. B.u[size] = s - B.u[size]
    B.size    = size

end

function Broyden_parada(fcurr, w, sig, sig_max, mu_wk, m, n, p, it, it_max, eps)

    _, wxl, wxu, _, wzl, wzu = get_variables_tulip(w, m, n)

    positivity = min(minimum(wxl), minimum(wzl), minimum(wxu), minimum(wzu))

    if norm(fcurr) < eps
        #      println("O Método de Broyden Convergiu.")
        if positivity > 0
            #          println("O Ponto encontrado pelo método de Broyden foi aceito.")
            return (true, true)
        else
            #          println("O Ponto encontrado pelo método de Broyden foi recusado, pois não satisfaz as condições de positividade.")
            return (true, false)
        end
    end

    mucurr = compute_mu(w, m, n, p)

    if 0 <= mucurr < sig*mu_wk && positivity > 0 #-sqrt(eps) # trocar < por <= se nao der certo
        #    println("Broyden interrompido: houve decréscimo suficiente de mu, e o ponto encontrado está em F_0.")
        return (true, true)
    else
        if it < it_max
            return (false, nothing)
        else
            #    println("Broyden interrompido: o limite de iterações foi atingido.")
            return (true, false)
        end
    end
end

function Broyden(F_tau, B, w, mu_wk, sig, sig_max, m, n, p, eps, it_max)

    status = false
    # primeira iteração
    it = 1
    sb = F_tau(w, sig*mu_wk)

    #display(sb)
    ldiv!(B, sb)
    #display(sb)

    #  println("||F(w_0)|| = ", norm(F_tau(w, sig*mu_wk)))
    @. w = w + sb
    #  println("||F(w_$(it))|| = ", norm(F_tau(w, sig*mu_wk)))
    # u = -F_tau(w, sig*mu_wk)
    # ldiv!(B, u)
    fcurr = F_tau(w, sig*mu_wk)

    update!(B, sb, fcurr)

    while true
        (parar, status) = Broyden_parada(fcurr, w, sig, sig_max, mu_wk, m, n, p, it, it_max, eps)

        if parar == true
            break
        end

        it += 1

        @. sb = (dot(B.sb[it-1], B.u[it-1]) / B.rho[it-1]) * B.u[it-1]
        @. sb = sb + B.u[it-1]

        @. w = w + sb
        #    println("||F(w_$(it))|| = ", norm(F_tau(w, sig*mu_wk)))
        fcurr .= F_tau(w, sig*mu_wk)
        update!(B, sb, fcurr)
    end
    global nitb += it
    println("Nº de iterações de Broyden: ", it)
    return status
end

function get_variables_tulip(w, m, n)

    wx =  @view w[1:n]
    wxl = @view w[n + 1:2 * n]
    wxu = @view w[2 * n + 1:3 * n]
    wy  = @view w[3 * n + 1:3 * n + m]
    wzl = @view w[3 * n + m + 1:4 * n + m]
    wzu = @view w[4 * n + m + 1:5 * n + m]

    return wx, wxl, wxu, wy, wzl, wzu

end

function f_tau_creator(mpc)

    return let

        m, n, _ = mpc.pt.m, mpc.pt.n, mpc.pt.p

        dat = mpc.dat

        (w, τ) -> begin

        wx, wxl, wxu, wy, wzl, wzu = get_variables_tulip(w, m, n)

        v = zeros(5 * n + m)

        rd, rl, ru, rp, rμl, rμu = get_variables_tulip(v, m, n)

        # Primal residual
        # rp = b - A*x
        rp .= dat.b
        mul!(rp, dat.A, wx, -one(Float64), one(Float64))

        # Lower-bound residual
        # rl_j = l_j - (x_j - xl_j)  if l_j ∈ R
        #      = 0                   if l_j = -∞
        @. rl = ((dat.l + wxl) - wx) * dat.lflag

        # Upper-bound residual
        # ru_j = u_j - (x_j + xu_j)  if u_j ∈ R
        #      = 0                   if u_j = +∞
        @. ru = (dat.u - (wx + wxu)) * dat.uflag

        # Dual residual
        # rd = c - (A'y + zl - zu)
        rd .= dat.c
        mul!(rd, transpose(dat.A), wy, -one(Float64), one(Float64))
        @. rd += wzu * dat.uflag - wzl * dat.lflag

        # Complementarity residual
        @. rμl = wxl * wzl - τ
        @. rμu = wxu * wzu - τ

        return v

        end

    end

end

function compute_mu(w, m, n, p)
    _, wxl, wxu, _, wzl, wzu = get_variables_tulip(w, m, n)
    return (dot(wxl, wzl) + dot(wxu, wzu)) / p
end

function Quasi_Newton_Corrector!(mpc :: MPC, sig_max = 1-1.0e-4, eps=1.0e-8, it_max = 5)

    #1-1.0e-4

    # Names
    # dat = mpc.dat
    pt = mpc.pt
    # res = mpc.res

    m, n, p = pt.m, pt.n, pt.p

    # A = dat.A
    # b = dat.b
    # c = dat.c

    #w = vcat(pt.x, pt.y, z)

    # Cria o vetor w_k
    w = vcat(pt.x, pt.xl, pt.xu, pt.y, pt.zl, pt.zu)

    # Constroi a F
    F_tau = f_tau_creator(mpc)

    # Neste momento mpc.Δ ja tem a direcao previsora


    # # Passo 1
    # d = - (fatoracao \ F_tau(w, 0)) #vcat(mpc.Δ.x, mpc.Δ.y, dz) # Espaço para otimização
    Δ = mpc.Δ
    d = vcat(Δ.x, Δ.xl, Δ.xu, Δ.y, Δ.zl, Δ.zu)
    # mpc.Δ.x = d[1:n]
    # mpc.Δ.y = d[n+1:n+m]

    # Passo 2
    
    alpha0 = alpha = min(mpc.αp, mpc.αd)

    # Passo 3

    sig = 0.5*sig_max

    # Passo 4

    mu_wk = compute_mu(w, m, n, p)

    yki = zeros(5 * n + m)
    t = 1

    while true
        global n_tent_broyden += 1

        println("Testagem: ", t)
        println("Alfa = ", alpha)
        println("Sigma = ", sig)

        @. yki = w + alpha * d

        println("Ponto inicial do Broyden")
        display(yki)

        println("mu (após newton) = ", compute_mu(yki, m, n, p))

        B = GoodBroyden(mpc, it_max)

        b_status = Broyden(F_tau, B, yki, mu_wk, sig, sig_max, m, n, p, eps, it_max)

        println("mu (final do broyden)=", compute_mu(yki, m, n, p))
        println("Status (Broyden) = ", b_status)
        if b_status == true # Se encontrar um ponto em F_0 com decrescimo de mu, pare.
            break
        end
        alpha *= 0.5
        sig = 0.5*(sig_max + sig)
        #    if abs(sig_max - sig) < 1.0e-8 # Parar se sig se aproximar muito de sig_max (e depois acusar erro: sig_max não é grande suficiente ou o ponto inicial tomado não está próximo o suficiente do caminho central)
        if t == 3 # 30 iterações é suficiente para praticamente zerar a diferença entre sig_max e sig (ela fica na ordem de 4.66e-10)
            #            error("Não foi possível determinar alpha e sigma de modo a obter a convergência do passo corretor. Tente outro ponto inicial que esteja mais próximo do caminho central ou tome sig_max ainda mais próximo de 1.")
#            println("AVISO: Não foi possível determinar alpha e sigma de modo a obter a convergência do passo corretor. Isso pode ter ocorrido pois o ponto inicial não estava próximo o suficiente do caminho central. Para contornar isso, será aplicado um método quasi-newton alternativo.")
            println("\t\tPASSO ALTERNATIVO")
            
            global n_corr_alt += 1
            an = alpha0
            @. yki = w + an * d
            while true
              if 0 <= compute_mu(yki, m, n, p) <= mu_wk
                break
              else
                an *= 0.5
                @. yki = w + an * d
              end
            end
            μnovo = compute_mu(yki, m, n, p)
            sig_bom = (μnovo / mu_wk)^3
            println("\t\tsig_bom = ", sig_bom)
            tau = sig_bom * μnovo

            w_n2 = zeros(length(yki))
            Btmp = GoodBroyden(mpc, 5)
            dc = zeros(length(yki))
            yk = zeros(length(yki))
            sk = zeros(length(yki))
            dc .= F_tau(yki, tau)

            # Broyden diferente, pois reduz antes de dar o proximo passo
            for k=1:5
                println("\t\tBroyden Alt ($k)")
                
                @. yk = dc
                ldiv!(Btmp, dc)

                global nitb += 1

                _, wn2xl, wn2xu, _, wn2zl, wn2zu = get_variables_tulip(yki, m, n)
                _, dcxl, dcxu, _, dczl, dczu     = get_variables_tulip(dc, m, n)
                axl = max_step_length(wn2xl, dcxl)
                axu = max_step_length(wn2xu, dcxu)
                azl = max_step_length(wn2zl, dczl)
                azu = max_step_length(wn2zu, dczu)

                ac = 0.9 * min(1.0, axl, axu, azl, azu)
                @. w_n2 = yki + ac * dc

                println("\t\t\tμnovo = $μnovo")
                println("\t\t\tac ini = $ac")

                while true

                    # if 0 <= dot(w_n2[1:n], w_n2[n+m+1:2*n+m])/n <= dot(yki[1:n], yki[n+m+1:2*n+m])/n && minimum([minimum(w_n2[1:n]), minimum(w_n2[n+m+1:2*n+m])]) > 0
                    if 0 <= compute_mu(w_n2, m, n, p) <= μnovo
                        break
                    else
                        ac *= 0.5
                        @. w_n2 = yki + ac * dc
                    end
                    
                    # Verifica se há is_nan
                    # Isso acontece quando os passo quasi-Newton sao muito pequenos
                    is_nan = any(isnan, w_n2)

                    if is_nan == true
                        println("\t\t\tParou por NaN.")
                        w_n2 .= yki 
                        break
                    end

                    if ac < 1.0e-8
                        w_n2 .= yki # se ac for muito pequeno, não faça nada.
                        break
                    end
                end

                println("\t\t\tac end = $ac")

                @. sk  = w_n2 - yki
                @. yki = w_n2
                
                dc .= F_tau(yki, tau)
                # Tem que ser ao contrario, pois o "-" ja esta dentro de F_tau
                @. yk  = yk - dc

                # J_temp = J_temp + ((yk - J_temp*sk)*(sk'))/(dot(sk,sk))
                update!(Btmp, sk, yk)
                μnovo = compute_mu(yki, m, n, p)
            end

            # Termina o loop infinito
            break

        end

        t += 1
    end

    # Passo 5 (adaptado: atualizar direcao)

    # Compute the new direction
    @. d = yki - w

    dx, dxl, dxu, dy, dzl, dzu = get_variables_tulip(d, m, n)
    
    @. mpc.Δc.x  = dx
    @. mpc.Δc.xl = dxl
    @. mpc.Δc.xu = dxu
    @. mpc.Δc.y  = dy
    @. mpc.Δc.zl = dzl
    @. mpc.Δc.zu = dzu

    if t == 30
        return true
    else
        return false
    end
end
