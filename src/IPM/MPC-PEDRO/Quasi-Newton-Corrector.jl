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

    # Resolve o caso base
    ldiv!(A.lu, b)
    for i = 1:A.size[]
        u = A.u[i]
        sb = A.sb[i]
        rho = A.rho[i]
        b .= b + (dot(sb, b) / rho) * u

    end

    nothing

end

"""
Updates the Good Broyden approximation.
"""
function update!(B::GoodBroyden, s, b)

    size = B.size + 1
    B.u[size] = b
    ldiv!(B, B.u[size])
    B.rho[size] = dot(s, s - B.u[size]) 
    B.sb[size] = s
    B.size    = size

end

function Broyden_parada(F_tau, w, sig, sig_max, mu_wk, m, n, it, it_max, eps)
    if norm(F_tau(w, sig*mu_wk)) < eps
        #      println("O Método de Broyden Convergiu.")
        if minimum([minimum(w[1:n]), minimum(w[n+m+1: 2*n+m])]) > 0
            #          println("O Ponto encontrado pelo método de Broyden foi aceito.")
            return (true, true)
        else
            #          println("O Ponto encontrado pelo método de Broyden foi recusado, pois não satisfaz as condições de positividade.")
            return (true, false)
        end
    end
    if (dot(w[1:n], w[n+m+1: 2*n+m]) / n) < sig*mu_wk && minimum([minimum(w[1:n]), minimum(w[n+m+1: 2*n+m])]) > 0 #-sqrt(eps) # trocar < por <= se nao der certo
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

function Broyden(F_tau, B, w, mu_wk, sig, sig_max, m, n, eps, it_max)

    status = false
    # primeira iteração
    it = 1
    sb = -F_tau(w, sig*mu_wk)
    ldiv!(B, sb)
    #  println("||F(w_0)|| = ", norm(F_tau(w, sig*mu_wk)))
    w .= w + sb
    #  println("||F(w_$(it))|| = ", norm(F_tau(w, sig*mu_wk)))
    u = -F_tau(w, sig*mu_wk)
    ldiv!(B, u)
    update!(B, sb, -F_tau(w, sig*mu_wk))


    while true
        (parar, status) = Broyden_parada(F_tau, w, sig, sig_max, mu_wk, m, n, it, it_max, eps)

        if parar == true
            break
        end

        it += 1

        sb = (dot(B.sb[it-1], B.u[it-1]) / B.rho[it-1]) * B.u[it-1]
        sb .= sb + B.u[it-1]

        w .= w + sb
        #    println("||F(w_$(it))|| = ", norm(F_tau(w, sig*mu_wk)))
        update!(B, sb, -F_tau(w, sig*mu_wk))
    end
    println("Nº de iterações de Broyden: ", it)
    return status
end

function Quasi_Newton_Corrector!(mpc :: MPC, z, dz, sig_max = 1-1.0e-4, eps=1.0e-8, it_max = 100)

    # Names
    dat = mpc.dat
    pt = mpc.pt
    res = mpc.res

    m, n, p = pt.m, pt.n, pt.p

    A = dat.A
    b = dat.b
    c = dat.c

    w = vcat(pt.x, pt.y, z)

    F_tau(w, tau) = begin
        x = w[1:n]
        lamb = w[n+1:n+m]
        s = w[n+m+1:2*n+m]
        v = zeros(2*n+m)

        for i=1:n
            v[i] = dot(A[:, i], lamb) + s[i] - c[i]
            if tau == 0
                v[n+m+i] = x[i]*s[i]
            else
                v[n+m+i] = x[i]*s[i] - tau
            end
        end
        for i=1:m
            v[n+i] = dot(A[i, :], x) - b[i]
        end

        return v    

    end
    J(w) = begin
        x = w[1:n]
        s = w[n+m+1:2*n+m]
        M = zeros(2*n+m, 2*n+m)
        M[1:n, n+1:n+m] = A'
        M[n+1:n+m, 1:n] = A
        for i=1:n
            M[i, n+m+i] = 1.0
            M[n+m+i, i] = s[i]
            M[n+m+i, n+m+i] = x[i]
        end
        return M
    end

    x = w[1:n]
    lamb = w[n+1:n+m]
    s = w[n+m+1:2*n+m]
    J_temp = J(w)

    # correção para garantir a decomposição lu (resolve a questão da invertibilidade da matriz, mas a convergencia ainda é muito improvavel)
    #  mult = 1
    #  while abs(det(J_temp)) < eps
    #    J_temp += (2^mult)*eps*I
    #    mult += 1
    #  end
    # ----------------------------------------

    Jw = GoodBroyden(lu(J_temp), it_max)
    # Passo 1
    d = vcat(mpc.Δ.x, mpc.Δ.y, dz) # Espaço para otimização
    # Passo 2
    alpha = min(1.0, 0.9*mpc.αp)
    # Passo 3
    sig = 0.5*sig_max
    # Passo 4
    mu_wk = dot(x, s)/n
    w_n = zeros(2*n+m)
    t = 1

    while true
        println("Testagem: ", t)
        println("Alfa = ", alpha)
        println("Sigma = ", sig)

        w_n .= w + alpha*d

        b_status = Broyden(F_tau, Jw, w_n, mu_wk, sig, sig_max, m, n, eps, it_max)
        Jw.size = 0 # Resetar a estrutura GoodBroyden para a próxima iteração

        println("mu (final do broyden)=", dot(w_n[1:n], w_n[n+m+1:2n+m])/n)
        println("Status (Broyden) = ", b_status)

        if b_status == true # Se encontrar um ponto em F_0 com decrescimo de mu, pare.
            break
        end
        alpha *= 0.5
        sig = 0.5*(sig_max + sig)
        #    if abs(sig_max - sig) < 1.0e-8 # Parar se sig se aproximar muito de sig_max (e depois acusar erro: sig_max não é grande suficiente ou o ponto inicial tomado não está próximo o suficiente do caminho central)
        if t == 30 # 30 iterações é suficiente para praticamente zerar a diferença entre sig_max e sig (ela fica na ordem de 4.66e-10)
            error("Não foi possível determinar alpha e sigma de modo a obter a convergência do passo corretor. Tente outro ponto inicial que esteja mais próximo do caminho central ou tome sig_max ainda mais próximo de 1.")
        end
        t += 1
    end
    # Passo 5 (adaptado: atualizar direcao)
#    println("w após broyden: ", w_n)
    w_n .= w_n .- w
    mpc.Δc.x .= w_n[1:n]
    mpc.Δc.y .= w_n[n+1:n+m] # talvez precise atualizar mais coisas em delta_c

end
