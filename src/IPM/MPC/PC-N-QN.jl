using LinearAlgebra
import LinearAlgebra: ldiv!

function Broyden_old(B, w, c, A, b, mu_wk, sig, m, n, eps, sig_max, it_max)
    N = size(B)[2]
    B0 = copy(B)
    w0 = copy(w)
    status = false
    F(w) = begin
        x = w[1:n]
        lamb = w[n+1:n+m]
        s = w[n+m+1:2*n+m]
        mu = (x'*s)/n
        v = zeros(N)
        v[1:n] = A'*lamb+s - c
        v[n+1: m+n] = A*x - b
        v[n+m+1:2*n+m] = diagm(x)*diagm(s)*ones(n) - sig*mu*ones(n)
        return v
    end
    k = 0
    while norm(F(w0)) > eps && k < it_max
        d = - B0 \ F(w0)
        w1 = w0 + d
        y = F(w1) - F(w0)
        B0 = B0 + ((y - B0*d)*d')/(norm(d, 1))
        w0 = w1
        k += 1
        x1 = w1[1:n]
        s1 = w1[n+m+1:2*n+m]
        mu1 = (x1'*s1)/n
        if mu1 < 0.5*(sig_max + sig)*mu_wk && minimum([minimum(x1), minimum(s1)]) > 0 && k < it_max
            status = true
            break
        end
    end
    println("Nº de iterações de Broyden: ", k)
    return (w0, status)
end

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
LinearAlgebra.ldiv!(A::GoodBroyden, b) = begin # WARNING: Essa função começa a consumir mais memória com o passar das iterações. Esse aumento é bem lento, e mesmo em um problema grande (QAP12) o uso de memória foi bem razoável para 100 iterações, então creio que não vai ser um problema.

    # Resolve o caso base
    println("⏰ Tempo para resolver um sistema envolvendo B_0:")
    @time ldiv!(A.lu, b)
    println("")
    println("⏰ Tempo dedicado a resolver o sistema original (Broyden)")
    u = nothing
    sb = nothing
    rho = nothing
    @time for i = 1:A.size[]
      u = A.u[i] # Sem intenção de fazer cópias
      sb = A.sb[i] # Sem intenção de fazer cópias
        rho = A.rho[i]
        rho = dot(sb, b) / rho
#        b .= b + (dot(sb, b) / rho) * u
        b .= b + rho * u

    end
    println("")

    nothing

end

"""
Updates the Good Broyden approximation.
"""
function update!(B::GoodBroyden, s, b)

    for i=1:length(s)
        if isnan(s[i])
            #         error("Existem NaN's no vetor s (broyden update)")
        end
        if isnan(b[i])
            #         error("Existem NaN's no vetor b (broyden update)")
        end
    end

    size = B.size + 1
    B.u[size] = b
    ldiv!(B, B.u[size])
    #    println("s (pro rho) = $(s)")
    #    println("u (pro rho) = $(B.u[size])")
    B.rho[size] = 0
    for i=1:length(s)
        B.rho[size] += s[i]*(s[i] - B.u[size][i])
    end

    #dot(s, B.u[size]) - dot(s, s)
    #    println("rho = $(B.rho[size])")
    B.sb[size] = s
    B.size    = size

end

function Broyden_parada(F_tau, w, sig, sig_max, mu_wk, m, n, it, it_max, eps)
    #  if norm(F_tau(w, sig*mu_wk)) < eps
    #      println("O Método de Broyden Convergiu.")
    #    return (true, true)
    #  end
    if (dot(w[1:n], w[n+m+1: 2*n+m]) / n) < sig*mu_wk && minimum([minimum(w[1:n]), minimum(w[n+m+1: 2*n+m])]) > 0#-sqrt(eps) # trocar < por <= se nao der certo
        println("Broyden interrompido: houve decréscimo suficiente de mu.")
        return (true, true)
    end
    if it < it_max
        return (false, nothing)
    else
        println("Broyden interrompido: o limite de iterações foi atingido.")
        return (true, false)
    end
end

function Broyden(F_tau, B, w, mu_wk, sig, sig_max, m, n, eps = 1.0e-8, it_max = 100)

    status = false
    #B = Jw #GoodBroyden(Jw, it_max)
    # primeira iteração
    it = 1
    sb = -F_tau(w, sig*mu_wk)
    ldiv!(B, sb)
    w .= w + sb
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

        update!(B, sb, -F_tau(w, sig*mu_wk))
    end
    println("Nº de iterações de Broyden: ", it)
    return status
end

function iteracao(F_tau, J, w, it_max, eps, sig_max, m, n)
    status = true
    x = view(w[1:n], :) # Não cria uma cópia do vetor na memória (ou pelo menos não deveria)
    lamb = view(w[n+1:n+m], :) # Sem cópia
    s = view(w[n+m+1:2*n+m], :) # Sem cópia
    Jw = J(w)
    println("⏰ Tempo para fatorar Jw (inclui o tempo que passa tentando corrigir a jacobiana até ser possível fatorar):")
    @time begin
    is_invertible=false
    while is_invertible==false
    try
    Jw = lu(Jw)
    is_invertible=true
  catch
        for i=1:2*n+m # Soma 1.0e-8*I em Jw para tentar fazer Jw se tornar inversivel
      Jw[i, i] += 1.0e-8
    end
  end
  end
end
    println("")
    Jw = GoodBroyden(Jw, it_max)

    d = -F_tau(w, 0) # F em sigma_k mu_k com sigma_k = 0.

    # Passo 1
    println("⏰ Tempo para encontrar d (direção de Newton):")
    @time ldiv!(Jw, d)
    println("")
    dx = view(d[1:n], :) # Sem cópia
    ds = view(d[n+m+1:2*n+m], :) # Sem cópia
    # Passo 2
    println("⏰ Tempo para encontrar alpha para o passo de Newton:")
    @time begin
    pctg = 0.99
    alpha = 1.0
    for i=1:n
        if dx[i] < 0
          alpha = min(-pctg*x[i]/dx[i], 1.0)
        end
        if ds[i] < 0
          alpha = min(-pctg*s[i]/ds[i], 1.0)
        end
    end
    alpha = max(alpha, 0) # calcula alpha máximo tal que algum xi ou si zera e depois toma 90% desse passo, ou passo 1 no caso em que nenhuma variável bloqueia o passo.
  end
  println("")
    # Passo 3
    println("⏰ Tempo gasto para encontrar sigma para o passo corretor:")
    # sig = 0.5*sig_max
    sig = min(sig_max, 1 - alpha) # OBS: contas recentes (2025) mostram que escolher sigma igual à 1 - alpha é mais interessante
    println("")
    # Passo 4
    mu_wk = dot(x, s)/n
    w_n = zeros(2*n+m)
    t = 1
    while true
        println("Testagem: ", t)
        println("Alfa = ", alpha)
        println("Sigma = ", sig)
        #    w_n .= w + alpha*d
        for i=1:2*n+m
            w_n[i] = w[i] + alpha*d[i]
        end
        b_status = Broyden(F_tau, Jw, w_n, mu_wk, sig, sig_max, m, n, eps, it_max)
        Jw.size = 0 # Resetar a estrutura GoodBroyden para a próxima iteração
        #    if minimum([minimum(w_n[1:n]), minimum(w_n[n+m+1: 2*n+m])]) < -sqrt(eps)
        #      error("O ponto é inviável!")
        #    end

        println("mu (final do broyden)=", dot(w_n[1:n], w_n[n+m+1:2n+m])/n)
        #    println("w_n= ", w_n)

        println("Status (Broyden) = ", b_status)
        #    mu_n = (w_n[1:n]'*w_n[n+m+1:2*n+m])/n
        if b_status == true # Se encontrar um ponto em F_0 com decrescimo de mu, pare.
            break
        end
        println("⏰ Tempo para recalcular alpha e sigma:")
        alpha *= 0.5
        sig = min(sig_max, 1 - alpha) # OBS: contas recentes (2025) mostram que escolher sigma igual à 1 - alpha é mais interessante
        #sig = 0.5*(sig_max + sig)
        println("")
        if abs(sig_max - sig) < eps # Parar se sig se aproximar muito de sig_max (e depois acusar erro: sig_max não é grande suficiente ou o ponto inicial tomado não está próximo o suficiente do caminho central)
            status = false
            break
        end
        t += 1
    end
    # Passo 5
    return (w_n, status)
end

function PC_NQN(c, A, b, w, sig_max = 1-1.0e-4, eps=1.0e-8, it_max1=1000, it_max2=100)
    m, n = size(A)
    k = 0
    w0 = copy(w)

    for i=1:n
        if w0[i] < 0
            error("Ponto inicial inviável.")
        end
        if w0[n+m+i] < 0
            error("Ponto inicial inviável.")
        end
    end


    F_tau(w, tau) = begin # TODO: A função F_tau é uma das que mais consomem memória. As vezes entra em garbage colector também. Uma das principais candidatas a receber melhorias.
      println("⏰ Tempo para calcular F_tau:")
      @time begin
        x = view(w[1:n], :) # Sem cópias
        lamb = view(w[n+1:n+m], :) # Sem cópias
        s = view(w[n+m+1:2*n+m], :) # Sem cópias
        v = zeros(2*n+m)

#        rc = v[1:n]
#        rb = v[n+1:n+m]
#        r_mu = v[n+m+1:2*n+m]
        prod = nothing
        for i=1:n
#            v[i] = dot(A[:, i], lamb) + s[i] - c[i]
            for j=1:m # Calcula o produto linha-coluna de A^T lambda
              prod = A[j, i]
              prod *= lamb[j]
              v[i] += prod
            end
            v[i] += s[i]
            v[i] -= c[i]
            prod = x[i]
            prod *= s[i]
            v[n+m+i] += prod
            v[n+m+i] -= tau
        end
        for i=1:m
#            v[n+i] = dot(A[i, :], x) - b[i]
             for j=1:n # Calcula o produto linha-coluna de A x
              prod = A[i, j]
              prod *= x[j]
              v[n+i] += prod
            end
            v[n+i] -= b[i]
        end
      end
      println("")
        return v    

        #    if tau == 0
        #      return vcat(A'*lamb + s - c, A*x-b, diagm(x)*diagm(s)*ones(n))
        #    end
        #    return vcat(A'*lamb + s - c, A*x-b, diagm(x)*diagm(s)*ones(n) - tau*ones(n))
    end
    J(w) = begin # TODO: A função J é uma das que mais consomem memória. As vezes entra em garbage colector também. Uma das principais candidatas a receber melhorias.
      println("⏰ Tempo para calcular J:")
      @time begin
        x = w[1:n]
        s = w[n+m+1:2*n+m]
        M = spzeros(2*n+m, 2*n+m) # Cria uma matriz de zeros, mas sem armazenar zeros
#        M[1:n, n+1:n+m] = A'
#        M[1:n, n+m+1:2*n+m] = I(n)
#        M[n+1:n+m, 1:n] = A
#        M[n+m+1:2*n+m, 1:n] = diagm(s)
#        M[n+m+1:2*n+m, n+m+1:2*n+m] = diagm(x)
        for i=1:m # Esse for preenche as matrizes A e A'
          for j=1:n
            if abs(A[i, j]) > 1.0e-8
              M[j, n+i] = A[i, j]
              M[n+i, j] = A[i, j]
            end
          end
        end
        for j=1:n # Esse for preenche a identidade I, e as matrizes diagonais X e S
          M[j, n+m+j] = 1.0
          if s[j] > 1.0e-8
            M[n+m+j, j] = s[j]
          end
          if x[j] > 1.0e-8
            M[n+m+j, n+m+j] = x[j]
          end
        end
        dropzeros!(M) # Remove quaisquer zeros remanescentes na memória
      end
      println("")
        return M
        #    return vcat(hcat(zeros(n, n), A', I),
        #             hcat(A, zeros(m, m+n)),
        #             hcat(diagm(s), zeros(n, m), diagm(x))
        #               )
    end

    #  for i=1:n
    #  if w0[i] < 0
    #    w0[i] = abs(w0[i])
    #  end
    #  if w0[n+m+i] < 0
    #    w0[n+m+i] = abs(w0[n+m+i])
    #  end
    #  end

    while k < it_max1
        k += 1
        println("")
        println("Nº da iteração = ", k)
        status = true
        println("mu_$(k-1) = ", (1/n)*w0[1:n]'w0[n+m+1:2*n+m])
        (w0, status) = iteracao(F_tau, J, w0, it_max2, eps, sig_max, m, n)
        mu = (w0[1:n]'*w0[n+m+1:2*n+m])/n
        if status == false
            error("Não foi possível determinar alpha e sigma de modo a obter a convergência do passo corretor. Tente outro ponto inicial que esteja mais próximo do caminho central ou tome sig_max ainda mais próximo de 1.")
        end
        if mu < eps
            break
        end
    end
    println("")
    println("RESULTADO FINAL:")
    println("Nº de Iterações: ", k)
    println("Solução:")
    println("x = ")
    display(w0[1:n])
    println("lambda = ")
    display(w0[n+1:n+m])
    println("s = ")
    display(w0[n+m+1:2*n+m])
    #return (w0, k)
end

using ForwardDiff

function Encontrar_Ponto_Inicial(c, A, b, M = 1.0, mu = 10.0, eps = 1.0e-4, it_max=1000)
    m, n = size(A)
    F(w) = begin
        x = w[1:n]
        lamb = w[n+1:n+m]
        s = w[n+m+1:2*n+m]
        return vcat(A'*lamb + s - c, A*x-b, diagm(x)*diagm(s)*ones(n) - mu*ones(n))
    end
    JF(w) = ForwardDiff.jacobian(F, w)
    w0 = rand(2*n+m)
    for i=1:(2n+m)
        w0[i] = abs(w0[i])
    end
    w0 = M*w0
    status = false
    pos = true
    k = 0
    while k < it_max
        J = JF(w0)
        d = - J'*((J*J') \ F(w0))
        alpha = 1.0
        while alpha >= eps
            w1 = w0 + alpha*d
            x1 = w1[1:n]
            s1 = w1[n+m+1:2*n+m]
            if (F(w1) <= F(w0)) && minimum(x1) > eps && minimum(s1) > eps
                pos = true
                break
            end
            pos = false
            alpha *= 0.5
            if alpha < eps
                return (w0, status)
            end
        end
        w0 = w0 + alpha*d
        if (norm(F(w0)) < eps) && (pos == true)
            status = true
        end
        k += 1
    end
    return (w0, status)
end

function chutar_e_encontrar(c, A, b, mu = 10, M = 1.0, eps = 1.0e-4, it_max=1000)
    # Essa procura a solução até encontrar (pode demorar muito)
    while true
        (w, status) = Encontrar_Ponto_Inicial(c, A, b, M, mu, eps, it_max)
        if status == true
            return w
        end
    end
end

function chutar_e_encontrar2(c, A, b, mu = 10, M = 1.0e12, eps = 1.0e-4, it_max=1000)
    # Essa procura a solução até encontrar (pode demorar muito)
    m, n = size(A)
    F(w) = begin
        x = w[1:n]
        lamb = w[n+1:n+m]
        s = w[n+m+1:2*n+m]
        return vcat(A'*lamb + s - c, A*x-b, diagm(x)*diagm(s)*ones(n) - (1/n)*dot(x, s)*ones(n))
    end
    w = ones(2*n+m)
    g(w) = begin
        x = w[1:n]
        lamb = w[n+1:n+m]
        s = w[n+m+1:2*n+m]

        F0 = F(w)
        sum = norm(F0)^2#dot(F0, F0)
        for i=1:n
            sum += M * (min(x[i], 0)^2 + min(s[i], 0)^2)
        end
        return sum
    end
    dg(w) = ForwardDiff.gradient(g, w)
    while norm(dg(w)) > 1.0e-4
        println(g(w))
        d = -dg(w)
        alpha = 1.0
        while g(w+alpha*d) > g(w) + 0.5*alpha*dot(d, dg(w))
            alpha *= 0.5
        end
        w = w + alpha*d
    end

    x = w[1:n]
    lamb = w[n+1:n+m]
    s = w[n+m+1:2*n+m]

    println("mu = $(dot(x, s)/n)")

    return w 

end

function chutar_e_encontrar3(c, A, b, mu = 10, M = 1.0e12, eps = 1.0e-4, it_max=1000)
    # Essa procura a solução até encontrar (pode demorar muito)
    m, n = size(A)
    F2(w) = begin
        x = w[1:n]
        lamb = w[n+1:n+m]
        s = w[n+m+1:2*n+m]
        return vcat(A'*lamb + s - c, A*x-b, diagm(x)*diagm(s)*ones(n) - mu*ones(n))
    end
    while true
        w = rand(2*n+m)
        DF2(w) = ForwardDiff.jacobian(F2, w)
        while norm(F2(w)) > 1.0e-4
            
            println(norm(F2(w)))
            DF2w = DF2(w)
            d = (DF2w'*DF2w) \ (- DF2w'*F2(w))

            x = w[1:n]
            lamb = w[n+1:n+m]
            s = w[n+m+1:2*n+m]
            alpha = 1.0
            it = 0
            while minimum(x+alpha*d[1:n]) <= 0 && minimum(s+alpha*d[m+n+1:2*n+m]) <= 0 && it <= 500
                alpha *= 0.5
                it += 1
            end
            w += alpha*d
            if it > 500
                break
            end
        end

    end

    

    return w
end

function Encontrar_Ponto_Inicial2(c, A, b, mu = 1.0, theta = 0.25, eps = 1.0e-4, it_max=1000)
    # Essa procura a solução até encontrar (pode demorar muito)
    while true
        w = Encontrar_Ponto_Inicial(c, A, b, mu, theta, eps, it_max)
        if !(w == false)
            return w
        end
    end
end

function chamar_problema_simples(i)

    if i == 1
        c = [1., 0]
        A = [1. 1]
        b = [1]
        return (c, A, b)
    elseif i == 2
        c = [-1., -1, 0, 0]
        A = [-1. 3 1 0; 5 -3 0 1]
        b = [9., 15]
        return (c, A, b)
    elseif i == 3
        c = [0, 1., 0]
        A = [1. 2 1]
        b = [6.]
        return (c, A, b)
    else
        error("Problema não encontrado.")
    end

end


# EX:
# min x_1
# s.a: x_1 + x_2 = 1
# (x_1, x_2) >= 0
# c = [1., 0]
# A = [1. 1]
# b = [1.]
