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
LinearAlgebra.ldiv!(A::GoodBroyden, b) = begin # WARNING: Essa função começa a consumir mais memória com o passar das iterações. Esse aumento é bem lento, e mesmo em um problema grande (QAP12) o uso de memória foi bem razoável para 100 iterações, então creio que não vai ser um problema.

    # Resolve o caso base
    println("⏰ Tempo para resolver um sistema envolvendo B_0:")
    @time ldiv!(A.lu, b)
    println("")
    println("⏰ Tempo dedicado a resolver o sistema original (Broyden)")
    m = A.size[]
    u = nothing
    sb = nothing
    rho = nothing
    prod = nothing
    @time for i = 1:m
      u = A.u[i] # Sem intenção de fazer cópias
      sb = A.sb[i] # Sem intenção de fazer cópias
#        rho = A.rho[i]
        rho = 0
        for j=1:m # produto interno dot(sb, b)
          prod = sb[j]
          prod *= b[j]
          rho += prod
        end
        rho /= A.rho[i]

#        rho = dot(sb, b) / rho
#        b .= b + (dot(sb, b) / rho) * u
#        b .= b + rho * u
        for j=1:m
          b[j] = b[j] + rho * u[j]
        end

    end
    println("")

    nothing

end

#LinearAlgebra.ldiv!(A::GoodBroyden, b) = begin
#
#    # Resolve o caso base
#    ldiv!(A.lu, b)
#    for i = 1:A.size[]
#        u = A.u[i]
#        sb = A.sb[i]
#        rho = A.rho[i]
#        b .= b + (dot(sb, b) / rho) * u
#
#    end
#
#    nothing
#
#end

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

#function update!(B::GoodBroyden, s, b)
#
#    size = B.size + 1
#    B.u[size] = b
#    ldiv!(B, B.u[size])
#    B.rho[size] = dot(s, s - B.u[size]) 
#    B.sb[size] = s
#    B.size    = size
#
#end

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
    if 0 <= (dot(w[1:n], w[n+m+1: 2*n+m]) / n) < sig*mu_wk && minimum([minimum(w[1:n]), minimum(w[n+m+1: 2*n+m])]) > 0 #-sqrt(eps) # trocar < por <= se nao der certo
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

function Broyden(F_tau, B, w, mu_wk, sig, sig_max, m, n, eps, it_max) # WARNING: Essa função faz algumas contas, principalmente produtos internos com matrizes grandes, e pode ser otimizada

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
    global nitb += it
    println("Nº de iterações de Broyden: ", it)
    return status
end

function Quasi_Newton_Corrector!(mpc :: MPC, z, dz, sig_max = 1-1.0e-4, eps=1.0e-8, it_max = 5)

    #1-1.0e-4

    # Names
    dat = mpc.dat
    pt = mpc.pt
    res = mpc.res

    m, n, p = pt.m, pt.n, pt.p

    A = dat.A
    b = dat.b
    c = dat.c

    w = vcat(pt.x, pt.y, z) # WARNING: NÃO usar vcat, isso consome muita memória


###
    for i=1:n
        if w[i] < 0 || w[n+m+i] < 0
            println("WARNING: O ponto atual não está em F_0. A escolha de alpha não faz mais sentido.")
            break
        end
    end
###

#    F_tau(w, tau) = begin
#        x = w[1:n]
#        lamb = w[n+1:n+m]
#        s = w[n+m+1:2*n+m]
#        v = zeros(2*n+m)
#
#        for i=1:n
#            v[i] = dot(A[:, i], lamb) + s[i] - c[i]
#            if tau == 0
#                v[n+m+i] = x[i]*s[i]
#            else
#                v[n+m+i] = x[i]*s[i] - tau
#            end
#        end
#        for i=1:m
#            v[n+i] = dot(A[i, :], x) - b[i]
#        end
#
#        return v    
#
#    end
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
#    J(w) = begin
#        x = w[1:n]
#        s = w[n+m+1:2*n+m]
#        M = zeros(2*n+m, 2*n+m)
#        M[1:n, n+1:n+m] = A'
#        M[n+1:n+m, 1:n] = A
#        for i=1:n
#            M[i, n+m+i] = 1.0
#            M[n+m+i, i] = s[i]
#            M[n+m+i, n+m+i] = x[i]
#        end
#        return M
#    end
    J(w) = begin # TODO: A função J é uma das que mais consomem memória. As vezes entra em garbage colector também. Uma das principais candidatas a receber melhorias.
      println("⏰ Tempo para calcular J:")
      @time begin
        x = view(w[1:n], :) # Sem cópias
        s = view(w[n+m+1:2*n+m], :) # Sem cópias
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
          if abs(s[j]) > 1.0e-8 # TODO: Remover o abs (coloquei somente como salvaguarda para erros numéricos, mas s teoricamente já deveria ser positivo)
            M[n+m+j, j] = s[j]
          end
          if abs(x[j]) > 1.0e-8 # TODO: Remover o abs
            M[n+m+j, n+m+j] = x[j]
          end
        end
#        dropzeros!(M) # Remove quaisquer zeros remanescentes na memória
      end
      println("")
        return M
        #    return vcat(hcat(zeros(n, n), A', I),
        #             hcat(A, zeros(m, m+n)),
        #             hcat(diagm(s), zeros(n, m), diagm(x))
        #               )
    end


#    x = w[1:n]
#    lamb = w[n+1:n+m]
#    s = w[n+m+1:2*n+m]
###
    x = pt.x
    lamb = pt.y
    s = z
###

#    J_temp = J(w)
##    println("Jac=")
##    display(J_temp)
#
#    # correção para garantir a decomposição lu (resolve a questão da invertibilidade da matriz, mas a convergencia ainda é muito improvavel)
#    corrigiu_jac = false
#    if true
#    fatoracao = nothing
#    mult = 1
#    while true
#        try fatoracao = lu(J_temp)
#            break
#        catch
#            J_temp += (2^mult)*eps*I
#            #J_temp[(n+m+1):(2*n+m), 1:n] += (2^mult)*eps*I
#            #J_temp[(n+m+1):(2*n+m), (n+m+1):(2*n+m)] += (2^mult)*eps*I
#            mult += 1
#            corrigiu_jac = true
#        end
#    end
#  else
#    fatoracao = lu(J_temp)
#  end
#    if corrigiu_jac == true
#        global n_corr_jac += 1
#    end
#    # ----------------------------------------
#
#    Jw = GoodBroyden(fatoracao, it_max)
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
    J_temp = copy(Jw) # TODO: Remover a necessidade de guardar a Jacobiana como uma matriz normal (não GoodBroyden) para o método alternativo, e passar a matriz GoodBroyden para ele.
    Jw = GoodBroyden(Jw, it_max)
    # Passo 1
#    d = - (fatoracao \ F_tau(w, 0)) #vcat(mpc.Δ.x, mpc.Δ.y, dz) # Espaço para otimização 
    d = - (J_temp \ F_tau(w, 0)) #vcat(mpc.Δ.x, mpc.Δ.y, dz) # Espaço para otimização 
    mpc.Δ.x = d[1:n] # TODO: Trocar por view
    mpc.Δ.y = d[n+1:n+m] # TODO: Trocar por view

    # Passo 2
    
#  alpha = 10.0
#  for i=1:n
#    if d[i] < 0
#      temp = - w[i]/d[i]
#      if temp < alpha
#        alpha = temp
#      end
#    end
#    if d[n+m+i] < 0
#      temp = - w[n+m+i]/d[n+m+i]
#      if temp < alpha
#        alpha = temp
#      end
#    end
#    if alpha < 0
#      alpha = 0
#      break
#    end
#  end
#
#    alpha0 = min(1.0, 0.9*alpha)
##    alpha0 = 0.9*alpha
#    mpc.αp, mpc.αd = alpha0, alpha0
#    alpha = alpha0
    println("⏰ Tempo para encontrar alpha para o passo de Newton:")
    @time begin
    dx = view(d, 1:n) # Sem cópias
    ds = view(d, n+m+1:2*n+m) # Sem cópias
    pctg = 0.99
    alpha = 1.0
    aerr = nothing
    for i=1:n
        if dx[i] < 0
            alpha = min(-pctg*x[i] / dx[i], 1.0)
        end
        if ds[i] < 0
            alpha = min(-pctg*s[i] / ds[i], 1.0)
        end
    end
    alpha = max(alpha, 0) # calcula alpha máximo tal que algum xi ou si zera e depois toma uma fração desse passo, ou passo 1 no caso em que nenhuma variável bloqueia o passo.
    mpc.αp, mpc.αd = alpha, alpha
    alpha0 = alpha
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
        global n_tent_broyden += 1
        println("Testagem: ", t)
        println("Alfa = ", alpha)
        println("Sigma = ", sig)

#        w_n .= w + alpha*d
        for i=1:2*n+m # Teoricamente é garantido que x e s são positivos, mas erros numéricos acontecem
            w_n[i] = w[i] + alpha*d[i]
            if 1 <= i <= n || n+m+1 <= i <= 2*n+m

                            if minimum(w_n[i]) < 0
                error("A estratégia para calcular alpha não funcionou: w_n[$(i)] = $(w_n[i]);\n $(w[i]) + $(alpha) * $(d[i])")
            end


              w_n[i] = abs(w_n[i])
              if w_n[i] == 0
                error("Talvez precisemos forçar w_n[i] > eps^2")
              end
            end
        end


        println("Ponto inicial do Broyden")
        display(w_n)

        println("mu (após newton) = ", dot(w_n[1:n], w_n[n+m+1:2*n+m])/n)

        b_status = Broyden(F_tau, Jw, w_n, mu_wk, sig, sig_max, m, n, eps, it_max)
        Jw.size = 0 # Resetar a estrutura GoodBroyden para a próxima iteração

        println("mu (final do broyden)=", dot(w_n[1:n], w_n[n+m+1:2n+m])/n)
        println("Status (Broyden) = ", b_status)
        if b_status == true # Se encontrar um ponto em F_0 com decrescimo de mu, pare.
            break
        end
        alpha *= 0.5
#        mpc.αp, mpc.αd = alpha, alpha # Acho que deveria atualizar esse também
#        sig = 0.5*(sig_max + sig)
        sig = min(sig_max, 1 - alpha) # OBS: contas recentes (2025) mostram que escolher sigma igual à 1 - alpha é mais interessante
        #    if abs(sig_max - sig) < 1.0e-8 # Parar se sig se aproximar muito de sig_max (e depois acusar erro: sig_max não é grande suficiente ou o ponto inicial tomado não está próximo o suficiente do caminho central)
        
###### AQUI COMEÇA O MÉTODO ALTERNATIVO

        if t == 3 # 30 iterações é suficiente para praticamente zerar a diferença entre sig_max e sig (ela fica na ordem de 4.66e-10)
            #            error("Não foi possível determinar alpha e sigma de modo a obter a convergência do passo corretor. Tente outro ponto inicial que esteja mais próximo do caminho central ou tome sig_max ainda mais próximo de 1.")
            println("AVISO: Não foi possível determinar alpha e sigma de modo a obter a convergência do passo corretor. Isso pode ter ocorrido pois o ponto inicial não estava próximo o suficiente do caminho central. Para contornar isso, será aplicado um método quasi-newton alternativo.")
#            error("AVISO: Não foi possível determinar alpha e sigma de modo a obter a convergência do passo corretor. Isso pode ter ocorrido pois o ponto inicial não estava próximo o suficiente do caminho central. Para contornar isso, será aplicado um método quasi-newton alternativo.")
            global n_corr_alt += 1
            an = alpha0
            for i=1:(2*n+m)
              w_n[i] = w[i] + an*d[i]
            if 1 <= i <= n || n+m+1 <= i <= 2*n+m
              w_n[i] = abs(w_n[i])
            end
            end
            while true
              if 0 <= dot(w_n[1:n], w_n[n+m+1:2*n+m])/n <= mu_wk #&& min(minimum(w_n[1:n]), minimum(w_n[n+m+1:2*n+m])) > eps^2
                break
              else
                an *= 0.5
                w_n .= w .+ an*d
              end
            end
#        mpc.αp, mpc.αd = an, an # Acho que deveria atualizar esse também
#            sig_bom = ((dot(w_n[1:n], w_n[n+m+1:2*n+m])/n)/(dot(w[1:n],w[n+m+1:2*n+m])/n))^3
            sig_bom = 1 - an # Sigma que vi recentemente que funciona (2025)
            println("sig_bom = ", sig_bom)
            tau = sig_bom*dot(w_n[1:n], w_n[n+m+1:2*n+m])/n
            w_n2 = zeros(size(w_n)[1])
            for k=1:5
                try
#                dc = - (J_temp \ F_tau(w_n, tau))
                dc = - F_tau(w_n, tau)
                ldiv!(Jw, dc)
                global nitb += 1
                ac = 1.0
                w_n2 .= w_n .+ dc
                while true
                  if 0 <= dot(w_n2[1:n], w_n2[n+m+1:2*n+m])/n <= dot(w_n[1:n], w_n[n+m+1:2*n+m])/n && minimum([minimum(w_n2[1:n]), minimum(w_n2[n+m+1:2*n+m])]) > 0
                  break
                else
                ac *= 0.5
                w_n2 .= w_n .+ ac*dc
            end
                is_nan = false
                for kk=1:length(w_n)
                    if isnan(w_n2[kk])
                        is_nan = true
                        break
                    end
                end
   
                if is_nan == true
                w_n2 .= w_n 
                    break
                end
            if ac < 1.0e-8
                w_n2 .= w_n # se ac for muito pequeno, não faça nada.
                break
            end
          end
        catch
          println("Eu sai com = $(k). Não devo sair com k = 1, pois isso seria ruim.")
          w_n .= w_n2
          break
      end
                yk = F_tau(w_n2, tau) - F_tau(w_n, tau)
                sk = w_n2 - w_n
                w_n .= w_n2
                Jw.lu = Jw.lu.L*Jw.lu.U # reconstroi Jw
                Jw.lu = Jw.lu + ((yk - Jw.lu*sk)*(sk'))/(dot(sk,sk))
        end
            break

        end

        t += 1
    end
    # Passo 5 (adaptado: atualizar direcao)
    #    println("w após broyden: ", w_n)
    mpc.pt.x = w_n[1:n]
    
    # atualizando xl e xu (para o tulip)
    
    mpc.pt.xl .= mpc.pt.x - mpc.dat.l
    mpc.pt.xu .= mpc.dat.u - mpc.pt.x
    
    # ---
    
    mpc.pt.y = w_n[n+1:n+m]
    mpc.pt.z .= w_n[n+m+1:2*n+m]


    
    # atualizando zl e zu (para o tulip) [provisorio]
    
    mpc.pt.zl .= copy(mpc.pt.z)
    mpc.pt.xu .= 0
    
    # ---
    
    w_n .= w_n - w
    mpc.Δc.x .= w_n[1:n]
    mpc.Δc.y .= w_n[n+1:n+m] # talvez precise atualizar mais coisas em delta_c

    if t == 30
        return true
    else
        return false
    end
end
