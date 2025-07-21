using LinearAlgebra
import LinearAlgebra: ldiv!

mutable struct GoodBroyden{S}

    qnc
    u :: Vector{Vector{S}}
    sb :: Vector{Vector{S}}
    rho :: Vector{S}
    size :: Int

    function GoodBroyden(qnc, max_size=10)
        return new{Float64}(qnc, Vector{Vector{Float64}}(undef, max_size), Vector{Vector{Float64}}(undef, max_size), Vector{Float64}(undef, max_size), 0)
    end
end

"""
    ldiv!(A::GoodBroyden, b, x)

Solves the linear system \$A x = b\$ where \$A\$ is given by the Good Broyden update.
"""
LinearAlgebra.ldiv!(A::GoodBroyden, sig) = begin # WARNING: Essa função começa a consumir mais memória com o passar das iterações. Esse aumento é bem lento, e mesmo em um problema grande (QAP12) o uso de memória foi bem razoável para 100 iterações, então creio que não vai ser um problema.

    # Resolve o caso base
    println("⏰ Tempo para resolver um sistema envolvendo B_0:")

    compute_corrector!(A.qnc, sig) # Pressupõe que o lado direito correto já está armazenado em qnc. Também pressupõe que o ponto atual armazenado em qnc seja o ponto antes do passo preditor (pois caso contrário, B_0 não seria a jacobiana utilizada no passo preditor).
    
    Δc = A.qnc.Δc

    b = vcat(Δc.x, Δc.xl, Δc.xu, Δc.y, Δc.zl, Δc.zu) # Constrói o lado direito

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

#    rp, rl, ru, rd, rxzl, rxzu = deconcatenate(A.qnc, b)
    Δc = A.qnc.Δc
    Δc.x, Δc.xl, Δc.xu, Δc.y, Δc.zl, Δc.zu = deconcatenate(A.qnc, b)

    nothing

end

#LinearAlgebra.ldiv!(A::GoodBroyden, b) = begin
#
#    # Resolve o caso base
#    ldiv!(A.qnc, b)
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
    mpc.nitb += it
    println("Nº de iterações de Broyden: ", it)
    return status
end

function positivity_test(mpc)
    pt = mpc.pt
    n = mpc.pt.n
    # Teste de positividade de xl, zl, xu, zu sem gambiarra pra compatibilizar com o Tulip
#    num1 = length(pt.xl)
#    num2 = length(pt.xu)
    for i=1:n
        if pt.xl[i] < 0 || pt.zl[i] < 0
            return false
        end
    end
    for i=1:n
        if pt.xu[i] < 0 || pt.zu[i] < 0
            return false
        end
    end
    return true
end

function decrease_and_feasibility_test(mpc, cp_mu, sig)
    pt = mpc.pt
    update_mu!(pt)
    if (pt.μ <= 0.5 * (1.0 + sig) * cp_mu) && positivity_test(mpc)
        return true
    end
    pt.μ = cp_mu
    return false
end

function Broyden_convergence_test(mpc, eps = 1.0e-8)
    for v in (mpc.ξp, mpc.ξl, mpc.ξu, mpc.ξd, mpc.ξxzl, mpc.ξxzu)
        if norm(v) > eps
            return false
        end
    end
    return true
end

function Broyden_parada2(mpc, cp_mu, it, it_max, eps, sig)
    convergence = false
    accept_point = false
    stop = false
    if Broyden_convergence_test(mpc, eps)
        convergence = true
    end
    if decrease_and_feasibility_test(mpc, cp_mu, sig)
       accept_point = true
       stop = true
    end
    if convergence || it >= it_max
        stop = true
    end
    return stop, convergence, accept_point
end 

function deconcatenate(mpc, b)
    m = mpc.pt.m
    n = mpc.pt.n
    rp = b[1:m]
    rl = b[m+1:m+n]
    ru = b[m+n+1:m+2*n] 
    rd = b[m+2*n+1:m+3*n] 
    rxzl = b[m+3*n+1:m+4*n] 
    rxzu = b[m+4*n+1:m+5*n] 
    return rp, rl, ru, rd, rxzl, rxzu
end

function Broyden2!(GB_mpc, alpha, sig, cp_mu, it_max, eps, cp_x, cp_xl, cp_xu, cp_y, cp_zl, cp_zu)
  mpc = GB_mpc.qnc
  pt = mpc.pt
  Δ = mpc.Δ
  Δc = mpc.Δc

  ### 1ª iteração de Broyden

  it = 1

  # Calcula os resíduos no ponto atual (lado direito em solve_newton_system!)

    compute_residuals!(mpc)

  # Recuperar a jacobiana original (na função solve_newton_system!, a jacobiana é calculada no ponto atual guardado em mpc. Isto significa que não estaríamos usando a mesma jacobiana do passo preditor. Portanto, ao retornar ao ponto anterior ao passo preditor, estamos garantindo que a mesma jacobiana do passo preditor será utilizada como B_0 pelo método de Broyden)

  pt.x  .-= alpha .* Δ.x
  pt.xl .-= alpha .* Δ.xl
  pt.xu .-= alpha .* Δ.xu
  pt.y  .-= alpha .* Δ.y
  pt.zl .-= alpha .* Δ.zl
  pt.zu .-= alpha .* Δ.zu
  
  compute_corrector!(mpc, sig) # Pressupõe que os resíduos após o passo de Newton estejam guardados em mpc

  # Recupera o ponto após o passo de Newton e já soma a direção de Broyden
  pt.x  .+= alpha .* Δ.x  .+ Δc.x
  pt.xl .+= alpha .* Δ.xl .+ Δc.xl
  pt.xu .+= alpha .* Δ.xu .+ Δc.xu
  pt.y  .+= alpha .* Δ.y  .+ Δc.y
  pt.zl .+= alpha .* Δ.zl .+ Δc.zl
  pt.zu .+= alpha .* Δ.zu .+ Δc.zu

  it += 1

  ### Fim da 1ª iteração de Broyden

  ### Iterações posteriores...

    while true # Main loop

    # Stopping criteria

    stop, convergence, accept_point = Broyden_parada2(mpc, cp_mu, it, it_max, eps, sig)
    if stop == true
        mpc.nitb += it # contabiliza as iterações de Broyden
        return accept_point
    end

#    # Calcula os resíduos no ponto atual (lado direito em solve_newton_system!)
#
#    compute_residuals!(mpc)
#    
#    # Copia o ponto atual
#
#    cp_x2  = copy(pt.x)
#    cp_xl2 = copy(pt.xl)
#    cp_xu2 = copy(pt.xu)
#    cp_y2  = copy(pt.y)
#    cp_zl2 = copy(pt.zl)
#    cp_zu2 = copy(pt.zu)
#
#    # Retorna ao ponto anterior ao passo preditor
#
#    pt.x  .= cp_x 
#    pt.xl .= cp_xl
#    pt.xu .= cp_xu
#    pt.y  .= cp_y 
#    pt.zl .= cp_zl
#    pt.zu .= cp_zu
#
#    ldiv!(GB_mpc, sig)

    it += 1

    end

end

function compute_residuals_for_Broyden!(rp, rd, rl, ru, x, y, xl, xu, zl, zu, alpha, mpc::MPC{T}) where{T} # TODO: Não deixar a função sobrescrever o mpc

#    pt, res = mpc.pt, mpc.res
    dat = mpc.dat

    # Primal residual
    # rp = b - A*x
    rp .= dat.b
    mul!(rp, dat.A, x, -one(T), one(T))

    # Lower-bound residual
    # rl_j = l_j - (x_j - xl_j)  if l_j ∈ R
    #      = 0                   if l_j = -∞
    @. rl = ((dat.l + xl) - x) * dat.lflag # TODO: Não sei o que fazer com isso ainda, pois o nosso método usa uma formulação diferente para o problema.

    # Upper-bound residual
    # ru_j = u_j - (x_j + xu_j)  if u_j ∈ R
    #      = 0                   if u_j = +∞
    @. ru = (dat.u - (x + xu)) * dat.uflag # TODO: Aqui também.

    # Dual residual
    # rd = c - (A'y + zl - zu)
    rd .= dat.c
    mul!(rd, transpose(dat.A), y, -one(T), one(T))
    @. rd += zu .* dat.uflag - zl .* dat.lflag # TODO: Aqui também.

    # Residuals norm
#    res.rp_nrm = norm(res.rp, Inf)
#    res.rl_nrm = norm(res.rl, Inf)
#    res.ru_nrm = norm(res.ru, Inf)
#    res.rd_nrm = norm(res.rd, Inf)
#
#    # Compute primal and dual bounds
#    mpc.primal_objective = dot(dat.c, pt.x) + dat.c0
#    mpc.dual_objective   = (
#        dot(dat.b, pt.y)
#        + dot(dat.l .* dat.lflag, pt.zl)
#        - dot(dat.u .* dat.uflag, pt.zu)
#    ) + dat.c0

    return nothing
end

function compute_first_system!(v, rp, rl, ru, rd, xl, xu, zl, zu, tau, mpc::MPC) # TODO: Certificar-se de usar a função compute_residuals_for_Broyden para atualizar os resíduos para o ponto inicial de Broyden

    # Newton RHS
    copyto!(mpc.ξp, rp)
    copyto!(mpc.ξl, rl)
    copyto!(mpc.ξu, ru)
    copyto!(mpc.ξd, rd)

    # TODO: O \mu calculado pelo Tulip diz respeito a um problema onde l <= x <= u. Nesse caso, eles introduzem variáveis de folga xl e xu, deixando x livre. Portanto, as variáveis de folga xl e xu são utilizadas para o calculo de \mu. O \mu calculado dessa forma só será o mesmo \mu que eu estou calculando no meu código se l = 0 e u = +\infty. Portanto, devo revisar o meu código para que tau = sig*mu_wk esteja de acordo com a forma esperada pelo Tulip nas equações abaixo.

    @. mpc.ξxzl = tau * mpc.dat.lflag .- (xl .* zl) .* mpc.dat.lflag 
    @. mpc.ξxzu = tau * mpc.dat.uflag .- (xu .* zu) .* mpc.dat.uflag

    # Compute affine-scaling direction
    @timeit mpc.timer "Newton" solve_newton_system!(v, mpc,
        mpc.ξp, mpc.ξl, mpc.ξu, mpc.ξd, mpc.ξxzl, mpc.ξxzu
    )

    # TODO: check Newton system residuals, perform iterative refinement if needed
    return nothing
end

function Quasi_Newton_Corrector!(mpc::QNC, params, sig_max = 1-1.0e-4, eps=1.0e-8, it_max = 5)

    #1-1.0e-4

    # Names
    dat = mpc.dat
    pt = mpc.pt
    res = mpc.res

    Δ  = mpc.Δ
    Δc = mpc.Δc

    # pt_x_cp = copy(pt.x) # Se eu precisar copiar objetos dentro dessa estrutura para não perdê-los, vou fazer mais ou menos assim. 

    m, n, p = pt.m, pt.n, pt.p

    A = dat.A
    b = dat.b
    c = dat.c

#    ###### TODO: É assim que vou chamar a função do Tulip que resolve os sistemas
#
#    alpha = 0.5
#    sig = 0.5
#    mu = 0.8
#
#    xl = false * mpc.Δ.xl
#    xu = false * mpc.Δ.xu
#    zl = false * mpc.Δ.zl
#    zu = false * mpc.Δ.zu
#    xl .+= alpha .* mpc.Δ.xl
#    xu .+= alpha .* mpc.Δ.xu
#    zl .+= alpha .* mpc.Δ.zl
#    zu .+= alpha .* mpc.Δ.zu
#
#    rp = false * mpc.res.rp
#    rd = false * mpc.res.rd
#    rl = false * mpc.res.rl
#    ru = false * mpc.res.ru
#
#    compute_residuals_for_Broyden!(rp, rd, rl, ru, pt.x, pt.y, xl, xu, zl, zu, alpha, mpc)
#    compute_first_system!(mpc.Δc, rp, rl, ru, rd, xl, xu, zl, zu, sig*mu, mpc)
#
#    # Esse script já está funcionando. Agora falta inseri-lo no local certo do código e fazer as devidas adaptações para que o Broyden resolva o mesmo sistema que a função do Tulip (solve_newton_system).
#    # Algumas coisas que eu faço no meu código que não fazem mais sentido depois dessa mudança:
#    # - A forma como eu calculo \mu só faz sentido para problemas da forma
#    #       min c^T x
#    #       s.a.: Ax  = b
#    #              x >= 0
#    #   No Tulip, os resíduos são calculados para um problema onde l <= x <= u. Isso muda um pouco a forma como o \mu é calculado, mas coincide com a minha forma quando l = 0 e u = + \infty. Então, eu vou precisar rever a forma como calculo \mu no meu código e substituir pelo \mu calculado pelo Tulip.
#    #   Também não faz mais sentido eu calcular um vetor z, visto que o problema tem uma formulação diferente, e ele não será mais necessário. Além disso, o meu z só fará sentido quando l = 0 e u = +\infty, onde z seria igual a zl, e zu = 0.
#    #
#    #   Também é bom levar em conta que o Tulip resolve um sistema envolvendo as seguintes condições KKT:
#    #   A^T \lambda + zl - zu = c
#    #                     A x = b
#    #             xl_i * zl_i = \tau, i = 1, ..., n
#    #             xu_i * zu_i = \tau, i = 1, ..., n
#    #         xl, xu, zl, zu >= 0
#    #   Então, eles estão em busca de um ponto com 5 entradas (que também são vetores): (x, xl, xu, zl, zu).
#    #   Também vou deixar aqui mais duas fórmulas para eu lembrar depois:
#    #   x - xl = l
#    #   x + xu = u
#    
#    ##############


#    w = vcat(pt.x, pt.y, z) # WARNING: NÃO usar vcat, isso consome muita memória


###
#    for i=1:n
#        if w[i] < 0 || w[n+m+i] < 0
#            error("O ponto atual não está em F_0. A escolha de alpha não faz mais sentido quando w tem entradas negativas.")
#            break
#        end
#    end

    # Teste de positividade de xl, zl, xu, zu sem gambiarra pra compatibilizar com o Tulip
    pos_cond = positivity_test(mpc)
    if pos_cond == false
            error("O ponto atual não está em F_0. A escolha de alpha não faz mais sentido quando w tem entradas negativas.")
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
#      F_tau(w, tau) = begin # TODO: A função F_tau é uma das que mais consomem memória. As vezes entra em garbage colector também. Uma das principais candidatas a receber melhorias.
#        println("⏰ Tempo para calcular F_tau:")
#        @time begin
#          x = view(w[1:n], :) # Sem cópias
#          lamb = view(w[n+1:n+m], :) # Sem cópias
#          s = view(w[n+m+1:2*n+m], :) # Sem cópias
#          v = zeros(2*n+m)
#
#  #        rc = v[1:n]
#  #        rb = v[n+1:n+m]
#  #        r_mu = v[n+m+1:2*n+m]
#          prod = nothing
#          for i=1:n
#  #            v[i] = dot(A[:, i], lamb) + s[i] - c[i]
#              for j=1:m # Calcula o produto linha-coluna de A^T lambda
#                prod = A[j, i]
#                prod *= lamb[j]
#                v[i] += prod
#              end
#              v[i] += s[i]
#              v[i] -= c[i]
#              prod = x[i]
#              prod *= s[i]
#              v[n+m+i] += prod
#              v[n+m+i] -= tau
#          end
#          for i=1:m
#  #            v[n+i] = dot(A[i, :], x) - b[i]
#              for j=1:n # Calcula o produto linha-coluna de A x
#                prod = A[i, j]
#                prod *= x[j]
#                v[n+i] += prod
#              end
#              v[n+i] -= b[i]
#          end
#        end
#        println("")
#          return v    
#
#          #    if tau == 0
#          #      return vcat(A'*lamb + s - c, A*x-b, diagm(x)*diagm(s)*ones(n))
#          #    end
#          #    return vcat(A'*lamb + s - c, A*x-b, diagm(x)*diagm(s)*ones(n) - tau*ones(n))
#      end
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
#      J(w) = begin # TODO: A função J é uma das que mais consomem memória. As vezes entra em garbage colector também. Uma das principais candidatas a receber melhorias.
#        println("⏰ Tempo para calcular J:")
#        @time begin
#          x = view(w[1:n], :) # Sem cópias
#          s = view(w[n+m+1:2*n+m], :) # Sem cópias
#          M = spzeros(2*n+m, 2*n+m) # Cria uma matriz de zeros, mas sem armazenar zeros
#  #        M[1:n, n+1:n+m] = A'
#  #        M[1:n, n+m+1:2*n+m] = I(n)
#  #        M[n+1:n+m, 1:n] = A
#  #        M[n+m+1:2*n+m, 1:n] = diagm(s)
#  #        M[n+m+1:2*n+m, n+m+1:2*n+m] = diagm(x)
#          for i=1:m # Esse for preenche as matrizes A e A'
#            for j=1:n
#              if abs(A[i, j]) > 1.0e-8
#                M[j, n+i] = A[i, j]
#                M[n+i, j] = A[i, j]
#              end
#            end
#          end
#          for j=1:n # Esse for preenche a identidade I, e as matrizes diagonais X e S
#            M[j, n+m+j] = 1.0
#            if abs(s[j]) > 1.0e-8 # TODO: Remover o abs (coloquei somente como salvaguarda para erros numéricos, mas s teoricamente já deveria ser positivo)
#              M[n+m+j, j] = s[j]
#            end
#            if abs(x[j]) > 1.0e-8 # TODO: Remover o abs
#              M[n+m+j, n+m+j] = x[j]
#            end
#          end
#  #        dropzeros!(M) # Remove quaisquer zeros remanescentes na memória
#        end
#        println("")
#          return M
#          #    return vcat(hcat(zeros(n, n), A', I),
#          #             hcat(A, zeros(m, m+n)),
#          #             hcat(diagm(s), zeros(n, m), diagm(x))
#          #               )
#      end


#    x = w[1:n]
#    lamb = w[n+1:n+m]
#    s = w[n+m+1:2*n+m]
###
###    x = pt.x
###    lamb = pt.y
###    s = z
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
#    Jw = J(w)
#    println("⏰ Tempo para fatorar Jw (inclui o tempo que passa tentando corrigir a jacobiana até ser possível fatorar):")
#    @time begin
#    is_invertible=false
#    while is_invertible==false
#    try
#    Jw = lu(Jw)
#    is_invertible=true
#  catch
#        for i=1:2*n+m # Soma 1.0e-8*I em Jw para tentar fazer Jw se tornar inversivel
#      Jw[i, i] += 1.0e-8
#    end
#  end
#  end
#end
#    println("")
###    J_temp = copy(Jw) # TODO: Remover a necessidade de guardar a Jacobiana como uma matriz normal (não GoodBroyden) para o método alternativo, e passar a matriz GoodBroyden para ele.
###    Jw = GoodBroyden(Jw, it_max)

GB_mpc = GoodBroyden(mpc, it_max)

    # Passo 1
####    d = - (fatoracao \ F_tau(w, 0)) #vcat(mpc.Δ.x, mpc.Δ.y, dz) # Espaço para otimização 
###    d = - (J_temp \ F_tau(w, 0)) #vcat(mpc.Δ.x, mpc.Δ.y, dz) # Espaço para otimização 
###    mpc.Δ.x = d[1:n] # TODO: Trocar por view
###    mpc.Δ.y = d[n+1:n+m] # TODO: Trocar por view

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
#    println("⏰ Tempo para encontrar alpha para o passo de Newton:")
#    @time begin
#    dx = view(d, 1:n) # Sem cópias
#    ds = view(d, n+m+1:2*n+m) # Sem cópias
#    pctg = 0.99 # TODO: Usar o StepDampFactor do próprio Tulip (padrão = 0.9995). Além disso, não embutir a porcentagem no alpha, ou remover o StepDampFactor na atualização do ponto no código do Tulip.
#    alpha = 1.0
#    for i=1:n
#        if dx[i] < 0
#          alpha = min(-pctg*(x[i] / dx[i]), 1.0, alpha)
#        end
#        if ds[i] < 0
#          alpha = min(-pctg*(s[i] / ds[i]), 1.0, alpha)
#        end
#    end
#    alpha = max(alpha, 0) # calcula alpha máximo tal que algum xi ou si zera e depois toma uma fração desse passo, ou passo 1 no caso em que nenhuma variável bloqueia o passo.
#    mpc.αp, mpc.αd = alpha, alpha
#    alpha0 = alpha
#  end

alpha = GB_mpc.qnc.αp * params.StepDampFactor # Pressupõe αp = αd

  println("")
    # Passo 3
#    println("⏰ Tempo gasto para encontrar sigma para o passo corretor:")
    # sig = 0.5*sig_max
    sig = min(sig_max, 1.0 - alpha) # OBS: contas recentes (2025) mostram que escolher sigma igual à 1 - alpha é mais interessante
    println("")
    # Passo 4
#    mu_wk = dot(x, s)/n
    cp_x, cp_y, cp_xl, cp_xu, cp_zl, cp_zu, cp_mu = copy(pt.x), copy(pt.y), copy(pt.xl), copy(pt.xu), copy(pt.zl), copy(pt.zu), copy(pt.μ) # Fazendo cópia do iterando
#    w_n = zeros(2*n+m)
    t = 1

    while true
        mpc.n_tent_broyden += 1
        println("Testagem: ", t)
        println("Alfa = ", alpha)
        println("Sigma = ", sig)

#        w_n .= w + alpha*d
      # Comentei esse teste, pois não parece essencial. Mas, se precisar, eu o adapto para o novo formato.
#        for i=1:2*n+m # Teoricamente é garantido que x e s são positivos, mas erros numéricos acontecem
#            w_n[i] = w[i] + alpha*d[i]
#            if 1 <= i <= n || n+m+1 <= i <= 2*n+m
#
#                            if minimum(w_n[i]) < 0
#                error("A estratégia para calcular alpha não funcionou: w_n[$(i)] = $(w_n[i]);\n $(w[i]) + $(alpha) * $(d[i])")
#            end
#
#
#              w_n[i] = abs(w_n[i])
#              if w_n[i] == 0
#                error("Talvez precisemos forçar w_n[i] > eps^2")
#              end
#            end
#        end

    # Anda na direção preditora, com o tamanho de passo especificado
    pt.x  .+= alpha .* Δ.x
    pt.xl .+= alpha .* Δ.xl
    pt.xu .+= alpha .* Δ.xu
    pt.y  .+= alpha .* Δ.y
    pt.zl .+= alpha .* Δ.zl
    pt.zu .+= alpha .* Δ.zu

#        println("Ponto inicial do Broyden")
#        display(w_n)

        compute_residuals!(GB_mpc.qnc) # Atualiza os resíduos
        update_mu!(pt) # Atualiza mu só pra printar ele
        println("mu (após newton) = ", pt.μ)
        pt.μ = cp_mu # Retorna pro valor original

#        b_status = Broyden(F_tau, Jw, w_n, mu_wk, sig, sig_max, m, n, eps, it_max)
        b_status = Broyden2!(GB_mpc, alpha, sig, cp_mu, it_max, eps, cp_x, cp_xl, cp_xu, cp_y, cp_zl, cp_zu)
#        Jw.size = 0 # Resetar a estrutura GoodBroyden para a próxima iteração

#        println("mu (final do broyden)=", dot(w_n[1:n], w_n[n+m+1:2n+m])/n)
#        println("Status (Broyden) = ", b_status)
        if b_status == true # Se encontrar um ponto em F_0 com decrescimo de mu, pare.
            break
        end
        alpha *= 0.5
#        mpc.αp, mpc.αd = alpha, alpha # Acho que deveria atualizar esse também
#        sig = 0.5*(sig_max + sig)
        sig = min(sig_max, 1.0 - alpha) # OBS: contas recentes (2025) mostram que escolher sigma igual à 1 - alpha é mais interessante
        #    if abs(sig_max - sig) < 1.0e-8 # Parar se sig se aproximar muito de sig_max (e depois acusar erro: sig_max não é grande suficiente ou o ponto inicial tomado não está próximo o suficiente do caminho central)

        # Descarta os deslocamentos feitos durante o método de Broyden e retorna mu para seu valor original

        pt.x  .= cp_x
        pt.xl .= cp_xl
        pt.xu .= cp_xu
        pt.y  .= cp_y
        pt.zl .= cp_zl
        pt.zu .= cp_zu
        pt.μ = cp_mu
        
###### AQUI COMEÇA O MÉTODO ALTERNATIVO

        if t == 100
            error("É necessário utilizar o método alternativo, mas por enquanto ele está desligado.")
        end

        if t == -1 # 3 # 30 iterações é suficiente para praticamente zerar a diferença entre sig_max e sig (ela fica na ordem de 4.66e-10)
            #            error("Não foi possível determinar alpha e sigma de modo a obter a convergência do passo corretor. Tente outro ponto inicial que esteja mais próximo do caminho central ou tome sig_max ainda mais próximo de 1.")
            println("AVISO: Não foi possível determinar alpha e sigma de modo a obter a convergência do passo corretor. Isso pode ter ocorrido pois o ponto inicial não estava próximo o suficiente do caminho central. Para contornar isso, será aplicado um método quasi-newton alternativo.")
#            error("AVISO: Não foi possível determinar alpha e sigma de modo a obter a convergência do passo corretor. Isso pode ter ocorrido pois o ponto inicial não estava próximo o suficiente do caminho central. Para contornar isso, será aplicado um método quasi-newton alternativo.")
            mpc.n_corr_alt += 1
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
                mpc.nitb += 1
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
                Jw.qnc = Jw.qnc.L*Jw.qnc.U # reconstroi Jw
                Jw.qnc = Jw.qnc + ((yk - Jw.qnc*sk)*(sk'))/(dot(sk,sk))
        end
            break

        end

        t += 1
    end
    # Passo 5 
    #    println("w após broyden: ", w_n)
#    mpc.pt.x = w_n[1:n]
    
    # atualizando xl e xu (para o tulip)
    
#    mpc.pt.xl .= mpc.pt.x - mpc.dat.l
#    mpc.pt.xu .= mpc.dat.u - mpc.pt.x
    
    # ---
    
#    mpc.pt.y = w_n[n+1:n+m]
#    mpc.pt.z .= w_n[n+m+1:2*n+m]


    
    # atualizando zl e zu (para o tulip) [provisorio]
    
#    mpc.pt.zl .= copy(mpc.pt.z)
#    mpc.pt.xu .= 0
    
    # ---
    
#    w_n .= w_n - w
#    mpc.Δc.x .= w_n[1:n]
#    mpc.Δc.y .= w_n[n+1:n+m] # talvez precise atualizar mais coisas em delta_c

    # Calcula a direção resultante após o passo preditor e as iterações do método de Broyden
    Δc.x =  pt.x - cp_x
    Δc.y =  pt.y - cp_y
    Δc.xl = pt.xl - cp_xl
    Δc.xu = pt.xu - cp_xu
    Δc.zl = pt.zl - cp_zl
    Δc.zu = pt.zu - cp_zu

    # Retorna o ponto para sua posição inicial.
        pt.x  .= cp_x
        pt.xl .= cp_xl
        pt.xu .= cp_xu
        pt.y  .= cp_y
        pt.zl .= cp_zl
        pt.zu .= cp_zu

    if t == 30
        return true
    else
        return false
    end
end
