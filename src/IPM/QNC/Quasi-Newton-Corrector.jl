# TODO: Usar alphas separados para x e z;
# TODO: Evitar cópias desnecessárias dos pontos;
# TODO: Tentar não modificar as partes presumidamente imutáveis da estrutura QNC pelo Tulip original;
# TODO: Abrir as contas do método de Broyden para evitar concatenar e desconcatenar vetores;
# TODO: Armazenar as cópias que precisam ser feitas dentro da estrutura GoodBroyden;


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
LinearAlgebra.ldiv!(A::GoodBroyden, cp_x, cp_xl, cp_xu, cp_y, cp_zl, cp_zu) = begin # WARNING: Essa função começa a consumir mais memória com o passar das iterações. Esse aumento é bem lento, e mesmo em um problema grande (QAP12) o uso de memória foi bem razoável para 100 iterações, então creio que não vai ser um problema.

  # Resolve o caso base

  pt = A.qnc.pt
  m  = pt.m
  n  = pt.n

  # Copiar o iterando atual de Broyden

  cp_x_b   = copy(pt.x)
  cp_xl_b  = copy(pt.xl)
  cp_xu_b  = copy(pt.xu)
  cp_y_b   = copy(pt.y)
  cp_zl_b  = copy(pt.zl)
  cp_zu_b  = copy(pt.zu)

  # Recuperar a jacobiana original (na função solve_newton_system!, a jacobiana é calculada no ponto atual guardado em qnc. Isto significa que não estaríamos usando a mesma jacobiana do passo preditor. Portanto, ao retornar ao ponto anterior ao passo preditor, estamos garantindo que a mesma jacobiana do passo preditor será utilizada como B_0 pelo método de Broyden)

  pt.x  .= cp_x 
  pt.xl .= cp_xl
  pt.xu .= cp_xu
  pt.y  .= cp_y 
  pt.zl .= cp_zl
  pt.zu .= cp_zu

  # O comando abaixo pressupõe que o lado direito correto já está armazenado em qnc. Também pressupõe que o ponto atual armazenado em qnc seja o ponto antes do passo preditor (pois caso contrário, B_0 não seria a jacobiana utilizada no passo preditor).
  qnc = A.qnc
  solve_newton_system!(qnc.Δc, qnc, qnc.ξp, qnc.ξl, qnc.ξu, qnc.ξd, qnc.ξxzl, qnc.ξxzu)

  pt.x  .=  cp_x_b 
  pt.xl .= cp_xl_b
  pt.xu .= cp_xu_b
  pt.y  .=  cp_y_b 
  pt.zl .= cp_zl_b
  pt.zu .= cp_zu_b

  Δc = qnc.Δc

  b = vcat(Δc.x, Δc.xl, Δc.xu, Δc.y, Δc.zl, Δc.zu) # Constrói o lado direito

  mm   = A.size[]
  u    = nothing
  sb   = nothing
  rho  = nothing
  prod = nothing
  for i = 1:mm
    u   = A.u[i] # Sem intenção de fazer cópias
    sb  = A.sb[i] # Sem intenção de fazer cópias
    #     rho = A.rho[i]
    rho = 0
    for j=1:(5*n+m) # produto interno dot(sb, b)
      prod  = sb[j]
      prod *= b[j]
      rho  += prod
    end
    rho /= A.rho[i]

    #        rho = dot(sb, b) / rho
    #        b .= b + (dot(sb, b) / rho) * u
    #        b .= b + rho * u
    for j=1:(5*n+m)
      b[j] = b[j] + rho * u[j]
    end

  end

  Δc.x, Δc.xl, Δc.xu, Δc.y, Δc.zl, Δc.zu = deconcatenate(A.qnc, b)

  nothing

end

"""
Updates the Good Broyden approximation.
"""
function update!(B::GoodBroyden, s, b)

  u           = b

  size        = B.size + 1
  B.u[size]   = u
  B.rho[size] = 0

  for i=1:length(s)
    B.rho[size] += s[i]*(s[i] - B.u[size][i])
  end

  # Substituição para evitar erros numéricos envolvendo divisão por zero
  
  if -1.0e-8 <= B.rho[size] <= 0
    B.rho[size] = -1.0e-8
  elseif 0 <= B.rho[size] <= 1.0e-8 
    B.rho[size] = 1.0e-8
  end

  B.sb[size] = s
  B.size     = size

end

function positivity_test(qnc)

  pt = qnc.pt
  n  = qnc.pt.n

  # Teste de positividade de xl, zl, xu, zu sem gambiarra pra compatibilizar com o Tulip
  
  for i=1:n
    if (pt.xl[i] < 0) || (pt.zl[i] < 0) || (pt.xu[i] < 0) || (pt.zu[i] < 0)
      return false
    end
  end

  return true

end

function decrease_and_feasibility_test(qnc, cp_mu, sig)

  pt = qnc.pt
  update_mu!(pt)
  if (pt.μ <= 0.5 * (1.0 + sig) * cp_mu) && positivity_test(qnc)
    pt.μ = cp_mu
    return true
  end
  pt.μ = cp_mu
  return false

end

function Broyden_convergence_test(qnc, eps = 1.0e-8)

  for v in (qnc.ξp, qnc.ξl, qnc.ξu, qnc.ξd, qnc.ξxzl, qnc.ξxzu) # soh faz sentido se eu recalcular os residuos antes de rodar essa funcao
    if norm(v) > eps
      return false
    end
  end
  return true

end

function Broyden_parada(qnc, cp_mu, it, it_max, eps, sig)

  convergence  = false
  accept_point = false
  stop         = false
  if Broyden_convergence_test(qnc, eps)
    convergence = true
  end
  if decrease_and_feasibility_test(qnc, cp_mu, sig)
    accept_point = true
    stop         = true
  end
  if convergence || it >= it_max
    stop = true
  end

  return stop, convergence, accept_point

end 

function deconcatenate(qnc, b)

  m    = qnc.pt.m
  n    = qnc.pt.n
  rp   = b[1:n]
  rl   = b[n+1:2*n]
  ru   = b[2*n+1:3*n] 
  rd   = b[3*n+1:3*n+m] 
  rxzl = b[3*n+m+1:4*n+m] 
  rxzu = b[4*n+m+1:5*n+m] 

  return rp, rl, ru, rd, rxzl, rxzu

end

function Broyden!(GB_struct, mult, sig, cp_mu, it_max, eps, cp_x, cp_xl, cp_xu, cp_y, cp_zl, cp_zu, params, b_alt = false)

  qnc = GB_struct.qnc
  dat = qnc.dat
  pt  = qnc.pt
  Δ   = qnc.Δ
  Δc  = qnc.Δc

  ### 1ª iteração de Broyden

  it = 1

  # Anda na direção preditora, com o tamanho de passo especificado
  pt.x  .+= (mult * params.StepDampFactor * qnc.αp) .* Δ.x
  pt.xl .+= (mult * params.StepDampFactor * qnc.αp) .* Δ.xl
  pt.xu .+= (mult * params.StepDampFactor * qnc.αp) .* Δ.xu
  pt.y  .+= (mult * params.StepDampFactor * qnc.αd) .* Δ.y
  pt.zl .+= (mult * params.StepDampFactor * qnc.αd) .* Δ.zl
  pt.zu .+= (mult * params.StepDampFactor * qnc.αd) .* Δ.zu

  # Calcula os resíduos no ponto atual (parte do lado direito em solve_newton_system!)

  compute_residuals!(qnc)
  copyto!(qnc.ξp, qnc.res.rp)
  copyto!(qnc.ξl, qnc.res.rl)
  copyto!(qnc.ξu, qnc.res.ru)
  copyto!(qnc.ξd, qnc.res.rd)
  pt.μ = cp_mu

  @. qnc.ξxzl = (sig * pt.μ .- pt.xl .* pt.zl) .* dat.lflag
  @. qnc.ξxzu = (sig * pt.μ .- pt.xu .* pt.zu) .* dat.uflag

  ldiv!(GB_struct, cp_x, cp_xl, cp_xu, cp_y, cp_zl, cp_zu) # Pressupõe que os resíduos após o passo de Newton estejam guardados em qnc

  if b_alt # Se for o metodo alternativo, controla o tamanho do passo de Broyden.
    alpha_b_p, alpha_b_d = max_step_length_pd(qnc.pt, qnc.Δc)  
    Δc.x    .= (params.StepDampFactor * alpha_b_p) .* Δc.x 
    Δc.xl   .= (params.StepDampFactor * alpha_b_p) .* Δc.xl
    Δc.xu   .= (params.StepDampFactor * alpha_b_p) .* Δc.xu
    Δc.y    .= (params.StepDampFactor * alpha_b_d) .* Δc.y 
    Δc.zl   .= (params.StepDampFactor * alpha_b_d) .* Δc.zl
    Δc.zu   .= (params.StepDampFactor * alpha_b_d) .* Δc.zu
  end

  # Atualiza o ponto

  pt.x  .+=  Δc.x
  pt.xl .+=  Δc.xl
  pt.xu .+=  Δc.xu
  pt.y  .+=  Δc.y
  pt.zl .+=  Δc.zl
  pt.zu .+=  Δc.zu

  sb = vcat(Δc.x, Δc.xl, Δc.xu, Δc.y, Δc.zl, Δc.zu)

  # Calcula -F_{sigma * mu} (w + d_b) para depois calcular u

  compute_residuals!(qnc)
  copyto!(qnc.ξp, qnc.res.rp)
  copyto!(qnc.ξl, qnc.res.rl)
  copyto!(qnc.ξu, qnc.res.ru)
  copyto!(qnc.ξd, qnc.res.rd)
  pt.μ = cp_mu

  @. qnc.ξxzl = (sig * pt.μ .- pt.xl .* pt.zl) .* dat.lflag
  @. qnc.ξxzu = (sig * pt.μ .- pt.xu .* pt.zu) .* dat.uflag

  ldiv!(GB_struct, cp_x, cp_xl, cp_xu, cp_y, cp_zl, cp_zu) # Pressupõe que os resíduos após o passo de Newton estejam guardados em qnc
  u = vcat(Δc.x, Δc.xl, Δc.xu, Δc.y, Δc.zl, Δc.zu)

  update!(GB_struct, sb, u)

  it += 1

  ### Fim da 1ª iteração de Broyden

  ### Iterações posteriores...

  while true # Main loop

    # Stopping criteria

    stop, convergence, accept_point = Broyden_parada(qnc, cp_mu, it, it_max, eps, sig)
    if stop == true
      params.OutputLevel > 0 &&  println("Parou por que? stop / convergence / accept_point : ", (stop, convergence, accept_point))
      qnc.nitb += it # contabiliza as iterações de Broyden
      if b_alt
        qnc.n_corr_alt += it
      end
      return accept_point
    end

    sb  = (dot(GB_struct.sb[it-1], GB_struct.u[it-1]) / GB_struct.rho[it-1]) * GB_struct.u[it-1]
    sb .= sb .+ GB_struct.u[it-1]

    Δc.x, Δc.xl, Δc.xu, Δc.y, Δc.zl, Δc.zu = deconcatenate(GB_struct.qnc, sb)

    # Multiplicar a direção pelo tamanho do passo e armazenar tanto no qnc quanto na estrutura GoodBroyden

    if b_alt # Se for o metodo alternativo, controla o tamanho do passo de Broyden.
      alpha_b_p, alpha_b_d = max_step_length_pd(qnc.pt, qnc.Δc)  
      Δc.x  *= alpha_b_p * params.StepDampFactor  
      Δc.xl *= alpha_b_p * params.StepDampFactor     
      Δc.xu *= alpha_b_p * params.StepDampFactor 
      Δc.y  *= alpha_b_d * params.StepDampFactor 
      Δc.zl *= alpha_b_d * params.StepDampFactor 
      Δc.zu *= alpha_b_d * params.StepDampFactor 
      sb = vcat(Δc.x, Δc.xl, Δc.xu, Δc.y, Δc.zl, Δc.zu)
    end

    # Atualizar o ponto

    pt.x  .+=  Δc.x
    pt.xl .+=  Δc.xl
    pt.xu .+=  Δc.xu
    pt.y  .+=  Δc.y
    pt.zl .+=  Δc.zl
    pt.zu .+=  Δc.zu

    # Calcular novo u

    compute_residuals!(qnc)
    copyto!(qnc.ξp, qnc.res.rp)
    copyto!(qnc.ξl, qnc.res.rl)
    copyto!(qnc.ξu, qnc.res.ru)
    copyto!(qnc.ξd, qnc.res.rd)
    pt.μ = cp_mu

    @. qnc.ξxzl = (sig * pt.μ .- pt.xl .* pt.zl) .* dat.lflag
    @. qnc.ξxzu = (sig * pt.μ .- pt.xu .* pt.zu) .* dat.uflag

    ldiv!(GB_struct, cp_x, cp_xl, cp_xu, cp_y, cp_zl, cp_zu) # Pressupõe que os resíduos do iterando mais atual estejam guardados em qnc
    u = vcat(Δc.x, Δc.xl, Δc.xu, Δc.y, Δc.zl, Δc.zu)

    update!(GB_struct, sb, u)

    it += 1

  end

end

function Quasi_Newton_Corrector!(qnc::QNC, params, sig_max = 1-1.0e-4, eps=1.0e-8, it_max = 5) # it_max padrao eh 5

  # Names
  
  dat = qnc.dat
  pt  = qnc.pt
  res = qnc.res

  Δ  = qnc.Δ
  Δc = qnc.Δc

  m, n, p = pt.m, pt.n, pt.p

  A = dat.A
  b = dat.b
  c = dat.c

######### É bom levar em conta que o Tulip resolve um sistema envolvendo as seguintes condições KKT:
######### A^T \lambda + zl - zu  = c
#########                   A x  = b
#########           xl_i * zl_i  = \tau, i = 1, ..., n
#########           xu_i * zu_i  = \tau, i = 1, ..., n
#########        xl, xu, zl, zu >= 0
######### Então, eles estão em busca de um ponto com 5 entradas (que também são vetores): (x, xl, xu, zl, zu).
######### Também vou deixar aqui mais duas fórmulas para eu lembrar depois:
######### x - xl = l
######### x + xu = u

  GB_struct = GoodBroyden(qnc, it_max)
  qncGB     = GB_struct.qnc

  mult = 1.0
  alpha_m = 0.5*(GB_struct.qnc.αp + GB_struct.qnc.αd)
  sig = min(sig_max, 1.0 - alpha_m) # OBS: contas recentes (2025) mostram que escolher sigma igual à 1 - alpha é mais interessante. Como na prática estamos usando dois alphas, estou considerando a média dos dois.
  
  cp_x, cp_y, cp_xl, cp_xu, cp_zl, cp_zu, cp_mu = copy(pt.x), copy(pt.y), copy(pt.xl), copy(pt.xu), copy(pt.zl), copy(pt.zu), copy(pt.μ) # Fazendo cópia do iterando
  
  t = 1

  while true
    qnc.n_tent_broyden += 1
    params.OutputLevel > 0 && println("Testagem: ", t)
    params.OutputLevel > 0 && println("Alfa = ", alpha)
    params.OutputLevel > 0 && println("Sigma = ", sig)

    b_status       = Broyden!(GB_struct, mult, sig, cp_mu, it_max, eps, cp_x, cp_xl, cp_xu, cp_y, cp_zl, cp_zu, params) 
    GB_struct.size = 0 # Resetar a estrutura GoodBroyden para a próxima iteração

    params.OutputLevel > 0 && println("Status (Broyden) = ", b_status)
    if b_status == true # Se encontrar um ponto em F_0 com decrescimo de mu, pare.
      break
    end
    mult *= 0.5
    sig   = min(sig_max, 1.0 - mult*alpha_m) # OBS: contas recentes (2025) mostram que escolher sigma igual à 1 - alpha é mais interessante

    # Descarta os deslocamentos feitos durante o método de Broyden e retorna mu para seu valor original

    qncGB.pt.x  .= cp_x
    qncGB.pt.xl .= cp_xl
    qncGB.pt.xu .= cp_xu
    qncGB.pt.y  .= cp_y
    qncGB.pt.zl .= cp_zl
    qncGB.pt.zu .= cp_zu
    qncGB.pt.μ   = cp_mu

##### AQUI COMEÇA O MÉTODO ALTERNATIVO #####

    if t == 3 # 3 por padrao # 30 iterações é suficiente para praticamente zerar a diferença entre sig_max e sig (ela fica na ordem de 4.66e-10)

      println("WARNING: Não foi possível determinar alpha e sigma de modo a obter a convergência do passo corretor. Isso pode ter ocorrido pois o ponto inicial não estava próximo o suficiente do caminho central. Para contornar isso, será aplicado um método quasi-newton alternativo.")

      sig = min(sig_max, 1.0 - 0.5*(GB_struct.qnc.αp + GB_struct.qnc.αd)) # OBS: contas recentes (2025) mostram que escolher sigma igual à 1 - alpha é mais interessante. Como na prática estamos usando dois alphas, estou considerando a média dos dois.

      b_status = Broyden!(GB_struct, mult, sig, cp_mu, it_max, eps, cp_x, cp_xl, cp_xu, cp_y, cp_zl, cp_zu, params, true)

      if !(b_status)

        println("WARNING: O método alternativo falhou. Andando apenas na direção preditora, sem correções...")
        
        # Descarta os deslocamentos feitos durante o método de Broyden e retorna mu para seu valor original

        qncGB.pt.x  .= cp_x
        qncGB.pt.xl .= cp_xl
        qncGB.pt.xu .= cp_xu
        qncGB.pt.y  .= cp_y
        qncGB.pt.zl .= cp_zl
        qncGB.pt.zu .= cp_zu
        qncGB.pt.μ   = cp_mu

        # Anda apenas na direção preditora
        
        qncGB.pt.x  .+= (params.StepDampFactor * qncGB.αp) .* Δ.x
        qncGB.pt.xl .+= (params.StepDampFactor * qncGB.αp) .* Δ.xl
        qncGB.pt.xu .+= (params.StepDampFactor * qncGB.αp) .* Δ.xu
        qncGB.pt.y  .+= (params.StepDampFactor * qncGB.αd) .* Δ.y
        qncGB.pt.zl .+= (params.StepDampFactor * qncGB.αd) .* Δ.zl
        qncGB.pt.zu .+= (params.StepDampFactor * qncGB.αd) .* Δ.zu

      end

      break

    end

    t += 1
  end

  # Calcula a direção resultante após o passo preditor e as iterações do método de Broyden
   
  Δc.x  =  qncGB.pt.x - cp_x
  Δc.y  =  qncGB.pt.y - cp_y
  Δc.xl = qncGB.pt.xl - cp_xl
  Δc.xu = qncGB.pt.xu - cp_xu
  Δc.zl = qncGB.pt.zl - cp_zl
  Δc.zu = qncGB.pt.zu - cp_zu

  # Retorna o ponto para sua posição inicial.
   
  pt.x  .= cp_x
  pt.xl .= cp_xl
  pt.xu .= cp_xu
  pt.y  .= cp_y
  pt.zl .= cp_zl
  pt.zu .= cp_zu

end
