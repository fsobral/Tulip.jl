# TODO: Tentar não modificar as partes presumidamente imutáveis da estrutura QNC pelo Tulip original; (fiz isso, com exceção do ponto, que tenho que atualizar para calcular os resíduos, mas retorno ao valor inicial sempre através de uma cópia)
# TODO: Abrir as contas do método de Broyden para evitar concatenar e desconcatenar vetores;
# TODO: Armazenar as cópias que precisam ser feitas dentro da estrutura GoodBroyden; (eu fiz isso, mas guardei na estrutura qnc)


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
LinearAlgebra.ldiv!(A::GoodBroyden) = begin # WARNING: Essa função começa a consumir mais memória com o passar das iterações. Esse aumento é bem lento, e mesmo em um problema grande (QAP12) o uso de memória foi bem razoável para 100 iterações, então creio que não vai ser um problema.

  qnc = A.qnc
  cp_x, cp_xl, cp_xu, cp_y, cp_zl, cp_zu, cp_mu = qnc.pt_cp.x, qnc.pt_cp.xl, qnc.pt_cp.xu, qnc.pt_cp.y, qnc.pt_cp.zl, qnc.pt_cp.zu, qnc.pt_cp.μ # Nomes

  # Resolve o caso base

  pt = A.qnc.pt
  m  = pt.m
  n  = pt.n

  #  # Copiar o iterando atual de Broyden
  #
  #  cp_x_b   = copy(pt.x)
  #  cp_xl_b  = copy(pt.xl)
  #  cp_xu_b  = copy(pt.xu)
  #  cp_y_b   = copy(pt.y)
  #  cp_zl_b  = copy(pt.zl)
  #  cp_zu_b  = copy(pt.zu)

  #  # Recuperar a jacobiana original (na função solve_newton_system!, a jacobiana é calculada no ponto atual guardado em qnc. Isto significa que não estaríamos usando a mesma jacobiana do passo preditor. Portanto, ao retornar ao ponto anterior ao passo preditor, estamos garantindo que a mesma jacobiana do passo preditor será utilizada como B_0 pelo método de Broyden)
  #
  #  pt.x  .= cp_x 
  #  pt.xl .= cp_xl
  #  pt.xu .= cp_xu
  #  pt.y  .= cp_y 
  #  pt.zl .= cp_zl
  #  pt.zu .= cp_zu

  # O comando abaixo pressupõe que o lado direito correto já está armazenado em qnc. Também pressupõe que o ponto atual armazenado em qnc seja o ponto antes do passo preditor (pois caso contrário, B_0 não seria a jacobiana utilizada no passo preditor).
  solve_newton_system!(qnc.Δc, qnc, qnc.ξp, qnc.ξl, qnc.ξu, qnc.ξd, qnc.ξxzl, qnc.ξxzu)

  #  pt.x  .=  cp_x_b 
  #  pt.xl .= cp_xl_b
  #  pt.xu .= cp_xu_b
  #  pt.y  .=  cp_y_b 
  #  pt.zl .= cp_zl_b
  #  pt.zu .= cp_zu_b

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
    #    rho = 0
    #    for j=1:(5*n+m) # produto interno dot(sb, b)
    #      prod  = sb[j]
    #      prod *= b[j]
    #      rho  += prod
    #    end
    rho = dot(sb, b)
    rho /= A.rho[i]

    #        rho = dot(sb, b) / rho
    #        b .= b + (dot(sb, b) / rho) * u
    #        b .= b + rho * u
    #    for j=1:(5*n+m)
    #      b[j] = b[j] + rho * u[j]
    #    end
    @. b += rho * u

  end

  deconcatenate(A.qnc, b, Δc.x, Δc.xl, Δc.xu, Δc.y, Δc.zl, Δc.zu)

  nothing

end

"""
Updates the Good Broyden approximation.
"""
function update!(B::GoodBroyden)

  #  u           = b

  size        = B.size + 1
  #  B.u[size]   = u

  #  B.rho[size] = 0
  #
  #  for i=1:length(s)
  #    B.rho[size] += s[i]*(s[i] - B.u[size][i])
  #  end

  s = B.sb[size]

  B.rho[size] = dot(s, s .- B.u[size])

  # Substituição para evitar erros numéricos envolvendo divisão por zero

  if -1.0e-8 <= B.rho[size] <= 0
    B.rho[size] = -1.0e-8
  elseif 0 <= B.rho[size] <= 1.0e-8 
    B.rho[size] = 1.0e-8
  end

  #B.sb[size] = s
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

  cp_mu = qnc.pt_cp.μ
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

function Broyden_parada(GB_struct, it, it_max, eps, sig, mult, params)

  qnc = GB_struct.qnc
  dat = qnc.dat
  pt  = qnc.pt
  pt_cp  = qnc.pt_cp
  Δ   = qnc.Δ
  Δc  = qnc.Δc

  cp_x, cp_xl, cp_xu, cp_y, cp_zl, cp_zu, cp_mu = qnc.pt_cp.x, qnc.pt_cp.xl, qnc.pt_cp.xu, qnc.pt_cp.y, qnc.pt_cp.zl, qnc.pt_cp.zu, qnc.pt_cp.μ # Nomes

  calculate_resulting_step!(GB_struct, mult, params)
  # Neste momento Δc possui todos os passos corretores até o momento incluindo o passo previsor (ja reduzido por mult).

  # Atualizar o ponto apenas para testar o criterio de parada

  @. pt.x  +=  Δc.x
  @. pt.xl +=  Δc.xl
  @. pt.xu +=  Δc.xu
  @. pt.y  +=  Δc.y
  @. pt.zl +=  Δc.zl
  @. pt.zu +=  Δc.zu

  cp_mu = qnc.pt_cp.μ

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

  # Retorna o iterando para seu valor original

  @. pt.x  = pt_cp.x 
  @. pt.xl = pt_cp.xl
  @. pt.xu = pt_cp.xu
  @. pt.y  = pt_cp.y 
  @. pt.zl = pt_cp.zl
  @. pt.zu = pt_cp.zu

  return stop, convergence, accept_point

end 

function deconcatenate(qnc, v, x, xl, xu, y, zl, zu)

  m   = qnc.pt.m
  n   = qnc.pt.n
  x  .= v[1:n]
  xl .= v[n+1:2*n]
  xu .= v[2*n+1:3*n] 
  y  .= v[3*n+1:3*n+m] 
  zl .= v[3*n+m+1:4*n+m] 
  zu .= v[4*n+m+1:5*n+m] 

end

function concatenate(qnc, v, x, xl, xu, y, zl, zu)

  m    = qnc.pt.m
  n    = qnc.pt.n
  v[1:n]           .= x
  v[n+1:2*n]       .= xl
  v[2*n+1:3*n]     .= xu
  v[3*n+1:3*n+m]   .= y
  v[3*n+m+1:4*n+m] .= zl
  v[4*n+m+1:5*n+m] .= zu 

end

"
Calcula a soma de todos os passos (afim escala e correções de Broyden) registrados até o momento em Δ e GB_struct.sb.

O passo afim escala é somado considerando o tamanho de passo máximo, controlado pelo multiplicador 'mult'.

Os passos de Broyden são dados de forma completa quando o algoritmo principal é utilizado. No caso do método alternativo, os vetores GB_struct.sb[i] já estão multiplicados de seus respectivos tamanhos de passo.

Por fim, o passo que resulta desta grande soma é finalmente guardado em Δc, sobreescrevendo qualquer informação salva nesta estrutura.

"
function calculate_resulting_step!(GB_struct, mult, params)

  qnc = GB_struct.qnc
  Δ   = qnc.Δ
  Δc  = qnc.Δc
  m   = qnc.pt.m
  n   = qnc.pt.n

  # Calcula a direção resultante do passo de Newton junto com os passos de Broyden

  # Guarda o passo de Newton em Δc. Isto sobreescreve qualquer valor previamente salvo neste "point".

  @. Δc.x  = (mult * params.StepDampFactor * qnc.αp) * Δ.x 
  @. Δc.xl = (mult * params.StepDampFactor * qnc.αp) * Δ.xl
  @. Δc.xu = (mult * params.StepDampFactor * qnc.αp) * Δ.xu
  @. Δc.y  = (mult * params.StepDampFactor * qnc.αd) * Δ.y 
  @. Δc.zl = (mult * params.StepDampFactor * qnc.αd) * Δ.zl
  @. Δc.zu = (mult * params.StepDampFactor * qnc.αd) * Δ.zu

  # Soma todas as direções de Broyden calculadas até o momento. OBS: A direção atual só estará inclusa aqui se a função update! já tiver sido executada.
  for i=1:GB_struct.size 
    @. Δc.x  += GB_struct.sb[i][1       : n]
    @. Δc.xl += GB_struct.sb[i][n+1     : 2*n]
    @. Δc.xu += GB_struct.sb[i][2*n+1   : 3*n]
    @. Δc.y  += GB_struct.sb[i][3*n+1   : 3*n+m]
    @. Δc.zl += GB_struct.sb[i][3*n+m+1 : 4*n+m]
    @. Δc.zu += GB_struct.sb[i][4*n+m+1 : 5*n+m]
  end

end

"
Corrige a direção de Broyden de acordo com o tamanho máximo de passo segundo max_step_length_pd. A direção corrigida é armazenada tanto em Δc quanto em sua forma concatenada sb para a utilização posterior.

"
function Broyden_alternative_step!(GB_struct, mult, params, sb)

  qnc = GB_struct.qnc
  dat = qnc.dat
  pt  = qnc.pt
  pt_cp  = qnc.pt_cp
  Δ   = qnc.Δ
  Δc  = qnc.Δc

  cp_x, cp_xl, cp_xu, cp_y, cp_zl, cp_zu, cp_mu = qnc.pt_cp.x, qnc.pt_cp.xl, qnc.pt_cp.xu, qnc.pt_cp.y, qnc.pt_cp.zl, qnc.pt_cp.zu, qnc.pt_cp.μ # Nomes

  # Calcula a soma de todos os passos desde o afim escala e guarda em Δc.

  calculate_resulting_step!(GB_struct, mult, params)

  # Recupera o iterando atual para o método alternativo

  @. pt.x  += Δc.x
  @. pt.xl += Δc.xl
  @. pt.xu += Δc.xu
  @. pt.y  += Δc.y
  @. pt.zl += Δc.zl
  @. pt.zu += Δc.zu

  # Recupera a direção nova (isso é necessário, pois Δc foi sobreescrito após a função calculate_resulting_step! ser executada)

  deconcatenate(qnc, sb, Δc.x, Δc.xl, Δc.xu, Δc.y, Δc.zl, Δc.zu)

  # Calcula o tamanho de passo na direção nova

  alpha_b_p, alpha_b_d = max_step_length_pd(qnc.pt, qnc.Δc) 

  # Atualiza a direção nova com o tamanho de passo calculado

  @. Δc.x    = (params.StepDampFactor * alpha_b_p) * Δc.x 
  @. Δc.xl   = (params.StepDampFactor * alpha_b_p) * Δc.xl
  @. Δc.xu   = (params.StepDampFactor * alpha_b_p) * Δc.xu
  @. Δc.y    = (params.StepDampFactor * alpha_b_d) * Δc.y 
  @. Δc.zl   = (params.StepDampFactor * alpha_b_d) * Δc.zl
  @. Δc.zu   = (params.StepDampFactor * alpha_b_d) * Δc.zu

  # Guarda a nova direção em sb

  concatenate(qnc, sb, Δc.x, Δc.xl, Δc.xu, Δc.y, Δc.zl, Δc.zu)

  # Retorna o iterando para seu valor original

  @. pt.x  = pt_cp.x 
  @. pt.xl = pt_cp.xl
  @. pt.xu = pt_cp.xu
  @. pt.y  = pt_cp.y 
  @. pt.zl = pt_cp.zl
  @. pt.zu = pt_cp.zu

end

"
Calcula os resíduos para a construção do lado direito de um sistema para ser resolvido pela função solve_newton_system!. Esses resíduos serão guardados em GB_struct.qnc, substituindo quaisquer valores anteriormente salvos.

"
function calculate_broyden_residuals!(GB_struct, cp_mu, sig)

  qnc = GB_struct.qnc
  dat = qnc.dat
  pt  = qnc.pt
  pt_cp  = qnc.pt_cp
  Δ   = qnc.Δ
  Δc  = qnc.Δc

  cp_x, cp_xl, cp_xu, cp_y, cp_zl, cp_zu, cp_mu = qnc.pt_cp.x, qnc.pt_cp.xl, qnc.pt_cp.xu, qnc.pt_cp.y, qnc.pt_cp.zl, qnc.pt_cp.zu, qnc.pt_cp.μ # Nomes

  # Calcula os resíduos no ponto atual (parte do lado direito em solve_newton_system!)

  compute_residuals!(qnc)
  copyto!(qnc.ξp, qnc.res.rp)
  copyto!(qnc.ξl, qnc.res.rl)
  copyto!(qnc.ξu, qnc.res.ru)
  copyto!(qnc.ξd, qnc.res.rd)
  pt.μ = cp_mu

  # Adiciona regularizacao referente aos pontos de referencia (ver GS, 2019 ou G, 2012)
  @. qnc.ξp += qnc.regD * (pt.y - cp_y)
  @. qnc.ξd -= qnc.regP * (pt.x - cp_x)
  # Adiciona os termos nao lineares
  @. qnc.ξxzl = (sig * pt.μ - pt.xl * pt.zl) * dat.lflag
  @. qnc.ξxzu = (sig * pt.μ - pt.xu * pt.zu) * dat.uflag

end

"
Calcula a primeira direção de Broyden e guarda em Δc. Esta função sobreescreve Δc, GB_struct.qnc.res e os ξ's.
"
function calculate_first_broyden_step!(GB_struct, mult, params, cp_mu, sig)

  qnc   = GB_struct.qnc
  dat   = qnc.dat
  pt    = qnc.pt
  pt_cp = qnc.pt_cp
  Δ     = qnc.Δ
  Δc    = qnc.Δc

  cp_x, cp_xl, cp_xu, cp_y, cp_zl, cp_zu, cp_mu = qnc.pt_cp.x, qnc.pt_cp.xl, qnc.pt_cp.xu, qnc.pt_cp.y, qnc.pt_cp.zl, qnc.pt_cp.zu, qnc.pt_cp.μ # Nomes

  # Anda na direção preditora, com o tamanho de passo especificado, apenas para calcular os resíduos

  calculate_resulting_step!(GB_struct, mult, params)

  # Aplica afim escala (ja com mult aplicado)
  @. pt.x  += Δc.x
  @. pt.xl += Δc.xl
  @. pt.xu += Δc.xu
  @. pt.y  += Δc.y
  @. pt.zl += Δc.zl
  @. pt.zu += Δc.zu

  calculate_broyden_residuals!(GB_struct, cp_mu, sig)

  # Retorna o iterando para seu valor original

  @. pt.x  = pt_cp.x 
  @. pt.xl = pt_cp.xl
  @. pt.xu = pt_cp.xu
  @. pt.y  = pt_cp.y 
  @. pt.zl = pt_cp.zl
  @. pt.zu = pt_cp.zu

  # Calcula a direção completa de Broyden e guarda em Δc.
  ldiv!(GB_struct) # Pressupõe que os resíduos após o passo de Newton estejam guardados em qnc

end

"
Calcula o novo vetor u, armazenando-o em GB_struct. AVISO: Durante o processo, Δc será sobreescrito. 

"
function calculate_broyden_u!(GB_struct, mult, params, sb, cp_mu, sig)

  qnc   = GB_struct.qnc
  dat   = qnc.dat
  pt    = qnc.pt
  pt_cp = qnc.pt_cp
  Δ     = qnc.Δ
  Δc    = qnc.Δc

  cp_x, cp_xl, cp_xu, cp_y, cp_zl, cp_zu, cp_mu = qnc.pt_cp.x, qnc.pt_cp.xl, qnc.pt_cp.xu, qnc.pt_cp.y, qnc.pt_cp.zl, qnc.pt_cp.zu, qnc.pt_cp.μ # Nomes

  m = qnc.pt.m
  n = qnc.pt.n


  calculate_resulting_step!(GB_struct, mult, params)

  # Atualizar o ponto apenas para calcular os resíduos

  @. pt.x  +=  Δc.x  + sb[1       : n]
  @. pt.xl +=  Δc.xl + sb[n+1     : 2*n]
  @. pt.xu +=  Δc.xu + sb[2*n+1   : 3*n]
  @. pt.y  +=  Δc.y  + sb[3*n+1   : 3*n+m]
  @. pt.zl +=  Δc.zl + sb[3*n+m+1 : 4*n+m]
  @. pt.zu +=  Δc.zu + sb[4*n+m+1 : 5*n+m]

  # Calcular novo u

  calculate_broyden_residuals!(GB_struct, cp_mu, sig)

  # Retorna o iterando para seu valor original

  @. pt.x  = pt_cp.x 
  @. pt.xl = pt_cp.xl
  @. pt.xu = pt_cp.xu
  @. pt.y  = pt_cp.y 
  @. pt.zl = pt_cp.zl
  @. pt.zu = pt_cp.zu

  # Cria o vetor u

  GB_struct.u[GB_struct.size + 1] = spzeros(5*n+m) 
  u = GB_struct.u[GB_struct.size + 1]

  ldiv!(GB_struct) # Pressupõe que os resíduos do iterando mais atual estejam guardados em qnc
  concatenate(qnc, u, Δc.x, Δc.xl, Δc.xu, Δc.y, Δc.zl, Δc.zu)

end

"
Esta função realiza o cálculo da direção de Broyden atual (passo cheio), e a armazena tanto em Δc quanto em sb para uso posterior. No caso do método alternativo, são guardados os passos já ajustados de acordo com o tamanho de passo dado por max_step_length_pd. 

"
function calculate_broyden_sb!(it, GB_struct, mult, params, cp_mu, sig, sb, b_alt)

  # Apelidos iniciais

  qnc     = GB_struct.qnc
  dat     = qnc.dat
  pt      = qnc.pt
  pt_cp   = qnc.pt_cp
  Δ       = qnc.Δ
  Δc      = qnc.Δc
  gb_size = GB_struct.size

  cp_x, cp_xl, cp_xu, cp_y, cp_zl, cp_zu, cp_mu = qnc.pt_cp.x, qnc.pt_cp.xl, qnc.pt_cp.xu, qnc.pt_cp.y, qnc.pt_cp.zl, qnc.pt_cp.zu, qnc.pt_cp.μ # Nomes

  m    = qnc.pt.m
  n    = qnc.pt.n

  if it == 1 # Na primeira iteração é diferente
    # Calcula a primeira direção de Broyden e guarda em Δc.
    calculate_first_broyden_step!(GB_struct, mult, params, cp_mu, sig)


    concatenate(qnc, sb, Δc.x, Δc.xl, Δc.xu, Δc.y, Δc.zl, Δc.zu) # preenche o sb com as entradas de Δc

  else

    @. sb = (dot(GB_struct.sb[gb_size], GB_struct.u[gb_size]) / GB_struct.rho[gb_size]) * GB_struct.u[gb_size]
    @. sb += GB_struct.u[gb_size]

    # !!! TODO Verificar a possível remoção disso!!
    deconcatenate(GB_struct.qnc, sb, Δc.x, Δc.xl, Δc.xu, Δc.y, Δc.zl, Δc.zu)
  end

  # Apenas para o método alternativo: controla o tamanho do passo de Broyden.
  if b_alt 
    Broyden_alternative_step!(GB_struct, mult, params, sb)
  end



end

function Broyden!(GB_struct, mult, sig, it_max, eps, params, b_alt = false)

  # Apelidos iniciais

  qnc   = GB_struct.qnc
  dat   = qnc.dat
  pt    = qnc.pt
  pt_cp = qnc.pt_cp
  Δ     = qnc.Δ
  Δc    = qnc.Δc

  cp_x, cp_xl, cp_xu, cp_y, cp_zl, cp_zu, cp_mu = qnc.pt_cp.x, qnc.pt_cp.xl, qnc.pt_cp.xu, qnc.pt_cp.y, qnc.pt_cp.zl, qnc.pt_cp.zu, qnc.pt_cp.μ # Nomes

  m    = qnc.pt.m
  n    = qnc.pt.n

  # Laço principal

  for it=1:it_max

    # 1 ETAPA: Calcula sb

    gb_size = GB_struct.size

    # Cria o vetor sb dentro de GB_struct e guarda a direção nele
    GB_struct.sb[GB_struct.size + 1] = spzeros(5*n+m) 
    sb = GB_struct.sb[GB_struct.size + 1]

    calculate_broyden_sb!(it, GB_struct, mult, params, cp_mu, sig, sb, b_alt)

    # 2 ETAPA: Calcula u

    # Calcula o vetor u e guarda na estrutura GB_struct
    calculate_broyden_u!(GB_struct, mult, params, sb, cp_mu, sig)

    # 3 ETAPA: Atualiza a estrutura Good Broyden (calcula rho e aumenta GB_struct.size)

    update!(GB_struct)

    # 4 ETAPA: Critério de parada

    # Stopping criteria

    stop, convergence, accept_point = Broyden_parada(GB_struct, it, it_max, eps, sig, mult, params)

    if stop == true
      params.OutputLevel > 0 &&  println("Parou por que? stop / convergence / accept_point : ", (stop, convergence, accept_point))
      qnc.nitb += it # contabiliza as iterações de Broyden
      if b_alt
        qnc.n_corr_alt += it
      end

      return accept_point
    end

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

  cp_x, cp_xl, cp_xu, cp_y, cp_zl, cp_zu, cp_mu = qncGB.pt_cp.x, qncGB.pt_cp.xl, qncGB.pt_cp.xu, qncGB.pt_cp.y, qncGB.pt_cp.zl, qncGB.pt_cp.zu, qncGB.pt_cp.μ # Nomes
  copyto!(cp_x, pt.x)
  copyto!(cp_xl, pt.xl)
  copyto!(cp_xu, pt.xu)
  copyto!(cp_y, pt.y)
  copyto!(cp_zl, pt.zl)
  copyto!(cp_zu, pt.zu)
  qncGB.pt_cp.μ = pt.μ
  #  cp_x, cp_y, cp_xl, cp_xu, cp_zl, cp_zu, cp_mu = copy(pt.x), copy(pt.y), copy(pt.xl), copy(pt.xu), copy(pt.zl), copy(pt.zu), copy(pt.μ) # Fazendo cópia do iterando

  t = 1

  while true
    qnc.n_tent_broyden += 1
    params.OutputLevel > 0 && println("Testagem: ", t)
    params.OutputLevel > 0 && println("Alfa (médio) = ", alpha_m)
    params.OutputLevel > 0 && println("Sigma = ", sig)

    b_status       = Broyden!(GB_struct, mult, sig, it_max, eps, params) 
    GB_struct.size = 0 # Resetar a estrutura GoodBroyden para a próxima iteração

    params.OutputLevel > 0 && println("Status (Broyden) = ", b_status)
    if b_status == true # Se encontrar um ponto em F_0 com decrescimo de mu, pare.
      break
    end
    mult *= 0.5
    sig   = min(sig_max, 1.0 - mult*alpha_m) # OBS: contas recentes (2025) mostram que escolher sigma igual à 1 - alpha é mais interessante

    # Descarta os deslocamentos feitos durante o método de Broyden e retorna mu para seu valor original

    @. qncGB.pt.x  = cp_x
    @. qncGB.pt.xl = cp_xl
    @. qncGB.pt.xu = cp_xu
    @. qncGB.pt.y  = cp_y
    @. qncGB.pt.zl = cp_zl
    @. qncGB.pt.zu = cp_zu
    qncGB.pt.μ     = cp_mu

    ##### AQUI COMEÇA O MÉTODO ALTERNATIVO #####

    if t == 3 # 3 por padrao # 30 iterações é suficiente para praticamente zerar a diferença entre sig_max e sig (ela fica na ordem de 4.66e-10)

      println("WARNING: Não foi possível determinar alpha e sigma de modo a obter a convergência do passo corretor. Isso pode ter ocorrido pois o ponto inicial não estava próximo o suficiente do caminho central. Para contornar isso, será aplicado um método quasi-newton alternativo.")

      sig = min(sig_max, 1.0 - 0.5*(GB_struct.qnc.αp + GB_struct.qnc.αd)) # OBS: contas recentes (2025) mostram que escolher sigma igual à 1 - alpha é mais interessante. Como na prática estamos usando dois alphas, estou considerando a média dos dois.

      b_status = Broyden!(GB_struct, mult, sig, it_max, eps, params, true)

      if !(b_status)

        println("WARNING: O método alternativo falhou. Andando apenas na direção preditora, sem correções...")

        # Descarta os deslocamentos feitos durante o método de Broyden e retorna mu para seu valor original

        @. qncGB.pt.x  = cp_x
        @. qncGB.pt.xl = cp_xl
        @. qncGB.pt.xu = cp_xu
        @. qncGB.pt.y  = cp_y
        @. qncGB.pt.zl = cp_zl
        @. qncGB.pt.zu = cp_zu
        qncGB.pt.μ     = cp_mu

        # Anda apenas na direção preditora

        #        qncGB.pt.x  .+= (params.StepDampFactor * qncGB.αp) .* Δ.x
        #        qncGB.pt.xl .+= (params.StepDampFactor * qncGB.αp) .* Δ.xl
        #        qncGB.pt.xu .+= (params.StepDampFactor * qncGB.αp) .* Δ.xu
        #        qncGB.pt.y  .+= (params.StepDampFactor * qncGB.αd) .* Δ.y
        #        qncGB.pt.zl .+= (params.StepDampFactor * qncGB.αd) .* Δ.zl
        #        qncGB.pt.zu .+= (params.StepDampFactor * qncGB.αd) .* Δ.zu

        @. Δc.x  = (params.StepDampFactor * qncGB.αp) * Δ.x
        @. Δc.xl = (params.StepDampFactor * qncGB.αp) * Δ.xl
        @. Δc.xu = (params.StepDampFactor * qncGB.αp) * Δ.xu
        @. Δc.y  = (params.StepDampFactor * qncGB.αd) * Δ.y
        @. Δc.zl = (params.StepDampFactor * qncGB.αd) * Δ.zl
        @. Δc.zu = (params.StepDampFactor * qncGB.αd) * Δ.zu


      end

      break

    end

    t += 1
  end

  # Calcula a direção resultante após o passo preditor e as iterações do método de Broyden

  #  Δc.x  =  qncGB.pt.x - cp_x
  #  Δc.y  =  qncGB.pt.y - cp_y
  #  Δc.xl = qncGB.pt.xl - cp_xl
  #  Δc.xu = qncGB.pt.xu - cp_xu
  #  Δc.zl = qncGB.pt.zl - cp_zl
  #  Δc.zu = qncGB.pt.zu - cp_zu

  # Retorna o ponto para sua posição inicial.

  pt.x  .= cp_x
  pt.xl .= cp_xl
  pt.xu .= cp_xu
  pt.y  .= cp_y
  pt.zl .= cp_zl
  pt.zu .= cp_zu

end
