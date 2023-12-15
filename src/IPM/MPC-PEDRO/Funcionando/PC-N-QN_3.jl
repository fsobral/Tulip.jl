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
    B.rho[size] = -dot(s, B.u[size]) + dot(s, s)
    B.sb[size] = s
    B.size    = size

end

function Broyden_parada(F_tau, w, sig, sig_max, mu_wk, m, n, it, it_max, eps)
#  if norm(F_tau(w, sig*mu_wk)) < eps
#      println("O Método de Broyden Convergiu.")
#    return (true, true)
#  end
  if (dot(w[1:n], w[n+m+1: 2*n+m]) / n) < sig*mu_wk && minimum([minimum(w[1:n]), minimum(w[n+m+1: 2*n+m])]) > 0 # trocar < por <= se nao der certo
    println("Broyden interrompido: houve decréscimo suficiente de mu.")
    return (true, true)
  end
  if it < it_max
    return (false, nothing)
  end
  println("Broyden interrompido: o limite de iterações foi atingido.")
  return (true, false)
end

function Broyden(F_tau, Jw, w, mu_wk, sig, sig_max, m, n, eps = 1.0e-8, it_max = 10)

  status = false
  B = GoodBroyden(lu(Jw), it_max)
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
  x = w[1:n]
  lamb = w[n+1:n+m]
  s = w[n+m+1:2*n+m]
  Jw = J(w)
  Fw = F_tau(w, 0) # F em sigma_k mu_k com sigma_k = 0.
  # Passo 1
  d = - Jw \ Fw
  dx = d[1:n]
  ds = d[n+m+1:2*n+m]
  # Passo 2
  v = [100.]
  for i=1:n
    if dx[i] < 0
      v = vcat(v, -x[i]/dx[i] )
    end
    if ds[i] < 0
      v = vcat(v, -s[i]/ds[i] )
    end
  end
  alpha = minimum([1.0, 0.9*minimum(v)]) # calcula alpha máximo tal que algum xi ou si zera e depois toma 90% desse passo, ou passo 1 no caso em que nenhuma variável bloqueia o passo.
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
    w_n = w + alpha*d
     b_status = Broyden(F_tau, Jw, w_n, mu_wk, sig, sig_max, m, n, eps, it_max)
    println("mu (final do broyden)=", (1/n)*w_n[1:n]'*w_n[n+m+1:2n+m])
    println("Status (Broyden) = ", b_status)
#    mu_n = (w_n[1:n]'*w_n[n+m+1:2*n+m])/n
    if b_status == true # Se encontrar um ponto em F_0 com decrescimo de mu, pare.
      break
    end
    alpha *= 0.5
    sig = 0.5*(sig_max + sig)
    if abs(sig_max - sig) < eps # Parar se sig se aproximar muito de sig_max (e depois acusar erro: sig_max não é grande suficiente ou o ponto inicial tomado não está próximo o suficiente do caminho central)
      status = false
      break
    end
    t += 1
  end
  # Passo 5
  return (w_n, status)
end

function PC_NQN(c, A, b, w, sig_max = 0.9999, eps=1.0e-8, it_max1=1000, it_max2=100)
  m, n = size(A)
  k = 0
  w0 = copy(w)

  F_tau(w, tau) = begin
    x = w[1:n]
    lamb = w[n+1:n+m]
    s = w[n+m+1:2*n+m]
    if tau == 0
      return vcat(A'*lamb + s - c, A*x-b, diagm(x)*diagm(s)*ones(n))
    end
    return vcat(A'*lamb + s - c, A*x-b, diagm(x)*diagm(s)*ones(n) - tau*ones(n))
  end
  J(w) = begin
    x = w[1:n]
    s = w[n+m+1:2*n+m]
    return vcat(hcat(zeros(n, n), A', I),
             hcat(A, zeros(m, m+n)),
             hcat(diagm(s), zeros(n, m), diagm(x))
               )
  end
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
