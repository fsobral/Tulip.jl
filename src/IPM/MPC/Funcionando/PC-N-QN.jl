using LinearAlgebra

function Broyden(B, w, sig, m, n, eps, sig_max, it_max)
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
    B0 = B0 + ((y - B0*d)*d')/(d'*d)
    w0 = w1
    k += 1
    x0 = w0[1:n]
    s0 = w0[n+m+1:2*n+m]
    mu0 = (x0'*s0)/n
    x1 = w1[1:n]
    s1 = w1[n+m+1:2*n+m]
    mu1 = (x1'*s1)/n
    if mu1 < 0.5*(sig_max + sig)*mu0 && minimum([minimum(x1), minimum(s1)]) > -eps && k < it_max
      status = true
      for i=1:(2*n+m)
        if abs(w0[i]) < eps
          w0[i] = 0.0
        end
      end
      break
    end
  end
  println("Nº de iterações de Broyden: ", k)
  return (w0, status)
end

function iteracao(c, A, b, w, it_max, eps, sig_max)
  status = true
  m, n = size(A)
  x = w[1:n]
  lamb = w[n+1:n+m]
  s = w[n+m+1:2*n+m]
  mu = (x'*s)/n
  J = vcat(hcat(zeros(n, n), A', I),
           hcat(A, zeros(m, m+n)),
           hcat(diagm(s), zeros(n, m), diagm(x))
          )
#  Fw = vcat(zeros(m), zeros(n), diagm(x)*diagm(s)*ones(n))
  Fw = vcat(A'*lamb + s - c, A*x-b, diagm(x)*diagm(s)*ones(n))
  # Passo 1
  d = - J \ Fw
  dx = d[1:n]
  ds = d[n+m+1:2*n+m]
  # Passo 2 (Esse passo vai ser reescrito para determinar alpha sem fazer uma busca linear)
  v = [1.0]
  for i=1:n
    if dx[i] < 0
      v = vcat(v, -x[i]/dx[i] )
    end
    if ds[i] < 0
      v = vcat(v, -s[i]/ds[i] )
    end
  end
  alpha = 0.9*minimum(v) # calcula alpha máximo tal que algum xi ou si zera e depois toma 90% desse passo
  # Passo 3
  sig = 0.5*sig_max
  # Passo 4
  w_n = zeros(2n+m)
  t = 1
  while true
    println("Testagem: ", t)
    println("Alfa = ", alpha)
    println("Sigma = ", sig)
    w_n, b_status = Broyden(J, w + alpha*d, sig, m, n, eps, sig_max, it_max)
    println("w_n = ")
    display(w_n)
    println("Status = ", b_status)
    mu_n = (w_n[1:n]'*w_n[n+m+1:2*n+m])/n
    if b_status == true # Se encontrar um ponto em F_0 com decrescimo de mu, pare.
      break
    end
    alpha *= 0.5
    sig = 0.5*(sig_max + sig)
    if abs(sig_max - sig) < eps # Parar se alpha chegar muito perto de 0
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
  while k < it_max1
    k += 1
    println("")
    println("Nº da iteração = ", k)
    (w0, status) = iteracao(c, A, b, w0, it_max2, eps, sig_max)
    k += 1
    mu = (w0[1:n]'*w0[n+m+1:2*n+m])/n
    if status == false
      error("Não foi possível determinar alpha e sigma de modo a obter a convergência do passo corretor. Tente outro ponto inicial ou tome sig_max ainda mais próximo de 1.")
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

function Encontrar_Ponto_Inicial(c, A, b, mu = 1.0, theta = 0.25, eps = 1.0e-4, it_max=1000)
  m, n = size(A)
  F(w) = begin
    x = w[1:n]
    lamb = w[n+1:n+m]
    s = w[n+m+1:2*n+m]
    mu = (x'*s)/n
    z = abs(norm(diagm(x)*diagm(s)*ones(n) - mu*ones(n)) - 0.5*theta*mu) # Gambiarra: na solução, ||XSe - \mu e|| = 0.5 \theta \mu <= \theta \mu
    return vcat(A'*lamb + s - c, A*x-b, z)
  end
  JF(w) = ForwardDiff.jacobian(F, w)
  w0 = mu*rand(2*n+m)
  k = 0
  while k < it_max
    J = JF(w0)
    d = - J'*((J*J') \ F(w0))
    alpha = 1.0
    while true
      w1 = w0 + alpha*d
      x1 = w1[1:n]
      s1 = w1[n+m+1:2*n+m]
      mu1 = (x1'*s1)/n
      if norm(diagm(x1)*diagm(s1)*ones(n) - mu1*ones(n)) <= theta*mu1
        break
      end
      alpha *= 0.5
      if alpha < eps
        break
      end
    end
    w0 = w0 + alpha*d
    k += 1
    if norm(F(w0)) < eps
      break
    end
  end
  if k == it_max
    #println("O ponto encontrado não é um ponto inicial confiável, pois o limite de iterações foi atingido.")
    return false
  end
  return w0
end

function Encontrar_Ponto_Inicial_alt(c, A, b, M = 1.0, eps = 1.0e-4, it_max=1000)
  m, n = size(A)
  F(w) = begin
    x = w[1:n]
    lamb = w[n+1:n+m]
    s = w[n+m+1:2*n+m]
    return vcat(A'*lamb + s - c, A*x-b)
  end
  JF(w) = ForwardDiff.jacobian(F, w)
  w0 = rand(2*n+m)
  for i=1:(2n+m)
    w0[i] = abs(w0[i])
  end
  w0 = M*w0
  status = false
  k = 0
  while k < it_max
    J = JF(w0)
    d = - J'*((J*J') \ F(w0))
    alpha = 1.0
    while true
      w1 = w0 + alpha*d
      x1 = w1[1:n]
      s1 = w1[n+m+1:2*n+m]
      mu1 = (x1'*s1)/n
      if (F(w1) < F(w0)) && minimum(x1) > 0 && minimum(s1) > 0 
        break
      end
      alpha *= 0.5
      if alpha < eps
        status = false
        break
      end
    end
    w0 = w0 + alpha*d
    k += 1
    if norm(F(w0)) < eps
      status = true
      break
    end
  end
  return (w0, status)
end

function chutar_e_encontrar(c, A, b, M = 1.0, eps = 1.0e-4, it_max=1000)
# Essa procura a solução até encontrar (pode demorar muito)
    while true
      (w, status) = Encontrar_Ponto_Inicial_alt(c, A, b, M, eps, it_max)
        if !(status == false)
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
