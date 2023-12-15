include("./IPM/MPC-PEDRO/load_problems.jl")
include("./IPM/MPC-PEDRO/PC-N-QN.jl")
println("Execute o comando testar(nome_ou_numero) para testar um problema.")

function testar(nome)
if nome == 1
    c = [1., 0]
    A = [1. 1]
    b = [1]
    x0, y0, s0 = make_feasible(A, b, c, 10.0)
  elseif nome == 2
    c = [-1., -1, 0, 0]
    A = [-1. 3 1 0; 5 -3 0 1]
    b = [9., 15]
    x0, y0, s0 = make_feasible(A, b, c, 10.0)
  elseif nome == 3
    c = [0, 1., 0]
    A = [1. 2 1]
    b = [6.]
    x0, y0, s0 = make_feasible(A, b, c, 10.0)
  else
    p = load_problem(nome)
    (c, A, b) = (p.c, p.A, p.b)
    x0, y0, s0 = make_feasible(p, 10.0)
  end

w = vcat(x0,y0,s0)
PC_NQN(c, A, b, w)

end
