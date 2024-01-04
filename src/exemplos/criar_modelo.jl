using Printf
using JuMP
#import Tulip

function testar(nome, ignore = false)

    if nome == 1
        c = [1., 0]
        A = [1. 1]
        b = [1]
    elseif nome == 2
        c = [-1., -1, 0, 0]
        A = [-1. 3 1 0; 5 -3 0 1]
        b = [9., 15]
    elseif nome == 3
        c = [0, 1., 0]
        A = [1. 2 1]
        b = [6.]
    elseif nome == 4
        c = [1., 1]
        A = [5 7.; 5 -3]
        b = [35., 10]
    else
        p = load_problem(nome)
        (c, A, b) = (p.c, p.A, p.b)
    end

    modelar_e_resolver(c, A, b, ignore)

end

function modelar_e_resolver(c, A, b, ignore)

    m, n = size(A)

    # Instantiate JuMP model
    global lp = Model(Tulip.Optimizer)

    # Create variables
    @variable(lp, x[1:n] >= 0)

    # Add constraints
    @constraint(lp, A * x .== b)

    # Set the objective
    @objective(lp, Min, c' * x)

    # Set some parameters
    set_optimizer_attribute(lp, "OutputLevel", 0)  # disable output
    set_optimizer_attribute(lp, "Presolve_Level", 0)     # disable presolve

    # Solve the problem

    if ignore == false
        optimize!(lp)

        # Check termination status
        st = termination_status(lp)
        println("Termination status: $st")

        # Query solution value
        objval = objective_value(lp)
        x_ = value.(x)

        @printf "Z* = %.4f\n" objval
        println("x* = ")
        display(x_)
    else
        try
            optimize!(lp)

            # Check termination status
            st = termination_status(lp)
            println("Termination status: $st")

            # Query solution value
            objval = objective_value(lp)
            x_ = value.(x)

            @printf "Z* = %.4f\n" objval
            println("x* = ")
            display(x_)
        catch err
            println("")
            println("⚠️ AVISO: Ocorreu um erro durante a resolução deste problema.")
            println("")
            println(err)
        end
    end
    #for i=1:n
    #@printf " %.4f\n" x_[i]
    #end
    #println("")

    #return x_

end
