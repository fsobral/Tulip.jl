using Printf
using JuMP
#import Tulip

# Instantiate JuMP model
lp = Model(Tulip.Optimizer)

# Create variables
@variable(lp, x >= 0)
@variable(lp, y >= 0)

# Add constraints
@constraint(lp, row1, -x +3*y <= 9)
@constraint(lp, row2, 5*x -3*y <= 15)

# Set the objective
@objective(lp, Min, -x - y)

# Set some parameters
set_optimizer_attribute(lp, "OutputLevel", 0)  # disable output
set_optimizer_attribute(lp, "Presolve_Level", 0)     # disable presolve

# Solve the problem
optimize!(lp)

# Check termination status
st = termination_status(lp)
println("Termination status: $st")

# Query solution value
objval = objective_value(lp)
x_ = value(x)
y_ = value(y)

@printf "Z* = %.4f\n" objval
@printf "x* = %.4f\n" x_
@printf "y* = %.4f\n" y_
