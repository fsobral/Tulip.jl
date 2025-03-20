module Tulip

global nitb = 0 # numero total de iterações de Broyden
global n_corr_alt = 0 # numero total de utilizações do método alternativo
global n_corr_jac = 0 # numero total de utilizações do método alternativo
using LinearAlgebra
using Logging
using Printf
using SparseArrays
using TOML

using TimerOutputs

# Read Tulip version from Project.toml file
const TULIP_VERSION = let project = joinpath(@__DIR__, "..", "Project.toml")
    Base.include_dependency(project)
    VersionNumber(TOML.parsefile(project)["version"])
end
version() = TULIP_VERSION

include("utils.jl")

# Linear algebra
include("LinearAlgebra/LinearAlgebra.jl")
import .TLPLinearAlgebra.construct_matrix
const TLA = TLPLinearAlgebra

# KKT solvers
include("KKT/KKT.jl")
using .KKT

# Commons data structures
# TODO: put this in a module

include("status.jl")    # Termination and solution statuses
include("problemData.jl")
include("solution.jl")
include("attributes.jl")

# Presolve module
include("Presolve/Presolve.jl")

# IPM solvers
include("./IPM/IPM.jl")
# incluir Corretor-Quasi-Newton
include("./IPM/MPC-PEDRO/Quasi-Newton-Corrector.jl")
# incluir funcoes uteis
include("./IPM/MPC-PEDRO/load_problems.jl")

include("parameters.jl")

# Model
include("model.jl")

# Interfaces
include("Interfaces/tulip_julia_api.jl")
include("Interfaces/MOI/MOI_wrapper.jl")

end # module
