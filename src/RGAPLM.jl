module RGAPLM

# import packages
using LinearAlgebra, Distributions, StatsFuns, SpecialFunctions, Roots, Printf

# Write your package code here.
include("utils.jl")
include("regression.jl")

export wls, loess, rloess, AM, RAM, RGAPLM

end
