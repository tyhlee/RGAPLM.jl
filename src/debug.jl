# testing
using BenchmarkTools
using LinearAlgebra
using RDatasets
using Distributions, StatsFuns, SpecialFunctions
using Roots
using Debugger
using Plots
using Printf
include("utils.jl")
include("regression.jl")

t = collect([1.0:1.0:365;])
tsq = t.^2
esp = rand(length(t))
X = [ones(length(t)) t]
beta = [1, 0.001]
T = [sin.((2*7*pi).*t./365) cos.((2*7*pi).*t./365)]
T = T .- sum(T,dims=1)
eta = vec(X * beta .+ sum(T,dims=2))
mu = exp.(eta)
y = rand.(Poisson.(mu))
youtlier = copy(y)
w = ones(length(y))
span = repeat([0.15],size(T)[2])
degree = ones(Int,size(T)[2])
c = 100.0
max_it = 50
family="NB"
sigma=1.0

tmp = RGAPLM(y,X,T,
    family=family, method = "Pan", link="log", verbose=true,
    span=span,loess_degree=degree,
    sigma= 1.0,
    beta=nothing,
    c=c,
    c_X=c,
    c_T=c,
    c_sigma=c,
    epsilon=1e-4,max_it=10,
    epsilon_T = 1e-4, max_it_T = 5,
    epsilon_X = 1e-4, max_it_X = 5,
    epsilon_RAM = 1e-4, max_it_RAM = 5,
    epsilon_sigma=1e-4,
    epsilon_eta=1e-4, max_it_eta = 5,
    initial_beta = false, maxmu=1e5,
    minmu=1e-10,min_sigma = 1e-3,max_sigma = 1e3)


Juno.@enter RGAPLM(y,X,T,
    family=family, method = "Pan", link="log", verbose=true,
    span=span,loess_degree=degree,
    sigma= 1.0,
    beta=nothing,
    c=c,
    c_X=c,
    c_T=c,
    c_sigma=c,
    epsilon_T = 1e-6, max_it_T = 5,
    epsilon_X = 1e-6, max_it_X = 5,
    epsilon_RAM = 1e-6, max_it_RAM = 10,
    epsilon_sigma=1e-6,
    epsilon_eta=1e-4, max_it_eta = 25,
    initial_beta = false, maxmu=1e5,
    minmu=1e-10,min_sigma = 1e-3,max_sigma = 1e3)
