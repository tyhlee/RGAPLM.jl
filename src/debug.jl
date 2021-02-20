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
c = 10000.0
max_it = 50
family="P"
sigma=1.0

beta = zeros(2)
# use the median for the initial estimate
beta[1] = g_link(family,median(y),link="log")
par = vec(X * beta)
eta = copy(par)
mu = g_invlink(family,eta)
s = sqrt.(g_var(family,mu,sigma))
z, w = g_ZW(family,"Tukey",y,mu,s,c,sigma)

Juno.@enter RGAPLM(y,X,T,
    family ="P", method = "Pan",
    span=span,loess_degree=degree,
    beta=nothing,
    sigma= 1.0,
    c=c, robust_type ="none",
    c_X=c, robust_type_X ="none",
    c_T=c, robust_type_T ="none",
    epsilon=1e-6, max_it = 50,
    epsilon_T = 1e-6, max_it_T = 50,
    epsilon_X = 1e-6, max_it_X = 50)
