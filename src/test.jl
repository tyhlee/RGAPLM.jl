# testing
using BenchmarkTools
using LinearAlgebra
using RDatasets
using Distributions
using Debugger
using Plots
include("utils.jl")
include("regression.jl")


# testing RAM, AM
t = collect([1.0:1.0:365;])
tsq = t.^2
esp = rand(length(t))
X = [t sin.((2*7*pi).*t./365) esp]
y = 0.2.* t .+ 3 .* sin.((2*7*pi).*t./365) .+ esp
youtlier = copy(y)
youtlier[230:232] = youtlier[230:232]*1.5
plot(t,y)
plot!(t,youtlier)
w = ones(length(y))
span = repeat([0.08],size(X)[2])
degree = 1
c = 3.0
max_it = 50

# debugging rloess
spanj = 0.1
tmp = loess(youtlier,t,t,span=spanj,degree=degree)
tmp2 = rloess(youtlier,t,t,span=spanj,degree=degree,c=1.0,max_it=100)
plot(t,youtlier)
plot!(t,tmp)
plot!(t,tmp2)

# AM, RAM
tmp = AM(youtlier,X,w,span=span,loess_degree=degree,max_it=max_it)
tmp2 = RAM(youtlier,X,w;
        span=span,loess_degree=degree,max_it=5,max_it_loess=10,c=c,sigma=0.2,max_it_global=5)
plot(t,youtlier)
plot!(t,sum(tmp[2],dims=2).+tmp[1])
plot!(t,sum(tmp2[2],dims=2).+tmp2[1])
plot!(t,y)

# g_ZW
n = 500
mu = ceil.(Int,rand(Uniform(0,500),n))
y = rand.(Poisson.(mu))
s = sqrt.(mu)
sigma = 2.0
c = 1.345

@benchmark g_ZW("P","Huber",y,mu,s,c,sigma)
@benchmark g_ZW("P","Tukey",y,mu,s,c,sigma)
@benchmark g_ZW("NB","Huber",y,mu,sqrt.(mu .+ mu .^2 .* sigma),c,sigma)
@benchmark g_ZW("NB","Tukey",y,mu,sqrt.(mu .+ mu .^2 .* sigma),c,sigma)
@benchmark g_ZW2("NB","Tukey",y,mu,sqrt.(mu .+ mu .^2 .* sigma),c,sigma)

tmp = g_ZW("NB","Tukey",y,mu,sqrt.(mu .+ mu .^2 .* sigma),c,sigma)
tmp2 =g_ZW2("NB","Tukey",y,mu,sqrt.(mu .+ mu .^2 .* sigma),c,sigma)
ttt = collect(1:n)
plot(ttt,tmp[1])
plot!(ttt,tmp2[1])
plot(ttt,tmp[1] .- tmp2[1])
look_index = abs.(tmp[1] .-tmp2[1]) .> 0.2
# 30 , 83
ttt[look_index.==1]
j1 = max.(ceil.(Int,mu .- c .* s),0)
j2 = floor.(Int,mu .+ c .* s)
j1[81]
j2[81]
tmp[1][30]
tmp2[1][30]
y[30]
mu[30]

# TODO: test with classic and let c= large number
