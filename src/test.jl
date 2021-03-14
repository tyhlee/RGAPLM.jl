# testing
using RGAPLM
using BenchmarkTools
using RDatasets
using Debugger
using Plots
using Printf

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
degree = ones(Int64,3)
tmp = AM(youtlier,X,w,span=span,loess_degree=degree,max_it=max_it)
tmp2 = RAM(youtlier,X,w;
        span=span,loess_degree=degree,max_it=5,max_it_loess=10,c=c,sigma=0.2,max_it_global=5)
plot(t,youtlier)
plot!(t,sum(tmp[2],dims=2).+tmp[1])
plot!(t,sum(tmp2[2],dims=2).+tmp2[1])
plot!(t,y)

# g_ZW
n = 25
mu = ceil.(Int,rand(Uniform(0,500),n))
y = rand.(Poisson.(mu))
s = sqrt.(mu)
sigma = 1.0
c = 1.345
ttt = collect(1:n)

c = 1000.0 # all identical
max_j = Int(1e10)
tmp = g_ZW("P","none",y,mu,s,c,sigma,max_j=max_j)
tmp1 =g_ZW("P","Huber",y,mu,s,c,sigma,max_j=max_j)
tmp2 = g_ZW("P","Tukey",y,mu,s,c,sigma,max_j=max_j)
plot(ttt,tmp[1])
plot!(ttt,tmp1[1])
plot!(ttt,tmp2[1])
plot(ttt,tmp[2])
plot!(ttt,tmp1[2])
plot!(ttt,tmp2[2])

c =  1.0 # should see some difference
tmp = g_ZW("P","none",y,mu,s,c,sigma)
tmp1 =g_ZW("P","Huber",y,mu,s,c,sigma)
tmp2 = g_ZW("P","Tukey",y,mu,s,c,sigma)
plot(ttt,tmp[1])
plot!(ttt,tmp1[1])
plot!(ttt,tmp2[1])
plot(ttt,tmp[2])
plot!(ttt,tmp1[2])
plot!(ttt,tmp2[2])

c = 500.0 # should be identical across all three
sigma = 2.0
tmp = g_ZW("NB","none",y,mu,sqrt.(mu .+ mu .^2 .* sigma),c,sigma)
tmp1 =g_ZW("NB","Huber",y,mu,sqrt.(mu .+ mu .^2 .* sigma),c,sigma)
tmp2 = g_ZW("NB","Tukey",y,mu,sqrt.(mu .+ mu .^2 .* sigma),c,sigma)
plot(ttt,tmp[1])
plot!(ttt,tmp1[1])
plot!(ttt,tmp2[1])
plot(ttt,tmp[2])
plot!(ttt,tmp1[2])
plot!(ttt,tmp2[2])

c = 1.0 # should see some differences
tmp = g_ZW("NB","none",y,mu,sqrt.(mu .+ mu .^2 .* sigma),c,sigma)
tmp1 =g_ZW("NB","Huber",y,mu,sqrt.(mu .+ mu .^2 .* sigma),c,sigma)
tmp2 = g_ZW("NB","Tukey",y,mu,sqrt.(mu .+ mu .^2 .* sigma),c,sigma)
plot(ttt,tmp[1])
plot!(ttt,tmp1[1])
plot!(ttt,tmp2[1])
plot(ttt,tmp[2])
plot!(ttt,tmp1[2])
plot!(ttt,tmp2[2])

# performance
sigma = 2.0
c = 1.345
@benchmark g_ZW("P","Huber",y,mu,s,c,sigma)
@benchmark g_ZW("P","Tukey",y,mu,s,c,sigma)
@benchmark g_ZW("NB","Huber",y,mu,sqrt.(mu .+ mu .^2 .* sigma),c,sigma)
@benchmark g_ZW("NB","Tukey",y,mu,sqrt.(mu .+ mu .^2 .* sigma),c,sigma)
@benchmark g_ZW2("NB","Tukey",y,mu,sqrt.(mu .+ mu .^2 .* sigma),c,sigma)

tmp = g_ZW("NB","Tukey",y,mu,sqrt.(mu .+ mu .^2 .* sigma),c,sigma)
tmp2 =g_ZW2("NB","Tukey",y,mu,sqrt.(mu .+ mu .^2 .* sigma),c,sigma)
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

# test rGAPLM - Poisson Pan
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
family="P"
sigma=1.0
par = vec(X * beta)
eta = copy(par)
mu = g_invlink(family,eta)
s = sqrt.(g_var(family,mu,sigma))
z, w = g_ZW(family,"Tukey",y,mu,s,c,sigma)

Pan_P = rGAPLM(y,X,T,
    family ="P", method = "Pan",
    span=span,loess_degree=degree,
    beta=nothing,
    sigma= 1.0,
    c=c, robust_type ="Huber",
    c_X=c, robust_type_X ="none",
    c_T=c, robust_type_T ="none",
    epsilon=1e-6, max_it = 50,
    epsilon_T = 1e-6, max_it_T = 50,
    epsilon_X = 1e-6, max_it_X = 50);

# 8 iterations!
plot(t,y)
plot!(t,Pan_P.mu)
plot(t,mu)
plot!(t,Pan_P.mu)
plot(beta,seriestype = :scatter,title="β")
plot!(Pan_P.beta,seriestype = :scatter)

# inject some outliers
y[100:103] = ceil.(Int,y[100:103].*2.5)
c=3.5
Pan_P = rGAPLM(y,X,T,
    family ="P", method = "Pan",
    span=span,loess_degree=degree,
    beta=nothing,
    sigma= 1.0,
    c=c, robust_type ="Tukey",
    c_X=c, robust_type_X ="Tukey",
    c_T=c, robust_type_T ="Tukey",
    epsilon=1e-6, max_it = 50,
    epsilon_T = 1e-6, max_it_T = 5,
    epsilon_X = 1e-6, max_it_X = 5);

plot(t,y)
plot!(t,Pan_P.mu)
png("figs/Poisson_Pan")
plot(t,mu)
plot!(t,Pan_P.mu)
png("figs/Poisson_Pan_mu")
plot(beta,seriestype = :scatter,title="β")
plot!(Pan_P.beta,seriestype = :scatter)
png("figs/Poisson_Pan_beta")

# test rGAPLM - Poisson Lee
method = "Lee"
family = "P"
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
c = 5000.0
max_it = 50

model = rGAPLM(y,X,T,
    family =family, method = method,
    span=span,loess_degree=degree,
    beta=nothing,
    sigma= 1.0,
    c=c, robust_type ="Tukey",
    c_X=c, robust_type_X ="Tukey",
    c_T=c, robust_type_T ="Tukey",
    epsilon=1e-6, max_it = 50,
    epsilon_T = 1e-6, max_it_T = 10,
    epsilon_X = 1e-6, max_it_X = 10,
    epsilon_RAM=1e-6, max_it_RAM=10);

model.convergence

plot(t,y)
plot!(t,Pan_P.mu)
plot!(t,model.mu)
plot(t,mu)
plot!(t,model.mu)
plot!(t,Pan_P.mu)
plot(beta,seriestype = :scatter,title="β")
plot!(model.beta,seriestype = :scatter)
plot!(Pan_P.beta,seriestype = :scatter)

# inject some outliers
y[100:103] = ceil.(Int,y[100:103].*2.5)
c=3.5
span = repeat([0.3],size(T)[2])
include("regression.jl")
model = rGAPLM(y,X,T,
    family =family, method = method,
    span=span,loess_degree=degree,
    beta=nothing,
    sigma= 1.0,
    c=c, robust_type ="Tukey",
    c_X=c, robust_type_X ="Tukey",
    c_T=c*1.5, robust_type_T ="Tukey",
    epsilon=1e-6, max_it = 15,
    epsilon_T = 1e-6, max_it_T = 5,
    epsilon_X = 1e-6, max_it_X = 5,
    epsilon_RAM=1e-6, max_it_RAM=5);

plot(t,y)
plot!(t,model.mu)
png("figs/Poisson_Lee")
plot(t,mu)
plot!(t,model.mu)
png("figs/Poisson_Lee_mu")
plot(beta,seriestype = :scatter,title="β")
plot!(model.beta,seriestype = :scatter)
png("figs/Poisson_Lee_beta")



# test sigma estimator
y_var = var(y)
y_mean  = mean(y)
hat_eta = y_var/y_mean
hat_theta = y_mean/hat_eta
hat_sigma = 1/hat_theta

sigma_estimator(1.0,y,mu,10.0)

function sigma_estimator(sigma, y,mu,c)
    loglkhd_sig1 = 0
    loglkhd_sig0 = 1
    epsilon_sigma = 1e-6
    iter_sigma=0
    max_it_sigma = 100
    max_sigma = 1000

    while (abs(loglkhd_sig1 - loglkhd_sig0) > epsilon_sigma) & (iter_sigma < max_it_sigma)
        if isnothing(c)
            sigma = abs(sigma - score_sigma(y,mu,sigma)/info_sigma(y,mu,sigma))
        else
            robust_sigma_uniroot = (xx -> robust_sigma(y,mu,xx,c;type="Tukey",family="NB"))
            roots = Roots.find_zero(robust_sigma_uniroot,(0.01,10),Roots.Bisection(),maxevals=25,tol=1e-4)
            # roots = Roots.find_zeros(robust_sigma_uniroot,0.01,10,maxevals=25,tol=1e-4)
            #roots = Roots.find_zeros(robust_sigma_uniroot,1e-1,1e1,verbose=true)
            # roots = Roots.fzero(robust_sigma_uniroot,1,order=2,verbose=true)
            # roots = Roots.fzero(robust_sigma_uniroot,0.01,50,verbose=true)
            # println("$roots")
            if length(roots) == 0
                # do not update
                @warn("Roots not found for sigma (current value: $sigma)")
            elseif length(roots) == 1
                #  update sigma
                sigma = roots[1]
            else
                # likely multipel roots close to zero
                roots = Roots.find_zeros(robust_sigma_uniroot,0.1,max_sigma,verose=true)
                if length(roots) == 0
                    sigma = 0.1
                else
                    sigma = roots[1]
                end
            end
        end
        if sigma > max_sigma
            sigma = max_sigma
        end
        loglkhd_sig0 = copy(loglkhd_sig1)
        loglkhd_sig1 = ll(y,mu,sigma)
        iter_sigma += 1
        print("Iter: $iter_sigma, sigma: $sigma, ll: $loglkhd_sig1 \n")
    end
    sigma
end

# test rGAPLM - NB Pan
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
c = 30.0 # don't set it too large; otherwise computation is too slow
max_it = 50
family="NB"
sigma=1.5
y = float.(rand.(NegativeBinomial.(1/sigma, 1 ./ (sigma .* mu .+ 1))))

# inject some outliers
y[100:103] = ceil.(Int,y[100:103].*2.5)
c=50.0

model =  rGAPLM(y,X,T,
    family=family, method = "Pan", link="log", verbose=true,
    span=span,loess_degree=degree,
    sigma= 1.0,
    beta=nothing,
    c=c,
    c_X=c,
    c_T=c,
    c_sigma=c*2,
    epsilon=1e-6,max_it=10,
    epsilon_T = 1e-5, max_it_T = 5,
    epsilon_X = 1e-5, max_it_X = 10,
    epsilon_RAM = 1e-5, max_it_RAM = 5,
    epsilon_sigma=1e-5, max_it_sigma=10,
    epsilon_eta=1e-5, max_it_eta = 10,
    initial_beta = false, maxmu=1e5,
    minmu=1e-5,min_sigma = 0.1,max_sigma = 15.0)

model_Lee =  rGAPLM(y,X,T,
    family=family, method = "Lee", link="log", verbose=true,
    span=span,loess_degree=degree,
    sigma= 1.0,
    beta=nothing,
    c=c,
    c_X=c,
    c_T=c,
    c_sigma=c*2,
    epsilon=1e-6,max_it=10,
    epsilon_T = 1e-5, max_it_T = 5,
    epsilon_X = 1e-5, max_it_X = 10,
    epsilon_RAM = 1e-5, max_it_RAM = 5,
    epsilon_sigma=1e-5, max_it_sigma=10,
    epsilon_eta=1e-5, max_it_eta = 10,
    initial_beta = false, maxmu=1e5,
    minmu=1e-5,min_sigma = 0.1,max_sigma = 15.0)

plot(t,y)
plot!(t,model.mu)
plot(t,mu)
plot!(t,model.mu)
plot(beta,seriestype = :scatter,title="β")
plot!(model.beta,seriestype = :scatter)
plot((beta .- model.beta) ./ beta .* 100,seriestype = :scatter,title="β")
print((sigma .- model.sigma) ./ sigma .* 100)
