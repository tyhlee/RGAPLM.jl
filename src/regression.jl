using BenchmarkTools
using LinearAlgebra
using RDatasets
using Distributions, StatsFuns
using Debugger
using Plots
using Printf
include("utils.jl")

# weighted least squares
function wls(y::type_VecFloatInt,
    X::type_VecOrMatFloatInt,
    w::type_VecFloatInt=ones(length(y)),
    method=:qr)

    if size(y) != size(w)
        throw(DimensionMisMatch("y and w must have the same length"))
    end

    if method == :qr
        w = sqrt.(w)
        return (Diagonal(w)*X \ (w .* y))
    else
        error("Method: $method is not supported. It should be :qr")
    end
end

# univariate loess
function loess(y::type_VecFloatInt,
    x::type_VecFloatInt,
    xi::type_VecFloatInt,
    w::type_VecFloatInt=ones(length(y));
    span::Float64=0.75,
    degree::Int64=2)

    # results
    results = similar(y)

    # compute the design matrix
    if all(x==xi)
        for j in 1:length(xi)

            tmp_x = outer_power(x.-xi[j],degree)
            tmp_weight = compute_kernel_weight(x.-xi[j],span)
            tmp_weight = w .* tmp_weight
            if !all(tmp_weight.>=0)
                throw(DomainError(j,"there is a negative weight"))
            end
            results[j] = wls(y,tmp_x,tmp_weight)[1]
        end
    else
        for j in 1:length(xi)
            tmp_x = outer_power(x.-xi[j],degree)
            tmp_weight = compute_kernel_weight(x.-xi[j],span)
            tmp_weight = w .* tmp_weight
            if !all(tmp_weight.>=0)
                throw(DomainError(j,"there is a negative weight"))
            end
            results[j] = sum( wls(y,tmp_x,(tmp_weight)) .* tmp_x[j,:] )
        end
    end
    results
end

# univariate robust loess
function rloess(y::type_VecFloatInt,
    x::type_VecFloatInt,
    xi::type_VecFloatInt,
    w::type_VecFloatInt=ones(length(y));
    sigma::type_FloatInt=1.0,
    span::type_FloatInt=0.75,
    degree::U=2,
    c::type_FloatInt = 1.345,
    type::String="Huber",
    epsilon::type_FloatInt = 1e-6,
    max_it::U=5) where {T <: Float64, U <: Int}

    # initalization
    w = w ./ sigma
    wy = w .* y

    # results
    results = similar(y)
    wr = similar(y)
    wtmp_x = similar(y)
    iter::Int64 =0
    crit::Float64=epsilon*10

    # compute the design matrix
    for j in 1:length(xi)
        wtmp_x = w .* outer_power(x.-xi[j],degree)
        ker_weight = compute_kernel_weight(x.-xi[j],span)

        iter = 0
        crit = epsilon*10
        beta = zeros(degree+1)

        while (crit> epsilon) & (iter < max_it)
            tmp_beta = copy(beta)
            wr = (wy .- (wtmp_x * beta))
            tmp_weight = psi_weight(wr,c,type) .* ker_weight
            beta = wls(wy,wtmp_x,tmp_weight)
            iter += 1
            crit = compute_crit(beta,tmp_beta)
        end

        results[j] = beta[1]

    end

    results
end

# AM using backfitting
function AM(y::type_VecFloatInt,X::type_VecOrMatFloatInt,
    w::type_VecFloatInt=ones(length(y));
    span::type_VecFloatInt,loess_degree::Vector{N},
    epsilon::type_FloatInt=1e-6,max_it::N=25, y_mean::type_FloatInt = mean(y)) where {U <: Float64, N<:Int}

    # initialization
    # if there is a vector of ones (intercept), exclude it
    if any(mapslices(x-> all(x.==1.0),X,dims=1))
        tmp_index = findall(x -> x==0, all(mapslices(x-> all(x.==1.0),X,dims=1)))
        X = X[:,tmp_index]
    end

    n,k = size_VecOrMat(X)

    crit::Float64 = 10*epsilon
    iter::Int64 = 0
    S = zeros(Float64,n,k)

    # fit
    while (crit > epsilon) & (iter < max_it)
        S_tmp = copy(S)

        if k==1
            S = loess(y .- y_mean,X,X,w,span=span[1],degree=loess_degree[1])
        else
            for j in 1:k
                S[:,j] = loess(y .- y_mean .- vec(sum(dropcol(S,j),dims=2)),X[:,j],X[:,j],w,span=span[j],degree=loess_degree[j])
            end
        end
        # centering
        S = S .- mean(S,dims=1)

        # update crit & iter
        iter += 1
        crit = compute_crit(S,S_tmp)/n
    end
    return (alpha=y_mean,S=S)
end

# helper function for robust AM
function robust_intercept(y::type_VecFloatInt,mu::type_VecFloatInt=median(y),w::type_VecFloatInt=ones(length(y));
    sigma::type_VecFloatInt=1.0,
    type::String="Huber",c::type_VecFloatInt=1.345,
    epsilon::type_VecFloatInt=1e-6,max_it::N=10) where{T<: Float64, N <:Int}

    iter::Int64 = 0
    crit::type_FloatInt = epsilon*10
    w = w ./ sigma
    wsq = w.^2
    ww = similar(w)
    tmp_mu = mu
    res = similar(y)
    while (crit > epsilon) & (iter < max_it)
        res = (y .- mu) .* w
        ww = psi_weight(res,c,type) .* wsq
        tmp_mu = copy(mu)
        mu = sum(ww .* y) / sum(ww)
        crit = compute_crit([mu],[tmp_mu],"norm2_change")
        iter += 1
    end
    mu
end

# robust AM using backfitting
function RAM(y::type_VecFloatInt,X::type_VecOrMatFloatInt,w::type_VecFloatInt=ones(length(y));
    span::type_VecFloatInt,loess_degree::Vector{N},
    max_it_loess::N = 10, epsilon_loess::type_FloatInt= 1e-6,
    type::String="Huber",c::type_FloatInt = 1.345,
    sigma::type_FloatInt = 0.0, epsilon::type_FloatInt=1e-6,max_it::N=10, alpha::type_FloatInt = median(y),
    epsilon_global::type_FloatInt = 1e-6, max_it_global::N = 5,
    update_intercept::Bool = true) where {U <: Float64, N<:Int64}

    # initialization
    # if there is a vector of ones (intercept), exclude it
    # intercept is updated separately
    if any(mapslices(x-> all(x.==1.0),X,dims=1))
        tmp_index = findall(x -> x==0, all(mapslices(x-> all(x.==1.0),X,dims=1)))
        X = X[:,tmp_index]
    end

    n::Int64,k::Int64 = size_VecOrMat(X)

    S = zeros(Float64,n,k)

    if sigma==0.0
        _, sigma = scaletau2(y)
    end

    if sigma < 1e-10
        sigma = 1e-10
    end

    global_crit::Float64 = 10*epsilon_global
    global_iter::Int64 = 0

    # fit
    while (global_crit > epsilon_global) & (global_iter < max_it_global)
        crit = 10*epsilon
        iter = 0
        # update f
        while (crit > epsilon) & (iter < max_it)
            S_tmp = copy(S)

            if k==1
                S = rloess(y .- alpha,X,X,w,sigma=sigma,span=span[1],degree=loess_degree[1],type=type,c=c,
                max_it=max_it_loess,epsilon=epsilon_loess)
            else
                for j in 1:k
                    S[:,j] = rloess(y .- alpha .- vec(sum(dropcol(S,j),dims=2)),X[:,j],X[:,j],
                    w,sigma=sigma,span=span[j],degree=loess_degree[j],type=type,c=c,
                    max_it=max_it_loess,epsilon=epsilon_loess)
                end
            end
            # centering
            S = S .- mean(S,dims=1)

            # update the intercept
            if update_intercept
                alpha_tmp = copy(alpha)
                alpha = robust_intercept(y .- vec(sum(S,dims=2)), alpha_tmp,w,
                sigma = sigma,type=type,c=c,epsilon=epsilon_loess,max_it=max_it_loess)
                crit = compute_crit(S,S_tmp)/n + compute_crit([alpha],[alpha_tmp])
            else
                crit = compute_crit(S,S_tmp)/n
            end

            # update crit & iter
            iter += 1
        end

        # update sigma
        tmp_sigma = copy(sigma)
        _, sigma = scaletau2(y .- alpha .- vec(sum(S,dims=2)))

        global_crit = copy(crit + compute_crit([sigma],[tmp_sigma]))
        global_iter += 1
    end

    return (alpha=alpha, S=S, sigma=simga)
end

# implement RGAPLM for Poisson and NB
function RGAPLM(y::type_VecFloatInt,X::type_NTVecOrMatFloatInt,T::type_NTVecOrMatFloatInt;
    family::St ="NB", method::St = "Pan", link::St="log", verbose=true,
    span::type_VecFloatInt,loess_degree::Vector{N},
    sigma::type_FloatInt= 1.0,
    beta::Union{Nothing,Float64,Int64,Vector{Float64},Vector{Int64}}=nothing,
    c::U=1.345, robust_type::St ="Tukey",
    c_X::U=1.345, robust_type_X::St ="Tukey",
    c_T::U=1.345, robust_type_T::St ="Tukey",
    epsilon::U=1e-6, max_it::N = 100,
    epsilon_T::U = 1e-6, max_it_T::N = 5,
    epsilon_X::U = 1e-6, maix_it_X::N = 5) where {U <: type_FloatInt, N <: Int64, St <: String}

    # initialization
    n::Int64 = length(y)
    _, p::Int64 = size_VecOrMat(X)
    _, q::Int64 = size_VecOrMat(T)
    crit::Float64 = copy(epsilon)
    crit_T::Float64 = copy(epsilon)
    crit_X::Float64 = copy(epsilon)
    iter::Int64 = 0
    iter_X::Int64 = 0
    iter_T::Int64 = 0

    if p ==0 & q == 0
        error("No data is provided.")
    end

    eta = similar(y)
    mu = similar(y)
    z = similar(y)
    w = similar(y)

    if p != 0
        if beta == nothing
            beta = zeros(p)
            # use the median for the initial estimate
            beta[1] = g_link(family,median(y),link=link)
        else
            if p != length(beta)
                error("The number of initial beta values does not match the number of variables in X.")
            end
        end
    end

    if q != 0
        S::type_VecOrMatFloatInt = zeros(n,q)
    end

    # assume p and q are not zero !
    if family == "P"
        # mean belongs to X
        if method == "Pan"

            # initialization
            par = vec(X * beta)
            eta = copy(par)
            mu = g_invlink(family,eta)
            s = sqrt.(g_var(family,mu,sigma))
            z, w = g_ZW(family,robust_type,y,mu,s,c,sigma)
            par_res = vec(z .- par)

            crit =  epsilon + 1
            iter = 0

            # start the loop
            while (crit > epsilon) & (iter < max_it)
                # estimate S
                tmp_S = copy(S)
                S = AM(par_res,T,w,span=span,loess_degree=loess_degree,epsilon=epsilon_T,max_it=max_it_T,y_mean=beta[1]).S

                nonpar = sum(S,dims=2)
                # estimate beta
                tmp_beta = copy(beta)
                beta = wls(vec(z .- nonpar),X,w)
                par = X * beta

                # update eta, mu, Z, W
                eta = vec(par .+ nonpar)
                mu = g_invlink(family,eta)
                s = sqrt.(g_var(family,mu,sigma))
                z,w = g_ZW(family,robust_type,y,mu,s,c,sigma)
                par_res = vec(z .- par)
                crit = compute_crit(S,tmp_S,"norm2_change") .+ compute_crit(beta,tmp_beta,"norm2_change")
                iter += 1
            end
            # end of Pan Poisson
            if verbose==1
                @sprintf("Pan Poisson fitted with %.0f iterations and converged with %.2E",iter,crit)
            end

            return (eta=eta,mu=mu,z=z,w=w,beta=beta,S=S,s=s)

        elseif method == "Lee"
            # initialization
            # non-robust !
            par = vec(X * beta)
            eta = copy(par)
            mu = g_invlink(family,eta)
            s = sqrt.(g_var(family,mu,sigma))
            z, w = g_ZW(family,"none",y,mu,s,c,sigma)
            robust_z = similar(z)
            robust_w = similar(w)
            par_res = vec(z .- par)

            crit =  epsilon + 1
            iter = 0

            # start the loop
            while (crit > epsilon) & (iter < max_it)
                # estimate S robustly
                tmp_S = copy(S)
                S = RAM(par_res,T,w,span=span,loess_degree=loess_degree,epsilon=epsilon_T,max_it=max_it_T,
                max_it_loess=10,epsilon_loess::type_FloatInt= 1e-6,type=robust_type_T,
                c=c_T,sigma=1.0, alpha=0.0,update_intercept=false,
                max_it_global=1).S

                nonpar = sum(S,dims=2)

                # construct robust Z,W to estimate beta
                while (crit_X > epsilon_X) & (iter_X < maix_it_X)
                    mu = g_invlink(family,eta= nonpar .+ par ;link=link)
                    s = sqrt.(g_var(family,mu,sigma))
                    robust_z, robust_w = g_ZW(family,robust_type,y,mu,s,c,sigma)
                    tmp_beta = copy(beta)
                    beta = wls(vec(robust_z .- nonpar),X,robust_w)
                    par = X * beta
                    crit_X = compute_crit(beta,tmp_beta,"norm2_change")
                    iter_X += 1
                end

                # update eta, mu, Z, W
                eta = vec(par .+ nonpar)
                mu = g_invlink(family,eta)
                s = sqrt.(g_var(family,mu,sigma))
                z, w = g_ZW(family,"none",y,mu,s,c,sigma)
                par_res = vec(z .- par)
                crit = compute_crit(S,tmp_S,"norm2_change") .+ compute_crit(beta,tmp_beta,"norm2_change")
                iter += 1
            end
            # end of Pan Poisson
            if verbose==1
                @printf("Lee Poisson fitted with %.0f iterations and converged with %.2E",iter,crit)
            end

            return (eta=eta,mu=mu,z=z,w=w,beta=beta,S=S,s=s)
        else
            error("Method $method is not supproted.")
        end # end of family Poisson

    elseif family =="NB"

        if method == "Pan"

        elseif method =="Lee"

        else
            error("Method $method is not supported.")
        end

    else
        error("Family $family is not supported.")
    end

    # S, beta, sigma

    # get the likelihood; convergence in likelihood!

    # write three versions: X, T, X & T

    # global while loop

        # local loop sigma

        # local loop beta + S
            # local loop beta
            # local loop S

    # return beta, S, mu, eta, sigma in dic format
    (a=3, b="hi")
end
