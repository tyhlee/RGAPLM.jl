using BenchmarkTools
using LinearAlgebra
using RDatasets
using Distributions, StatsFuns
using Debugger
using Plots
include("utils.jl")

# weighted least squares
function wls(y::Vector{T},
    X::VecOrMat{T},
    w::Vector{R}=ones(length(y)),
    method=:qr) where {T <: Float64, R <: Real}

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
function loess(y::Vector{T},
    x::Vector{T},
    xi::Vector{T},
    w::Vector{T}=ones(length(y));
    span::T=0.75,
    degree::Integer=2) where T <: Float64

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
function rloess(y::Vector{T},
    x::Vector{T},
    xi::Vector{T},
    w::Vector{T}=ones(length(y));
    sigma::T=1.0,
    span::T=0.75,
    degree::U=2,
    c::T = 1.345,
    type::String="Huber",
    epsilon::T = 1e-6,
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
function AM(y::Vector{U},X::VecOrMat{U},
    w::Vector{U}=ones(length(y));
    span::Vector{U},loess_degree::Vector{N},
    epsilon::U=1e-6,max_it::N=25, y_mean::U = mean(y)) where {U <: Float64, N<:Int}

    # initialization
    # if there is a vector of ones (intercept), exclude it
    if any(mapslices(x-> all(x.==1.0),X,dims=1))
        tmp_index = findall(x -> x==0, all(mapslices(x-> all(x.==1.0),X,dims=1)))
        X = X[:,tmp_index]
    end

    n,k = size_VecOrMat(X)

    crit::U = 10*epsilon
    iter::N = 0
    S = zeros(U,n,k)

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
    return y_mean,S
end

# helper function for RAM
function robust_intercept(y::Vector{T},mu::T=median(y),w::Vector{T}=ones(length(y));sigma::T=1.0,
    type::String="Huber",c::T=1.345,epsilon::T=1e-6,max_it::N=10) where{T<: Float64, N <:Int}

    iter::Int = 0
    crit::T = epsilon*10
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
function RAM(y::Vector{U},X::VecOrMat{U},w::Vector{U}=ones(length(y));
    span::Vector{U},loess_degree::Vector{U},
    max_it_loess::N = 10, epsilon_loess::U= 1e-6,
    type::String="Huber",c::U = 1.345,
    sigma::U = 0.0, epsilon::U=1e-6,max_it::N=10, alpha::U = median(y),
    epsilon_global = 1e-6, max_it_global = 5) where {U <: Float64, N<:Int}

    # initialization
    # if there is a vector of ones (intercept), exclude it
    # intercept is updated separately
    if any(mapslices(x-> all(x.==1.0),X,dims=1))
        tmp_index = findall(x -> x==0, all(mapslices(x-> all(x.==1.0),X,dims=1)))
        X = X[:,tmp_index]
    end

    n,k = size_VecOrMat(X)

    S = zeros(U,n,k)

    if sigma==0.0
        _, sigma = scaletau2(y)
    end

    if sigma < 1e-10
        sigma = 1e-10
    end

    global_crit::U = 10*epsilon_global
    global_iter::N = 0

    # fit
    while (global_crit > epsilon_global) & (global_iter < max_it_global)
        crit = 10*epsilon
        iter = 0
        # update f
        while (crit > epsilon) & (iter < max_it)
            S_tmp = copy(S)
            alpha_tmp = copy(alpha)

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
            alpha = robust_intercept(y .- vec(sum(S,dims=2)), alpha_tmp,w,
            sigma = sigma,type=type,c=c,epsilon=epsilon_loess,max_it=max_it_loess)
            println("alpha: ",alpha)

            # update crit & iter
            iter += 1
            crit = compute_crit(S,S_tmp)/n + compute_crit([alpha],[alpha_tmp])
        end

        # update sigma
        tmp_sigma = copy(sigma)
        _, sigma = scaletau2(y .- alpha .- vec(sum(S,dims=2)))

        global_crit = copy(crit + compute_crit([sigma],[tmp_sigma]))
        global_iter += 1
        println(global_iter,"th sigma:",sigma)
        println(global_iter,"th Global Crit:",global_crit)
    end

    return alpha, S, sigma
end

# implement RGAPLM for Normal, Poisson, NB
function RGAPLM(y::Vector{U},X::Union{NT,VecOrMat{U}},T::Union{NT,VecOrMat{U}};
    span::Vector{U},degree::Vector{N},
    c::U=1.34.5, robust_type::St ="Tukey",
    c_X::U=1.345, robust_type_X::St ="Tukey",
    c_T::U=1.345, robust_type_T::St ="Tukey",
    epsilon::U=1e-6, max_it::N = 100,
    epsilon_T::U = 1e-6, max_it_T::N = 5,
    epsilon_X::U = 1e-6, maix_it_X::N = 5) where {U <: Float64, N <: Int, St <: String, NT <: Nothing}

    # initialization
    n::N = length(y)
    _, p::N = size_VecOrMat(X)
    _, q::N = size_VecOrMat(T)
    # S, beta, sigma

    # get the likelihood; convergence in likelihood!

    # global while loop

        # local loop sigma

        # local loop beta + S
            # local loop beta
            # local loop S

    # return beta, S, mu, eta, sigma in dic format

end

function f1(y::Vector{U},X::VecOrMat{U}) where {U <: Float64}
    return size_VecOrMat(X)
end
