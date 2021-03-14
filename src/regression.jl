# using BenchmarkTools
# using LinearAlgebra
# using RDatasets
# using Distributions, StatsFuns, SpecialFunctions
# using Roots
# using Debugger
# using Plots
# using Printf
# include("utils.jl")

# weighted least squares
function wls(y::type_VecFloatInt,
    X::type_VecOrMatFloatInt,
    w::type_VecRealFloatInt=ones(length(y)),
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
    degree::Int=2)

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
    iter::Int =0
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
    iter::Int = 0
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

    iter::Int = 0
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
    update_intercept::Bool = true) where {U <: Float64, N<:Int}

    # initialization
    # if there is a vector of ones (intercept), exclude it
    # intercept is updated separately
    if any(mapslices(x-> all(x.==1.0),X,dims=1))
        tmp_index = findall(x -> x==0, all(mapslices(x-> all(x.==1.0),X,dims=1)))
        X = X[:,tmp_index]
    end

    n::Int,k::Int = size_VecOrMat(X)

    S = zeros(Float64,n,k)

    if sigma==0.0
        _, sigma = scaletau2(y)
    end

    if sigma < 1e-10
        sigma = 1e-10
    end

    global_crit::Float64 = 10*epsilon_global
    global_iter::Int = 0

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

    return (alpha=alpha, S=S, sigma=sigma)
end

# RGAPLM for Poisson and NB
function rGAPLM(y::type_VecFloatInt,X::type_NTVecOrMatFloatInt,T::type_NTVecOrMatFloatInt;
    family::String ="NB", method::String = "Pan", link::String="log", verbose=true,
    span::type_VecFloatInt,loess_degree::Vector{Int},
    sigma::type_FloatInt= 1.0,
    beta::Union{Nothing,Float64,Int,Vector{Float64},Vector{Int}}=nothing,
    c::type_FloatInt=1.345, robust_type::String ="Tukey",
    c_X::type_FloatInt=1.345, robust_type_X::String ="Tukey",
    c_T::type_FloatInt=1.345, robust_type_T::String ="Tukey",
    c_sigma::type_FloatInt=1.345, robust_type_c::String = "Tukey",
    epsilon::type_FloatInt=1e-6, max_it::Int = 100,
    epsilon_T::type_FloatInt = 1e-6, max_it_T::Int = 5,
    epsilon_X::type_FloatInt = 1e-6, max_it_X::Int = 5,
    epsilon_RAM::type_FloatInt = 1e-6, max_it_RAM::Int = 10,
    epsilon_sigma::type_FloatInt=1e-4, max_it_sigma::Int = 10,
    epsilon_eta::type_FloatInt=1e-4, max_it_eta::Int = 25,
    initial_beta::Bool = false, maxmu::type_FloatInt=1e5,
    minmu::type_FloatInt=1e-10,min_sigma::type_FloatInt = 0.1,max_sigma::type_FloatInt = 10)
 # where {U <: type_FloatInt, N <: Int, St <: String}

    # initialization
    n::Int = length(y)
    _, p::Int = size_VecOrMat(X)
    _, q::Int = size_VecOrMat(T)
    crit::Float64 = copy(epsilon)
    crit_T::Float64 = copy(epsilon)
    crit_X::Float64 = copy(epsilon)
    iter::Int = 0
    iter_X::Int = 0
    iter_T::Int = 0

    if p ==0 & q == 0
        error("No data is provided.")
    end

    eta = similar(y)
    mu = similar(y)
    z = similar(y)
    w = similar(y)
    alpha::Float64 = 0.0

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

                nonpar = vec(sum(S,dims=2))
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
                crit = compute_crit(S,tmp_S,"norm2_change") + compute_crit(beta,tmp_beta,"norm2_change")
                iter += 1
            end
            # end of Pan Poisson
            if verbose==1
                @printf("Pan Poisson fitted with %.0f iterations and converged with %.2E",iter,crit)
            end

            return (eta=eta,mu=mu,z=z,w=w,beta=beta,S=S,s=s,convergence=Dict(:iter => iter, :crit => crit))

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
            tmp_beta = copy(beta)

            crit =  epsilon + 1
            iter = 0

            # start the loop
            while (crit > epsilon) & (iter < max_it)
                # estimate S robustly
                tmp_S = copy(S)
                alpha, S, _ = RAM(par_res,T,w,span=span,loess_degree=loess_degree,epsilon=epsilon_T,max_it=max_it_T,
                max_it_loess=max_it_RAM,epsilon_loess= epsilon_RAM,type=robust_type_T,
                c=c_T,sigma=1.0, alpha=0.0,update_intercept=false,
                max_it_global=1)

                nonpar = vec(sum(S,dims=2))
                # beta[1] = beta[1] + alpha
                # par = X * beta

                # construct robust Z,W to estimate beta
                crit_X = 10 * epsilon_X
                iter_X = 0
                while (crit_X > epsilon_X) & (iter_X < max_it_X)
                    mu = g_invlink(family,nonpar .+ par ;link=link)
                    s = sqrt.(g_var(family,mu,sigma))
                    robust_z, robust_w = g_ZW(family,robust_type,y,mu,s,c_X,sigma)
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
                if verbose ==1
                    @show(beta)
                    @printf("%.0f iter with %.2E crit \n",iter,crit)
                end
            end
            # end of Pan Poisson
            if verbose
                @printf("Lee Poisson fitted with %.0f iterations and converged with %.2E",iter,crit)
            end

            return (eta=eta,mu=mu,z=z,w=w,beta=beta,S=S,s=s,convergence=Dict(:iter => iter, :crit => crit))
        else
            error("Method $method is not supproted.")
        end # end of family Poisson

    elseif family =="NB"
        # vectorized; could be more computationally efficient
        # GLM components

        # initial mu computed through Poisson GLM
        # replace this with my GRBF
        if !initial_beta
            # ML estimations of mu and sig in an alternating fashion
            full_loglkhd1 = 0
            full_loglkhd0 = full_loglkhd1+epsilon+1
            iter = 0

            # outer loop
            while (abs(full_loglkhd1 - full_loglkhd0) > epsilon) & (iter < 5)

                # estimate mu
                GAPLM_MLE = RGAPLM(y,X,T,
                    family="P", method= "Pan", link="log", verbose=false,
                    span=span,loess_degree=loess_degree,
                    sigma= sigma,
                    beta=beta,
                    c=500, robust_type ="Huber",
                    c_X=500, robust_type_X ="Huber",
                    c_T=500, robust_type_T ="Huber",
                    epsilon=epsilon, max_it = 5)


                beta = copy(GAPLM_MLE.beta)
                mu = copy(GAPLM_MLE.mu)
                mu[mu .> maxmu] .= maxmu
                mu[mu .< minmu] .= minmu

                # sig MLE based on initial mu, with starting value = moment based
                sigma = mean((y ./ mu .- 1) .^2)
                loglkhd_sig1 = 0
                loglkhd_sig0 = loglkhd_sig1 + epsilon_sigma +1
                iter_sigma = 0

                # Newton Raphson to calc
                while (abs(loglkhd_sig1 - loglkhd_sig0) > epsilon_sigma) & (iter_sigma < max_it_sigma)
                    sigma = abs(sigma - score_sigma(y,mu,sigma)/info_sigma(y,mu,sigma))
                    if sigma > max_sigma
                        sigma = max_sigma
                    end
                    loglkhd_sig0 = copy(loglkhd_sig1)
                    loglkhd_sig1 = ll(y,mu,float(sigma))
                    iter_sigma += 1
                end

                # update
                full_loglkhd0 = full_loglkhd1
                full_loglkhd1 = ll(y,mu,sigma)
                iter += 1
            end # if initial values are provided
        #TODO: Check whether I need this beta[1] = beta[1] + nonpar$alpha
        else  # use the supplied initial values;
            mu = g_invlink(family,X * beta;link=link)
            mu[mu .> maxmu] .= maxmu
            mu[mu .< minmu] .= minmu
        end

        if verbose
            print("Parameter initialization completed.\n
            Initial parameter values are:")
            @show sigma
            @show beta
            @show mu[1:5]
        end

        # Start estimation
        if method == "Pan"

            # initialization
            s = sqrt.(g_var(family,mu,sigma))
            z, w = g_ZW(family,robust_type,y,mu,s,c,sigma)

            crit =  epsilon + 1
            iter = 0
            tmp_sigma = copy(sigma)
            tmp_eta = similar(eta)
            tmp_beta = similar(beta)
            tmp_S = similar(S)


            if verbose
                println("Starting the loop . . .")
            end
            # start the loop
            while (crit > epsilon) & (iter < max_it)
                # estimate sigma
                tmp_sigma = copy(sigma)
                robust_sigma_uniroot = (xx -> robust_sigma(y,mu,xx,c_sigma;type=robust_type_c,family=family))
                roots = Roots.find_zero(robust_sigma_uniroot,(min_sigma,max_sigma),Roots.Bisection(),
                verbose=false,tol=epsilon_sigma,maxevals=max_it_sigma)
                #roots = Roots.find_zero(robust_sigma_uniroot,min_sigma,max_sigma,verbose=verbose,xrtol=epsilon_sigma)
                if length(roots) == 0
                    # do not update
                    @warn("Roots not found for sigma (current value: $sigma)")
                elseif length(roots) == 1
                    #  update sigma
                    sigma = float(roots[1])
                else
                    # likely multipel roots close to zero
                    roots = Roots.find_zeros(robust_sigma_uniroot,0.1,max_sigma,verose=true)
                    if length(roots) == 0
                        sigma = 0.1
                    else
                        sigma = float(roots[1])
                    end
                end

                # update robust components involving sigma
                s = sqrt.(g_var(family,mu,sigma))
                z, w = g_ZW(family,robust_type,y,mu,s,c,sigma)
                par = X * beta
                par_res = vec(z .- par)

                # estimate eta (beta + f(T))
                tmp_eta = copy(eta)
                iter_eta = 0
                crit_eta = epsilon_eta + 1.0

                while (crit_eta > epsilon_eta) & (iter_eta < max_it_eta)
                    tmp_S = copy(S)
                    S = AM(par_res,T,w,span=span,loess_degree=loess_degree,epsilon=epsilon_T,max_it=max_it_T,y_mean=beta[1]).S

                    nonpar = vec(sum(S,dims=2))
                    # estimate beta
                    tmp_beta = copy(beta)
                    beta = wls(vec(z .- nonpar),X,w)
                    par = X * beta

                    # update eta, mu, Z, W
                    eta = vec(par .+ nonpar)
                    mu = g_invlink(family,eta)
                    mu[mu .> maxmu] .= maxmu
                    mu[mu .< minmu] .= minmu
                    s = sqrt.(g_var(family,mu,sigma))
                    z,w = g_ZW(family,robust_type,y,mu,s,c,sigma)
                    par = X * beta
                    par_res = vec(z .- par)

                    crit_eta = (compute_crit(S,tmp_S,"norm2_change") + compute_crit(beta,tmp_beta,"norm2_change"))/2
                    iter_eta += 1
                end

                # compute sigma crit and eta crit
                crit = max(abs(tmp_sigma - sigma),compute_crit(eta,tmp_eta,"norm_inf"))
                iter += 1

                if verbose
                    @printf("Iter %.0f; Crit %.2E; beta0: %.2E; beta1:  %.2E; sigma: %.2E \n",iter,crit,beta[1],beta[2],sigma)
                    @show beta
                end
            end
            # end of estimation
            if verbose
                @printf("Pan NB fitted with %.0f iterations and converged with %.2E",iter,crit)
            end

            return (eta=eta,mu=mu,z=z,w=w,beta=beta,sigma=sigma,S=S,s=s,convergence=Dict(:iter => iter, :crit => crit))

        elseif method=="Lee"

            # initialization
            s = sqrt.(g_var(family,mu,sigma))
            z, w = g_ZW(family,robust_type,y,mu,s,c,sigma)

            crit =  epsilon + 1
            iter = 0
            tmp_sigma = copy(sigma)
            tmp_eta = similar(eta)
            tmp_beta = similar(beta)
            tmp_S = similar(S)


            if verbose
                println("Starting the loop . . .")
            end

            # start the loop
            while (crit > epsilon) & (iter < max_it)
                # estimate sigma
                tmp_sigma = copy(sigma)
                robust_sigma_uniroot = (xx -> robust_sigma(y,mu,xx,c_sigma;type=robust_type_c,family=family))
                roots = Roots.find_zero(robust_sigma_uniroot,(min_sigma,max_sigma),Roots.Bisection(),
                verbose=false,tol=epsilon_sigma,maxevals=max_it_sigma)
                #roots = Roots.find_zero(robust_sigma_uniroot,min_sigma,max_sigma,verbose=verbose,xrtol=epsilon_sigma)
                if length(roots) == 0
                    # do not update
                    @warn("Roots not found for sigma (current value: $sigma)")
                elseif length(roots) == 1
                    #  update sigma
                    sigma = float(roots[1])
                else
                    # likely multipel roots close to zero
                    roots = Roots.find_zeros(robust_sigma_uniroot,0.1,max_sigma,verose=true)
                    if length(roots) == 0
                        sigma = 0.1
                    else
                        sigma = float(roots[1])
                    end
                end

                # update components involving sigma
                s = sqrt.(g_var(family,mu,sigma))
                z, w = g_ZW(family,"none",y,mu,s,c,sigma)
                robust_z = similar(z)
                robust_w = similar(w)
                par = X * beta
                par_res = vec(z .- par)
                tmp_beta = copy(beta)

                # initalize paramters for updating η
                tmp_eta = copy(eta)
                iter_eta = 0
                crit_eta = epsilon_eta + 1.0

                # update η
                while (crit_eta > epsilon_eta) & (iter_eta < max_it_eta)
                    # update f(T) using
                    # non-robust components with robust univariate smoothers
                    tmp_S = copy(S)

                    alpha, S, _ = RAM(par_res,T,w,span=span,loess_degree=loess_degree,epsilon=epsilon_T,max_it=max_it_T,
                    max_it_loess=max_it_RAM,epsilon_loess= epsilon_RAM,type=robust_type_T,
                    c=c_T,sigma=sigma, alpha=0.0,update_intercept=false,
                    max_it_global=1)

                    nonpar = vec(sum(S,dims=2))
                    # beta[1] = beta[1] + alpha
                    # par = X * beta

                    # estimate beta
                    crit_X = 10 * epsilon_X
                    iter_X = 0
                    while (crit_X > epsilon_X) & (iter_X < max_it_X)
                        mu = g_invlink(family,nonpar .+ par ;link=link)
                        # prevent overflow/underflow
                        mu[mu .> maxmu] .= maxmu
                        mu[mu .< minmu] .= minmu
                        s = sqrt.(g_var(family,mu,sigma))
                        robust_z, robust_w = g_ZW(family,robust_type,y,mu,s,c_X,sigma)
                        tmp_beta = copy(beta)
                        beta = wls(vec(robust_z .- nonpar),X,robust_w)
                        par = X * beta
                        crit_X = compute_crit(beta,tmp_beta,"norm_inf")
                        iter_X += 1
                    end

                    tmp_beta = copy(beta)
                    beta = wls(vec(z .- nonpar),X,w)
                    par = X * beta

                    # update eta, mu, Z, W
                    tmp_eta = copy(eta)
                    eta = vec(par .+ nonpar)
                    mu = g_invlink(family,eta)
                    mu[mu .> maxmu] .= maxmu
                    mu[mu .< minmu] .= minmu
                    s = sqrt.(g_var(family,mu,sigma))
                    z,w = g_ZW(family,"none",y,mu,s,c,sigma)
                    par_res = vec(z .- par)

                    crit_eta = max(compute_crit(S,tmp_S,"norm_inf"),compute_crit(beta,tmp_beta,"norm_inf"))
                    iter_eta += 1
                end

                # compute sigma crit and eta crit
                crit = max(abs(tmp_sigma - sigma),compute_crit(eta,tmp_eta,"norm_inf"))
                iter += 1

                if verbose
                    @printf("Iter %.0f; Crit %.2E; beta0: %.2E; beta1:  %.2E; sigma: %.2E \n",iter,crit,beta[1],beta[2],sigma)
                    @show beta
                end
            end
            # end of estimation
            if verbose
                @printf("Lee NB fitted with %.0f iterations and converged with %.2E",iter,crit)
            end

            return (eta=eta,mu=mu,z=z,w=w,beta=beta,sigma=sigma,S=S,s=s,convergence=Dict(:iter => iter, :crit => crit))
        else
            error("Method $method is not supported.")
        end

    else
        error("Family $family is not supported.")
    end
    # end of RGAPLM
end
