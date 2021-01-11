function tricube(u::Vector{Float64})
    map(x -> abs(x) <= 1.0 ? (1-x^3)^3 : 0,u)
    # (1 .- u.^3).^3
end

# create the design matrix
function outer_power(u::Vector{Float64},pow_degree::Integer)
    tmp = Array{Float64,2}(undef,length(u),pow_degree+1)
    for i in 0:pow_degree
        tmp[:,i+1] = u.^i
    end
    tmp
end

function compute_kernel_weight(x::Vector{Float64}, span::Float64)
    tricube(x./sort(abs.(x))[ceil(Int,span*length(x))])
end

function rho(r::Union{T,Array{T}}, c::T=1.345,type::String="Huber") where T <: Real
    if type=="Huber"
        return map(t -> abs(t)<=c ? t^2 : 2*c*abs(t)-c^2,r)
    elseif type =="Tukey"
        return map(t -> abs(t)<=c ? 1-(1-(t/c)^2)^3 : 1,r)
    else
        error("Unknown type")
    end
end

function psi(r::Union{T,Array{T}}, c::T,type::String="Huber") where T <: Real
    if type=="Huber"
        return map(t -> abs(t)<=c ? t : sign(t)*c,r)
    elseif type =="Tukey"
        return map(t -> abs(t)<=c ? t*(1-(t/c)^2)^2 : 0,r)
    else
        throw(error("Unknown type"))
    end
end

function psi_weight(r::Union{T,Array{T}}, c::T,type::String="Huber") where T <: Real
    if type=="Huber"
        return map(t -> abs(t)<=c ? 1 : c/abs(t),r)
    elseif type =="Tukey"
        return map(t -> abs(t)<=c ? (1-(t/c)^2)^2 : 0,r)
    else
        throw(error("Unknown type"))
    end
end

function compute_crit(X::VecOrMat{T},XX::VecOrMat{T},type="norm2") where T <: Real
    if type=="norm2"
        return norm(sum(X .- XX,dims=2))
    elseif type=="norm2_change"
        return norm(X-XX)/norm(XX)
    else
        error("Compute crit type is not supported")
    end
end

dropcol(M::AbstractMatrix,ind) = M[:,deleteat!(collect(axes(M,2)),ind)]

# REF: R's robustbase::scaeltau2
function scaletau2(x::Vector{T};c1::T=4.5,c2::T=3.0,consistency=true,
    sigma::T=1.0) where T<:Float64

    n::Int64 = length(x)
    medx::Float64 = median(x)
    xx = abs.(x .- medx)
    sigma = median(xx)
    if sigma <= 0
        return medx, 0
    end

    w = similar(xx)
    mu::T = 0.0

    if c1 > 0
        xx = xx ./ (sigma * c1)
        w = 1 .- xx.^2
        w = ((abs.(w) .+ w) ./2).^2
        mu = sum(x .* w) / sum(w)
    else
        mu = medx
    end

    x = (x .- mu) ./ sigma
    w = x.^2
    w[ w .> c2^2 ] .= c2^2
    if consistency
        input::T = c2*quantile(Normal(0,1),3/4)
        nEs2::T = n * (2 * ((1-input^2) * cdf(Normal(0,1),input) - input * pdf(Normal(0,1),input) + input^2) - 1)
    else
        nEs2 = n
    end
    return mu, sigma * sqrt(sum(w)/nEs2)
end

# determine the size[2] of a vector or matrix
function size_VecOrMat(x::Union{Nothing,VecOrMat{Number}})
    if typeof(x)== Nothing
        return 0,0
    elseif typeof(x) == Array{typeof(x[1]),1}
        return length(x), 1
    else
        return size(x)
    end
end

# neg binom helper: cdf and pmf
function nb2_cdf_mu_sigma(mu::T,sigma::T,y::T) where {T <: Union{Int64, Float64,Vector{Float64},Vector{Int64}}}
        StatsFuns.nbinomcdf.(1 ./ sigma, 1 ./ (sigma .* mu .+ 1),y)
end

# it is really pmf; use pdf for consistency with Distributions pkg
function nb2_pdf_mu_sigma(mu::T,sigma::T,y::T) where {T <: Union{Int64, Float64,Vector{Float64},Vector{Int64}}}
        StatsFuns.nbinompdf.(1 ./ sigma, 1 ./ (sigma .* mu .+ 1),y)
end

# GLM fam link functions
function g_link(family::String,mu::Union{Vector{Int64},Vector{Float64}};link::String="log")
    if link == "log"
        return log.(mu)
    else
        erro("$link is not supported.")
    end
end

function g_invlink(family::String,eta::Union{Vector{Int64},Vector{Float64}};link::String="log")
    if link == "log"
        return exp.(eta)
    else
        erro("$link is not supported.")
    end
end

function g_derlink(family::String,mu::Union{Vector{Int64},Vector{Float64}};link::String="log")
    if link == "log"
        return 1 ./ mu
    else
        erro("$link is not supported.")
    end
end

function g_weight(family::String,mu::Union{Vector{Int64},Vector{Float64}};link::String="log")
    if link == "log"
        log.(mu)
    else
        erro("$link is not supported.")
    end
end

function g_var(family::String,mu::Union{Vector{Int64},Vector{Float64}},sigma::Union{Nothing,Int64,Float64}=nothing)
    if link == "log"
        return mu
    else
        erro("$link is not supported.")
    end
end

function g_ZW(family::String,robust_type::String,y::Union{Vector{Int64},Vector{Float64}},mu::Union{Vector{Int64},Vector{Float64}},s::Vector{Float64},c::Float64,sigma::Union{Nothing,Float64,Int64})

    E1 = similar(y)
    E2 = similar(y)
    musq::Vector{Float64} = mu .^ 2
    nb_r = (typeof(sigma)==Nothing, nothing, 1/sigma)
    nb_p = similar(y)

    j1::Vector{Int} = max.(ceil.(Int,mu .- c .* s),0)
    j2::Vector{Int} = floor.(Int,mu .+ c .* s)
    zero_index = j1 .> j2
    # make this into floor
    j1 .-= 1

    if robust_type =="Huber"

        if family=="P"

            tmp = Poisson.(mu)
            # E(psi(r;c))
            E1 = c .* ((1 .- cdf.(tmp,j2)) .- cdf.(tmp,j1)) .+ mu ./ s .* (pdf.(tmp,j1) .- pdf.(tmp,j2))
            # E(r psi(r;c))
            E2 = (c .* mu) .* (pdf.(tmp,j1) .+ pdf.(tmp,j2)) .+
            1 ./ s .* ( s .^ 2 .* (cdf.(tmp,j2 .- 2) .- cdf.(tmp,j1 .- 2)) .+
            (musq .+ mu) .* (pdf.(tmp,j1 .- 1) .- pdf.(tmp,j2 .- 1)) .-
            musq .* (pdf.(tmp,j1) .- pdf.(tmp,j2)) )
            E2 = E2 ./ s


        elseif family=="NB"
            nb_r = 1 / sigma
            nb_p = 1 ./ (sigma .* mu .+ 1)
            E1 =  c .* ((1 .- StatsFuns.nbinomcdf.(nb_r,nb_p,j2)) .- StatsFuns.nbinomcdf.(nb_r,nb_p,j1)) .+ mu ./ s .*
            (StatsFuns.nbinompdf.(nb_r,nb_p,j1) .* (sigma .* j1 .+ 1) .-
             StatsFuns.nbinompdf.(nb_r,nb_p,j2) .* (sigma .* j2 .+ 1))
            E2 = (c .* mu) .*
            (StatsFuns.nbinompdf.(nb_r,nb_p,j1) .* (sigma .* j2  .+ 1) .+ StatsFuns.nbinompdf.(nb_r,nb_p,j2) .* (sigma .* j1 .+1)) .+
            1 ./ s .* ( s .^ 2 .* (StatsFuns.nbinomcdf.(nb_r,nb_p,j2 .- 2) .- StatsFuns.nbinomcdf.(nb_r,nb_p,j1 .- 2)) .+
            StatsFuns.nbinompdf.(nb_r,nb_p,j1) .* (sigma .* mu .* j1 .^ 2 .- (2*sigma) .* musq .* j1 .- musq) .+
            StatsFuns.nbinompdf.(nb_r,nb_p,j1 .- 1) .* ( sigma .* musq .* (sigma+1) .* (j1 .- 1) .- mu .+ musq) .-
            StatsFuns.nbinompdf.(nb_r,nb_p,j2) .* ( sigma .* mu .* j2 .^ 2 .- (2*sigma) .* musq .* j2 .- musq) .-
            StatsFuns.nbinompdf.(nb_r,nb_p,j2 .- 1) .* ( sigma .* mu .^ 2 .* (sigma+1) .* (j2 .- 1) .- mu .+ mu .^ 2))
            E2 = E2 ./ s

        else
            error("$family is not supported.")
        end

    elseif robust_type =="Tukey"

        if family=="P"

            tmp = Poisson.(mu)
            # E(psi(r;c))
            E1 = map(i -> sum(map(k -> psi((k-mu[i])/s[i],c,robust_type) * pdf(tmp[i],k),collect((j1[i]+1):j2[i]))),collect(1:length(y)))
            # E(r psi(r;c))
            E2 = map(i -> sum(map(k -> psi((k-mu[i])/s[i],c) * (k-mu[i]) * pdf(tmp[i],k),collect((j1[i]+1):j2[i]))),collect(1:length(y)))
            E2 = E2 ./ s


        elseif family=="NB"
            nb_r = 1 / sigma
            nb_p = 1 ./ (sigma .* mu .+ 1)
            # E(psi(r;c))
            E1 = map(i -> sum(map(k -> psi((k-mu[i])/s[i],c) * StatsFuns.nbinompdf(nb_r,nb_p[i],k),collect((j1[i]+1):j2[i]))),collect(1:length(y)))
            # E(r psi(r;c))
            E2 = map(i -> sum(map(k -> psi((k-mu[i])/s[i],c) * (k-mu[i]) * StatsFuns.nbinompdf(nb_r,nb_p[i],k),collect((j1[i]+1):j2[i]))),collect(1:length(y)))
            E2 = E2 ./ s

        else
            error("$family is not supported.")
        end

    elseif robust_type=="none"
        return g_link(family,mu) .+ (y .- mu) .* g_derlink(family,mu),
        1 ./ (g_derlink(family,mu) .^ 2 .* s .^2)

    else
        error("$robust_type is not supported.")
    end

    E1[zero_index] .= 0
    E2[zero_index] .= 0

    # return Z and W
    g_link(family,mu) .+ s .* (psi( (y .- mu) ./ s, c,robust_type) .- E1) .*
    g_derlink(family,mu) ./ E2, E2 ./ (s .* g_derlink(family,mu)) .^2

end


function g_ZW2(family::String,robust_type::String,y::Union{Vector{Int64},Vector{Float64}},mu::Union{Vector{Int64},Vector{Float64}},s::Vector{Float64},c::Float64,sigma::Union{Nothing,Float64,Int64})

    E1 = similar(y)
    E2 = similar(y)
    musq::Vector{Float64} = mu .^ 2
    nb_r = (typeof(sigma)==Nothing, nothing, 1/sigma)
    nb_p = similar(y)

    j1::Vector{Int} = floor.(Int,mu .- c .* s)
    j2::Vector{Int} = floor.(Int,mu .+ c .* s)

    if robust_type =="Huber"

        if family=="P"

            tmp = Poisson.(mu)
            # E(psi(r;c))
            E1 = c .* ((1 .- cdf.(tmp,j2)) .- cdf.(tmp,j1)) .+ mu ./ s .* (pdf.(tmp,j1) .- pdf.(tmp,j2))
            # E(r psi(r;c))
            E2 = (c .* mu) .* (pdf.(tmp,j1) .+ pdf.(tmp,j2)) .+
            1 ./ s .* ( s .^ 2 .* (cdf.(tmp,j2 .- 2) .- cdf.(tmp,j1 .- 2)) .+
            (musq .+ mu) .* (pdf.(tmp,j1 .- 1) .- pdf.(tmp,j2 .- 1)) .-
            musq .* (pdf.(tmp,j1) .- pdf.(tmp,j2)) )
            E2 = E2 ./ s

        elseif family=="NB"
            nb_r = 1 / sigma
            nb_p = 1 ./ (sigma .* mu .+ 1)
            E1 =  c .* ((1 .- StatsFuns.nbinomcdf.(nb_r,nb_p,j2)) .- StatsFuns.nbinomcdf.(nb_r,nb_p,j1)) .+ mu ./ s .*
            (StatsFuns.nbinompdf.(nb_r,nb_p,j1) .* (sigma .* j1 .+ 1) .-
             StatsFuns.nbinompdf.(nb_r,nb_p,j2) .* (sigma .* j2 .+ 1))
            E2 = (c .* mu) .*
            (StatsFuns.nbinompdf.(nb_r,nb_p,j1) .* (sigma .* j2  .+ 1) .+ StatsFuns.nbinompdf.(nb_r,nb_p,j2) .* (sigma .* j1 .+1)) .+
            1 ./ s .* ( s .^ 2 .* (StatsFuns.nbinomcdf.(nb_r,nb_p,j2 .- 2) .- StatsFuns.nbinomcdf.(nb_r,nb_p,j1 .- 2)) .+
            StatsFuns.nbinompdf.(nb_r,nb_p,j1) .* (sigma .* mu .* j1 .^ 2 .- (2*sigma) .* musq .* j1 .- musq) .+
            StatsFuns.nbinompdf.(nb_r,nb_p,j1 .- 1) .* ( sigma .* musq .* (sigma+1) .* (j1 .- 1) .- mu .+ musq) .-
            StatsFuns.nbinompdf.(nb_r,nb_p,j2) .* ( sigma .* mu .* j2 .^ 2 .- (2*sigma) .* musq .* j2 .- musq) .-
            StatsFuns.nbinompdf.(nb_r,nb_p,j2 .- 1) .* ( sigma .* mu .^ 2 .* (sigma+1) .* (j2 .- 1) .- mu .+ mu .^ 2))
            E2 = E2 ./ s


        else
            error("$family is not supported.")
        end



    elseif robust_type =="Tukey"

        if family=="P"

            tmp = Poisson.(mu)
            # E(psi(r;c))
            E1 = map(i -> sum(map(k -> psi((k-mu[i])/s[i],c,robust_type) * pdf(tmp[i],k),collect((j1[i]+1):j2[i]))),collect(1:length(y)))
            # E(r psi(r;c))
            E2 = map(i -> sum(map(k -> psi((k-mu[i])/s[i],c) * (k-mu[i]) * pdf(tmp[i],k),collect((j1[i]+1):j2[i]))),collect(1:length(y)))
            E2 = E2 ./ s


        elseif family=="NB"
            nb_r = 1 / sigma
            nb_p = 1 ./ (sigma .* mu .+ 1)
            # E(psi(r;c))
            E1 = map(i -> sum(map(k -> psi((k-mu[i])/s[i],c) * StatsFuns.nbinompdf(nb_r,nb_p[i],k),collect((j1[i]+1):j2[i]))),collect(1:length(y)))
            # E(r psi(r;c))
            E2 = map(i -> sum(map(k -> psi((k-mu[i])/s[i],c) * (k-mu[i]) * StatsFuns.nbinompdf(nb_r,nb_p[i],k),collect((j1[i]+1):j2[i]))),collect(1:length(y)))
            E2 = E2 ./ s

        else
            error("$family is not supported.")
        end

    elseif robust_type=="none"
        return g_link(family,mu) .+ (y .- mu) .* g_derlink(family,mu),
        1 ./ (g_derlink(family,mu) .^ 2 .* s .^2)

    else
        error("$robust_type is not supported.")
    end

    return g_link(family,mu) .+ s .* (psi( (y .- mu) ./ s, c,robust_type) .- E1) .*
    g_derlink(family,mu) ./ E2, E2 ./ (s .* g_derlink(family,mu)) .^2

end
