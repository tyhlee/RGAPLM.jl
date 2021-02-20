# type declarations
type_VecFloat = Union{Float64,Vector{Float64}}
type_VecInt = Vector{Int}
type_VecReal = Vector{Real}
type_VecFloatInt = Union{type_VecInt,type_VecFloat}
type_VecRealFloatInt = Union{type_VecInt,type_VecFloat,type_VecReal}
type_VecReal = Vector{Real}
type_VecOrMatFloat = VecOrMat{Float64}
type_VecOrMatInt = VecOrMat{Int}
type_VecOrMatFloatInt = Union{type_VecOrMatFloat,type_VecOrMatInt}
type_FloatInt = Union{Int,Float64}
type_VecFloatIntFloatInt = Union{Int,Float64,type_VecInt,type_VecFloat}
type_NTVecOrMatFloatInt = Union{Nothing,type_VecOrMatFloat,type_VecOrMatInt}

function tricube(u::type_VecFloatInt)
    # (1 .- u.^3).^3
    map(x -> abs(x) <= 1.0 ? (1-x^3)^3 : 0,u)
end

# create the design matrix for loess
function outer_power(u::type_VecFloatInt,pow_degree::Integer)
    tmp = Array{Float64,2}(undef,length(u),pow_degree+1)
    for i in 0:pow_degree
        tmp[:,i+1] = u.^i
    end
    tmp
end

# compute the kernel weight for loess
function compute_kernel_weight(x::type_VecFloatInt, span::type_FloatInt)
    tricube(x./sort(abs.(x))[ceil(Int,span*length(x))])
end

function rho(r::type_VecFloatIntFloatInt, c::type_FloatInt=1.345,type::String="Huber") where T <: Real
    if type=="Huber"
        return map(t -> abs(t)<=c ? t^2 : 2*c*abs(t)-c^2,r)
    elseif type =="Tukey"
        return map(t -> abs(t)<=c ? 1-(1-(t/c)^2)^3 : 1,r)
    else
        error("Unknown type")
    end
end

function psi(r::type_VecFloatIntFloatInt, c::type_FloatInt,type::String="Huber")
    if type=="Huber"
        return map(t -> abs(t)<=c ? t : sign(t)*c,r)
    elseif type =="Tukey"
        return map(t -> abs(t)<=c ? t*(1-(t/c)^2)^2 : 0,r)
    else
        throw(error("Unknown type"))
    end
end

function psi_weight(r::type_VecFloatIntFloatInt, c::type_FloatInt,type::String="Huber")
    if type=="Huber"
        return map(t -> abs(t)<=c ? 1 : c/abs(t),r)
    elseif type =="Tukey"
        return map(t -> abs(t)<=c ? (1-(t/c)^2)^2 : 0,r)
    else
        throw(error("Unknown type"))
    end
end

function compute_crit(X::type_VecOrMatFloatInt,XX::type_VecOrMatFloatInt,type="norm2")
    if type=="norm2"
        return norm(sum(X .- XX,dims=2))
    elseif type=="norm2_change"
        return norm(X .- XX)/norm(XX)
    elseif type =="norm_inf"
        return norm(X .- XX,Inf)
    else
        error("Compute crit $type is not supported")
    end
end

# drop the ind column of a matrix
dropcol(M::AbstractMatrix,ind) = M[:,deleteat!(collect(axes(M,2)),ind)]

# compute a robust scale estimate using R's robustbase::scaeltau2
function scaletau2(x::type_VecFloatInt;c1::type_VecFloat=4.5,c2::type_VecFloat=3.0,consistency=true,
    sigma::type_VecFloat=1.0) where T<:Float64

    n::Int = length(x)
    medx::Float64 = median(x)
    xx = abs.(x .- medx)
    sigma = median(xx)
    if sigma <= 0
        return medx, 0
    end

    w = similar(xx)
    mu::type_VecFloat = 0.0

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
        input::type_VecFloat = c2*quantile(Normal(0,1),3/4)
        nEs2::type_VecFloat = n * (2 * ((1-input^2) * cdf(Normal(0,1),input) - input * pdf(Normal(0,1),input) + input^2) - 1)
    else
        nEs2 = n
    end
    return mu, sigma * sqrt(sum(w)/nEs2)
end

# determine the number of samples and the number of columns in a matrix
# returns 0 num of columns for a vector
function size_VecOrMat(x::Union{Nothing,VecOrMat{Float64},VecOrMat{Int}})
    if typeof(x)== Nothing
        return 0,0
    elseif typeof(x) == Array{typeof(x[1]),1}
        return length(x), 1
    else
        return size(x)
    end
end

# neg binom helper using mu and sigma: cdf
function nb2_cdf_mu_sigma(mu::T,sigma::T,y::Union{Int,Vector{Int},Float64,Vector{Float64}}) where {T <: Union{Int, Float64,Vector{Float64},Vector{Int}}}
        StatsFuns.nbinomcdf.(1 ./ sigma, 1 ./ (sigma .* mu .+ 1),y)
end

# neg binom helper using mu and sigma: pmf
function nb2_pdf_mu_sigma(mu::T,sigma::T,y::Union{Int,Vector{Int},Float64,Vector{Float64}}) where {T <: Union{Int, Float64,Vector{Float64},Vector{Int}}}
        StatsFuns.nbinompdf.(1 ./ sigma, 1 ./ (sigma .* mu .+ 1),y)
end

# GLM link functions
function g_link(family::String,mu::type_VecFloatInt;link::String="log")
    if link == "log"
        return log.(mu)
    else
        erro("$link is not supported.")
    end
end

# GLM inverse link functions
function g_invlink(family::String,eta::type_VecFloatInt;link::String="log")
    if link == "log"
        return exp.(eta)
    else
        erro("$link is not supported.")
    end
end

# GLM derivative of link functions
function g_derlink(family::String,mu::type_VecFloatInt;link::String="log")
    if link == "log"
        return 1 ./ mu
    else
        erro("$link is not supported.")
    end
end

# GLM weight functions
function g_weight(family::String,mu::type_VecFloatInt;link::String="log")
    if link == "log"
        log.(mu)
    else
        erro("$link is not supported.")
    end
end

# GLM variance functions
function g_var(family::String,mu::type_VecFloatInt,sigma::Union{Nothing,Int,Float64}=nothing)
    if family == "P"
        return mu
    elseif family =="NB"
        return mu .+ (mu .^2) .* sigma
    else
        erro("$link is not supported.")
    end
end

# compute GLM robust Z (adjusted response variable) and W (weights)
function g_ZW(family::String,robust_type::String,y::type_VecFloatInt,mu::type_VecFloatInt,
    s::type_VecFloatInt,c::type_FloatInt,sigma::Union{Nothing,Float64,Int})

    if robust_type=="none"
        return g_link(family,mu) .+ (y .- mu) .* g_derlink(family,mu),
        1 ./ (g_derlink(family,mu) .^ 2 .* s .^2)
    end
    E1 = similar(y)
    E2 = similar(y)
    musq = mu .^ 2
    nb_r = (typeof(sigma)==Nothing, nothing, 1/sigma)
    nb_p = similar(y)

    j1::Vector{Int} = max.(ceil.(Int,mu .- c .* s),0)
    j2::Vector{Int} = floor.(Int,mu .+ c .* s)
    # j1 = max.(ceil.(Int,mu .- c .* s),0)
    # j2= min.(floor.(Int,mu .+ c .* s),1e4)
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
    return g_link(family,mu) .+ s .* (psi( (y .- mu) ./ s, c,robust_type) .- E1) .*
    g_derlink(family,mu) ./ E2, E2 ./ (s .* g_derlink(family,mu)) .^2

end

# log-likelihood
function ll(y::type_VecRealFloatInt,mu::type_VecRealFloatInt,sigma::Float64 ;type::String="NB")
    sum(loggamma.(y .+ 1/sigma) .- loggamma.(y .+ 1)) - length(y)*loggamma(1/sigma) -
    (1/sigma)* sum(log.(sigma .* mu .+ 1)) + transpose(y) * log.( sigma .* mu ./ (sigma .* mu .+ 1))
end

# score function for NB sigma
function score_sigma(y::type_VecRealFloatInt,mu::type_VecRealFloatInt,sigma::Float64;type::String="NB")
    sum(digamma.(y .+ 1/sigma) .- digamma(1/sigma) .- log.(sigma .* mu .+ 1) .- sigma .* (y .- mu) ./ (sigma .* mu .+1))
end

# info function for NB sigma
function info_sigma(y::type_VecRealFloatInt,mu::type_VecRealFloatInt,sigma::Float64;family::String="NB")
    (-1/sigma^2) * ( sum(trigamma.(y .+ 1/sigma)) - length(y)*trigamma(1/sigma)) -
      sum((sigma .* mu.^2 .+ y) ./ ((sigma .* mu .+ 1) .^2))
end

function psi_sigma_MLE(r::type_VecRealFloatInt,mu::type_VecRealFloatInt,sigma::Float64)
    digamma.(r .* sqrt.(mu .* (sigma .* mu .+ 1)) .+ mu .+ 1/sigma ) .-
    sigma .* r .* sqrt.(mu ./ (sigma .* mu .+ 1)) .-
    digamma(1/sigma) .- log.(sigma .* mu .+ 1)
end

function robust_sigma(y,mu,sigma,c;type="Tukey",family="NB")
    if sigma <=0
        return Inf
    end
    if (family == "NB") & (type =="Tukey")
        r = (y .- mu) ./ sqrt.(g_var(family,mu,float(sigma)))
        wi = psi_weight(r,c,type)
        return sum(wi .* psi_sigma_MLE(r,mu,sigma) .-
        [robust_sigma_correction(x,sigma,c) for x in mu])
    else
        error("$Type or $family is not supported")
    end
end

function robust_sigma_correction(mui,sigma,c)
    sqrtVmui = sqrt(mui * (sigma * mui + 1))
    invsig = 1/sigma
    j1 = max(ceil(Int,mui-c*sqrtVmui),0)
    j2 = floor(Int,mui+c*sqrtVmui)

    if j1>j2
        return 0
    else
        j12 = Int.(j1:j2)
        return sum((((j12 .- mui) ./ (c*sqrtVmui)).^2 .- 1) .^2 .*
            (digamma.( j12 .+ invsig) .- digamma(invsig) .- log(mui/invsig+1) .- (j12 .- mui) ./ (mui+invsig)) .*
            nb2_pdf_mu_sigma(mui,sigma,j12))
    end
end
