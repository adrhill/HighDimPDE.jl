"""
Sampling method for the Monte Carlo integration
"""
abstract type MCSampling{T} end
Base.eltype(::MCSampling{T}) where T = eltype(T)


struct UniformSampling{T} <: MCSampling{T} 
    a::T
    b::T
    invdims::Array{Int64} # dimensions where no integrations
end
@functor UniformSampling (a,b,)
"""
    function UniformSampling(a, b, [dims])
Uniform sampling method for the Monte Carlo integration, in the hypercube `[a, b]^2`.
If `dims` is specified, only `dims` dimensions are selected for integration.
"""
UniformSampling(a, b) = UniformSampling(a, b, Int64[])
# UniformSampling(a, b, dims, d) = UniformSampling(a, b, d, filter(i -> !(i in dims), 1:d))

function (mc_sample::UniformSampling{T})(x_mc, kwargs...) where T
    Tel = eltype(T)
    idim = mc_sample.invdims
    rand!(x_mc)
    m = (mc_sample.b + mc_sample.a) ./ convert(Tel,2)
    x_mc .= (x_mc .- convert(Tel,0.5)) .* (mc_sample.b - mc_sample.a) .+ m
    alldims = ntuple(i->:,ndims(x_mc)-1)
    if !isempty(kwargs)
         x = kwargs[1]
        x_mc[idim, alldims...] .= x[idim, alldims...] # can not do partial without x
    end
    nothing
end

struct NormalSampling{T} <: MCSampling{T}
    σ::T
    shifted::Bool # if true, we shift integration by x when invoking mc_sample::MCSampling(x)
    invdims::Array{Int64} # dimensions where no integrations
end
@functor NormalSampling (σ,shifted,)

"""
Normal sampling method for the Monte Carlo integration.

Arguments:
* `σ`: the standard devation of the sampling
* `shifted` : if true, the integration is shifted by `x`
* If `invdims` is specified, only `Not(invdims)` dimensions are selected for integration.
"""
NormalSampling(σ) = NormalSampling(σ, false, Int64[])
NormalSampling(σ, shifted::Bool) = NormalSampling(σ, shifted, Int64[])
# NormalSampling(σ, shifted, dims, d) = NormalSampling(σ, shifted, filter(i -> !(i in dims), 1:d))

function (mc_sample::NormalSampling{T})(x_mc, kwargs...) where T
    idim = mc_sample.invdims
    randn!(x_mc)
    x_mc .*=  mc_sample.σ  
    if !isempty(kwargs)
        x = kwargs[1]
        alldims = ntuple(i->:,ndims(x_mc)-1)
        mc_sample.shifted ? x_mc .+= x : nothing
        x_mc[idim, alldims...] .= x[idim, alldims...]
    end
    nothing
end

struct NoSampling <: MCSampling{Nothing} end

(mc_sample::NoSampling)(x) = nothing

function _integrate(::MCS) where {MCS <: MCSampling}
    if MCS <: NoSampling
        return false
    else
        return true
    end
end