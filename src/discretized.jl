import POMDPs.initialstate
const IVec8 = SVector{8, Int}

@with_kw struct AODiscreteVDPTagPOMDP <: POMDP{TagState, TagAction, IVec8}
    cpomdp::VDPTagPOMDP             = VDPTagPOMDP()
    angles::Array{Float64, 1}       = range(0, stop=2*pi, length=11)[1:end-1]
    binsize::Float64                = 0.5
end
AODiscreteVDPTagPOMDP(p::VDPTagPOMDP, n_angles::Int, binsize::Float64) = AODiscreteVDPTagPOMDP(p, range(0, stop=2*pi, length=n_angles+1)[1:end-1], binsize)

@with_kw struct ADiscreteVDPTagPOMDP <: POMDP{TagState, TagAction, Vec8}
    cpomdp::VDPTagPOMDP             = VDPTagPOMDP()
    angles::Array{Float64, 1}       = range(0, stop=2*pi, length=11)[1:end-1]
end
ADiscreteVDPTagPOMDP(p::VDPTagPOMDP, n_angles::Int) = ADiscreteVDPTagPOMDP(p, range(0, stop=2*pi, length=n_angles+1)[1:end-1])

const DiscreteVDPTagProblem = Union{AODiscreteVDPTagPOMDP, ADiscreteVDPTagPOMDP}

cproblem(p::DiscreteVDPTagProblem) = p.cpomdp
mdp(p::DiscreteVDPTagProblem) = mdp(cproblem(p))

convert_s(::Type{T}, x::T, p) where T = x
convert_a(::Type{T}, x::T, p) where T = x
convert_o(::Type{T}, x::T, p) where T = x

# observation
function convert_o(::Type{IVec8}, o::Vec8, p::AODiscreteVDPTagPOMDP)
    return floor.(Int, (o./p.binsize)::Vec8)::IVec8
end
# convert_o(::Type{Vec8}, o::Int, p::DiscreteVDPTagProblem) = (o-0.5)*2*pi/p.n_obs_angles

n_states(p::DiscreteVDPTagProblem) = Inf
n_actions(p::DiscreteVDPTagProblem) = 2*length(p.angles)
POMDPs.discount(p::DiscreteVDPTagProblem) = discount(cproblem(p))
POMDPs.actions(p::DiscreteVDPTagProblem) = [TagAction(look, angle) for angle in p.angles for look in [true, false]]

function POMDPs.gen(p::ADiscreteVDPTagPOMDP, s::TagState, a::TagAction, rng::AbstractRNG)
    return gen(cproblem(p), s, a, rng)
end

function POMDPs.gen(p::AODiscreteVDPTagPOMDP, s::TagState, a::TagAction, rng::AbstractRNG)
    csor = @gen(:sp,:o,:r)(cproblem(p), s, a, rng)
    return (sp=csor[1], o=convert_o(IVec8, csor[2], p), r=csor[3])
end

function POMDPs.observation(p::ADiscreteVDPTagPOMDP, s::TagState, a::TagAction, sp::TagState)
    return POMDPs.observation(cproblem(p), s, a, sp)
end

function POMDPs.observation(p::AODiscreteVDPTagPOMDP, s::TagState, a::TagAction, sp::TagState)
    ImplicitDistribution(p, s, a, sp) do p, s, a, sp, rng
        co = rand(rng, observation(cproblem(p), s, a, sp))
        return convert_o(IVec8, co, p)
    end
end

POMDPs.initialstate(p::DiscreteVDPTagProblem) = VDPInitDist()

#=
gauss_cdf(mean, std, x) = 0.5*(1.0+erf((x-mean)/(std*sqrt(2))))
function obs_weight(p::AODiscreteVDPTagPOMDP, a::Int, sp::TagState, o::Int)
    cp = cproblem(p)
    @assert cp.bearing_std <= 2*pi/6.0 "obs_weight assumes σ <= $(2*pi/6.0)"
    ca = convert_a(actiontype(cp), a, p)
    co = convert_o(obstype(cp), o, p) # float between 0 and 2pi
    upper = co + 0.5*2*pi/p.n_angles
    lower = co - 0.5*2*pi/p.n_angles
    if ca.look
        diff = sp.target - sp.agent
        bearing = atan(diff[2], diff[1])
        # three cases: o is in bin, below, or above
        if bearing <= upper && bearing > lower
            cdf_up = gauss_cdf(bearing, cp.bearing_std, upper)
            cdf_low = gauss_cdf(bearing, cp.bearing_std, lower)
            prob = cdf_up - cdf_low
        elseif bearing <= lower
            cdf_up = gauss_cdf(bearing, cp.bearing_std, upper)
            cdf_low = gauss_cdf(bearing, cp.bearing_std, lower)
            below_cdf_up = gauss_cdf(bearing, cp.bearing_std, upper-2*pi)
            below_cdf_low = gauss_cdf(bearing, cp.bearing_std, lower-2*pi)
            prob = cdf_up - cdf_low + below_cdf_up - below_cdf_low
        else # bearing > upper
            cdf_up = gauss_cdf(bearing, cp.bearing_std, upper)
            cdf_low = gauss_cdf(bearing, cp.bearing_std, lower)
            above_cdf_up = gauss_cdf(bearing, cp.bearing_std, upper+2*pi)
            above_cdf_low = gauss_cdf(bearing, cp.bearing_std, lower+2*pi)
            prob = cdf_up - cdf_low + above_cdf_up - above_cdf_low
        end
        return prob
    else
        return 1.0
    end
end

function obs_weight(p::ADiscreteVDPTagPOMDP, a::Int, sp::TagState, o::Float64)
    ca = convert_a(TagAction, a, p)
    return obs_weight(cproblem(p), ca, sp, o)
end
=#
