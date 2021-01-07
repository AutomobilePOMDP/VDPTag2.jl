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
POMDPs.isterminal(p::DiscreteVDPTagProblem, s::TagState) = mdp(p).tag_terminate && norm(s.agent-s.target) < mdp(p).tag_radius
POMDPs.actions(p::DiscreteVDPTagProblem) = [TagAction(look, angle) for look in [false, true] for angle in p.angles]
POMDPs.actionindex(p::DiscreteVDPTagProblem, a::TagAction) = a.look * length(p.angles) + findfirst(x->x==a.angle, p.angles)

POMDPs.transition(p::DiscreteVDPTagProblem, s::TagState, a::TagAction) = transition(cproblem(p), s, a)
POMDPs.initialstate(p::DiscreteVDPTagProblem) = VDPInitDist()
POMDPs.reward(p::DiscreteVDPTagProblem, s::TagState, a::TagAction, sp::TagState) = reward(cproblem(p), s, a, sp)

POMDPs.observation(p::ADiscreteVDPTagPOMDP, a::TagAction, sp::TagState) = observation(cproblem(p), a, sp)

struct DiscreteBeamDist
    beam_dist::BeamDist
    pomdp::POMDP
end

POMDPs.observation(p::AODiscreteVDPTagPOMDP, a::TagAction, sp::TagState) = DiscreteBeamDist(observation(cproblem(p), a, sp), p)
rand(rng::AbstractRNG, d::DiscreteBeamDist) = convert_o(IVec8, rand(rng, d.beam_dist), d.pomdp)

function POMDPs.pdf(d::DiscreteBeamDist, o::IVec8)
    p = 1.0
    lower = o .* d.pomdp.binsize
    upper = (o .+ 1) .* d.pomdp.binsize
    d = d.beam_dist
    for i in 1:length(o)
        if i == d.abeam
            p *= cdf(d.an, upper[i]) - cdf(d.an, lower[i])
        else
            p *= cdf(d.n, upper[i]) - cdf(d.n, lower[i])
        end
    end
    return p
end