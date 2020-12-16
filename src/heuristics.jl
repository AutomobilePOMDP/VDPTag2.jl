struct ToNextML{P<:Union{VDPTagProblem, DiscreteVDPTagProblem}, RNG<:AbstractRNG} <: Policy
    p::P
    rng::RNG
end

ToNextML(p::Union{VDPTagProblem, DiscreteVDPTagProblem}; rng=Random.GLOBAL_RNG) = ToNextML(p, rng)

function POMDPs.action(p::ToNextML{VDPTagMDP}, s::TagState)
    next = next_ml_target(p.p, s.target)
    diff = next-s.agent
    return atan(diff[2], diff[1])
end

function POMDPs.action(p::ToNextML, s::TagState)
    next = next_ml_target(p.p, s.target)
    diff = next-s.agent
    return TagAction(false, atan(diff[2], diff[1]))
end

POMDPs.action(p::ToNextML, b::AbstractParticleBelief) = TagAction(false, action(p, rand(p.rng, b)))

struct ToNextMLSolver <: Solver
    rng::AbstractRNG
end

POMDPs.solve(s::ToNextMLSolver, p::Union{VDPTagProblem, DiscreteVDPTagProblem}) = ToNextML(p, s.rng)

struct ManageUncertainty <: Policy
    p::Union{VDPTagPOMDP, DiscreteVDPTagProblem}
    max_norm_std::Float64
end

function POMDPs.action(p::ManageUncertainty, b::AbstractParticleBelief)
    agent = first(particles(b)).agent
    prob_dict = ParticleFilters.probdict(b)
    target_particles = Array{Float64}(undef, 2, length(prob_dict))
    for (i, s) in enumerate(keys(prob_dict))
        target_particles[:,i] = s.target
    end
    normal_dist = fit(MvNormal, target_particles, collect(values(prob_dict))) # particles should be unique
    angle = POMDPs.action(ToNextML(mdp(p.p)), TagState(agent, mean(normal_dist)))
    return TagAction(sqrt(det(cov(normal_dist))) > p.max_norm_std, angle)
end

POMDPs.action(p::ManageUncertainty, b::Any) = TagAction(false, action(ToNextML(mdp(p.p)), rand(b)))
POMDPs.action(p::ManageUncertainty, s::TagState) = TagAction(false, action(ToNextML(mdp(p.p)), s))


mutable struct NextMLFirst{RNG<:AbstractRNG}
    p::VDPTagMDP
    rng::RNG
end

function next_action(gen::NextMLFirst, mdp::Union{POMDP, MDP}, s::TagState, snode)
    if n_children(snode) < 1
        return POMDPs.action(ToNextML(gen.p, gen.rng), s)
    else
        return 2*pi*rand(gen.rng)
    end
end

function next_action(gen::NextMLFirst, pomdp::Union{POMDP, MDP}, b, onode)
    s = rand(gen.rng, b)
    return TagAction(false, next_action(gen, pomdp, s, onode))
end