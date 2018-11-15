module CCP

using Random: shuffle!
using Statistics: median
using StatsBase: sample, weights
using Serialization: serialize
using GZip
using JLD
using ArgParse
using Parameters


nbgens = 10000
nbruns = 4
nbsteps = 5000
popsize = 100
const beta = 1.
const a = 5.
const b = 3. 
const maxcoop = 20
logdir = "juliatest2"

mutable struct Agent
    id::Int64
    fitness::Float64

    # Inputs
    seen_coop::Float64
    seen_punish::Float64
    new_partner::Bool

    # Outputs
    coop::Float64
    punish::Float64
    leave::Float64

    # inside
    coef::Float64

    # Layers
    layers::Array{Matrix{Float64}}
end

@with_kw struct Conf
    a::Float64 = 5
    b::Float64 = 3
    beta::Float64 = 1
    nbgens::Int = 20000
    nbruns::Int = 4
    nbsteps::Int = 5000
    popsize::Int = 100
    fakeprop::Int = 2
    maxcoop::Float64 = 30
    nbmut::Int = 1
    logdir::String = "."
    partnercontrol::Bool = false
    nointeract::Bool = false
end

Agent(id::Int) = Agent(id, 0, 0, 0, true, 0, 0, 0, 1, init_layers(-5., 5., [5]))
Agent(id::Int, layers::Array{Matrix{Float64}}) = Agent(id, 0, 0, 0, true, 0, 0, 0, 1, deepcopy(layers))
Agent(id::Int, parent::Agent) = Agent(id, parent.layers)


function Base.show(io::Base.IO, a::Agent)
    print(io, "Agent ", a.id)
end

function init_layers(low::Float64, high::Float64, hidden_dim::Array{Int64})::Array{Matrix{Float64}}
    dim::Array{Int64} = [4 hidden_dim 3]
    layers = Matrix{Float64}[]
    for (i, j) in zip(dim, dim[2:end])
        push!(layers, rand(Float64, (j, i)) .* (high - low) .+ low)
    end
    return layers
end

function decision(agent::Agent)::Agent
    output::Array{Float64} = [1.; agent.seen_coop / maxcoop;
                              agent.seen_punish / maxcoop;
                              convert(Float64, agent.new_partner)]
    for layer in agent.layers
        output = tanh.(layer * output)
    end
    agent.leave = output[1]
    agent.coop = (output[2] + 1) / 2 * maxcoop
    agent.punish = (output[3] + 1) / 2 * maxcoop
    return agent
end

function clear_inputs!(agent::Agent)::Agent
    agent.seen_coop = 0.
    agent.seen_punish = 0.
    agent.new_partner = true
    return agent
end

function create_pairs(pool, beta::Real)::Tuple{Array{Array{Agent}}, Array{Agent}}
    new_pairs = Array{Agent}[]
    alone_pool = Agent[]
    shuffle!(pool)
    for i = 1:2:length(pool)
        if rand(Float64) < beta
            push!(new_pairs, [pool[i], pool[i+1]])
        else
            push!(alone_pool, pool[i], pool[i+1])
        end
    end
    return new_pairs, alone_pool
end

function mutate!(agent::Agent)::Agent
    il = rand(1:length(agent.layers))
    i = rand(1:size(agent.layers[il])[1])
    j = rand(1:size(agent.layers[il])[2])
    agent.layers[il][i, j] = rand(Float32) * 10 - 5  # [-5, 5[
    return agent
end

isfake(agent::Agent)::Bool = agent.coef != 1
getcoop(agent::Agent)::Float64 = clamp(agent.coef * agent.coop, 0, maxcoop)
doleave(agent::Agent)::Bool = agent.leave > 0




function payoff(agent::Agent, pair, a::Real, b::Real)::Real
    x = getcoop(agent)
    x0 = sum(getcoop.(a for a in pair)) - x
    n = length(pair)
    return (a * (x + x0) + b * x0)/n - 1/2 * x*x
end


function main(conf::Conf)
    nbgens = conf.nbgens
    nbruns = conf.nbruns
    nbsteps = conf.nbsteps
    beta = conf.beta
    fakeprop = conf.fakeprop
    a = conf.a
    b = conf.b
    logdir = conf.logdir

    @assert (nbruns % fakeprop == 0) "nbruns should be a multiple of fakeprop so that each robot is fake the same amount of runs"

    mkpath(logdir)

    population = Agent[]
    nbfakes = div(popsize, fakeprop)
    GZip.open("$logdir/fitnesslog.txt.gz", "w") do ffit
            for agent in population
                write(ffit, "gen, agent, fitness\n")
            end
        end

    for i in 1:popsize
        push!(population, Agent(i))
    end
    for igen in 1:nbgens
        log = igen % 1000 == 0
        println(igen)
        if log
            JLD.save("$logdir/population_$igen.jld", "population", population)
            println("genome written")
            runlog = GZip.open("$logdir/jrunlog_$igen.txt.gz", "w")
            write(runlog, "gen, run, step, agent, pair, coef, seencoop, seenpunish, newpartner, coop, punish, leave\n")
        end
        fakes = zeros(Int, popsize)
        for irun in 1:nbruns
            if all(fakes .== 2)
                fakes = cat(zeros(Int, popsize - nbfakes), ones(Int, nbfakes), dims=1)
            else
                to_do = sample([i for i in 1:popsize if fakes[i] == 0], nbfakes, replace=false)
                fakes[to_do] .= 1
            end
            nbfakesdone = 0
            fakelistiter = collect(zip(population, fakes))
            shuffle!(fakelistiter)
            for (agent, fake) in fakelistiter
                if fake == 1
                    agent.coef = nbfakesdone / (nbfakes - 1) * 2
                    nbfakesdone += 1
                else
                    agent.coef = 1
                end
            end

            alone_pool = copy(population)
            pairs = Array{Agent}[]
            clear_inputs!.(population)
            for istep in 1:nbsteps
                newpairs, alone_pool = create_pairs(alone_pool, beta)
                push!(pairs, newpairs...)
                next_pairs = Array{Agent}[]
                leavers = Agent[]
                for agent in alone_pool
                    write(runlog, "$igen, $irun, $istep, -1, $(agent.id), $(agent.coef)," *
                                  "$(agent.seen_coop), $(agent.seen_punish), $(agent.new_partner)," *
                                  "$(agent.coop), $(agent.punish), $(agent.leave)\n")
                end
                if conf.nointeract
                    clear_inputs!.(population)
                end
                for (i_pair, pair) in enumerate(pairs)
                    for agent in pair
                        decision(agent)
                        if log
                            write(runlog, "$igen, $irun, $istep, $i_pair, $(agent.id), $(agent.coef)," *
                                  "$(agent.seen_coop), $(agent.seen_punish), $(agent.new_partner)," *
                                  "$(agent.coop), $(agent.punish), $(agent.leave)\n")
                        end
                    end
                    if !conf.partnercontrol && any([doleave(agent) for agent in pair])
                        clear_inputs!.([agent for agent in pair])
                        push!(leavers, pair...)
                    else
                        push!(next_pairs, pair)
                        total_coop::Float64 = 0.
                        total_spite::Float64 = 0.
                        for agent in pair
                            if !isfake(agent)
                                agent.fitness += payoff(agent, pair, a, b)
                                agent.fitness -= agent.punish
                            end
                            total_coop += getcoop(agent)
                            total_spite += agent.punish
                            for o_agent in pair
                                if o_agent != agent && !isfake(o_agent)
                                    o_agent.fitness -= 3 * agent.punish
                                end
                            end
                        end
                        for agent in pair
                            agent.new_partner = false
                            agent.seen_coop = total_coop - getcoop(agent)
                            agent.seen_punish = total_spite - agent.punish
                        end
                    end
                end
                pairs = next_pairs
                if length(leavers) > 0
                    push!(alone_pool, leavers...)
                end
            end # step
            fakes[fakes .== 1] .= 2
        end # run
        @assert all(fakes .== 2) "not all robots has been faked at the end"
        fitnesses = Float64[agent.fitness for agent in population]
        GZip.open("$logdir/fitnesslog.txt.gz", "a") do ffit
            for agent in population
                write(ffit, "$igen, $(agent.id), $(agent.fitness)\n")
            end
        end
        println(minimum(fitnesses), " ",  median(fitnesses), " ", maximum(fitnesses))
        minfit = minimum(fitnesses)
        if minfit <= 0
            wfit = weights(fitnesses .- (minfit .- 1))
        else
            wfit = weights(fitnesses)
        end
        parents = sample(population, wfit, popsize, replace=true)
        population::Array{Agent} = [Agent(i, parent) for (i, parent) in enumerate(parents)]
        for mutant in sample(population, conf.nbmut)
            mutate!(mutant)
        end
        if log
            close(runlog)
        end
    end  # gen
end

s = ArgParseSettings()
@add_arg_table s begin
    "--logdir"
        arg_type = String
        default = "."
    "--partnercontrol"
        action = :store_true
    "--nointeract"
        action = :store_true
    "--nbgens"
        arg_type = Int
        default = 10000
    "--nbruns"
        arg_type = Int
        default = 4
    "--nbsteps"
        arg_type = Int
        default = 5000
    "--popsize"
        arg_type = Int
        default = 100
    "--nbmut"
        arg_type = Int
        default = 1
    "--fakeprop"
        arg_type = Int
        default = 2
end

parsed_args = parse_args(ARGS, s)

conf = Conf(nbgens=parsed_args["nbgens"],
            logdir=parsed_args["logdir"],
            partnercontrol=parsed_args["partnercontrol"],
            nointeract=parsed_args["nointeract"],
            nbruns=parsed_args["nbruns"],
            nbsteps=parsed_args["nbsteps"],
            popsize=parsed_args["popsize"],
            nbmut=parsed_args["nbmut"],
            fakeprop=parsed_args["fakeprop"])
main(conf)


end
