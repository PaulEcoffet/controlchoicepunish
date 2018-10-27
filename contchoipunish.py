import numpy as np
import copy
import sys

def dummy_log(*args, **kwargs): pass

logi = print
loge = print
logd = dummy_log

class Agent():
    def __init__(self, id, hlayers_sizes):
        self.id = id
        self.fitness = 0

        self.prev_coop = 0
        self.prev_punish = 0
        self.knows_partner = 0
        self.leave = 0
        self.coop = 0
        self.spite = 0
        if type(hlayers_sizes) == Agent:
            agent = hlayers_sizes
            self.hlayers_sizes = copy.deepcopy(agent.hlayers_sizes)
            self.layers = copy.deepcopy(agent.layers)
        else:
            self.hlayers_sizes = [4] + hlayers_sizes + [3]
            self.init_layers(hlayers_sizes)

    def init_layers(self, hlayers_sizes):
        nblayers = 1 + len(hlayers_sizes)
        self.layers = []
        true_hlayers_sizes = [4] + hlayers_sizes + [3] # input = bias + prev_coop + prev_punish + knows_partner
        for ninput, noutput in zip(true_hlayers_sizes, true_hlayers_sizes[1:]):
            self.layers.append(np.random.uniform(-1, 1, size=(noutput, ninput)))

    def step_decision(self):
        input = np.array([1, self.prev_coop, self.prev_punish, self.knows_partner])
        output = None
        for layer in self.layers:
            output = np.tanh(layer @ input)
            input = output
        self.leave = output[0] > 0
        self.coop = (output[1] + 1) / 2 * 10
        self.spite = (output[2] + 1) / 2 * 10
        assert(0 <= self.coop <= 10)
        assert(0 <= self.spite <= 10)

    def mutate(self):
        il = np.random.choice(len(self.layers))
        ir = np.random.choice(len(self.layers[il]))
        ic = np.random.choice(len(self.layers[il][ir]))
        self.layers[il][ir][ic] = np.random.uniform(-10, 10)

    def clear_inputs(self):
        self.knows_partner = 0
        self.prev_coop = 0
        self.prev_punish = 0

    def fulldesc(self):
        return f"""Agent {self.id}
self.knows_partner = {self.knows_partner}
prev_coop: {self.prev_coop}
prev_punish: {self.prev_punish}
coop: {self.coop}
punish: {self.spite}
leave: {self.leave}
"""

    def __str__(self):
        return str(self.id)

    def __repr__(self):
        return f"agent {self.id}"

def create_pairs(pool_set, beta):
    new_pairs = []
    pool = list(pool_set)
    alone_pool = set()
    np.random.shuffle(pool)
    for i in range(0, len(pool), 2):
        if np.random.uniform() < beta:
            new_pairs.append([pool[i], pool[i+1]])
        else:
            alone_pool.add(pool[i])

    return new_pairs, alone_pool

def payoff(agent, pair):
    n = len(pair)
    x = agent.coop
    x0 = sum(o_agent.coop for o_agent in pair if o_agent != agent)
    res = (a * (x + x0) + b * x0) / n - 1/2*x*x
    #print(f"p({x}, {x0}, {n}) = {res}")
    return res

#########
# Const #
#########

a = 5
b = 3
beta = 1
nbgens = 100
nbstep = 100
popsize = 100
nbruns = 20

def main():
    population = [Agent(i, [5]) for i in range(popsize)]

    for igen in range(nbgens):
        for irun in range(nbruns):
            alone_pool = set(population)
            pairs = []
            for agent in population:
                agent.clear_inputs()
            for step in range(nbstep):
                # Pool management
                new_pairs, alone_pool = create_pairs(alone_pool, beta)
                pairs += new_pairs
                next_pairs = []
                leavers = set()
                for i_pair, pair in enumerate(pairs):
                    # Agent decision
                    for agent in pair:
                        agent.step_decision()
                    # Leave is prioritary on other actions
                    if (any([agent.leave for agent in pair])):
                        logd(f"pair {i_pair} with {pair[0].id} and {pair[1].id} breaks")
                        for agent in pair:
                            agent.clear_inputs()
                            leavers.add(agent)
                    else: # If they play
                        next_pairs.append(pair)
                        logd(f"pair {i_pair} with {pair[0].id} and {pair[1].id} plays")
                        total_coop = 0
                        total_spite = 0
                        for agent in pair:
                            logd(f"agent {agent.id} plays {agent.coop} and gains {payoff(agent, pair)}")
                            agent.fitness += payoff(agent, pair)
                            total_coop += agent.coop
                            logd(f"agent {agent.id} spite {agent.spite}")
                            #agent.fitness -= agent.spite
                            total_spite += agent.spite
                            for o_agent in pair :
                                if o_agent != agent: pass
                                    #o_agent.fitness -= 3 * agent.spite
                        for agent in pair: # update inputs
                            agent.knows_partner = 1
                            agent.prev_coop = total_coop - agent.coop
                            agent.prev_punish = total_spite - agent.spite

                # Removing broken appart pairs, and adding the agent to the pool
                alone_pool |= leavers
                pairs = next_pairs
        # end runs
        fitnesses = np.array([agent.fitness for agent in population])
        i_max = np.argmax(fitnesses)
        print(population[i_max].fulldesc())
        print(igen, np.min(fitnesses), np.median(fitnesses), np.max(fitnesses))
        min_fit = np.min(fitnesses)
        if min_fit >= 0: min_fit = -1
        fitnesses -= min_fit
        sumfit = np.sum(fitnesses)
        parents = np.random.choice(population, size=popsize, p=fitnesses/sumfit, replace=True)
        population = [Agent(i, agent) for i, agent in enumerate(parents)]
        np.random.choice(population).mutate()


main()