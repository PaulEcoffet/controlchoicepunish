import numpy as np
import copy
from typing import Iterable
import gzip
from pathlib import Path
import argparse
from os import makedirs
import time
import logging


class Agent:
    def __init__(self, id, hlayers_sizes):
        self.id = id
        self.fitness = 0

        self.prev_coop = 0
        self.coef = 1
        self.prev_punish = 0
        self.knows_partner = 0
        self.trueleave = False
        self.truecoop = 0
        self.truespite = 0
        if type(hlayers_sizes) == Agent:
            agent = hlayers_sizes
            self.hlayers_sizes = copy.deepcopy(agent.hlayers_sizes)
            self.layers = copy.deepcopy(agent.layers)
        else:
            self.hlayers_sizes = [4] + hlayers_sizes + [3]
            self.init_layers(hlayers_sizes)

    @property
    def coop(self):
        return np.clip(self.truecoop * self.coef, 0, 10)

    @property
    def fake(self):
        return self.coef != 1

    @property
    def leave(self):
        global PARTNER_CONTROL
        return self.trueleave and not PARTNER_CONTROL

    @property
    def spite(self):
        return self.truespite

    def init_layers(self, hlayers_sizes):
        self.layers = []
        true_hlayers_sizes = [4] + hlayers_sizes + [3]  # input = bias + prev_coop + prev_punish + knows_partner
        for ninput, noutput in zip(true_hlayers_sizes, true_hlayers_sizes[1:]):
            self.layers.append(np.random.uniform(-1, 1, size=(noutput, ninput)))


    def step_decision(self):
        global PARTNER_INTERACTION
        if not PARTNER_INTERACTION:
            seen_coop = 0
        else:
            seen_coop = self.prev_coop / 10
        seen_punish = self.prev_punish / 10
        output = np.array([1, seen_coop, seen_punish, self.knows_partner])
        for layer in self.layers:
            output = np.tanh(layer @ output)
        self.trueleave = output[0] > 0
        self.truecoop = (output[1] + 1) / 2 * 10
        self.truespite = (output[2] + 1) / 2 * 10

    def mutate(self):
        il = np.random.choice(len(self.layers))
        ir = np.random.choice(len(self.layers[il]))
        ic = np.random.choice(len(self.layers[il][ir]))
        self.layers[il][ir][ic] = np.random.uniform(-5, 5)

    def clear_inputs(self):
        self.knows_partner = 0
        self.prev_coop = 0
        self.prev_punish = 0

    def fulldesc(self):
        return f"""Agent {self.id}
coef: {self.coef} (fake : {self.fake})
knows_partner: {self.knows_partner}
prev_coop: {self.prev_coop}
prev_punish: {self.prev_punish}
coop: {self.coop} (true: {self.truecoop})
punish: {self.spite}
leave: {self.leave}"""

    def __str__(self):
        return f"agent {self.id}"

    def __repr__(self):
        return f"agent {self.id}"


def create_pairs(pool_set: Iterable[Agent], beta : float):
    new_pairs = []
    pool = list(pool_set)
    alone_pool = set()
    np.random.shuffle(pool)
    for i in range(0, len(pool), 2):
        if np.random.uniform() < beta:
            new_pairs.append([pool[i], pool[i+1]])
        else:
            alone_pool.add(pool[i])
            alone_pool.add(pool[i+1])

    return new_pairs, alone_pool


def payoff(agent, pair):
    global a, b
    n = len(pair)
    x = agent.coop
    x0 = sum(o_agent.coop for o_agent in pair if o_agent != agent)
    res = (a * (x + x0) + b * x0) / n - 1/2*x*x
    return res


#########
# Const #
#########

a = 5
b = 3
beta = 1
nbgens = 10000
nbstep = 500
popsize = 100
nbruns = 6
nbfakes = popsize//2
PARTNER_CONTROL = False
PARTNER_INTERACTION = False
logdir = Path('.')

def main():

    population = [Agent(i, [5]) for i in range(popsize)]

    fitnesslogf = gzip.open(logdir / 'fitnesslog.txt.gz', 'wt')
    print("gen,agent,fitness", file=fitnesslogf)
    fitnesslogf.close()
    ###############
    # GENERATIONS #
    ###############
    for igen in range(nbgens):
        fakeswap = False
        fakes = np.array([True] * nbfakes + [False] * (popsize - nbfakes))
        log = (igen + 1) % 100 == 0
        if log:
            runlogf = gzip.open(logdir / f'runlog_{igen+1}.txt.gz', 'wt')
            print("gen,run,step,pair,agent,fake,coop,punish,knows,ownCoop,ownPunish,leave", file=runlogf)

        ########
        # RUNS #
        ########
        for irun in range(nbruns):
            alone_pool = set(population)
            pairs = []
            if fakeswap:
                fakes = ~fakes  # invert every true and false
            else:
                np.random.shuffle(fakes)

            nbfakesdone = 0
            fakelistiter = list(zip(population, fakes))
            np.random.shuffle(fakelistiter)
            for agent, isfake in fakelistiter:
                agent.clear_inputs()
                if isfake:
                    agent.coef = nbfakesdone / (nbfakes - 1) * 2
                    nbfakesdone += 1
                else:
                    agent.coef = 1
            fakeswap = not fakeswap
            #########
            # STEPS #
            #########
            for step in range(nbstep):
                # Pool management
                new_pairs, alone_pool = create_pairs(alone_pool, beta)
                pairs += new_pairs
                next_pairs = []
                leavers = set()
                if log:
                    for agent in alone_pool:
                        print(igen, irun, step, -1, agent.id, agent.fake, 0, 0, 0, 0, 0, 0,
                              sep=", ", file=runlogf, flush=False)

                for i_pair, pair in enumerate(pairs):
                    # Agent decision
                    for agent in pair:
                        agent.step_decision()
                        if log:
                            print(igen, irun, step, i_pair, agent.id, agent.fake, agent.prev_coop,
                                  agent.prev_punish, agent.knows_partner, agent.truecoop, agent.spite,
                                  agent.leave, sep=", ", file=runlogf, flush=False)
                    # Leave is prioritized on other actions
                    if any(agent.leave for agent in pair):
                        logging.debug(f"pair {i_pair} with {pair[0].id} and {pair[1].id} breaks")
                        for agent in pair:
                            agent.clear_inputs()
                            leavers.add(agent)
                    else:  # If they play
                        next_pairs.append(pair)
                        logging.debug(f"pair {i_pair} with {pair[0].id} and {pair[1].id} plays")
                        total_coop = 0
                        total_spite = 0
                        for agent in pair:
                            logging.debug(f"agent {agent.id} plays {agent.coop} and gains {payoff(agent, pair)}")
                            if not agent.fake:
                                agent.fitness += payoff(agent, pair)
                            total_coop += agent.coop
                            logging.debug(f"agent {agent.id} spite {agent.spite}")
                            if not agent.fake:
                                agent.fitness -= agent.spite
                            total_spite += agent.spite
                            for o_agent in pair:
                                if o_agent != agent and not o_agent.fake:
                                    o_agent.fitness -= 3 * agent.spite
                        for agent in pair:  # update inputs
                            agent.knows_partner = 1
                            agent.prev_coop = total_coop - agent.coop
                            agent.prev_punish = total_spite - agent.spite

                # Removing broken appart pairs, and adding the agent to the pool
                alone_pool |= leavers
                pairs = next_pairs
            # before starting a new run, let's flush the logs
            if log:
                runlogf.flush()
            # end steps
        # end runs
        if log:
            runlogf.close()
        fitnesses = np.array([agent.fitness for agent in population])
        fitnesslogf = gzip.open(logdir / 'fitnesslog.txt.gz', 'at')
        for agent in population:
            print(igen, agent.id, agent.fitness, sep=',', file=fitnesslogf, flush=False)
        fitnesslogf.close()
        i_max = np.argmax(fitnesses)
        logging.info(population[i_max].fulldesc())
        print(igen, np.min(fitnesses), np.median(fitnesses), np.max(fitnesses))
        min_fit = np.min(fitnesses)
        if min_fit >= 0:
            min_fit = -1
        fitnesses -= min_fit
        sumfit = np.sum(fitnesses)
        parents = np.random.choice(population, size=popsize, p=fitnesses/sumfit, replace=True)
        population = [Agent(i, agent) for i, agent in enumerate(parents)]
        [agent.mutate() for agent in np.asarray(np.random.choice(population, size=(1,)))]
    runlogf.close()
    fitnesslogf.close()


ap = argparse.ArgumentParser()
ap.add_argument("--no-partner-choice", action="store_true", dest="partner_control")
ap.add_argument("--no-partner-interaction", action="store_false", dest="partner_interaction")
ap.add_argument("-d", "--dir", type=Path, default=Path('.'))
ap.add_argument("-v", "--verbosity", action="count", default=0)
args = ap.parse_args()
PARTNER_INTERACTION = args.partner_interaction
PARTNER_CONTROL = args.partner_control
logging.basicConfig(level=int(40-args.verbosity*10))
logdir = args.dir
makedirs(logdir, exist_ok=True)

main()
