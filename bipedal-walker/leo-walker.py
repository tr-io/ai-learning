import gym
import numpy as np
import random
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as factor
import torch.optim as optim
import copy
import math
from gym.wrappers import Monitor
from collections import OrderedDict
import torch.distributions as tdist
# use q-learning
# bipedal-walker has 24 objects in each state
# size 4 array for action

# Initialize Q(s1-24, action)
# Observe current state (s1-24)
# Based on exploration strategy, choose action to take, a
# Take a and observe resulting reward, r, and new state of the environment, (s1'-24')
# Update Q w/ this rule:
# Q'(s1-24, a) = (1 - w) * Q(s1-24) + w(r+d*Q(s1'-24', argmax a'Q(s1'-24', a')))
# w is learning rate, d is discount rate
# repeat 2-5 until convergence

# use an epsilon-greedy exploration strategy

# env - environment
# lr - learning rate
# discount - discount factor
# epsilon - epsilon for epsilon-greedy
# min_eps - minimum episodes
# eps - max episodes
def QLearning(env, lr, discount, epsilon, min_eps, eps):
    qtable = np.zeros((24, 4))

    
    if random.uniform(0, 1) < epsilon:
        # exploration: select random action
        print("not defined")
    else:
        # exploit: select action with max q-value
        print("not defined")

"""
Helper Functions for GeneticAlgorithm
"""

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

"""
NeuralNetwork class using Multilayer Perceptron
- Started out with 2 layers (1 hidden)
"""
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        # first layer
        self.weights1 = np.random.randn(self.input_size, self.hidden_size)
        self.bias1 = np.random.randn(self.hidden_size)
        # second layer
        self.weights2 = np.random.randn(self.hidden_size, self.output_size)
        self.bias2 = np.random.randn(self.output_size)

    def feedforward(self, input):
        layer1 = sigmoid(np.dot(input, self.weights1) + self.bias1)
        layer2 = sigmoid(np.dot(layer1, self.weights2) + self.bias2)
        return layer2

# Helper function to get a neural net
# doing a neural net from scratch is hard lmfao
def gen_model(i_size, h_size, o_size):
    return torch.nn.Sequential(
        torch.nn.Linear(i_size, h_size).cuda(),
        torch.nn.ReLU().cuda(),
        torch.nn.Linear(h_size, o_size).cuda(),
        torch.nn.Sigmoid().cuda()
    )

# Helper function to calculate the fitness function of an individual agent
def run_agent(model, eps=500, render=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    env = gym.make('BipedalWalker-v3')
    obs = env.reset()
    fitness = 0.0
    for _ in range(eps):
        if render:
            env.render()
        obs = torch.from_numpy(obs).float().to(device)
        action = (model(obs).detach()).cpu().numpy()
        obs, reward, done, info = env.step(action)
        fitness += reward
        if done:
            break
    
    # just for getting the model parameters and stuff
    # parameters are weights and biases and then we concat them into one singular object
    weights = model.state_dict().values()
    weights = [w.flatten() for w in weights]
    weights = torch.cat(weights, 0)

    env.close()

    return fitness, weights

# Data structure for an Agent
class WalkerAgent:
    # Walker properties:
    # input (obs space) size: 24
    # hidden layer size: arbitrary (choosing 16)
    # output (action space) size: 4
    def __init__(self):
        self.input_size = 24
        self.hidden_size = 16
        self.output_size = 4

        # instantiate new neural net for the walker
        self.model = gen_model(self.input_size, self.hidden_size, self.output_size).cuda()

        # reward/fitness
        self.fitness = 0.0
        self.weights = None
    
    def calculate_fitness(self):
        self.fitness, self.weights = run_agent(self.model)

    def update_model(self):
        self.model.load_state_dict(weights_to_statedict(self.model.state_dict(), self.weights))
    
def crossover(child1_weights, child2_weights):
    # randomly choose crossover point
    cross_point = np.random.randint(0, child1_weights.shape[0])
    child1_copied_weights = child1_weights.clone()
    child2_copied_weights = child2_weights.clone()

    # now just do some swapping
    # crossover: pick crossover point
    # then from that point, we take the weights (genes) from child1 from 0 - cross point
    # and swap them with child2's weights (genes) from 0 - cross point
    temp = child1_copied_weights[:cross_point].clone()
    child1_copied_weights[:cross_point] = child2_copied_weights[:cross_point]
    child2_copied_weights[:cross_point] = temp

    return child1_copied_weights, child2_copied_weights

# mutation function
def mutate(parent_weights, p=0.6):
    child_weights = parent_weights.clone()
    if np.random.rand() < p:
        cross_point = np.random.randint(0, parent_weights.shape[0])
        n = tdist.Normal(child_weights.mean(), child_weights.std())
        child_weights[cross_point] = n.sample() + np.random.randint(-20, 20)
    
    return child_weights

# method to turn weights back into torch tensor state dict
# taken from github: 
def weights_to_statedict(model_dict, model_weights) -> OrderedDict:
    shapes = [x.shape for x in model_dict.values()]
    shapes_prod = [torch.tensor(s).numpy().prod() for s in shapes]

    partial_split = model_weights.split(shapes_prod)
    model_values = []
    for i in range(len(shapes)):
        model_values.append(partial_split[i].view(shapes[i]))
    
    state_dict = OrderedDict((key, value) for (key, value) in zip(model_dict.keys(), model_values))
    return state_dict

def run_video_agent(model, eps=500):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    env = gym.make('BipedalWalker-v3')
    env = Monitor(env, './vid', video_callable=lambda episode_id: True, force=True)
    obs = env.reset()
    last_obs = obs

    fitness = 0.0

    for _ in range(eps):
        env.render()

        obs = torch.from_numpy(obs).float().to(device)
        action = (model(obs).detach()).cpu().numpy()
        new_obs, reward, done, info = env.step(action)
        fitness += reward
        obs = new_obs

        if done:
            break

    env.close()
    print("Best score ", fitness)
"""
Parameters for GeneticAlgorithm:
pop: population size
top_limit: how many agents we take to use for reproduction; the top # of agents we take
eps: generations to run; number of iterations we want to loop

Pseudocode for the genetic algorithm approach:
START
Generate the initial population
Compute fitness
REPEAT
    Selection
    Crossover
    Mutation
    Compute fitness
UNTIL population has converged
STOP
"""
def GeneticAlgorithm(pop, top_limit, gen):
    # Initialize population
    population = []
    for _ in range(pop):
        new_agent = WalkerAgent()
        population.append(new_agent)

    # Start main loop
    for _ in range(gen):
        print("\n\n\n")
        print("Running generation ", _)
        print("\n")
        # Compute fitness for each fucking agent
        print("Computing fitness...")
        for p in population:
            p.calculate_fitness()

        # each agent should have its self.fitness values calculated
        # Selection time
        # sort the populations by fitness
        print("Doing selection...")
        sorted_pop = sorted(population, key=lambda x: x.fitness, reverse=True) # sort in desc order
        sorted_pop = sorted_pop[:top_limit] # get top_limit # of top scoring individuals

        print("\n")
        print("Best scores")
        for x in range(0, len(sorted_pop)):
            print("top ", (x + 1), " score: ", sorted_pop[x].fitness)

        # TODO: might add some random shuffling that way genetics is even more shuffled
        # right now its more like eugenics to get fastest and bestest children quickly
        # i think shuffling would slow it down tho

        # Crossover time
        # help from: https://www.geeksforgeeks.org/crossover-in-genetic-algorithm/
        # "Crossover is sexual reproduction" (GeeksForGeeks)
        # from now on i require anyone who reads this and wants to have kids
        # to say "ayy bby lets do some crossover ;)"
        # parent selection: https://www.tutorialspoint.com/genetic_algorithms/genetic_algorithms_parent_selection.htm
        # roulette wheel selection
        sum_fit = abs(sum(x.fitness for x in sorted_pop))
        print("Crossover time...")
        new_pop = []
        for i in range(0, len(population) - 1):
            # roulette selection:
            # pick random number between 0 and sum_fit
            # for parent 1 (parent a)
            pa_fit = random.uniform(0.0, sum_fit)
            pb_fit = random.uniform(0.0, sum_fit)

            pa_sum = 0.0
            pb_sum = 0.0
            parent1 = None
            parent2 = None
            for j in range(0, len(sorted_pop)):
                pa_sum += abs(sorted_pop[j].fitness)
                pb_sum += abs(sorted_pop[j].fitness)
                if pa_sum >= pa_fit:
                    parent1 = sorted_pop[j]
                if pb_sum >= pb_fit:
                    parent2 = sorted_pop[j]

            print("pa_fit, pb_fit: ", pa_fit, ", ", pb_fit)
            print("pa_sum, pb_sum: ", pa_sum, ", ", pb_sum)
            print("Selected parents: ", parent1, " and ", parent2)

            child1 = copy.deepcopy(parent1)
            child2 = copy.deepcopy(parent2)

            print("Crossing over weights...")
            child1.weights, child2.weights = crossover(parent1.weights, parent2.weights)

            # mutation step
            # mutation by normal distribution
            print("Mutation chance...")
            child1.weights = mutate(child1.weights)
            child2.weights = mutate(child2.weights)

            # Update the weights
            print("Updating weights...")
            child1.update_model()
            child2.update_model()

            # add it to new_pop, but add the higher scoring one first
            print("Adding children to new population...")
            child1.calculate_fitness()
            child2.calculate_fitness()

            child1_added = False
            if child1.fitness > child2.fitness:
                child1_added = True
                new_pop.append(child1)
            else:
                new_pop.append(child2)

            if len(new_pop) >= (len(population) - 1):
                break

            if child1_added:
                new_pop.append(child2)
            else:
                new_pop.append(child1)
        
        # add best scoring agent to the new pop
        new_pop.append(sorted_pop[0])

        # set population to be new population
        print("Setting new population...")
        population = copy.deepcopy(new_pop)
    
    # play best one
    print("Playing best agent!")
    sorted_pop = sorted(population, key=lambda x: x.fitness, reverse=True) # sort in desc order
    run_video_agent(sorted_pop[0].model)

# Main function for testing
def main():
    pop = 100
    top_limit = 20
    eps = 3
    #trials = 4

    """
    env = gym.make('BipedalWalker-v3')
    for _ in range(10):
        obs = env.reset()
        for t in range(500):
            env.render()
            print("\n\n")
            print(obs)
            print(obs.shape)
            print("\n\n")
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            if done:
                print("episode finished after {} timesteps".format(t+1))
        
    env.close()
    """

    GeneticAlgorithm(pop, top_limit, eps)

if __name__ == "__main__":
    main()