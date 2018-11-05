import numpy as np
from bandits.env.BanditEnv import BanditEnv
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")

class EpsilonGreedy():
    def __init__(self, numBandits, epsilon):
        self.optimism = 3
        self.numBandits = numBandits
        self.epsilon = epsilon
        self.clock = 0

        self.expectedVals = self.optimism * np.ones(numBandits)
        self.banditCounter = np.zeros(numBandits)
        self.env = BanditEnv(numBandits)

        self.actionHist = []
        self.rewardHist = [[] for i in range(numBandits)]
    def choose(self):
        if np.random.uniform(0, 1) > self.epsilon:
            # Exploit by choosing the option with the greatest return
            option = np.argmax(self.expectedVals)
        else:
            # Explore a random option
            option = np.random.randint(0, self.numBandits)
        self.banditCounter[option] += 1
        self.actionHist.append(option)
        return option
    def updateVal(self, option, reward):
        # Update the histories
        self.rewardHist[option].append(reward)
        n = self.banditCounter[option]
        self.expectedVals[option] += 1 / n * (reward - self.expectedVals[option])
    def step(self):
        option = self.choose()
        # print(option)
        reward = self.env.select(option)
        self.updateVal(option, reward)
        self.clock += 1
         # print("{0}: {1}".format(self.clock, self.expectedVals))
    def getClock(self):
        return self.clock

if __name__ == '__main__':
    greedy = EpsilonGreedy(5, 0.1)
    while(greedy.getClock() < 300):
        greedy.step()
    plt.figure(0)
    sns.violinplot(data=greedy.rewardHist)
    plt.figure(1)
    plt.plot(greedy.actionHist)