
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class BanditEnv():
    def __init__(self, numBandits):
        self.numBandits = numBandits
        self.time = 0
        # A list of tupless in the form of
        # (expectedValue, variance, ? drift)
        self.bandits = []
        self.setBandits()
    def setBandits(self):
        values = np.random.uniform(-5, 5, (self.numBandits))
        variance = np.random.uniform(1, 5, (self.numBandits))
        self.bandits = np.stack([values, variance], axis=1)
    def select(self, option):
        p = self.bandits[option]
        return np.random.normal(*p)
    def peekValues(self):
        return self.bandits[: , 0]

if __name__ == '__main__':
    simpleEnv = BanditEnv(5)
    test = [[simpleEnv.select(i) for j in range(100)] for i in range(5)]
    ax = sns.violinplot(data=test)



