from bandits.env.BanditEnv import BanditEnv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")


class BanditDriftEnv(BanditEnv):
    def __init__(self, numBandits):
        BanditEnv.__init__(self, numBandits)

    def select(self, option):
        p = self.bandits[option]
        self.bandits[:, 0] += np.random.normal(0, 0.05, self.numBandits)
        return np.random.normal(*p)

    def setBandits(self):
        values = np.ones(self.numBandits)
        variance = 0.5 * np.ones(self.numBandits)
        self.bandits = np.stack([values, variance], axis=1)

if __name__ == '__main__':
    bandits = 5
    history = np.ones(5)
    drift = BanditDriftEnv(5)
    for i in range(50):
        drift.select(0)
        newMeans = drift.peekValues()
        history = np.vstack((history, newMeans))
    # print(history)
    print(history[:, 0])
    plt.plot(history)
    plt.show()

