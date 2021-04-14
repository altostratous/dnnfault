from matplotlib import pyplot as plt
import pickle

from matplotlib.ticker import PercentFormatter

variants = (('randbet', 'RandomBET'),
            ('smoothing', 'Smoothing'))

attacks = (('none', 'None'),
           ('spray', 'Spray'),
           ('upspray', 'Up Spray'))

step = 2
resolution = 1000
interval = (60, 75)

for i, attack in enumerate(attacks):
    attack_key, attack_title = attack
    plt.subplot(len(attacks), 1, i + 1)
    for variant_key, variant_title in variants:
        with open(attack_key + '.pkl', mode='rb') as f:
            evaluation = pickle.load(f)
        probability = []
        x = []
        for p in range(int(interval[0] / 100 * resolution) - 1,
                       int(interval[1] / 100 * resolution) + 1, step):
            percentage = p / resolution
            x.append(percentage)
            # probability.append(len([i[variant_key] for i in evaluation if i[variant_key] < percentage]) / len(evaluation))
            probability.append(len([i[variant_key] for i in evaluation if percentage - step / resolution <= i[variant_key] < percentage]) / len(evaluation))
        plt.plot(x, probability, label=variant_title)
        plt.legend()
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        plt.gca().xaxis.set_major_formatter(PercentFormatter(1))
        plt.title(attack_title)
plt.tight_layout()
plt.show()