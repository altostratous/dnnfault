import os
import pickle
from collections import defaultdict

import copy

import math
from matplotlib import pyplot as plt

filenames = os.listdir('result_2')
accuracy = []


def deserialize(filename):
    filename, _ = os.path.splitext(filename)
    results = {}
    for i in filename.split('-'):
        key, value = i.split(':')
        results[key] = value
    if 'mc_dropout' not in results:
        results['mc_dropout'] = 'False'
    return results


has_error = False
for filename in filenames:
    if filename.endswith('.lock'):
        continue
    try:
        accs = pickle.load(open('results_1/' + filename, mode='rb'))
        z = 1.96  # 95%
        accs_ = sum(accs) / len(accs)
        accuracy.append((accs_, filename, accs, deserialize(filename), math.sqrt(accs_ * (1 - accs_) / len(accs) / 79)))
    except (pickle.UnpicklingError, EOFError):
        print(filename)
        has_error = True
if has_error:
    exit()
conf_vals = defaultdict(set)

for e in accuracy:
    # if e[3]['e_mapping'] != 'RowHammerSprayAttackMapping':
    #     continue
    for k, v in e[3].items():
        conf_vals[k].add(v)

for k, vals in conf_vals.items():
    for v in vals:
        accs = [a[0] for a in accuracy if a[3][k] == v]
        print(k, v, sum(accs) / len(accs))

uvul, svul = [], []
for i in range(len(accuracy)):
    for j in range(i + 1, len(accuracy), 1):
        if (all(accuracy[i][3][k] == accuracy[j][3][k] for k in accuracy[i][3] if k != 'e_mapping') and
            {accuracy[i][3]['e_mapping'], accuracy[j][3]['e_mapping']} == {'RowHammerSprayAttackMapping', 'RowHammerUpSprayAttackMapping'}):
            if accuracy[i][3]['e_mapping'] == 'RowHammerSprayAttackMapping':
                spray = accuracy[i]
                up_spray = accuracy[j]
            else:
                up_spray = accuracy[i]
                spray = accuracy[j]
            if abs(spray[0] - up_spray[0]) < 0.05:
                continue
            if spray[0] < up_spray[0]:
                svul.append((spray, up_spray))
            else:
                uvul.append((spray, up_spray))
print(len(uvul), len(svul))
print(len([s for s, u in uvul if (s[3]['t_mapping'] == 'RandomBETMapping')]),
      len([s for s, u in svul if not(s[3]['t_mapping'] == 'RandomBETMapping')]))
print(*[s for s, u in uvul if (s[3]['t_mapping'] != 'RandomBETMapping')], sep='\n')

print(*[a for a in accuracy if (
        # a[3]['t_mapping'] == 'RandomBETMapping' and
        a[3]['t_mapping'] == 'None' and
        a[3]['sigma'] == '0.0' and
        a[3]['e_mapping'] == 'RowHammerSprayAttackMapping' and
        # a[3]['e_mapping'] == 'None' and
        a[3]['dropout_p'] == '0.5' and
        a[3]['first_dropout'] == 'True' and
        a[3]['last_dropout'] == 'False' and
        a[3]['mc_dropout'] == 'False' and
        a[3]['weight_clip'] == 'True'
)], sep='\n')


base_filters = {
    't_mapping': 'None',
    'sigma': '0.0',
    'e_mapping': 'None',
    'dropout_p': '0.5',
    'first_dropout': 'True',
    'last_dropout': 'False',
    'mc_dropout': 'False',
    'weight_clip': 'False',
}

def _get_accuracy(filters):
    filters = {
        **base_filters,
        **filters
    }
    applicable = [a for a in accuracy if all(a[3][k] == v for k, v in filters.items())]
    assert len(applicable) == 1
    return applicable[0]


def get_accuracy(filters):
    return _get_accuracy(filters)[0]


plt.figure(figsize=(12, 8))
for attack in ('RowHammerUpSprayAttackMapping',
               'RowHammerSprayAttackMapping'
               ):

    color = 'blue' if attack == 'RowHammerSprayAttackMapping' else 'orange'
    legend = ', Spray' if attack == 'RowHammerSprayAttackMapping' else ', UpSpray'
    y_01 = []
    y_025 = []
    y_08 = []
    y_last = []
    y_wc = []
    x_01 = []
    x_025 = []
    x_08 = []
    x_last = []
    x_wc = []
    for acc in [a for a in accuracy if (
        a[3]['e_mapping'] == attack and
        a[3]['mc_dropout'] == 'False' and
        a[3]['first_dropout'] == 'True' and
        a[3]['t_mapping'] != 'RandomBET' and
        a[0] > 0.60
    )]:
        x = get_accuracy({**acc[3], 'e_mapping': 'None'})
        y = acc[0]
        if acc[3]['sigma'] == '0.1':
            x_01.append(x)
            y_01.append(y)
        if acc[3]['sigma'] == '0.25':
            x_025.append(x)
            y_025.append(y)
        if acc[3]['last_dropout'] == 'True':
            x_last.append(x)
            y_last.append(y)
        if acc[3]['weight_clip'] == 'True':
            y_wc.append(y)
            x_wc.append(x)
        if acc[3]['dropout_p'] == '0.8':
            x_08.append(x)
            y_08.append(y)
    plt.scatter(x_08, y_08, s=32, linewidth=1, facecolor=color, marker=5, label='q=0.8' + legend)
    plt.scatter(x_wc, y_wc, s=32, linewidth=1, facecolor=color, marker='_', label='Weight Clip' + legend)
    plt.scatter(x_last, y_last, s=32, linewidth=1, facecolor=color, marker='|', label='Last Dropout' + legend)
    plt.scatter(x_025, y_025, s=128, linewidth=1, facecolor='none', marker='o', edgecolors=[color], label='σ=0.25'+ legend)
    plt.scatter(x_01, y_01, s=64, linewidth=1, facecolor='none', marker='o', edgecolors=[color], label='σ=0.1' + legend)
plt.legend()
plt.xlabel('Clean Accuracy')
plt.ylabel('Robust Accuracy')
plt.title('Trade-off')
plt.savefig('../../adversehw/doc/tradeoff.png')
plt.show()

with open('../../adversehw/doc/fullresults.tex', mode='w', encoding='utf-8') as f:
    f.write('hello')


def get_tabular_key(acc):
    key = (acc[3]['weight_clip'],
           acc[3]['t_mapping'] == 'RandomBETMapping',
           acc[3]['dropout_p'],
           acc[3]['sigma'],
           acc[3]['first_dropout'],
           acc[3]['last_dropout'],
           acc[3]['mc_dropout'])
    return key

tabular = defaultdict(dict)
for acc in accuracy:
    key = get_tabular_key(acc)
    tabular[key][acc[3]['e_mapping']] = (100 * acc[0], 100 * acc[4])

cc = 0
with open('../../adversehw/doc/fullresults.tex', mode='w') as f:
    for key, value in sorted(tabular.items()):
        print(*key, '{:.2f}±{:.2f}'.format(*value['None']),
              '{:.2f}±{:.2f}'.format(*value['RowHammerSprayAttackMapping']),
              '{:.2f}±{:.2f}'.format(*value['RowHammerUpSprayAttackMapping']),
              sep=' & ', end=' \\\\ \n \hline \n',  file=f)
print(len(tabular))


def plot_accuracy(title, variants, mutations=None, size=None):

    if mutations is None:
        mutations = (({'e_mapping': 'None'}, 'Clean'),
                    ({'e_mapping': 'RowHammerSprayAttackMapping'}, 'Bit Error Rate 1%'),
                    # ({'e_mapping': 'RowHammerUpSprayAttackMapping'}, 'UpSpray')
                     )

    if size is not None:
        plt.figure(figsize=size)

    for i, (filters, variant) in enumerate(variants):
        x = []
        yerr = []
        x_tick_labels = []
        for mutation in mutations:
            filters.update(mutation[0])
            x_tick_labels.append(mutation[1])
            x.append(get_accuracy(filters))
            yerr.append(_get_accuracy(filters)[4])
        width = 1 / (len(variants) + 1)
        plt.bar([j + width / 2 + i * width for j in range(len(x))],
                height=x, yerr=yerr, capsize=5, width=width, align='edge' if (len(variants) % 2 == 0) else 'center',
                label=variant, tick_label=x_tick_labels if i == len(variants) // 2 else None,)
    plt.legend()
    plt.title(title[1])
    plt.ylabel('Accuracy')
    plt.savefig('../../adversehw/doc/' + title[0] + '.png')
    plt.show()


plot_accuracy(('bestofbests', 'Robustness of Compound Methods'),
              (
              ({}, 'Original AlexNet'),
              # ({'sigma': '0.25', 'last_dropout': 'True'}, 'Thinner'),
              ({'weight_clip': 'True'}, 'Weight Clip'),
              # ({'t_mapping': 'RandomBETMapping', 'weight_clip': 'True'}, 'RandBET'),
              ), size=(12, 4))

plot_accuracy(('dropoutrobustness', 'MC Dropout Robustness'),
              (
              ({'dropout_p': '0.5', 'mc_dropout': 'True'}, 'q = 0.5, 2-layers'),
              ({'dropout_p': '0.5', 'mc_dropout': 'True', 'last_dropout': 'True'}, 'q = 0.5, 3-layers'),
              ({'dropout_p': '0.8', 'mc_dropout': 'True'}, 'q = 0.8, 2-layers'),
              ({'dropout_p': '0.8', 'mc_dropout': 'True', 'last_dropout': 'True'}, 'q = 0.8, 3-layers'),
              ), size=(12, 4))

plot_accuracy(('grsrobustness', 'Robustness'),
               (({}, 'Original AlexNet'),
                ({'sigma': '0.25'}, 'Random Smoothing'),
              ({'t_mapping': 'RandomBETMapping', 'weight_clip': 'True'}, 'RandBET'),
                ({'weight_clip': 'True'}, 'Weight Clip'),
                ))