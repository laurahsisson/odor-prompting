# python3 -m charts.descriptor_frequency
from collections import Counter

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

import numpy as np

import util.read_dataset as dataset
INPUT_FILE = "all_lemmatized"
dataset.read_all_words(INPUT_FILE)

DESCRIPTOR_SAMPLE_COUNT = 20
MEAN_FREQUENCY_DEVIATION = .3

total = 0

freqs = []
for d, f in dataset.occurs.most_common():
    if f > 1:
        freqs.append(f)

lower_bound = np.quantile(freqs, .5 - MEAN_FREQUENCY_DEVIATION)
upper_bound = np.quantile(freqs, .5 + MEAN_FREQUENCY_DEVIATION)
print(lower_bound, upper_bound)
mean_descriptors = [
    d for d, f in dataset.occurs.most_common() if f > lower_bound and f < upper_bound
]

print(f"Found {len(dataset.occurs)} total dataset.occurs")
print("Most common")
print(dataset.occurs.most_common(DESCRIPTOR_SAMPLE_COUNT))
print("Middle frequency")
print(
    sorted([(d, dataset.occurs[d])
            for d in np.random.choice(mean_descriptors,
                                      size=DESCRIPTOR_SAMPLE_COUNT)],
           key=lambda x: x[1],
           reverse=True))
print("Least common")
print(dataset.occurs.most_common()[-1 * DESCRIPTOR_SAMPLE_COUNT:-1])

occurs_by_freq = Counter()
for descriptor, freq in dataset.occurs.items():
    occurs_by_freq[freq] += freq


occurs_by_freq = sorted(occurs_by_freq.items(), key=lambda of: of[0])
dataset.occurs = [f for f, o in occurs_by_freq]
occurs = [o for f, o in occurs_by_freq]
percentage = np.cumsum(occurs) / np.sum(occurs) * 100

#define aesthetics for plot
color1 = 'steelblue'
color2 = 'red'
line_size = 1

#create basic bar plot
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
import matplotlib.pyplot as plt
ax.set_xlabel('word frequency')

ax.set_xscale('log')

ax.bar(dataset.occurs, occurs, color=color1)
ax.set_ylabel('total occurences at word frequency')

#add cumulative percentage line to plot

ax2 = ax.twinx()
ax2.plot(dataset.occurs, percentage, color=color2, marker="", ms=line_size)
ax2.yaxis.set_major_formatter(PercentFormatter())
ax2.set_ylabel('total occurences at word frequency')

#specify axis colors
ax.tick_params(axis='y', colors=color1)
ax2.tick_params(axis='y', colors=color2)

# 
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300

import matplotlib.pyplot as plt
plt.savefig("figures/descriptor_frequency.svg")