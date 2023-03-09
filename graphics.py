import matplotlib.pyplot as plt
import numpy as np


def plot_scatter(predicted, target, r, title):
    fig = plt.figure()
    grid = plt.GridSpec(4, 4)

    ax_main = fig.add_subplot(grid[1:4, 0:3])
    ax_right = fig.add_subplot(grid[1:4, 3])
    ax_top = fig.add_subplot(grid[0, 0:3])

    c = 0.003 * target / 0.003 * predicted
    ax_main.scatter(x=predicted, y=target, c=c)
    ax_main.set_xlabel('predicted')
    ax_main.set_ylabel('target')

    ax_top.hist(predicted, histtype='stepfilled', orientation='vertical')

    ax_right.hist(target, histtype='stepfilled', orientation='horizontal')

    fig.suptitle(f'{title} \n R: {r}')


def plot_features(features, scores, title):
    plt.rcdefaults()
    fig, ax = plt.subplots()
    ax.barh(np.arange(len(features)), scores)
    ax.set_yticks(np.arange(len(features)), labels=features)
    ax.invert_yaxis()
    ax.set_xlabel('MI score')
    ax.set_title(f'{title}')
