from zad1a import KnapSack
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from random import randint
import gc


def make_graph(graph_title: str, x_coords: list[int], y_coords: list[int]) -> None:
    plt.clf()
    ax = plt.subplot()
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    plt.plot(x_coords, y_coords)
    plt.xlabel('number of items to choose from')
    plt.ylabel('time taken to choose (sec)')
    plt.title(graph_title)
    plt.savefig(f'{graph_title}.png')
    plt.show()


def measure_time() -> np.array[float]:
    weights = np.array([randint(1, 12) for x in range(1, 25)])
    capacity = 9
    profits = np.array([randint(1, 12) for x in range(1, 25)])
    times_measured = np.array([])
    knapsack = KnapSack(profits[:4], weights[:4], capacity)
    for i in range(1, 22):
        gc.collect()
        start = time.process_time()
        knapsack.solve_knapsack_brute_force()
        end = time.process_time()
        times_measured = np.append(times_measured, end - start)

        knapsack.weights = weights[:4+i]
        knapsack.profits = profits[:4+i]

    return times_measured

def main():
    x_coords = [x for x in range(4, 25)]
    y_coords = measure_time()
    print(x_coords)
    print(y_coords)
    make_graph("Brute Force Knapsack Time", x_coords, y_coords)

if __name__ == "__main__":
    main()