import numpy as np
import itertools as it
import matplotlib.pyplot as plt
import gc, time
from typing import Tuple, Union
from matplotlib.ticker import FormatStrFormatter
from random import randint

class KnapSack:
    def __init__(self, profits, weights, capacity):
        self.profits = profits
        self.weights = weights
        self.capacity = capacity
        self.item_id = [i for i in range(len(self.profits))]
        self.combo: list[Tuple[int, int, int]] = []
        # combo is a list containing tuples, each representing an item;
        for profit, weight, id in zip(self.profits, self.weights, self.item_id):
            self.combo.append((profit, weight, id))

    def solve_knapsack_brute_force(self) -> Union[Tuple[int, int, list[int]], None]:
        if not self.check_weights_and_profits():
            return None

        solution: Tuple[int, int, list[int]]
        max_profit: int = 0

        for comb_length in range(1, len(self.weights) + 1):
            combinations = it.combinations(self.combo, comb_length)

            for item in combinations:
                current_profit: int = 0
                current_weight: int = 0
                id_list: list[int] = []

                for profit, weight, id in item:
                    current_profit += profit
                    current_weight += weight
                    id_list.append(id)

                if current_weight <= self.capacity and current_profit > max_profit:
                    solution = (current_profit, current_weight, id_list)

        return solution

    def solve_knapsack_pw_ratio(self) -> Union[Tuple[int, int, list[int]], None]:
        if not self.check_weights_and_profits():
            return None

        pw_ratio = list(self.profits/self.weights)
        combo_with_pw_ratio: list[Tuple[int, int, int, float]] = []
        # combo_with_pw_ratio is a list containing tuples, each representing an item;
        for item, ratio in zip(self.combo, pw_ratio):
            combo_with_pw_ratio.append((item[0], item[1], item[2], ratio))

        combo_with_pw_ratio.sort(key=lambda x: x[3], reverse=True)

        current_profit: int = 0
        current_weight: int = 0
        id_list: list[int] = []

        for item in combo_with_pw_ratio:
            if current_weight + item[1] <= self.capacity:
                current_profit += item[0]
                current_weight += item[1]
                id_list.append(item[2])
            else: break

        return (current_profit, current_weight, id_list)

    def check_weights_and_profits(self):
        return True if len(self.weights) == len(self.profits) and len(self.weights > 0) else False

def make_graph(graph_title: str, x_coords: list[int], y_coords: list[int]) -> None:
    plt.clf()
    ax = plt.subplot()
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    plt.plot(x_coords, y_coords)
    plt.xlabel('number of items to choose from')
    plt.ylabel('time taken to choose (sec)')
    plt.title(graph_title)
    plt.savefig(f'{graph_title}.png')
    plt.show()

def measure_time(profits: list[int], weights: list[int], capacity: int) -> np.ndarray:
    times_measured = np.array([])
    knapsack = KnapSack(profits[:4], weights[:4], capacity)
    for i in range(1, 22):
        gc.collect()
        start = time.process_time()
        knapsack.solve_knapsack_brute_force()
        end = time.process_time()
        times_measured = np.append(times_measured, end - start)

        knapsack = KnapSack(profits[:4+i], weights[:4+i], capacity)

    return times_measured

def main():
    weights = np.array([8, 3, 5, 2])
    capacity = 9
    profits = np.array([16, 8, 9, 6])
    KS = KnapSack(profits, weights, capacity)
    answer_bf = KS.solve_knapsack_brute_force()
    answer_pwr = KS.solve_knapsack_pw_ratio()

    print(f"Solution (brute force): \n profit: {answer_bf[0]} \n weight: {answer_bf[1]} \n ID's of items taken: {answer_bf[2]}")
    print("\n")
    print(f"Solution (pw ratio): \n profit: {answer_pwr[0]} \n weight: {answer_pwr[1]} \n ID's of items taken: {answer_pwr[2]}")

    # measuring time of solving knapsack for 4 to 24 items
    number_of_items = 24
    for x in range(20):
      weights = np.append(weights, randint(1, 12))
      profits = np.append(profits, randint(1, 12))
    x_coords = [x for x in range(4, number_of_items+1)]
    y_coords = measure_time(profits, weights, capacity)
    # print(f"x coordinates are: {x_coords} \n y coordinates are: {y_coords} \n")
    make_graph("Brute force Knapsack Time", x_coords, y_coords)


if __name__ == '__main__':
    main()