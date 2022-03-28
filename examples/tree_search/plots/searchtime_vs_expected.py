import pickle
from functools import partial
from time import perf_counter

import botbowl
import examples.tree_search as ts
import matplotlib.pyplot as plt
import numpy as np
from examples.tree_search import MCTS_Info
import examples.tree_search.evaluation_scenarios as scenarios
from tests.util import get_custom_game_turn

times_evaluated = 3
num_rollouts = 5000

search_heuristics = [
    #ts.HeuristicVector(score=1, ball_marked=0.05, ball_carried=0.2, ball_position=0.01, tv_on_pitch=0),
    ts.HeuristicVector(score=1, ball_marked=0, ball_carried=0, ball_position=0, tv_on_pitch=0),
]


evaluation_heuristic = ts.HeuristicVector(score=1, ball_marked=0, ball_carried=0, ball_position=0, tv_on_pitch=0)
#evaluation_heuristic = search_heuristics[0]

def calculate_expected_value(node):
    return max(ts.get_node_value(child, evaluation_heuristic) for child in node.children)


def get_best_action_value(node):
    info: MCTS_Info = node.info
    return max(np.matmul(info.action_values, evaluation_heuristic) / (info.visits + (info.visits == 0)))


def main(tree_searcher):
    game = scenarios.five_player_hopeless()

    vals = np.zeros(shape=(len(search_heuristics), times_evaluated, num_rollouts, 3))

    for i, search_heuristic in enumerate(search_heuristics):
        trees = [ts.SearchTree(game) for _ in range(times_evaluated)]
        policy = ts.MockPolicy()
        start_time = perf_counter()

        for k in range(num_rollouts):
            # explore
            for j, tree in enumerate(trees):
                tree_searcher(tree, policy, weights=search_heuristic, exploration_coeff=1)

                # What is the expected value
                if tree_searcher is ts.deterministic_tree_search_rollout:
                    expected_value = calculate_expected_value(tree.root_node)
                elif tree_searcher is ts.mcts_ucb_rollout:
                    expected_value = get_best_action_value(tree.root_node)
                else:
                    raise ValueError()

                num_nodes = len(tree.all_action_nodes)

                vals[i, j, k, 0] = expected_value
                vals[i, j, k, 1] = num_nodes
                vals[i, j, k, 2] = perf_counter() - start_time

            if k > 20 and k % 20 == 0:
                with open("searchtime_vs_expected.data", 'wb') as f:
                    pickle.dump(obj=vals, file=f)

                for j in range(len(trees)):
                    plt.plot(vals[i, j, :k, 1],
                             vals[i, j, :k, 0],
                             label='expected value')
                plt.draw()
                plt.pause(0.1)
                print(f"new plot available {k}  {vals[i, :, k, 1]}")


if __name__ == "__main__":
    main(ts.deterministic_tree_search_rollout)
    #main(ts.mcts_ucb_rollout)
