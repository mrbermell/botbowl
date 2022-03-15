import pickle
from functools import partial
from time import perf_counter
import matplotlib.pyplot as plt

import botbowl
import numpy as np
from botbowl import Action, ActionType
import examples.tree_search as ts
from tests.util import get_custom_game_turn

times_evaluated = 5
num_rollouts = 500

search_heuristics = [
    #ts.HeuristicVector(score=1, ball_marked=0.1, ball_carried=0.2, ball_position=0.01, tv_on_pitch=1),
    ts.HeuristicVector(score=1, ball_marked=0, ball_carried=0, ball_position=0, tv_on_pitch=0),
    ts.HeuristicVector(score=1, ball_marked=0, ball_carried=0, ball_position=0, tv_on_pitch=0),
    ts.HeuristicVector(score=1, ball_marked=0, ball_carried=0, ball_position=0, tv_on_pitch=0),
    ts.HeuristicVector(score=1, ball_marked=0, ball_carried=0, ball_position=0, tv_on_pitch=0),
]

evaluation_heuristic = ts.HeuristicVector(score=1, ball_marked=0, ball_carried=0, ball_position=0, tv_on_pitch=0)

node_evaluation = partial(ts.get_node_value, weights=evaluation_heuristic)


def main():
    game, *_ = get_custom_game_turn(player_positions=[(7, 2), (7, 4), (10, 3)],
                                    opp_player_positions=[(4, 2), (4, 4), (6, 3)],
                                    ball_position=(7, 2),
                                    size=3,
                                    rerolls=0,
                                    pathfinding_enabled=True)

    env_conf = botbowl.ai.env.EnvConf(size=3, pathfinding=True)
    env = botbowl.ai.env.BotBowlEnv(env_conf)
    env.reset(skip_observation=True)
    env.game = game
    renderer = botbowl.ai.env_render.EnvRenderer(env)
    renderer.render()

    input("quit")


    vals = np.zeros(shape=(len(search_heuristics), times_evaluated, num_rollouts, 3))

    for i, search_heuristic in enumerate(search_heuristics):
        trees = [ts.SearchTree(game) for _ in range(times_evaluated)]
        policy = ts.MockPolicy()
        start_time = perf_counter()

        for k in range(num_rollouts):
            # explore
            for j, tree in enumerate(trees):
                ts.do_mcts_branch(tree, policy, weights=search_heuristic, exploration_coeff=1)
                # What is the expected value
                expected_value = max(map(node_evaluation, tree.root_node.children))
                num_nodes = len(tree.all_action_nodes)

                vals[i, j, k, 0] = expected_value
                vals[i, j, k, 1] = num_nodes
                vals[i, j, k, 2] = perf_counter() - start_time

            if k > 10 and k % 10 == 0:
                with open("searchtime_vs_expected.data", 'wb') as f:
                    pickle.dump(obj=vals, file=f)

                for j in range(len(trees)):
                    plt.plot(vals[i, j, :k, 1],
                             vals[i, j, :k, 0],
                             label='expected value')
                plt.draw()
                plt.pause(0.1)
                print(f"new plot available {k}  {vals[i, :, k, 1]}")


#def only_plot():
#    with open()
#    vals = pickle.load()


if __name__ == "__main__":
    main()
