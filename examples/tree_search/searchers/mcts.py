import random
from functools import partial
from typing import Callable, List, Tuple, Dict

import numpy as np

import botbowl
import examples.tree_search as ts
import examples.tree_search.hashmap as hashmap
import examples.tree_search.searchers.search_util as search_util


def setup_node(new_node: ts.ActionNode, game: botbowl.Game, policy):
    if new_node.info is None:
        _, probabilities, actions = policy(game)
        num_actions = len(actions)
        heuristic = np.array(search_util.get_heuristic(game))

        new_node.info = ts.MCTS_Info(probabilities=probabilities / probabilities.mean(),
                                     actions=actions,
                                     action_values=np.zeros((num_actions, len(heuristic))),
                                     visits=np.zeros(num_actions, dtype=np.int),
                                     heuristic=heuristic,
                                     reward=None,
                                     state_value=0)


def mcts_with_game_engine(root_node: ts.ActionNode,
                          game: botbowl.Game,
                          nodes: Dict[str, ts.ActionNode],
                          policy: ts.Policy,
                          weights: ts.HeuristicVector,
                          sample_action: Callable[[ts.ActionNode, ts.HeuristicVector], botbowl.Action],
                          cc_cond: search_util.ContinueCondition = None,
                          n: int = 1,
                          ) -> None:

    assert hashmap.create_gamestate_hash(game) == root_node.simple_hash
    assert game.trajectory.enabled
    step_num = game.get_step()
    assert step_num == root_node.step_nbr

    if cc_cond is None:
        cc_cond = search_util.ContinueCondition()

    weights = np.array(weights)

    turn_counter_adjust = 1 * (game.get_proc(botbowl.core.procedure.Kickoff) is not None or
                               game.get_proc(botbowl.core.procedure.LandKick) is not None)
    end_turn_at = search_util.get_team_turn_num(game, game.active_team) + turn_counter_adjust + cc_cond.turns

    continue_expension = partial(search_util.continue_expansion,
                                 game=game, cc_cond=cc_cond, scores=search_util.get_score_sum(game),
                                 half=game.state.half, end_turn_at=end_turn_at, team=game.active_team)

    setup_node(root_node, game, policy)

    for _ in range(n):
        node = root_node
        back_prop_queue: List[Tuple[ts.ActionNode, botbowl.Action]] = []

        while continue_expension(node):
            action = sample_action(node, weights)
            back_prop_queue.append((node, action))

            game.step(action)

            # figure out if the new state already exists.
            node_hash = hashmap.create_gamestate_hash(game)

            if node_hash in nodes:
                node = nodes[node_hash]
            else:
                node = ts.ActionNode(game, parent=None)
                setup_node(node, game, policy)
                nodes[node_hash] = node

        # do backpropagation
        final_heuristic = node.info.heuristic
        while len(back_prop_queue) > 0:
            node, action = back_prop_queue.pop()
            action_index = node.info.actions.index(action)
            node.info.action_values[action_index] += final_heuristic - node.info.heuristic

        game.revert(step_num)


def vanilla_action_sampling(node: ts.ActionNode, _) -> botbowl.Action:
    index = random.randint(0, len(node.info.actions)-1)
    node.info.visits[index] += 1
    return node.info.actions[index]


vanilla_mcts_rollout = partial(mcts_with_game_engine,
                               sample_action=vanilla_action_sampling)
