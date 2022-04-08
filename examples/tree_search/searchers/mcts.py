from typing import Callable, List, Tuple, Dict

import botbowl
import examples.tree_search as ts
import examples.tree_search.searchers.search_util as search_util
import examples.tree_search.hashmap as hashmap
import numpy as np
from functools import partial


def setup_node(new_node: ts.ActionNode, game: botbowl.Game, policy):
    if new_node.info is None:
        _, probabilities, actions_ = policy(game)
        num_actions = len(actions_)
        heuristic = np.array(search_util.get_heuristic(game))
        reward = np.zeros(shape=heuristic.shape)

        if new_node.parent is not None:
            for parent in new_node.get_all_parents(include_self=False):
                if isinstance(parent, ts.ActionNode):
                    reward = heuristic - parent.info.heuristic
                    break

        new_node.info = ts.MCTS_Info(probabilities=probabilities / probabilities.mean(),
                                     actions=actions_,
                                     action_values=np.zeros((num_actions, len(reward))),
                                     visits=np.zeros(num_actions, dtype=np.int),
                                     heuristic=heuristic,
                                     reward=reward,
                                     state_value=0)


def vanilla_mcts(root_node: ts.ActionNode,
                 game: botbowl.Game,
                 nodes: Dict[str, ts.ActionNode],
                 policy: search_util.Policy,
                 weights: search_util.HeuristicVector,
                 sample_action: Callable[[ts.ActionNode, ts.HeuristicVector], botbowl.Action],
                 cc_cond: search_util.ContinueCondition = None,
                 ) -> None:

    assert hashmap.create_gamestate_hash(game) == root_node.simple_hash
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

    node = root_node
    back_prop_queue: List[Tuple[ts.ActionNode, botbowl.Action]] = []

    while continue_expension(node):
        action = sample_action(node, weights)
        back_prop_queue.append((node, action))

        game.step(action)

        # figure out if the new state already existed.
        node_hash = hashmap.create_gamestate_hash(game)

        if node_hash in nodes:
            node = nodes[node_hash]
        else:
            node = ts.ActionNode(game, parent=None)
            setup_node(node, game, policy)
            nodes[node_hash] = node

    # do backpropagation
    propagated_value = np.copy(node.info.reward)
    while len(back_prop_queue) > 0:
        node, action = back_prop_queue.pop()
        action_index = node.info.actions.index(action)
        node.info.action_values[action_index] += propagated_value
        propagated_value += node.info.reward
