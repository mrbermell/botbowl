import queue
from functools import partial
from typing import List

import botbowl
import examples.tree_search.searchers.search_util as search_util
import numpy as np
from examples.tree_search.SearchTree import SearchTree, ActionNode, ChanceNode
from pytest import approx


def mcts_ucb_rollout(tree: SearchTree,
                     policy: search_util.Policy,
                     weights: search_util.HeuristicVector,
                     exploration_coeff=1,
                     cc_cond: search_util.ContinueCondition = None) -> None:

    if cc_cond is None:
        cc_cond = search_util.ContinueCondition()

    weights = np.array(weights)
    tree.set_game_to_node(tree.root_node)
    game = tree.game

    scores = search_util.get_score_sum(game)
    half = game.state.half
    my_team = game.active_team

    turn_counter_adjust = 1 * (game.get_proc(botbowl.core.procedure.Kickoff) is not None or
                               game.get_proc(botbowl.core.procedure.LandKick) is not None)
    end_turn_at = search_util.get_team_turn_num(game, my_team) + turn_counter_adjust + cc_cond.turns

    continue_expension = partial(search_util.continue_expansion,
                                 game=game, cc_cond=cc_cond, scores=scores, half=half,
                                 end_turn_at=end_turn_at, team=my_team)

    def setup_node(new_node: ActionNode):
        if type(new_node.info) is not search_util.MCTS_Info:
            tree.set_game_to_node(new_node)
            _, probabilities, actions_ = policy(tree.game)
            num_actions = len(actions_)
            heuristic = np.array(search_util.get_heuristic(tree.game))
            reward = np.zeros(shape=heuristic.shape)

            if new_node.parent is not None:
                for parent in new_node.get_all_parents(include_self=False):
                    if isinstance(parent, ActionNode):
                        reward = heuristic - parent.info.heuristic
                        break

            new_node.info = search_util.MCTS_Info(probabilities=probabilities / probabilities.mean(),
                                                  actions=actions_,
                                                  action_values=np.zeros((num_actions, len(reward))),
                                                  visits=np.zeros(num_actions, dtype=np.int),
                                                  heuristic=heuristic,
                                                  reward=reward,
                                                  state_value=0)

    def back_propagate(final_node: ActionNode):
        propagated_value = np.copy(final_node.info.reward)  # todo: add final_node.info.state_value too

        n = final_node
        while True:
            if isinstance(n.parent, ChanceNode):
                pass
            elif isinstance(n.parent, ActionNode):
                action_object = n.parent.get_child_action(n)
                action_index = n.parent.info.actions.index(action_object)
                n.parent.info.action_values[action_index] += propagated_value
                propagated_value += n.parent.info.reward
            else:
                raise ValueError()

            if n.parent.parent is None:
                break
            n = n.parent

    node = tree.root_node
    setup_node(node)

    while True:

        # pick next action
        mcts_info = node.info
        weighted_action_vals = np.matmul(mcts_info.action_values, weights)
        visits = mcts_info.visits + (mcts_info.visits == 0)  # last term prevents ZeroDivisionError

        if node.is_home:
            a_index = np.argmax((weighted_action_vals + exploration_coeff * mcts_info.probabilities) / visits)
        else:
            a_index = np.argmin((weighted_action_vals - exploration_coeff * mcts_info.probabilities) / visits)

        mcts_info.visits[a_index] += 1

        # expand action
        action: botbowl.Action = mcts_info.actions[a_index]
        if action not in node.explored_actions:
            tree.expand_action_node(node, action)

        # select next node
        children: List[ActionNode] = list(node.get_children_from_action(action))
        prob = [child.get_accum_prob(end_node=node) for child in children]
        node = np.random.choice(children, 1, p=prob)[0]

        setup_node(node)
        tree.set_game_to_node(node)
        if not continue_expension(node):
            back_propagate(node)
            break
