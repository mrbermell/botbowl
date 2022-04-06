from functools import partial
from operator import itemgetter
from typing import Union, Callable, List, Iterable

import botbowl
import examples.tree_search.searchers.search_util as search_util
import examples.tree_search as ts
import numpy as np
from examples.tree_search.SearchTree import SearchTree, ActionNode, ChanceNode, Node, get_action_node_children
from pytest import approx


def generic_tree_search_rollout(tree: SearchTree,
                                policy: search_util.Policy,
                                weights: search_util.HeuristicVector,
                                expand_chance_node: Callable[[ts.ChanceNode], Iterable[ts.ActionNode]],
                                sample_action: Callable[[ts.ActionNode, ts.HeuristicVector], botbowl.Action],
                                cc_cond: search_util.ContinueCondition = None,
                                back_propagate_with_probability: bool = False,
                                ) -> None:
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
        if type(new_node.info) is not ts.MCTS_Info:
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

            new_node.info = ts.MCTS_Info(probabilities=probabilities / probabilities.mean(),
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
            if back_propagate_with_probability and isinstance(n.parent, ChanceNode):
                propagated_value *= n.parent.get_child_prob(n)
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

    setup_node(tree.root_node)
    node_queue: List[ts.ActionNode] = [tree.root_node]

    while len(node_queue) > 0:
        node = node_queue.pop()

        # pick next action
        action = sample_action(node, weights)

        # expand action
        if action not in node.explored_actions:
            tree.expand_action_node(node, action)

        # handle child nodes
        direct_child = node.children[node.explored_actions.index(action)]

        if type(direct_child) is ts.ActionNode:
            children = (direct_child,)
        else:
            children = expand_chance_node(direct_child)

        for child_node in children:
            setup_node(child_node)
            tree.set_game_to_node(child_node)
            if continue_expension(child_node):
                node_queue.append(child_node)
            else:
                back_propagate(child_node)


def determinstic_chance_node_expansion(node: ts.ChanceNode) -> Iterable[ts.ActionNode]:
    return get_action_node_children(node)


def uct_action_sample(node: ts.ActionNode, weights: ts.HeuristicVector) -> botbowl.Action:
    mcts_info = node.info
    weighted_action_vals = np.matmul(mcts_info.action_values, weights)
    visits = mcts_info.visits + (mcts_info.visits == 0)  # last term prevents ZeroDivisionError

    if node.is_home:
        a_index = np.argmax((weighted_action_vals + mcts_info.probabilities) / visits)
    else:
        a_index = np.argmin((weighted_action_vals - mcts_info.probabilities) / visits)

    mcts_info.visits[a_index] += 1
    return mcts_info.actions[a_index]


def single_stocastic_chance_node_exp(node: ts.ChanceNode) -> Iterable[ts.ActionNode]:
    children: List[ActionNode] = list(get_action_node_children(node) )
    prob = [child.get_accum_prob(end_node=node) for child in children]
    return np.random.choice(children, 1, p=prob)[0],


def get_node_value(node: Union[Node, ActionNode, ChanceNode], weights: search_util.HeuristicVector) -> float:
    recursive_self = partial(get_node_value, weights=weights)

    if isinstance(node, ActionNode):
        reward = np.dot(node.info.reward, weights)
        if len(node.children) == 0:
            return node.info.state_value + reward
        elif node.is_home:
            return max(map(recursive_self, node.children))
        else:  # not node.is_home:
            return min(map(recursive_self, node.children))
    elif isinstance(node, ChanceNode):
        assert sum(node.child_probability) == approx(1.0, abs=1e-9)
        return sum(prob * recursive_self(child) for prob, child in zip(node.child_probability, node.children))
    else:
        raise ValueError()


def get_best_action(root_node: ActionNode, weights: search_util.HeuristicVector) -> botbowl.Action:
    assert len(root_node.children) == len(root_node.explored_actions)

    child_node_values = (get_node_value(node, weights) for node in root_node.children)

    if root_node.is_home:
        action_index = np.argmax(child_node_values)
    else:
        action_index = np.argmin(child_node_values)

    return root_node.explored_actions[action_index]


def show_best_path(tree: SearchTree, weights: search_util.HeuristicVector):
    node = tree.root_node
    tree.set_game_to_node(node)
    report_index = len(tree.game.state.reports)

    while len(node.children) > 0:
        if isinstance(node, ActionNode):
            assert len(node.children) == len(node.explored_actions)
            child_node_values = [get_node_value(node, weights) for node in node.children]

            if node.is_home:
                action_index = np.argmax(child_node_values)
            else:
                action_index = np.argmin(child_node_values)

            best_action = node.explored_actions[action_index]
            child = node.children[action_index]

            tree.set_game_to_node(node)
            for r in tree.game.state.reports[report_index:]:
                print(f"    {r}")
            report_index = len(tree.game.state.reports)

            action_type = str(best_action.action_type).split('.')[-1]
            pos = best_action.position
            expected_value = child_node_values[action_index]
            print(f"{action_type}, {pos}, value={expected_value:.3f}")

        elif isinstance(node, ChanceNode):
            child, prob = max(zip(node.children, node.child_probability), key=itemgetter(1))

        else:
            raise ValueError()

        node = child

    assert len(node.children) == 0
