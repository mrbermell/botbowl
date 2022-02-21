import queue
from functools import partial
from operator import attrgetter, itemgetter
from typing import Tuple, Union, Optional, Callable, List

import numpy as np
from pytest import approx

import botbowl
from examples.tree_search2.SearchTree import SearchTree, ActionNode, ChanceNode, Node
from collections import namedtuple

HeuristicVector = namedtuple('HeuristicVector', ['score',
                                                 'tv_on_pitch',
                                                 'ball_position',
                                                 'ball_carried',
                                                 'ball_marked'])


def get_heuristic(game: botbowl.Game) -> HeuristicVector:
    """
    Heuristic based on game state, calculated from home teams perspective
    zero sum, meaning away team's heuristic is negative of home team's heuristic
    :returns: array with different heuristics, multiply it with
    """
    score, tv_on_pitch, ball_position, ball_carried, ball_marked = 0.0, 0.0, 0.0, 0.0, 0.0
    home = game.state.home_team
    away = game.state.away_team

    score += home.state.score - away.state.score

    tv_on_pitch += sum(p.role.cost for p in game.get_players_on_pitch(team=home))
    tv_on_pitch -= sum(p.role.cost for p in game.get_players_on_pitch(team=away))
    tv_on_pitch /= 50000.0  # normalized to cost of lineman

    ball = game.get_ball()
    if ball is not None and ball.position is not None:
        ball_position -= ball.position.x  # negative because home team scores at x = 0

        home_marking_ball = len(game.get_adjacent_players(ball.position, team=home, standing=True)) > 0
        away_marking_ball = len(game.get_adjacent_players(ball.position, team=away, standing=True)) > 0

        if ball.is_carried:
            ball_carrier = game.get_player_at(ball.position)
            if ball_carrier.team == home:
                ball_carried += 1
                ball_marked -= away_marking_ball
            elif ball_carrier.team == away:
                ball_carried -= 1
                ball_marked += home_marking_ball
        else:
            ball_marked += (home_marking_ball - away_marking_ball)

    return HeuristicVector(score=score,
                           tv_on_pitch=tv_on_pitch,
                           ball_position=ball_position,
                           ball_carried=ball_carried,
                           ball_marked=ball_marked)


MCTS_Info = namedtuple('MCTS_Info', 'probabilities actions action_values visits heuristic reward state_value')
Policy = Callable[[botbowl.Game], Tuple[float, np.ndarray, List[botbowl.Action]]]


def do_mcts_branch(tree: SearchTree, policy: Policy, weights: HeuristicVector, exploration_coeff=1) -> None:
    weights = np.array(weights)
    tree.set_game_to_node(tree.root_node)
    game = tree.game

    scores = game.state.home_team.state.score + game.state.away_team.state.score
    my_team = game.actor
    my_turn_num = tree.root_node.turn

    def continue_expansion(a_node: ActionNode) -> bool:
        tree.set_game_to_node(a_node)
        if game.actor is my_team and my_turn_num != a_node.turn:
            return False
        if game.state.home_team.state.score + game.state.away_team.state.score != scores:
            return False
        return True

    def setup_node(new_node: ActionNode):
        if type(new_node.info) is not MCTS_Info:
            tree.set_game_to_node(new_node)
            _, probabilities, actions_ = policy(tree.game)
            num_actions = len(actions_)
            heuristic = np.array(get_heuristic(tree.game))
            reward = np.zeros(shape=heuristic.shape)

            if new_node.parent is not None:
                for parent in new_node.get_all_parents(include_self=False):
                    if isinstance(parent, ActionNode):
                        reward = heuristic - parent.info.heuristic
                        break

            new_node.info = MCTS_Info(probabilities=probabilities,
                                      actions=actions_,
                                      action_values=np.zeros((num_actions, len(reward))),
                                      visits=np.zeros(num_actions, dtype=np.int),
                                      heuristic=heuristic,
                                      reward=reward,
                                      state_value=0)

    def back_propagate(final_node: ActionNode):
        propagated_value = final_node.info.reward  # todo: add final_node.info.state_value too

        n = final_node
        while True:
            if isinstance(n.parent, ChanceNode):
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

    node_queue = queue.Queue()
    node_queue.put(tree.root_node)

    while node_queue.qsize() > 0:
        node: ActionNode = node_queue.get()
        setup_node(node)

        # pick next action
        mcts_info = node.info
        weighted_action_vals = np.matmul(mcts_info.action_values, weights)
        visits = mcts_info.visits + (mcts_info.visits == 0)  # last term prevents ZeroDivisionError

        if node.is_home:
            a_index = np.argmax((weighted_action_vals + exploration_coeff * mcts_info.probabilities) / visits)
        else:
            a_index = np.argmin((weighted_action_vals - exploration_coeff * mcts_info.probabilities) / visits)

        mcts_info.visits[a_index] += 1

        # expand action and handle new nodes
        action: botbowl.Action = mcts_info.actions[a_index]
        if action not in node.explored_actions:
            tree.expand_action_node(node, action)

        for child_node in node.get_children_from_action(action):
            setup_node(child_node)
            if continue_expansion(child_node):
                node_queue.put(child_node)
            else:
                back_propagate(child_node)


def get_node_value(node: Union[Node, ActionNode, ChanceNode], weights: HeuristicVector) -> float:
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


def get_best_action(root_node: ActionNode, weights: HeuristicVector) -> botbowl.Action:
    assert len(root_node.children) == len(root_node.explored_actions)

    child_node_values = (get_node_value(node, weights) for node in root_node.children)

    if root_node.is_home:
        action_index = np.argmax(child_node_values)
    else:
        action_index = np.argmin(child_node_values)

    return root_node.explored_actions[action_index]


def show_best_path(tree: SearchTree, weights: HeuristicVector):
    node = tree.root_node
    tree.set_game_to_node(node)
    report_index = len(tree.game.state.reports)

    while len(node.children) > 0:
        if isinstance(node, ActionNode):
            assert len(node.children) == len(node.explored_actions)
            child_node_values = (get_node_value(node, weights) for node in node.children)

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
            print(best_action)

        elif isinstance(node, ChanceNode):
            child, prob = max(zip(node.children, node.child_probability), key=itemgetter(1))

        else:
            raise ValueError()

        node = child

    assert len(node.children) == 0
