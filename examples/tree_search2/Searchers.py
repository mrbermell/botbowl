from functools import partial
from operator import attrgetter, itemgetter
from typing import Tuple, Union, Optional, Callable, List
import queue

import numpy as np
from pytest import approx

import botbowl
from examples.tree_search2.SearchTree import SearchTree, ActionNode, ChanceNode


def get_heuristic(game: botbowl.Game,
                  coeff_score: float = 0,
                  coeff_tv_on_pitch: float = 0,
                  coeff_ball_x: float = 0,
                  coeff_ball_marked: float = 0,
                  coeff_ball_carried: float = 0) -> float:
    """ heuristic based on game state, zero sum, calculated from home teams perspective """
    result: float = 0.0
    home = game.state.home_team
    away = game.state.away_team
    ball = game.get_ball()

    result += coeff_score * (home.state.score - away.state.score)

    result += coeff_tv_on_pitch * sum(map(attrgetter('role.cost'), game.get_players_on_pitch(team=home)))
    result -= coeff_tv_on_pitch * sum(map(attrgetter('role.cost'), game.get_players_on_pitch(team=away)))

    if ball is not None and ball.position is not None:
        result -= coeff_ball_x * ball.position.x  # negative because home team scores at x = 0

        home_marking_ball = len(game.get_adjacent_players(ball.position, team=home, standing=True)) > 0
        away_marking_ball = len(game.get_adjacent_players(ball.position, team=away, standing=True)) > 0

        if ball.is_carried:
            ball_carrier = game.get_player_at(ball.position)
            if ball_carrier.team == home:
                result += coeff_ball_carried
                result -= coeff_ball_marked * away_marking_ball
            elif ball_carrier.team == away:
                result -= coeff_ball_carried
                result += coeff_ball_marked * home_marking_ball
        else:
            result += coeff_ball_marked * (home_marking_ball - away_marking_ball)

    return result


gotebot_heuristic = partial(get_heuristic, coeff_score=1)#,
                            #coeff_ball_x=0.05,
                            #coeff_ball_marked=0.1,
                            #coeff_ball_carried=0.2)


def get_best_action(node: Union[ChanceNode, ActionNode]) -> Tuple[Optional[botbowl.Action], float]:
    if type(node) is ChanceNode:
        assert sum(node.child_probability) == approx(1.0, abs=1e-9), f"{node} {sum(node.child_probability)=} should be 1.0"
        return None, sum( prob * get_best_action(child)[1] for prob, child in zip(node.child_probability, node.children))

    elif type(node) is ActionNode:
        if len(node.children) == 0:
            return None, node.info['value']
        else:
            assert len(node.children) == len(node.explored_actions)
            action, value = max(((action, get_best_action(child)[1])
                                for action, child in zip(node.explored_actions, node.children)), key=itemgetter(1))
            return action, value


Policy = Callable[[botbowl.Game], Tuple[float, np.ndarray, List[botbowl.Action]]]
def do_mcts_branch(tree: SearchTree, policy: Policy) -> None:

    def continue_expansion(a_node: ActionNode) -> bool:
        raise NotImplementedError() # todo

    def setup_node(new_node: ActionNode):
        if 'probabilities' not in new_node.info:
            tree.set_game_to_node(new_node)
            state_value, probabilities, actions = policy(tree.game)
            new_node.info['actions'] = actions
            new_node.info['probabilities'] = probabilities
            new_node.info['state_value'] = state_value
            new_node.info['action_values'] = np.zeros(len(probabilities))
            new_node.info['visit_count'] = np.zeros(len(probabilities), dtype=np.int)

    def back_propagate(final_node: ActionNode):
        propagated_value = final_node.info['state_value'] + final_node.reward

        n = final_node
        while True:
            if isinstance(n.parent, ChanceNode):
                propagated_value *= n.parent.get_child_prob(n)
            elif isinstance(n.parent, ActionNode):
                action_object = n.parent.get_child_action(n)
                action_index = n.parent.info['actions'].index(action_object)
                n.parent.info['action_values'][action_index] += propagated_value
                propagated_value += n.parent.reward
            else:
                raise ValueError()

            if n.parent is tree.root_node:
                break
            n = n.parent

    node_queue = queue.Queue()
    node_queue.put(tree.root_node)

    while node_queue.qsize() > 0:
        node: ActionNode = node_queue.get()
        setup_node(node)

        # pick next action
        probs = node.info['probabilities']
        action_values = node.info['action_values']
        visits = node.info['visit_count']
        a_index = np.argmax(action_values + 0.25*probs/(1+visits))
        visits[a_index] += 1  # increment visit count

        # expand action and handle new nodes
        action: botbowl.Action = node.info['actions'][a_index]
        if action not in node.explored_actions:
            tree.expand_action_node(node, action)

        for child_node in node.get_children_from_action(action):
            setup_node(child_node)
            if continue_expansion(child_node):
                node_queue.put(child_node)
            else:
                # backpropagation!
                back_propagate(child_node)

