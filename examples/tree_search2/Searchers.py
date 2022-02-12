import queue
from collections import namedtuple
from operator import attrgetter, itemgetter
from operator import attrgetter, itemgetter
from typing import Tuple, Union, Optional, Callable, List

import numpy as np
from pytest import approx

import botbowl
from examples.tree_search2.SearchTree import SearchTree, ActionNode, ChanceNode

NUM_HEURISTICS = 5
gbg_heuristic_weights = np.array([1, # score
                                  0, # tv on pitch
                                  0.05, # ball position
                                  0.2, # ball carried
                                  0.1]) # ball marked

def get_heuristic(game: botbowl.Game) -> np.ndarray:
    """
    Heuristic based on game state, calculated from home teams perspective
    zero sum, meaning away team's heuristic is negative of home team's heuristic
    :returns: array with different heuristics, multiply it with
    """
    result = np.zeros(NUM_HEURISTICS)
    home = game.state.home_team
    away = game.state.away_team
    ball = game.get_ball()

    result[0] = home.state.score - away.state.score

    result[1] += sum(map(attrgetter('role.cost'), game.get_players_on_pitch(team=home)))
    result[1] -= sum(map(attrgetter('role.cost'), game.get_players_on_pitch(team=away)))

    if ball is not None and ball.position is not None:
        result[2] -= ball.position.x  # negative because home team scores at x = 0

        home_marking_ball = len(game.get_adjacent_players(ball.position, team=home, standing=True)) > 0
        away_marking_ball = len(game.get_adjacent_players(ball.position, team=away, standing=True)) > 0

        if ball.is_carried:
            ball_carrier = game.get_player_at(ball.position)
            if ball_carrier.team == home:
                result[3] += 1
                result[4] -= away_marking_ball
            elif ball_carrier.team == away:
                result[3] -= 1
                result[4] += home_marking_ball
        else:
            result[4] += (home_marking_ball - away_marking_ball)

    return result


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


MCTS_Info = namedtuple('MCTS_Info', 'probabilities actions action_values visits heuristic reward state_value')


Policy = Callable[[botbowl.Game], Tuple[float, np.ndarray, List[botbowl.Action]]]
def do_mcts_branch(tree: SearchTree, policy: Policy) -> None:
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

            heuristic = get_heuristic(tree.game)

            reward = np.zeros(NUM_HEURISTICS)
            if new_node.parent is not None:
                for parent in new_node.get_all_parents(include_self=False):
                    if isinstance(parent, ActionNode):
                        reward = heuristic - parent.info.reward
                        break

            assert len(reward) == NUM_HEURISTICS
            assert len(heuristic) == NUM_HEURISTICS

            new_node.info= MCTS_Info(probabilities=probabilities,
                                     actions=actions_,
                                     action_values=np.zeros((num_actions, NUM_HEURISTICS)),
                                     visits=np.zeros(len(probabilities), dtype=np.int),
                                     heuristic=heuristic,
                                     reward=reward,
                                     state_value=None)


    def back_propagate(final_node: ActionNode):
        #propagated_value = final_node.info.state_value + final_node.info.reward
        propagated_value = final_node.info.reward

        n = final_node
        while True:
            if isinstance(n.parent, ChanceNode):
                propagated_value *= n.parent.get_child_prob(n)
            elif isinstance(n.parent, ActionNode):
                action_object = n.parent.get_child_action(n)
                action_index = n.parent.info.actions.index(action_object)
                n.parent.info.action_values[action_index] += propagated_value
                propagated_value += n.parent.reward
            else:
                raise ValueError()

            assert len(propagated_value) == NUM_HEURISTICS

            if n.parent is tree.root_node:
                break
            n = n.parent

    node_queue = queue.Queue()
    node_queue.put(tree.root_node)

    while node_queue.qsize() > 0:
        node: ActionNode = node_queue.get()
        setup_node(node)

        # pick next action
        mcts_info = node.info
        weighted_action_vals = np.matmul(mcts_info.action_values, gbg_heuristic_weights)
        visits = mcts_info.visits + (mcts_info.visits==0)
        a_index = np.argmax(weighted_action_vals/visits + 10*mcts_info.probabilities/visits)
        mcts_info.visits[a_index] += 1  # increment visit count

        # expand action and handle new nodes
        action: botbowl.Action = mcts_info.actions[a_index]
        if action not in node.explored_actions:
            if action.action_type is botbowl.ActionType.BLOCK and action.position.x == 1 and action.position.y == 1:
                print("")
            tree.expand_action_node(node, action)

        for child_node in node.get_children_from_action(action):
            setup_node(child_node)
            if continue_expansion(child_node):
                node_queue.put(child_node)
            else:
                # backpropagation!
                back_propagate(child_node)

