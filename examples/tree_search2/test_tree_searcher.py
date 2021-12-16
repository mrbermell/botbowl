from random import randint

import pytest
from pytest import approx

from examples.tree_search2.Searchers import gotebot_heuristic, get_best_action
from tests.util import get_custom_game_turn, only_fixed_rolls
from examples.tree_search2.SearchTree import ActionNode, expand_action, ChanceNode, Node, SearchTree
from examples.tree_search2.Samplers import ActionSampler
# import botbowl
from botbowl import Square, Action, ActionType
# import numpy as np


@pytest.mark.parametrize("data", [(Square(2, 2), [1.0]),
                                  (Square(4, 4), [4 / 6, 2 / 6]),
                                  (Square(3, 3), [4 / 6, 2 / 6]),
                                  (Square(9, 9), [(4 * 5 * 5) / (6 ** 3), 1 - (4 * 5 * 5) / (6 ** 3)])
                                  ])
def test_expand_move(data):
    move_target, outcome_probs = data
    assert sum(outcome_probs) == 1.0
    game, (player, _, _) = get_custom_game_turn(player_positions=[(1, 1)],
                                                opp_player_positions=[(1, 3), (3, 1)],
                                                forward_model_enabled=True,
                                                pathfinding_enabled=True)

    game.step(Action(ActionType.START_MOVE, position=player.position))

    action = Action(ActionType.MOVE, position=move_target)
    parent_action_node: ActionNode = ActionNode(game, parent=None)
    next_node: Node = expand_action(game, action, parent_action_node)

    assert next_node.parent is parent_action_node

    if len(outcome_probs) == 1:
        assert type(next_node) is ActionNode
    else:
        next_node: ChanceNode
        assert type(next_node) is ChanceNode
        assert sum(next_node.child_probability) == 1.0
        assert all(
            y == approx(x, abs=1e-12) for x, y in zip(sorted(next_node.child_probability), sorted(outcome_probs)))


@pytest.mark.parametrize("data", [(Square(3, 3), [4 / 6, 2 / 6]),  # Not marked ball
                                  (Square(3, 2), [3 / 6, 3 / 6]),  # Marked one tz
                                  (Square(4, 2), [4 / 6, 2 / 6])  # Marked two tz
                                  ])
def test_expand_pickup(data):
    ball_square, outcome_probs = data
    assert sum(outcome_probs) == 1.0
    game, (player, _, _) = get_custom_game_turn(player_positions=[(1, 1)],
                                                opp_player_positions=[(3, 1), (5, 1)],
                                                ball_position=ball_square,
                                                forward_model_enabled=True,
                                                pathfinding_enabled=True)

    game.step(Action(ActionType.START_MOVE, position=player.position))
    action = Action(ActionType.MOVE, position=ball_square)
    parent_action_node: ActionNode = ActionNode(game, parent=None)
    next_node: Node = expand_action(game, action, parent_action_node)

    next_node: ChanceNode
    assert next_node.parent is parent_action_node
    assert type(next_node) is ChanceNode
    assert sum(next_node.child_probability) == 1.0
    assert all(y == approx(x, abs=1e-12) for x, y in zip(sorted(next_node.child_probability), sorted(outcome_probs)))


def test_tree_searcher():
    game, _ = get_custom_game_turn(player_positions=[(5, 5)],
                                   ball_position=(3, 3),
                                   forward_model_enabled=True,
                                   pathfinding_enabled=True)

    with only_fixed_rolls(game, d6=[6]):
        action = ActionSampler(game).get_action()
        game.step(action)
        action = ActionSampler(game).get_action()
        game.step(action)
        action = ActionSampler(game).get_action()
        game.step(action)

    assert game.state.home_team.state.score + game.state.away_team.state.score == 1


def test_tree_searcher2():
    game, _ = get_custom_game_turn(player_positions=[(5, 5)],
                                   opp_player_positions=[(6, 6)],
                                   ball_position=(3, 3),
                                   forward_model_enabled=True,
                                   pathfinding_enabled=True)
    tree = SearchTree(game)

    for _ in range(100):
        possible_nodes = []
        for node in tree.all_action_nodes:
            if node.depth < 3:
                if 'action_sampler' not in node.info:
                    possible_nodes.append(node)
                elif len(node.info['action_sampler']) > 0:
                    possible_nodes.append(node)

        node_to_explore = possible_nodes[randint(0, len(possible_nodes)-1)]
        if 'action_sampler' not in node_to_explore.info:
            tree.set_game_to_node(node_to_explore)
            node_to_explore.info['action_sampler'] = ActionSampler(tree.game)

        action = node_to_explore.info['action_sampler'].get_action()
        assert action is not None
        tree.expand_action_node(node_to_explore, action)

    for node in tree.all_action_nodes:
        if 'value' not in node.info:
            tree.set_game_to_node(node)
            node.info['value'] = gotebot_heuristic(tree.game)

    tree.set_game_to_node(tree.root_node)
    best_action, value = get_best_action(tree.root_node)
    print(f"{best_action} with {value=}")
