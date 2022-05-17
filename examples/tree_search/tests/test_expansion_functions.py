import examples.tree_search as ts
import pytest
from botbowl import Square, Action, ActionType, Skill, BBDieResult
from pytest import approx
from tests.util import get_custom_game_turn, only_fixed_rolls


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
    parent_action_node: ts.ActionNode = ts.ActionNode(game, parent=None)
    next_node: ts.Node = ts.expand_action(game, action, parent_action_node)

    assert next_node.parent is parent_action_node

    if len(outcome_probs) == 1:
        assert type(next_node) is ts.ActionNode
    else:
        next_node: ts.ChanceNode
        assert type(next_node) is ts.ChanceNode
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
    parent_action_node: ts.ActionNode = ts.ActionNode(game, parent=None)
    next_node: ts.Node = ts.expand_action(game, action, parent_action_node)

    next_node: ts.ChanceNode
    assert next_node.parent is parent_action_node
    assert type(next_node) is ts.ChanceNode
    assert sum(next_node.child_probability) == 1.0
    assert all(y == approx(x, abs=1e-12) for x, y in zip(sorted(next_node.child_probability), sorted(outcome_probs)))


def test_bounce():
    game, (attacker, defender) = get_custom_game_turn(player_positions=[(5, 5)],
                                                      opp_player_positions=[(5, 6)],
                                                      ball_position=(5, 6),
                                                      forward_model_enabled=True,
                                                      pathfinding_enabled=True)

    with only_fixed_rolls(game, block_dice=[BBDieResult.DEFENDER_DOWN]):
        game.step(Action(ActionType.START_BLOCK, position=attacker.position))
        game.step(Action(ActionType.BLOCK, position=defender.position))
        game.step(Action(ActionType.SELECT_DEFENDER_DOWN))
        game.step(Action(ActionType.PUSH, position=Square(5, 7)))

    action = Action(ActionType.FOLLOW_UP, position=attacker.position)

    tree = ts.SearchTree(game)
    n = tree.expand_action_node(tree.root_node, action)


def test_expand_block():
    game, (attacker, _, defender) = get_custom_game_turn(player_positions=[(5, 5), (7, 7)],
                                                         opp_player_positions=[(6, 6)],
                                                         forward_model_enabled=True)
    defender.extra_skills.append(Skill.DODGE)
    tree = ts.SearchTree(game)

    next_node, = tree.expand_action_node(tree.root_node, Action(ActionType.START_BLOCK, player=attacker))

    assert len(tree.all_action_nodes) == 2
    next_node, *_ = tree.expand_action_node(next_node, Action(ActionType.BLOCK, position=defender.position))

    assert len(tree.all_action_nodes) == 6
    next_node, = tree.expand_action_node(next_node, Action(ActionType.SELECT_DEFENDER_DOWN))

    assert len(tree.all_action_nodes) == 7
    next_node, = tree.expand_action_node(next_node, Action(ActionType.PUSH, position=Square(7, 6)))

    assert len(tree.all_action_nodes) == 8
    next_node, *_ = tree.expand_action_node(next_node, Action(ActionType.FOLLOW_UP, position=Square(6, 6)))

    assert len(tree.all_action_nodes) == 11


@pytest.mark.parametrize("pitch_size", [1, 3, 5, 7, 11])
def test_expand_throw_in(pitch_size):
    game, (attacker, defender) = get_custom_game_turn(player_positions=[(3, 2)],
                                                      opp_player_positions=[(3, 1)],
                                                      ball_position=(3, 1),
                                                      size=pitch_size,
                                                      forward_model_enabled=True,
                                                      pathfinding_enabled=True)

    with only_fixed_rolls(game, block_dice=[BBDieResult.DEFENDER_DOWN]):
        game.step(Action(ActionType.START_BLOCK, position=attacker.position))
        game.step(Action(ActionType.BLOCK, position=defender.position))
        game.step(Action(ActionType.SELECT_DEFENDER_DOWN))

    action = Action(ActionType.PUSH, position=Square(3, 0))

    tree = ts.SearchTree(game)
    tree.expand_action_node(tree.root_node, action)
    assert len(tree.all_action_nodes) == 2


def test_expand_handoff():
    for i in range(10):
        game, (scorer, carrier, opp1, opp2) = get_custom_game_turn(player_positions=[(2, 2), (7, 7)],
                                                                   opp_player_positions=[(7, 8), (7, 6)],
                                                                   ball_position=(7, 7),
                                                                   pathfinding_enabled=True,
                                                                   rerolls=3)

        game.step(Action(ActionType.START_HANDOFF, position=carrier.position))
        tree = ts.SearchTree(game)
        tree.expand_action_node(tree.root_node, Action(ActionType.HANDOFF, position=scorer.position))
        reroll_node = None
        for key, node in tree.all_action_nodes.data.items():
            if key.find("Reroll (Dodge)") != -1:
                reroll_node = node[0]
                break
        tree.expand_action_node(reroll_node, Action(ActionType.USE_REROLL))
