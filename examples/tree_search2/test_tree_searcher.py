from copy import deepcopy
from random import randint

import pytest
from pytest import approx

from botbowl import Square, Action, ActionType, Skill, BBDieResult
from examples.tree_search2.Samplers import ActionSampler, MockPolicy
from examples.tree_search2.SearchTree import ActionNode, expand_action, ChanceNode, Node, SearchTree, \
    get_action_node_children
from examples.tree_search2.Searchers import get_best_action, get_heuristic, do_mcts_branch
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


def test_dodge_pickup_score():
    game, _ = get_custom_game_turn(player_positions=[(5, 5)],
                                   opp_player_positions=[(6, 6)],
                                   ball_position=(3, 3),
                                   forward_model_enabled=True,
                                   pathfinding_enabled=True)

    def on_every_action_node(search_tree: SearchTree, node):
        search_tree.set_game_to_node(node)
        node.info['action_sampler'] = ActionSampler(search_tree.game)
        node.info['value'] = get_heuristic(search_tree.game, coeff_score=1.0)

    tree = SearchTree(game, on_every_action_node)

    def filter_func(node):
        return node.depth < 3 and len(node.info['action_sampler']) > 0

    for _ in range(100):
        possible_nodes = list(filter(filter_func, tree.all_action_nodes))
        node_to_explore = possible_nodes[randint(0, len(possible_nodes)-1)]

        action = node_to_explore.info['action_sampler'].get_action()
        assert action is not None
        tree.expand_action_node(node_to_explore, action)

    tree.set_game_to_node(tree.root_node)
    best_action, value = get_best_action(tree.root_node)
    assert value == approx((4/6)**2)  # corresponding to 3+, 3+ w/o rerolls


def test_expand_block():
    game, (attacker, _, defender) = get_custom_game_turn(player_positions=[(5, 5), (7, 7)],
                                                         opp_player_positions=[(6, 6)],
                                                         forward_model_enabled=True)
    defender.extra_skills.append(Skill.DODGE)
    tree = SearchTree(game)

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

def test_expand_throw_in():
    game, (attacker, defender) = get_custom_game_turn(player_positions=[(5, 2)],
                                                      opp_player_positions=[(5, 1)],
                                                      ball_position=(5, 1),
                                                      forward_model_enabled=True,
                                                      pathfinding_enabled=True)

    with only_fixed_rolls(game, block_dice=[BBDieResult.DEFENDER_DOWN]):
        game.step(Action(ActionType.START_BLOCK, position=attacker.position))
        game.step(Action(ActionType.BLOCK, position=defender.position))
        game.step(Action(ActionType.SELECT_DEFENDER_DOWN))

    action = Action(ActionType.PUSH, position=Square(5, 0))

    tree = SearchTree(game)
    tree.expand_action_node(tree.root_node, action)
    assert len(tree.all_action_nodes) == 2


def test_set_new_root():
    game, (player1, player2, opp_player) = get_custom_game_turn(player_positions=[(5, 5), (6, 6)],
                                                                opp_player_positions=[(4, 4)],
                                                                ball_position=(5, 5),
                                                                pathfinding_enabled=True,
                                                                forward_model_enabled=True)

    action_p2_1 = Action(ActionType.START_BLITZ, position=player2.position)
    action_p2_2 = Action(ActionType.BLOCK, position=Square(4, 4))

    action_p1_1 = Action(ActionType.START_MOVE, position=player1.position)
    action_p1_2 = Action(ActionType.MOVE, position=Square(1, 5))

    tree = SearchTree(deepcopy(game))
    assert tree.root_node.depth == 0


    # Move player 2
    new_node, = tree.expand_action_node(tree.root_node, action_p2_1)
    new_nodes = tree.expand_action_node(new_node, action_p2_2)

    assert new_node.depth == 1
    assert new_nodes[0].depth == 2
    assert len(tree.all_action_nodes) == 2 + 4

    # Move player 1
    new_node, = tree.expand_action_node(tree.root_node, action_p1_1)
    assert len(tree.all_action_nodes) == 2 + 4 + 1
    new_nodes = tree.expand_action_node(new_node, action_p1_2)

    assert new_node.depth == 1
    assert new_nodes[0].depth == 2
    assert len(tree.all_action_nodes) == 2 + 4 + 1 + 7

    game.step(action_p1_1)
    tree.set_new_root(game)

    assert len(tree.all_action_nodes) == 8
    assert new_node is tree.root_node
    assert new_node.depth == 0
    assert new_nodes[0].depth == 1

    with only_fixed_rolls(game, d6=[6]):
        game.step(action_p1_2)

    tree.set_new_root(game)
    tree.expand_action_node(tree.root_node, Action(ActionType.SETUP_FORMATION_SPREAD))
    assert new_nodes[0] is tree.root_node
    assert len(tree.all_action_nodes) == 2

    game.step(Action(ActionType.SETUP_FORMATION_SPREAD))
    game.step(Action(ActionType.END_SETUP))
    tree.set_new_root(game)
    assert len(tree.all_action_nodes) == 1
    assert len(tree.root_node.children) == 0


def test_mcts():
    game, _ = get_custom_game_turn(player_positions=[(6, 6), (7, 7)],
                                opp_player_positions=[(5, 6)],
                                ball_position=(6, 6),
                                pathfinding_enabled=True)

    tree = SearchTree(game)
    policy = MockPolicy()
    for i in range(100):
        do_mcts_branch(tree, policy)

    from more_itertools import first
    setup_node_name = first(filter(lambda s: s.find('Setup')>0,  tree.all_action_nodes.data))
    setup_node: ActionNode = tree.all_action_nodes.data[setup_node_name][0]
    chance_node = setup_node.parent
    assert isinstance(chance_node, ChanceNode)

    action_children = list(get_action_node_children(chance_node))
    assert len(action_children) > 0
    assert len(action_children) > 1


    print("")
    mcts_info = tree.root_node.info['mcts']
    for action, visits, action_val in zip(mcts_info.actions, mcts_info.visits, mcts_info.action_values):
        action.player = None
        print(f"{action}, {visits=}, {action_val=}")

    print("")
    print(f"{len(tree.all_action_nodes)=}")