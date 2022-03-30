import gym
import numpy as np
import pytest
from more_itertools import first
from pytest import approx

import examples.tree_search as ts
import examples.tree_search.evaluation_scenarios as scenarios
from botbowl import Square, Action, ActionType, Skill, BBDieResult, botbowl
from botbowl.core import procedure
from examples.tree_search import hashmap, get_node_value
from tests.util import get_custom_game_turn, only_fixed_rolls


default_weights = ts.HeuristicVector(score=1, ball_marked=0.1, ball_carried=0.2, ball_position=0.01, tv_on_pitch=1)


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


def test_dodge_pickup_score():
    game, players = get_custom_game_turn(player_positions=[(5, 5), (6, 6)],
                                         opp_player_positions=[(4, 4), (4, 2)],
                                         ball_position=(3, 3),
                                         forward_model_enabled=True,
                                         pathfinding_enabled=True)

    weights = ts.HeuristicVector(score=1, ball_marked=0, ball_carried=0, ball_position=0.001, tv_on_pitch=0)

    tree = ts.SearchTree(game)
    policy = ts.MockPolicy()

    def search_select_step():
        for i in range(10):
            ts.deterministic_tree_search_rollout(tree, policy, weights, exploration_coeff=0.5)
        info: ts.MCTS_Info = tree.root_node.info
        print("   ")
        print_node(tree.root_node, weights)
        return info.actions[np.argmax(info.visits)]

    a = search_select_step()
    game.step(a)
    tree.set_new_root(game)

    a = search_select_step()
    game.step(a)
    tree.set_new_root(game)

    a = search_select_step()
    game.step(a)


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


def test_pickup_score():
    game, (player,) = get_custom_game_turn(player_positions=[(5, 5)],
                                           ball_position=(3, 3),
                                           forward_model_enabled=True,
                                           pathfinding_enabled=True)

    player.role.ma = 4
    game.set_available_actions()

    weights = ts.HeuristicVector(score=1, ball_marked=0, ball_carried=0, ball_position=0, tv_on_pitch=0)

    tree = ts.SearchTree(game)
    policy = ts.MockPolicy()

    def search_select_step():
        for i in range(40):
            ts.deterministic_tree_search_rollout(tree, policy, weights, exploration_coeff=0.5)
        info: ts.MCTS_Info = tree.root_node.info
        print("   ")
        print_node(tree.root_node, weights)
        return info.actions[np.argmax(info.visits)]

    a = search_select_step()
    assert a.action_type == ActionType.START_MOVE and a.position == player.position
    with only_fixed_rolls(game):
        game.step(a)

    tree.set_new_root(game)

    setup_node: ts.ActionNode = first(filter(lambda n: n.simple_hash.find('Setup') > 0, tree.all_action_nodes))
    assert setup_node.get_accum_prob() == approx(2/3)

    a = search_select_step()
    assert a.action_type == ActionType.MOVE and a.position == Square(3, 3)
    with only_fixed_rolls(game, d6=[3]):
        game.step(a)
    tree.set_new_root(game)

    a = search_select_step()
    assert a.action_type == ActionType.MOVE and a.position.x == 1
    with only_fixed_rolls(game):
        game.step(a)
    tree.set_new_root(game)


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

    tree = ts.SearchTree(game)
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

    tree = ts.SearchTree(game)
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
    assert len(tree.all_action_nodes) == 2 + 4 + 1 + 13

    game.step(action_p1_1)
    tree.set_new_root(game)

    assert len(tree.all_action_nodes) == 14
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


def print_node(node, weights):
    mcts_info = node.info

    for action, visits, action_val in zip(mcts_info.actions, mcts_info.visits, mcts_info.action_values):
        if visits > 0:
            action_value = np.dot(action_val, weights)
            a_index = node.explored_actions.index(action)
            child_node = node.children[a_index]
            assert child_node is not None
            expected_value = ts.get_node_value(child_node, weights)
            action.player = None
            print(f"{action}, {visits=}, avg(AV)={action_value/visits:.2f}, EV={expected_value:.2f}")


@pytest.mark.parametrize("tree_searcher", [ts.deterministic_tree_search_rollout,
                                           ts.mcts_ucb_rollout])
def test_mcts(tree_searcher):
    game, _ = get_custom_game_turn(player_positions=[(6, 6), (7, 7)],
                                   opp_player_positions=[(5, 6)],
                                   ball_position=(6, 6),
                                   pathfinding_enabled=True)

    weights = ts.HeuristicVector(score=1, ball_marked=0.1, ball_carried=0.2, ball_position=0.01, tv_on_pitch=1)

    tree = ts.SearchTree(game)
    policy = ts.MockPolicy()
    while len(tree.all_action_nodes) < 2000:
        tree_searcher(tree, policy, weights, exploration_coeff=5)

    print("")
    print(f"{tree_searcher.__name__}, num explored nodes = {len(tree.all_action_nodes)}")
    mcts_info = tree.root_node.info
    for action, visits, action_val in zip(mcts_info.actions, mcts_info.visits, mcts_info.action_values):
        action.player = None
        action_value = np.dot(action_val, weights)/(visits + (visits == 0))
        print(f"{action}, {visits=}, {action_value=:.4f}")
    print("")


@pytest.mark.parametrize("max_ma", [2, 1])
def test_blitz_reroll(max_ma):
    if max_ma == 2:
        fixed_d6 = [6, 1]
    elif max_ma == 1:
        fixed_d6 = [6, 6, 1]
    else:
        raise ValueError()

    game, (player, victim, *_) = get_custom_game_turn(player_positions=[(2, 5)],
                                                      opp_player_positions=[(5, 5), (4, 2), (4, 4), (4, 6), (4, 8)],
                                                      ball_position=(5, 5),
                                                      pathfinding_enabled=True)

    player.role.ma = max_ma
    game.state.home_team.state.rerolls = 3

    game.step(Action(ActionType.START_BLITZ, position=player.position))

    with only_fixed_rolls(game, d6=fixed_d6):
        game.step(Action(ActionType.BLOCK, position=victim.position))

    action = Action(ActionType.USE_REROLL)
    assert type(game.get_procedure()) is procedure.Reroll
    assert type(game.state.stack.items[-2]) is procedure.GFI

    tree = ts.SearchTree(game)
    tree.expand_action_node(tree.root_node, action)
    print("")


@pytest.mark.parametrize("name", ['botbowl-v4',
                                  'botbowl-11-v4',
                                  'botbowl-7-v4',
                                  'botbowl-5-v4',
                                  'botbowl-3-v4',
                                  'botbowl-1-v4'])
def test_game_state_hash(name):
    env = gym.make(name)
    random_bot = botbowl.RandomBot("randombot")
    for _ in range(10):
        env.reset()
        game: botbowl.Game = env.game
        game.away_agent.human = True
        game.home_agent.human = True

        assert len(game.get_available_actions()) > 0

        hashes = set()
        hashes.add(hashmap.create_gamestate_hash(game))
        latest_hashes = [hashmap.create_gamestate_hash(game)]
        end_setup = False

        while not game.state.game_over:
            if isinstance(game.get_procedure(), botbowl.core.procedure.Setup):
                if game.get_procedure().reorganize:
                    end_setup = True

            if end_setup:
                action = Action(ActionType.END_SETUP)
                end_setup = False
            else:
                aa_types = {ac.action_type for ac in game.get_available_actions()}
                if ActionType.SETUP_FORMATION_WEDGE in aa_types:
                    action = Action(ActionType.SETUP_FORMATION_WEDGE)
                    end_setup = True
                elif ActionType.SETUP_FORMATION_SPREAD in aa_types:
                    action = Action(ActionType.SETUP_FORMATION_SPREAD)
                    end_setup = True
                else:
                    action = random_bot.act(game)
                    while action.action_type == ActionType.UNDO:
                        action = random_bot.act(game)

            game.step(action)
            new_hash = hashmap.create_gamestate_hash(game)
            if new_hash in hashes:
                for r in game.state.reports:
                    print(r)
                print(f"action={action}")
                print(f"new hash:\n{new_hash}")

                print("Latest hashes:")
                if len(latest_hashes) > 5:
                    hashes_to_print = latest_hashes[-5:]
                else:
                    hashes_to_print = latest_hashes
                for h in reversed(hashes_to_print):
                    print(h)

                raise AssertionError("not unique game state")
            assert new_hash not in hashes
            hashes.add(new_hash)
            latest_hashes.append(new_hash)


def test_mock_policy():
    game, _ = get_custom_game_turn(player_positions=[(6, 6), (1, 1)],
                                   opp_player_positions=[(4, 4)],
                                   ball_position=(3, 3),
                                   pathfinding_enabled=True)

    policy = ts.MockPolicy()

    _, probs, actions = policy(game)

    game.step(actions[1])

    _, probs, actions = policy(game)

    print("")


def test_xml_tree():
    game, _ = get_custom_game_turn(player_positions=[(6, 6), (7, 7)],
                                   opp_player_positions=[(5, 6)],
                                   ball_position=(6, 6),
                                   pathfinding_enabled=True)

    weights = ts.HeuristicVector(score=1, ball_marked=0.1, ball_carried=0.2, ball_position=0.01, tv_on_pitch=1)

    tree = ts.SearchTree(game)
    policy = ts.MockPolicy()
    ts.deterministic_tree_search_rollout(tree, policy, weights, exploration_coeff=1)
    ts.deterministic_tree_search_rollout(tree, policy, weights, exploration_coeff=1)
    ts.deterministic_tree_search_rollout(tree, policy, weights, exploration_coeff=1)

    root = tree.to_xml()

    import xml.etree.ElementTree as ET
    print("")
    ET.dump(root)


@pytest.mark.parametrize("data", [(scenarios.five_player_hopeless, -1), ])
def test_deterministic_scenario_outcomes(data):
    scenario, approx_expected_value = data
    game = scenario()
    weights = ts.HeuristicVector(score=1, ball_marked=0, ball_carried=0, ball_position=0, tv_on_pitch=0)
    tree = ts.SearchTree(game)
    policy = ts.MockPolicy()

    cc_cond = ts.ContinueCondition(probability=0.01)

    for _ in range(1000):
        ts.deterministic_tree_search_rollout(tree, policy, weights, cc_cond=cc_cond, exploration_coeff=1)

    expected = np.mean([ts.get_node_value(child, weights) for child in tree.root_node.children])

    print("")
    print(f"{expected=}")