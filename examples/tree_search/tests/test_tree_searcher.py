import examples.tree_search as ts
import examples.tree_search.evaluation_scenarios as scenarios
import gym
import numpy as np
import pytest
from botbowl import Square, Action, ActionType, botbowl
from examples.tree_search import hashmap
from more_itertools import first
from pytest import approx
from tests.util import get_custom_game_turn, only_fixed_rolls

default_weights = ts.HeuristicVector(score=1, ball_marked=0.1, ball_carried=0.2, ball_position=0.01, tv_on_pitch=1)


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
        ts.deterministic_tree_search_rollout(tree, policy, weights)
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
        ts.deterministic_tree_search_rollout(tree, policy, weights)
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
    assert setup_node.get_accum_prob() == approx(2 / 3)

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
            print(f"{action}, {visits=}, avg(AV)={action_value / visits:.2f}, EV={expected_value:.2f}")


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
        tree_searcher(tree, policy, weights)
    print("")
    print(f"{tree_searcher}, num explored nodes = {len(tree.all_action_nodes)}")
    mcts_info = tree.root_node.info
    for action, visits, action_val in zip(mcts_info.actions, mcts_info.visits, mcts_info.action_values):
        action.player = None
        action_value = np.dot(action_val, weights) / (visits + (visits == 0))
        print(f"{action}, {visits=}, {action_value=:.4f}")
    print("")


def test_vanilla_mcts():
    game, _ = get_custom_game_turn(player_positions=[(6, 6), (7, 7)],
                                   opp_player_positions=[(5, 6)],
                                   ball_position=(6, 6),
                                   pathfinding_enabled=True,
                                   forward_model_enabled=True)

    weights = ts.HeuristicVector(score=1, ball_marked=0.1, ball_carried=0.2, ball_position=0.01, tv_on_pitch=1)
    policy = ts.MockPolicy()
    all_action_nodes = dict()
    root_node = ts.ActionNode(game, parent=None)
    ts.vanilla_mcts_rollout(root_node, game, all_action_nodes, policy, weights, n=200)

    print(f"num explored nodes = {len(all_action_nodes)}")
    mcts_info = root_node.info
    for action, visits, action_val in zip(mcts_info.actions, mcts_info.visits, mcts_info.action_values):
        action.player = None
        action_value = np.dot(action_val, weights) / (visits + (visits == 0))
        print(f"{action}, {visits=}, {action_value=:.4f}")
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


def test_xml_tree():
    game, _ = get_custom_game_turn(player_positions=[(6, 6), (7, 7)],
                                   opp_player_positions=[(5, 6)],
                                   ball_position=(6, 6),
                                   pathfinding_enabled=True)

    weights = ts.HeuristicVector(score=1, ball_marked=0.1, ball_carried=0.2, ball_position=0.01, tv_on_pitch=0)
    tree = ts.SearchTree(game)
    policy = ts.MockPolicy()
    ts.deterministic_tree_search_rollout(tree, policy, weights)
    root = tree.to_xml(weights)
    root.write('test_output.xml')


@pytest.mark.parametrize("data", [(scenarios.five_player_hopeless, -1), ])
def test_deterministic_scenario_outcomes(data):
    scenario, approx_expected_value = data
    game = scenario()
    weights = ts.HeuristicVector(score=1, ball_marked=0, ball_carried=0, ball_position=0, tv_on_pitch=0)
    tree = ts.SearchTree(game)
    policy = ts.MockPolicy()
    cc_cond = ts.ContinueCondition(probability=0.01)
    ts.deterministic_tree_search_rollout(tree, policy, weights, cc_cond=cc_cond, search_time=10)
    expected = np.mean([ts.get_node_value(child, weights) for child in tree.root_node.children])

    print("")
    print(f"{expected=}")

