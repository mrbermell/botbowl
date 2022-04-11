import copy
import time
from functools import partial
from typing import Callable, Dict

import botbowl
import examples.tree_search as ts
from botbowl import ActionType, Action
from examples.tree_search import hashmap
from examples.tree_search.searchers import search_util
from examples.tree_search.searchers.mcts import vanilla_mcts_rollout


class SearchAgent(botbowl.Agent):
    def __init__(self, name,
                 tree_search_rollout: Callable[[ts.SearchTree], None],
                 final_action_choice_strategy: Callable[[ts.ActionNode], botbowl.Action],
                 seconds_per_action: int,
                 policy: ts.Policy = None,
                 ):
        super().__init__(name)
        self.tree = None
        self.policy = policy
        self.tree_search_rollout = tree_search_rollout
        self.final_action_choice_strategy = final_action_choice_strategy
        self.seconds_per_action = seconds_per_action

    def new_game(self, game, team):
        self.tree = ts.SearchTree(game)

    def end_game(self, game):
        pass

    def act(self, game: botbowl.Game) -> Action:

        scripted_action = SearchAgent.get_scripted_action(game, self.policy)
        if scripted_action is not None:
            return scripted_action

        self.tree.set_new_root(game)
        start_time = time.perf_counter()
        while True:
            self.tree_search_rollout(self.tree)

            if time.perf_counter() - start_time > self.seconds_per_action:
                break

        action = self.final_action_choice_strategy(self.tree.root_node)

        return action

    @staticmethod
    def get_scripted_action(game, policy):
        scripted_action = ts.Samplers.scripted_action(game)
        if scripted_action is None and policy is not None:
            _, _, actions = policy(game)
            if len(actions) == 1:
                scripted_action = actions[0]

        if scripted_action is not None:
            return scripted_action

        return None


class VanillaMCTSSearchAgent(botbowl.Agent):
    def __init__(self, name,
                 tree_search_rollout: Callable[[ts.ActionNode, botbowl.Game, Dict[str, ts.ActionNode]], None],
                 final_action_choice_strategy: Callable[[ts.ActionNode], botbowl.Action],
                 seconds_per_action: int,
                 policy: ts.Policy = None,
                 ):
        super().__init__(name)
        self.all_action_nodes = dict()
        self.root_node = None
        self.policy = policy
        self.final_action_choice_strategy = final_action_choice_strategy
        self.tree_search_rollout = tree_search_rollout
        self.seconds_per_action = seconds_per_action

    def new_game(self, game, team):
        self.all_action_nodes = dict()
        self.root_node = None

    def end_game(self, game):
        pass

    def act(self, game: botbowl.Game) -> Action:

        scripted_action = SearchAgent.get_scripted_action(game, self.policy)
        if scripted_action is not None:
            print(f"scripted action: {scripted_action}")
            return scripted_action

        node_hash = hashmap.create_gamestate_hash(game)
        if node_hash in self.all_action_nodes:
            self.root_node = self.all_action_nodes[node_hash]
        else:
            self.all_action_nodes = {node_hash: self.root_node}
            self.root_node = ts.ActionNode(game, parent=None)

        start_time = time.perf_counter()
        my_game = copy.deepcopy(game)
        my_game.enable_forward_model()
        while True:
            self.tree_search_rollout(self.root_node, my_game, self.all_action_nodes)

            if time.perf_counter() - start_time > self.seconds_per_action:
                break

        action = self.final_action_choice_strategy(self.root_node)
        print(f"searched action: {action}")
        return action


def create_baseline_mcts_agent(weights) -> botbowl.Agent:
    policy = ts.MockPolicy()

    rollout = partial(vanilla_mcts_rollout,
                      policy=policy,
                      weights=weights,
                      )
    return VanillaMCTSSearchAgent(name='baseline mcts agent',
                                  tree_search_rollout=rollout,
                                  seconds_per_action=15,
                                  policy=policy,
                                  final_action_choice_strategy=partial(search_util.highest_valued_action,
                                                                       weights=weights))


def main():
    weights = ts.HeuristicVector(score=1, ball_marked=0.1, ball_carried=0.2, ball_position=0.01, tv_on_pitch=0.3)
    agent = create_baseline_mcts_agent(weights)

    env_conf = botbowl.ai.env.EnvConf(size=3, pathfinding=True)
    env = botbowl.ai.env.BotBowlEnv(env_conf)
    env.reset(skip_observation=True)

    game = env.game

    agent.new_game(game, game.state.home_team)
    game.step(Action(ActionType.START_GAME))
    renderer = botbowl.ai.env_render.EnvRenderer(env)

    def calc_score():
        return game.state.home_team.state.score, game.state.away_team.state.score

    current_score = calc_score()

    while not game.state.game_over:
        action = agent.act(game)
        game.step(action)
        renderer.render()
        if current_score != calc_score():
            current_score = calc_score()
            print(f"Goal! {current_score}")

    print(f"home: {game.state.home_team.state.score}, away: {game.state.away_team.state.score}")


if __name__ == "__main__":
    main()
