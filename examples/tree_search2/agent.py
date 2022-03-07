import queue
from copy import deepcopy
from time import perf_counter
from typing import Optional

import botbowl
import botbowl.core.procedure as procedure
import numpy as np
from SearchTree import SearchTree
from botbowl import Game, ActionType
from botbowl.core.model import Agent, Action
from examples.tree_search2.Samplers import MockPolicy
from examples.tree_search2.Searchers import do_mcts_branch, HeuristicVector, get_node_value
from more_itertools import first
from tests.util import get_custom_game_turn


class SearchAgent(Agent):
    tree: Optional[SearchTree]
    team: Optional[botbowl.Team]
    queued_actions: queue.Queue

    def __init__(self, name):
        super().__init__(name)
        self.tree = None
        self.queued_actions = queue.Queue()
        self.team = None
        self.policy = MockPolicy()
        self.weights = HeuristicVector(score=1, ball_marked=0.01, ball_carried=0.1, ball_position=0.001, tv_on_pitch=0)
        self.width = None
        self.height = None

    def act(self, game: Game) -> Action:
        assert game.active_team is self.team
        scripted_action = self.scripted_action(game)
        if scripted_action is not None:
            if not game._is_action_allowed(scripted_action):
                raise ValueError()
            print(f"scripted action: {scripted_action}")
            return scripted_action

        if self.tree is None:
            self.tree = SearchTree(game)
        else:
            self.tree.set_new_root(game)

        while True:
            for _ in range(5):
                do_mcts_branch(self.tree, self.policy, self.weights, exploration_coeff=0.5)

            if self.tree.root_node.info.visits.sum() > 25 or game.get_seconds_left() < 5:
                break

        child_values = [get_node_value(child_node, self.weights) for child_node in self.tree.root_node.children]
        action = self.tree.root_node.explored_actions[np.argmax(child_values)]

        print(f"action: {action}")
        return action

    def scripted_action(self, game: botbowl.Game):
        if self.queued_actions.qsize() > 0:
            return self.queued_actions.get()

        proc = game.get_procedure()
        aa = game.get_available_actions()

        if isinstance(proc, procedure.Setup):
            not_allowed = {ActionType.END_SETUP, ActionType.PLACE_PLAYER}
            action_choice = first(filter(lambda ac: ac.action_type not in not_allowed, aa))
            self.queued_actions.put(Action(ActionType.END_SETUP))
            return Action(action_choice.action_type)

        if isinstance(proc, procedure.PlaceBall):
            x = self.width // 4 + 1
            if self.team is game.state.away_team:
                x *= 3
            y = self.height // 2 + 1
            return Action(ActionType.PLACE_BALL, position=botbowl.Square(x, y))

        if isinstance(proc, procedure.Touchback):
            return Action(ActionType.SELECT_PLAYER, player=aa[0].players[0])

        if isinstance(proc, procedure.CoinTossFlip):
            return Action(ActionType.HEADS)

        if isinstance(proc, procedure.CoinTossKickReceive):
            return Action(ActionType.RECEIVE)

        if isinstance(proc, procedure.Interception):
            return Action(ActionType.SELECT_PLAYER, player=aa[0].players[0])

        _, _, policy_actions = self.policy(game)
        if len(policy_actions) == 1:
            return policy_actions[0]

        return None

    def new_game(self, game, team):
        self.tree = None
        self.team = team
        self.height = len(game.square_shortcut)
        self.width = len(game.square_shortcut[0])

    def end_game(self, game):
        pass


def main():
    env_conf = botbowl.ai.env.EnvConf(size=3, pathfinding=True)
    env = botbowl.ai.env.BotBowlEnv(env_conf)
    env.reset(skip_observation=True)

    game = env.game
    agent = SearchAgent("searcher agent")
    agent.new_game(game, game.state.home_team)
    game.step(Action(ActionType.START_GAME))

    def calc_score():
        return game.state.home_team.state.score, game.state.away_team.state.score

    current_score = calc_score()

    while not game.state.game_over:
        action = agent.act(game)
        game.step(action)
        if current_score != calc_score():
            current_score = calc_score()
            print(f"Goal! {current_score}")

    print(f"home: {game.state.home_team.state.score}, away: {game.state.away_team.state.score}")


if __name__ == "__main__":
    main()
