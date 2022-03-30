import queue
import numpy as np
from typing import Optional

import botbowl
from botbowl import ActionType, Action
import examples.tree_search as ts


class SearchAgent(botbowl.Agent):
    tree: Optional[ts.SearchTree]
    team: Optional[botbowl.Team]
    queued_actions: queue.Queue

    def __init__(self, name):
        super().__init__(name)
        self.tree = None
        self.queued_actions = queue.Queue()
        self.policy = ts.MockPolicy()
        self.weights = ts.HeuristicVector(score=1, ball_marked=0.01, ball_carried=0.1, ball_position=0.001, tv_on_pitch=0)

    def act(self, game: botbowl.Game) -> Action:

        scripted_action = self.get_scripted_action(game)
        if scripted_action is not None:
            return scripted_action

        self.tree.set_new_root(game)
        while True:
            ts.deterministic_tree_search_rollout(self.tree, self.policy, self.weights, exploration_coeff=0.5)
            if self.tree.root_node.info.visits.sum() > 25 or game.get_seconds_left() < 5:
                break

        # expectimax
        child_values = [ts.get_node_value(child_node, self.weights) for child_node in self.tree.root_node.children]
        action = self.tree.root_node.explored_actions[np.argmax(child_values)]

        print(f"num_visits={self.tree.root_node.info.visits.sum()} action: {action}")
        return action

    def get_scripted_action(self, game):
        scripted_action = ts.Samplers.scripted_action(game)
        if scripted_action is None:
            _, _, actions = self.policy(game)
            if len(actions) == 1:
                scripted_action = actions[0]

        if scripted_action is not None:
            print(f"scripted action: {scripted_action}")
            return scripted_action

        return None

    def new_game(self, game, team):
        self.tree = ts.SearchTree(game)

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
