import botbowl
from botbowl import Action, ActionType
import examples.tree_search as ts

times_evaluated = 5
num_rollouts = 1000

search_heuristics = [
    ts.HeuristicVector(score=1, ball_marked=0, ball_carried=0, ball_position=0, tv_on_pitch=0),
    ts.HeuristicVector(score=1, ball_marked=0, ball_carried=0, ball_position=0, tv_on_pitch=0),
    ts.HeuristicVector(score=1, ball_marked=0, ball_carried=0, ball_position=0, tv_on_pitch=0),
    ts.HeuristicVector(score=1, ball_marked=0, ball_carried=0, ball_position=0, tv_on_pitch=0),
    ts.HeuristicVector(score=1, ball_marked=0.1, ball_carried=0.2, ball_position=0.01, tv_on_pitch=1),
]

evaluation_heuristic = ts.HeuristicVector(score=1, ball_marked=0, ball_carried=0, ball_position=0, tv_on_pitch=0)


def get_game() -> botbowl.Game:
    env_conf = botbowl.ai.env.EnvConf(size=3, pathfinding=True)
    env = botbowl.ai.env.BotBowlEnv(env_conf)
    env.reset(skip_observation=True)

    game = env.game
    game.away_agent.human = True
    game.step(Action(ActionType.START_GAME))
    game.step(Action(ActionType.TAILS))
    game.step(Action(ActionType.TAILS))

    # renderer = botbowl.ai.env_render.EnvRenderer(env)

    # todo
    return game


def main():
    game = get_game()

    for search_heuristic in search_heuristics:
        for _ in range(times_evaluated):
            tree = ts.SearchTree(game)
            assert tree.root_node.is_home
            policy = ts.MockPolicy()

            for _ in range(num_rollouts):
                # explore
                ts.do_mcts_branch(tree, policy, weights=search_heuristic, exploration_coeff=1)

                # What is the expected value
                expected_value = max(map(ts.get_node_value, tree.root_node.children))


if __name__ == "__main__":
    main()
