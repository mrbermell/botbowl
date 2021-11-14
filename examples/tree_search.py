

import ffai
import gym
from typing import Tuple, List

from TreeSearcher import TreeSearcher


def get_random_action(game):
    return ffai.ai.make_bot('random').act(game)



def random_action_heuristic(game) -> Tuple[List[ffai.Action], float]:
    action = get_random_action(game)
    return [action], 0.0


def hash_game_state(game: ffai.Game) -> int:
    s = "".join([f"{p.position.x}{p.position.y}{int(p.state.up)}"
                         for p in game.get_players_on_pitch()])
    s += "".join([f"{ball_pos.x}{ball_pos.y}" for ball_pos in game.get_ball_positions()])
    return hash(s)


def get_action_with_roll(game: ffai.core.Game):
    aa = game.get_available_actions()
    if aa[0].action_type == ffai.ActionType.MOVE:
        action_choice = aa[0]
        for i, roll in enumerate(action_choice.rolls):
            if len(roll) > 0:
                return ffai.Action(ffai.ActionType.MOVE, position=action_choice.positions[i]), roll
    return None


def main():
    env = gym.make('FFAI-3-v3')
    env.reset()
    game = env.game

    for _ in range(10):
        game.step(get_random_action(game))


    ts = TreeSearcher(game, random_action_heuristic)
    ts.explore(max_time=10)
    # action = ts.get_best_action()


    exit()



    for _ in range(20):
        t = get_action_with_roll(game)
        if t is None:
            print(".")
            action = get_random_action(game)
        else:
            action, roll = t
            player = game.get_active_player()
            print(f"Player {player.nr}: Move action with roll {roll}")

            succeed = input("fail? [y/n]") == "Y"

            if succeed:
                ffai.D6.fix(roll[0])
            else:
                ffai.D6.fix(roll[0] - 1)

        game.step(action)

        env.render()

if __name__ == "__main__":
    main()