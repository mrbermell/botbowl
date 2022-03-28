from typing import Callable

import botbowl
from tests.util import get_custom_game_turn


def three_player_after_kickoff():
    game, _ = get_custom_game_turn(player_positions=[],
                                   opp_player_positions=[],
                                   ball_position=(1, 1),
                                   rerolls=3,
                                   size=3,
                                   pathfinding_enabled=True)

    return game


def five_player_away_threatens_score():
    game, _ = get_custom_game_turn(player_positions=[(13, 2), (12, 3), (14, 5), (12, 7), (13, 8)],
                                   opp_player_positions=[(11, 2), (10, 3), (11, 4), (10, 5), (12, 5)],
                                   ball_position=(10, 3),
                                   rerolls=1,
                                   size=5,
                                   turn=7,
                                   pathfinding_enabled=True)

    return game


def five_player_hopeless():
    game, _ = get_custom_game_turn(player_positions=[(13, 2)],
                                   opp_player_positions=[(11, 2), (10, 3), (11, 4), (10, 5), (12, 5)],
                                   ball_position=(10, 3),
                                   rerolls=1,
                                   size=5,
                                   turn=7,
                                   pathfinding_enabled=True)

    return game


def draw_game(game):
    env = botbowl.ai.env.BotBowlEnv()
    env.game = game
    renderer = botbowl.ai.env_render.EnvRenderer(env)

    renderer.render()
    input()


if __name__ == "__main__":
    draw_game(five_player_hopeless())
