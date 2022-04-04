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


def seven_player_short_kick():
    game, _ = get_custom_game_turn(player_positions=[(11, 3), (11, 4), (11, 5), (11, 6), (11, 7), (13, 5), (17, 5)],
                                   opp_player_positions=[(10, 4), (10, 6), (8, 2), (8, 8), (8, 5), (7, 3), (7, 7)],
                                   ball_position=(12, 3),
                                   rerolls=2,
                                   size=7,
                                   turn=4,
                                   pathfinding_enabled=True)

    return game


def seven_player_break_screen():
    game, _ = get_custom_game_turn(player_positions=[(6, 5), (7, 7), (8, 5), (7, 8), (9, 5), (13, 5), (17, 5)],
                                   opp_player_positions=[(5, 2), (4, 2), (5, 5), (4, 5), (5, 8), (4, 8), (4, 6)],
                                   ball_position=(12, 3),
                                   rerolls=2,
                                   size=7,
                                   turn=5,
                                   pathfinding_enabled=True)

    return game


def eleven_easy_two_turn_score():
    game, _ = get_custom_game_turn(player_positions=[(13, 3), (5, 4), (12, 5), (13, 7), (19, 3), (7, 4), (18, 6)],
                                   opp_player_positions=[(20, 10), (21, 9), (18, 7), (19, 10)],
                                   ball_position=(13, 3),
                                   rerolls=0,
                                   size=11,
                                   turn=7,
                                   pathfinding_enabled=True)

    return game


def five_player_hopeless():
    game, _ = get_custom_game_turn(player_positions=[(2, 2)],
                                   opp_player_positions=[(13, 5)],
                                   ball_position=(13, 5),
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
