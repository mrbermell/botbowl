import pytest

from tests.util import get_custom_game_turn
from .TreeSearcher import expand
import botbowl
from botbowl import Square, Action, ActionType


@pytest.mark.parametrize("data", [(Square(2, 2), [1.0]),
                                  (Square(3, 3), [4/6, 2/6])])
def test_expand_move(data):
    move_target, outcome_probs = data
    game, player, _, _ = get_custom_game_turn(player_positions=[(1, 1)],
                                              opp_player_positions=[(1, 3), (3, 1)],
                                              ball_position=(5, 5))
    game.config.pathfinding_enabled = True
    game.set_available_actions()

    game.step(Action(ActionType.START_MOVE, position=player.position))

    action = botbowl.Action(botbowl.ActionType.MOVE, position=move_target)
    steps, probs = expand(game, action)

    assert len(probs) == len(outcome_probs)
    assert probs == outcome_probs
