from functools import partial
from operator import attrgetter, itemgetter
from typing import Tuple, Union, Optional

from pytest import approx

import botbowl
from examples.tree_search2.SearchTree import SearchTree, ActionNode, ChanceNode


def get_heuristic(game: botbowl.Game,
                  coeff_score: float = 0,
                  coeff_tv_on_pitch: float = 0,
                  coeff_ball_x: float = 0,
                  coeff_ball_marked: float = 0,
                  coeff_ball_carried: float = 0) -> float:
    """ heuristic based on game state, zero sum, calculated from home teams perspective """
    result: float = 0.0
    home = game.state.home_team
    away = game.state.away_team
    ball = game.get_ball()

    result += coeff_score * (home.state.score - away.state.score)

    result += coeff_tv_on_pitch * sum(map(attrgetter('role.cost'), game.get_players_on_pitch(team=home)))
    result -= coeff_tv_on_pitch * sum(map(attrgetter('role.cost'), game.get_players_on_pitch(team=away)))

    if ball is not None and ball.position is not None:
        result -= coeff_ball_x * ball.position.x  # negative because home team scores at x = 0

        home_marking_ball = len(game.get_adjacent_players(ball.position, team=home, standing=True)) > 0
        away_marking_ball = len(game.get_adjacent_players(ball.position, team=away, standing=True)) > 0

        if ball.is_carried:
            ball_carrier = game.get_player_at(ball.position)
            if ball_carrier.team == home:
                result += coeff_ball_carried
                result -= coeff_ball_marked * away_marking_ball
            elif ball_carrier.team == away:
                result -= coeff_ball_carried
                result += coeff_ball_marked * home_marking_ball
        else:
            result += coeff_ball_marked * (home_marking_ball - away_marking_ball)

    return result


gotebot_heuristic = partial(get_heuristic, coeff_score=1)#,
                            #coeff_ball_x=0.05,
                            #coeff_ball_marked=0.1,
                            #coeff_ball_carried=0.2)


def get_best_action(node: Union[ChanceNode, ActionNode]) -> Tuple[Optional[botbowl.Action], float]:
    if type(node) is ChanceNode:
        assert sum(node.child_probability) == approx(1.0, abs=1e-9), f"{node} {sum(node.child_probability)=} should be 1.0"
        return None, sum( prob * get_best_action(child)[1] for prob, child in zip(node.child_probability, node.children))

    elif type(node) is ActionNode:
        if len(node.children) == 0:
            return None, node.info['value']
        else:
            assert len(node.children) == len(node.explored_actions)
            action, value = max(((action, get_best_action(child)[1])
                                for action, child in zip(node.explored_actions, node.children)), key=itemgetter(1))
            return action, value
