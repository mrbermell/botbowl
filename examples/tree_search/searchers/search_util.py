import botbowl
from collections import namedtuple
from examples.tree_search.SearchTree import ActionNode, HeuristicVector
import dataclasses
from typing import Callable, Tuple, List
import numpy as np


Policy = Callable[[botbowl.Game], Tuple[float, np.ndarray, List[botbowl.Action]]]


@dataclasses.dataclass
class ContinueCondition:
    probability: float = 0.02
    single_drive: bool = True
    turns: int = 1  # Note: turn counter increases after kickoff is resolved!
    opp_as_final_turn: bool = True


def get_score_sum(g) -> int:
    return g.state.home_team.state.score + g.state.away_team.state.score


def get_team_turn_num(g: botbowl.Game, team: botbowl.Team) -> int:
    return g.state.half * g.config.rounds + team.state.turn


def continue_expansion(node: ActionNode, game, cc_cond, scores, half, end_turn_at, team) -> bool:
    if game.state.game_over:
        return False
    if cc_cond.single_drive and (get_score_sum(game) != scores or game.state.half != half):
        return False
    if get_team_turn_num(game, team) >= end_turn_at:
        return False

    # This ends the search after {my_team} has played its final turn, if opp_as_final_turn
    if (not cc_cond.opp_as_final_turn) and team.state.turn == end_turn_at - 1:
        turn = game.current_turn
        if turn is not None and turn.team is not team:
            return False

    if cc_cond.probability > 0 and node.get_accum_prob() < cc_cond.probability:
        return False
    return True


def get_heuristic(game: botbowl.Game) -> HeuristicVector:
    """
    Heuristic based on game state, calculated from home teams perspective
    zero-sum, meaning away team's heuristic is negative of home team's heuristic
    :returns: array with different heuristics, multiply it with
    """
    score, tv_on_pitch, ball_position, ball_carried, ball_marked = 0.0, 0.0, 0.0, 0.0, 0.0
    home = game.state.home_team
    away = game.state.away_team

    score += home.state.score - away.state.score

    tv_on_pitch += sum(p.role.cost for p in game.get_players_on_pitch(team=home))
    tv_on_pitch -= sum(p.role.cost for p in game.get_players_on_pitch(team=away))
    tv_on_pitch /= 50000.0  # normalized to cost of lineman

    ball = game.get_ball()
    if ball is not None and ball.position is not None:
        ball_position -= ball.position.x  # negative because home team scores at x = 0

        home_marking_ball = len(game.get_adjacent_players(ball.position, team=home, standing=True)) > 0
        away_marking_ball = len(game.get_adjacent_players(ball.position, team=away, standing=True)) > 0

        if ball.is_carried:
            ball_carrier = game.get_player_at(ball.position)
            if ball_carrier.team == home:
                ball_carried += 1
                ball_marked -= away_marking_ball
            elif ball_carrier.team == away:
                ball_carried -= 1
                ball_marked += home_marking_ball
        else:
            ball_marked += (home_marking_ball - away_marking_ball)

    return HeuristicVector(score=score,
                           tv_on_pitch=tv_on_pitch,
                           ball_position=ball_position,
                           ball_carried=ball_carried,
                           ball_marked=ball_marked)
