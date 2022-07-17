import dataclasses
from typing import Callable, Tuple, List

import numpy as np

import botbowl
import examples.tree_search as ts
from botbowl.core import procedure


def most_visited_action(node: ts.ActionNode, weights) -> botbowl.Action:
    index = np.argmax(node.info.visits)
    return node.info.actions[index]


def highest_valued_action(node: ts.ActionNode, weights: ts.HeuristicVector) -> botbowl.Action:
    visits = node.info.visits
    not_visited = visits == 0

    avg_action_values = np.matmul(node.info.action_values, weights) / (visits + not_visited)

    if node.is_home:
        avg_action_values[not_visited] -= 9999
        index = np.argmax(avg_action_values)
    else:
        avg_action_values[not_visited] += 9999
        index = np.argmin(avg_action_values)

    return node.info.actions[index]


def highest_expectimax_action(node: ts.ActionNode, weights: ts.HeuristicVector) -> botbowl.Action:
    child_node_values = [ts.get_node_value(child_node, weights) for child_node in node.children]
    if node.is_home:
        index = np.argmax(child_node_values)
    else:
        index = np.argmin(child_node_values)
    return node.explored_actions[np.argmax(index)]


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


def continue_expansion(node: ts.ActionNode, game: botbowl.Game, cc_cond, scores, half, end_turn_at, team) -> bool:
    if game.state.game_over:
        return False
    if cc_cond.single_drive and (get_score_sum(game) != scores or game.state.half != half):
        return False
    if get_team_turn_num(game, team) >= end_turn_at:
        return False

    # This ends the search after {my_team} has played its final turn, if opp_as_final_turn
    #if (not cc_cond.opp_as_final_turn) and team.state.turn == end_turn_at - 1:
    #    turn = game.current_turn
    #    if turn is not None and turn.team is not team:
    #        return False

    #if cc_cond.probability > 0 and node.get_accum_prob() < cc_cond.probability:
    #    return False

    if type(game.get_procedure() is procedure.Turn) and not node.info.visited_once:
        node.info.visited_once = True
        return False

    return True


def get_heuristic(game: botbowl.Game) -> ts.HeuristicVector:
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

    return ts.HeuristicVector(score=score,
                              tv_on_pitch=tv_on_pitch,
                              ball_position=ball_position,
                              ball_carried=ball_carried,
                              ball_marked=ball_marked)
