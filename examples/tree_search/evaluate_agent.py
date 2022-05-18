import multiprocessing as mp
from typing import Tuple, Union

import botbowl
import examples.tree_search.evaluation_scenarios as scenarios
import more_itertools
import numpy as np
from examples.scripted_bot_example import MyScriptedBot

verbose = False


def runner(game: botbowl.Game, agent: botbowl.Agent, agent_as_home, opp_agent: botbowl.Agent, end_of_drive) \
        -> Union[Tuple[int, int], None]:
    if agent_as_home:
        game.replace_home_agent(agent)
        agent.new_game(game, game.state.home_team)

        game.replace_away_agent(opp_agent)
        opp_agent.new_game(game, game.state.away_team)

    else:
        game.replace_away_agent(agent)
        agent.new_game(game, game.state.away_team)

        game.replace_home_agent(opp_agent)
        opp_agent.new_game(game, game.state.home_team)

    if end_of_drive:
        game.config.fast_mode = False
        score_and_half = game.state.home_team.state.score + game.state.away_team.state.score + game.state.half
        game.step(botbowl.Action(botbowl.ActionType.CONTINUE))

        while not game.state.game_over:
            try:
                game.step()
            except Exception as e:
                print(f"Caught exception: {e}")
                return None

            new_score_and_half = game.state.home_team.state.score + game.state.away_team.state.score + game.state.half
            if score_and_half != new_score_and_half:
                if verbose:
                    print(f"h{game.state.home_team.state.score}-a{game.state.away_team.state.score}. "
                          f"half{game.state.half}:round{game.state.round}, reports={len(game.state.reports)}")
                break
    else:
        try:
            game.step(botbowl.Action(botbowl.ActionType.CONTINUE))
            assert game.state.game_over
        except Exception as e:
            print(f"Caught exception: {e}")
            return None

    if agent_as_home:
        agent_score = game.state.home_team.state.score
        opp_score = game.state.away_team.state.score
    else:
        agent_score = game.state.away_team.state.score
        opp_score = game.state.home_team.state.score

    return agent_score, opp_score


def benchmark_agent(create_agent_func, create_game_func, n=16, *, agent_as_home=True, create_opp_agent_func=None,
                    end_of_drive=False):

    experiments = ((create_game_func(), create_agent_func(), agent_as_home, create_opp_agent_func(), end_of_drive)
                   for _ in range(n))

    with mp.Pool(8) as p:
        results = np.stack(p.starmap(runner, experiments))
    results, none_results = map(list, more_itertools.partition(lambda x: x is None, results))

    td_rate = np.array(results)[:, 0].mean()
    opp_td_rate = np.array(results)[:, 1].mean()
    fail_rate = len(none_results) / (len(none_results) + len(results))

    return td_rate, opp_td_rate, fail_rate


def main():
    def create_agent():
        return MyScriptedBot('scripted')

    def create_opp_agent():
        return botbowl.RandomBot('random')  # MyScriptedBot('opp scripted')

    num_games = 64

    td_rate, opp_td_rate, fail_rate = benchmark_agent(create_agent, scenarios.eleven_after_kickoff, n=num_games,
                                                      end_of_drive=True, create_opp_agent_func=create_opp_agent)
    print(f"as home: {td_rate=:0.3f}, {opp_td_rate=:0.3f}, {fail_rate=:0.3f}")

    td_rate, opp_td_rate, fail_rate = benchmark_agent(create_agent, scenarios.eleven_after_kickoff, n=num_games,
                                                      end_of_drive=True, agent_as_home=False,
                                                      create_opp_agent_func=create_opp_agent)
    print(f"as away: {td_rate=:0.3f}, {opp_td_rate=:0.3f}, {fail_rate=:0.3f}")


if __name__ == '__main__':
    main()
