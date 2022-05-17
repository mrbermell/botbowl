import random
from typing import Optional

import botbowl
import more_itertools
from examples.scripted_bot_example import MyScriptedBot

import examples.tree_search.evaluation_scenarios as scenarios
import examples.tree_search.agent as agents
import numpy as np
import multiprocessing as mp


def runner(game: botbowl.Game, agent: botbowl.Agent, agent_as_home=True, opp_agent: botbowl.Agent = None,
           end_of_drive=False):
    if opp_agent is None:
        opp_agent = botbowl.RandomBot('random')

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
            except IndexError as e:
                print(f"Caught exception: {e}")
                return None, None

            new_score_and_half = game.state.home_team.state.score + game.state.away_team.state.score + game.state.half
            if score_and_half != new_score_and_half:
                break
    else:
        try:
            game.step(botbowl.Action(botbowl.ActionType.CONTINUE))
            assert game.state.game_over
        except IndexError as e:
            print(f"Caught exception: {e}")
            return None, None

    if agent_as_home:
        agent_score = game.state.home_team.state.score
        opp_score = game.state.away_team.state.score
    else:
        agent_score = game.state.away_team.state.score
        opp_score = game.state.home_team.state.score

    return agent_score, opp_score


def benchmark_agent(agent, game,  n=16, *, agent_as_home=True, opp_agent: botbowl.Agent = None,
                    end_of_drive=False):

    experiments = [(game, agent, agent_as_home, opp_agent, end_of_drive)] * n

    with mp.Pool(8) as p:
        results = np.stack(p.starmap(runner, experiments))
    results, none_results = map(list, more_itertools.partition(lambda x: x[0] is None, results))

    td_rate = np.array(results)[:, 0].mean()
    fail_rate = len(none_results) / (len(none_results) + len(results))

    return td_rate, fail_rate


def main_test():
    agent = MyScriptedBot('scripted')
    game = scenarios.seven_player_short_kick()

    runner(game, agent)


def main2():
    agent = MyScriptedBot('scripted')
    game = scenarios.seven_player_short_kick()

    td_rate, fail_rate = benchmark_agent(agent, game, n=8, end_of_drive=True)

    print(f"{td_rate=:0.3f}, {fail_rate=:0.3f}")


def main():
    pass
    agent = agents.SearchAgent()
    game = scenarios.five_player_away_threatens_score()

    runner(game, agent)

    #td_rate, fail_rate = benchmark_agent(agent, game, n=4, end_of_drive=True)
    #print(f"{td_rate=:0.3f}, {fail_rate=:0.3f}")


if __name__ == '__main__':
    main()
