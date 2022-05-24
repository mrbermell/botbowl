import multiprocessing as mp
import random
import time
from typing import Tuple, Union

import botbowl
import examples.tree_search.evaluation_scenarios as scenarios
import more_itertools
import numpy as np
from examples.scripted_bot_example import MyScriptedBot
import examples.tree_search.searchers.mcts_example as mcts_example
from more_itertools import first




class BenchmarkRunner:

    def __init__(self, create_agent_func):
        self.experiments = {}
        self.create_agent_func = create_agent_func

    def add_benchmark(self, name, create_game, create_opp_agent, n=16, *, agent_as_home=True, end_of_drive=False):

        exp = ((create_game(), self.create_agent_func(), agent_as_home, create_opp_agent(), end_of_drive)
               for _ in range(n))
        self.experiments[name] = exp
        return self

    def run(self):
        with mp.Pool(4) as p:
            async_results = {name: p.starmap_async(runner, experiment) for name, experiment in self.experiments.items()}

            while len(async_results) > 0:
                result_name = None
                for name, async_result in async_results.items():
                    if async_result.ready():
                        result_name = name
                        break

                if result_name is None:
                    time.sleep(1)
                    continue

                results = async_results.pop(result_name).get(1)
                results, none_results = map(list, more_itertools.partition(lambda x: x is None, results))
                results = np.array(results)
                fail_rate = len(none_results) / (len(none_results) + len(results))

                print(f"{result_name}: {results.mean(axis=0)}, fail rate={fail_rate}")


def runner(game: botbowl.Game,
           agent: botbowl.Agent,
           agent_as_home,
           opp_agent: botbowl.Agent,
           end_of_drive,
           draw=False,
           verbose=False) \
        -> Union[Tuple[int, int], None]:

    env = botbowl.ai.env.BotBowlEnv()
    env.game = game
    renderer = botbowl.ai.env_render.EnvRenderer(env)


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

        num_reports = len(game.state.reports)

        while not game.state.game_over:
            try:
                game.step()
            except Exception as e:
                print(f"Caught exception: {e}")
                return None

            if verbose and num_reports < len(game.state.reports):
                for r in game.state.reports[num_reports:]:
                    print(r)
                num_reports = len(game.state.reports)
            if draw:
                renderer.render()

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
    num_games = 200
    scenario = scenarios.five_player_easy_home_score
    mp_enabled = False

    def create_agent():
        # return MyScriptedBot('scripted')
        return mcts_example.MCTSBot("mcts bot", seconds=25)

    def create_opp_agent():
        return botbowl.RandomBot('random')  # MyScriptedBot('opp scripted')

    def create_game():
        game = scenario()
        game.set_seed(random.randint(0, 2**32))
        return game

    if mp_enabled:
        manager = BenchmarkRunner(create_agent) \
            .add_benchmark(scenario.__name__ + '(home)', create_game, create_opp_agent, num_games, end_of_drive=True, agent_as_home=True) \
            #.add_benchmark(scenario.__name__ + '(away)', create_game, create_opp_agent, num_games, end_of_drive=True, agent_as_home=False)
        manager.run()
    else:
        r = runner(game=create_game(), agent=create_agent(), agent_as_home=True, opp_agent=create_opp_agent(),
                   end_of_drive=True, draw=True, verbose=True)
        print(f"result = {r}, is none={r is None}")


if __name__ == '__main__':
    main()
