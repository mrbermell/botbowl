from botbowl.core.model import Agent
from botbowl import Game
from copy import deepcopy

from time import perf_counter

from SearchTree import SearchTree

class SearchAgent(Agent):
    tree: SearchTree

    def __init__(self, name):
        super().__init__(name)
        self.tree = None

    def act(self, game: Game):
        if self.tree is None:
            self.tree = SearchTree(deepcopy(game))
        else:
            self.tree.set_new_root(game)


        time_left = game.get_seconds_left()
        if time_left < 10:
            assert False

        # expand the tree
        start_time = perf_counter()
        while perf_counter() - start_time < 10:
            pass


        # pick action to return


    def new_game(self, game, team):
        self.tree = None

    def end_game(self, game):
        pass
