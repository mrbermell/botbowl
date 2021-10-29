from abc import ABC, abstractmethod

import ffai
import gym
from dataclasses import dataclass, field
from typing import Optional, Any, Callable, Tuple, List
import ffai.core.forward_model as forward_model


@dataclass
class Node(ABC):
    parent: Optional['Node']
    children: list = field(init=False, default_factory=lambda: [])
    change_log: List[forward_model.Step] = field(init=False, default_factory=lambda: [])

    @abstractmethod
    def get_value(self):
        pass

    def do_thing(self):
        print(f"in do thing, {self.children} ")


@dataclass
class ActionNode(Node):
    reward: float = field(init=False, default=0.0)
    value: float = field(init=False, default=0.0)
    actions: List[ffai.Action] = field(default_factory=lambda: [])

    def get_value(self):
        pass


class ChanceNode(Node):
    def get_value(self):
        pass


@dataclass
class TreeSearcher:
    game: ffai.Game
    action_value_func: Callable[[ffai.Game], Tuple[List[ffai.Action], float]]
    root_node: ActionNode = field(init=False, default=None)

    def set_new_root(self, game: ffai.Game) -> None:
        pass

    def explore(self, num_node: int = None, max_time: int = None) -> None:
        # init root node - assume nothing saved
        assert self.root_node is None
        root_node = ActionNode(parent=None,  )
        root_node.actions, root_node.value = self.action_value_func(self.game)




    def get_best_action(self) -> ffai.Action:
        pass








def random_action_heuristic(game) -> Tuple[ffai.Action, float]:
    pass


def hash_game_state(game: ffai.Game) -> int:
    s = "".join([f"{p.position.x}{p.position.y}{int(p.state.up)}"
                         for p in game.get_players_on_pitch()])
    s += "".join([f"{ball_pos.x}{ball_pos.y}" for ball_pos in game.get_ball_positions()])
    return hash(s)


def get_action_with_roll(game: ffai.core.Game):
    aa = game.get_available_actions()
    if aa[0].action_type == ffai.ActionType.MOVE:
        action_choice = aa[0]
        for i, roll in enumerate(action_choice.rolls):
            if len(roll) > 0:
                return ffai.Action(ffai.ActionType.MOVE, position=action_choice.positions[i]), roll
    return None


if __name__ == "__main__":
    env = gym.make('FFAI-3-v3')
    env.reset()
    game = env.game

    ts = TreeSearcher(game, random_action_heuristic)
    ts.explore(max_time=10)
    action = ts.get_best_action()



    #n = ActionNode(None)
    #print(n.change_log)

    #exit()

    random_bot = ffai.ai.make_bot('random')
    for _ in range(10):
        action = random_bot.act(game)
        game.step(action)

    for _ in range(10):
        action = random_bot.act(game)
        game.step(action)
        print(f"{hash_game_state(game)} - {action.action_type}")
    #env.render()
    exit()


    for _ in range(20):
        t = get_action_with_roll(game)
        if t is None:
            print(".")
            action = random_bot.act(game)
        else:
            action, roll = t
            player = game.get_active_player()
            print(f"Player {player.nr}: Move action with roll {roll}")

            succeed = input("fail? [y/n]") == "Y"

            if succeed:
                ffai.D6.fix(roll[0])
            else:
                ffai.D6.fix(roll[0] - 1)

        game.step(action)

        env.render()
