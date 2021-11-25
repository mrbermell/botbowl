import botbowl
from botbowl import ActionType, Action
from typing import Optional, Callable, Tuple, List
from abc import ABC, abstractmethod
import botbowl.core.forward_model as forward_model
from tests.util import get_game_turn, get_custom_game_turn


class Node(ABC):
    parent: Optional['Node']
    children: List['Node']
    change_log: List[forward_model.Step]

    def __init__(self, parent: Optional['Node']):
        self.parent = parent
        self.children = []
        self.change_log = []

    @abstractmethod
    def get_value(self):
        pass

    def do_thing(self):
        print(f"in do thing, {self.children} ")


class ActionNode(Node):
    reward: float
    value: float
    actions: List[botbowl.Action]

    def __init__(self, parent, reward = 0.0, actions=None):
        super().__init__(parent)
        self.reward = reward
        self.actions = [] if actions is None else actions

    def get_value(self):
        pass


class ChanceNode(Node):
    def get_value(self):
        pass



class TreeSearcher:
    game: botbowl.Game
    action_value_func: Callable[[botbowl.Game], Tuple[List[botbowl.Action], float]]
    root_node: ActionNode

    def __init__(self, game, action_value_func):
        self.game = game
        self.action_value_func = action_value_func
        self.root_node = None

    def set_new_root(self, game: botbowl.Game) -> None:
        pass

    def explore(self, num_node: int = None, max_time: int = None) -> None:
        # init root node - assume nothing saved
        assert self.root_node is None
        root_node = ActionNode(parent=None )
        root_node.actions, root_node.value = self.action_value_func(self.game)
        assert len(root_node.actions) > 0

        self.root_node = root_node






    def get_best_action(self) -> botbowl.Action:
        pass


def expand(game: botbowl.Game, action: botbowl.Action) -> List[Tuple[forward_model.Step, float]]:
    """
    :param game - game object used for calculations. Will be reverted to original state.
    :param action - action to be evaluated.
    :returns - list of tuples containing (Steps, probability) for each possible outcome.
             - probabilities sums to 1.0
    """
    assert game._is_action_allowed(action)

    game.enable_forward_model()
    assert game.trajectory.enabled
    report_idx = len(game.state.reports)

    # init
    steps = []
    probs = []
    root_step = game.get_step()

    # action_choice =

    # RE-ROLL proc

    # BLITZ

    # BLOCK

    # MOVE

    if action.action_type == ActionType.MOVE:
        print("MOVE!")
        path = game.state.stack.peek().paths[action.position]
        print(f"{path.prob=}")
        print(f"{path.rolls=}")

        is_pickup = game.get_ball().position == action.position
        print(f"{is_pickup=}")

        if path.prob < 1.0:
            fail_squares = []
            fail_rolls = []
            for sq, rolls in zip(path.steps, path.rolls):
                if len(rolls) > 0:
                    fail_squares.append(sq)
                    fail_rolls.append(rolls)

            # Success scenario
            rolls = [x for xs in fail_rolls for x in xs]  # concat the list
            for _ in rolls:
                botbowl.D6.fix(6)

            game.step(action)
            print(f"{game.get_player_at(action.position)}")
            assert len(botbowl.D6.FixedRolls) == 0

            for r in game.state.reports[report_idx:]:
                print(r.outcome_type)

            steps.append(game.trajectory.action_log[root_step:])
            probs.append(path.prob)

            # get fail squares and their rolls (dodge, gfi)
            # pickup ball  (should always be last square)
        else:
            pass

    return list(zip(steps, probs))


def main():
    ball_pos = botbowl.Square(9, 9)
    game, player, opp_player = get_custom_game_turn(player_positions=[(2, 2)],
                                                    opp_player_positions=[(4, 4)],
                                                    ball_position=ball_pos)
    game.config.pathfinding_enabled = True
    game.set_available_actions()

    game.step(Action(ActionType.START_MOVE, position=player.position))

    expansion = expand(game, Action(ActionType.MOVE, position=ball_pos))
    for steps, prob in expansion:
        print(steps)
        print(prob)

    print("hej")


if __name__ == "__main__":
    main()