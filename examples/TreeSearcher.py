import ffai
from ffai import ActionType, Action
from typing import Optional, Callable, Tuple, List
from abc import ABC, abstractmethod
import ffai.core.forward_model as forward_model
from tests.util import get_game_turn


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
    actions: List[ffai.Action]

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
    game: ffai.Game
    action_value_func: Callable[[ffai.Game], Tuple[List[ffai.Action], float]]
    root_node: ActionNode

    def __init__(self, game, action_value_func):
        self.game = game
        self.action_value_func = action_value_func
        self.root_node = None

    def set_new_root(self, game: ffai.Game) -> None:
        pass

    def explore(self, num_node: int = None, max_time: int = None) -> None:
        # init root node - assume nothing saved
        assert self.root_node is None
        root_node = ActionNode(parent=None )
        root_node.actions, root_node.value = self.action_value_func(self.game)
        assert len(root_node.actions) > 0

        self.root_node = root_node






    def get_best_action(self) -> ffai.Action:
        pass


def expand(game: ffai.Game, action: ffai.Action) -> List[Tuple[forward_model.Step, float]]:
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
            rolls = [x for xs in fail_rolls for x in xs] #concat the list
            for _ in rolls:
                ffai.D6.fix(6)

            game.step(action)
            print(f"{game.get_player_at(action.position)}")
            assert len(ffai.D6.FixedRolls) == 0

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
    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    game.clear_board()
    game.config.pathfinding_enabled = True

    player = team.players[0]
    start_square = ffai.Square(2,2)
    target_square = ffai.Square(9,9)
    game.put(player, ffai.Square(2,2))

    opp_player = game.get_opp_team(team).players[0]
    game.put(opp_player, ffai.Square(4, 4))

    game.get_ball().move_to(target_square)
    game.get_ball().is_carried = False

    game.set_available_actions()

    game.step(Action(ActionType.START_MOVE, position=start_square))

    expansion = expand(game, Action(ActionType.MOVE, position=target_square))
    for steps, prob in expansion:
        print(steps)
        print(prob)

    print("hej")

if __name__ == "__main__":
    main()