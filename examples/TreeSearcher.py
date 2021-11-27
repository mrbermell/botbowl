import botbowl
import numpy as np
from botbowl import ActionType, Action
from typing import Optional, Callable, Tuple, List
from abc import ABC, abstractmethod
import botbowl.core.forward_model as forward_model
from tests.util import get_game_turn, get_custom_game_turn

from botbowl.core.pathfinding.python_pathfinding import Path

accumulated_prob_2d_roll = np.array([36, 36, 36, 35, 33, 30, 26, 21, 15, 10, 6, 3, 1]) / 36

NodeAction = botbowl.Action


class ActionSampler:
    actions: List[NodeAction]

    def __init__(self, game: botbowl.Game):

        actions = []
        for action_choice in game.get_available_actions():
            positions = action_choice.positions
            if len(positions)>0:
                for sq in positions:
                    actions.append(botbowl.Action(action_choice.action_type, position=sq))
            else:
                actions.append(botbowl.Action(action_choice.action_type))

        self.actions = actions

    def get_action(self) -> NodeAction:
        return np.random.choice(self.actions, 1)[0]

    def __len__(self):
        return len(self.actions)


class Node(ABC):
    parent: Optional['Node']
    children: List['Node']
    change_log: List[forward_model.Step]

    def __init__(self, parent: Optional['Node'], steps: [List[forward_model.Step]]):
        self.parent = parent
        self.children = []
        self.change_log = steps

    @abstractmethod
    def get_value(self):
        pass

    def do_thing(self):
        print(f"in do thing, {self.children} ")


class ActionNode(Node):
    reward: float
    value: float
    action_sampler: ActionSampler

    def __init__(self, parent: Optional[Node], steps: List[forward_model.Step], action_sampler: ActionSampler, reward = 0.0):
        super().__init__(parent, steps)
        self.reward = reward
        self.action_sampler = action_sampler

    def get_value(self):
        pass


class ChanceNode(Node):
    """
    Contains different outcomes of dices rolls.

    If there are not any connected children,
    then step the game from here until there are available actions
    could possibly be converted to an ActionNode
    """

    child_probability: List[float]

    def __init__(self, parent: Optional[Node], steps: List[forward_model.Step]):
        super().__init__(parent, steps)
        self.child_probability = []

    def connect_child(self, child: Node, prob: float):
        self.children.append(child)
        self.child_probability.append(prob)

    def get_value(self):
        pass


class TreeSearcher:
    game: botbowl.Game
    action_value_func: Callable[[botbowl.Game], ActionSampler]
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


def expand(game: botbowl.Game, action: botbowl.Action) \
        -> Node:
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

        path: Path = game.get_procedure().paths[action.position]
        is_pickup = game.get_ball().position == action.position

        if path.prob < 1.0:

            # Concat list of rolls
            rolls = [x for xs in path.rolls for x in xs]
            if is_pickup:
                rolls.pop()

            # Decide which roll to fail. step until there and create the chance node.

            # which roll shall fail?
            p = np.array(rolls) / 6
            p /= np.sum(p)
            index_of_failure = np.random.choice(range(len(rolls)), 1, p=p)[0]

            # fix all rolls up until the failure, and step them
            for _ in range(index_of_failure):
                botbowl.D6.fix(6)

            game.step(action, slow_step=True)
            while len(botbowl.D6.FixedRolls) > 0:
                game.step(slow_step=True)

            # Create the ChanceNode
            steps = game.trajectory.action_log[root_step:]
            chance_node = ChanceNode(parent=None, steps=steps)

            #reset the root step to new chance node
            root_step = game.get_step()

            ### SUCCESS SCENARIO ###
            for _ in range(len(rolls) - index_of_failure):
                botbowl.D6.fix(6)
            while len(botbowl.D6.FixedRolls) > 0:
                game.step(slow_step=True)

            if is_pickup:
                # need to expand a pickup node, gaah!
                raise NotImplementedError()
            else:
                while len(game.get_available_actions()) == 0:
                    game.step(slow_step=True)
                steps = game.trajectory.action_log[root_step:]
                action_node = ActionNode(parent=chance_node, steps=steps, action_sampler=ActionSampler(game))
                chance_node.connect_child(action_node, path.prob)

            ### FAILURE SCENARIO ###
            game.revert(root_step)
            game.set_available_actions()
            player = game.get_active_player()
            assert player is not None
            if player.has_skill(botbowl.Skill.DODGE) or game.can_use_reroll(player.team):
                # how to handle this? Gaah!
                raise NotImplementedError()

            fix_step_connect(game, chance_node, d6_fixes=[1], prob=1-path.prob,
                             step_condition=lambda g: not type(g.get_procedure()) is botbowl.KnockDown)

            return chance_node

        else:
            # Do the action, create next action node. Leave Game in new state.
            root_step = game.get_step()
            game.step(action)
            steps = game.trajectory.action_log[root_step:]

            return ActionNode(None, steps, action_sampler=ActionSampler(game))


def fix_step_connect(game: botbowl.Game,
                     parent: ChanceNode,
                     d6_fixes: List[int],
                     prob: float,
                     step_condition: Callable[[botbowl.Game], bool]):
    for roll_fix in d6_fixes:
        botbowl.D6.fix(roll_fix)
    root_step = game.get_step()
    while step_condition(game):
        game.step(slow_step=True)
    steps = game.trajectory.action_log[root_step:]
    new_chance_node = ChanceNode(parent=parent, steps=steps)
    parent.connect_child(new_chance_node, prob=prob)
    assert len(botbowl.D6.FixedRolls) == 0
    game.revert(root_step)


def expand_knockdown(node: ChanceNode, game: botbowl.Game) -> None:
    """
    :param node: node to store the different outcomes in.
    :param game:
    """
    # noinspection PyTypeChecker
    proc: botbowl.KnockDown = game.get_procedure()
    assert type(proc) is botbowl.KnockDown
    assert proc.armor_roll or proc.in_crowd

    def step_condition(g: botbowl.Game):
        return proc in g.state.stack.items

    def _fix_step_connect(d6_fixes: List[int], prob: float):
        fix_step_connect(game, node, d6_fixes, prob, step_condition)

    if proc.in_crowd:
        _fix_step_connect(d6_fixes=[1, 2], prob=1.0)  # injury roll is a KO
    else:
        p_armorbreak = accumulated_prob_2d_roll[ proc.player.get_av() + 1]
        p_injury_removal = accumulated_prob_2d_roll[8]  # KO

        _fix_step_connect(d6_fixes=[1, 2], prob=1.0 - p_armorbreak)  # No armorbreak
        _fix_step_connect(d6_fixes=[6, 5, 1, 2], prob=p_armorbreak * (1.0 - p_injury_removal))  # Stunned
        _fix_step_connect(d6_fixes=[6, 5, 4, 5], prob=p_armorbreak * p_injury_removal)  # KO


#if __name__ == "__main__":
#    pass