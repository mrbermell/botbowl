import time
from abc import ABC, abstractmethod
from collections import Counter
from functools import partial
from queue import PriorityQueue
from typing import Optional, Callable, List, Union, Dict, Any

import numpy as np
from more_itertools import collapse

import botbowl
import botbowl.core.forward_model as forward_model
import botbowl.core.procedure as procedures
from tests.util import only_fixed_rolls

accumulated_prob_2d_roll = np.array([36, 36, 36, 35, 33, 30, 26, 21, 15, 10, 6, 3, 1]) / 36

remove_randomness = partial(only_fixed_rolls, assert_fixes_consumed=False, assert_no_prev_fixes=False)


class Node(ABC):
    parent: Optional['Node']
    children: List['Node']
    change_log: List[forward_model.Step]
    step_nbr: int  # forward model's step count

    def __init__(self, game: botbowl.Game, parent: Optional['Node']):
        self.step_nbr = game.get_step()
        self.parent = parent
        self.children = []
        self.change_log = game.trajectory.action_log[self.step_nbr:] if parent is not None else []

    def _connect_child(self, child_node: 'Node'):
        assert child_node.parent is self
        self.children.append(child_node)


class ActionNode(Node):
    reward: float
    value: float
    team: botbowl.Team
    explored_actions: List[botbowl.Action]
    turn: int
    depth: int
    info: dict  # Only purpose is to store information for users of SearchTree

    def __init__(self, game: botbowl.Game, parent: Optional[Node], reward=0.0):
        super().__init__(game, parent)
        self.reward = reward
        self.team = game.state.available_actions[0].team
        self.explored_actions = []
        self.turn = self.team.state.turn
        self.info = {}

        if parent is None:
            self.depth = 0
        else:
            node: ActionNode = parent
            while type(node) != type(self):
                node = node.parent
            self.depth = node.depth + 1

    def connect_child(self, child_node: Node, action: botbowl.Action):
        super()._connect_child(child_node)
        self.explored_actions.append(action)


class ChanceNode(Node):
    """
    Contains different outcomes of dices rolls.

    If there are not any connected children,
    then step the game from here until there are available actions
    could possibly be converted to an ActionNode
    """

    child_probability: List[float]

    def __init__(self, game: botbowl.Game, parent: Optional[Node]):
        super().__init__(game, parent)
        self.child_probability = []

    def connect_child(self, child_node: Node, prob: float):
        super()._connect_child(child_node)
        self.child_probability.append(prob)


class SearchTree:
    game: botbowl.Game
    root_node: ActionNode
    all_action_nodes: List[ActionNode]

    def __init__(self, game):
        self.game = game
        self.root_node = ActionNode(game, None)
        self.all_action_nodes = [self.root_node]

    def set_new_root(self, game: botbowl.Game) -> None:
        pass  # todo

    def set_game_to_node(self, target_node: ActionNode) -> None:
        """Uses forward model to set self.game to the state of Node"""

        if target_node is self.root_node:
            self.game.revert(target_node.step_nbr)
            return

        # todo: should be able to handle game being in any if all_action_nodes
        assert self.root_node.step_nbr == self.game.get_step()
        assert target_node in self.all_action_nodes

        step_list = []
        node = target_node
        while node is not self.root_node:
            step_list.append(node.change_log)
            node = node.parent

        for steps in reversed(step_list):
            self.game.forward(steps)

        assert target_node.step_nbr == self.game.get_step()

    def expand_action_node(self, node: ActionNode, action: botbowl.Action) -> None:
        assert action not in node.explored_actions
        assert node in self.all_action_nodes
        self.set_game_to_node(node)
        new_node = expand_action(self.game, action, node)
        node.connect_child(new_node, action)
        self.set_game_to_node(self.root_node)

        # find all newly added action nodes
        self._look_for_action_nodes(new_node)

    def _look_for_action_nodes(self, node: Node):
        if isinstance(node, ActionNode):
            assert node not in self.all_action_nodes
            self.all_action_nodes.append(node)
        for child_node in node.children:
            self._look_for_action_nodes(child_node)


def expand_action(game: botbowl.Game, action: botbowl.Action, parent: ActionNode) -> Node:
    """
    :param game: game object used for calculations. Will be reverted to original state.
    :param action: action to be evaluated.
    :param parent: parent node
    :returns - list of tuples containing (Steps, probability) for each possible outcome.
             - probabilities sums to 1.0
    Not called recursively
    """
    # noinspection PyProtectedMember
    assert game._is_action_allowed(action)
    assert game.trajectory.enabled
    game.config.fast_mode = False

    with remove_randomness(game):
        game.step(action)

    return expand_none_action(game, parent)


def expand_none_action(game: botbowl.Game, parent: Node, moving_handled=False, pickup_handled=False) -> Node:
    """
    :param game: the game state is changed during expansion but restored to state of argument 'parent'
    :param parent: shall represent the current state of argument 'game'. game state is restored to parent.step_nbr
    :param moving_handled:
    :param pickup_handled:
    :returns: A subclass of Node:
                - ChanceNode in a nestled structure with multiple ActionNode as leaf nodes.
                - ActionNode if only one possible outcome.
        param game is changed but restored to initial state af
    Called recursively.
    """

    return_node = None
    while len(game.state.available_actions) == 0:
        proc = game.get_procedure()
        proc_type = type(proc)

        if proc_type in {procedures.Dodge, procedures.GFI} and not moving_handled:
            return_node = expand_moving(game, parent)
        elif proc_type is procedures.Pickup and not pickup_handled:
            return_node = expand_pickup(game, parent)
        elif proc_type in proc_to_function:
            return_node = proc_to_function[proc_type](game, parent)

        if return_node is not None:
            game.revert(parent.step_nbr)
            return return_node

        with remove_randomness(game):
            game.step()

    action_node = ActionNode(game, parent)
    game.revert(parent.step_nbr)
    assert parent.step_nbr == game.get_step()
    return action_node


# def expand_template(game: botbowl.Game, parent: Node) -> Node:
#    raise NotImplementedError()


def expand_bounce(game: botbowl.Game, parent: Node) -> Node:
    # noinspection PyTypeChecker
    active_proc: procedures.Bounce = game.get_procedure()
    assert type(active_proc) is procedures.Bounce

    new_parent = ChanceNode(game, parent)
    debug_step_count = game.get_step()

    ball_pos = active_proc.piece.position

    active_player = game.get_active_player()
    adj_squares = game.get_adjacent_squares(ball_pos, occupied=False)
    sq_to_num_tz = {sq: game.num_tackle_zones_at(active_player, sq) for sq in adj_squares}
    num_tz_to_sq = {}

    for sq, num_tz in sq_to_num_tz.items():
        num_tz_to_sq.setdefault(num_tz, []).append(sq)

    for num_tz, count in Counter(sq_to_num_tz.values()).items():
        possible_squares = num_tz_to_sq[num_tz]
        square = np.random.choice(possible_squares, 1)[0]
        assert type(square) == botbowl.Square

        delta_x = square.x - ball_pos.x
        delta_y = square.y - ball_pos.y

        roll = botbowl.D8.d8_from_xy[(delta_x, delta_y)]
        with only_fixed_rolls(game, d8=[roll]):
            game.step()
        new_node = expand_none_action(game, new_parent)
        new_parent.connect_child(new_node, prob=count / 8)
        assert debug_step_count == game.get_step() == new_parent.step_nbr

    assert debug_step_count == game.get_step() == new_parent.step_nbr
    return new_parent


def expand_handoff(game: botbowl.Game, parent: Node) -> Node:
    raise NotImplementedError()


def expand_pickup(game: botbowl.Game, parent: Node) -> Node:
    # noinspection PyTypeChecker
    active_proc: procedures.Pickup = game.get_procedure()
    assert type(active_proc) is procedures.Pickup
    probability_success = game.get_pickup_prob(active_proc.player, active_proc.ball.position)

    new_parent = ChanceNode(game, parent)
    debug_step_count = game.get_step()

    # SUCCESS SCENARIO
    with only_fixed_rolls(game, d6=[6]):
        game.step()
    success_node = expand_none_action(game, new_parent, pickup_handled=True)
    new_parent.connect_child(success_node, probability_success)

    assert debug_step_count == game.get_step() == new_parent.step_nbr

    # FAILURE SCENARIO
    with only_fixed_rolls(game, d6=[1]):
        game.step()
    fail_node = expand_none_action(game, new_parent, pickup_handled=True)
    new_parent.connect_child(fail_node, 1 - probability_success)

    assert debug_step_count == game.get_step() == new_parent.step_nbr

    return new_parent


def expand_moving(game: botbowl.Game, parent: Node) -> Node:
    # noinspection PyTypeChecker
    active_proc: Union[procedures.GFI, procedures.Dodge] = game.get_procedure()
    assert type(active_proc) is procedures.Dodge or type(active_proc) is procedures.GFI

    procs = [proc for proc in reversed(game.state.stack.items) if isinstance(proc, procedures.MoveAction)]
    assert len(procs) == 1
    move_action_proc: procedures.MoveAction = procs[0]
    if move_action_proc.steps is not None:
        final_step = move_action_proc.steps[-1]
    else:
        final_step = active_proc.position

    path = move_action_proc.paths[final_step]
    probability_success = path.prob  # TODO: pickup should not be considered by this probability.
    rolls = list(collapse(path.rolls))

    if game.get_ball().position == final_step:
        rolls.pop()

    p = np.array(rolls) / sum(rolls)
    index_of_failure = np.random.choice(range(len(rolls)), 1, p=p)[0]

    # fix all rolls up until the failure, and step them
    for _ in range(index_of_failure):
        botbowl.D6.fix(6)
    while len(botbowl.D6.FixedRolls) > 0:
        game.step()

    new_parent = ChanceNode(game, parent)
    debug_step_count = game.get_step()

    # SUCCESS SCENARIO
    for _ in range(len(rolls) - index_of_failure):
        botbowl.D6.fix(6)
    success_node = expand_none_action(game, new_parent, moving_handled=True)
    new_parent.connect_child(success_node, probability_success)

    assert debug_step_count == game.get_step()

    # FAILURE SCENARIO
    botbowl.D6.fix(1)
    fail_node = expand_none_action(game, new_parent, moving_handled=True)
    new_parent.connect_child(fail_node, 1 - probability_success)

    assert debug_step_count == game.get_step()

    return new_parent


def expand_armor(game: botbowl.Game, parent: Node) -> Node:
    # noinspection PyTypeChecker
    proc: procedures.Armor = game.get_procedure()
    assert not proc.skip_armor
    botbowl.D6.fix(1)
    botbowl.D6.fix(1)
    while proc in game.state.stack.items:
        game.step()
    return expand_none_action(game, parent)


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
    new_chance_node = ChanceNode(game, parent)
    game.revert(root_step)
    parent.connect_child(new_chance_node, prob=prob)
    assert len(botbowl.D6.FixedRolls) == 0


def expand_block(game: botbowl.Game, parent: Node) -> Node:
    raise NotImplementedError()


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
        p_armorbreak = accumulated_prob_2d_roll[proc.player.get_av() + 1]
        p_injury_removal = accumulated_prob_2d_roll[8]  # KO

        _fix_step_connect(d6_fixes=[1, 2], prob=1.0 - p_armorbreak)  # No armorbreak
        _fix_step_connect(d6_fixes=[6, 5, 1, 2], prob=p_armorbreak * (1.0 - p_injury_removal))  # Stunned
        _fix_step_connect(d6_fixes=[6, 5, 4, 5], prob=p_armorbreak * p_injury_removal)  # KO


# move to other file
def explore(self, max_time: int = None) -> None:
    """Builds the search tree for 'max_time' seconds."""

    assert self.root_node.step_nbr == self.game.get_step()

    start_time = time.time()
    while time.time() - start_time < max_time:
        node = self.get_next_action_node_to_explore()
        self.set_game_to_node(node)
        # action = node.action_sampler.get_action()


# move to other file
def explore_condition(self, node: ActionNode, prob) -> bool:
    if self.root_node.team == node.team and self.root_node.turn != node.turn:
        return False
    elif prob < 0.05:
        return False
    else:
        return True


proc_to_function: Dict[Any, Callable[[botbowl.Game, Node], Node]] = \
    {procedures.Block: expand_block,
     procedures.Armor: expand_armor,
     # procedures.Pickup: expand_pickup,
     procedures.Bounce: expand_bounce,
     # procedures.Catch: expand_template,
     # procedures.Intercept: expand_template,
     # procedures.Foul: expand_template,
     # procedures.KickoffTable: expand_template,
     # procedures.PassAttempt: expand_template,
     # procedures.Scatter: expand_template,
     # procedures.ThrowIn: expand_template,
     procedures.Handoff: expand_handoff}
