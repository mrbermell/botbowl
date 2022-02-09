from abc import ABC
from collections import Counter
from copy import deepcopy
from functools import partial
from typing import Optional, Callable, List, Union, Iterable

import more_itertools.more
import numpy as np
from more_itertools import collapse
from pytest import approx

import botbowl
import botbowl.core.forward_model as forward_model
import botbowl.core.procedure as procedures
from botbowl import Skill, BBDieResult
from tests.util import only_fixed_rolls
from .hashmap import HashMap

accumulated_prob_2d_roll = np.array([36, 36, 36, 35, 33, 30, 26, 21, 15, 10, 6, 3, 1]) / 36

remove_randomness = partial(only_fixed_rolls, assert_fixes_consumed=False, assert_no_prev_fixes=False)


class Node(ABC):
    parent: Optional['Node']
    children: List['Node']
    change_log: List[forward_model.Step]
    step_nbr: int  # forward model's step count
    top_proc: str

    def __init__(self, game: botbowl.Game, parent: Optional['Node']):
        self.step_nbr = game.get_step()
        self.parent = parent
        self.children = []
        self.change_log = game.trajectory.action_log[parent.step_nbr:] if parent is not None else []
        self.top_proc = str(game.get_procedure())
        assert parent is None or len(self.change_log) > 0

    def _connect_child(self, child_node: 'Node'):
        assert child_node.parent is self
        self.children.append(child_node)

    def __repr__(self):
        self_type = str(type(self)).split(".")[-1]
        return f"{self_type}({self.step_nbr=}, {self.top_proc}"


class ActionNode(Node):
    reward: float
    value: float
    team: botbowl.Team
    explored_actions: List[botbowl.Action]
    turn: int
    info: dict  # Only purpose is to store information for users of SearchTree
    simple_hash: str

    def __init__(self, game: botbowl.Game, parent: Optional[Node], reward=0.0):
        super().__init__(game, parent)
        self.reward = reward
        self.team = game.state.available_actions[0].team
        self.explored_actions = []
        self.turn = self.team.state.turn
        self.info = {}
        self.simple_hash = ActionNode.hash_game_state(game)

        if parent is not None:
            node: ActionNode = parent
            while type(node) != type(self):
                node = node.parent

    @property
    def depth(self):
        return len(list(filter(lambda n: type(n) is ActionNode, self.get_all_parents(include_self=False))))

    def connect_child(self, child_node: Node, action: botbowl.Action):
        super()._connect_child(child_node)
        self.explored_actions.append(action)

    def get_child_action(self, child: Node) -> botbowl.Action:
        assert child in self.children
        return self.explored_actions[self.children.index(child)]

    def make_root(self):
        self.parent = None
        self.change_log.clear()

    def get_all_parents(self, include_self) -> Iterable[Node]:
        if include_self:
            yield self

        node = self.parent
        while node is not None:
            yield node
            node = node.parent
        return

    def get_children_from_action(self, action: botbowl.Action) -> Iterable['ActionNode']:
        if action not in self.explored_actions:
            return
        child = self.children[self.explored_actions.index(action)]
        return get_action_node_children(child)

    def __hash__(self):
        return hash(self.simple_hash)

    def __eq__(self, other):
        raise NotImplementedError()

    @staticmethod
    def hash_game_state(game: botbowl.Game) -> str:
        s = ""
        s += "h" if game.active_team is game.state.home_team else "a"
        s += str(game.state.round)
        s += str(game.active_team.state.turn)
        s += type(game.get_procedure()).__name__
        s += f"{hash(game.get_ball_position())}-"
        s += " ".join(str(hash(p.position)) for p in game.get_players_on_pitch())+"-"
        return s


def get_action_node_children(node: Node) -> Iterable[ActionNode]:
    if isinstance(node, ActionNode):
        yield node
        return
    elif isinstance(node, ChanceNode):
        return more_itertools.collapse(map(get_action_node_children, node.children))
    else:
        raise ValueError()

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

    def get_child_prob(self, child_node: Node) -> float:
        assert child_node in self.children
        return self.child_probability[self.children.index(child_node)]

class SearchTree:
    game: botbowl.Game
    root_node: ActionNode
    all_action_nodes: HashMap
    current_node: ActionNode
    on_every_action_node: Callable[['SearchTree', ActionNode], None]

    def __init__(self, game, on_every_action_node=None):
        self.game = deepcopy(game)

        self.game.home_agent.human = True
        self.game.away_agent.human = True
        if not self.game.trajectory.enabled:
            self.game.enable_forward_model()

        self.root_node = ActionNode(game, None)
        self.all_action_nodes = HashMap([self.root_node])
        self.current_node = self.root_node
        self.on_every_action_node = on_every_action_node

        if self.on_every_action_node is not None:
            self.on_every_action_node(self, self.root_node)

    def set_new_root(self, game: botbowl.Game) -> None:
        if self.game is game:
            raise ValueError("Can't search the tree for its own game object.")

        target_node = ActionNode(game, None)
        found_node = None

        # compare with all nodes that have the same hash
        for node in self.all_action_nodes[target_node]:
            self.set_game_to_node(node)
            diff = self.game.state.compare(game.state)
            if len(diff) == 0:
                found_node = node
                break

        if found_node is None:
            self.__init__(deepcopy(game), self.on_every_action_node)
        else:
            self.root_node = found_node
            self.root_node.make_root()
            self.set_game_to_node(self.root_node)
            self.all_action_nodes = HashMap()
            self._look_for_action_nodes(self.root_node)  # add all children to the 'self.all_action_nodes'

    def set_game_to_node(self, target_node: ActionNode) -> None:
        """Uses forward model to set self.game to the state of Node"""
        assert self.current_node.step_nbr == self.game.get_step(), \
            f"gamestate {self.game.get_step()} and SearchTree {self.current_node.step_nbr} are not synced, big fault!"

        assert target_node in self.all_action_nodes, "target node is not in SearchTree, major fault"

        if target_node is self.current_node:
            return

        # Always revert to root node to begin with. Could be improved...
        self.game.revert(self.root_node.step_nbr)
        self.current_node = self.root_node
        assert self.root_node.step_nbr == self.game.get_step()

        if target_node is self.root_node:
            return

        for node in reversed( list(target_node.get_all_parents(include_self=True))):
            self.game.forward(node.change_log)

        self.current_node = target_node

        assert target_node.step_nbr == self.game.get_step(), f"{target_node.step_nbr} != {self.game.get_step()}"

    def expand_action_node(self, node: ActionNode, action: botbowl.Action) -> List[ActionNode]:
        assert action not in node.explored_actions, f"{action} has already been explored in this node"
        assert node in self.all_action_nodes, f"{node} is not in all_action_nodes"
        self.set_game_to_node(node)
        new_node = expand_action(self.game, action, node)
        node.connect_child(new_node, action)
        self.set_game_to_node(self.root_node)

        # find all newly added action nodes
        return self._look_for_action_nodes(new_node)

    def _look_for_action_nodes(self, node: Node) -> List[ActionNode]:
        new_action_nodes = []
        if isinstance(node, ActionNode):
            assert node not in self.all_action_nodes
            new_action_nodes.append(node)
            self.all_action_nodes.add(node)
            if self.on_every_action_node is not None:
                self.on_every_action_node(self, node)

        for child_node in node.children:
            new_action_nodes.extend(self._look_for_action_nodes(child_node))
        return new_action_nodes


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


def get_expanding_function(proc, moving_handled, pickup_handled) -> Callable[[botbowl.Game, Node], Node]:
    proc_type = type(proc)
    if proc_type in {procedures.Dodge, procedures.GFI} and not moving_handled:
        return expand_moving
    elif proc_type is procedures.Pickup and not pickup_handled:
        return expand_pickup
    elif proc_type is procedures.Block and proc.roll is None:
        return expand_block
    elif proc_type is procedures.Armor:
        return expand_armor
    elif proc_type is procedures.Injury:
        return expand_injury
    elif proc_type is procedures.Bounce:
        return expand_bounce
    elif proc_type is procedures.Catch:
        return expand_catch
    elif proc_type is procedures.ThrowIn:
        return expand_throw_in
    else:
        return None

    # saved for later
    # procedures.Foul
    # procedures.KickoffTable
    # procedures.PassAttempt
    # procedures.Intercept
    # procedures.Scatter


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

    while len(game.state.available_actions) == 0:
        proc = game.get_procedure()
        expand_func = get_expanding_function(proc, moving_handled, pickup_handled)

        if expand_func is not None:
            return_node = expand_func(game, parent)
            game.revert(parent.step_nbr)
            return return_node

        with remove_randomness(game):
            game.step()

    action_node = ActionNode(game, parent)
    game.revert(parent.step_nbr)
    assert parent.step_nbr == game.get_step()
    return action_node


def expand_throw_in(game: botbowl.Game, parent: Node) -> Node:
    # noinspection PyTypeChecker
    active_proc: procedures.ThrowIn = game.get_procedure()
    assert type(active_proc) is procedures.ThrowIn

    d6_fixes = [3, 4] if game.config.arena.height > 7 else [1, 2]

    with only_fixed_rolls(game, d3=[2], d6=d6_fixes):
        game.step()

    assert active_proc is not game.get_procedure()

    return expand_none_action(game, parent)


def expand_bounce(game: botbowl.Game, parent: Node) -> Node:
    # noinspection PyTypeChecker
    active_proc: procedures.Bounce = game.get_procedure()
    assert type(active_proc) is procedures.Bounce

    new_parent = ChanceNode(game, parent)

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
        assert game.get_step() == new_parent.step_nbr

    sum_prob = sum(new_parent.child_probability)
    new_parent.child_probability = [prob/sum_prob for prob in new_parent.child_probability]

    assert sum(new_parent.child_probability) == approx(1.0, abs=1e-9)
    assert game.get_step() == new_parent.step_nbr
    return new_parent


def expand_pickup(game: botbowl.Game, parent: Node) -> Node:
    # noinspection PyTypeChecker
    active_proc: procedures.Pickup = game.get_procedure()
    assert type(active_proc) is procedures.Pickup
    probability_success = game.get_pickup_prob(active_proc.player, active_proc.ball.position)

    new_parent = ChanceNode(game, parent)

    # SUCCESS SCENARIO
    with only_fixed_rolls(game, d6=[6]):
        game.step()
    success_node = expand_none_action(game, new_parent, pickup_handled=True)
    new_parent.connect_child(success_node, probability_success)

    assert game.get_step() == new_parent.step_nbr

    # FAILURE SCENARIO
    with only_fixed_rolls(game, d6=[1]):
        game.step()
    fail_node = expand_none_action(game, new_parent, pickup_handled=True)
    new_parent.connect_child(fail_node, 1 - probability_success)

    assert game.get_step() == new_parent.step_nbr

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
    probability_success = path.prob
    rolls = list(collapse(path.rolls))

    if game.get_ball().position == final_step:
        # remove the pickup roll and probability
        rolls.pop()
        probability_success /= game.get_pickup_prob(active_proc.player, final_step)

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
    assert not proc.foul

    p_armorbreak = accumulated_prob_2d_roll[proc.player.get_av() + 1]
    new_parent = ChanceNode(game, parent)
    expand_with_fixes(game, new_parent, p_armorbreak, d6=[6, 6])  # Armor broken
    expand_with_fixes(game, new_parent, 1 - p_armorbreak, d6=[1, 1])  # Armor not broken
    return new_parent


def expand_injury(game: botbowl.Game, parent: Node) -> Node:
    # noinspection PyTypeChecker
    proc: procedures.Injury = game.get_procedure()
    assert not proc.foul

    if proc.in_crowd:
        with only_fixed_rolls(game, d6=[5, 4]):  # straight to KO
            game.step()
        return expand_none_action(game, parent)

    p_removal = accumulated_prob_2d_roll[8]
    new_parent = ChanceNode(game, parent)
    expand_with_fixes(game, new_parent, p_removal, d6=[5, 4])  # KO
    expand_with_fixes(game, new_parent, 1 - p_removal, d6=[1, 1])  # Stun
    return new_parent


def expand_block(game: botbowl.Game, parent: Node) -> Node:

    proc: botbowl.Block = game.get_procedure()
    assert type(proc) is botbowl.Block
    assert not proc.gfi, "Can't handle GFI:s here =( "
    assert proc.roll is None

    attacker: botbowl.Player = proc.attacker
    defender: botbowl.Player = proc.defender
    dice = game.num_block_dice(attacker, defender)
    num_dice = abs(dice)

    # initialize as 1d block without skills
    dice_outcomes = np.array([2, 2, 1, 1], dtype=int)
    DEF_DOWN, NOONE_DOWN, ALL_DOWN, ATT_DOWN = (0, 1, 2, 3)

    die_results = ([BBDieResult.DEFENDER_DOWN, BBDieResult.DEFENDER_STUMBLES],
                   [BBDieResult.PUSH],
                   [BBDieResult.BOTH_DOWN],
                   [BBDieResult.ATTACKER_DOWN])

    who_has_block = (attacker.has_skill(Skill.BLOCK), defender.has_skill(Skill.BLOCK))

    if any(who_has_block):
        dice_outcomes[ALL_DOWN] = 0
        die_results[ALL_DOWN].clear()

        if who_has_block == (True, True):  # both
            dice_outcomes[NOONE_DOWN] += 1
            die_results[NOONE_DOWN].append(BBDieResult.BOTH_DOWN)
        elif who_has_block == (True, False):  # only attacker
            dice_outcomes[DEF_DOWN] += 1
            die_results[DEF_DOWN].append(BBDieResult.BOTH_DOWN)
        elif who_has_block == (False, True):  # only defender
            dice_outcomes[ATT_DOWN] += 1
            die_results[ATT_DOWN].append(BBDieResult.BOTH_DOWN)

    crowd_surf: bool = game.get_push_squares(attacker.position, defender.position)[0].out_of_bounds

    if crowd_surf:
        dice_outcomes[DEF_DOWN] += 2
        dice_outcomes[NOONE_DOWN] -= 2
        die_results[DEF_DOWN].append(BBDieResult.PUSH)
        die_results[NOONE_DOWN].remove(BBDieResult.PUSH)
    elif defender.has_skill(Skill.DODGE):  # and not attacker.has_skill(Skill.TACKLE):
        dice_outcomes[DEF_DOWN] -= 1
        dice_outcomes[NOONE_DOWN] += 1
        die_results[DEF_DOWN].remove(BBDieResult.DEFENDER_STUMBLES)
        die_results[NOONE_DOWN].append(BBDieResult.DEFENDER_STUMBLES)

    prob = np.zeros(4)
    probability_left = 1.0
    available_dice = 6
    evaluation_order = [DEF_DOWN, NOONE_DOWN, ALL_DOWN, ATT_DOWN]
    if dice < 0:
        evaluation_order = reversed(evaluation_order)

    for i in evaluation_order:
        prob[i] = probability_left * (1 - (1 - dice_outcomes[i] / available_dice) ** num_dice)
        available_dice -= dice_outcomes[i]
        probability_left -= prob[i]

    assert available_dice == 0 and probability_left == approx(0) and prob.sum() == approx(1)

    new_parent = ChanceNode(game, parent)

    for prob, die_res in zip(prob, die_results):
        if prob == approx(0) or len(die_res) == 0:
            assert prob == approx(0) and len(die_res) == 0
            continue

        expand_with_fixes(game, new_parent, prob,
                          block_dice=np.random.choice(die_res, num_dice))

    assert sum(new_parent.child_probability) == approx(1.0)
    return new_parent


def expand_catch(game: botbowl.Game, parent: Node) -> Node:
    # noinspection PyTypeChecker
    proc: procedures.Catch = game.get_procedure()
    assert type(proc) is procedures.Catch

    p_catch = game.get_catch_prob(proc.player, accurate=proc.accurate, handoff=proc.handoff)

    new_parent = ChanceNode(game, parent)

    # Success scenario
    expand_with_fixes(game, new_parent, p_catch, d6=[6])

    # Failure scenario
    if proc.player.has_skill(Skill.CATCH):
        expand_with_fixes(game, new_parent, 1 - p_catch, d6=[1, 1])
    else:
        expand_with_fixes(game, new_parent, 1 - p_catch, d6=[1])

    return new_parent


def expand_with_fixes(game, parent, probability, **fixes):
    with only_fixed_rolls(game, **fixes):
        game.step()
    new_node = expand_none_action(game, parent)
    parent.connect_child(new_node, probability)
