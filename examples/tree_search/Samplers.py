import numpy as np
from abc import ABC, abstractmethod
from random import shuffle
from typing import Optional, List, Iterable, Tuple, Callable, Dict
from more_itertools import collapse

import botbowl
from botbowl import ActionType, Action, EnvConf, Setup
from tests.util import get_game_turn
from examples.tree_search import ActionNode

NodeAction = botbowl.Action


class TreeExplorationStrategy(ABC):
    @abstractmethod
    def get_node_and_action(self, TreeSearcher) -> Tuple['ActionNode', Iterable[Action]]:
        pass


def get_filtered_available_actions(game: botbowl.Game):
    allowed_avail_actions = []
    disallowed = {ActionType.START_FOUL, ActionType.START_PASS, ActionType.START_HANDOFF}

    for action_choice in game.get_available_actions():
        if action_choice.action_type in disallowed:
            continue
        allowed_avail_actions.append(action_choice)
    return allowed_avail_actions


def get_priority_move_square(action_choice, game) -> Optional[botbowl.Square]:
    player = game.get_active_player()
    ball_pos = game.get_ball().position
    paths = action_choice.paths
    priority_square = None


    # Score or move towards endzone
    if player.position == ball_pos:
        endzone_x = game.get_opp_endzone_x(player.team)
        endzone_paths = [path for path in paths if path.get_last_step().x == endzone_x]
        if len(endzone_paths) > 0:
            path = max(endzone_paths, key=lambda p: p.prob)
            priority_square = path.get_last_step()
        else:
            max_x_path = max(paths, key=lambda p: p.prob * (30 - abs(p.get_last_step().x - endzone_x)))
            priority_square = max_x_path.get_last_step()

    # Pickup ball
    elif ball_pos in action_choice.positions:
        priority_square = ball_pos

    return priority_square


def convert_to_actions(action_choice: botbowl.ActionChoice) -> Iterable[Action]:
    if len(action_choice.positions) > 0:
        return (Action(action_choice.action_type, position=sq) for sq in action_choice.positions)
    elif len(action_choice.players) > 0:
        return (Action(action_choice.action_type, player=p) for p in action_choice.players)
    else:
        return (Action(action_choice.action_type) for _ in range(1))


class ActionSampler:
    actions: List[NodeAction]
    priority_actions: List[NodeAction]

    def __init__(self, game: botbowl.Game):
        self.actions = []
        self.priority_actions = []

        for action_choice in game.get_available_actions():
            if action_choice.action_type not in {ActionType.START_MOVE,
                                                 ActionType.MOVE,
                                                 ActionType.END_PLAYER_TURN}:
                continue

            if action_choice.action_type == ActionType.MOVE:
                prio_move_square = get_priority_move_square(action_choice, game)

                if prio_move_square is not None:
                    self.priority_actions.append(Action(action_choice.action_type, position=prio_move_square))

            self.actions.extend(convert_to_actions(action_choice))

        if len(self.actions) == 0:
            self.actions.extend(collapse(convert_to_actions(action_choice)
                                         for action_choice in game.get_available_actions()))

        assert len(self.actions) > 0
        shuffle(self.actions)

    def get_action(self) -> Optional[Action]:
        if len(self.priority_actions) > 0:
            return self.priority_actions.pop()
        elif len(self.actions):
            return self.actions.pop()
        else:
            return None

    def __len__(self):
        return len(self.actions)


use_directly_action_types = [ActionType.START_GAME,
                             ActionType.TAILS,
                             ActionType.RECEIVE,
                             ]


def scripted_action(game: botbowl.Game) -> Optional[Action]:
    available_action_types = {action_choice.action_type for action_choice in game.get_available_actions()}

    for at in use_directly_action_types:
        if at in available_action_types:
            return Action(at)

    if ActionType.PLACE_BALL in available_action_types:
        x = game.arena.width // 4
        if game.active_team is game.state.away_team:
            x *= 3
        y = game.arena.height // 2
        return Action(ActionType.PLACE_BALL, position=botbowl.Square(x, y))

    proc = game.get_procedure()
    if type(proc) is Setup:
        if game.is_setup_legal(game.active_team):
            return Action(ActionType.END_SETUP)
        for at in available_action_types:
            if at not in {ActionType.END_SETUP, ActionType.PLACE_PLAYER}:
                return Action(at)

    return None


class MockPolicy:
    ActionProbList = List[Tuple[Action, float]]

    def __init__(self):
        self.env_conf = EnvConf()
        self.positional_types = set(self.env_conf.positional_action_types)
        self.simple_types = set(self.env_conf.simple_action_types)
        self.simple_types.remove(ActionType.END_PLAYER_TURN)


        self.convert_function: Dict[ActionType, Callable[[botbowl.Game], MockPolicy.ActionProbList]] = {
                                 ActionType.MOVE: self.move_actions,
                                 ActionType.START_HANDOFF: self.start_handoff_actions,
                                 #ActionType.START_BLITZ: self.move_actions,
                                 #ActionType.START_BLITZ: self.move_actions,
                                 #ActionType.START_BLITZ: self.move_actions,
        }

        self.end_setup = False

        self.player_actiontypes_without_move_actions = {botbowl.core.table.PlayerActionType.HANDOFF,
                                                        botbowl.core.table.PlayerActionType.BLITZ}

    def start_handoff_actions(self, game: botbowl.Game, action_choice) -> ActionProbList:
        ball = game.get_ball()
        if ball is not None and ball.is_carried:
            if ball.position in action_choice.positions:
                return [(Action(ActionType.START_HANDOFF, position=[ball.position]), 1)]
        return []

    def move_actions(self, game: botbowl.Game, action_choice) -> ActionProbList:
        if game.get_player_action_type() in self.player_actiontypes_without_move_actions:
            return []

        action_probs = []

        player = game.get_active_player()
        ball_carrier = game.get_ball_carrier()
        is_ball_carrier = player is game.get_ball_carrier()

        if player.state.moves > 0 and not is_ball_carrier:
            return []
        elif is_ball_carrier and player.state.moves > 0:
            action_probs.append((Action(ActionType.END_PLAYER_TURN), 1))

        ball = game.get_ball()

        ball_on_floor_pos = game.get_ball().position if not ball.is_carried else None
        opp_ball_carrier = ball_carrier if ball_carrier is not None and ball_carrier.team is not game.active_team else None

        is_home = player.team is game.state.home_team

        for pos in action_choice.positions:
            prob = 1
            if ball_on_floor_pos is not None:
                if pos == ball_on_floor_pos:
                    prob = 3
                elif pos.distance(ball_on_floor_pos) == 1:
                    prob = 2
            elif opp_ball_carrier is not None and pos.distance(opp_ball_carrier.position):
                prob = 2

            elif is_ball_carrier and game.arena.is_in_opp_endzone(pos, is_home):
                prob = 2

            action = Action(ActionType.MOVE, position=pos)
            action_probs.append((action, prob))

        return action_probs

    def __call__(self, game: botbowl.Game) -> Tuple[float, np.ndarray, List[botbowl.Action]]:
        if game.state.game_over:
            return 0.0, np.array([1.0]), [None]

        action = scripted_action(game)
        if action is not None:
            return 0.0, np.array([1.0]), [action]

        actions: MockPolicy.ActionProbList = []

        if self.end_setup:
            self.end_setup = False
            return 0.0, np.array([1.0]), [Action(ActionType.END_SETUP)]

        for action_choice in game.get_available_actions():
            action_type = action_choice.action_type

            if action_type in self.convert_function:
                actions.extend(self.convert_function[action_type](game, action_choice))

            elif action_type in self.positional_types:
                if len(action_choice.positions)>0:
                    positions = action_choice.positions
                else:
                    positions = [p.position for p in action_choice.players]

                for pos in positions:
                    actions.append((Action(action_type, position=pos), 1))

            elif action_choice.action_type in self.simple_types:
                actions.append((Action(action_type), 1))

            elif type(game.get_procedure()) is Setup and action_type not in {ActionType.END_SETUP, ActionType.PLACE_PLAYER}:
                actions.append((Action(action_type), 1))
                self.end_setup = True

        if len(actions) == 0 and ActionType.END_PLAYER_TURN in [ac.action_type for ac in game.state.available_actions]:
            actions.append((Action(ActionType.END_PLAYER_TURN), 1))

        action_objects, probabilities = zip(*actions)
        probabilities = np.array(probabilities, dtype=np.float)
        probabilities += 0.0001*np.random.random(len(probabilities))
        probabilities /= sum(probabilities)

        return 0.0, probabilities, action_objects


def main():
    policy = MockPolicy()

    game = get_game_turn()
    game.config.pathfinding_enabled = True
    game.set_available_actions()

    while not game.state.game_over:
        _, _, actions = policy(game)
        action = np.random.choice(actions, 1)[0]
        game.step(action)
        print(action)

if __name__ == "__main__":
    main()