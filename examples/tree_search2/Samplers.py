from abc import ABC, abstractmethod
from random import shuffle
from typing import Optional, List, Iterable, Tuple

import numpy as np
from more_itertools import collapse

import botbowl
from botbowl import ActionType, Action

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


