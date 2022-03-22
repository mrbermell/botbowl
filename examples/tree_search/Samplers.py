from typing import Optional, List, Tuple, Callable, Dict

import numpy as np

import botbowl
from botbowl import ActionType, Action, EnvConf, Setup, PlayerActionType
from tests.util import get_game_turn

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
            # ActionType.START_BLITZ: self.move_actions,
            # ActionType.START_BLITZ: self.move_actions,
            # ActionType.START_BLITZ: self.move_actions,
        }

        self.end_setup = False

        self.player_actiontypes_without_move_actions = {botbowl.core.table.PlayerActionType.HANDOFF,
                                                        botbowl.core.table.PlayerActionType.BLITZ}

    def start_handoff_actions(self, game: botbowl.Game, action_choice) -> ActionProbList:
        ball = game.get_ball()
        if ball is not None and ball.position is not None:
            if ball.is_carried and ball.position in action_choice.positions:
                # only allow the ball carrier to do a handoff 
                return [(Action(ActionType.START_HANDOFF, position=ball.position), 1)]
            elif ball.on_ground:
                # ball on ground, allow handoff but force the pickup in move_actions()
                return [(Action(ActionType.START_HANDOFF, position=player.position), 1)
                        for player in action_choice.players]
        return []

    def move_actions(self, game: botbowl.Game, action_choice: botbowl.ActionChoice) -> ActionProbList:
        if game.get_player_action_type() in self.player_actiontypes_without_move_actions:
            return []

        action_probs = []

        player = game.get_active_player()
        ball_carrier = game.get_ball_carrier()
        is_ball_carrier = player is game.get_ball_carrier()
        allow_multiple_movements = is_ball_carrier or player.state.has_blocked

        if player.state.moves > 0:
            if allow_multiple_movements:
                # This is needed because END_PLAYER_TURN is masked if there are other available actions
                action_probs.append((Action(ActionType.END_PLAYER_TURN), 1))
            else:
                # End the movement if you've already made moves unless it was after
                return []

        ball = game.get_ball()
        ball_on_floor_pos = game.get_ball().position if not ball.is_carried else None
        opp_ball_carrier = ball_carrier if ball_carrier is not None and ball_carrier.team is not game.active_team else None

        must_pickup_ball = game.get_player_action_type() is PlayerActionType.HANDOFF and \
                           (not is_ball_carrier) and \
                           ball_on_floor_pos in action_choice.positions
        if must_pickup_ball:
            return [(Action(ActionType.MOVE), 1)]

        is_home = player.team is game.state.home_team

        for pos, path in zip(action_choice.positions, action_choice.paths):
            prob = path.prob
            if ball_on_floor_pos is not None:
                if pos == ball_on_floor_pos:
                    prob += 2
                elif pos.distance(ball_on_floor_pos) == 1:
                    prob += 1
            elif opp_ball_carrier is not None and pos.distance(opp_ball_carrier.position) == 1:
                prob += 1

            elif is_ball_carrier and game.arena.is_in_opp_endzone(pos, is_home):
                prob += 1

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
                if len(action_choice.positions) > 0:
                    positions = action_choice.positions
                else:
                    positions = [p.position for p in action_choice.players]

                for pos in positions:
                    actions.append((Action(action_type, position=pos), 1))

            elif action_choice.action_type in self.simple_types:
                actions.append((Action(action_type), 1))

            elif type(game.get_procedure()) is Setup and action_type not in {ActionType.END_SETUP,
                                                                             ActionType.PLACE_PLAYER}:
                actions.append((Action(action_type), 1))
                self.end_setup = True

        if len(actions) == 0 and ActionType.END_PLAYER_TURN in [ac.action_type for ac in game.state.available_actions]:
            actions.append((Action(ActionType.END_PLAYER_TURN), 1))

        action_objects, probabilities = zip(*actions)
        probabilities = np.array(probabilities, dtype=np.float)
        probabilities += 0.0001 * np.random.random(len(probabilities))
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
