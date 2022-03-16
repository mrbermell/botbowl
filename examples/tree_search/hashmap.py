from typing import Optional, Iterable, Protocol, List, Dict
import botbowl


class HasSimpleHash(Protocol):
    simple_hash: str


class HashMap:
    data: Dict[str, List[HasSimpleHash]]

    def __init__(self, values: Optional[Iterable[HasSimpleHash]]=None):
        self.data = {}
        if values is not None:
            for item in values:
                self.add(item)

    def __contains__(self, item: HasSimpleHash):
        if item.simple_hash in self.data:
            for possible_match in self.data[item.simple_hash]:
                if item is possible_match:
                    return True
        return False

    def add(self, item: HasSimpleHash):
        self.data.setdefault(item.simple_hash, []).append(item)

    def __getitem__(self, item):
        return self.data.setdefault(item.simple_hash, [])

    def __len__(self):
        return sum(map(len, self.data.values()))

    def __iter__(self):
        for item_list in self.data.values():
            for item in item_list:
                yield item


def create_position_hash(sq: botbowl.Square) -> str:
    return f"{sq.x * 17 + sq.y:3.0f}"


def create_playerstate_hash(game: botbowl.Game, player: botbowl.core.model.Player) -> str:
    bool_attr = ('used', 'stunned', 'has_blocked')

    assert player.position is not None

    s = ""
    s += "h" if player.team is game.state.home_team else "a"
    s += create_position_hash(player.position)
    s += f"{player.state.moves:2.0f}"

    for attr in bool_attr:
        s += f"{1*getattr(player.state, attr)}"

    s += f"{player.role.name}"
    return s


def create_gamestate_hash(game: botbowl.Game) -> str:
    """
    Based the GameState, provides a str that can be used for fast and approximate game state comparisons
    """
    assert len(game.state.available_actions) > 0

    s = ""
    s += "h" if game.active_team is game.state.home_team else "a"
    s += str(game.state.round)
    s += str(game.state.half)
    s += str(game.active_team.state.turn)
    s += str(game.state.home_team.state.score)
    s += str(game.state.away_team.state.score)
    s += type(game.get_procedure()).__name__ if not game.state.game_over else "GAME_OVER"
    ball_pos = game.get_ball_position()
    s += create_position_hash(ball_pos) if ball_pos is not None else " "
    s += "".join(create_playerstate_hash(game, p) for p in game.get_players_on_pitch())
    s += "".join(f"{ac}" for ac in game.get_available_actions())

    return s
