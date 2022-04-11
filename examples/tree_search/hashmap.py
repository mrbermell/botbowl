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
    assert len(game.state.available_actions) > 0 or game.state.game_over, f"len(aa)=0 when game is not over!"

    s = ""
    s += "h" if game.active_team is game.state.home_team else "a"
    s += str(game.state.round)
    s += str(game.state.half)
    s += str(game.state.home_team.state.score)
    s += str(game.state.away_team.state.score)
    s += f"{len(game.state.reports):4.0f}"
    if not game.state.game_over:
        proc = game.get_procedure()
        s += f" {type(proc).__name__} "
        if isinstance(proc, botbowl.core.procedure.Setup):
            s += str(1*proc.reorganize)
        elif isinstance(proc, botbowl.core.procedure.Reroll):
            s += f"({type(proc.context).__name__})"
    else:
        s += "GAME_OVER"

    ball_pos = game.get_ball_position()
    s += f"ball={create_position_hash(ball_pos)} " if ball_pos is not None else " "

    for ac in game.get_available_actions():
        s += f" {ac.action_type.name}{len(ac.positions)},{hash(tuple(ac.positions))}"

    s += "".join(create_playerstate_hash(game, p) for p in game.get_players_on_pitch())

    return s
