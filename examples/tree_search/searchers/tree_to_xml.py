from contextlib import contextmanager

from examples.tree_search import SearchTree, HeuristicVector
from examples.tree_search.searchers.mcts_ucb import mcts_ucb_rollout


@contextmanager
def xml_entry(name: str, **attributes):
    s = f"<{name}"
    for key, val in attributes.items():
        _val = f'"{val}"' if type(val) is str else f'{val}'
        s += f' {key}="{val}"'
    s+= '>'

    try:
        yield s
    finally:
        s += f'</{name}>'


def create_xml(tree: SearchTree) -> str:
     pass


def main():
    from examples.tree_search.Samplers import MockPolicy
    from tests.util import get_custom_game_turn

    game, _ = get_custom_game_turn(player_positions=[(6, 6), (7, 7)],
                                   opp_player_positions=[(5, 6)],
                                   ball_position=(6, 6),
                                   pathfinding_enabled=True)

    weights = HeuristicVector(score=1, ball_marked=0.1, ball_carried=0.2, ball_position=0.01, tv_on_pitch=1)

    tree = SearchTree(game)
    policy = MockPolicy()

    for _ in range(300):
        mcts_ucb_rollout(tree, policy, weights, exploration_coeff=1)

    s = create_xml(tree)

    print(s)


def main2():
    root = None
    with xml_entry("Root", value=1337) as f:
        root = f
        f += "vad vill du mig\n"
    print(root)


if __name__ == "__main__":
    main2()
