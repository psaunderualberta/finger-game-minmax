from numba import int64, njit
from numba.experimental import jitclass

# Move Types
ATTACK = 0
SWAP = 1

# Hands
LEFT_HAND = 0
RIGHT_HAND = 1

# Amounts
ONE = 1
TWO = 2
THREE = 3
FOUR = 4
FIVE = 5

spec = [
    ("move_type", int64),  # Type of move (attack or swap)
    ("source_hand", int64),  # Source hand (left or right)
    ("target_hand", int64),  # Target hand (left or right)
    ("amount", int64),  # Amount of cards to move
]


@jitclass(spec)
class Move(object):
    def __init__(self, move_type, source_hand, target_hand, amount):
        self.move_type = move_type
        self.source_hand = source_hand
        self.target_hand = target_hand
        self.amount = amount


@njit
def create_all_moves():
    moves = []

    move_type = ATTACK
    for source_hand in [LEFT_HAND, RIGHT_HAND]:
        for target_hand in [LEFT_HAND, RIGHT_HAND]:
            moves.append(Move(move_type, source_hand, target_hand, 0))

    move_type = SWAP
    for source_hand in [LEFT_HAND, RIGHT_HAND]:
        for target_hand in [LEFT_HAND, RIGHT_HAND]:
            if source_hand != target_hand:
                for amount in [ONE, TWO, THREE, FOUR]:
                    moves.append(Move(move_type, source_hand, target_hand, amount))

    return moves


def move2str(move: Move):
    if move is None:
        return ""
    if move.move_type == ATTACK:
        return "A{}{}".format(
            "L" if move.source_hand == LEFT_HAND else "R",
            "L" if move.target_hand == LEFT_HAND else "R"
        )
    elif move.move_type == SWAP:
        return "S{}{}{}".format(
            move.amount,
            "L" if move.source_hand == LEFT_HAND else "R",
            "L" if move.target_hand == LEFT_HAND else "R"
        )
    else:
        raise Exception(f"Move type {move.move_type} not known.")
    
def str2move(s: str):
    if s is None:
        return None
    s = s.upper()
    str_move_type = s[0]
    assert str_move_type in ["A", "S"]
    if str_move_type == "A":
        move_type = ATTACK
        str_src_hand = s[1]
        str_target_hand = s[2]
        assert str_src_hand in ["L", "R"]
        assert str_target_hand in ["L", "R"]

        count = 0

    elif str_move_type == "S":
        move_type = SWAP
        count = s[1]
        assert count in ["1", "2", "3", "4"]
        str_src_hand = s[2]
        str_target_hand = s[3]
        assert str_src_hand in ["L", "R"]
        assert str_target_hand in ["L", "R"]

        count = int(s[1])

    src_hand = LEFT_HAND if str_src_hand == "L" else RIGHT_HAND
    target_hand = LEFT_HAND if str_target_hand == "L" else RIGHT_HAND
        
    return Move(
        move_type,
        src_hand,
        target_hand,
        count
    )