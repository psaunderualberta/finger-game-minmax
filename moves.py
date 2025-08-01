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
    if move.move_type == ATTACK:
        return "A{}{}".format(

        )
    elif move.move_type == SWAP:
        return "S{}{}{}".format(

        )
    else:
        raise Exception(f"Move type {move.move_type} not known.")
    
