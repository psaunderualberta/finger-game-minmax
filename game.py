import numpy as np
from numba import boolean, double, njit
from numba.experimental import jitclass

from moves import *

spec = [
    ("cache", double[:]),  # Cache for storing game states
]


@jitclass(spec)
class Cache(object):
    def __init__(self, cache):
        self.cache = cache

    def __getitem__(self, key):
        return self.cache[key]

    def __setitem__(self, key, value):
        self.cache[key] = value

    def __contains__(self, key):
        return ~np.isnan(self.cache[key])
    
    def __len__(self):
        return len(self.cache) - np.isnan(self.cache).sum()

    def reset_cache(self):
        self.cache.fill(np.nan)


spec = [
    ("state", int64[:]),  # 2D array for the game state
    ("player_1_turn", boolean),  # Boolean for whose turn it is
    ("seen_states", double[:]),  # Seen states for fast lookup
    ("source_idxs", int64[:]),  # Indices for source hands
    ("target_idxs", int64[:]),  # Indices for target hands
]


@jitclass(spec)
class Game(object):
    def __init__(self, state, player_1_turn, seen_states):
        self.state = state
        self.player_1_turn = player_1_turn
        self.seen_states = seen_states
        self.source_idxs = np.array([0, 1]) if player_1_turn else np.array([2, 3])
        self.target_idxs = np.array([2, 3]) if player_1_turn else np.array([0, 1])

    @staticmethod
    def get_initial_state():
        return np.array([1, 1, 1, 1]), True, get_initial_cache().cache

    def is_valid_move(self, move):
        source_hand_idx = self.source_idxs[0 if move.source_hand == LEFT_HAND else 1]
        target_hand_idx = self.target_idxs[0 if move.target_hand == LEFT_HAND else 1]
        if move.move_type == ATTACK:
            return self.state[source_hand_idx] >= 0

        # Handle SWAP move
        source_before = self.state[source_hand_idx]
        target_before = self.state[target_hand_idx]
        source_after = source_before - move.amount
        target_after = target_before + move.amount

        # cannot swap more fingers than source, or target will exceed 4 fingers
        if source_after < 0 or target_after > 4:
            return False

        # the source hand must have at least one finger
        if source_before <= 0:
            return False

        # if the target hand has more than 4 fingers after the move, it's invalid
        if target_after > 4:
            return False

        # if swapping all fingers, the target hand must have at least one finger
        if move.amount == source_before and target_before <= 0:
            return False

        return True

    def get_valid_moves(self):
        moves = create_all_moves()
        valid_moves = []
        for move in moves:
            if self.is_valid_move(move):
                valid_moves.append(move)

        return valid_moves

    def play(self, move: Move) -> "Game":
        new_state = self.state.copy()
        if move.move_type == ATTACK:
            source_hand_idx = self.source_idxs[
                0 if move.source_hand == LEFT_HAND else 1
            ]
            target_hand_idx = self.target_idxs[
                0 if move.target_hand == LEFT_HAND else 1
            ]

            new_state[target_hand_idx] = (
                min(new_state[target_hand_idx] + new_state[source_hand_idx], 5) % 5
            )

        else:
            # Handle SWAP move
            source_hand_idx = self.source_idxs[
                0 if move.source_hand == LEFT_HAND else 1
            ]
            target_hand_idx = self.source_idxs[
                0 if move.source_hand == LEFT_HAND else 1
            ]

            new_state[source_hand_idx] -= move.amount
            new_state[target_hand_idx] += move.amount
            assert new_state[source_hand_idx] >= 0, "Source hand cannot go negative"
            assert (
                new_state[target_hand_idx] <= 4
            ), "Target hand cannot exceed 4 fingers"

        hs = hash(self)
        new_seen_states = self.seen_states.copy()
        cache = Cache(new_seen_states)
        cache[hs] = 0.0
        assert hs in cache
        return Game(new_state, not self.player_1_turn, cache.cache)

    def is_terminal(self):
        hs = hash(self)
        cache = Cache(self.seen_states)
        return (
            self.state[:2].sum() == 0
            or self.state[2:].sum() == 0
            or hs in cache
        )

    def get_state_value(self):
        # Implement logic to calculate the value of the state for the given player
        cache = Cache(self.seen_states)
        if self.is_terminal():
            hs = hash(self)
            p1_dead = self.state[:2].sum() == 0
            p2_dead = self.state[2:].sum() == 0

            if hs in cache:
                return cache[hs]

            if p1_dead and not p2_dead:
                return -1.0 if self.player_1_turn else 1.0
            if p2_dead and not p1_dead:
                return 1.0 if self.player_1_turn else -1.0

        return 0.0

    def __hash__(self):
        # Implement logic to hash the current state for fast lookup
        state = self.state.copy()
        state[:2] = np.sort(state[:2])
        state[2:] = np.sort(state[2:])

        base = 5
        return int(
            state[0] * base**4
            + state[1] * base**3
            + state[2] * base**2
            + state[3] * base
            + int(self.player_1_turn)
        )

    @staticmethod
    def hash_limit():
        # Calculate the magnitude of the hash value
        return 4 * 5**4 + 4 * 5**3 + 4 * 5**2 + 4 * 5 + 1


@njit
def get_initial_cache():
    return Cache(
        np.zeros(4 * 5**4 + 4 * 5**3 + 4 * 5**2 + 4 * 5 + 2, dtype=np.float64) + np.nan
    )


if __name__ == "__main__":
    init = Game.get_initial_state()
    print(init)
    g = Game(init[0], init[1], init[2])
    