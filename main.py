from numba import njit
from game import Game, Cache, get_initial_cache
from moves import create_all_moves
import time
import numpy as np
from itertools import product
from tqdm import tqdm


# @njit
def alpha_beta_search(state: Game, depth: int, alpha: float, beta: float, endgame_db: Cache):
    state_hash = hash(state)
    if depth == 0 or state.is_terminal() or state_hash in endgame_db:
        if state_hash in endgame_db:
            return endgame_db[state_hash], 1
        return state.get_state_value(), 1
    
    level_cache = get_initial_cache()

    state_value = np.nan
    total_count = 0
    for move in state.get_valid_moves():
        next_state = state.play(move)
        key = hash(next_state)
        if key in level_cache:
            continue
        value, cnt = alpha_beta_search(next_state, depth - 1, -beta, -alpha, endgame_db)
        total_count += cnt
        if ~np.isnan(value):
            state_value = max(-value, state_value) if ~np.isnan(state_value) else -value

        level_cache[key] = 1

        if state_value >= beta:
            break

        alpha = max(alpha, state_value)
    
    return state_value, total_count


def generate_endgame_db():
    endgame_db = get_initial_cache()    
    states = list(product(range(5), repeat=4))
    for i1, i2, i3, i4 in tqdm(states):
        if (i1, i2) == (0, 0) or (i3, i4) == (0, 0):
            continue
        for p1_turn in [True, False]:
            initial_state, _, seen_states = Game.get_initial_state()
            initial_state = np.array([i1, i2, i3, i4], dtype=np.int64)
            game = Game(initial_state, p1_turn, seen_states)

            result = alpha_beta_search(game, 10, -1, 1, get_initial_cache())
            if ~np.isnan(result[0]):
                state_hash = hash(game)
                endgame_db[state_hash] = result[0]
            else:
                print(f"Invalid state: {initial_state}, p1_turn: {p1_turn}")
    
    return endgame_db


def main():
    endgame_db = generate_endgame_db()
    initial_state, player_1_turn, seen_states = Game.get_initial_state()

    initial_state = np.array([1, 1, 1, 1], dtype=np.int64)  # Example initial state

    game = Game(initial_state, player_1_turn, seen_states)

    # Start the alpha-beta search
    print("Starting alpha-beta search...")
    for depth in range(1, 40):
        start = time.time()
        best_value = alpha_beta_search(game, depth, -1, 1, endgame_db)
        end = time.time()

        print(f"Best value at depth {depth}: {best_value}")
        print(f"Time taken: {end - start:.4f} seconds")

        if best_value[0] != 0:
            break


if __name__ == "__main__":
    main()