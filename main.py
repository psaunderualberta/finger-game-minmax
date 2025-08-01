import time
from itertools import product

import numpy as np
from numba import njit
from tqdm import tqdm

from game import Cache, Game, get_initial_cache
from moves import move2str, Move


# TODO: Add optimal move tracking
@njit
def alpha_beta_search(
    state: Game, depth: int, alpha: float, beta: float, endgame_db: Cache
):
    state_hash = hash(state)
    if state.is_terminal() or state_hash in endgame_db:
        if state_hash in endgame_db:
            return endgame_db[state_hash], 1, True, [Move(1, 1, 1, 1)]
        return state.get_state_value(), 1, True, [Move(1, 1, 1, 2)]
    if depth == 0:
        return state.get_state_value(), 1, False, [Move(1, 1, 1, 3)]

    level_cache = get_initial_cache()

    state_value = -1.0
    game_solved = True
    total_count = 0
    best_move = Move(1, 1, 1, 4)
    best_following_moves = [best_move]
    for move in state.get_valid_moves():
        next_state = state.play(move)
        key = hash(next_state)
        if key in level_cache:
            continue
        level_cache[key] = 1
        value, cnt, move_solved, follow_moves = alpha_beta_search(next_state, depth - 1, -beta, -alpha, endgame_db)
        game_solved = game_solved and move_solved
        total_count += cnt

        if -value >= state_value:
            state_value = -value
            best_move = move
            best_following_moves = [best_move] + follow_moves

        alpha = max(alpha, state_value)

    if state_value != 0:
        game_solved = True  # this player can guarantee a win or cannot avoid a loss

    return state_value, total_count, game_solved, best_following_moves


def generate_endgame_db(depth=1, existing_db=get_initial_cache()):
    endgame_db = existing_db.copy()
    analyzed_states = get_initial_cache()
    states = list(product(range(5), repeat=4))
    num_solved = 0
    for i1, i2, i3, i4 in tqdm(states):
        if (i1, i2) == (0, 0) or (i3, i4) == (0, 0):
            continue
        for p1_turn in [True, False]:
            initial_state, _, seen_states = Game.get_initial_state()
            initial_state = np.array([i1, i2, i3, i4], dtype=np.int64)
            game = Game(initial_state, p1_turn, seen_states)

            state_hash = hash(game)
            if state_hash in analyzed_states or state_hash in existing_db:
                continue
            analyzed_states[state_hash] = 1

            result, _, solved, best_move = alpha_beta_search(game, depth, -1, 1, existing_db)
            if solved:
                smove = list(map(move2str, best_move))
                print(f"Solved state: {initial_state}, Result: {result}, Player 1 Turn: {p1_turn}, Best Move: {smove}")
                endgame_db[state_hash] = result
                num_solved += 1

    print(f"Endgame database generated with {num_solved} | {len(endgame_db)} solved states.")
    return endgame_db


def repeated_endgame_search(depth=3):
    endgame_db = get_initial_cache()
    assert len(endgame_db) == 0, "Endgame database should be empty before generation."
    new_endgame_db = generate_endgame_db(depth)
    while len(new_endgame_db) > len(endgame_db):
        print(f"New endgame database size: {len(new_endgame_db)}")
        endgame_db = new_endgame_db
        new_endgame_db = generate_endgame_db(depth, existing_db=endgame_db)
    
    print(f"Final endgame database size: {len(endgame_db)}")
    return endgame_db


def main():
    endgame_db = repeated_endgame_search(depth=1)
    initial_state, player_1_turn, seen_states = Game.get_initial_state()

    initial_state = np.array([1, 1, 1, 1], dtype=np.int64)  # Example initial state

    game = Game(initial_state, player_1_turn, seen_states)

    # # Start the alpha-beta search
    # print("Starting alpha-beta search...")
    # for depth in range(1, 50):
    #     start = time.time()
    #     best_value = alpha_beta_search(game, depth, -1, 1, endgame_db)
    #     end = time.time()

    #     print(f"Best value at depth {depth}: {best_value}")
    #     print(f"Time taken: {end - start:.4f} seconds")

    #     if best_value[0] != 0:
    #         break


if __name__ == "__main__":
    main()
