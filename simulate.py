#!/usr/bin/env python3
"""
simulate.py: Run a headless UnoEnvironment game with random legal moves
and print the first finishing player and step count.
"""
import numpy as np
from environment import UnoEnvironment

def simulate_game(player_count=4, seed=None):
    """
    Simulate one game with `player_count` players, all choosing random legal moves.
    Prints which player finishes first (wins) and at which step.
    """
    if seed is not None:
        np.random.seed(seed)
    env = UnoEnvironment(player_count)
    step = 0
    while True:
        # gather legal actions
        legal = [a for a in range(env.action_count()) if env.legal_move(a)]
        # pick random legal action
        action = np.random.choice(legal)
        _, _, done, info = env.step(action)
        step += 1
        # check if a player finished (first finishing event)
        if info.get('player') == 2:
            print(f"Player {info['turn']} finished (won) at step {step}.")
            break

if __name__ == '__main__':
    simulate_game()