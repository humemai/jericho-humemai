"""
Utility functions for the agent.
"""

from glob import glob
from jericho import FrotzEnv


def get_game_paths(only_33=True) -> list:
    """
    Get the paths to the games in the Jericho Game Suite.

    Args:
        only_33 (bool): If True, only return the 33 games used in the paper.
    Returns:
        game_paths (list): List of paths to the games.
    """

    game_paths = glob("./z-machine-games-master/jericho-game-suite/*")

    if only_33:
        possible_games = [
            "905",
            "acorncourt",
            "advent",
            "adventureland",
            "afflicted",
            "anchor",
            "awaken",
            "balances",
            "deephome",
            "detective",
            "dragon",
            "enchanter",
            "gold",
            "inhumane",
            "jewel",
            "karn",
            "library",
            "ludicorp",
            "moonlit",
            "omniquest",
            "pentari",
            "reverb",
            "snacktime",
            "sorcerer",
            "spellbrkr",
            "spirit",
            "temple",
            "tryst205",
            "yomomma",
            "zenon",
            "zork1",
            "zork3",
            "ztuu",
        ]
        game_paths = [
            game_path
            for game_path in game_paths
            if game_path.split("/")[-1].split(".")[0] in possible_games
        ]

    return game_paths


def create_game(game_path: str):
    """
    Returns a Jericho FrotzEnv game instance given the path to the game.
    """
    return FrotzEnv(game_path)
