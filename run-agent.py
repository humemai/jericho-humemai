import json
import argparse
from utils import get_game_paths, create_game
from tqdm.auto import tqdm
from agent import RandomAgent, WalkThroughAgent, LLMAgent


def get_game_name(game_path: str) -> str:
    """
    Get the name of the game from the game path.

    Args:
        game_path (str): The path to the game file.
    Returns:
        game_name (str): The name of the game.
    """

    game_name = game_path.split("/")[-1].split(".")[0]

    return game_name


def run_games(
    agent_type="RandomAgent",
    max_steps=None,
    only_33=True,
    output_file="results.json",
    api_key=None,
    model=None,
    max_chat_history_size=10,
    fallback_to_random=True,
    timeout=10,
) -> None:
    """
    Runs a specified agent on a set of games and saves the results to a JSON file.

    Args:
        agent_type (str): The type of agent to use (default: "RandomAgent").
        max_steps (int): The maximum number of steps for each game (default: None).
        only_33 (bool): Whether to only use the games in the '33' set (default: True).
        output_file (str): The name of the file to save the results to (default:
            "results.json").
        api_key (str): The API key for the LLMAgent (default: None).
        model (str): The model to use for the LLMAgent (default: None).
        max_chat_history_size (int): The maximum chat history size for the LLMAgent
            (default: 10).
        fallback_to_random (bool): Whether to fallback to random for the LLMAgent
            (default: True).
        timeout (int): The timeout for the LLMAgent (default: 10).

    """

    game_paths = get_game_paths(only_33=only_33)

    results = {}
    for game_path in tqdm(game_paths, desc="Running games"):
        env = create_game(game_path)

        if agent_type == "RandomAgent":
            agent = RandomAgent(env, max_steps=max_steps)
        elif agent_type == "WalkThroughAgent":
            agent = WalkThroughAgent(env, max_steps=max_steps)
        elif agent_type == "LLMAgent":
            if not api_key or not model:
                raise ValueError("API key and model must be specified for LLMAgent")
            agent = LLMAgent(
                api_key=api_key,
                model=model,
                env=env,
                max_steps=max_steps,
                max_chat_history_size=max_chat_history_size,
                fallback_to_random=fallback_to_random,
                timeout=timeout,
            )
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")

        agent.run()

        num_observations = len(agent.observations)
        num_actions = len(agent.actions)
        num_rewards = len(agent.rewards)
        total_reward = sum(agent.rewards.values())

        result = {
            "agent": agent.__class__.__name__,
            "game_path": game_path,
            "max_steps": agent.max_steps,
            "steps": agent.step,
            "num_observations": num_observations,
            "num_actions": num_actions,
            "num_rewards": num_rewards,
            "total_reward": total_reward,
            "max_score": env.get_max_score(),
            "info": agent.info,
        }
        results[get_game_name(game_path)] = result

        print(f"Game: {game_path}")
        print(f"  Agent: {agent.__class__.__name__}")
        print(f"  Max Steps: {agent.max_steps}")
        print(f"  Steps: {agent.step}")
        print(f"  Observations: {num_observations}")
        print(f"  Actions: {num_actions}")
        print(f"  Rewards: {num_rewards}")
        print(f"  Total Reward: {total_reward}")
        print(f"  Max Score: {env.get_max_score()}")
        print(f"  Walkthrough length: {len(env.get_walkthrough())}")
        print(f"  Info: {agent.info}")
        print()

        # Save the results to a JSON file
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)

        print(f"Results saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run games with a specified agent.")
    parser.add_argument(
        "--agent_type",
        type=str,
        default="RandomAgent",
        help="The type of agent to use.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="The maximum number of steps for each game.",
    )
    parser.add_argument(
        "--only_33", action="store_true", help="Only use the games in the '33' set."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="results.json",
        help="The name of the file to save the results to.",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="The API key for the LLMAgent.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="The model to use for the LLMAgent.",
    )
    parser.add_argument(
        "--max_chat_history_size",
        type=str,
        default=10,
        help="The maximum chat history size for the LLMAgent.",
    )
    parser.add_argument(
        "--fallback_to_random",
        type=bool,
        default=True,
        help="Whether to fallback to random for the LLMAgent.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=10,
        help="The timeout for the LLMAgent.",
    )

    args = parser.parse_args()

    run_games(
        args.agent_type,
        args.max_steps,
        args.only_33,
        args.output_file,
        args.api_key,
        args.model,
        args.max_chat_history_size,
        args.fallback_to_random,
        args.timeout,
    )
