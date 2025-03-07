import json
import random
import re
from typing import Any
import requests

from .agent import Agent


class LLMAgent(Agent):
    """
    An agent that uses a Large Language Model (LLM) to play text-based adventure games.

    This agent sends the game state to an LLM through the OpenRouter API and interprets
    the model's responses to select actions from the valid actions list. It maintains a
    chat history to provide context to the model about previous game states and actions.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        env: Any,
        max_steps: int | None,
        max_chat_history_size: int = 10,
        fallback_to_random: bool = True,
        timeout: int = 10,
    ) -> None:
        """
        Initialize the LLM-based agent.

        Args:
            api_key (str): API key for the OpenRouter service.
            model (str): The LLM model identifier to use from OpenRouter (e.g.,
                'meta-llama/llama-3.2-1b-instruct', 'meta-llama/llama-3.2-3b-instruct',
                'meta-llama/llama-3.3-70b-instruct').
            env (Any): The game environment object.
            max_steps (int | None): Maximum number of steps to take before terminating.
            max_chat_history_size (int): Maximum number of conversation turns to keep in
                history.
            fallback_to_random (bool): Whether to choose a random action when the LLM
                fails.
            timeout (int): HTTP request timeout in seconds.
        """
        super().__init__(env, max_steps)
        self.api_key = api_key
        self.model = model

        self.max_chat_history_size = max_chat_history_size
        self.fallback_to_random = fallback_to_random
        self.timeout = timeout
        self.info: dict[str, Any] = {}

        self.system_message = {
            "role": "system",
            "content": (
                "You are an AI agent playing a text-based adventure game. Your goal "
                "is to maximize your score by solving puzzles, exploring the "
                "environment, and collecting important items.\nGuidelines:\n"
                "1. Carefully read and remember descriptions, noting items, locations, "
                "and potential clues.\n"
                "2. Maintain awareness of your inventory and use inventory items "
                "creatively to overcome challenges.\n"
                "3. Prioritize actions that explore new areas or interact meaningfully "
                "with your surroundings.\n"
                "4. Pay attention to hints in textual descriptions that may imply "
                "hidden puzzles or useful interactions.\n"
                "5. Avoid repeating actions unless you have new reasons to try "
                "them.\n"
                "IMPORTANT: Respond ONLY with the exact action from the provided valid "
                "actions list."
            ),
        }
        self.chat_history: list[dict[str, str]] = [self.system_message]
        self.chat_history.append(
            {
                "role": "user",
                "content": f"Step: {self.step}: Observation: {self.observations[self.step]}",
            }
        )  # Add the first observation

    def choose_action(
        self,
        valid_actions: list[str],
    ) -> str:
        """
        Select an action from the list of valid actions using the LLM.

        This method sends the current game state and valid actions to the LLM,
        and extracts a valid action from the LLM's response. If the LLM fails to
        produce a valid action, it will either fall back to choosing a random action
        or raise an exception based on the fallback_to_random setting.

        Args:
            valid_actions (list[str]): List of valid actions the agent can take.

        Returns:
            str: The selected action to perform.

        Raises:
            ValueError: If no valid action is found in the LLM response and
                fallback_to_random is False.
            requests.exceptions.RequestException: If there's an API communication error
                and fallback_to_random is False.
        """
        if len(self.chat_history) > self.max_chat_history_size * 3:
            self.chat_history = [self.system_message] + self.chat_history[1:][
                -self.max_chat_history_size * 3 :
            ]

        messages = self.chat_history + [
            {
                "role": "user",
                "content": (
                    f"Current observation: {self.observations[self.step]}\n"
                    f"Valid actions: {valid_actions}\n"
                    "Which action do you want to take? Respond with the action only."
                ),
            }
        ]

        try:
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "http://localhost:8888",
                    "X-Title": "Jericho Agent",
                },
                data=json.dumps({"model": self.model, "messages": messages}),
                timeout=self.timeout,
            )
            response.raise_for_status()
            raw_response = response.json()["choices"][0]["message"]["content"].strip()

            # Use regex to find the exact matching valid action
            for action in valid_actions:
                if re.search(rf"\b{re.escape(action)}\b", raw_response, re.IGNORECASE):
                    return action

            if self.fallback_to_random:
                print(
                    f"No valid action found in LLM response. Using random action. "
                    f"Response: {raw_response}"
                )
                return random.choice(valid_actions)
            else:
                raise ValueError(
                    f"No valid action found in LLM response and fallback_to_random is "
                    f"False. Raw response: {raw_response}"
                )

        except (requests.exceptions.RequestException, KeyError, ValueError) as e:
            print(f"Error calling OpenRouter API: {e}")
            if self.fallback_to_random:
                return random.choice(valid_actions)
            else:
                raise

    def update_action(self, action: str) -> None:
        """
        Updates the agent's internal state with the chosen action.

        This method records the action taken at the current step and adds it to
        the chat history for context in future decisions.

        Args:
            action (str): The chosen action.
        """
        self.actions[self.step] = action
        self.chat_history.append(
            {"role": "assistant", "content": f"Step {self.step}: Action: {action}"}
        )

    def update_reward(self, reward: float) -> None:
        """
        Updates the agent's internal state with the received reward.

        This method records the reward received after taking an action and adds it
        to the chat history for context in future decisions.

        Args:
            reward (float): The reward received after taking the action.
        """
        self.rewards[self.step] = reward

        self.chat_history.append(
            {"role": "user", "content": f"Step {self.step}: Reward: {reward}"}
        )

    def update_observation(self, observation: str) -> None:
        """
        Updates the agent's internal state with the received observation.

        This method records the observation received after taking an action,
        and adds it to the chat history for context in future decisions.

        Args:
            observation (str): The observation after taking the action.
        """
        self.observations[self.step] = observation
        self.chat_history.append(
            {"role": "user", "content": f"Step {self.step}: Observation: {observation}"}
        )
