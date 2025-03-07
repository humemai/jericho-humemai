"""
This module provides a base Agent class and a specialized LLMAgent class. The Agent
class outlines a method for choosing actions, which must be implemented by subclasses.
The LLMAgent extends Agent, integrating with an external LLM to determine next steps in
a text-based adventure game.
"""

import random


class Agent:
    """
    A base class for agents that interact with text-based environments. Subclasses
    should implement the choose_action method to select actions based on the current
    observation and list of valid actions
    """

    def __init__(self, env, max_steps: int = None) -> None:
        """
        Initializes the Agent with an empty history list.

        Args:
            env (FrotzEnv): The FrotzEnv environment object
            max_steps: The maximum number of steps to take in an episode.
                if None, the episode will run until the environment returns done=True

        """
        self.env = env
        self.max_steps = max_steps

        self.reset_env()

    def reset_env(self):
        """
        Resets the environment and the agent's internal state.
        """
        self.step = 0
        self.observations = {}
        self.actions = {}
        self.rewards = {}

        observation, info = self.env.reset()
        self.observations[self.step] = observation  # Add the first observation

    def choose_action(self, valid_actions: list[str]) -> str:
        """
        Raises a NotImplementedError indicating that subclasses must implement a method
        to choose actions.

        Args:
            valid_actions (list[str]): A list of valid actions available in the
                environment.

        Returns:
            str: The chosen action (subclasses should implement the method body).
        """
        raise NotImplementedError("This method should be overridden by subclasses")

    def close_env(self):
        """
        Closes the environment.
        """
        self.env.close()

    def update_action(self, action: str) -> None:
        """
        Updates the agent's internal state with the chosen action.

        Args:
            action (str): The chosen action.
        """
        self.actions[self.step] = action

    def update_reward(self, reward: float) -> None:
        """
        Updates the agent's internal state with the received reward

        Args:
            reward (float): The reward received after taking the action.
        """
        self.rewards[self.step] = reward

    def update_observation(self, observation: str) -> None:
        """
        Updates the agent's internal state with the received observation.

        Args:
            observation (str): The observation after taking the action.
        """
        self.observations[self.step] = observation

    def run(self) -> None:
        """
        Runs the agent in the environmen.
        """
        done = False
        while not done and (not self.max_steps or self.step < self.max_steps):
            valid_actions = self.env.get_valid_actions()
            if len(valid_actions) == 0:
                print("No valid actions left. Ending episode.")
                break
            action = self.choose_action(valid_actions)
            self.update_action(action)

            observation, reward, done, info = self.env.step(action)

            self.step += 1

            self.update_reward(reward)
            self.update_observation(observation)

        self.info = info
        self.close_env()


class WalkThroughAgent(Agent):
    """
    An agent that follows a predefined walkthrough of actions in a text-based
    adventure game.
    """

    def __init__(self, env, max_steps: int = None) -> None:
        """
        Initializes the WalkThroughAgent with a predefined walkthrough of actions.

        Args:
            env (FrotzEnv): The FrotzEnv environment object.
            max_steps (int): The maximum number of steps to take in an episode.
                If None, the episode will run until the environment returns done=True.
        """
        super().__init__(env, max_steps)

    def run(self) -> None:
        """
        Runs the agent in the environment.
        """
        done = False
        for action in self.env.get_walkthrough():
            self.update_action(action)

            observation, reward, done, info = self.env.step(action)

            self.step += 1

            self.update_reward(reward)
            self.update_observation(observation)

        self.info = info
        self.close_env()


class RandomAgent(Agent):
    """
    An agent that chooses actions randomly from the list of valid actions.
    """

    def __init__(self, env, max_steps: int = None) -> None:
        """
        Initializes the RandomAgent.
        """
        super().__init__(env, max_steps)

    def choose_action(self, valid_actions: list[str]) -> str:
        """
        Chooses an action randomly from the list of valid actions.

        Args:
            valid_actions (list[str]): A list of valid actions available in the
                environment.

        Returns:
            str: The chosen action.
        """
        return random.choice(valid_actions)
