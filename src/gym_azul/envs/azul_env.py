from typing import Tuple, Dict, List, Optional

import numpy as np  # type: ignore
from gym import logger  # type: ignore

from gym_azul.agents.greedy_agent import GreedyAgent
from gym_azul.envs.mu_zero_env import MuzeroEnv
from gym_azul.game.game_engine import AzulGame
from gym_azul.spaces.azul_spaces import observation_space, action_space
from gym_azul.spaces.from_azul_spaces import game_action_from_action
from gym_azul.spaces.to_azul_spaces import observation_from_state, \
    action_from_game_action
from gym_azul.util.format_utils import format_observation


class AzulEnv(MuzeroEnv):
    """
   Description:
       Azul game environment
   Episode Termination:
       Azul gameover (one full row)
       Episode length is greater than max_turns.
   """
    render_mode: str
    num_players: int
    internal_seed: List[int]
    max_turns: int
    game: AzulGame

    metadata = {"render.modes": ["human"]}

    def __init__(self, render_mode: str = "human", num_players: int = 2,
        max_turns: int = 500) -> None:
        super(AzulEnv, self).__init__()

        self.render_mode = render_mode
        self.num_players = num_players
        self.max_turns = max_turns
        self.internal_seed = []
        self.expert_agent = GreedyAgent()

        self.action_space = action_space()
        self.observation_space = observation_space(self.num_players)
        self.game = AzulGame(num_players=self.num_players)

    def seed(self, seed: Optional[int] = None) -> List[int]:
        if seed is not None:
            set_seed = self.game.seed(seed)
            self.internal_seed = [set_seed]
        return self.internal_seed

    def step(self, action: int) -> Tuple[
        np.ndarray, float, bool, Dict[str, int]]:
        reward = 0.0
        info_before = {
            "turn": self.get_turn(),
            "round": self.get_round(),
            "player": self.to_play()
        }
        move_info: Dict[str, int] = {}
        slot, color, line = game_action_from_action(action)

        if self.is_done():
            logger.warn(
                "Game over, call reset. Undefined action.")
        else:
            reward, move_info = self.game.action_handler(slot, color, line)

        info_after = {
            "next_turn": self.get_turn(),
            "next_round": self.get_round(),
            "next_player": self.to_play()
        }

        info = {**info_before, **move_info, **info_after}

        return self.get_observation(), reward, self.is_done(), info

    def reset(self) -> np.ndarray:
        self.game.reset(start_player=0)
        return self.get_observation()

    def render(self, mode="human"):
        observation = self.get_observation()
        print(f"Next turn: {self.get_turn()}")
        print(f"Next round: {self.get_round()}")
        print(f"Next player: {self.to_play()}")
        print(format_observation(observation))

    def close(self) -> None:
        pass

    def to_play(self) -> int:
        return self.game.current_player

    def legal_actions(self) -> List[int]:
        parsed_legal_actions = self.game.legal_actions()
        legal_actions = [action_from_game_action(action) for action in
                         parsed_legal_actions]

        return legal_actions

    def expert_action(self) -> int:
        return self.expert_agent.act(self.to_play(),
                                     self.legal_actions(),
                                     self.get_observation())

    def get_observation(self) -> np.ndarray:
        return observation_from_state(self.game.state)

    def is_done(self) -> bool:
        game_over = self.game.game_over
        too_many_turns = (self.game.turns_count > self.max_turns)
        legal_actions = len(self.legal_actions())
        return game_over or too_many_turns or legal_actions == 0

    def get_round(self) -> int:
        return self.game.rounds_count

    def get_turn(self) -> int:
        return self.game.turns_count
