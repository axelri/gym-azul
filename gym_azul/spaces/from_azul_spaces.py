from typing import Tuple
import numpy as np

from gym_azul.game.game_state import AzulState, AzulPlayerState
from gym_azul.game.game_utils import TOTAL_COLORS, TOTAL_LINES


def game_action_from_action(action: int) -> Tuple[int, int, int]:
    slot = action // (TOTAL_COLORS * TOTAL_LINES)
    after_slot = action % (TOTAL_COLORS * TOTAL_LINES)

    color = after_slot // TOTAL_LINES
    after_color = after_slot % TOTAL_LINES

    line = after_color

    return slot, color, line


def state_pattern_lines(pattern_lines: np.ndarray) -> np.ndarray:
    lines = []
    for line in range(TOTAL_LINES):
        color = pattern_lines[line, 0]
        count = 0
        if color != 6:
            # do not count unset values
            for column in range(5):
                if pattern_lines[line, column] == color:
                    count += 1
        lines.append([color, count])

    return np.array(lines)


def state_wall(wall: np.ndarray) -> np.ndarray:
    return np.copy(wall)


def state_floor_line(floor_line: np.ndarray) -> np.ndarray:
    # grab one column
    return floor_line[:, 0].T


def state_points(points: np.ndarray) -> int:
    return points[0]


def state_starting_marker(starting_marker: np.ndarray) -> bool:
    return bool(starting_marker[0])


def player_state(
    player_observation: np.ndarray) -> Tuple[AzulPlayerState, bool]:
    obs_pattern_lines = player_observation[0:5, 0:5]
    obs_wall = player_observation[5:10, 0:5]
    obs_floor_line = player_observation[0:7, 5:10]
    obs_points = player_observation[7, 5:10]
    obs_has_starting_marker = player_observation[8, 5:10]

    pattern_lines = state_pattern_lines(obs_pattern_lines)
    wall = state_wall(obs_wall)
    floor_line = state_floor_line(obs_floor_line)
    points = state_points(obs_points)
    has_starting_marker = state_starting_marker(obs_has_starting_marker)

    return AzulPlayerState(points, pattern_lines, wall,
                           floor_line), has_starting_marker


def slots_state(slots: np.ndarray) -> np.ndarray:
    return np.copy(slots)


def bag_state(bag: np.ndarray) -> np.ndarray:
    # grab one column
    return bag[:, 0].T


def lid_state(lid: np.ndarray) -> np.ndarray:
    # grab one column
    return lid[:, 0].T


def state_from_observation(observation: np.ndarray) -> AzulState:
    channels, _, _ = observation.shape

    num_players = channels - 1
    starting_marker = num_players

    players = []
    for player_idx in range(num_players):
        player_channel = observation[player_idx, :, :]
        p_state, has_token = player_state(player_channel)
        players.append(p_state)
        if has_token:
            starting_marker = player_idx

    slots_obs = observation[-1, 0:10, 0:5]
    bag_obs = observation[-1, 0:5, 5:10]
    lid_obs = observation[-1, 5:10, 5:10]

    slots = slots_state(slots_obs)
    bag = bag_state(bag_obs)
    lid = lid_state(lid_obs)

    azul_state = AzulState(num_players, players, slots, bag, lid,
                           starting_marker)

    return azul_state
