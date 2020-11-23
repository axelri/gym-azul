from typing import Tuple, List, Dict

import numpy as np  # type: ignore

from gym_azul.constants import ColorTile, \
    Tile, Color, Line, Column, LineAmount, FloorLineTile, Slot, Player, \
    PLAYER_INACTIVE_OBS, NumPlayers, StartingMarker
from gym_azul.model.state import AzulState, AzulPlayerState, PatternLine


def state_pattern_lines(obs_pattern_lines: np.ndarray) -> List[PatternLine]:
    pattern_lines = []
    for line in Line:
        color = obs_pattern_lines[line, 0]
        amount = 0
        if color != ColorTile.EMPTY:
            # do not count unset values
            for column in Column:
                if obs_pattern_lines[line, column] == color:
                    amount += 1
        pattern_lines.append(PatternLine(color, LineAmount(amount)))

    return pattern_lines


def state_wall(obs_wall: np.ndarray) -> List[List[ColorTile]]:
    wall = []
    for line in Line:
        wall_line = []
        for column in Column:
            color = obs_wall[line, column]
            wall_line.append(ColorTile(color))
        wall.append(wall_line)

    return wall


def state_floor_line(obs_floor_line: np.ndarray) -> List[Tile]:
    floor_line = []
    for tile in FloorLineTile:
        tile_value = obs_floor_line[tile, 0]
        floor_line.append(Tile(tile_value))
    return floor_line


def state_points(obs_points: np.ndarray) -> int:
    return obs_points[0]


def state_starting_marker(obs_starting_marker: np.ndarray) -> bool:
    return bool(obs_starting_marker[0])


def state_next_turn(obs_next_turn: np.ndarray) -> bool:
    return bool(obs_next_turn[0])


def player_state(
    player_observation: np.ndarray
) -> Tuple[AzulPlayerState, bool, bool, bool]:
    if player_observation[0, 0] == PLAYER_INACTIVE_OBS:
        return AzulPlayerState(), False, False, False

    obs_pattern_lines = player_observation[0:5, 0:5]
    obs_wall = player_observation[5:10, 0:5]
    obs_floor_line = player_observation[0:7, 5:10]
    obs_points = player_observation[7, 5:10]
    obs_has_starting_marker = player_observation[8, 5:10]
    obs_has_next_turn = player_observation[9, 5:10]

    pattern_lines = state_pattern_lines(obs_pattern_lines)
    wall = state_wall(obs_wall)
    floor_line = state_floor_line(obs_floor_line)
    points = state_points(obs_points)
    has_starting_marker = state_starting_marker(obs_has_starting_marker)
    has_next_turn = state_next_turn(obs_has_next_turn)

    state = AzulPlayerState(points, pattern_lines, wall, floor_line)

    return state, has_starting_marker, has_next_turn, True


def slots_state(obs_slots: np.ndarray) -> List[Dict[Color, int]]:
    slots = []
    for slot in Slot:
        slot_dict: Dict[Color, int] = {}
        for color in Color:
            slot_dict[color] = obs_slots[slot, color]
        slots.append(slot_dict)
    return slots


def bag_state(obs_bag: np.ndarray) -> Dict[Color, int]:
    bag: Dict[Color, int] = {}
    for color in Color:
        bag_value = obs_bag[color, 0]
        bag[color] = bag_value
    return bag


def lid_state(obs_lid: np.ndarray) -> Dict[Color, int]:
    lid: Dict[Color, int] = {}
    for color in Color:
        lid_value = obs_lid[color, 0]
        lid[color] = lid_value
    return lid


def state_from_observation(observation: np.ndarray) -> AzulState:
    channels, _, _ = observation.shape

    starting_marker = StartingMarker.CENTER
    current_player = Player.PLAYER_1

    num_players = 0

    players = []
    for player in Player:
        player_channel = observation[player, :, :]
        state, has_starting_token, has_next_turn, is_active = player_state(
            player_channel)
        players.append(state)
        if has_starting_token:
            starting_marker = StartingMarker(player)
        if has_next_turn:
            current_player = player
        if is_active:
            num_players += 1

    slots_obs = observation[-1, 0:10, 0:5]
    bag_obs = observation[-1, 0:5, 5:10]
    lid_obs = observation[-1, 5:10, 5:10]

    slots = slots_state(slots_obs)
    bag = bag_state(bag_obs)
    lid = lid_state(lid_obs)

    azul_state = AzulState(players, slots, bag, lid, starting_marker,
                           current_player, NumPlayers(num_players))

    return azul_state
