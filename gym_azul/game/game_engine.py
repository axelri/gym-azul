from typing import Dict, List, Tuple

import numpy as np
from numpy.random import default_rng
from numpy.random._generator import Generator

from gym_azul.game.game_calcs import calc_move, is_next_round, \
    is_game_over, calc_wall_score, calc_bonus_points
from gym_azul.game.game_state import AzulState
from gym_azul.game.game_utils import wall_color_column, \
    max_tiles_for_line, TOTAL_COLORS, get_num_factories, PENALTIES, \
    generate_legal_actions


class AzulGame:
    random: Generator
    num_players: int
    state: AzulState
    turns_count: int
    rounds_count: int
    current_player: int
    game_over: bool

    def __init__(self, num_players: int) -> None:
        self.random = default_rng()

        self.num_players = num_players
        self.state = AzulState(num_players)
        self.turns_count = 0
        self.rounds_count = 0
        self.current_player = -1
        self.game_over = False

    def seed(self, seed: int) -> int:
        """
        Sets and return the seed
        """
        self.random = default_rng(seed)
        return seed

    def reset(self, start_player: int = 0) -> None:
        self.state = AzulState(self.num_players)
        self.turns_count = 1
        self.rounds_count = 1
        self.current_player = start_player
        self.game_over = False
        self.next_round()

    def legal_actions(self) -> List[Tuple[int, int, int]]:
        return generate_legal_actions(self.state.slots)

    def play_turn(self, slot: int, color: int, line: int,
        place_pattern_line: int, place_floor_line: List[Tuple[int, int]],
        discard: List[int]) -> None:
        """
        Modify board with action
        """
        slots = self.state.slots
        player_board = self.state.players[self.current_player]

        # Draw from slot
        slots[slot, color] = 0
        if slot != 0:
            # move rest to center if from factory
            for move_color in range(TOTAL_COLORS):
                slots[0, move_color] += slots[slot, move_color]
                slots[slot, move_color] = 0

        # Put in pattern line if putting > 0 tiles
        pattern_lines = player_board.pattern_lines
        if place_pattern_line > 0:
            pattern_lines[line, 0] = color
            pattern_lines[line, 1] += place_pattern_line

        # Put in floor line
        floor_line = player_board.floor_line
        for index, tile in place_floor_line:
            floor_line[index] = tile
            # Player took starting token
            if tile == 5:
                self.state.starting_marker = self.current_player

        # Discard to lid
        for tile in discard:
            if tile == 5:
                self.state.starting_marker = self.current_player
            else:
                self.state.lid[tile] += 1

    def process_players_new_round(self) -> None:
        """
        Place and discard tiles for players
        """

        for player_board in self.state.players:
            pattern_lines = player_board.pattern_lines
            wall = player_board.wall

            for line in range(5):
                max_tiles = max_tiles_for_line(line)
                color = pattern_lines[line, 0]
                current_tiles = pattern_lines[line, 1]

                # place, score and discard if full pattern line
                if current_tiles == max_tiles:
                    # calculate score
                    round_score, _ = calc_wall_score(wall, pattern_lines, color,
                                                     line)
                    player_board.points += round_score

                    # place one tile on wall
                    wall_column = wall_color_column(color, line)
                    wall[line, wall_column] = 1

                    # place rest of tiles in lid
                    self.state.lid[color] += (current_tiles - 1)

                    # clear pattern lines
                    pattern_lines[line, 0] = 6
                    pattern_lines[line, 1] = 0

            for floor_line_idx, tile in enumerate(player_board.floor_line):
                # ignore starting marker
                if tile < 5:
                    # calculate penalty
                    penalty = PENALTIES[floor_line_idx]
                    # do not go below zero points
                    player_board.points = max(0, player_board.points - penalty)

                    # discard to lid
                    self.state.lid[tile] += 1

            player_board.floor_line = np.full(7, 6, dtype=np.int8)

    def process_board_new_round(self) -> None:
        """
        Deal the tiles to factories
        """
        slots = self.state.slots

        num_factories = get_num_factories(self.num_players)
        left_to_deal = 4
        factory = 0

        # fill every factory with as much as possible
        while factory < num_factories:
            tiles_left: np.ndarray = np.sum(self.state.bag)
            to_sample = min(tiles_left, left_to_deal)

            all_tiles = []
            for color in range(TOTAL_COLORS):
                all_tiles += [color] * self.state.bag[color]

            tiles = self.random.choice(all_tiles, to_sample, replace=False)

            for color in tiles:
                slots[factory + 1, color] += 1
                self.state.bag[color] -= 1
                left_to_deal -= 1

            # move to next if factory was filled
            if left_to_deal == 0:
                factory += 1
                left_to_deal = 4

            # bag is empty
            if not np.any(self.state.bag):
                # lid is also empty
                if not np.any(self.state.lid):
                    # stop dealing
                    break
                # fill bag with lid and empty lid
                self.state.bag = np.copy(self.state.lid)
                self.state.lid = np.full(TOTAL_COLORS, 0, dtype=np.int8)

        self.state.starting_marker = self.num_players

    def next_round(self) -> None:
        """
        Prepare for next round
        """

        players = self.state.players

        self.process_players_new_round()

        if is_game_over(players):
            self.game_over = True
            for player_board in players:
                bonus = calc_bonus_points(player_board.wall)
                player_board.points += bonus
            return

        self.process_board_new_round()

        # set next player to who has the starting marker
        starting_marker = self.state.starting_marker
        if starting_marker != self.num_players:
            self.current_player = starting_marker
        # reset starting marker to center
        self.state.starting_marker = self.num_players
        self.rounds_count += 1

    def action_handler(self, slot: int, color: int, line: int) -> Tuple[
        float, Dict]:

        if self.game_over:
            # Game over, return
            return 0, {}

        starting_marker = self.state.starting_marker

        player_board = self.state.players[self.current_player]
        slots = self.state.slots

        result = calc_move(
            player_board,
            slots,
            starting_marker == self.num_players,
            slot,
            color,
            line)

        if result is None:
            # Invalid action, do not update
            return 0, {}

        actions, rewards = result
        place_pattern_line, place_floor_line, discard = actions

        self.play_turn(slot, color, line,
                       place_pattern_line,
                       place_floor_line,
                       discard)

        move_round_reward, move_bonus_reward, move_round_penalty = rewards
        points = player_board.points
        move_reward = move_round_reward + move_bonus_reward - move_round_penalty
        # can not go below zero points
        move_reward_capped_loss = max(move_reward, -points)

        self.turns_count += 1
        self.current_player = (self.current_player + 1) % self.num_players

        if is_next_round(slots):
            self.next_round()

        move_info = {
            "round_reward": move_round_reward,
            "bonus_reward": move_bonus_reward,
            "round_penalty": move_round_penalty
        }

        return float(move_reward_capped_loss), move_info
