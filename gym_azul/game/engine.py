from typing import Dict, List, Tuple, Optional, cast

from numpy.random import default_rng, Generator  # type: ignore

from gym_azul.constants import max_tiles_for_line, PENALTIES, \
    get_num_factories, Tile, STARTING_MARKER_CENTER, \
    TILES_PER_FACTORY, Slot, Color, Line, ColorTile
from gym_azul.game.calculations import calc_move, is_next_round, is_game_over, \
    calc_bonus_points, calc_wall_score
from gym_azul.game.move_model import PlacePattern, ActionResult, \
    PlaceFloorLine
from gym_azul.game.rules import generate_legal_actions, wall_color_column
from gym_azul.model import Action, new_state, AzulState, new_floor_line


class AzulGame:
    random: Generator
    num_players: int
    advanced: bool
    state: AzulState
    game_over: bool

    def __init__(
        self,
        num_players: int,
        seed: Optional[int] = None,
        start_player: int = 0,
        advanced: bool = False
    ) -> None:
        if seed is None:
            self.random = default_rng()
        else:
            self.random = default_rng(seed)

        self.num_players = num_players
        self.advanced = advanced
        self.state = new_state(num_players, start_player)
        self.game_over = False

    def seed(self, seed: int) -> int:
        """
        Sets and return the seed
        """
        self.random = default_rng(seed)
        return seed

    def reset(self, start_player: int = 0) -> None:
        self.state = new_state(self.num_players, start_player)
        self.game_over = False
        self.next_round()

    def legal_actions(self) -> List[Action]:
        return generate_legal_actions(
            self.state.slots,
            self.state.players[self.state.player].wall,
            advanced=self.advanced)

    def play_turn(
        self,
        action: Action,
        place_pattern_line: PlacePattern,
        place_floor_line: List[PlaceFloorLine],
        discard: List[Tile]
    ) -> None:
        """
        Modify board with action
        """
        slot, color, line, column = action
        place_line, place_color, place_amount = place_pattern_line
        slots = self.state.slots
        player_state = self.state.players[self.state.player]

        # Draw from slot
        slots[slot][color] = 0
        if slot != Slot.CENTER:
            # move rest to center if from factory
            for move_color in Color:
                slots[Slot.CENTER][move_color] += slots[slot][move_color]
                slots[slot][move_color] = 0

        # Put in pattern line if putting > 0 tiles
        pattern_lines = player_state.pattern_lines
        if place_amount > 0:
            pattern_lines[place_line].color = place_color
            pattern_lines[place_line].amount += place_amount

        # Put in floor line
        floor_line = player_state.floor_line
        for column, floor_line_color in place_floor_line:
            floor_line[column] = floor_line_color
            # Player took starting token
            if floor_line_color == Tile.STARTING_TOKEN.value:
                self.state.starting_marker = self.state.player

        # Discard to lid
        for discard_tile in discard:
            if discard_tile == Tile.STARTING_TOKEN.value:
                # Player took starting token
                self.state.starting_marker = self.state.player
            else:
                self.state.lid[cast(Color, discard_tile)] += 1

    def process_players_new_round(self) -> None:
        """
        Place and discard tiles for players
        """

        for player_board in self.state.players:
            pattern_lines = player_board.pattern_lines
            wall = player_board.wall

            for line in Line:
                max_tiles = max_tiles_for_line(line)
                line_color_tile = pattern_lines[line].color
                line_amount = pattern_lines[line].amount

                # place, score and discard if full pattern line
                if line_amount == max_tiles:
                    line_color = cast(Color, line_color_tile)
                    # calculate score
                    round_score_delta, _ = calc_wall_score(
                        wall, pattern_lines, line_color, line, deep_check=False)
                    player_board.points += round_score_delta

                    # place one tile on wall
                    # TODO: advanced mode, choose a column
                    wall_column = wall_color_column(line_color, line)
                    wall[line][wall_column] = cast(ColorTile, line_color)

                    # place rest of tiles in lid
                    self.state.lid[line_color] += (line_amount - 1)

                    # clear pattern lines
                    pattern_lines[line].color = ColorTile.EMPTY
                    pattern_lines[line].amount = 0

            for column, tile in enumerate(player_board.floor_line):
                # ignore starting marker
                if tile != Tile.EMPTY:
                    # calculate penalty
                    penalty = PENALTIES[column]
                    # do not go below zero points
                    player_board.points = max(0, player_board.points - penalty)

                    # discard to lid
                    if tile != Tile.STARTING_TOKEN:
                        self.state.lid[cast(Color, tile)] += 1

            player_board.floor_line = new_floor_line()

    def process_board_new_round(self) -> None:
        """
        Deal the tiles to factories
        """
        slots = self.state.slots
        bag = self.state.bag
        lid = self.state.lid

        num_factories = get_num_factories(self.num_players)
        left_to_deal = TILES_PER_FACTORY
        factory = 0

        # fill every factory with as much as possible
        while factory < num_factories:
            tiles_left = sum(bag.values())
            to_sample = min(tiles_left, left_to_deal)

            all_tiles: List[Color] = []
            for color in Color:
                all_tiles += [color] * self.state.bag[color]

            drawn_tiles = self.random.choice(
                all_tiles, to_sample, replace=False)

            for color in drawn_tiles:
                slots[factory + 1][color] += 1
                self.state.bag[color] -= 1
                left_to_deal -= 1

            # move to next if factory was filled
            if left_to_deal == 0:
                factory += 1
                left_to_deal = TILES_PER_FACTORY

            # bag is empty
            if sum(bag.values()) == 0:
                # lid is also empty
                if sum(lid.values()) == 0:
                    # stop dealing
                    break
                # fill bag with lid and empty lid
                for lid_color, lid_count in lid.items():
                    bag[lid_color] = lid_count
                for lid_color in Color:
                    lid[lid_color] = 0

        self.state.starting_marker = STARTING_MARKER_CENTER

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
        if starting_marker != STARTING_MARKER_CENTER:
            self.state.player = starting_marker
        # reset starting marker to center
        self.state.starting_marker = STARTING_MARKER_CENTER
        self.state.round += 1

    def action_handler(self, action: Action) -> Tuple[float, Dict[str, int]]:
        if self.game_over:
            # Game over, return
            return 0, {}

        starting_marker = self.state.starting_marker

        player_board = self.state.players[self.state.player]
        slots = self.state.slots

        result = calc_move(
            player_board,
            slots,
            starting_marker == STARTING_MARKER_CENTER,
            action)

        if result is None:
            # Invalid action, do not update
            print("Invalid action!")
            return 0, {}

        move, reward = result
        place_pattern_line, place_floor_line, discard = move

        self.play_turn(action,
                       place_pattern_line,
                       place_floor_line,
                       discard)

        points = player_board.points
        round_reward, bonus_reward, round_penalty = reward
        move_reward = round_reward + bonus_reward - round_penalty
        # can not go below zero points
        move_reward_capped = max(move_reward, -points)

        self.state.turn += 1
        self.state.player = (self.state.player + 1) % self.num_players

        if is_next_round(slots):
            self.next_round()

        move_info = {
            "round_reward": round_reward,
            "bonus_reward": bonus_reward,
            "round_penalty": round_penalty
        }

        return ActionResult(float(move_reward_capped), move_info)