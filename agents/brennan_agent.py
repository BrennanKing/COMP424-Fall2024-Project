from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from helpers import random_move, count_capture, execute_move, check_endgame, get_valid_moves


@register_agent("student_agent")
class StudentAgent(Agent):
    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "student_agent"
        self.time_limit = 2.0  # Maximum allowed time per move
        self.depth_reached = 0

    def step(self, chess_board, player, opponent):
        start_time = time.time()
        best_move = None
        alpha = float('-inf')
        beta = float('inf')

        valid_moves = get_valid_moves(chess_board, player)
        num_moves = len(valid_moves)
        if num_moves == 0:
            return None

        # No point in running any simulations if there is only one possible move
        if num_moves == 1:
            return valid_moves[0]

        # If there is an opportunity to take a corner, I want to grab it quickly (greedy)
        corners = [(0, 0), (0, chess_board.shape[1] - 1), (chess_board.shape[1] - 1, 0),
                   (chess_board.shape[1] - 1, chess_board.shape[1] - 1)]
        for move in valid_moves:
            if move in corners:
                return move

        depth = 0
        # Iterative deepening search
        # Keep going until the time limit is reached
        while time.time() - start_time <= self.time_limit - 0.05:
            depth += 1
            try:
                _, best_move = self.alpha_beta(
                    chess_board, depth, alpha, beta, True, player, opponent, start_time
                )
                self.depth_reached = depth
            except TimeoutError:
                break

        print(f"Depth reached: {self.depth_reached}, Best move: {best_move}")
        return best_move

    def alpha_beta(self, chess_board, depth, alpha, beta, maximizing_player, player, opponent, start_time):

        if time.time() - start_time >= self.time_limit - 0.05:
            raise TimeoutError("Time limit exceeded.")

        # Possible moves depend on who's turn it is
        valid_moves = get_valid_moves(chess_board, player if maximizing_player else opponent)
        if not valid_moves or depth == 0:
            # Don't return a move for leaf nodes
            eval_score = self.evaluate(chess_board, player, opponent)
            return eval_score, None

        best_move = None
        if maximizing_player:
            # Ordering the moves will hopefully lead to a higher pruning ratio
            ordered_moves = self.order_moves(chess_board, valid_moves, player, opponent)
            value = float('-inf')
            for move in ordered_moves:
                # Don't make changes to the original board, just make copies and then move on those
                new_board = deepcopy(chess_board)
                execute_move(new_board, move, player)
                score, _ = self.alpha_beta(new_board, depth - 1, alpha, beta, False, player, opponent, start_time)
                if score > value:
                    # Update the highest value if the current score is higher
                    value = score
                    best_move = move
                alpha = max(alpha, value)
                # print(alpha)
                # Prune the branch if beta is less than alpha
                if beta <= alpha:
                    break
        else:
            ordered_moves = self.order_moves(chess_board, valid_moves, opponent, player)
            value = float('inf')
            for move in ordered_moves:
                new_board = deepcopy(chess_board)
                execute_move(new_board, move, opponent)
                score, _ = self.alpha_beta(new_board, depth - 1, alpha, beta, True, player, opponent, start_time)
                if score < value:
                    value = score
                    best_move = move
                beta = min(beta, value)
                if beta <= alpha:
                    break  # Alpha cutoff

        return value, best_move

    def order_moves(self, chess_board, moves, player, opponent):
        # Use a simplified heuristic to order the moves
        # This is similar but not the same as what the evaluate function would return
        # For some reason, things go wrong when I use evaluate on these moves...
        # Something about how it causes weird pruning?
        scored_moves = []
        for move in moves:
            simulated_board = self.simulate_move(chess_board, move, player)
            corner_score = self.corner_heuristic(simulated_board, player)
            stability_score = self.stability_heuristic(simulated_board, player)
            mobility_score = -len(get_valid_moves(simulated_board, opponent))
            total_score = 15 * corner_score + 5 * stability_score + 3 * mobility_score
            scored_moves.append((move, total_score))
        scored_moves.sort(key=lambda x: x[1], reverse=True)
        return [move for move, _ in scored_moves]

    def simulate_move(self, chess_board, move, player):
        # Basically just the execute_move command but on a copy of the board
        new_board = deepcopy(chess_board)
        execute_move(new_board, move, player)
        return new_board

    def evaluate(self, chess_board, player, opponent):
        # This is where we combine all the heuristics to get a score on how good a
        # board state is for my agent

        # Board properties
        board_size = chess_board.shape[0]
        total_squares = board_size * board_size
        filled_squares = np.sum(chess_board != 0)
        empty_squares = total_squares - filled_squares

        # Determine game phase so that I can change how much each heuristic matters as the game goes on
        # This is a pretty janky way of doing it.
        # It might be better if I could have the weights update in a continuous manner
        # throughout the game (instead of changing abruptly at these semi arbitrary divisions...)
        if empty_squares > total_squares * 0.6:  # More than 60% empty -> Early game
            game_phase = "early"
        elif empty_squares > total_squares * 0.2:  # Between 20% and 60% empty -> Mid game
            game_phase = "mid"
        else:  # Less than 20% empty -> Late game
            game_phase = "late"

        # Compute heuristic scores
        # Notice that I'm always taking the score achieved by my agent and subtracting the score achieved
        # by the other agent. This is because it is all relative.
        # For example, one corner for me is a "good" thing, but not if the opponent has the other three
        corner_score = self.corner_heuristic(chess_board, player) - self.corner_heuristic(chess_board, opponent)
        stability_score = self.stability_heuristic(chess_board, player) - self.stability_heuristic(chess_board,
                                                                                                   opponent)
        mobility_score = len(get_valid_moves(chess_board, player)) - len(get_valid_moves(chess_board, opponent))
        border_score = self.border_heuristic(chess_board, player) - self.border_heuristic(chess_board, opponent)
        x_square_penalty = self.x_square_penalty(chess_board, player) - self.x_square_penalty(chess_board, opponent)

        # Dynamic weights based on game phase and board size
        # I really need to do more testing on the weight scaling.
        # Right now it is somewhat random as it is just based on my understanding of
        # how the different strategies vary in importance during gameplay.
        # It would be nice to have things change between the board sizes too

        # Mobility is more important than stability to start the game
        if game_phase == "early":
            weights = {
                "corner": 3,
                "stability": 1,
                "mobility": 2,
                "border": 1,
                "x_square": -3,
            }
        elif game_phase == "mid":
            weights = {
                "corner": 3,
                "stability": 1,
                "mobility": 1,
                "border": 1,
                "x_square": -3,
            }
        # Stability matters more than mobility at the end of the game
        else:  # Late game
            weights = {
                "corner": 2,
                "stability": 2,
                "mobility": 1,
                "border": 1,
                "x_square": -3,
            }

        # Combine scores with dynamic weights
        return (
                weights["corner"] * corner_score +
                weights["stability"] * stability_score +
                weights["mobility"] * mobility_score +
                weights["border"] * border_score +
                weights["x_square"] * x_square_penalty
        )

    def corner_heuristic(self, chess_board, player):
        # Corners are arguably the most important tile on the board, so we want to make sure
        # that the agent values game states that have its own discs in the corners.
        corners = [(0, 0), (0, chess_board.shape[1] - 1), (chess_board.shape[0] - 1, 0),
                   (chess_board.shape[0] - 1, chess_board.shape[1] - 1)]
        return sum(1 for corner in corners if chess_board[corner] == player)

    def stability_heuristic(self, chess_board, player):
        # Stability is all about securing discs that the opponent cant flip back
        stable_discs = 0
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        # Count to see is discs are piled up against other stable discs (around corners)
        for r in range(chess_board.shape[0]):
            for c in range(chess_board.shape[1]):
                if chess_board[r, c] == player:
                    is_stable = all(self.check_stable(chess_board, r, c, dr, dc, player) for dr, dc in directions)
                    stable_discs += is_stable
        return stable_discs

    def check_stable(self, chess_board, r, c, dr, dc, player):
        # Helper for stability heuristic
        board_size = chess_board.shape[0]
        while 0 <= r < board_size and 0 <= c < board_size:
            if chess_board[r, c] != player:
                return False
            r += dr
            c += dc
        return True

    def border_heuristic(self, chess_board, player):
        # To keep track of what tiles are on the border of a game board
        # (depends on the size of the board)
        board_size = chess_board.shape[0]
        edges = []

        # Add top and bottom borders, excluding corners
        edges += [(0, i) for i in range(2, board_size - 2)]
        edges += [(board_size - 1, i) for i in range(2, board_size - 2)]

        # Add left and right borders, excluding corners
        edges += [(i, 0) for i in range(2, board_size - 2)]
        edges += [(i, board_size - 1) for i in range(2, board_size - 2)]

        return sum(1 for edge in edges if chess_board[edge] == player)

    def x_square_penalty(self, chess_board, player):
        # Penalty for moves that will likely lead the opponent to scoring the corners
        board_size = chess_board.shape[0]
        penalty = 0

        # X-squares (tiles adjacent to corners)
        x_squares = {
            (0, 0): [(0, 1), (1, 0), (1, 1)],
            (0, board_size - 1): [(0, board_size - 2), (1, board_size - 1), (1, board_size - 2)],
            (board_size - 1, 0): [(board_size - 2, 0), (board_size - 1, 1), (board_size - 2, 1)],
            (board_size - 1, board_size - 1): [
                (board_size - 2, board_size - 1),
                (board_size - 1, board_size - 2),
                (board_size - 2, board_size - 2)
            ],
        }

        # Evaluate penalty for each corner and its X-squares
        for corner, adjacent_tiles in x_squares.items():
            if chess_board[corner] == 0:  # Corner not occupied by anyone
                for x_tile in adjacent_tiles:
                    if chess_board[x_tile] == player:  # Player occupies risky X-square
                        penalty += 1

        return penalty
