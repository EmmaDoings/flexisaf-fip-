import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd


# ======================
# Part 1: Tic-Tac-Toe AI
# ======================

BOARD_SIZE = 3
EMPTY = " "
HUMAN = "X"
AI = "O"


def create_board() -> List[List[str]]:
    return [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]


def print_board(board: List[List[str]]) -> None:
    for r in range(BOARD_SIZE):
        row = " | ".join(board[r])
        print(" " + row)
        if r < BOARD_SIZE - 1:
            print("---+---+---")


def check_winner(board: List[List[str]]) -> Optional[str]:
    # Rows
    for r in range(BOARD_SIZE):
        if board[r][0] != EMPTY and board[r][0] == board[r][1] == board[r][2]:
            return board[r][0]

    # Columns
    for c in range(BOARD_SIZE):
        if board[0][c] != EMPTY and board[0][c] == board[1][c] == board[2][c]:
            return board[0][c]

    # Diagonals
    if board[0][0] != EMPTY and board[0][0] == board[1][1] == board[2][2]:
        return board[0][0]
    if board[0][2] != EMPTY and board[0][2] == board[1][1] == board[2][0]:
        return board[0][2]

    return None


def is_tie(board: List[List[str]]) -> bool:
    return all(board[r][c] != EMPTY for r in range(BOARD_SIZE) for c in range(BOARD_SIZE)) and check_winner(board) is None


def board_to_key(board: List[List[str]]) -> str:
    # Stable encoding for dictionary keys
    return "".join(board[r][c] for r in range(BOARD_SIZE) for c in range(BOARD_SIZE))


def available_moves(board: List[List[str]]) -> List[Tuple[int, int]]:
    return [(r, c) for r in range(BOARD_SIZE) for c in range(BOARD_SIZE) if board[r][c] == EMPTY]


@dataclass
class QLearner:
    # Simple Q-learning for state-action values.
    alpha: float = 0.4
    gamma: float = 0.9
    epsilon: float = 0.2

    def __post_init__(self) -> None:
        # Q[state][action] where action is an index 0..8
        self.Q = {}  # type: dict[str, dict[int, float]]

    def get_q(self, state_key: str, action: int) -> float:
        return self.Q.get(state_key, {}).get(action, 0.0)

    def set_q(self, state_key: str, action: int, value: float) -> None:
        if state_key not in self.Q:
            self.Q[state_key] = {}
        self.Q[state_key][action] = value

    def choose_action(self, board: List[List[str]]) -> int:
        state_key = board_to_key(board)
        moves = available_moves(board)
        if not moves:
            raise ValueError("No moves available")

        # ε-greedy
        if random.random() < self.epsilon:
            r, c = random.choice(moves)
            return r * BOARD_SIZE + c

        # Choose best known action
        best_action = None
        best_q = float("-inf")
        for r, c in moves:
            a = r * BOARD_SIZE + c
            q = self.get_q(state_key, a)
            if q > best_q:
                best_q = q
                best_action = a

        return best_action if best_action is not None else (moves[0][0] * BOARD_SIZE + moves[0][1])

    def update(
        self,
        state_key: str,
        action: int,
        reward: float,
        next_state_key: Optional[str],
        next_board: Optional[List[List[str]]],
        done: bool,
    ) -> None:
        current_q = self.get_q(state_key, action)

        if done or next_state_key is None or next_board is None:
            target = reward
        else:
            # Max over next valid actions
            next_moves = available_moves(next_board)
            if not next_moves:
                target = reward
            else:
                max_next_q = max(self.get_q(next_state_key, r * BOARD_SIZE + c) for r, c in next_moves)
                target = reward + self.gamma * max_next_q

        new_q = current_q + self.alpha * (target - current_q)
        self.set_q(state_key, action, new_q)


def train_agent(episodes: int = 5000) -> QLearner:
    agent = QLearner()

    for _ in range(episodes):
        board = create_board()
        # Randomly decide who starts to improve robustness.
        turn = random.choice([HUMAN, AI])

        while True:
            winner = check_winner(board)
            if winner is not None or is_tie(board):
                break

            if turn == AI:
                state_key = board_to_key(board)
                action = agent.choose_action(board)
                r, c = divmod(action, BOARD_SIZE)
                board[r][c] = AI

                winner = check_winner(board)
                done = winner is not None or is_tie(board)

                # Reward shaping
                if winner == AI:
                    reward = 1.0
                    next_state_key = None
                    next_board = None
                elif winner == HUMAN:
                    reward = -1.0
                    next_state_key = None
                    next_board = None
                elif is_tie(board):
                    reward = 0.2
                    next_state_key = None
                    next_board = None
                else:
                    reward = 0.0
                    next_state_key = board_to_key(board)
                    next_board = board

                agent.update(
                    state_key=state_key,
                    action=action,
                    reward=reward,
                    next_state_key=next_state_key,
                    next_board=next_board,
                    done=done,
                )

                turn = HUMAN
            else:
                # Human during training plays random moves
                moves = available_moves(board)
                r, c = random.choice(moves)
                board[r][c] = HUMAN
                turn = AI

    # Reduce exploration for play
    agent.epsilon = 0.05
    return agent


def get_human_move(board: List[List[str]]) -> Tuple[int, int]:
    while True:
        raw = input("Enter your move as row,col (1-3,1-3): ").strip()
        try:
            row_str, col_str = raw.split(",")
            r = int(row_str) - 1
            c = int(col_str) - 1
            if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and board[r][c] == EMPTY:
                return r, c
            print("Invalid move. Cell must be empty and within 1-3.")
        except Exception:
            print("Invalid input format. Example: 2,3")


def main_game_loop(agent: QLearner) -> None:
    board = create_board()
    print("Tic-Tac-Toe: You are X, Computer is O")
    print_board(board)

    turn = HUMAN
    while True:
        winner = check_winner(board)
        if winner is not None:
            print_board(board)
            if winner == HUMAN:
                print("You win!")
            else:
                print("Computer wins!")
            return
        if is_tie(board):
            print_board(board)
            print("It's a tie!")
            return

        if turn == HUMAN:
            r, c = get_human_move(board)
            board[r][c] = HUMAN
            print_board(board)
            turn = AI
        else:
            action = agent.choose_action(board)
            r, c = divmod(action, BOARD_SIZE)
            board[r][c] = AI
            print(f"Computer moves at row {r+1}, col {c+1}")
            print_board(board)
            turn = HUMAN


def main() -> None:
    # Train quickly (can adjust episodes).
    agent = train_agent(episodes=4000)
    main_game_loop(agent)


# ============================
# Part 2: Matplotlib Exercises
# ============================

DATA_URL = "https://pynative.com/wp-content/uploads/2019/01/company_sales_data.csv"


def load_sales_data(url: str = DATA_URL) -> pd.DataFrame:
    # pandas reads remote CSVs directly
    df = pd.read_csv(url)
    return df


def exercise_1_total_profit_line_plot(df: pd.DataFrame) -> None:
    # Total profit for each month
    df["Total Profit"] = df["Profit"]  # ensure column exists conceptually

    # The dataset has columns: Month, Sales, Expense, Profit
    # but the task says "Total profit of all months".
    # So plot Profit per Month.
    months = df["Month"]
    profits = df["Profit"]

    plt.figure(figsize=(10, 5))
    plt.plot(months, profits, marker="o")
    plt.title("Total Profit of All Months")
    plt.xlabel("Month")
    plt.ylabel("Profit")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def exercise_2_subplot_bathing_soap_facewash(df: pd.DataFrame) -> None:
    # Dataset columns: Bathing soap, Facewash, Month
    plt.figure(figsize=(12, 5))

    # Subplot 1: Bathing soap
    plt.subplot(1, 2, 1)
    plt.plot(df["Month"], df["Bathing soap"], marker="o", color="#1f77b4")
    plt.title("Bathing Soap Sales")
    plt.xlabel("Month")
    plt.ylabel("Sales")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.xticks(rotation=45)

    # Subplot 2: Facewash
    plt.subplot(1, 2, 2)
    plt.plot(df["Month"], df["Facewash"], marker="o", color="#ff7f0e")
    plt.title("Facewash Sales")
    plt.xlabel("Month")
    plt.ylabel("Sales")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Simple menu so both tasks can be run from one file.
    print("Choose an option:")
    print("1) Train AI + play Tic-Tac-Toe")
    print("2) Matplotlib Exercise 1 (line plot total profit)")
    print("3) Matplotlib Exercise 2 (subplot for Bathing soap & Facewash)")

    choice = input("Enter 1, 2, or 3: ").strip()
    df = None

    if choice == "1":
        main()
    elif choice in {"2", "3"}:
        df = load_sales_data()
        if choice == "2":
            exercise_1_total_profit_line_plot(df)
        else:
            exercise_2_subplot_bathing_soap_facewash(df)
    else:
        print("Invalid choice. Exiting.")

