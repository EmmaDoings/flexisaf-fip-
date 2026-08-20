import random


def get_user_choice(choices: set[str]) -> str:
    while True:
        user_input = input(
            "Enter your choice (rock/paper/scissors): "
        ).strip().lower()

        if user_input in choices:
            return user_input

        print("Invalid choice. Please type one of: rock, paper, scissors.")


def determine_winner(user_choice: str, computer_choice: str) -> str:
    # Rules:
    # rock beats scissors
    # scissors beats paper
    # paper beats rock
    if user_choice == computer_choice:
        return "tie"

    wins = {
        "rock": "scissors",
        "scissors": "paper",
        "paper": "rock",
    }

    return "user" if wins[user_choice] == computer_choice else "computer"


def main() -> None:
    choices = {"rock", "paper", "scissors"}

    user_choice = get_user_choice(choices)
    computer_choice = random.choice(tuple(choices))

    result = determine_winner(user_choice, computer_choice)

    print(f"You chose: {user_choice}")
    print(f"Computer chose: {computer_choice}")

    if result == "tie":
        print("Result: It's a tie!")
    elif result == "user":
        print("Result: You win!")
    else:
        print("Result: Computer wins!")


if __name__ == "__main__":
    main()

