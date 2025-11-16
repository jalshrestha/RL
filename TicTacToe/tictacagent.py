"""
TicTacToe Reinforcement Learning Agent
=====================================

This module implements a Q-learning agent that learns to play TicTacToe through
reinforcement learning. The agent uses a Q-table to store state-action values
and learns optimal strategies through exploration and exploitation.

Key Components:
- TicTacToe: Game environment class
- Q-learning functions: get_Q, choose_action, update_Q
- Training function: Trains the agent through self-play
- Interactive play: Allows human vs agent gameplay

"""

import random
import numpy as np
import pickle

class TicTacToe:
    """
    TicTacToe game environment for reinforcement learning.
    
    This class represents the game board and handles game logic including:
    - Board state management
    - Move validation
    - Win condition checking
    - Game termination detection
    """
    
    def __init__(self):
        """Initialize the game environment."""
        self.reset()

    def reset(self):
        """
        Reset the game to initial state.
        
        Returns:
            str: Initial board state as a string
        """
        self.board = [' ']*9  # 3x3 board represented as 1D list
        self.current_winner = None
        return self.get_state()

    def get_state(self):
        """
        Get current board state as a string representation.
        
        Returns:
            str: Board state string, e.g. 'X O  X   '
                 Empty cells are represented by spaces
        """
        return ''.join(self.board)

    def available_actions(self):
        """
        Get list of available (empty) positions on the board.
        
        Returns:
            list: Indices (0-8) of empty cells
                 Board layout: 0|1|2
                              ---+---+---
                              3|4|5
                              ---+---+---
                              6|7|8
        """
        return [i for i, v in enumerate(self.board) if v == ' ']

    def make_move(self, action, letter):
        """
        Make a move on the board.
        
        Args:
            action (int): Board position (0-8)
            letter (str): Player symbol ('X' or 'O')
            
        Returns:
            bool: True if move was successful, False if position was occupied
        """
        if self.board[action] == ' ':  # Check if position is empty
            self.board[action] = letter
            if self.is_winner(action, letter):
                self.current_winner = letter
            return True
        return False

    def is_winner(self, action, letter):
        """
        Check if the given move results in a winning condition.
        
        Args:
            action (int): The position that was just played
            letter (str): The player symbol ('X' or 'O')
            
        Returns:
            bool: True if this move creates a winning line
        """
        # Check row (horizontal line)
        row_ind = action // 3
        row = self.board[row_ind*3:(row_ind+1)*3]
        if all([s == letter for s in row]): 
            return True

        # Check column (vertical line)
        col_ind = action % 3
        column = [self.board[col_ind+i*3] for i in range(3)]
        if all([s == letter for s in column]): 
            return True

        # Check diagonals (only if action is on a diagonal)
        if action % 2 == 0:  # Diagonal positions: 0, 2, 4, 6, 8
            diag1 = [self.board[i] for i in [0,4,8]]  # Main diagonal
            diag2 = [self.board[i] for i in [2,4,6]]  # Anti-diagonal
            if all([s == letter for s in diag1]) or all([s == letter for s in diag2]):
                return True
        return False

    def is_full(self):
        """
        Check if the board is completely filled.
        
        Returns:
            bool: True if no empty spaces remain (draw condition)
        """
        return ' ' not in self.board


# Q-learning Implementation
# ========================
# Global Q-table to store state-action values
# Key: (state, action) tuple
# Value: Q-value (expected future reward)
Q = {}

def get_state_from_perspective(board, my_symbol):
    """
    Convert board state to agent's perspective.

    The agent always sees itself as 'X' and opponent as 'O'.
    This allows a single Q-table to work for both players.

    Args:
        board (list): Current board state
        my_symbol (str): The symbol of the current player ('X' or 'O')

    Returns:
        str: Board state from agent's perspective
    """
    if my_symbol == 'X':
        return ''.join(board)
    else:
        # Swap X and O so agent always sees itself as 'X'
        return ''.join(['X' if c=='O' else 'O' if c=='X' else ' ' for c in board])

def get_Q(state, action):
    """
    Retrieve Q-value for a given state-action pair.

    Args:
        state (str): Current board state
        action (int): Action taken

    Returns:
        float: Q-value for the state-action pair (0.0 if not found)
    """
    return Q.get((state, action), 0.0)


def choose_action(state, available_actions, epsilon):
    """
    Choose an action using epsilon-greedy policy.
    
    With probability epsilon: explore (random action)
    With probability (1-epsilon): exploit (best known action)
    
    Args:
        state (str): Current board state
        available_actions (list): List of valid actions
        epsilon (float): Exploration rate (0.0 = pure exploitation, 1.0 = pure exploration)
        
    Returns:
        int: Chosen action
    """
    if random.random() < epsilon:
        # Exploration: choose random action
        return random.choice(available_actions)
    
    # Exploitation: choose action with highest Q-value
    q_values = [get_Q(state, a) for a in available_actions]
    max_q = max(q_values)
    
    # If multiple actions have the same max Q-value, choose randomly among them
    return random.choice([a for a, q in zip(available_actions, q_values) if q == max_q])

def update_Q(state, action, reward, next_state, alpha, gamma, available_actions):
    """
    Update Q-value using the Q-learning update rule.
    
    Q(s,a) = Q(s,a) + α[r + γ*max_a'Q(s',a') - Q(s,a)]
    
    Where:
    - α (alpha): learning rate
    - γ (gamma): discount factor
    - r: immediate reward
    - s': next state
    - max_a'Q(s',a'): maximum Q-value for next state
    
    Args:
        state (str): Current state
        action (int): Action taken
        reward (float): Immediate reward received
        next_state (str): Next state (None if terminal)
        alpha (float): Learning rate
        gamma (float): Discount factor
        available_actions (list): Available actions in next state
    """
    old_value = get_Q(state, action)
    
    if next_state is None:
        # Terminal state: no future rewards
        next_max = 0
    else:
        # Calculate maximum Q-value for next state
        next_max = max([get_Q(next_state, a) for a in available_actions], default=0)
    
    # Q-learning update rule
    Q[(state, action)] = old_value + alpha * (reward + gamma * next_max - old_value)

def train(episodes=200000, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.9999, epsilon_min=0.1):
    """
    Train the Q-learning agent through self-play.

    Both players learn from the same Q-table but from their own perspective.
    This allows the agent to learn both offensive and defensive strategies.

    Args:
        episodes (int): Number of training games to play
        alpha (float): Learning rate - how much new info overrides old info
        gamma (float): Discount factor - importance of future rewards
        epsilon (float): Initial exploration rate (1.0 = 100% random)
        epsilon_decay (float): Rate at which exploration decreases
        epsilon_min (float): Minimum exploration rate
    """
    env = TicTacToe()

    for ep in range(episodes):
        # Start a new game
        env.reset()
        player = 'X'  # Always start with X

        # Track history of moves for both players
        history = []  # [(player, state_from_perspective, action), ...]

        while True:
            # Get available moves for current player
            available = env.available_actions()

            # Get state from current player's perspective
            state_perspective = get_state_from_perspective(env.board, player)

            # Choose action using epsilon-greedy policy
            action = choose_action(state_perspective, available, epsilon)

            # Store this move in history
            history.append((player, state_perspective, action))

            # Make the move
            env.make_move(action, player)

            # Check game outcome
            if env.current_winner == player:
                # Current player won!
                # Give +1 reward to winner, -1 to loser
                for p, s, a in history:
                    if p == player:
                        # Winner's moves get positive reward
                        reward = 1
                        update_Q(s, a, reward, None, alpha, gamma, [])
                    else:
                        # Loser's moves get negative reward
                        reward = -1
                        update_Q(s, a, reward, None, alpha, gamma, [])
                break

            elif env.is_full():
                # Game ended in a draw
                # Small positive reward for draws (better than losing)
                for p, s, a in history:
                    reward = 0.5  # Slight reward for drawing
                    update_Q(s, a, reward, None, alpha, gamma, [])
                break

            else:
                # Game continues - switch to other player
                player = 'O' if player == 'X' else 'X'

        # Gradually reduce exploration (epsilon decay)
        # This allows the agent to exploit learned knowledge more over time
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # Progress indicator
        if (ep + 1) % 100000 == 0:
            print(f"Completed {ep + 1}/{episodes} episodes... (epsilon: {epsilon:.4f})")

    print("Training finished.")

def play_against_agent():
    """
    Interactive gameplay between human and trained agent.

    Allows a human player to play against the trained Q-learning agent.
    The human can choose to be either 'X' or 'O', and the agent will
    play as the opposite symbol.

    The agent uses pure exploitation (epsilon=0.0) during gameplay,
    meaning it always chooses the action with the highest Q-value.
    """
    while True:
        env = TicTacToe()
        env.reset()

        # Let human choose who starts
        print("\nWho should start?")
        print("1. You (Human)")
        print("2. Agent")
        starter_choice = input("Enter choice (1/2): ").strip()

        if starter_choice == '1':
            # Human starts (plays as X)
            human = 'X'
            agent = 'O'
        elif starter_choice == '2':
            # Agent starts (plays as X)
            human = 'O'
            agent = 'X'
        else:
            print("Invalid choice! Defaulting to human starting first.")
            human = 'X'
            agent = 'O'

        print(f"\n{human} = You, {agent} = Agent")
        print("\nBoard positions:")
        print("0|1|2")
        print("-----")
        print("3|4|5")
        print("-----")
        print("6|7|8\n")

        # Game loop
        while True:
            # Display current board state
            print(f"\nBoard: {env.board[0:3]}\n       {env.board[3:6]}\n       {env.board[6:9]}")

            # Check for game end conditions
            if env.is_full() or env.current_winner:
                if env.current_winner == human:
                    print("\nYou win!")
                elif env.current_winner == agent:
                    print("\nAgent wins!")
                else:
                    print("\nIt's a draw!")
                break

            # Determine whose turn it is based on board state
            # X always goes first, count moves to determine turn
            x_count = env.board.count('X')
            o_count = env.board.count('O')

            # It's X's turn if counts are equal
            current_player = 'X' if x_count == o_count else 'O'

            if current_player == human:
                # Human's turn
                try:
                    move = int(input("Your move (0-8): "))
                    if move not in env.available_actions():
                        print("Invalid move! Try again.")
                        continue
                    env.make_move(move, human)
                except (ValueError, IndexError):
                    print("Invalid input! Please enter a number between 0-8.")
                    continue
            else:
                # Agent's turn
                state_perspective = get_state_from_perspective(env.board, agent)
                action = choose_action(state_perspective, env.available_actions(), 0.0)  # Pure greedy
                print(f"Agent chooses position {action}")
                env.make_move(action, agent)

        # Ask if user wants to play again
        play_again = input("\nDo you want to play again? (y/n): ").strip().lower()
        if play_again != 'y':
            print("\nThanks for playing! Goodbye!")
            break


def save_Q(filename="q_table.pkl"):
    """
    Save the learned Q-table to a file using pickle.
    
    Args:
        filename (str): File name to save Q-table (default: 'q_table.pkl')
    """
    with open(filename, "wb") as f:
        pickle.dump(Q, f)
    print(f"✅ Q-table saved to {filename}")

def load_Q(filename="q_table.pkl"):
    """
    Load a previously saved Q-table from file.
    
    Args:
        filename (str): File name from which to load Q-table (default: 'q_table.pkl')
    """
    global Q
    try:
        with open(filename, "rb") as f:
            Q = pickle.load(f)
        print(f"✅ Q-table loaded from {filename}")
    except FileNotFoundError:
        print("⚠️ No saved Q-table found. You need to train first.")

if __name__ == "__main__":
    print("=" * 50)
    print("TicTacToe Q-Learning Agent")
    print("=" * 50)

    # Ask user if they want to train or use pretrained model
    choice = input("\nDo you want to:\n1. Train a new agent\n2. Use pretrained agent\n\nEnter choice (1/2): ").strip()

    if choice == '1':
        # Train new agent
        print("\nTraining the Q-learning agent... (this may take several minutes)")
        episodes = int(input("Enter number of episodes (recommended: 1000000): ") or "1000000")
        train(episodes=episodes, epsilon_decay=0.99995, epsilon_min=0.01)
        save_Q()
        print("\nTraining complete! Let's play!")
    elif choice == '2':
        # Load pretrained model
        print("\nLoading pretrained Q-table...")
        load_Q()
        if not Q:
            print("No pretrained model found. Training a new one...")
            train(episodes=1000000, epsilon_decay=0.99995, epsilon_min=0.01)
            save_Q()
    else:
        print("Invalid choice. Exiting...")
        exit()

    # Play against the agent
    play_against_agent()
