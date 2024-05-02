import random
import numpy as np
import time
import matplotlib.pyplot as plt

class Card:
    def __init__(self, suit, value):
        self.suit = suit
        self.value = value

    def __repr__(self):
        return f"{self.value} of {self.suit}"

class Deck:
    def __init__(self):
        suits = ["Hearts", "Diamonds", "Clubs", "Spades"]
        values = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "Jack", "Queen", "King", "Ace"]
        self.cards = [Card(suit, value) for suit in suits for value in values]
        self.shuffle()

    def shuffle(self):
        random.shuffle(self.cards)

    def deal(self):
        if len(self.cards) == 0:
            self.__init__()
        return self.cards.pop()

class Blackjack:
    def __init__(self, alpha=0.5, gamma=0.9, epsilon=1.0):
        self.deck = Deck()
        self.player_hand = []
        self.dealer_hand = []
        self.game_over = False
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = {}

    def compute_state(self):
        player_total = self.score_hand(self.player_hand)
        dealer_visible_card = self.dealer_hand[0].value if self.dealer_hand else None
        has_ace = any(card.value == 'Ace' for card in self.player_hand if self.score_hand([card]) == 11)
        return (player_total, dealer_visible_card, has_ace)

    def update_q_value(self, state, action, reward, next_state):
        current_q = self.q_table.get(state, [0, 0])[action]
        max_future_q = max(self.q_table.get(next_state, [0, 0]))
        new_q = current_q + self.alpha * (reward + self.gamma * max_future_q - current_q)
        self.q_table[state][action] = new_q

    def ai_decision(self):
        state = self.compute_state()
        if state not in self.q_table:
            self.q_table[state] = [0, 0]
        if random.random() < self.epsilon:
            action = random.choice([0, 1])
        else:
            action = np.argmax(self.q_table[state])
        self.epsilon *= 0.995
        print(f"\nAI's current state: {state}, Q-values: {self.q_table[state]}, Action taken: {'Hit' if action == 0 else 'Stand'}, Epsilon: {self.epsilon}")
        return action

    def deal_initial_cards(self):
        self.player_hand = [self.deck.deal(), self.deck.deal()]
        self.dealer_hand = [self.deck.deal(), self.deck.deal()]

    def score_hand(self, hand):
        score = 0
        ace_count = 0
        for card in hand:
            if card.value in ["Jack", "Queen", "King"]:
                score += 10
            elif card.value == "Ace":
                ace_count += 1
                score += 11
            else:
                score += int(card.value)
        while score > 21 and ace_count:
            score -= 10
            ace_count -= 1
        return score

    def player_hit(self):
        self.player_hand.append(self.deck.deal())
        if self.score_hand(self.player_hand) > 21:
            self.game_over = True

    def dealer_turn(self):
        while self.score_hand(self.dealer_hand) < 17:
            self.dealer_hand.append(self.deck.deal())
        self.game_over = True

    def get_winner(self):
        player_score = self.score_hand(self.player_hand)
        dealer_score = self.score_hand(self.dealer_hand)
        print(f"Dealer's final hand: {self.dealer_hand} with a total of {dealer_score}")
        if player_score > 21:
            return -10, "Player busts! Dealer wins."  # Increased penalty for busting
        elif dealer_score > 21 or player_score > dealer_score:
            return 2, "Player wins!"  # Positive reward for winning
        elif player_score < dealer_score:
            return -1, "Dealer wins."  # Lesser penalty for losing without busting
        else:
            return 0, "It's a tie."  # Neutral outcome

def main():
    game = Blackjack()
    auto_play = False
    rounds_to_play = 0
    round_count = 0
    total_wins = 0
    total_losses = 0
    total_ties = 0
    win_rates = []

    print("****************************************************")
    print("*               Jaren's Blackjack AI               *")
    print("*                (Machine Learning)                *")
    print("****************************************************")
    print("This program runs a simplified version of Blackjack and demonstrates Machine Learning.")
    print("Here's how it works:")
    print("\n- You guide and watch an AI playing Blackjack against a computer dealer.")
    print("  The AI's only options are to hit or stand. The dealer follows casino rules.")
    print("- The AI starts off by making random decisions, but will learn to play")
    print("  better over time using reinforcement learning techniques.")
    print("- The AI makes decisions based on Q-values, which represent the")
    print("  expected rewards of taking specific actions in certain game situations.")
    print("- The AI uses the epsilon-greedy strategy, which means it starts off more likely")
    print("  to make random decisions (explore) than it is to use learned strategies (exploit).")
    print("- Exploration means the AI tries random actions to discover potentially better strategies.")
    print("- Exploitation means the AI relies on what strategies have worked to maximize rewards.")
    print("- Epsilon represents the percentage likelihood that the AI will explore/exploit. In")
    print("  this case, epsilon decays at a rate of 0.995 each round - so it explores less over time.")
    print("- Statistically speaking, we always expect to see a low total win percentage after the first")
    print("  50 or so games, gradually increasing and averaging out to a cap of about 40.5% as the")
    print("  number of games played goes to the tens of thousands. The house always wins in the long run.")
    print("\n\n'y' to play a hand, 'n' to stop, 'auto' to auto-play several hands quickly (recommended), or 'plot' to plot a graph.")
    print("****************************************************\n")

    while True:
        if not auto_play:
            continue_playing = input("\nPlay a hand? (y/n/auto/plot): ").lower()
            if continue_playing == 'n':
                break
            elif continue_playing == 'y':
                pass
            elif continue_playing == 'plot':
                if win_rates:
                    plt.figure(figsize=(12, 8))
                    plt.plot(win_rates, label='Win Rate %', marker='o', markersize=1)
                    plt.xlabel('Round Number')
                    plt.ylabel('Win Rate %')
                    plt.title('AI Win Rate Over Time')
                    plt.legend()
                    plt.grid(True)
                    plt.show()
                else:
                    print("No data to plot yet.")
                continue
            elif continue_playing == 'auto':
                rounds_input = input("Enter the number of rounds for auto-play (10-5000): ")
                try:
                    rounds_to_play = int(rounds_input)
                    if 10 <= rounds_to_play <= 5000:
                        auto_play = True
                        initial_rounds = rounds_to_play
                    else:
                        print("Invalid number of rounds. Please enter a number between 10 and 5000.")
                except ValueError:
                    print("Invalid input. Please enter a number.")
                continue
            else:
                print("Invalid input. Please use 'y', 'n', 'auto', or 'plot'.")
                continue
        else:
            if rounds_to_play > 0:
                rounds_to_play -= 1
                time.sleep(5 / max(initial_rounds, 1))
            else:
                auto_play = False
                continue

        round_count += 1
        print("\n-----------------------------------")
        print(f"Round {round_count}:")
        game.deal_initial_cards()
        print(f"Player's initial hand: {game.player_hand} with a total of {game.score_hand(game.player_hand)}")
        print(f"Dealer's visible card: {game.dealer_hand[0]}")

        prev_state = game.compute_state()
        while not game.game_over:
            action = game.ai_decision()
            if action == 0:
                game.player_hit()
                print(f"Player hits. Hand: {game.player_hand} with a total of {game.score_hand(game.player_hand)}")
                if game.score_hand(game.player_hand) > 21:
                    print("Player busts!")
                    break
            elif action == 1:
                game.dealer_turn()
                print("Player stands.")

        next_state = game.compute_state()
        reward, result_string = game.get_winner()
        print(result_string)

        if reward == 2:
            total_wins += 1
        elif reward == -1 or reward == -10:
            total_losses += 1
        else:
            total_ties += 1

        if total_wins + total_losses > 0:
            win_percentage = (total_wins / (total_wins + total_losses)) * 100
        else:
            win_percentage = 0

        win_rates.append(win_percentage)

        print(f"\nEnd of Round {round_count} stats:")
        print(f"Total Wins: {total_wins}, Total Losses: {total_losses}, Total Ties: {total_ties}")
        print(f"Win Percentage (excluding ties): {win_percentage:.2f}%")

        game.update_q_value(prev_state, action, reward, next_state)
        game.game_over = False

    print("\nGoodbye!")

if __name__ == "__main__":
    main()