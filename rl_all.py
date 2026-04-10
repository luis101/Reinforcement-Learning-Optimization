
#%% Load required libraries:

import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import random

import torch
import torch.nn as nn
import torch.optim as optim

import yfinance as yf
import talib as ta
import quantstats as qs

from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

from sklearn.covariance import LedoitWolf
from sklearn.model_selection import train_test_split

#%% Load financial data

# Load S&P 500 data

stocks = ...

# Load stock data for the past 10 years:
start = dt.date.today() - dt.timedelta(days=365*10)
end = dt.date.today()

# Create a list to store tickers with complete data:
stocks_valid = []

# Define the expected number of business days in the date range
trading_days = pd.date_range(start=start, end=end, freq='B')

for ticker in stocks:
    data = yf.download(ticker, start=start, end=end)
    # Reindex the data to include all business days within the date range
    data = data.reindex(trading_days, method='ffill').dropna()
    # Check if the length of the data matches the number of trading days
    if len(data) == len(trading_days):
        stocks_valid.append(ticker)

print("Tickers with complete data:", stocks_valid)

tickers=stocks_valid

# Load data for valid_stocks:

stock_df = yf.download(tickers, start=start, end=end, progress=False)
returns=stock_df['Close'].pct_change().dropna()

# Add to market features:

gspc_df=yf.download("^GSPC", start=start, end=end, progress=False)

vix_df=yf.download("^VIX", start=start, end=end, progress=False)

# Calculate volatility (standard deviation of returns) for each ticker:
volatility = returns.std()*np.sqrt(252)

#%% Feature engineering

# Technical indicators

def calculate_indicators(tickers, start, end):
    macd = []
    rsi = []
    cci = []
    adx = []
    stoch = []
    willr = []
    bb_upper = []
    bb_middle = []
    bb_lower = []
    mfi = []
    ema = []
    atr = []
    sar = []
    obv = []

    for ticker in tickers:
        try:
            stock_data = yf.download(ticker, start=start, end=end, progress=False)

            close_prices = stock_data['Close'].astype(np.float64).values.flatten()
            high_prices = stock_data['High'].astype(np.float64).values.flatten()
            low_prices = stock_data['Low'].astype(np.float64).values.flatten()
            volume = stock_data['Volume'].astype(np.float64).values.flatten()

            if close_prices.ndim > 1:
                close_prices = close_prices.flatten()
            if high_prices.ndim > 1:
                high_prices = high_prices.flatten()
            if low_prices.ndim > 1:
                low_prices = low_prices.flatten()
            if volume.ndim > 1:
                volume = volume.flatten()

            macd.append(ta.MACD(close_prices)[0])
            rsi.append(ta.RSI(close_prices))
            cci.append(ta.CCI(high_prices, low_prices, close_prices))
            adx.append(ta.ADX(high_prices, low_prices, close_prices))
            stoch_k, stoch_d = ta.STOCH(high_prices, low_prices, close_prices)
            stoch.append(stoch_k)  # Assuming you want to append %K line of Stochastic
            willr.append(ta.WILLR(high_prices, low_prices, close_prices))
            bb_upper_ticker, bb_middle_ticker, bb_lower_ticker = ta.BBANDS(close_prices)
            bb_upper.append(bb_upper_ticker)
            bb_middle.append(bb_middle_ticker)
            bb_lower.append(bb_lower_ticker)
            ema.append(ta.EMA(close_prices))
            atr.append(ta.ATR(high_prices, low_prices, close_prices))
            sar.append(ta.SAR(high_prices, low_prices ))
            obv.append(ta.OBV(close_prices, volume))
            
            # Calculate the Money Flow Index (MFI)
            mfi.append(ta.MFI(high=high_prices, low=low_prices, close=close_prices, volume=volume))
            
        except Exception as e:
            print(f"Error downloading data for {ticker}: {e}")

    return np.array(macd), np.array(rsi), np.array(cci), np.array(adx), np.array(stoch), np.array(willr), np.array(bb_upper), np.array(bb_middle), np.array(bb_lower), np.array(mfi), np.array(ema), np.array(atr), np.array(sar), np.array(obv)

# Function to handle NaNs
def handle_nans(indicators, fill_value=0):
    inds_nan = np.isnan(indicators)
    if inds_nan.any():
        indicators = np.where(inds_nan, fill_value, indicators)
    return indicators

# Function to normalize indicators
def normalize_indicators(indicators):
    indicators = handle_nans(indicators)  # Handle NaNs before normalization
    if indicators.ndim == 1:
        indicators = indicators.reshape(-1, 1)  # Reshape 1D array to 2D array

    min_val = np.min(indicators, axis=1, keepdims=True)
    max_val = np.max(indicators, axis=1, keepdims=True)

    # Avoid divide-by-zero error by adding a small epsilon
    epsilon = 1e-8
    normalized = (indicators - min_val) / (max_val - min_val + epsilon)

    print(f"Normalized indicators shape: {normalized.shape}")
    return normalized

# Normalize GSPC and VIX
normalized_gspc = normalize_indicators(gspc_df['Close'].values)
normalized_vix = normalize_indicators(vix_df['Close'].values)
print(f"Normalized GSPC shape: {normalized_gspc.shape}")
print(f"Normalized VIX shape: {normalized_vix.shape}")


indicators = calculate_indicators(Tickers, start, end)

# Normalize indicators and store them in a dictionary:

normalized_indicators = {}
indicator_names = ['macd', 'rsi', 'cci', 'adx', 'stoch', 'willr', 'bb_upper', 'bb_middle', 'bb_lower', 'mfi', 'ema', 'atr', 'sar', 'obv']
for i, name in enumerate(indicator_names):
    normalized_indicators[name] = normalize_indicators(indicators[i])

# Checking shapes
for name in normalized_indicators:
    print(f"{name} shape: {normalized_indicators[name].shape}")
    
normalized_gspc=normalize_indicators(gspc_df['Close'].values)
normalized_vix=normalize_indicators(vix_df['Close'].values)

# Split the data into training and testing sets using time-based splitting (80% train, 20% test):

train_df, test_df =
train_gspc, test_gspc = 
train_vix, test_vix =

#%% Define the RL environment

# Initialize parameters for model:

D = len(tickers)

state_dim = 15 * D + 4  # (14 technical indicators + holdings)*D + GSPC + VIX + balance + portfolio value
action_dim = D * 1  # Actions: Weights between 0 and 1 for each ticker

print(f"Calculated state dimension: {state_dim}")

# Build a standard Advantage-Actor-Critic (A2C):

class A2C(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(A2C, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state):
        action_probs = torch.softmax(self.actor(state), dim=-1)
        state_value = self.critic(state)
        return action_probs, state_value

# Ensure proper initialization

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

#%% Training loop

# Hyperparameters
num_episodes = 1000 
gamma = 0.99
learning_rate = 0.001
model = A2C(state_dim, action_dim)
model.apply(init_weights)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
mse_loss = nn.MSELoss()

device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")


def train_a2c(train_df, episodes=100, gamma=0.99, epsilon=0.1, early_stop_threshold=0.001, patience=10):

    last_weights = np.zeros(D)

    total_iterations = 0

    final_holdings = None
    final_portfolio_value = None
    rewards_history = []
    portfolio_values_history = []
    best_avg_reward = -float('inf')
    patience_counter = 0

    for episode in range(episodes):
        initial_holdings = np.zeros(D)
        initial_balance = 1000000
        initial_portfolio_value = initial_balance + np.sum(train_df['Close'].iloc[0] * initial_holdings)
        current_step = 0

        holdings = initial_holdings.copy()
        balance = initial_balance
        portfolio_value = initial_portfolio_value

        episode_rewards = []

        while current_step < len(train_df) - 1:
            state_components = []

            for i, ticker in enumerate(tickers):
                for name in indicator_names:
                    state_components.append(normalized_indicators[name][i, current_step])

            state_components.extend([
                normalized_gspc[current_step, 0],
                normalized_vix[current_step, 0]
            ])

            state_components.extend(holdings)
            state_components.extend([balance, portfolio_value])

            state = torch.FloatTensor(state_components).to(device)

            action_probs, state_value = model(state)

            if torch.isnan(action_probs).any() or torch.isinf(action_probs).any():
                raise ValueError("NaNs or Infs found in action probabilities")

            if random.random() < epsilon:
                # action random real number between 0 and 1
                action = random.random()
            else:
                action = torch.argmax(action_probs).item()$

            last_weights.add(tickers[action]) = action_probs[action].item()  # Store the last action's weight for the corresponding ticker

            next_prices = train_df['Close'].iloc[current_step + 1]
            next_portfolio_value = balance + np.sum(next_prices * holdings)

            if action_desc == 'buy' and balance >= next_prices.iloc[ticker_idx]:
                holdings[ticker_idx] += 1
                balance -= next_prices.iloc[ticker_idx]
            elif action_desc == 'sell' and holdings[ticker_idx] > 0:
                holdings[ticker_idx] -= 1
                balance += next_prices.iloc[ticker_idx]

            reward = (next_portfolio_value - portfolio_value) / portfolio_value 
            episode_rewards.append(reward)
            portfolio_value = next_portfolio_value

            next_state_components = []

            for i, ticker in enumerate(Tickers):
                for name in indicator_names:
                    next_state_components.append(normalized_indicators[name][i, current_step + 1])

            next_state_components.extend([
                normalized_gspc[current_step + 1, 0],
                normalized_vix[current_step + 1, 0]
            ])

            next_state_components.extend(holdings)
            next_state_components.extend([balance, portfolio_value])

            next_state = torch.FloatTensor(next_state_components).to(device)

            _, next_state_value = model(next_state)

            advantage = reward + gamma * next_state_value.item() - state_value.item()

            critic_loss = mse_loss(state_value, torch.FloatTensor([reward + gamma * next_state_value.item() + 1e-8]).to(device))
            actor_loss = -torch.log(action_probs[action] + 1e-8) * advantage

            loss = actor_loss + critic_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            current_step += 1
            total_iterations += 1

        final_holdings = holdings
        final_portfolio_value = portfolio_value
        total_reward = np.sum(episode_rewards)
        rewards_history.append(total_reward)
        portfolio_values_history.append(portfolio_value)

        avg_reward = np.mean(rewards_history[-patience:])
        print(f"Episode {episode+1}/{episodes}, Reward: {total_reward}, Avg Reward: {avg_reward}")

        if avg_reward > best_avg_reward + early_stop_threshold:
            best_avg_reward = avg_reward
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at episode {episode+1}")
            break

    tickers_list = list(tickers_with_last_buy_or_hold)
    print(f"Tickers with 'buy' or 'hold' actions at the end of the last iteration: {tickers_with_last_buy_or_hold}")
    print("Final portfolio value:", final_portfolio_value)
    print("Final holdings:")
    for i, ticker in enumerate(Tickers):
        print(f"{ticker} holdings: {final_holdings[i]}")
    print("Total iterations:", total_iterations)

    # Calculate final weights of tickers
    # After the portfolio simulation is complete
    print("Final portfolio value:", final_portfolio_value)
    final_prices = train_df['Close'].iloc[-1]
    final_weights = (final_holdings * final_prices) / final_portfolio_value

    # Normalize the weights to ensure they sum to one
    weight_sum = sum(final_weights)
    normalized_weights = final_weights / weight_sum

    print("Normalized final weights of tickers:")
    for i, ticker in enumerate(Tickers):
        print(f"{ticker} weight: {normalized_weights.iloc[i]}")

    # Create a DataFrame to store final selected tickers and their weights
    weights_df = pd.DataFrame({
        'Ticker': Tickers,
        'Weight': normalized_weights
    })

    # Filter the DataFrame to include only the tickers with non-zero weights
    final_weights_df = weights_df[weights_df['Weight'] > 0]

    # Ensure tickers_list is consistent with the final_weights_df
    tickers_list = final_weights_df['Ticker'].tolist()

    # Reset the index to avoid any indexing issues
    final_weights_df.reset_index(drop=True, inplace=True)

    print("Final selected tickers and their weights:")
    print(final_weights_df)

    # Plot rewards history and portfolio value history
    plt.figure(figsize=(14, 7))

    plt.subplot(2, 1, 1)
    plt.plot(rewards_history, label='Total Reward')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Rewards History')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(portfolio_values_history, label='Portfolio Value')
    plt.xlabel('Episode')
    plt.ylabel('Portfolio Value')
    plt.title('Portfolio Value History')
    plt.legend()

    plt.tight_layout()
    plt.show()
    
    return final_holdings, final_portfolio_value, tickers_list, normalized_weights, final_weights_df
