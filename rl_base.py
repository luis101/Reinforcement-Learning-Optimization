# Load required libraries:
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import random
import datetime as dt
import yfinance as yf
import talib as ta
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from sklearn.covariance import LedoitWolf
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
import quantstats as qs


"""
Gängige Algorithmen

    DDPG (Deep Deterministic Policy Gradient): Einer der am häufigsten genutzten Algorithmen. Er ist ein Actor-Critic-Verfahren, das speziell für kontinuierliche Aktionen entwickelt wurde.
    PPO (Proximal Policy Optimization): Beliebt wegen seiner stabilen Updates und einfachen Implementierung. Er verhindert zu grosse Sprünge in der Strategie-Anpassung.
    TD3 (Twin Delayed DDPG): Eine Weiterentwicklung von DDPG, die Überoptimierung (Overestimation Bias) reduziert und dadurch stabiler in volatilen Märkten performt.
    A2C / A3C (Advantage Actor-Critic): Diese Algorithmen nutzen einen "Critic", um die Varianz der Strategie-Updates zu verringern. Sie sind einfach zu implementieren und können in verschiedenen Marktszenarien gut funktionieren.
    SAC (Soft Actor-Critic): Ein Algorithmus, der Entropie maximiert, um eine bessere Exploration zu fördern. Er ist besonders effektiv in komplexen Umgebungen mit vielen Unsicherheiten.
    Rainbow DQN: Eine Kombination mehrerer Verbesserungen des klassischen DQN, die sowohl diskrete als auch kontinuierliche Aktionen unterstützen kann. Er ist robust gegenüber verschiedenen Marktbedingungen.
"""

# Load stock data that includes pandemic period:
start = dt.date.today() - dt.timedelta(days=365*4)
end = dt.date.today()

# Select five arbitrary stocks from some volatile sectors:

banking_stocks = ['JPM', 'BAC', 'WFC', 'RY', 'HSBC']
energy_stocks = ['XOM', 'CVX', 'PCCYF', 'COP', 'SLB']
chip_stocks = ['NVDA', 'INTC', 'TSM', 'AMD', 'AVGO']
ai_stocks = ['GOOGL', 'MSFT', 'IBM', 'AMZN', 'NVDA']
consumer_stocks = ['PG', 'KO', 'NESN', 'UL', 'PEP']
utilities_stocks = ['NEE', 'SO', 'DUK', 'EXC', 'D']
blockchain_stocks = ['COIN', 'XRP', 'MSTR', 'HIVE', 'HBAR']
technology_stocks = ['AAPL', 'CSCO', 'CRM', 'ADBE', 'INTC']
healthcare_stocks = ['JNJ', 'PFE', 'MRK', 'ABBV', 'MRNA']
real_estate_stocks = ['AMT', 'PLD', 'SPG', 'O', 'EQIX']
financial_stocks = ['V', 'MA', 'GS', 'AXP', 'MS']
consumer_discretionary_stocks = ['AMZN', 'TSLA', 'NKE', 'SBUX', 'HD']
industrial_stocks = ['BA', 'GE', 'CAT', 'MMM', 'UNP']

combined_stocks = list(set(chip_stocks + ai_stocks+banking_stocks+consumer_stocks+energy_stocks+utilities_stocks+blockchain_stocks+technology_stocks+healthcare_stocks+real_estate_stocks+financial_stocks+consumer_discretionary_stocks+industrial_stocks))

print("Combined list without duplicates:", combined_stocks)

# Create a list to store tickers with complete data:
valid_stocks = []

# Define the expected number of business days in the date range
trading_days = pd.date_range(start=start, end=end, freq='B')

for ticker in combined_stocks:
    data = yf.download(ticker, start=start, end=end)
    # Reindex the data to include all business days within the date range
    data = data.reindex(trading_days, method='ffill').dropna()
    # Check if the length of the data matches the number of trading days
    if len(data) == len(trading_days):
        valid_stocks.append(ticker)

print("Tickers with complete data:", valid_stocks)

Tickers=valid_stocks
print(f" you have {len(valid_stocks)} mix stocks in your portfolio.")

# Load data for valid_stocks:

stock_df = yf.download(Tickers, start=start, end=end, progress=False)
returns=stock_df['Close'].pct_change().dropna()

# Add to market features:

gspc_df=yf.download("^GSPC", start=start, end=end, progress=False)

vix_df=yf.download("^VIX", start=start, end=end, progress=False)

# Calculate volatility (standard deviation of returns) for each ticker:
volatility = returns.std()*np.sqrt(252)

# Sort tickers by volatility in descending order:
sorted_volatility = volatility.sort_values(ascending=False)

print("Tickers sorted by volatility:")
print(sorted_volatility)

# Add technical indicators as features:

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

# Initialize parameters for model:

D = len(Tickers)

# state_dim= (14 technical indicators+ holdings)*D+ GSPC+VIX+balance+portfolio value:

state_dim = 15 * D + 4  # (14 technical indicators+ holdings)*D+ 
action_dim = D * 3  # Actions: Buy, Sell, Hold for each ticker
print(f"Calculated state dimension: {state_dim}")


# Split the data into training and testing sets:

train_df, test_df = train_test_split(stock_df, test_size=0.2, shuffle=False)
train_gspc, test_gspc = train_test_split(gspc_df, test_size=0.2, shuffle=False)
train_vix, test_vix = train_test_split(vix_df, test_size=0.2, shuffle=False)

# Build a standard Advantage-Actor-Critic (A2C):

class A2C(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(A2C, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.actor = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        action_probs = torch.softmax(self.actor(x), dim=-1)
        state_value = self.critic(x)
        return action_probs, state_value

# Ensure proper initialization
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)



# Train A2C model:

def train_a2c(train_df, episodes=100, gamma=0.99, epsilon=0.1, early_stop_threshold=0.001, patience=10):
    indicators = calculate_indicators(Tickers, start, end)
    normalized_indicators = {name: normalize_indicators(ind) for name, ind in zip(indicator_names, indicators)}

    normalized_gspc = normalize_indicators(train_gspc['Close'].values)
    normalized_vix = normalize_indicators(train_vix['Close'].values)

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = A2C(state_dim, action_dim).to(device)
    model.apply(init_weights)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    mse_loss = nn.MSELoss()

    tickers_with_last_buy_or_hold = set()
    total_iterations = 0

    final_holdings = None
    final_portfolio_value = None
    rewards_history = []
    portfolio_values_history = []
    best_avg_reward = -float('inf')
    patience_counter = 0

    for episode in range(episodes):
        initial_holdings = np.zeros(D)
        initial_balance = 100000
        initial_portfolio_value = initial_balance + np.sum(train_df['Close'].iloc[0] * initial_holdings)
        current_step = 0

        holdings = initial_holdings.copy()
        balance = initial_balance
        portfolio_value = initial_portfolio_value

        episode_rewards = []

        while current_step < len(train_df) - 1:
            state_components = []

            for i, ticker in enumerate(Tickers):
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
                action = random.randint(0, action_dim - 1)
            else:
                action = torch.argmax(action_probs).item()

            action_type = action % 3
            ticker_idx = action // 3
            action_desc = ['hold', 'buy', 'sell'][action_type]

            if action_desc in ['buy', 'hold']:
                tickers_with_last_buy_or_hold.add(Tickers[ticker_idx])
            else:
                tickers_with_last_buy_or_hold.discard(Tickers[ticker_idx])

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

# Run the function with the dataset:
final_holdings, final_portfolio_value, tickers_list, normalized_weights, final_weights_df = train_a2c(train_df)

# Define comparison portfolios:

def get_IVP(cov, **kargs):
    ivp = 1. / np.diag(cov)
    ivp /= ivp.sum()
    return ivp

def get_cluster_var(cov, cItems):
    cov_ = cov.loc[cItems, cItems]
    w_ = get_IVP(cov_).reshape(-1, 1)
    cVar = np.dot(np.dot(w_.T, cov_), w_)[0, 0]
    return cVar

def get_quasi_diag(link):
    link = link.astype(int)
    sortIx = pd.Series([link[-1, 0], link[-1, 1]])
    numItems = link[-1, 3]
    while sortIx.max() >= numItems:
        sortIx.index = range(0, sortIx.shape[0] * 2, 2)
        df0 = sortIx[sortIx >= numItems]
        i = df0.index
        j = df0.values - numItems
        sortIx[i] = link[j, 0]
        df0 = pd.Series(link[j, 1], index=i + 1)
        sortIx = pd.concat([sortIx, df0])
        sortIx = sortIx.sort_index()
        sortIx.index = range(sortIx.shape[0])
    return sortIx.tolist()

def get_rec_bipart(cov, sortIx):
    w = pd.Series(1.0, index=sortIx)  # Ensure w is of float type
    cItems = [sortIx]
    while len(cItems) > 0:
        cItems = [i[int(j):int(k)] for i in cItems for j, k in ((0, len(i) / 2), (len(i) / 2, len(i))) if len(i) > 1]
        for i in range(0, len(cItems), 2):
            cItems0 = cItems[i]
            cItems1 = cItems[i + 1]
            cVar0 = get_cluster_var(cov, cItems0)
            cVar1 = get_cluster_var(cov, cItems1)
            alpha = float(1 - cVar0 / (cVar0 + cVar1))  # Explicitly cast alpha to float
            w[cItems0] *= alpha
            w[cItems1] *= 1 - alpha
    return w



def HRP_Allocation(returns):
    cov = returns.cov()
    corr = returns.corr()
    dist = squareform(((1 - corr) / 2.)**.5)
    link = linkage(dist, 'single')
    sortIx = get_quasi_diag(link)
    sortIx = returns.columns[sortIx].tolist()
    hrp = get_rec_bipart(cov, sortIx)
    return hrp.sort_index()

  

# Function to perform Modern Portfolio Optimization (MVO):
def optimize_portfolio(returns):
    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    def portfolio_performance(weights):
        portfolio_returns = np.dot(weights, mean_returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return portfolio_returns, portfolio_volatility

    def negative_sharpe_ratio(weights, risk_free_rate=0):
        p_returns, p_volatility = portfolio_performance(weights)
        return - (p_returns - risk_free_rate) / p_volatility

    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    bounds = tuple((0, 1) for _ in range(returns.shape[1]))
    initial_guess = returns.shape[1] * [1. / returns.shape[1]]

    optimized_result = minimize(negative_sharpe_ratio, initial_guess,
                                method='SLSQP', bounds=bounds,
                                constraints=constraints)

    return optimized_result.x




# Display the head of train and test returns dataframes:
print("Train Returns:")
print(train_returns.head())

print("\nTest Returns:")
print(test_returns.head())


# Function to calculate portfolio returns:
def calculate_portfolio_returns(weights, test_returns):
    portfolio_returns = (weights * test_returns).sum(axis=1)
    cumulative_returns = (1 + portfolio_returns).cumprod()-1
    return cumulative_returns


# Align indices of returns with stock_df
returns_df = returns.reindex(stock_df.index)

# Split the data into train and test sets
train_df, test_df = train_test_split(stock_df, test_size=0.2, shuffle=False)

# Get the indices for train and test data
train_indices = train_df.index
test_indices = test_df.index

# Filter the returns dataframe using the train and test indices
train_returns = returns_df.loc[train_indices]
test_returns = returns_df.loc[test_indices]


# Filter the test returns to include only the selected tickers
selected_tickers = final_weights_df['Ticker']
test_returns_filtered = test_returns[selected_tickers]

# Convert the weights to a numpy array
drl_weights = final_weights_df['Weight'].values

# Calculate the portfolio returns on the test data
drl_portfolio_returns = test_returns_filtered.dot(drl_weights)

# Calculate cumulative returns for the portfolio
drl_cumulative_returns = (1 + drl_portfolio_returns).cumprod() - 1

# Display the portfolio returns and cumulative returns
print("Portfolio Returns on Test Data:")
print(drl_portfolio_returns.head())

print("\nCumulative Returns on Test Data:")
print(drl_cumulative_returns.head())

# MVO portfolio on test data:

# Optimize portfolio using train returns
mvo_weights = optimize_portfolio(train_returns)
#print(mvo_weights)
mvo_portfolio_returns = test_returns.dot(mvo_weights)
mvo_cumulative_returns = calculate_portfolio_returns(mvo_weights, test_returns)

# HRP portfolio on test data:

hrp_weights = HRP_Allocation(train_returns)

print("HRP train weights:\n")
print(hrp_weights)
print()

hrp_portfolio_returns = test_returns.dot(hrp_weights)
hrp_cumulative_returns = calculate_portfolio_returns(hrp_weights, test_returns)

# Create DataFrames to compare HRP , MVO and DRL  performance:

performance_df = pd.DataFrame({
    'DRL Cumulative Returns': drl_cumulative_returns,
    'MVO Cumulative Returns': mvo_cumulative_returns,
    'HRP Cumulative Returns': hrp_cumulative_returns
})

#print("Performance Comparison:\n", performance_df)

# Plot the performance:
performance_df.plot(title="DRL vs HRP vs MVO Performance")

print(" MVO last cumulative returns:\n", mvo_cumulative_returns.iloc[-1])
print(" HRP last cumulative returns:\n", hrp_cumulative_returns.iloc[-1])
print(" DRL  last cumulative returns:\n", drl_cumulative_returns.iloc[-1])


# Use quantstats to define the function to calculate performance metrics :

def calculate_performance_metrics(returns, risk_free_rate=0.0):
    # Convert returns to a pandas Series
    returns_series = pd.Series(returns).dropna()

    # Use quantstats to calculate individual performance metrics
    sharpe_ratio = qs.stats.sharpe(returns_series, rf=risk_free_rate, periods=252, annualize=True)

    # Calculation of Omega ratio
    try:
        omega_ratio = qs.stats.omega(returns_series.to_frame(),  required_return=0, rf=risk_free_rate,periods=252)  # Avoid rf parameter here to prevent the error
    except AttributeError as e:
        print(f"Error calculating Omega ratio: {e}")
        omega_ratio = None

    volatility = qs.stats.volatility(returns_series)
    max_drawdown = qs.stats.max_drawdown(returns_series)
    sortino_ratio = qs.stats.sortino(returns_series, rf=risk_free_rate, periods=252)
    cvar = qs.stats.conditional_value_at_risk(returns_series, sigma=1, confidence=.99)
    calmar_ratio=qs.stats.calmar(returns_series,prepare_returns=True )
    tail_ratio=qs.stats.tail_ratio(returns_series, cutoff=0.95, prepare_returns=True)
    risk_return=qs.stats.risk_return_ratio(returns_series)
    skew=qs.stats.skew(returns_series)
    kurtosis=qs.stats.kurtosis(returns_series)

    print("\nQuantStats Performance Metrics:")
    print(f"Sharpe Ratio: {sharpe_ratio}")
    print(f"Omega Ratio: {omega_ratio}")
    print(f"Volatility: {volatility}")
    print(f"Max Drawdown: {max_drawdown}")
    print(f"Sortino Ratio: {sortino_ratio}")
    print(f"Calmar Ratio: {calmar_ratio}")
    print(f"Tail Ratio: {tail_ratio}")
    print(f"Risk Return: {risk_return}")
    print(f"Skew: {skew}")
    print(f"Kurtosis: {kurtosis}")

risk_free_rate = 0.0419

portfolios={"MVO Portfolio": mvo_portfolio_returns,
"HRP Portfolio": hrp_portfolio_returns,
"DRL Portfolio": drl_portfolio_returns
}


# Calculate and print performance metrics for each portfolio:

for name, returns in portfolios.items():
    print(f"\nPerformance Metrics for {name}:")
    calculate_performance_metrics(returns, risk_free_rate)

Ouput analysis:

DRL= Deep Reinforcement Learning

Sharpe Ratio:
Definition: Measures risk-adjusted return. Higher values are better.
Omega Ratio
Definition: Ratio of gains to losses; values above 1 indicate more gains than losses.

Volatility:
Definition: Measures portfolio risk (standard deviation of returns). Lower is generally better if returns are positive.

Max Drawdown:
Definition: Largest peak-to-trough loss; smaller values are better.

Sortino Ratio:
Definition: Similar to the Sharpe ratio but penalizes only downside risk. Higher is better.

Calmar Ratio:
Definition: Measures return relative to maximum drawdown. Higher is better.

Tail Ratio:
Definition: Measures the likelihood of extreme positive returns compared to extreme negative returns. Higher values are better.

Risk Return Ratio:
Definition: Return per unit of risk. Higher values are better.

Skew:
Definition: Measures asymmetry of return distribution. Negative values indicate a tendency toward losses.

Kurtosis:
Definition: Measures the "tailedness" of return distribution. Higher kurtosis indicates a higher likelihood of extreme returns.

Tickers with complete data: ['AMD', 'NEE', 'NVDA', 'SLB', 'D', 'MS', 'SBUX', 'NKE', 'PG', 'SO', 'XOM', 'AXP', 'AMZN', 'ADBE', 'BA', 'PEP', 'ABBV', 'V', 'EQIX', 'PCCYF', 'HSBC', 'TSLA', 'GE', 'RY', 'UNP', 'AMT', 'MA', 'EXC', 'MMM', 'JPM', 'BAC', 'JNJ', 'TSM', 'UL', 'PFE', 'WFC', 'GOOGL', 'MSFT', 'DUK', 'MRK', 'AAPL', 'CSCO', 'COP', 'IBM', 'AVGO', 'KO', 'INTC', 'MRNA', 'CAT', 'O', 'HIVE', 'HD', 'CVX', 'GS', 'CRM', 'PLD', 'SPG', 'MSTR']
 you have 58 mix stocks in your portfolio.
Tickers sorted by volatility:
Ticker
HIVE     1.001022
MSTR     0.988629
MRNA     0.659377
TSLA     0.602654
NVDA     0.528309
AMD      0.501814
PCCYF    0.428024
INTC     0.406123
AVGO     0.392050
SLB      0.385739
BA       0.365840
TSM      0.364212
ADBE     0.361914
CRM      0.360782
AMZN     0.352761
COP      0.332212
NKE      0.328933
WFC      0.309349
GOOGL    0.308413
GE       0.306672
SBUX     0.303638
AXP      0.286922
CAT      0.284576
SPG      0.280512
EQIX     0.277438
XOM      0.277358
MS       0.275628
MMM      0.272046
BAC      0.269746
NEE      0.267244
PLD      0.266127
AAPL     0.265693
AMT      0.262101
GS       0.260980
MSFT     0.260497
CVX      0.253947
HSBC     0.251686
PFE      0.249797
MA       0.244965
JPM      0.240356
HD       0.238153
V        0.227086
UNP      0.225458
D        0.222684
CSCO     0.221708
IBM      0.216740
ABBV     0.215183
EXC      0.210039
MRK      0.209236
UL       0.202947
O        0.193125
SO       0.186668
DUK      0.181443
RY       0.178050
PG       0.168397
PEP      0.164471
JNJ      0.159091
KO       0.153076

Performance Metrics for MVO Portfolio:

QuantStats Performance Metrics:
Sharpe Ratio: 0.9341008221603543
Omega Ratio: 1.1657562286379817
Volatility: 0.16928401416121625
Max Drawdown: -0.0875326071820387
Sortino Ratio: 1.3046875039755903
Calmar Ratio: 1.5612773004410159
Tail Ratio: 0.9287629324238612
Risk Return: 0.07411811800179373
Skew: -0.45551856788472694
Kurtosis: 0.488036597639554

Performance Metrics for HRP Portfolio:

QuantStats Performance Metrics:
Sharpe Ratio: 0.9136680148114422
Omega Ratio: 1.1630486188856834
Volatility: 0.09728648607362742
Max Drawdown: -0.0590252744306794
Sortino Ratio: 1.2868080709535321
Calmar Ratio: 1.5356917079496806
Tail Ratio: 1.0966492304428817
Risk Return: 0.08413556006397206
Skew: -0.5262709747010359
Kurtosis: 2.081772299321921

Performance Metrics for DRL Portfolio:

QuantStats Performance Metrics:
Sharpe Ratio: 1.5967298308877962
Omega Ratio: 1.3264001354122792
Volatility: 0.24208460472305315
Max Drawdown: -0.10680257630082846
Sortino Ratio: 2.4160420872372663
Calmar Ratio: 2.9755590169776496
Tail Ratio: 0.9903893346782576
Risk Return: 0.11126617734223067
Skew: 0.01194119465415206
Kurtosis: 3.8445446474827545

Best Overall Performance over test period: DRL Portfolio, excelling in key metrics such as Sharpe Ratio, Sortino Ratio, Calmar Ratio, and Omega Ratio.
Balanced Performance: HRP Portfolio, showing low volatility and strong drawdown management.
Poor Performance: MVO Portfolio, with negative risk-adjusted metrics and the highest volatility.

Also,the  cumulative returns plot strongly supports the quantitative analysis of the performanc