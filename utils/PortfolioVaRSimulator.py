# !pip install mgarch
import numpy as np
import yfinance as yf
import pandas as pd
from pandas.tseries.offsets import BDay
from typing import Dict
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
import seaborn as sb

class PortfolioVaRSimulator:
    def __init__(self, weights:Dict[str,float], start_date, n_sim=10, alpha=0.05,lookback:int=252):
        self.tickers = list(weights.keys())
        self.weights = np.array(list(weights.values()))
        self.lookback = lookback
        self.start_date = start_date
        self.n_sim = n_sim
        self.alpha = alpha
    
    def SimMultiGBMexact_av(self,n_sim, S0, mu, Sigma, Deltat=1/252, T=1):
        m = int(T/Deltat)  # number of periods.
        p = len(S0)  # number of assets.
        print(f'Simulating for {len(S0)} assets for {m} days with {n_sim} simulations')
        #cholesky decomposition for correlated multivariate normals
        L = np.linalg.cholesky(Sigma)
        
        #the shape is now (n_sim, p, m+1) to store n_sim paths for p assets over m+1 time points.
        S = np.zeros((n_sim, p, m+1))
        S[:, :, 0] = S0 
        
        mid = n_sim//2
        for i in range(mid):
            Z = np.dot(np.random.randn(m, p), L.T)#generate correlated random variables
            Z_anti = -Z
            for j in range(1, m+1):
                drift = (mu - 0.5 * np.diag(Sigma)) * Deltat
                diffusion1 = np.sqrt(Deltat) * Z[j-1, :]
                diffusion2 = np.sqrt(Deltat) * Z_anti[j-1, :]
                S[i, :, j] = S[i, :, j-1] * np.exp(drift + diffusion1)
                S[i+mid, :, j] = S[i+mid, :, j-1] * np.exp(drift + diffusion2)
        return S
    
    def SimMultiGBMexact(self,n_sim, S0, v, Sigma, Deltat=1/252, T=1):
        m = int(T/Deltat)  #number of periods.
        p = len(S0)  #number of assets.
        print(f'Simulating for {len(S0)} assets for {m} days')
        
        #cholesky decomposition for correlated multivariate normals
        L = np.linalg.cholesky(Sigma)
        
        #the shape is now (n_sim, p, m+1) to store n_sim paths for p assets over m+1 time points.
        S = np.zeros((n_sim, p, m+1))
        S[:, :, 0] = S0 
    
        for i in range(n_sim):
            Z = np.dot(np.random.randn(m, p), L.T)#generate correlated random variables
            for j in range(1, m+1):
                drift = (v - 0.5 * np.diag(Sigma)) * Deltat
                diffusion = np.sqrt(Deltat) * Z[j-1, :]
                S[i, :, j] = S[i, :, j-1] * np.exp(drift + diffusion)
        
        return S

    def _fetch_data_and_estimate_params(self):
        start_datetime = pd.to_datetime(self.start_date)
        historical_start_date = start_datetime - BDay(self.lookback)
        data = yf.download(self.tickers, start=historical_start_date, end=self.start_date)["Adj Close"]
        log_returns = np.log(data / data.shift(1)).dropna()
        mu = log_returns.mean().values * 252
        Sigma = log_returns.cov().values * 252
        S0 = data.iloc[-1].values
        return S0, mu, Sigma
    
    def _fetch_portfolio_historical_data(self):
        start_datetime = pd.to_datetime(self.start_date)
        historical_start_date = start_datetime - BDay(self.lookback)
        data = yf.download(self.tickers, start=historical_start_date, end=self.start_date)["Adj Close"]
        log_returns = np.log(data / data.shift(1)).dropna()
        portfolio_log_returns = (log_returns * self.weights).sum(axis=1)
        mu = portfolio_log_returns.mean() * 252
        std = portfolio_log_returns.std() * np.sqrt(252)
        standardized_returns = (portfolio_log_returns - mu*1/252) / (std* np.sqrt(1/252))

        return standardized_returns,mu,std
    
    def simulate_ged_portfolio(self):
        portfolio_log_returns,mu,std = self._fetch_portfolio_historical_data()
        print(mu,std)
        def ged_pdf(z,xi) : 
                _lambda = ((2**(-2/xi)*math.gamma(1/xi)) /  math.gamma(3/xi))**0.5
                return (xi*np.exp(-0.5*abs(z/_lambda)**xi) ) / (_lambda*2**(1+1/xi)*math.gamma(1/xi))
        def generate_ged_vectorized(xi, size=100):
            y_values = np.linspace(0, 10, 10000)
            f_values = 2*ged_pdf(y_values,xi)*np.exp(y_values)
            c = max(f_values)
            _lambda = ((2 ** (-2 / xi) * math.gamma(1 / xi)) / math.gamma(3 / xi)) ** 0.5
            z_values = np.random.exponential(scale=1, size=size)
            u_values = np.random.uniform(low=0, high=1, size=size)
            ged_values = (xi * np.exp(-0.5 * np.abs(z_values / _lambda) ** xi)) / (_lambda * 2 ** (1 + 1 / xi) * math.gamma(1 / xi))
            condition = u_values <= (2 * ged_values * np.exp(z_values)/ c)
            z_values = z_values[condition]
            v_values = np.random.uniform(low=0, high=1, size=len(z_values))
            z_values[v_values < 0.5] *= -1
            return z_values
        def estimate_xi(data):
            def neg_log_likelihood(xi):
                return -np.sum(ged_pdf(data, xi))
            result = minimize(neg_log_likelihood, x0=1, bounds=[(0.3, 5)])
            return result.x[0]
        xi = estimate_xi(portfolio_log_returns)
        print("lambda:",xi)
        Z = np.array([])
        while len(Z) < self.n_sim:
            Z = np.append(Z, generate_ged_vectorized(xi, size=self.n_sim- len(Z)))
        R = mu*1/252 + std*Z*np.sqrt(1/252)
        sorted_returns = np.sort(R)
        print(sorted_returns)
        VaR = np.percentile(sorted_returns, self.alpha * 100)
        VaR_index = int(self.alpha * len(sorted_returns))
        CVaR = np.mean(sorted_returns[:VaR_index])
        print(CVaR,VaR)
        sb.histplot(R)

        return sorted_returns

    
    def simulate_returns(self):
        S0,mu,Sigma= self._fetch_data_and_estimate_params()
        # print(S0,mu,Sigma)
        simulated_paths = self.SimMultiGBMexact_av(self.n_sim, S0, mu, Sigma)
        reshaped_weights = self.weights.reshape(1, -1, 1)
        
        initial_portfolio_value = np.sum(S0 * self.weights)
        scaled_simulated_paths = simulated_paths / initial_portfolio_value

        weighted_paths = scaled_simulated_paths * reshaped_weights
        portfolio_value_paths = np.sum(weighted_paths, axis=1)

        end_values = portfolio_value_paths[:, -1]
        portfolio_returns = end_values - 1  #since initial value is 1
        
        self.portfolio_returns = portfolio_returns
        self.portfolio_value_paths = portfolio_value_paths

        #sort returns and determine VaR
        sorted_returns = np.sort(portfolio_returns)
        VaR = np.percentile(sorted_returns, self.alpha * 100)
        self.VaR = VaR
        VaR_index = int(self.alpha * len(sorted_returns))
        CVaR = np.mean(sorted_returns[:VaR_index])
        self.CVaR = CVaR
        print(CVaR,VaR)
        return portfolio_returns, portfolio_value_paths, VaR,CVaR
    
    def plot_results(self):
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
        for path in self.portfolio_value_paths:
            ax[0].plot(path, alpha=0.4)  # alpha for transparency
        ax[0].set_title('Portfolio Value Paths')
        ax[0].set_xlabel('Time (Days)')
        ax[0].set_ylabel('Portfolio Value')

        ax[1].hist(self.portfolio_returns, bins=50, alpha=0.7, color='blue')
        ax[1].axvline(self.VaR, color='red', linestyle='dashed', linewidth=2)
        ax[1].annotate('VaR', xy=(self.VaR, 0), xytext=(-self.VaR, 50), arrowprops=dict(facecolor='black', shrink=0.05))
        ax[1].set_title('Histogram of Portfolio Returns')
        ax[1].set_xlabel('Returns')
        ax[1].set_ylabel('Frequency')
        plt.tight_layout()
        plt.show()
    def calculate_path_dependent_metrics(self, threshold_value):
        num_paths = self.portfolio_value_paths.shape[0]
        num_timepoints = self.portfolio_value_paths.shape[1]
        cummax = np.maximum.accumulate(self.portfolio_value_paths, axis=1)
        drawdown = (cummax - self.portfolio_value_paths) / cummax
        max_drawdown = drawdown.max(axis=1)
        average_max_drawdown = np.mean(max_drawdown)

        above_threshold_count = np.sum(np.all(self.portfolio_value_paths >= threshold_value, axis=1))

        end_values = self.portfolio_value_paths[:, -1]
        average_end_value = np.mean(end_values)
        std_dev_end_value = np.std(end_values)

        changes = np.diff(self.portfolio_value_paths, axis=1)
        positive_changes = np.where(changes > 0, changes, 0)
        negative_changes = np.where(changes < 0, changes, 0)
        avg_upward_movement = np.mean(positive_changes)
        avg_downward_movement = np.mean(negative_changes)
        positive_return_count = np.sum(end_values > self.portfolio_value_paths[:, 0])
        metrics = {
            "Average Maximum Drawdown": average_max_drawdown,
            "Paths Above Threshold Value": above_threshold_count,
            "Average End Value": average_end_value,
            "Standard Deviation of End Value": std_dev_end_value,
            "Average Upward Movement": avg_upward_movement,
            "Average Downward Movement": avg_downward_movement,
            "Paths with Positive Returns": positive_return_count
        }

        return pd.DataFrame(metrics, index=[0])
    
# tickers = tickers
# start_date = '2023-06-01'
# weights = weights
# simulator = PortfolioVaRSimulator(weights, start_date,n_sim=1000)
# simulator.simulate_returns()
# simulator.plot_results()
