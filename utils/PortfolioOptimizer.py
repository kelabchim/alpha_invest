from typing import Dict, List, Union
import numpy as np
import pandas as pd
import yfinance as yf
# from pypfopt.efficient_frontier import EfficientFrontier
# from pypfopt import risk_models
# from pypfopt import expected_returns
np.set_printoptions(precision=10, suppress=True)
from concurrent.futures import ThreadPoolExecutor



class PortfolioOptimizer:
    def __init__(self, tickers, start, end):
        """
        Initialize the PortfolioSimulator with data from Yahoo Finance.
        """
        self.tickers = tickers
        self.historical_prices = self._fetch_data(tickers, start, end)
        self.daily_returns = self.historical_prices.pct_change().fillna(0)
    
    def _fetch_data(self, tickers, start, end):
        """
        Fetch historical adjusted close prices from Yahoo Finance.
        """
        data = yf.download(tickers, start=start, end=end)['Adj Close']
        return data
    
    # def optimize_portfolio_MVO(self):
    #     """
    #     Perform portfolio optimization targeting a specific risk level and return optimal weights and statistics.

    #     :param risk: Desired level of risk aversion.
    #     :return: Optimal weights as a pandas Series and portfolio statistics as a DataFrame.
    #     """
    #     try:
    #         mu = expected_returns.mean_historical_return(self.historical_prices)
    #         Sigma = risk_models.sample_cov(self.historical_prices)

    #         ef = EfficientFrontier(mu, Sigma)
    #         # weights = ef.max_quadratic_utility(risk_aversion=risk)
    #         weights = ef.max_sharpe()
    #         cleaned_weights = ef.clean_weights()
    #         # cleaned_weights = ef.clean_weights()

    #         # Convert weights to the format expected by calculate_stats_vectorized
    #         weights_array = np.array(list(cleaned_weights.values())).reshape(1, -1)
    #         stats = self.calculate_stats_vectorized(weights_array)

    #         weights_series = pd.Series(cleaned_weights).T
    #         stats_df = pd.DataFrame(stats, index=[0])

    #         return stats_df,weights_series
        
    #     except ValueError as e:
    #         print(f"An error occurred: {e}")
    #         return None, None
    #     except np.linalg.LinAlgError as e:
    #         print(f"Linear algebra error: {e}")
    #         return None, None

        
    def differential_evolution(self, population_size=50, mutation_factor=0.5, crossover_probability=0.5, generations=200, metric:str = 'sharpe_ratio',min_weight_constraints=None):
        """
        Perform portfolio optimization using Differential Evolution algorithm with minimum weight constraints.
        """
        # Initialize population with constrained weights
        if min_weight_constraints is None:
            min_weight_constraints = np.zeros(len(self.tickers))
        
        population = np.array([self.generate_constrained_weights(min_weight_constraints) for _ in range(population_size)])
        all_weights = []
        all_stats = []

        for _ in range(generations):
            for j in range(population_size):
                #mutation
                all_weights.append(population[j].copy())

                idxs = [idx for idx in range(population_size) if idx != j]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + mutation_factor * (b - c), 0, 1)
                mutant = self.adjust_weights_to_constraints(mutant, min_weight_constraints)

                #crossover
                trial = np.where(np.random.rand(len(self.tickers)) < crossover_probability, mutant, population[j])
                trial = self.adjust_weights_to_constraints(trial, min_weight_constraints)  #ensure trial meets constraints

                #selection
                trial_stats = self.calculate_stats(np.array([trial]))
                candidate_stats = self.calculate_stats(np.array([population[j]]))
                if self.fitness(trial_stats, metric) > self.fitness(candidate_stats, metric):
                    population[j] = trial

                #record the portfolio and its statistics
                all_stats.append(trial_stats)

        #compile data into DataFrames
        weights_df = pd.DataFrame(all_weights, columns=self.tickers)
        stats_df = pd.DataFrame(all_stats)

        return stats_df, weights_df

    def generate_constrained_weights(self, min_weight_constraints):
        """
        Generate initial portfolio weights respecting the minimum weight constraints.
        """
        num_tickers = len(self.tickers)
        remaining_weight = 1 - np.sum(min_weight_constraints)
        assert remaining_weight >= 0, "Sum of minimum weights exceeds 1."

        additional_weights = np.random.uniform(size=num_tickers)
        additional_weights /= np.sum(additional_weights)
        additional_weights *= remaining_weight

        weights = min_weight_constraints + additional_weights
        return weights / np.sum(weights)

    def adjust_weights_to_constraints(self, weights, min_weight_constraints):
        """
        Adjust weights to meet minimum weight constraints and normalize them.
        """
        constrained_weights = np.maximum(weights, min_weight_constraints)
        over_allocated = np.sum(constrained_weights) - 1
        adjustment = over_allocated / len(self.tickers)
        constrained_weights -= adjustment
        constrained_weights = np.clip(constrained_weights, min_weight_constraints, None)
        return constrained_weights / np.sum(constrained_weights)


    def fitness(self, stats, metric: str):
        """
        Calculate a specific metric from the portfolio statistics based on the provided weights.
        :param metric: The specific metric to calculate (e.g., 'sharpe_ratio', 'sortino_ratio').
        :return: The calculated metric value or an error message if the metric is not found.
        """        
        if metric in stats:
            return stats[metric]
        else:
            return f"Error: The requested metric '{metric}' is not available in the calculated statistics."
        
        
    def calculate_stats(self, weights)->Dict[str,float]:
        portfolio_statistics = self.calculate_stats_vectorized(weights)
        for key in portfolio_statistics:
            portfolio_statistics[key]=portfolio_statistics.get(key).item()
        return portfolio_statistics

    
    def calculate_stats_vectorized(self, weights)->Dict[str,List[float]]:
        """
        Calculate various statistics for given portfolios using efficient matrix operations.
        Weights is a 2D array where each row represents a different portfolio.
        """
        portfolio_returns = np.dot(self.daily_returns, weights.T)

        portfolio_return = np.mean(portfolio_returns, axis=0) * 252
        portfolio_std = np.std(portfolio_returns, axis=0) * np.sqrt(252)
        sharpe = portfolio_return / portfolio_std

        negative_returns = np.where(portfolio_returns < 0, portfolio_returns, np.nan)
        negative_std = np.nanstd(negative_returns, axis=0) * np.sqrt(252)
        sortino = portfolio_return / negative_std

        cumulative_returns = np.cumprod(1 + portfolio_returns, axis=0)
        rolling_max = np.maximum.accumulate(cumulative_returns, axis=0)
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = np.nanmin(drawdown, axis=0)

        portfolio_risk = np.sqrt(np.einsum('ij,ji->i', weights, np.dot(self.daily_returns.cov() * 252, weights.T)))
        
        proportions = weights / np.sum(weights, axis=1, keepdims=True)
        epsilon = 1e-10
        shannon_entropy = -np.sum(proportions * np.log(proportions + epsilon), axis=1)
        shannon_index = shannon_entropy / np.log(weights.shape[1])
        
        portfolio_statistics = {
        "portfolio_return": portfolio_return,
        "portfolio_std": portfolio_std,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_drawdown,
        "risk_measure": portfolio_risk,
        "shannon_index": shannon_index
        }
        return portfolio_statistics
        

    def generate_weights(self,num_assets, num_simulations, method, min_weight_constraints=None):
        if method == 'uniform':
            raw_weights = np.random.uniform(size=(num_simulations, num_assets))
        elif method == 'softmax_normal':
            raw_weights = np.random.normal(size=(num_simulations, num_assets))
            raw_weights = np.exp(raw_weights) / np.sum(np.exp(raw_weights), axis=1, keepdims=True)

        if min_weight_constraints is not None:
            min_weight_constraints = np.array(min_weight_constraints)
            remaining_weight = 1 - min_weight_constraints.sum()
            scaled_weights = raw_weights * remaining_weight / raw_weights.sum(axis=1, keepdims=True)
            final_weights = scaled_weights + min_weight_constraints
            normalized_weights = final_weights / final_weights.sum(axis=1, keepdims=True)
        else:
            normalized_weights = raw_weights / raw_weights.sum(axis=1, keepdims=True)
        return normalized_weights

    def simulate_portfolios(self, num_simulations=100, method='uniform', min_weight_constraints=None)->Union[pd.DataFrame,pd.DataFrame]:
        """
        Simulate portfolio weights with minimum constraints and calculate statistics for each portfolio.
        
        :param num_simulations: Number of portfolio simulations to run.
        :param method: Method to generate weights ('uniform' or 'softmax_normal').
        :param min_weight_constraints: Minimum weight constraints for each asset.
        :return: DataFrame containing the statistics for each simulated portfolio.
        """
        num_assets = len(self.tickers)
        
        if min_weight_constraints is not None:
            if len(min_weight_constraints) != num_assets:
                raise ValueError("Length of min_weight_constraints must be equal to the number of assets.")
            if sum(min_weight_constraints) >= 1:
                raise ValueError("Sum of minimum constraints must be less than 1.")

        chunks = 10 
        chunk_size = num_simulations // chunks

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.generate_weights, num_assets, chunk_size, method, min_weight_constraints)
                    for _ in range(chunks)]
            results = [future.result() for future in futures]

        all_normalized_weights = np.vstack(results)
        stats = self.calculate_stats_vectorized(all_normalized_weights)
        stats_df = pd.DataFrame(stats)
        weights_df = pd.DataFrame(all_normalized_weights, columns=self.tickers)

        return stats_df, weights_df

# tickers = ["AAPL", "BRK-B", "UNH", "XOM", "WMT", "LLY", "AVGO", "PEP", "ADBE", 'CMCSA', 'CSCO', 'CSX', 'DAL', 'DIS', 'DISH', 'DVN', 'ET', 'F', 'FCX', 'GOOG', 'HBAN', 'HBI', 'HPE', 'HPQ', 'INTC', 'JBLU', 'KEY', 'KHC', 'KMI', 'KO', 'KOS', 'LUV', 'M', 'MARA', 'MDLZ', 'META', 'MRO', 'MSFT', 'MU', 'NCLH', 'NEE', 'NEM', 'NVDA', 'NYCB', 'OKE', 'ORCL', 'PARA', 'PCG', 'PFE', 'PLUG', 'PR', 'PYPL', 'RF', 'RIG', 'RIOT', 'RTX', 'SCHW', 'SHOP', 'SIRI', 'SLNO', 'SNAP', 'SQ', 'SWN', 'T', 'TLRY', 'TSLA', 'UEC', 'USB', 'VFC', 'VZ', 'WBD', 'WFC']
# tickers = ["AAPL", "BRK-B", "UNH", "XOM", "WMT", "LLY", "META", "AVGO", "PEP", "ADBE"]
# tickers = ["AAPL", "BRK-B", "UNH", "XOM", "WMT"]
# start_date = '2022-06-01'
# end_date = '2023-01-01'
# simulator = PortfolioOptimizer(tickers, start_date, end_date)
# stats_uniform,weights_uniform = simulator.simulate_portfolios(num_simulations=100000, method='uniform')
# stats_uniform.sort_values(by='sharpe_ratio')