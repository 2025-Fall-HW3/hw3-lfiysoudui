"""
Package Import
"""
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import quantstats as qs
import gurobipy as gp
import warnings
import argparse
import sys

"""
Project Setup
"""
warnings.simplefilter(action="ignore", category=FutureWarning)

assets = [
    "SPY",
    "XLB",
    "XLC",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLRE",
    "XLU",
    "XLV",
    "XLY",
]

# Initialize Bdf and df
Bdf = pd.DataFrame()
for asset in assets:
    raw = yf.download(asset, start="2012-01-01", end="2024-04-01", auto_adjust = False)
    Bdf[asset] = raw['Adj Close']

df = Bdf.loc["2019-01-01":"2024-04-01"]

"""
Strategy Creation

Create your own strategy, you can add parameter but please remain "price" and "exclude" unchanged
"""


class MyPortfolio:
    """
    NOTE: You can modify the initialization function
    """

    def __init__(self, price, exclude, lookback=15, gamma=0):
        self.price = price
        self.returns = price.pct_change().fillna(0)
        self.exclude = exclude
        self.lookback = lookback
        self.gamma = gamma

    def calculate_weights(self):
        # Get the assets by excluding the specified column
        assets = self.price.columns[self.price.columns != self.exclude]

        # Calculate the portfolio weights
        self.portfolio_weights = pd.DataFrame(
            index=self.price.index, columns=self.price.columns
        )

        """
        TODO: Complete Task 4 Below
        """
        # Enhanced Mean-Variance optimization with strong momentum filtering
        for i in range(self.lookback + 1, len(self.price)):
            R_n = self.returns.copy()[assets].iloc[i - self.lookback : i]
            
            # Momentum filter: calculate cumulative returns over lookback period
            cum_returns = (1 + R_n).prod() - 1
            
            # Sort assets by momentum and select top performers
            top_assets = cum_returns.nlargest(6).index.tolist()  # Select top 6 assets
            
            # Further filter: only invest if cumulative return is positive
            top_assets = [asset for asset in top_assets if cum_returns[asset] > 0]
            
            if len(top_assets) > 0:
                # Optimize only on top momentum assets
                R_n_filtered = R_n[top_assets]
                weights_filtered = self.mv_opt(R_n_filtered, self.gamma)
                
                # Assign weights
                for j, asset in enumerate(top_assets):
                    self.portfolio_weights.loc[self.price.index[i], asset] = weights_filtered[j]
                
                # Set other assets to 0
                for asset in assets:
                    if asset not in top_assets:
                        self.portfolio_weights.loc[self.price.index[i], asset] = 0
            else:
                # If no positive momentum assets, use equal weight on all
                equal_weight = 1.0 / len(assets)
                for asset in assets:
                    self.portfolio_weights.loc[self.price.index[i], asset] = equal_weight
        
        # Set excluded asset weight to 0
        self.portfolio_weights[self.exclude] = 0
        """
        TODO: Complete Task 4 Above
        """

        self.portfolio_weights.ffill(inplace=True)
        self.portfolio_weights.fillna(0, inplace=True)

    def mv_opt(self, R_n, gamma):
        Sigma = R_n.cov().values
        mu = R_n.mean().values
        n = len(R_n.columns)

        with gp.Env(empty=True) as env:
            env.setParam("OutputFlag", 0)
            env.setParam("DualReductions", 0)
            env.start()
            with gp.Model(env=env, name="portfolio") as model:
                # Initialize decision variables: portfolio weights
                w = model.addMVar(n, name="w", lb=0, ub=1)
                
                # Mean-Variance objective: maximize return - (gamma/2) * variance
                # Expected return = mu^T * w
                portfolio_return = mu @ w
                
                # Variance = w^T * Sigma * w (quadratic form)
                # Note: The 1/2 factor is standard in Markowitz formulation
                portfolio_variance = w @ Sigma @ w
                
                # Objective: maximize expected return - (gamma/2) * variance
                # When gamma = 0: pure return maximization
                # When gamma > 0: trade off between return and risk
                model.setObjective(portfolio_return - 0.5 * gamma * portfolio_variance, gp.GRB.MAXIMIZE)
                
                # Constraint: weights must sum to 1 (fully invested portfolio)
                model.addConstr(w.sum() == 1, "budget")

                model.optimize()

                # Check if the status is INF_OR_UNBD (code 4)
                if model.status == gp.GRB.INF_OR_UNBD:
                    print(
                        "Model status is INF_OR_UNBD. Reoptimizing with DualReductions set to 0."
                    )
                elif model.status == gp.GRB.INFEASIBLE:
                    # Handle infeasible model
                    print("Model is infeasible.")
                elif model.status == gp.GRB.INF_OR_UNBD:
                    # Handle infeasible or unbounded model
                    print("Model is infeasible or unbounded.")

                if model.status == gp.GRB.OPTIMAL or model.status == gp.GRB.SUBOPTIMAL:
                    # Extract the solution
                    solution = []
                    for i in range(n):
                        var = model.getVarByName(f"w[{i}]")
                        solution.append(var.X)

        return solution

    def calculate_portfolio_returns(self):
        # Ensure weights are calculated
        if not hasattr(self, "portfolio_weights"):
            self.calculate_weights()

        # Calculate the portfolio returns
        self.portfolio_returns = self.returns.copy()
        assets = self.price.columns[self.price.columns != self.exclude]
        self.portfolio_returns["Portfolio"] = (
            self.portfolio_returns[assets]
            .mul(self.portfolio_weights[assets])
            .sum(axis=1)
        )

    def get_results(self):
        # Ensure portfolio returns are calculated
        if not hasattr(self, "portfolio_returns"):
            self.calculate_portfolio_returns()

        return self.portfolio_weights, self.portfolio_returns


if __name__ == "__main__":
    # Import grading system (protected file in GitHub Classroom)
    from grader_2 import AssignmentJudge
    
    parser = argparse.ArgumentParser(
        description="Introduction to Fintech Assignment 3 Part 12"
    )

    parser.add_argument(
        "--score",
        action="append",
        help="Score for assignment",
    )

    parser.add_argument(
        "--allocation",
        action="append",
        help="Allocation for asset",
    )

    parser.add_argument(
        "--performance",
        action="append",
        help="Performance for portfolio",
    )

    parser.add_argument(
        "--report", action="append", help="Report for evaluation metric"
    )

    parser.add_argument(
        "--cumulative", action="append", help="Cumulative product result"
    )

    args = parser.parse_args()

    judge = AssignmentJudge()
    
    # All grading logic is protected in grader_2.py
    judge.run_grading(args)
