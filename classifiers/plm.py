"""
This module is dedicated to PanelOLS and PooledOLS classes.
It contains the PanelOLS class for fixed effects and random effects models,
and the PooledOLS class for pooled OLS models.
"""
from clm import LinearRegression, RegressionResults
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats

class FixedEffects(LinearRegression):
    def __init__(self, *args, **kwargs):
        kwargs['constant'] = False
        super().__init__(*args, **kwargs)

    def _demean(self, df, group_level, columns):
        """
        Demean columns in df by subtracting group means on index level group_level.

        Parameters:
        - df: DataFrame with MultiIndex
        - group_level: index level name or number for grouping
        - columns: list of columns to demean

        Returns:
        - DataFrame with demeaned columns only
        """
        group_means = df.groupby(level=group_level)[columns].transform('mean')
        return df[columns] - group_means

    def fit(self, data, dependent, exog, entity_effects=False, time_effects=False):
        if not isinstance(data.index, pd.MultiIndex):
            raise ValueError("Data must have a MultiIndex with entity and time levels")

        entity_level = data.index.names[0] if entity_effects else None
        time_level = data.index.names[1] if time_effects else None

        X = data[exog].copy()
        y = data[dependent].copy()

        self._check_input(X, y)

        df = X.copy()
        df['y'] = y

        if entity_level is not None:
            df[exog] = self._demean(df, entity_level, exog)
            df['y'] = self._demean(df, entity_level, ['y'])

        if time_level is not None:
            df[exog] = self._demean(df, time_level, exog)
            df['y'] = self._demean(df, time_level, ['y'])

        return super().fit(df[exog], df['y'])
