"""
Module with Linear Regression models.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats


class LinearRegression:
    def __init__(
        self, lr=0.01, max_iter=2000000, method='normal',
        verbose=False, constant=False, loss_history=False,
        tolerance=1e-20, convergence_stop=True, convergence_iter=5,
        cov_type='nonrobust'
    ):
        self.__method = method
        self.__cov_type = cov_type

        self.__lr = lr
        self.__max_iter = max_iter

        self.__verbose = verbose

        self.__constant = constant

        self.__loss_history = []
        self.__loss_history_ind = loss_history or verbose

        self.__conv_stop = convergence_stop
        self.__conv_iter = convergence_iter
        self.__tolerance = tolerance

        self.__theta = None

    def __check_input(self, X, y):
        if not isinstance(X, (pd.DataFrame, np.ndarray)):
            raise ValueError(
                "X should be a pandas DataFrame or numpy ndarray.")
        if not isinstance(y, (pd.DataFrame, pd.Series, np.ndarray)):
            raise ValueError(
                "y should be a pandas DataFrame, pandas Series, or numpy ndarray.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("The number of samples in X and y must match.")
        if isinstance(X, pd.DataFrame):
            if X.select_dtypes(include=[np.number]).shape[1] != X.shape[1]:
                raise ValueError(
                    "DataFrame must contain only numeric columns.")

    def fit(self, X, y):
        self.__check_input(X, y)
        self.__columns = X.columns if isinstance(X, pd.DataFrame) else [
            f"x{i}" for i in range(X.shape[1])]
        self.__y_name = y.name if isinstance(
            y, pd.Series) else y.columns[0] if isinstance(y, pd.DataFrame) else 'y'
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, (pd.DataFrame, pd.Series)):
            y = y.values
        y = y.reshape(-1, 1)

        self.__mean = X.mean(axis=0)
        self.__std = X.std(axis=0)
        self.__std[self.__std == 0] = 1
        X_scaled = (X - self.__mean) / self.__std

        if self.__constant:
            X_scaled = np.c_[np.ones(X.shape[0]), X_scaled]
            self.__columns = ["const"] + list(self.__columns)

        if self.__method == 'normal':
            self.__theta = np.linalg.pinv(
                X_scaled.T @ X_scaled) @ X_scaled.T @ y
        else:
            self.__theta = np.zeros((X_scaled.shape[1], 1))
            prev_loss = float('inf')
            converged_iteration = 0
            for i in range(self.__max_iter):
                predictions = X_scaled @ self.__theta
                error = predictions - y
                gradients = (1 / X_scaled.shape[0]) * X_scaled.T @ error
                self.__theta -= self.__lr * gradients

                loss = np.mean(error ** 2)
                if self.__loss_history_ind:
                    self.__loss_history.append(loss)
                if self.__conv_stop and abs(prev_loss - loss) < self.__tolerance:
                    if converged_iteration >= self.__conv_iter:
                        if self.__verbose:
                            print(
                                f"Converged at iteration {i}: MSE = {loss:.4f}")
                        break
                    else:
                        converged_iteration += 1
                else:
                    converged_iteration = 0

                prev_loss = loss
                if self.__verbose and i % 100 == 0:
                    print(f"Iter {i}: MSE = {loss:.4f}")

        if self.__constant:
            beta_0 = self.__theta[0, 0]
            coefs = self.__theta[1:, 0]
        else:
            beta_0 = 0
            coefs = self.__theta[:, 0]

        original_coefs = coefs / self.__std
        original_intercept = beta_0 - np.sum(self.__mean * original_coefs)

        final_theta = np.concatenate(
            [[original_intercept], original_coefs]) if self.__constant else original_coefs
        self.__theta = final_theta.reshape(-1, 1)

        if self.__constant:
            X_transformed = np.c_[np.ones(X.shape[0]), X]
        else:
            X_transformed = X

        return RegressionResults(X_transformed, y, self.__theta, self.__loss_history, self.__columns, self.__y_name, self.__cov_type)

    def predict(self, X):
        if self.__theta is None:
            raise AttributeError("Model should be trained first!")
        if isinstance(X, pd.DataFrame):
            X = X.values
        if self.__constant:
            X = np.c_[np.ones(X.shape[0]), X]
        return (X @ self.__theta).flatten()


class RegressionResults:
    def __init__(self, X, y, theta, loss_history=None, columns=None, y_name=None, cov_type="nonrobust"):
        self.__X = X
        self.__y = y
        self.__dependent_name = y_name
        self.__theta = theta
        self.__n, self.__p = X.shape
        self.__loss_history = loss_history
        self.__columns = columns
        self.__cov_type = cov_type.lower()

        self.__y_hat = X @ theta
        self.__residuals = y - self.__y_hat
        self.__mse = np.mean(self.__residuals ** 2)
        self.__r2 = self.__compute_r2()
        self.__adj_r2 = 1 - (1 - self.__r2) * \
            (self.__n - 1) / (self.__n - self.__p)

        self.__stderr, self.__tvalues, self.__pvalues = self.__compute_inference()
        self.__aic = self.__compute_aic()
        self.__bic = self.__compute_bic()

    @property
    def n(self):
        return self.__n

    @property
    def p(self):
        return self.__p

    @property
    def columns(self):
        return self.__columns

    @property
    def params(self):
        return self.__theta.flatten().copy()

    @property
    def cov_type(self):
        return self.__cov_type

    @property
    def fittedvalues(self):
        return self.__y_hat.copy()

    @property
    def residuals(self):
        return self.__residuals.copy()

    @property
    def rsquared(self):
        return self.__r2

    @property
    def rsquared_adj(self):
        return self.__adj_r2

    @property
    def mse(self):
        return self.__mse

    @property
    def stderr(self):
        return self.__stderr.copy()

    @property
    def tvalues(self):
        return self.__tvalues.copy()

    @property
    def pvalues(self):
        return self.__pvalues.copy()

    @property
    def aic(self):
        return self.__aic

    @property
    def bic(self):
        return self.__bic

    @property
    def f_statistic(self):
        y = self.__y
        theta = self.__theta
        n, p = self.__n, self.__p
        cov_type = self.__cov_type.lower()

        has_const = "const" in self.__columns
        df_model = p - 1 if has_const else p
        df_resid = n - p

        if cov_type == "nonrobust":
            ssr = np.sum((self.__y_hat - np.mean(y)) ** 2)
            sse = np.sum(self.__residuals ** 2)
            msr = ssr / df_model
            mse = sse / df_resid
            f_stat = msr / mse
            p_value = 1 - stats.f.cdf(f_stat, df_model, df_resid)
            return f_stat, p_value

        R = np.eye(p)
        if has_const:
            R = R[1:]
            df_model = p - 1
        else:
            df_model = p

        r = np.zeros(df_model)
        beta = theta.flatten()
        cov = self.__covariance_matrix
        Rbeta = R @ beta
        RcRT = R @ cov @ R.T

        try:
            RcRT_inv = np.linalg.pinv(RcRT)
            f_stat = (Rbeta - r).T @ RcRT_inv @ (Rbeta - r) / df_model
            p_value = 1 - stats.f.cdf(f_stat, df_model, df_resid)
            return f_stat, p_value
        except np.linalg.LinAlgError:
            return np.nan, np.nan

    def conf_int(self, alpha=0.05):
        z = stats.t.ppf(1 - alpha / 2, df=self.__n - self.__p)
        lower = self.__theta.flatten() - z * self.__stderr
        upper = self.__theta.flatten() + z * self.__stderr
        return np.vstack([lower, upper]).T

    def coefficients(self):
        headers = ["coef", "std err", "t", "P>|t|", "[0.025", "0.975]"]
        ci = self.conf_int()
        rows = []
        for i in range(len(self.__theta)):
            rows.append([
                self.__theta[i, 0],
                self.__stderr[i],
                self.__tvalues[i],
                self.__pvalues[i],
                ci[i, 0],
                ci[i, 1]
            ])
        df = pd.DataFrame(rows, columns=headers, index=self.__columns)
        return df

    def __compute_r2(self):
        ss_res = np.sum(self.__residuals ** 2)
        ss_tot = np.sum((self.__y - np.mean(self.__y)) ** 2)
        return 1 - ss_res / ss_tot

    def __compute_inference(self):
        X = self.__X
        theta = self.__theta
        n, p = self.__n, self.__p
        cov_type = self.__cov_type.lower()
        e = self.__residuals
        XTX_inv = np.linalg.pinv(X.T @ X)

        if cov_type == "nonrobust":
            sigma_squared = self.__mse
            cov = sigma_squared * XTX_inv

        else:
            h = np.sum(X * (X @ XTX_inv), axis=1)
            e2 = (e ** 2).flatten()

            if cov_type == "hc0":
                scaled_e2 = e2
            elif cov_type == "hc1":
                scaled_e2 = e2 * n / (n - p)
            elif cov_type == "hc2":
                scaled_e2 = e2 / (1 - h)
            elif cov_type == "hc3":
                scaled_e2 = e2 / (1 - h) ** 2
            else:
                raise ValueError(f"Unknown cov_type: {cov_type}")

            X_weighted = X * scaled_e2[:, None]
            cov = XTX_inv @ (X.T @ X_weighted) @ XTX_inv

        std_err = np.sqrt(np.diag(cov))
        t_vals = theta.flatten() / std_err
        p_vals = 2 * (1 - stats.t.cdf(np.abs(t_vals), df=n - p))
        self.__covariance_matrix = cov
        return std_err, t_vals, p_vals

    def __compute_aic(self):
        return -2 * self.__compute_llf() + 2 * self.__p

    def __compute_bic(self):
        k = self.__X.shape[1] + 1
        return -2 * self.__compute_llf() + k * np.log(self.__n)

    def __repr__(self):
        summary_str = "OLS Regression Results\n"
        summary_str += "-" * 70 + "\n"
        summary_str += f"Number of observations: {self.__n}\n\n"
        summary_str += f"Degrees of freedom (model): {self.__p - 1 if 'const' in self.__columns else self.__p}\n"
        summary_str += f"Degrees of freedom (residuals): {self.__n - self.__p}\n\n"
        summary_str += f"Covariance Type: {self.__cov_type.upper()}\n\n"
        summary_str += f"R-squared: {self.__r2:.4f}\n"
        summary_str += f"Adj. R-squared: {self.__adj_r2:.4f}\n\n"
        summary_str += "-" * 70 + "\n"

        df_summary = self.coefficients().copy()
        summary_str += df_summary.to_string()
        summary_str += "\n" + "-" * 70

        summary_str += f"\nF-statistic{' (robust)' if self.__cov_type != 'nonrobust' else ''}: {self.f_statistic[0]:.4f}\n"
        summary_str += f"Prob ({'robust ' if self.__cov_type != 'nonrobust' else ''}F-statistic): {self.f_statistic[1]:.4g}\n\n"
        summary_str += f"AIC: {self.__aic:.4f}" + " "*5
        summary_str += f"BIC: {self.__bic:.4f}\n"
        return summary_str

    def __compute_llf(self):
        n = self.__n
        mse = self.__mse
        return -n / 2 * (np.log(2 * np.pi) + np.log(mse) + 1)

    def _repr_html_(self):
        f_stat, f_pval = self.f_statistic
        df_model = self.__p - 1 if 'const' in self.__columns else self.__p
        df_resid = self.__n - self.__p

        html = f"""
        <style>
            .ols-summary {{
                border-collapse: collapse;
                width: 100%;
            }}
            .ols-header {{
                font-size: 18px;
                font-weight: bold;
                padding: 10px 0;
            }}
            .ols-subtable {{
                margin-right: 40px;
            }}
            .ols-summary td {{
                padding: 4px 8px;
            }}
            .ols-title-row td {{
                border: none;
            }}
            .ols-flex {{
                display: flex;
                justify-content: space-between;
                align-items: flex-start;
                max-width: 800px
            }}
            .ols-summary{{
                max-width: 750px
            }}
            .ols-summary caption {{
                font-weight: bold;
            }}
        </style>

        <div class="ols-header">OLS Regression Results</div>
        <div class="ols-flex">
            <table class="ols-summary ols-subtable">
                <tr><td>Dep. Variable:</td><td>{self.__dependent_name}</td></tr>
                <tr><td>Model:</td><td>OLS</td></tr>
                <tr><td>Method:</td><td>Least Squares</td></tr>
                <tr><td>No. Observations:</td><td>{self.__n}</td></tr>
                <tr><td>Df Residuals:</td><td>{df_resid}</td></tr>
                <tr><td>Df Model:</td><td>{df_model}</td></tr>
                <tr><td>Covariance Type:</td><td>{self.__cov_type.upper()}</td></tr>
            </table>

            <table class="ols-summary ols-subtable">
                <tr><td>R-squared:</td><td>{self.__r2:.3f}</td></tr>
                <tr><td>Adj. R-squared:</td><td>{self.__adj_r2:.3f}</td></tr>
                <tr><td>F-statistic{' (robust)' if self.__cov_type != 'nonrobust' else ''}:</td><td>{f_stat:.2f}</td></tr>
                <tr><td>Prob ({'robust ' if self.__cov_type != 'nonrobust' else ''}F-statistic):</td><td>{f_pval:.3g}</td></tr>
                <tr><td>Log-Likelihood:</td><td>{self.__compute_llf():.3f}</td></tr>
                <tr><td>AIC:</td><td>{self.__aic:.0f}</td></tr>
                <tr><td>BIC:</td><td>{self.__bic:.0f}</td></tr>
            </table>
        </div>

        <div class="ols-title-row">
            <table class="ols-summary">
                <caption style="text-align:left; padding-top:20px;">Coefficients:</caption>
                {self.coefficients().to_html(float_format="%.3f", classes="ols-summary", border=0)}
            </table>
        </div>
        """
        return html

    # Plots section:
    def plot_history(self):
        """
        Plots the loss history (MSE) over iterations using seaborn.
        """
        if not self.__loss_history:
            raise ValueError(
                "Loss history is empty. Only available when using gradient descent.")

        sns.set(style="whitegrid")
        sns.lineplot(x=range(len(self.__loss_history)), y=self.__loss_history)
        plt.xlabel("Iteration")
        plt.ylabel("MSE")
        plt.title("Loss History (Gradient Descent)")
        plt.tight_layout()
        plt.show()

    def plot_actual_vs_predicted(self):
        """
        Plots actual vs predicted values (y vs y_hat).
        """
        sns.set(style="whitegrid")
        y_flat = self.__y.flatten()
        y_hat_flat = self.__y_hat.flatten()
        sns.scatterplot(x=y_flat, y=y_hat_flat, alpha=0.7)
        sns.lineplot(x=y_flat, y=y_flat, color="red", label="Ideal Fit")
        plt.xlabel("Actual y")
        plt.ylabel("Predicted y_hat")
        plt.title("Actual vs Predicted Values")
        plt.legend()
        plt.show()

    def plot_residuals_vs_fitted(self):
        """
        Generates a scatter plot of residuals against the fitted values.

        This plot is useful for assessing the assumption of homoscedasticity
        (constant variance of errors) in a linear regression model.
        Ideally, the residuals should be randomly scattered around zero with no
        discernible pattern.
        """
        sns.set(style="whitegrid")
        sns.scatterplot(x=self.__y_hat.flatten(),
                        y=self.__residuals.flatten(), alpha=0.7)
        plt.axhline(0, color="red", linestyle="--")
        plt.xlabel("Fitted values (y_hat)")
        plt.ylabel("Residuals")
        plt.title("Residuals vs Fitted")
        plt.tight_layout()
        plt.show()

    def plot_residuals_hist(self):
        """
        Generates a histogram of the residuals.

        This plot helps in assessing the assumption that the errors (and thus
        the residuals) are normally distributed. A roughly bell-shaped histogram
        suggests that this assumption might hold. The Kernel Density Estimate (KDE)
        is overlaid to provide a smoothed representation of the distribution.
        """
        sns.histplot(self.__residuals, bins=20, kde=True, legend=False)
        plt.xlabel("Residuals")
        plt.title("Histogram of Residuals")
        plt.tight_layout()
        plt.show()

    def plot_qq(self):
        """
        Generates a Quantile-Quantile (QQ) plot of the standardized residuals
        against the theoretical quantiles of a standard normal distribution.

        This plot is used to assess the normality of the residuals. If the
        residuals are normally distributed, the points in the QQ plot should
        roughly fall along the red dashed line, which represents the ideal
        fit to a normal distribution. Deviations from this line suggest
        departures from normality.
        """
        residuals_flat = self.__residuals.flatten()
        residuals_standardized = (
            residuals_flat - np.mean(residuals_flat)) / np.std(residuals_flat)
        sorted_residuals = np.sort(residuals_standardized)
        normal_quantiles = stats.norm.ppf(
            np.linspace(0.01, 0.99, len(sorted_residuals)))
        plt.figure(figsize=(8, 6))
        plt.scatter(normal_quantiles, sorted_residuals, alpha=0.7)
        plt.plot(normal_quantiles, normal_quantiles,
                 color="red", linestyle="--", label="Ideal Fit")
        plt.title("Custom QQ Plot of Residuals")
        plt.xlabel("Theoretical Quantiles (Standard Normal Distribution)")
        plt.ylabel("Sample Quantiles (Standardized Residuals)")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_diagnostics(self):
        """
        Generates a set of diagnostic plots to evaluate the residuals of the regression model. 

        The plots include:
        1. Residuals vs Fitted: Helps assess homoscedasticity (constant variance of errors).
        2. Histogram of Residuals: Helps assess the normality of residuals.
        3. QQ Plot of Residuals: Further helps assess normality by comparing residuals to the theoretical normal distribution.

        The diagnostic plots allow for an easy visual check on the assumptions of linear regression, including 
        constant variance of errors (homoscedasticity) and normality of the residuals.
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle("Residuals Diagnostics", fontsize=16)

        # Residuals vs Fitted Plot
        axes[0].scatter(self.__y_hat.flatten(),
                        self.__residuals.flatten(), alpha=0.7)
        axes[0].axhline(0, color="red", linestyle="--")
        axes[0].set_xlabel("Fitted values (y_hat)")
        axes[0].set_ylabel("Residuals")
        axes[0].set_title("Residuals vs Fitted")

        # Histogram of Residuals
        sns.histplot(self.__residuals, bins=20,
                     kde=True, legend=False, ax=axes[1])
        axes[1].set_xlabel("Residuals")
        axes[1].set_title("Histogram of Residuals")

        # QQ Plot
        residuals_flat = self.__residuals.flatten()
        residuals_standardized = (
            residuals_flat - np.mean(residuals_flat)) / np.std(residuals_flat)
        sorted_residuals = np.sort(residuals_standardized)
        normal_quantiles = stats.norm.ppf(
            np.linspace(0.01, 0.99, len(sorted_residuals)))
        axes[2].scatter(normal_quantiles, sorted_residuals, alpha=0.7)
        axes[2].plot(normal_quantiles, normal_quantiles,
                     color="red", linestyle="--", label="Ideal Fit")
        axes[2].set_title("QQ Plot of Residuals")
        axes[2].set_xlabel(
            "Theoretical Quantiles (Standard Normal Distribution)")
        axes[2].set_ylabel("Sample Quantiles (Standardized Residuals)")
        axes[2].legend()

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()


AIC = 0
BIC = 1
COMBINED = 2


def selection(X, y, criterion=AIC, model_params: dict = None):
    """
    Performs model selection based on AIC, BIC, or a combination of both.
    """
    def get_score(results):
        if criterion == AIC:
            return results.aic
        if criterion == BIC:
            return results.bic
        if criterion == COMBINED:
            return results.aic + results.bic
        raise ValueError("Invalid criterion. Use AIC, BIC, or COMBINED.")

    model = LinearRegression(
        **model_params) if model_params else LinearRegression()
    best_results = get_score(model.fit(X, y))
    best_features = X.columns.tolist() if isinstance(
        X, pd.DataFrame) else list(range(X.shape[1]))
    dropped_features = []

    while True:
        best_score = best_results
        best_feature = None
        for feature in best_features:
            features_to_test = [f for f in best_features if f != feature]
            X_test = X[features_to_test] if isinstance(
                X, pd.DataFrame) else X[:, features_to_test]
            results = LinearRegression(**model_params).fit(X_test, y)
            score = get_score(results)
            if score < best_score:
                best_score = score
                best_feature = feature

        if best_score < best_results:
            best_results = best_score
            best_features.remove(best_feature)
            dropped_features.append(best_feature)
        else:
            break

    return best_features, dropped_features


def compare(results1: 'RegressionResults', results2: 'RegressionResults', names: list[str] = ('Model1', 'Model2'), tolerance=1e-30) -> pd.DataFrame:
    """
    Compares two RegressionResults objects and returns a pandas DataFrame
    summarizing their comparison, including average difference.
    """
    comparison_data = []

    def compare_attribute(name, val1, val2):
        try:
            val1_arr = np.asarray(val1, dtype=float).flatten()
            val2_arr = np.asarray(val2, dtype=float).flatten()
            are_equal = np.allclose(val1_arr, val2_arr, atol=tolerance)
            avg_diff = np.mean(np.abs(val1_arr - val2_arr)
                               ) if not are_equal else 0
        except Exception:
            are_equal = val1 == val2
            avg_diff = 0 if are_equal else np.nan

        return {
            'Metric': name,
            names[0]: val1,
            names[1]: val2,
            'Equal': are_equal,
            'Avg. Diff': avg_diff
        }

    comparison_data.append(compare_attribute(
        'Number of observations', results1.n, results2.n))
    comparison_data.append(compare_attribute(
        'Number of parameters', results1.p, results2.p))

    if set(results1.columns) != set(results2.columns):
        comparison_data.append(compare_attribute(
            'Columns (' + names[0] + ')', results1.columns, np.nan))
        comparison_data.append(compare_attribute(
            'Columns (' + names[1] + ')', np.nan, results2.columns))
    else:
        comparison_data.append(compare_attribute(
            'Columns', results1.columns, results2.columns))

    comparison_data.append(compare_attribute(
        "Coefficients", results1.params, results2.params))
    comparison_data.append(compare_attribute(
        "Std. Errors", results1.stderr, results2.stderr))
    comparison_data.append(compare_attribute(
        "T-values", results1.tvalues, results2.tvalues))
    comparison_data.append(compare_attribute(
        "P-values", results1.pvalues, results2.pvalues))
    comparison_data.append(compare_attribute(
        "R-squared", results1.rsquared, results2.rsquared))
    comparison_data.append(compare_attribute(
        "Adj. R-squared", results1.rsquared_adj, results2.rsquared_adj))
    comparison_data.append(compare_attribute(
        "MSE", results1.mse, results2.mse))
    comparison_data.append(compare_attribute(
        "AIC", results1.aic, results2.aic))
    comparison_data.append(compare_attribute(
        "BIC", results1.bic, results2.bic))

    f_stat1 = results1.f_statistic[0] if isinstance(
        results1.f_statistic, tuple) else results1.f_statistic
    f_stat2 = results2.f_statistic[0] if isinstance(
        results2.f_statistic, tuple) else results2.f_statistic
    comparison_data.append(compare_attribute("F-statistic", f_stat1, f_stat2))

    f_pval1 = results1.f_statistic[1] if isinstance(
        results1.f_statistic, tuple) else np.nan
    f_pval2 = results2.f_statistic[1] if isinstance(
        results2.f_statistic, tuple) else np.nan
    comparison_data.append(compare_attribute(
        "F-statistic P-value", f_pval1, f_pval2))

    comparison_data.append(compare_attribute(
        "Fitted Values", results1.fittedvalues, results2.fittedvalues))
    comparison_data.append(compare_attribute(
        "Residuals", results1.residuals, results2.residuals))

    return pd.DataFrame(comparison_data)
