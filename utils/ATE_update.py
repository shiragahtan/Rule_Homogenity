import json
import numpy as np
import pandas as pd
from scipy import stats
from linear_model_unlearning import CertifiableUnlearningLogisticRegression, BaseLinearRegression
from sklearn.linear_model import LogisticRegression
from dowhy import CausalModel

# ---------------------------------------------------------
# 1. LOAD CONFIG & IMMUTABLE ATTRIBUTES
# ---------------------------------------------------------
try:
    with open('../configs/config.json', 'r') as f:
        config = json.load(f)
    
    TREATMENT_COL = config.get('TREATMENT_COL', 'TempTreatment')
    
    # Dynamically load Immutable Attributes based on the chosen dataset
    # Defaults to 'stackoverflow' if CHOSEN_DATASET is missing
    CHOSEN_DS = config.get('CHOSEN_DATASET', 'stackoverflow') 
    
    if CHOSEN_DS in config['DATASETS']:
        IMMUTABLE_ATTRS = set(config['DATASETS'][CHOSEN_DS].get('IMMUTABLE_ATTRIBUTES', []))
        print(f"🔹 Loaded {len(IMMUTABLE_ATTRS)} immutable attributes for {CHOSEN_DS}")
    else:
        IMMUTABLE_ATTRS = set()
        print(f"⚠️ Warning: Dataset '{CHOSEN_DS}' not found in config. Immutable attributes empty.")

except Exception as e:
    print(f"⚠️ Config Load Error: {e}. Using defaults.")
    TREATMENT_COL = 'TempTreatment'
    IMMUTABLE_ATTRS = set()


# ---------------------------------------------------------
# 2. CATE FUNCTION (DoWhy backdoor.linear_regression)
# ---------------------------------------------------------
class EstimationError(Exception):
    """The effect could not be estimated for this data (e.g. a missing treatment arm)."""


def get_adjustment_columns(df, treatment_col, outcome_col):
    """
    The single place that decides which columns the regression adjusts for.
    PENDING confirmation from the instructor; for now: every column except treatment and outcome.
    The same rule is applied to the overall rule population and to every subgroup.
    """
    return [c for c in df.columns if c not in (treatment_col, TREATMENT_COL, outcome_col)]


def estimate_cate(df, treatment_col, outcome_col):
    """
    CATE of treatment_col (0/1) on outcome_col for the given rows, using DoWhy
    CausalModel.estimate_effect with method "backdoor.linear_regression".

    Uses every row it is given: no trimming, no effect caps, no minimum arm sizes, no dropping of
    rows. The input is assumed to be null-free (see preprocessing.clean_dataset).
    Raises EstimationError when the effect cannot be estimated.
    """
    if df[treatment_col].nunique() < 2:
        raise EstimationError(f"'{treatment_col}' has a single value in {len(df)} rows; "
                              f"both a treated and a control group are required")
    try:
        model = CausalModel(
            data=df,
            treatment=treatment_col,
            outcome=outcome_col,
            common_causes=get_adjustment_columns(df, treatment_col, outcome_col),
        )
        estimand = model.identify_effect(proceed_when_unidentifiable=True)
        estimate = model.estimate_effect(
            estimand,
            method_name="backdoor.linear_regression",
            control_value=0,
            treatment_value=1,
            target_units="ate",
        )
        value = float(estimate.value)
    except Exception as e:
        raise EstimationError(f"DoWhy failed on {len(df)} rows: {e}") from e
    if not np.isfinite(value):
        raise EstimationError(f"DoWhy returned a non-finite estimate ({value}) on {len(df)} rows")
    return value


def calculate_ate_safe(df, treatment_col, outcome_col, delta=None):
    """
    Backwards-compatible wrapper used by all algorithms: returns the CATE, or NaN when it cannot
    be estimated (the failure is printed, not silent). `delta` is accepted for compatibility and
    ignored: the minimum subgroup size is enforced by the algorithms, not by the estimator.
    TODO: callers still treat NaN as "skip"; the final policy is pending the instructor's answer.
    """
    try:
        return estimate_cate(df, treatment_col, outcome_col)
    except EstimationError as e:
        print(f"⚠️ CATE not estimated: {e}")
        return np.nan


# ---------------------------------------------------------
# 3. ROBUST LINEAR CLASS (Handles Tier 3 Fallback)
# ---------------------------------------------------------
class ATEUpdateLinear:
    def __init__(self, X, T, Y, find_confounders=False):
        # Handle Tier 3 Fallback (Empty Features) safely
        if isinstance(X, list) or (isinstance(X, pd.DataFrame) and X.empty):
            self.X = pd.DataFrame()
            self.n_features = 0
            # Explicit design matrix: Intercept + Treatment only
            intercept = pd.Series(1, index=range(len(T)), name='intercept')
            # Reset indices for safe concat
            T_series = T.copy() if isinstance(T, pd.Series) else pd.Series(T)
            T_series.index = range(len(T))
            
            self.design_matrix = pd.concat([intercept, T_series], axis=1)
            self.design_matrix.columns = ['intercept', 'treatment']
            self.X_matrix = self.design_matrix.values
        else:
            # Standard Case (Tier 1 & 2)
            self.X = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=[f"X{i}" for i in range(X.shape[1])])
            self.n_features = self.X.shape[1]
            self.T = T.copy() if isinstance(T, pd.Series) else pd.Series(T, index=self.X.index)
            # Design Matrix Construction
            intercept = pd.Series(1, index=self.X.index, name='intercept')
            self.design_matrix = pd.concat([intercept, self.T, self.X], axis=1)
            self.design_matrix.columns = ['intercept', 'treatment'] + [str(c) for c in self.X.columns]
            self.X_matrix = self.design_matrix.values

        self.Y = Y.copy() if isinstance(Y, pd.Series) else pd.Series(Y, index=range(len(Y)))
        self.Y_matrix = self.Y.values.reshape(-1, 1)
        self.n_samples = self.X_matrix.shape[0]
        
        # Run Regression
        self.original_model = BaseLinearRegression(self.X_matrix, self.Y_matrix)
        self.original_ate = float(self.original_model.beta[1].item())

    def get_original_ate(self):
        return self.original_ate

    def calculate_p_value(self):
        """
        Calculate p-value for treatment effect using standard OLS inference.
        """
        try:
            y_pred = self.X_matrix @ self.original_model.beta
            residuals = self.Y_matrix - y_pred
            rss = np.sum(residuals ** 2)
            
            # Note: n_features here comes from X shape. 
            # In Tier 3, X is empty, n_features=0. 
            # But X_matrix has 2 cols (Intercept, Treatment).
            # So actual degrees of freedom used is 2 + n_features.
            df_resid = self.n_samples - (self.design_matrix.shape[1])
            
            if df_resid < 3:
                return 1.0
                
            mse = rss / df_resid
            
            if hasattr(self.original_model, 'XTX_inv'):
                xtx_inv = self.original_model.XTX_inv
            else:
                return 1.0 
            
            var_beta = mse * np.diag(xtx_inv)
            var_beta = np.clip(var_beta, 0, None)
            
            with np.errstate(invalid='ignore'):
                se_beta = np.sqrt(var_beta)
            
            se_beta = np.where((se_beta < 1e-10) | ~np.isfinite(se_beta), np.inf, se_beta)
            t_stat = self.original_model.beta.flatten() / se_beta
            p_values = 2 * (1 - stats.t.cdf(np.abs(t_stat), df_resid))
            
            return p_values[1] 
        except Exception:
            return 1.0

    def get_ate_difference(self, removed_indices, approx=False, update=True):
        if not removed_indices:
            return 0.0
        
        current_ate = self.original_ate
        
        if isinstance(removed_indices, int):
            removed_indices = [removed_indices]
        
        # Handle index alignment if Tier 3 reset indices
        # We assume removed_indices correspond to positions in X_matrix/Y_matrix
        X_remove = self.X_matrix[removed_indices]
        Y_remove = self.Y_matrix[removed_indices]
        
        if approx:
            XTX_inv_updated = self.original_model.neumann_update(X_remove)
        else:
            XTX_inv_updated = self.original_model.woodbury_update(X_remove)
        beta_updated = XTX_inv_updated @ (self.X_matrix.T @ self.Y_matrix - X_remove.T @ Y_remove)
        
        new_ate = float(beta_updated[1].item())
        if update:
            self.original_model.XTX_inv = XTX_inv_updated
            self.original_model.beta = beta_updated
            self.original_ate = new_ate
            
            # Update internal matrices is tricky with numpy array indices.
            # Simplified: we just update the model stats since that's what matters for ATE.
            # Re-slicing self.X/self.Y for Tier 3 logic is complex, usually not needed for simple removal.
        
        return new_ate - current_ate

    def calculate_updated_ATE(self, removed_indices, approx=False):
        if not removed_indices:
            return self.original_ate
        self.get_ate_difference(removed_indices, approx=approx, update=True)
        return self.original_ate


class ATEUpdateLogistic:
    def __init__(self, X, T, Y, lambda_reg=0.1, max_iter=1000):
        # Store the original dataset
        self.X = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=[f"X{i}" for i in range(X.shape[1])])
        self.T = T.copy() if isinstance(T, pd.Series) else pd.Series(T)
        self.Y = Y.copy() if isinstance(Y, pd.Series) else pd.Series(Y)
        
        # Model parameters
        self.lambda_reg = lambda_reg
        self.max_iter = max_iter
        
        # Initialize and train the original model
        self.original_model = CertifiableUnlearningLogisticRegression(lambda_reg=lambda_reg, max_iter=max_iter)
        self.original_model.fit(self.X.values, self.T.values)
        
        # Compute the original ATE
        self.original_ate = self._compute_ate_ipw_unlearning(self.T, self.Y, self.X, model=self.original_model)
        
        # Store all available indices
        self.available_indices = list(range(len(self.X)))
    
    def _compute_ate_ipw_unlearning(self, T, Y, X, model=None, removed_index=None):
        if model is None:
            model = CertifiableUnlearningLogisticRegression(lambda_reg=self.lambda_reg, max_iter=self.max_iter)
            model.fit(X.values, T.values)
        elif removed_index is not None:
            if isinstance(removed_index, int):
                removed_index = [removed_index]
            X_remove = X.iloc[removed_index].values
            T_remove = T.iloc[removed_index].values
            
            keep_indices = [i for i in range(len(X)) if i not in removed_index]
            X_keep = X.iloc[keep_indices].values
            T_keep = T.iloc[keep_indices].values

            model.fit_incremental_mini_batch(
                X_keep, T_keep, X_remove, T_remove, sigma=0, batch_size=len(X_remove)
            )
        
        propensity_scores = []
        for i in range(len(X)):
            prob = model.sigmoid(X.iloc[i:i+1].values @ model.theta)
            propensity_scores.append(prob[0])
        propensity_scores = np.array(propensity_scores)
        
        if removed_index is not None:
            if isinstance(removed_index, int):
                removed_index = [removed_index]
            include_mask = np.array([i not in removed_index for i in range(len(X))])
            T_filtered = T[include_mask]
            Y_filtered = Y[include_mask]
            ps_filtered = propensity_scores[include_mask]
            
            treated_mask = (T_filtered == 1)
            control_mask = (T_filtered == 0)
            
            weighted_sum_treated = np.sum(Y_filtered[treated_mask] / ps_filtered[treated_mask])
            weight_total_treated = np.sum(1 / ps_filtered[treated_mask])
            weighted_mean_treated = weighted_sum_treated / weight_total_treated
    
            weighted_sum_control = np.sum(Y_filtered[control_mask] / (1 - ps_filtered[control_mask]))
            weight_total_control = np.sum(1 / (1 - ps_filtered[control_mask]))
            weighted_mean_control = weighted_sum_control / weight_total_control
        else:
            treated_mask = (T == 1)
            control_mask = (T == 0)
            
            weighted_sum_treated = np.sum(Y[treated_mask] / propensity_scores[treated_mask])
            weight_total_treated = np.sum(1 / propensity_scores[treated_mask])
            weighted_mean_treated = weighted_sum_treated / weight_total_treated
    
            weighted_sum_control = np.sum(Y[control_mask] / (1 - propensity_scores[control_mask]))
            weight_total_control = np.sum(1 / (1 - propensity_scores[control_mask]))
            weighted_mean_control = weighted_sum_control / weight_total_control

        return weighted_mean_treated - weighted_mean_control
    
    def get_ate_difference(self, removed_indices, method='unlearning'):
        if not removed_indices:
            return 0.0
        current_ate = self.original_ate
        
        if isinstance(removed_indices, int):
            removed_indices = [removed_indices]
        
        if method == 'unlearning':
            X_remove = self.X.iloc[removed_indices].values
            T_remove = self.T.iloc[removed_indices].values
            
            keep_indices = [i for i in range(len(self.X)) if i not in removed_indices]
            X_keep = self.X.iloc[keep_indices].values
            T_keep = self.T.iloc[keep_indices].values
            
            self.original_model.fit_incremental_mini_batch(
                X_keep, T_keep, X_remove, T_remove, sigma=0, batch_size=len(X_remove)
            )
        
        elif method == 'retrain':
            keep_indices = [i for i in range(len(self.X)) if i not in removed_indices]
            X_keep = self.X.iloc[keep_indices].values
            T_keep = self.T.iloc[keep_indices].values
            
            self.original_model = CertifiableUnlearningLogisticRegression(lambda_reg=self.lambda_reg, max_iter=self.max_iter)
            self.original_model.fit(X_keep, T_keep)
        
        else:
            raise ValueError("Method must be either 'unlearning' or 'retrain'")
        
        keep_indices = [i for i in range(len(self.X)) if i not in removed_indices]
        self.X = self.X.iloc[keep_indices]
        self.T = self.T.iloc[keep_indices]
        self.Y = self.Y.iloc[keep_indices]
        
        self.available_indices = [idx for i, idx in enumerate(self.available_indices) if i not in removed_indices]
        self.original_ate = self._compute_ate_ipw_unlearning(self.T, self.Y, self.X, model=self.original_model)
        
        return self.original_ate - current_ate

    def get_original_ate(self):
        return self.original_ate

    def calculate_updated_ate(self, removed_indices, method='unlearning'):
        old_ate = self.original_ate
        self.get_ate_difference(removed_indices, method=method)
        return self.original_ate
    
    def _identify_confounders(self):
        try:
            import dowhy
            from dowhy import CausalModel
            import warnings
            warnings.filterwarnings('ignore')
            
            data = self.X.copy()
            data['treatment'] = self.T.values
            data['outcome'] = self.Y.values
            
            feature_names = self.X.columns.tolist()
            edges = []
            for feat in feature_names:
                edges.append(f"{feat} -> treatment")
                edges.append(f"{feat} -> outcome")
            edges.append("treatment -> outcome")
            
            graph = "digraph {" + "; ".join(edges) + "}"
            
            model = CausalModel(
                data=data,
                treatment='treatment',
                outcome='outcome',
                graph=graph,
                approach="backdoor"
            )
            
            identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
            
            if hasattr(identified_estimand, 'backdoor_variables') and identified_estimand.backdoor_variables:
                return identified_estimand.backdoor_variables
            else:
                return self.X.columns.tolist()
                
        except ImportError:
            print("DoWhy not installed. Using all variables as potential confounders.")
            return self.X.columns.tolist()
        except Exception as e:
            print(f"Error in confounder identification: {e}. Using all variables.")
            return self.X.columns.tolist()