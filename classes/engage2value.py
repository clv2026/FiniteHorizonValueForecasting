import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score, precision_score
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from scipy.stats import entropy
import plotly.colors as pc
import os
import wesanderson
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

class EngagementStateModel:
    def __init__(
        self,
        train_df,
        test_df,
        freq,
        period,
        reduced_features=False
    ):  
        self.transition_matrix = None
        self.clf = None
        self.train_df = train_df
        self.test_df = test_df
        self.freq = freq
        self.period = period
        self.seed = np.random.seed(42)
        if not reduced_features:
            self.feature_cols = [
                'avg_unit_price', 'num_invoices', 'avg_rev_per_invoice', 'quantity_per_invoice',
                f'avg_revenue_past_{self.period}T', f'activity_freq_past_{self.period}T',
                f'revenue_volatility_past_{self.period}T', 'periods_since_last_purchase'
            ]
        else:
            self.feature_cols = [
                'avg_unit_price', 'num_invoices',
                'avg_rev_per_invoice', 'quantity_per_invoice',
                'periods_since_last_purchase'
            ]
        self.state_mapping = {'active': 0, 'dormant': 1, 'inactive': 2}

    def build_markov_transition_matrix(self):
        '''
        Build the Markov transition matrix from the training data.
        '''
        transition_counts = pd.crosstab(self.train_df['engagement_t0'], self.train_df['engagement_t1'])
        self.transition_matrix = transition_counts.div(transition_counts.sum(axis=1), axis=0)

    def train_classifier(self):
        ''' Train a classifier to predict the next engagement state. '''
        X = self.train_df[['engagement_t0_code'] + self.feature_cols]
        y = self.train_df['engagement_t1_code']

        # Compute class weights
        classes = np.unique(y)
        class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
        class_weight_dict = dict(zip(classes, class_weights))

        # Map class weights to each sample
        sample_weights = y.map(class_weight_dict)
    
        self.clf = GradientBoostingClassifier(
            random_state=42
        )
        self.clf.fit(X, y, sample_weight=sample_weights)

    def forecast_next_state_with_model(self, current_state, features, lambda_weight=0.5):
        '''
        Forecast the next state probabilities using the trained model.
        '''
        # base_probs = self.transition_matrix.loc[current_state].values
        state_code = self.state_mapping[current_state]
        input_dict = {'engagement_t0_code': [state_code]}
        for i, col in enumerate(self.feature_cols):
            input_dict[col] = [features[i]]
        X_feat = pd.DataFrame(input_dict)
        feat_probs = self.clf.predict_proba(X_feat)[0]
        # You can blend with Markov if desired:
        # final_probs = (1 - lambda_weight) * base_probs + lambda_weight * feat_probs
        final_probs = feat_probs
        return final_probs

    def row_forecast(self, row):
        '''
        Forecast the next state and probabilities for a single row.
        '''
        current_state = row['engagement_t0']
        features = [row[col] for col in self.feature_cols]
        probs = self.forecast_next_state_with_model(current_state, features)
        pred_code = np.argmax(probs)
        return pred_code, probs

    def test_classifier(self):
        '''
        Test the classifier on the test set and return accuracy, F1, and log-loss.
        '''
        results = self.test_df.apply(self.row_forecast, axis=1)
        self.test_df['model_predicted_code_T1'] = [r[0] for r in results]
        self.test_df['model_predicted_probs_T1'] = [r[1] for r in results]

        y_true = self.test_df['engagement_t1_code']
        y_pred = self.test_df['model_predicted_code_T1']
        # accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0.0)
        f1 = f1_score(y_true, y_pred, average='weighted')
        ll = log_loss(y_true, np.stack(self.test_df['model_predicted_probs_T1'].values))
        
        return precision, f1, ll

    def plot_confusion_matrix_T1(
        self,
        model,
        y_true_col='engagement_t1_code',
    ):
        '''
        Plot the confusion matrix for predicted vs. true engagement states.
        '''
        # Get the first three colors from the 'Hotel Chevalier' palette
        colors = wesanderson.film_palette('Hotel Chevalier')[:3]

        # Create a custom colormap (reverse if you like the order better)
        custom_cmap = LinearSegmentedColormap.from_list('custom_wes', colors[::-1], N=256)

        # Capitalize your state labels
        labels = [k.capitalize() for k, v in sorted(self.state_mapping.items(), key=lambda item: item[1])]

        y_true = self.test_df[y_true_col]
        if model == 'model':
            y_pred = self.test_df[f'{model}_predicted_code_T1']
        else:
            y_pred = self.test_df[f'{model}_pred_code_T1']

        cm = confusion_matrix(y_true, y_pred)
        cm_percent = cm / cm.sum(axis=1, keepdims=True) * 100

        plt.figure(figsize=(8, 6))
        sns.set(style="whitegrid")

        # Plot heatmap
        ax = sns.heatmap(
            cm_percent,
            annot=True,
            fmt=".1f",
            cmap=custom_cmap,
            xticklabels=labels,
            yticklabels=labels,
            cbar_kws={'label': 'Percent (%)'}
        )

        # Customize the colorbar (the "legend" for a heatmap)
        cbar = ax.collections[0].colorbar
        cbar.set_label('Percent (%)', fontsize=14, color='#4f4e4e')
        cbar.ax.yaxis.label.set_color('#4f4e4e')
        cbar.ax.tick_params(labelsize=12, colors='#4f4e4e')

        plt.xlabel("Predicted state", fontsize=16, color='#4f4e4e')
        plt.ylabel("True state", fontsize=16, color='#4f4e4e')
        plt.xticks(fontsize=14, color='#4f4e4e')
        plt.yticks(fontsize=14, color='#4f4e4e')
        os.makedirs('classes/plots', exist_ok=True)
        plt.savefig(f"classes/plots/{self.freq}_T1_engagement_{model}_cm.png", dpi=300)
        plt.close()

    def naive_markov_predict(self, current_state):
        '''
        Predict next state probabilities using the Markov transition matrix.
        '''
        return self.transition_matrix.loc[current_state].values

    def test_markov(self):
        '''
        Test the Markov model on the test set and return accuracy, F1, and log-loss.
        '''
        self.test_df['markov_probs_T1'] = self.test_df['engagement_t0'].apply(lambda s: self.naive_markov_predict(s))
        self.test_df['markov_pred_code_T1'] = self.test_df['markov_probs_T1'].apply(np.argmax)

        y_true = self.test_df['engagement_t1_code']
        y_pred_naive = self.test_df['markov_pred_code_T1']
        # accuracy_markov = accuracy_score(y_true, y_pred_naive)
        precision_markov = precision_score(y_true, y_pred_naive, average='weighted', zero_division=0.0)
        f1_markov = f1_score(y_true, y_pred_naive, average='weighted')
        ll_markov = log_loss(y_true, np.stack(self.test_df['markov_probs_T1'].values))

        return precision_markov, f1_markov, ll_markov

    # 4. Global mode
    @staticmethod
    def _cumulative_global_mode(series):
        '''
        Compute the cumulative global mode for a pandas Series.
        For each position in the series, returns the most frequent value seen so far.
        '''
        modes = []
        for i in range(len(series)):
            modes.append(series[:i+1].value_counts().idxmax())
        return modes

    # 5. Entity-level mode
    @staticmethod
    def _cumulative_entity_mode(series):
        '''
        Compute the cumulative mode for a pandas Series within an entity (e.g., per customer).
        For each position, returns the most frequent value seen so far for that entity.
        '''
        modes = []
        for i in range(len(series)):
            modes.append(series[:i+1].value_counts().idxmax())
        return modes
    
    def run_benchmarks(self, verbose=True):
        '''
        Run benchmark comparisons for different models:
        - Markov Model
        - Gradient Boosting Classifier
        - Naive persistence (repeat last state)
        - Global-level mode
        - Entity-level mode

        Computes precision, F1, and log-loss for each approach and displays the results.
        '''
        # 1. Markov Model
        y_true = self.test_df['engagement_t1_code']
        y_pred_naive = self.test_df['markov_pred_code_T1']
        accuracy_naive = accuracy_score(y_true, y_pred_naive)
        precision_naive = precision_score(y_true, y_pred_naive, average='weighted', zero_division=0.0)
        f1_naive = f1_score(y_true, y_pred_naive, average='weighted')
        ll_naive = log_loss(y_true, np.stack(self.test_df['markov_probs_T1'].values))

        # 2. Gradient Boosting Classifier
        y_pred_model = self.test_df['model_predicted_code_T1']
        accuracy_model = accuracy_score(y_true, y_pred_model)
        precision_model = precision_score(y_true, y_pred_model, average='weighted', zero_division=0.0)
        f1_model = f1_score(y_true, y_pred_model, average='weighted')
        ll_model = log_loss(y_true, np.stack(self.test_df['model_predicted_probs_T1'].values))

        # 3. Naive persistence (repeat last state)
        self.train_df['naive_pred'] = self.train_df['engagement_t0']
        mask = self.train_df['engagement_t1'].notna()
        y_true_repeat = self.train_df.loc[mask, 'engagement_t1']
        y_pred_repeat = self.train_df.loc[mask, 'naive_pred']
        accuracy_repeat = accuracy_score(y_true_repeat, y_pred_repeat)
        precision_repeat = precision_score(y_true_repeat, y_pred_repeat, average='weighted', zero_division=0.0)
        f1_repeat = f1_score(y_true_repeat, y_pred_repeat, average='weighted')
        classes = np.unique(y_true_repeat)
        y_pred_proba = np.zeros((len(y_pred_repeat), len(classes)))
        for i, label in enumerate(classes):
            y_pred_proba[:, i] = (y_pred_repeat == label).astype(float)
        ll_repeat = log_loss(y_true_repeat, y_pred_proba, labels=classes)

        bench_df = self.train_df.copy()
        bench_df = bench_df.sort_values(['CustomerID', 'timestamp']).reset_index(drop=True)
        bench_df['global_mode'] = self._cumulative_global_mode(bench_df['engagement_t1'])
        bench_df['entity_mode'] = (
            bench_df.groupby('CustomerID')['engagement_t1']
            .transform(self._cumulative_entity_mode)
        )

        mask = bench_df['engagement_t1'].notna()
        y_true_mode = bench_df.loc[mask, 'engagement_t1']

        # Global mode
        y_pred_global = bench_df.loc[mask, 'global_mode']
        acc_global = accuracy_score(y_true_mode, y_pred_global)
        precision_global = precision_score(y_true_mode, y_pred_global, average='weighted', zero_division=0.0)
        f1_global = f1_score(y_true_mode, y_pred_global, average='weighted')
        y_pred_proba_global = np.zeros((len(y_pred_global), len(classes)))
        for i, label in enumerate(classes):
            y_pred_proba_global[:, i] = (y_pred_global == label).astype(float)
        ll_global = log_loss(y_true_mode, y_pred_proba_global, labels=classes)

        # Entity-level mode
        y_pred_entity = bench_df.loc[mask, 'entity_mode']
        acc_entity = accuracy_score(y_true_mode, y_pred_entity)
        precision_entity = precision_score(y_true_mode, y_pred_entity, average='weighted', zero_division=0.0)
        f1_entity = f1_score(y_true_mode, y_pred_entity, average='weighted')
        y_pred_proba_entity = np.zeros((len(y_pred_entity), len(classes)))
        for i, label in enumerate(classes):
            y_pred_proba_entity[:, i] = (y_pred_entity == label).astype(float)
        ll_entity = log_loss(y_true_mode, y_pred_proba_entity, labels=classes)

        # Results DataFrame
        results = {
            'Model': [
                'Markov Matrix', 'GradientBoostCl', 'Naive repeat',
                'Global-level mode', 'Entity-level mode'
            ],
            'Precision': [
                precision_naive, precision_model, precision_repeat, precision_global, precision_entity
            ],
            'F1': [
                f1_naive, f1_model, f1_repeat, f1_global, f1_entity
            ],
            'Log-loss': [
                ll_naive, ll_model, ll_repeat, ll_global, ll_entity
            ]
        }
        results_df = pd.DataFrame(results)
        if verbose:
            display(results_df)


class ValueForecastingModel:
    def __init__(
        self,
        train_df,
        test_df,
        markov_matrix,
        model,
        freq,
        period,
        reduced_features=False,
        value_col='Revenue',
        ablation=False
    ):
        self.train_df = train_df
        self.test_df = test_df
        self.value_col = value_col
        self.markov_matrix = markov_matrix
        self.model = model
        self.freq = freq
        self.period = period
        self.seed = np.random.seed(42)
        self.state_avg = None
        self.state_regressors = {}
        self.state_mapping = {'active': 0, 'dormant': 1, 'inactive': 2}
        self.value_state_mapping = {0: 'active', 1: 'dormant', 2: 'inactive'}
        if not reduced_features:
            self.feature_cols = [
                'avg_unit_price', 'num_invoices', 'avg_rev_per_invoice',
                'quantity_per_invoice', f'avg_revenue_past_{self.period}T',
                f'activity_freq_past_{self.period}T', f'revenue_volatility_past_{self.period}T',
                'periods_since_last_purchase'
            ]
        else:
            self.feature_cols = [
                    'avg_unit_price', 'num_invoices', 'avg_rev_per_invoice',
                    'quantity_per_invoice', 'periods_since_last_purchase'
                ]
        if ablation:
            self.state_proportions = self.get_engagement_state_proportions()

    def _markov_multi_step_forecast(self, current_state):
        '''
        Compute multi-step Markov forecast probabilities for a given current state.
        '''
        states = self.markov_matrix.columns.tolist()
        n_states = len(states)

        # Create the current state vector (one-hot)
        current_state_vector = np.zeros(n_states)
        current_state_vector[states.index(current_state)] = 1

        # Raise the transition matrix to the desired power
        transition_matrix_np = self.markov_matrix.values
        transition_matrix_x = np.linalg.matrix_power(
            transition_matrix_np, self.period)
        
        # Compute future state probabilities
        future_state_probs = current_state_vector @ transition_matrix_x
        return dict(zip(states, future_state_probs))

    def _row_markov_forecast(self, row):
        '''
        Apply the multi-step Markov forecast to a single row and return predicted state and probabilities.
        '''
        current_state = row['engagement_t0']
        probs = self._markov_multi_step_forecast(current_state)
        pred_code = np.argmax(list(probs.values()))
        return pred_code, list(probs.values())

    # For markov model, forecasting is repeated matrix multiplication
    def markov_forecast_engagement(self):
        '''
        Forecast the state distribution after a given number of periods for all test data using the Markov model.
        '''
        self.test_df[[
            f'markov_predicted_code_T{self.period}',
            f'markov_predicted_probs_T{self.period}']
        ] = self.test_df.apply(
            lambda row: pd.Series(self._row_markov_forecast(row)), axis=1
        )

    def get_engagement_state_proportions(self):
        '''
        Compute engagement states proportions
        '''
        all_states = pd.concat([self.train_df['engagement_t0_code'], self.test_df['engagement_t0_code']])
        n_states = len(self.state_mapping)
        counts = np.bincount(all_states, minlength=n_states)
        probs = counts / counts.sum()
        return probs

    def _model_multi_step_forecast(
        self,
        current_state,
        features,
        random_probs=False
    ):
        '''
        Compute multi-step forecast probabilities using the trained model and current features,
        or random probabilities if random_probs is True.
        '''
        n_states = len(self.state_mapping)
        if random_probs:
            # Get empirical proportions
            empirical_props = self.state_proportions
            state_codes = np.arange(n_states)

            # Randomly select a state according to empirical proportions
            random_state = np.random.choice(state_codes, p=empirical_props)

            # Return a one-hot vector for the randomly selected state
            state_probs = np.zeros(n_states)
            state_probs[random_state] = 1.0
            return state_probs
        else:
            state_probs = np.zeros(n_states)
            state_probs[self.state_mapping[current_state]] = 1.0

            for _ in range(self.period):
                next_state_probs = np.zeros(n_states)
                for _, state_code in self.state_mapping.items():
                    # Build the input dict in the correct order
                    input_dict = {'engagement_t0_code': [state_code]}
                    for i, col in enumerate(self.feature_cols):
                        input_dict[col] = [features[i]]
                    X_feat = pd.DataFrame(input_dict)[['engagement_t0_code'] + self.feature_cols]  # enforce order
                    probs = self.model.predict_proba(X_feat)[0]
                    next_state_probs += state_probs[state_code] * probs
                state_probs = next_state_probs
            state_probs = state_probs / np.sum(state_probs)
            return state_probs

    def _row_model_forecast(
        self,
        row,
        random_probs
    ):
        '''
        Apply the multi-step model forecast to a single row and return predicted state and probabilities,
        or random probabilities if random_probs is True.
        '''
        current_state = row['engagement_t0']
        features = [row[col] for col in self.feature_cols]
        probs = self._model_multi_step_forecast(
            current_state,
            features,
            random_probs=random_probs
        )
        pred_code = np.argmax(probs)
        return pred_code, probs

    def model_forecast_engagement(self, random_probs=False):
        '''
        Forecast the state distribution after a given number of periods for all test data using the trained model,
        or random probabilities if random_probs is True.
        '''
        self.test_df[
            [
                f'model_predicted_code_T{self.period}',
                f'model_predicted_probs_T{self.period}'
            ]
        ] = self.test_df.apply(
            lambda row: pd.Series(self._row_model_forecast(
                row,
                random_probs=random_probs
                )), axis=1
        )

    def get_offset(self):
        '''
        Return the appropriate pandas DateOffset object based on the frequency and period.
        '''
        if self.freq == 'M':
            return pd.DateOffset(months=self.period)
        elif self.freq == '2M':
            return pd.DateOffset(months=2*self.period)
        elif self.freq == 'W':
            return pd.DateOffset(weeks=self.period)
        elif self.freq == '2W':
            return pd.DateOffset(weeks=2*self.period)
        else:
            raise ValueError("Unsupported frequency")

    # Testing Step 4
    def step4_metrics(self):
        '''
        Evaluate and display accuracy, F1, and log-loss for Markov and model forecasts at the specified horizon.
        '''
        # Create a lookup for (CustomerID, timestamp_ts) -> engagement_t0_code
        future_lookup = self.test_df.set_index(['CustomerID', 'timestamp_ts'])['engagement_t0_code'].to_dict()

        eval_rows = []
        for _, row in self.test_df.iterrows():
            # Capture X periods ahead t0 code
            offset = self.get_offset()
            future_date = row['timestamp_ts'] + offset
            # future_date = row['timestamp_ts'] + pd.DateOffset(months=3)

            key = (row['CustomerID'], future_date)
            actual_state_TX = future_lookup.get(key, None)
            if actual_state_TX is not None:
                eval_rows.append({
                    'CustomerID': row['CustomerID'],
                    'timestamp': future_date,
                    'y_true': int(actual_state_TX),
                    'y_pred_markov': int(row[f'markov_predicted_code_T{self.period}']),
                    'probs_markov': row[f'markov_predicted_probs_T{self.period}'],
                    'y_pred_model': int(row[f'model_predicted_code_T{self.period}']),
                    'probs_model': row[f'model_predicted_probs_T{self.period}']
                })

        eval_df = pd.DataFrame(eval_rows)

        if eval_df.empty:
            print(
                f"[Warning] No evaluation rows found in step4_metrics. "
                f"This usually means your selected period (horizon={self.period}) is too large, "
                f"or your test set does not cover enough future periods. "
                f"Try using a smaller period or check your data coverage."
            )
            return  # Exit the function gracefully

        y_true = eval_df['y_true']
        y_pred_markov = eval_df['y_pred_markov']
        y_pred_model = eval_df['y_pred_model']
        probs_markov = np.stack(eval_df['probs_markov'].values)
        probs_model = np.stack(eval_df['probs_model'].values)

        step4_results = {
            'Model': ['Naive Markov Matrix', 'GradientBoostCl'],
            'F1': [
                round(f1_score(y_true, y_pred_markov, average='weighted'), 5),
                round(f1_score(y_true, y_pred_model, average='weighted'), 5)
            ],
            'Log-loss': [
                round(log_loss(y_true, probs_markov), 5),
                round(log_loss(y_true, probs_model), 5)
            ],
            'Precision': [
                round(precision_score(y_true, y_pred_markov, average='weighted', zero_division=0.0), 5),
                round(precision_score(y_true, y_pred_model, average='weighted', zero_division=0.0), 5)
            ]
        }

        if self.freq == 'W':
            unit = "weeks"
            horizon = self.period
        elif self.freq == '2W':
            unit = "weeks"
            horizon = 2 * self.period
        elif self.freq == '2M':
            unit = "months"
            horizon = 2 * self.period
        else:  # Default to monthly
            unit = "months"
            horizon = self.period

        print(f"\nState predictions at a forecast horizon of {horizon} {unit}.")
        display(pd.DataFrame(step4_results))
        
    def _compute_state_averages(self):
        '''
        Compute and store the average value for each engagement state from the training data.
        '''
        self.state_avg = self.train_df.groupby('engagement_t0_code')[self.value_col].mean().to_dict()
        print(f"\nEmpirical state averages of {self.value_col}:")
        for k, v in self.state_avg.items():
            print(f" State {k} ({self.value_state_mapping.get(k)}), avg: {round(v, 5)}")

    def _expected_value_from_avg(self, probs):
        '''
        Given a probability vector, return the expected value using precomputed state averages.
        '''
        # Ensure state_avg is computed
        if not hasattr(self, 'state_avg') or self.state_avg is None:
            self._compute_state_averages()
        # The order of states is assumed to be 0, 1, 2, ...
        avg_list = [self.state_avg.get(i, 0) for i in range(len(probs))]
        return float(np.dot(probs, avg_list))

    def apply_empirical_state_averages(self):
        '''
        Assign hard and soft value predictions to test data using empirical state averages.
        '''
        # Ensure state_avg is computed
        if not hasattr(self, 'state_avg') or self.state_avg is None:
            self._compute_state_averages()

        # Markov predicted probs
        # Hard assignment
        hard_col = f'markov_predicted_{self.value_col}_T{self.period}_hard'
        code_col = f'markov_predicted_code_T{self.period}'
        self.test_df[hard_col] = self.test_df[code_col].map(self.state_avg)

        # Soft assignment
        soft_col = f'markov_predicted_{self.value_col}_T{self.period}_soft'
        probs_col = f'markov_predicted_probs_T{self.period}'
        self.test_df[soft_col] = self.test_df[probs_col].apply(self._expected_value_from_avg)

        # Gradient Boosting predicted probs
        # Hard assignment
        hard_col = f'model_predicted_{self.value_col}_T{self.period}_hard'
        code_col = f'model_predicted_code_T{self.period}'
        self.test_df[hard_col] = self.test_df[code_col].map(self.state_avg)

        # Soft assignment
        soft_col = f'model_predicted_{self.value_col}_T{self.period}_soft'
        probs_col = f'model_predicted_probs_T{self.period}'
        self.test_df[soft_col] = self.test_df[probs_col].apply(self._expected_value_from_avg)

    def train_state_regressors(self):
        '''
        Train a separate regressor for each engagement state.
        '''
        self.state_mapping = {'active': 0, 'dormant': 1, 'inactive': 2}
        for state_code in self.train_df['engagement_t0_code'].unique():
            state_data = self.train_df[self.train_df['engagement_t0_code'] == state_code]
            X_state = state_data[self.feature_cols]
            y_state = state_data[self.value_col]
            reg = GradientBoostingRegressor(random_state=42)
            reg.fit(X_state, y_state)
            self.state_regressors[state_code] = reg

    def train_regressor_directly(self):
        '''
        Train a regressor directly to predict the target value without modeling engagement states.
        '''
        # Build lookup for (CustomerID, timestamp) -> value at T
        value_lookup = self.train_df.set_index(
            ['CustomerID', 'timestamp_ts']
            )[self.value_col].to_dict()

        train_rows = []
        for _, row in self.train_df.iterrows():
            if self.freq == 'M':
                offset = pd.DateOffset(months=self.period)
            elif self.freq == 'W':
                offset = pd.DateOffset(weeks=self.period)
            future_date = row['timestamp_ts'] + offset
            key = (row['CustomerID'], future_date)
            target_value_T = value_lookup.get(key, None)
            if target_value_T is not None:
                row_dict = {col: row[col] for col in self.feature_cols}
                row_dict['target'] = target_value_T
                train_rows.append(row_dict)
        train_df = pd.DataFrame(train_rows)
        if train_df.empty:
            print("[Warning] No training rows found in train_model_ablation. "
                f"This usually means your selected period ({self.period}) is too large, "
                "or your training set does not cover enough future periods. "
                "Try using a smaller period or check your data coverage.")
            return None, None, None

        # Train regressor on current features to predict value at T
        X_train = train_df[self.feature_cols]
        y_train = train_df['target']
        reg = GradientBoostingRegressor()
        reg.fit(X_train, y_train)

        value_lookup = self.test_df.set_index(
            ['CustomerID', 'timestamp_ts']
            )[self.value_col].to_dict()

        test_rows = []
        for _, row in self.test_df.iterrows():
            if self.freq == 'M':
                offset = pd.DateOffset(months=self.period)
            elif self.freq == 'W':
                offset = pd.DateOffset(weeks=self.period)
            future_date = row['timestamp_ts'] + offset
            key = (row['CustomerID'], future_date)
            target_value_T = value_lookup.get(key, None)
            if target_value_T is not None:
                row_dict = {col: row[col] for col in self.feature_cols}
                row_dict['target'] = target_value_T
                test_rows.append(row_dict)
        test_df = pd.DataFrame(test_rows)
        if test_df.empty:
            print("[Warning] No test rows found in train_model_ablation. "
                f"This usually means your selected period ({self.period}) is too large, "
                "or your test set does not cover enough future periods. "
                "Try using a smaller period or check your data coverage.")
            return None, None, None


        # Predict for eval_df
        X_test = test_df[self.feature_cols]
        y_true = test_df['target']
        y_pred = reg.predict(X_test)

        # Metrics
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)

        return mae, rmse, r2

    def _predict_value_per_state(self, row):
        '''
        Predict the value for each state for a given row.
        '''
        preds = {
            0: None,
            1: None,
            2: None
        }

        cols = self.feature_cols        
        X_feat = pd.DataFrame([row[cols]], columns=cols)
        for state_code, reg in self.state_regressors.items():
            preds[state_code] = reg.predict(X_feat)[0]
        return preds

    def apply_state_regressors(self):
        '''
        Apply state-specific regressors to all test data and store predictions.
        '''
        self.test_df[f'state_reg_{self.value_col}_T{self.period}_dict'] = self.test_df.apply(
            lambda row: self._predict_value_per_state(row), axis=1
            )

    def _expected_value_from_state_regressors(self, probs, state_value_preds):
        '''
        Compute the expected value from state regressors using predicted state probabilities.
        '''
        state_codes = sorted(state_value_preds.keys())
        values = np.array([state_value_preds[code] for code in state_codes])
        return float(np.dot(probs, values))

    def compute_expected_value_from_regressors(self, ablation=False):
        '''
        Compute and store the expected value for each test row using state regressors and predicted probabilities.
        '''
        if not ablation:
            self.test_df[f'markov_predicted_{self.value_col}_T{self.period}_reg_soft'] = self.test_df.apply(
                lambda row: self._expected_value_from_state_regressors(
                    row[f'markov_predicted_probs_T{self.period}'],
                    row[f'state_reg_{self.value_col}_T{self.period}_dict']), axis=1
            )

        self.test_df[f'model_predicted_{self.value_col}_T{self.period}_reg_soft'] = self.test_df.apply(
            lambda row: self._expected_value_from_state_regressors(
                row[f'model_predicted_probs_T{self.period}'],
                row[f'state_reg_{self.value_col}_T{self.period}_dict']), axis=1
        )

    def step5_6_metrics(self):
        '''
        Evaluate and display MAE, RMSE, and R2 for value predictions using both empirical averages and state regressors.
        '''
        # Create a lookup for (CustomerID, timestamp_ts) -> true value (target column)
        revenue_lookup = self.test_df.set_index(['CustomerID', 'timestamp_ts'])[self.value_col].to_dict()

        # Build evaluation rows for each prediction row (for which we have a true value 3 periods ahead)
        eval_rows = []
        for _, row in self.test_df.iterrows():
            # Capture X periods ahead t0 code
            offset = self.get_offset()
            future_date = row['timestamp_ts'] + offset
            # future_date = row['timestamp_ts'] + pd.DateOffset(months=self.period)
            
            key = (row['CustomerID'], future_date)
            actual_revenue_TX = revenue_lookup.get(key, None)
            if actual_revenue_TX is not None:
                eval_rows.append({
                    'CustomerID': row['CustomerID'],
                    'timestamp': future_date,
                    'y_true_state': row['engagement_t0_code'],
                    'y_true': actual_revenue_TX,
                    'y_pred_markov_code': row[f'markov_predicted_code_T{self.period}'],
                    'y_pred_model_code': row[f'model_predicted_code_T{self.period}'],
                    'y_pred_6A_markov': row[f'markov_predicted_{self.value_col}_T{self.period}_soft'],
                    'y_pred_6A_model': row[f'model_predicted_{self.value_col}_T{self.period}_soft'],
                    'y_pred_6B_markov': row[f'markov_predicted_{self.value_col}_T{self.period}_reg_soft'],
                    'y_pred_6B_model': row[f'model_predicted_{self.value_col}_T{self.period}_reg_soft']
                })

        eval_df = pd.DataFrame(eval_rows)
        if eval_df.empty:
            print(
                "[Warning] No evaluation rows found in step5_6_metrics. "
                f"This usually means your selected period ({self.period}) is too large, "
                "or your test set does not cover enough future periods. "
                "Try using a smaller period or check your data coverage."
                )
            return

        y_true_value = eval_df['y_true']  # Target revenue value
        y_true_state = eval_df['y_true_state']  # Target engagement state
        y_markov_code = eval_df['y_pred_markov_code']
        y_model_code = eval_df['y_pred_model_code']
        y_pred_6A_markov = eval_df['y_pred_6A_markov']
        y_pred_6A_model = eval_df['y_pred_6A_model']
        y_pred_6B_markov = eval_df['y_pred_6B_markov']
        y_pred_6B_model = eval_df['y_pred_6B_model']

        results = {
            'Method': [
                '5A-Markov: empirical states avg.',
                '5A-GBC: empirical states avg.',
                '5B-Markov: State-conditional regr.',
                '5B-GBC: State-conditional regr.',
                ],
            'MAE': [
                mean_absolute_error(y_true_value, y_pred_6A_markov),
                mean_absolute_error(y_true_value, y_pred_6A_model),
                mean_absolute_error(y_true_value, y_pred_6B_markov),
                mean_absolute_error(y_true_value, y_pred_6B_model)
                ],
            'RMSE': [
                np.sqrt(mean_squared_error(y_true_value, y_pred_6A_markov)),
                np.sqrt(mean_squared_error(y_true_value, y_pred_6A_model)),
                np.sqrt(mean_squared_error(y_true_value, y_pred_6B_markov)),
                np.sqrt(mean_squared_error(y_true_value, y_pred_6B_model)),
                ],
            'R2': [
                r2_score(y_true_value, y_pred_6A_markov),
                r2_score(y_true_value, y_pred_6A_model),
                r2_score(y_true_value, y_pred_6B_markov),
                r2_score(y_true_value, y_pred_6B_model),
                ]
        }

        if self.freq == 'W':
            unit = "weeks"
            horizon = self.period
        elif self.freq == '2W':
            unit = "weeks"
            horizon = 2 * self.period
        elif self.freq == '2M':
            unit = "months"
            horizon = 2 * self.period
        else:  # Default to monthly
            unit = "months"
            horizon = self.period

        print(f"\nComparison of predicted vs. actual {self.value_col} at {horizon}-{unit} horizon:")
        display(pd.DataFrame(results))

    def regressor_metrics(
        self,
        ablation=False
    ):
        '''
        Return MAE, RMSE, and R2 of the regressor for the ablation study.
        '''
        if ablation:
            output_col = f'model_predicted_{self.value_col}_T{self.period}_reg_soft'
        else:
            output_col = f'state_reg_{self.value_col}_T{self.period}_dict'

        # Create a lookup for (CustomerID, timestamp_ts) -> true value (target column)
        revenue_lookup = self.test_df.set_index(['CustomerID', 'timestamp_ts'])[self.value_col].to_dict()
        state_lookup = self.test_df.set_index(['CustomerID', 'timestamp_ts'])['engagement_t0_code'].to_dict()

        # Build evaluation rows for each prediction row (for which we have a true value x periods ahead)
        eval_rows = []
        for _, row in self.test_df.iterrows():
            # Capture X periods ahead t0 code
            offset = self.get_offset()
            future_date = row['timestamp_ts'] + offset
            
            key = (row['CustomerID'], future_date)
            actual_revenue_TX = revenue_lookup.get(key, None)
            if ablation:
                if actual_revenue_TX is not None:
                    eval_rows.append({
                        'CustomerID': row['CustomerID'],
                        'timestamp': future_date,
                        'y_true': actual_revenue_TX,
                        'y_pred_6B': row[output_col],
                        'y_pred_code': row[f'model_predicted_code_T{self.period}']
                    })
            else:
                actual_state_TX = state_lookup.get(key, None)
                if actual_revenue_TX is not None and actual_state_TX is not None:
                    # Get the regressor's prediction for the true state at T
                    reg_pred_dict = row[output_col]
                    if isinstance(reg_pred_dict, dict) and actual_state_TX in reg_pred_dict:
                        reg_pred = reg_pred_dict[actual_state_TX]
                        eval_rows.append({
                            'CustomerID': row['CustomerID'],
                            'timestamp': future_date,
                            'y_true': actual_revenue_TX,
                            'y_pred_6B': reg_pred
                        })

        eval_df = pd.DataFrame(eval_rows)
        if eval_df.empty:
            print(
                "[Warning] No evaluation rows found in regressor_metrics. "
                f"This usually means your selected period ({self.period}) is too large, "
                "or your test set does not cover enough future periods. "
                "Try using a smaller period or check your data coverage.")
            return None, None, None

        if ablation:
            # Filter to only rows predicted as 'active' 
            # add rows where real state is active
            eval_df_active = eval_df[eval_df['y_pred_code'] == self.state_mapping.get('active')]
            y_true = eval_df_active['y_true']
            y_pred_6B = eval_df_active['y_pred_6B']    
        else:
            y_true = eval_df['y_true']
            y_pred_6B = eval_df['y_pred_6B']

        mae = mean_absolute_error(y_true, y_pred_6B)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred_6B))
        r2 = r2_score(y_true, y_pred_6B)

        return mae, rmse, r2

    @staticmethod
    def _jensen_shannon_divergence(p, q):
        '''
        Compute the Jensen-Shannon divergence between two probability distributions.
        '''
        p = np.asarray(p)
        q = np.asarray(q)
        # Avoid division by zero
        p = np.where(p == 0, 1e-12, p)
        q = np.where(q == 0, 1e-12, q)
        m = 0.5 * (p + q)
        return 0.5 * entropy(p, m) + 0.5 * entropy(q, m)

    def compare_state_distributions(
        self,
        plot_cm=False,
        plot_drift=False
    ):
        '''
        Compare predicted and actual state distributions at the forecast horizon, optionally plotting results.
        '''
        # Build a lookup for (CustomerID, timestamp) -> actual state at T
        actual_lookup = self.test_df.set_index(['CustomerID', 'timestamp_ts'])['engagement_t0_code'].to_dict()
        eval_rows = []
        for _, row in self.test_df.iterrows():
            offset = self.get_offset()
            future_date = row['timestamp_ts'] + offset
            key = (row['CustomerID'], future_date)
            actual_state_T = actual_lookup.get(key, None)
            if actual_state_T is not None:
                eval_rows.append({
                    'CustomerID': row['CustomerID'],
                    'timestamp': future_date,
                    'markov_predicted_state': int(row[f'markov_predicted_code_T{self.period}']),
                    'model_predicted_state': int(row[f'model_predicted_code_T{self.period}']),
                    'actual_state': int(actual_state_T)
                })

        eval_df = pd.DataFrame(eval_rows)
        if eval_df.empty:
            print(
                "[Warning] No evaluation rows found in compare_state_distributions. "
                f"This usually means your selected period ({self.period}) is too large, "
                "or your test set does not cover enough future periods. "
                "Try using a smaller period or check your data coverage.")
            return
        # Compute predicted and actual distributions
        model_pred_dist = np.bincount(eval_df['model_predicted_state'], minlength=len(self.state_mapping))
        model_pred_dist = model_pred_dist / model_pred_dist.sum()

        markov_pred_dist = np.bincount(eval_df['markov_predicted_state'], minlength=len(self.state_mapping))
        markov_pred_dist = markov_pred_dist / markov_pred_dist.sum()
        
        actual_dist = np.bincount(eval_df['actual_state'], minlength=len(self.state_mapping))
        actual_dist = actual_dist / actual_dist.sum()
        
        # Compute JSD
        jsd_model = self._jensen_shannon_divergence(model_pred_dist, actual_dist)
        jsd_markov = self._jensen_shannon_divergence(markov_pred_dist, actual_dist)

        state_labels = [self.value_state_mapping[i] for i in range(len(self.state_mapping))]

        if self.freq == 'W':
            unit = "weeks"
            horizon = self.period
        elif self.freq == '2W':
            unit = "weeks"
            horizon = 2 * self.period
        elif self.freq == '2M':
            unit = "months"
            horizon = 2 * self.period
        else:  # Default to monthly
            unit = "months"
            horizon = self.period
        
        print(f"\nComparison of state distribution at {horizon}-{unit} horizon")
        print("Model predicted state distribution:")
        for label, value in zip(state_labels, np.round(model_pred_dist, 3)):
            print(f"  {label}: {round(value*100, 5)}%")
        print("\nMarkov predicted state distribution:")
        for label, value in zip(state_labels, np.round(markov_pred_dist, 3)):
            print(f"  {label}: {round(value*100, 5)}%")
        print("\nActual state distribution:")
        for label, value in zip(state_labels, np.round(actual_dist, 3)):
            print(f"  {label}: {round(value*100, 5)}%")

        print(f"\nJensen-Shannon divergence (Model vs Actual): {jsd_model:.4f}")
        print(f"Jensen-Shannon divergence (Markov vs Actual): {jsd_markov:.4f}")

        # Plot confusion matrix
        if plot_cm:
            # Get the first three colors from the 'Hotel Chevalier' palette
            colors = wesanderson.film_palette('Hotel Chevalier')[:3]

            # Create a custom colormap (reverse if you like the order better)
            custom_cmap = LinearSegmentedColormap.from_list('custom_wes', colors[::-1], N=256)

            # Capitalize your state labels
            labels = [k.capitalize() for k, v in sorted(self.state_mapping.items(), key=lambda item: item[1])]

            # Compute confusion matrix and normalize
            y_true = eval_df['actual_state']
            y_pred = eval_df['model_predicted_state']
            cm = confusion_matrix(y_true, y_pred)
            cm_percent = cm / cm.sum(axis=1, keepdims=True) * 100

            plt.figure(figsize=(8, 6))
            sns.set(style="whitegrid")

            # Plot heatmap
            ax = sns.heatmap(
                cm_percent,
                annot=True,
                fmt=".1f",
                cmap=custom_cmap,
                xticklabels=labels,
                yticklabels=labels,
                cbar_kws={'label': 'Percent (%)'}
            )

            # Customize the colorbar (the "legend" for a heatmap)
            cbar = ax.collections[0].colorbar
            cbar.set_label('Percent (%)', fontsize=14, color='#4f4e4e')
            cbar.ax.yaxis.label.set_color('#4f4e4e')
            cbar.ax.tick_params(labelsize=12, colors='#4f4e4e')

            plt.xlabel("Predicted state", fontsize=16, color='#4f4e4e')
            plt.ylabel("True state", fontsize=16, color='#4f4e4e')
            plt.xticks(fontsize=14, color='#4f4e4e')
            plt.yticks(fontsize=14, color='#4f4e4e')
            os.makedirs('classes/plots', exist_ok=True)
            plt.savefig(f"classes/plots/{self.freq}_T{self.period}_drift_model_cm.png", dpi=300)
            plt.close()

            y_true = eval_df['actual_state']
            y_pred = eval_df['markov_predicted_state']

            cm = confusion_matrix(y_true, y_pred)
            cm_percent = cm / cm.sum(axis=1, keepdims=True) * 100

            plt.figure(figsize=(8, 6))
            sns.set(style="whitegrid")

            # Plot heatmap
            ax = sns.heatmap(
                cm_percent,
                annot=True,
                fmt=".1f",
                cmap=custom_cmap,
                xticklabels=labels,
                yticklabels=labels,
                cbar_kws={'label': 'Percent (%)'}
            )

            # Customize the colorbar (the "legend" for a heatmap)
            cbar = ax.collections[0].colorbar
            cbar.set_label('Percent (%)', fontsize=14, color='#4f4e4e')
            cbar.ax.yaxis.label.set_color('#4f4e4e')
            cbar.ax.tick_params(labelsize=12, colors='#4f4e4e')

            plt.xlabel("Predicted state", fontsize=16, color='#4f4e4e')
            plt.ylabel("True state", fontsize=16, color='#4f4e4e')
            plt.xticks(fontsize=14, color='#4f4e4e')
            plt.yticks(fontsize=14, color='#4f4e4e')
            plt.savefig(f"classes/plots/{self.freq}_T{self.period}_drift_markov_cm.png", dpi=300)
            plt.close()

        if plot_drift:
            colors = wesanderson.film_palette('Hotel Chevalier')[:3]
            state_labels = [label.capitalize() for label in state_labels]

            # Prepare data in long format
            df = pd.DataFrame({
                'State': state_labels * 3,
                'Proportion': np.concatenate([model_pred_dist, markov_pred_dist, actual_dist]),
                'Type': ['GradientBoost'] * len(state_labels) + ['Markov'] * len(state_labels) + ['True data'] * len(state_labels)
            })

            plt.figure(figsize=(10, 6))
            sns.set(style="whitegrid")
            sns.barplot(data=df, x='State', y='Proportion', hue='Type', palette=colors)
            plt.ylabel("Proportion", fontsize=16, color='#4f4e4e')
            plt.xlabel("")
            plt.xticks(size=14, color='#4f4e4e')
            plt.yticks(size=14, color='#4f4e4e')
            plt.legend()
            plt.ylim(0, 1)
            os.makedirs('classes/plots', exist_ok=True)
            sns.despine(left=True)
            plt.legend(
                loc='upper left',
                frameon=True,
                fontsize=14,
                labelcolor='#4f4e4e'
            )
            plt.savefig(f"classes/plots/{self.freq}_T{self.period}_drift.png", dpi=300)
            plt.close()
