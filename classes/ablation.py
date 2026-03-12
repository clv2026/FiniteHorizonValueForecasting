from preprocessing import DataPreprocessing, Dataset
from engage2value import EngagementStateModel, ValueForecastingModel
import pandas as pd
from IPython.display import display
import yaml


def load_config(config_path='classes/config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


if __name__ == '__main__':
    config = load_config()

    period = config['period']
    freq = config['freq']
    test_mode = config['test_mode']
    data_path = config['data_path']
    data_file = config['data_file']
    save_csv = config['save_csv']
    benchmarks = config['benchmarks']

    data_preparator = DataPreprocessing(
        path=data_path,
        file_name=data_file,
        save=save_csv,
        period=period,
        freq=freq
        )
    data_preparator.preprocess_data()
    df = data_preparator.return_df()

    dataset = Dataset(
        freq=data_preparator.freq,
        period=data_preparator.period,
        df=df,
        test_mode=test_mode
        )
    train_df, test_df = dataset.split_train_test()

    engagement = EngagementStateModel(
        train_df=train_df,
        test_df=test_df,
        freq=data_preparator.freq,
        period=data_preparator.period)

    engagement.train_classifier()
    engagement.test_classifier()

    # Step 4-5-6
    value_model = ValueForecastingModel(
        train_df=engagement.train_df,
        test_df=engagement.test_df,
        markov_matrix=engagement.transition_matrix,
        model=engagement.clf,
        freq=data_preparator.freq,
        period=data_preparator.period,
        ablation=True
    )

    # Step 4 - 5B - 6 but with random probs
    print("# Step 4 - 5B - 6 but with random probs")
    value_model.model_forecast_engagement(random_probs=True)
    value_model.train_state_regressors()
    value_model.apply_state_regressors()
    value_model.compute_expected_value_from_regressors(ablation=True)
    ran_mae, ran_rmse, ran_r2 = value_model.regressor_metrics(ablation=True)

    # Train directly the regressor without modeling engagement
    print("# Train directly the regressor without modeling engagement")
    dir_mae, dir_rmse, dir_r2 = value_model.train_regressor_directly()

    print("# Ablation study on a reduced set of features")
    # Ablation study on a reduced set of features
    df = data_preparator.return_df()

    dataset = Dataset(
        freq=data_preparator.freq,
        period=data_preparator.period,
        df=df,
        test_mode=test_mode
        )
    train_df, test_df = dataset.split_train_test()

    engagement = EngagementStateModel(
        train_df=train_df,
        test_df=test_df,
        freq=data_preparator.freq,
        period=data_preparator.period,
        reduced_features=True)

    engagement.train_classifier()
    engagement.test_classifier()

    # Step 4-5-6
    value_model = ValueForecastingModel(
        train_df=engagement.train_df,
        test_df=engagement.test_df,
        markov_matrix=engagement.transition_matrix,
        model=engagement.clf,
        freq=data_preparator.freq,
        period=data_preparator.period,
        reduced_features=True
    )

    # Step 4 - 5B - 6 but on a reduced set of features
    value_model.model_forecast_engagement()
    value_model.train_state_regressors()
    value_model.apply_state_regressors()
    value_model.compute_expected_value_from_regressors(ablation=True)
    reg_mae, reg_rmse, reg_r2 = value_model.regressor_metrics(ablation=True)

    # Results
    ablation_results = {
        'Model': [
            'Regressor on reduced set of features',
            'Regressor w/ random state probability',
            'Regressor w/out engagement modeling',
        ],
        'MAE': [
            round(reg_mae, 5),
            round(ran_mae, 5),
            round(dir_mae, 5)
        ],
        'RMSE': [
            round(reg_rmse, 5),
            round(ran_rmse, 5),
            round(dir_rmse, 5)
        ],
        'R2': [
            round(reg_r2, 5),
            round(ran_r2, 5),
            round(dir_r2, 5)
        ]
    }

    display(pd.DataFrame(ablation_results))
