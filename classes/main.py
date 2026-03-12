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
    engagement.build_markov_transition_matrix()
    precision_markov, f1_markov, ll_markov = engagement.test_markov()
    
    engagement.train_classifier()
    precision_gbc, f1_gbc, ll_gbc = engagement.test_classifier()
    engagement.plot_confusion_matrix_T1(model='model')
    engagement.plot_confusion_matrix_T1(model='markov')

    print("\nPredict engagement state in next T")
    if benchmarks:
        # Run naive repeating, global-level mode, entity-level mode
        engagement.run_benchmarks()
    else:
        model_naive_results = {
            'Model': ['Markov Matrix', 'GradientBoostCl'],
            'Precision': [precision_markov, precision_gbc],
            'F1': [f1_markov, f1_gbc],
            'Log-loss': [ll_markov, ll_gbc]
        }
        display(pd.DataFrame(model_naive_results))

    # Step 4-5-6
    value_model = ValueForecastingModel(
        train_df=engagement.train_df,
        test_df=engagement.test_df,
        markov_matrix=engagement.transition_matrix,
        model=engagement.clf,
        freq=data_preparator.freq,
        period=data_preparator.period
    )
    # Step 4
    # Forecast engagement state X months ahead
    value_model.markov_forecast_engagement()
    value_model.model_forecast_engagement()
    value_model.step4_metrics()

    # Step 5A - 6
    value_model.apply_empirical_state_averages()

    # Step 5B - 6
    value_model.train_state_regressors()
    value_model.apply_state_regressors()

    value_model.regressor_metrics()
    value_model.compute_expected_value_from_regressors()

    value_model.step5_6_metrics()
    value_model.compare_state_distributions(plot_cm=True, plot_drift=True)
