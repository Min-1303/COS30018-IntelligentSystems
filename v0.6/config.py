# config.py

CONFIG = {
    '1': {
        'epochs': 25,
        'batch_size': 32,
        # LSTM Configuration
        'lstm_layers': 2,
        'lstm_units': 50,
        'lstm_dropout': 0.3,

        # GRU Configuration
        'gru_layers': 2,
        'gru_units': 50,
        'gru_dropout': 0.3,

        # RNN type and Bidirectionality
        'rnn_type': 'LSTM',  # Can be 'LSTM' or 'GRU'
        'use_bidirectional': True,  # True or False

        # ARIMA Configuration
        'arima_order': (1, 1, 0),

        # SARIMA Configuration
        'sarima_order': (1, 1, 0),
        'sarima_seasonal_order': (0, 1, 1, 12),

        # Random Forest Configuration
        'rf_n_estimators': 50,
        'rf_max_depth': 10,
        'rf_min_samples_split': 2,
    },

    '2': {
        'epochs': 30,            # Updated batch size and epochs
        'batch_size': 64,       # Updated batch size and epochs
        
        'lstm_layers': 4,
        'lstm_units': 100,
        'lstm_dropout': 0.2,

        'gru_layers': 4,
        'gru_units': 100,
        'gru_dropout': 0.2,

        # RNN type and Bidirectionality
        'rnn_type': 'GRU',  # Can be 'LSTM' or 'GRU'
        'use_bidirectional': False,  # True or False

        'arima_order': (3, 1, 2),

        'sarima_order': (2, 1, 2),
        'sarima_seasonal_order': (1, 1, 1, 12),

        'rf_n_estimators': 150,
        'rf_max_depth': 20,
        'rf_min_samples_split': 2,
    },

    '3': {
        'epochs': 25,            # Updated batch size and epochs
        'batch_size': 64,       # Updated batch size and epochs
        'lstm_layers': 3,
        'lstm_units': 75,
        'lstm_dropout': 0.5,

        'gru_layers': 3,
        'gru_units': 75,
        'gru_dropout': 0.5,

        # RNN type and Bidirectionality
        'rnn_type': 'LSTM',  # Can be 'LSTM' or 'GRU'
        'use_bidirectional': False,  # True or False

        'arima_order': (5, 1, 1),

        'sarima_order': (1, 1, 1),
        'sarima_seasonal_order': (1, 1, 0, 6),

        'rf_n_estimators': 100,
        'rf_max_depth': 15,
        'rf_min_samples_split': 5,
    },

    '4': {
        'epochs': 15,            # Updated batch size and epochs
        'batch_size': 8,        # Updated batch size and epochs
        'lstm_layers': 1,
        'lstm_units': 50,
        'lstm_dropout': 0.1,

        'gru_layers': 1,
        'gru_units': 50,
        'gru_dropout': 0.1,

        # RNN type and Bidirectionality
        'rnn_type': 'GRU',  # Can be 'LSTM' or 'GRU'
        'use_bidirectional': True,  # True or False

        'arima_order': (1, 1, 1),

        'sarima_order': (1, 1, 0),
        'sarima_seasonal_order': (0, 1, 0, 6),

        'rf_n_estimators': 200,
        'rf_max_depth': 5,
        'rf_min_samples_split': 10,
    }
}

