"""
Auto-implemented improvement from GitHub
Source: arcarigiorgia/OptionPricing-LSTM-Greeks/main.py
Implemented: 2025-12-09T11:48:21.010311
Usefulness Score: 100
Keywords: def , optimize, tensorflow, sklearn, model, train, predict, fit, loss, volatility, size, stop, loss, greek
"""

# Original source: arcarigiorgia/OptionPricing-LSTM-Greeks
# Path: main.py


# Function: create_lstm_model
def create_lstm_model(input_shape, output_size):
    model = Sequential()
    model.add(LSTM(100, activation='tanh', return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.3))
    model.add(LSTM(50, activation='tanh'))
    model.add(Dropout(0.3))
    model.add(Dense(output_size)) 
    model.compile(optimizer=Adam(learning_rate=0.001), loss=tf.keras.losses.Huber())
    return model

# Visualizzazione Greche

