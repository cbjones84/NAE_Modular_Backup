"""
Auto-implemented improvement from GitHub
Source: dmackenz/Keras-Neuro-Evolution-Trading-Bot-Skeleton/evolution.py
Implemented: 2025-12-09T10:59:22.604339
Usefulness Score: 100
Keywords: def , optimize, model, loss, size, loss
"""

# Original source: dmackenz/Keras-Neuro-Evolution-Trading-Bot-Skeleton
# Path: evolution.py


# Function: build_model
def build_model():
    num_inputs = 4
    hidden_nodes = 16
    num_outputs = 3

    model = Sequential()
    model.add(Dense(hidden_nodes, activation='relu', input_dim=num_inputs))
    model.add(Dense(num_outputs, activation='softmax'))
    model.compile(loss='mse', optimizer='adam')
    
    return model


