import tensorflow as tf
import numpy as np

# Your data
data = np.array([
    [2, 4],
    [4, 8],
    [6, 12],
    [8, 16],
    [10, 20],
    [12, 24],
    [14, 28],
    [16, 32],
    [18, 36],
    [20, 40],
    [22, 44],
    [24, 48],
    [26, 52],
    [28, 56]
], dtype=np.float32)

# Splitting the data into features (X) and labels (y)
X_train = data[:, 0]
y_train = data[:, 1]

# Define the neural network architecture
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(1,)),
    tf.keras.layers.Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train, y_train, epochs=1000, verbose=0)

# Make predictions
X_test = np.array([30, 32, 34], dtype=np.float32)
predictions = model.predict(X_test)

# Display predictions
for x, prediction in zip(X_test, predictions):
    print(f"Input: {x}, Predicted Output: {prediction[0]}")
