import numpy as np
import matplotlib.pyplot as plt


def model(x, w, b):
    return w * x + b


def mse(y, y_hat):
    return np.mean((y - y_hat) ** 2)

def gradient(x, y, y_hat):
    n = len(x)

    error = y - y_hat

    dw = -(2 / n) * np.sum(x * error)
    db = -(2 / n) * np.sum(error)

    return dw, db


def train(x, y, epochs = 1000, learning_rate = 0.01):
    w, b = 0.0, 0.0
    losses = []

    for epoch in range(epochs):
        y_hat = model(x, w, b)
        loss = mse(y, y_hat)
        losses.append(loss)

        dw, db = gradient(x, y, y_hat)
        w = w - learning_rate * dw
        b = b - learning_rate * db

        if epoch % 100 == 0:
            print(f"epoch: {epoch}, loss: {loss:.4f}, w: {w:.4f}, b: {b:.4f}")

    return w, b, losses

x = np.array([50, 100, 150, 200], dtype=float)
y = np.array([1000, 2000, 3000, 4000], dtype=float)

x_max = np.max(x)
y_max = np.max(y)

x_norm = x / x_max
y_norm = y / y_max

w_norm, b_norm, losses = train(x_norm, y_norm, epochs = 5000)

w = w_norm * (y_max / x_max)
b = b_norm * y_max

print(f"\nFinal model:")
print(f"y = {w:.4f}x + {b:.4f}")

plt.plot(losses)
plt.xlabel("Epochs")
plt.ylabel("Loss (MSE)")
plt.title("Training Loss")
plt.show()