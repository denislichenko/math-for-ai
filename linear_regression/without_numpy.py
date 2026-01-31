import matplotlib.pyplot as plt

# -------------------------------
# Training data
# x — input feature (independent variable)
# y — target value we want to predict
# -------------------------------
x = [50, 100, 150, 200]
y = [1000, 2000, 3000, 4000]


# ======================================================
# MODEL FUNCTION
# ======================================================
def model(x, w, b):
    """
    Linear regression model: y_hat = w * x + b

    Why this function is needed:
    - This is our hypothesis (assumption) about how x relates to y
    - We assume the relationship is linear
    - The model produces predictions (y_hat) based on current parameters

    Parameters:
    - x: input data (list of numbers)
    - w: weight (slope of the line)
    - b: bias (intercept)

    Returns:
    - y_hat: predicted values for each x
    """
    return [w * xi + b for xi in x]


# ======================================================
# LOSS FUNCTION (MSE)
# ======================================================
def mse(y, y_hat):
    """
    Mean Squared Error (MSE)

    Why this function is needed:
    - The model must know how 'bad' its predictions are
    - Loss converts prediction error into a single number
    - Squaring penalizes large errors more than small ones
    - Averaging makes loss independent of dataset size

    Formula:
    MSE = (1 / n) * Σ(yi - y_hat_i)^2

    Parameters:
    - y: true target values
    - y_hat: predicted values from the model

    Returns:
    - scalar loss value
    """
    n = len(y)
    squared_errors = [(yi - y_hati) ** 2 for yi, y_hati in zip(y, y_hat)]
    return sum(squared_errors) / n


# ======================================================
# GRADIENT COMPUTATION
# ======================================================
def gradient(x, y, y_hat):
    """
    Computes gradients of the MSE loss with respect to w and b

    Why this function is needed:
    - Gradient tells us how to change parameters to reduce loss
    - dw = how much loss changes if w changes
    - db = how much loss changes if b changes
    - These values guide gradient descent updates

    Derived gradients for linear regression + MSE:
    ∂L/∂w = -(2/n) * Σ xi * (yi - y_hat_i)
    ∂L/∂b = -(2/n) * Σ (yi - y_hat_i)

    Parameters:
    - x: input values
    - y: true target values
    - y_hat: model predictions

    Returns:
    - dw: gradient w.r.t. weight
    - db: gradient w.r.t. bias
    """
    n = len(x)

    # Gradient of loss with respect to w
    dw = -(2 / n) * sum(
        [xi * (yi - y_hati) for xi, yi, y_hati in zip(x, y, y_hat)]
    )

    # Gradient of loss with respect to b
    db = -(2 / n) * sum(
        [yi - y_hati for yi, y_hati in zip(y, y_hat)]
    )

    return dw, db


# ======================================================
# TRAINING LOOP (GRADIENT DESCENT)
# ======================================================
def train(x, y, epochs=100, learning_rate=0.00001):
    """
    Trains the linear regression model using gradient descent

    Why this function is needed:
    - This is the optimization process
    - Repeatedly:
        1. Make predictions
        2. Measure error
        3. Compute gradients
        4. Update parameters
    - Over time, w and b converge to values that minimize loss

    Parameters:
    - x: training inputs
    - y: training targets
    - epochs: number of optimization steps
    - learning_rate: step size for gradient descent

    Returns:
    - w: learned weight
    - b: learned bias
    - losses: history of loss values
    """
    # Initialize parameters (model knows nothing at start)
    w, b = 0.0, 0.0
    losses = []

    for epoch in range(epochs):
        # Step 1: forward pass (prediction)
        y_hat = model(x, w, b)

        # Step 2: compute loss
        loss = mse(y, y_hat)
        losses.append(loss)

        # Step 3: compute gradients
        dw, db = gradient(x, y, y_hat)

        # Step 4: update parameters (gradient descent step)
        w -= learning_rate * dw
        b -= learning_rate * db

        # Logging for debugging and learning
        if epoch % 100 == 0:
            print(
                f"epoch: {epoch}, "
                f"loss: {loss:.6f}, "
                f"w: {w:.6f}, "
                f"b: {b:.6f}"
            )

    return w, b, losses


# ======================================================
# NORMALIZATION
# ======================================================
# Why normalize:
# - Prevents large values from causing unstable gradients
# - Speeds up convergence
# - Makes learning rate easier to choose
max_x = max(x)
max_y = max(y)

x_norm = [xi / max_x for xi in x]
y_norm = [yi / max_y for yi in y]


# ======================================================
# TRAIN ON NORMALIZED DATA
# ======================================================
w_norm, b_norm, losses = train(
    x_norm,
    y_norm,
    epochs=3000,
    learning_rate=0.01
)


# ======================================================
# DENORMALIZATION
# ======================================================
# Convert parameters back to original scale
w = w_norm * (max_y / max_x)
b = b_norm * max_y

print(f"Final model: y = {w:.4f}x + {b:.4f}")


# ======================================================
# VISUALIZATION
# ======================================================
# Plot loss to verify convergence
plt.plot(losses)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss (MSE)")
plt.show()
