# Imports
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from sklearn.preprocessing import add_dummy_feature
from sklearn.linear_model import LinearRegression
import matplotlib as mpl

# Get Dataset from URL and read it into pandas dataframe
dataset_url = "https://raw.githubusercontent.com/plotly/datasets/master/imports-85.csv"
df = pd.read_csv(dataset_url)
np.random.seed(42)  # The answer to univserse and everything
pd.set_option('display.max_columns', None)

# Initialize variables with information, to including the feature columns 2 data points, curb weight and engine size
target_variable_column = df[['city-mpg']]
feature_columns = df[['curb-weight', 'engine-size']]

# Initializing where to save images for save_fig()
IMAGES_PATH = Path() / "images" / "training_linear_models"
IMAGES_PATH.mkdir(parents=True, exist_ok=True)


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = IMAGES_PATH / f"{fig_id}.{fig_extension}"
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

### Linear Regression ###


# Formatting data like the link so it fits
m = len(df)
X = target_variable_column.values
y = feature_columns.values

# Plotting first points to see curb weight and engine size in relation to MPG
plt.figure(figsize=(6, 4))
plt.plot(X, y, "b.")
plt.xlabel("$City MPG$")
plt.ylabel("$CW and ES$", rotation=0)
plt.grid()
save_fig("generated_data_plot")
plt.show()

# Adding dummy feature for data size and Matmul
X_b = add_dummy_feature(X)
theta_best = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y

# Setting line(s) to predict
X_new = np.array([[0], [50]])
X_new_b = add_dummy_feature(X_new)
y_predict = X_new_b @ theta_best
y_predict  # Prediction

# Plot with prediction lines from
plt.figure(figsize=(6, 4))
plt.plot(X_new, y_predict, "r-", label="Predictions")
plt.plot(X, y, "b.")
plt.xlabel("$City MPG$")
plt.ylabel("$CW and ES$", rotation=0)
plt.grid()
save_fig("linear_model_predictions_plot")
plt.show()

# Linear Regression Setup
lin_reg = LinearRegression()
lin_reg.fit(X, y)
lin_reg.intercept_, lin_reg.coef_
lin_reg.predict(X_new)
theta_best_svd, residuals, rank, s = np.linalg.lstsq(X_b, y, rcond=1e-6)
np.linalg.pinv(X_b) @ y

# Setup for the image below
theta_path_sgd = []
n_epochs = 50
t0, t1 = 5, 50


def learning_schedule(t):
    return t0 / (t + t1)


# Random Initialization
theta = np.random.randn(2, 1)

n_shown = 20
plt.figure(figsize=(6, 4))

for epoch in range(n_epochs):
    for iteration in range(m):
        # Draw prediction lines on graph
        if epoch == 0 and iteration < n_shown:
            y_predict = X_new_b @ theta_best
            color = mpl.colors.rgb2hex(plt.cm.OrRd(iteration / n_shown + 0.15))
            plt.plot(X_new, y_predict, color=color)

        # Plot points
        random_index = np.random.randint(m)
        xi = X_b[random_index: random_index + 1]
        yi = y[random_index: random_index + 1]

        # Plot gradient lines from above
        gradients = 2 * xi.T @ (xi @ theta - yi)
        eta = learning_schedule(epoch * m + iteration)
        theta = theta - eta * gradients

        # Generate Figure
        theta_path_sgd.append(theta)

# Beautify the plot
plt.plot(X, y, "b.")
plt.xlabel("$City MPG$")
plt.ylabel("$CW and ES$", rotation=0)
plt.grid()
plt.show()
