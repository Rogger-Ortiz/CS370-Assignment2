import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from sklearn.preprocessing import add_dummy_feature
from sklearn.linear_model import LinearRegression
import matplotlib as mpl

dataset_url = "https://raw.githubusercontent.com/plotly/datasets/master/imports-85.csv"
df = pd.read_csv(dataset_url)
np.random.seed(42)
pd.set_option('display.max_columns', None)

target_variable_column = df[['city-mpg']]
feature_columns = df[['curb-weight', 'engine-size']]

IMAGES_PATH = Path() / "images" / "training_linear_models"
IMAGES_PATH.mkdir(parents=True, exist_ok=True)


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = IMAGES_PATH / f"{fig_id}.{fig_extension}"
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

### Linear Regression ###


m = len(df)
X = target_variable_column.values
y = feature_columns.values

plt.figure(figsize=(6, 4))
plt.plot(X, y, "b.")
plt.xlabel("$City MPG$")
plt.ylabel("$CW and ES$", rotation=0)
plt.grid()
save_fig("generated_data_plot")
plt.show()

X_b = add_dummy_feature(X)  # add x0 = 1 to each instance
theta_best = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y

X_new = np.array([[0], [50]])
X_new_b = add_dummy_feature(X_new)
y_predict = X_new_b @ theta_best
y_predict

plt.figure(figsize=(6, 4))  # extra code – not needed, just formatting
plt.plot(X_new, y_predict, "r-", label="Predictions")
plt.plot(X, y, "b.")
plt.xlabel("$City MPG$")
plt.ylabel("$CW and ES$", rotation=0)
plt.grid()
save_fig("linear_model_predictions_plot")
plt.show()

lin_reg = LinearRegression()
lin_reg.fit(X, y)
lin_reg.intercept_, lin_reg.coef_
lin_reg.predict(X_new)
theta_best_svd, residuals, rank, s = np.linalg.lstsq(X_b, y, rcond=1e-6)
np.linalg.pinv(X_b) @ y

theta_path_sgd = []
n_epochs = 50
t0, t1 = 5, 50  # learning schedule hyperparameters


def learning_schedule(t):
    return t0 / (t + t1)


theta = np.random.randn(2, 1)  # random initialization

n_shown = 20  # extra code – just needed to generate the figure below
plt.figure(figsize=(6, 4))  # extra code – not needed, just formatting
for epoch in range(n_epochs):
    for iteration in range(m):
        # extra code – these 4 lines are used to generate the figure
        if epoch == 0 and iteration < n_shown:
            y_predict = X_new_b @ theta_best
            color = mpl.colors.rgb2hex(plt.cm.OrRd(iteration / n_shown + 0.15))
            plt.plot(X_new, y_predict, color=color)

        random_index = np.random.randint(m)
        xi = X_b[random_index: random_index + 1]
        yi = y[random_index: random_index + 1]
        # for SGD, do not divide by m
        gradients = 2 * xi.T @ (xi @ theta - yi)
        eta = learning_schedule(epoch * m + iteration)
        theta = theta - eta * gradients
        # extra code – to generate the figure
        theta_path_sgd.append(theta)

# extra code – this section beautifies and saves Figure 4–10
plt.plot(X, y, "b.")
plt.grid()
plt.show()
