import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow import keras
import pandas as pd
from itertools import product
import csv

TF_ENABLE_ONEDNN_OPTS=0


# We define the function f which we would like to approximate
def f(x,y):
    return (x-0.5) ** 2 + (y-0.5) ** 2

# 'outputgenerator' calculated f on a given input set and saves the corresponding output set
def outputgenerator(inputarray):
    outputs = list()
    inputarray2 = np.array(inputarray, dtype=float)
    i=0
    while  i < len(inputarray2):
        outputs.append(round((inputarray2[i, 0]-0.5) ** 2 + (inputarray2[i, 1]-0.5) ** 2,5))
        i += 1
    return list(outputs)


# Generate points arranged in a grid and filter those inside the ball
# We do this to see whether there are differences depending on the structure of the training set (random vs. grid)

points_per_dim = 100

grid_values = np.linspace(0, 1, points_per_dim)
grid_check = np.linspace(0, 1, points_per_dim*5)
grid_points = np.array(list(product(grid_values, repeat=2)))
grid_points_check = np.array(list(product(grid_values, repeat=2)))

center = np.array([0.5] * 2)

radius = 0.5

distances = np.linalg.norm(grid_points - center, axis=1)

filtered_points = grid_points[distances <= radius]
filtered_points_check = grid_points_check[distances <= radius]

grid_input = np.array(filtered_points)
grid_check = np.array(filtered_points_check)
grid_output = np.array(outputgenerator(grid_input))


# 'infinity_norm_difference' calculates the difference of a network to another function on a given input set.
def infinity_norm_difference(network, func, inputs):

    inputs = np.array(inputs)
    x, y = inputs[:, 0], inputs[:, 1]

    network_outputs = network(inputs).numpy()
    if len(network_outputs.shape) > 1:
        network_outputs = network_outputs.flatten()

    func_outputs = func(x, y)

    differences = np.abs(network_outputs - func_outputs)

    argmax_index = np.argmax(differences)

    argmax_point = inputs[argmax_index]

    return np.max(differences), tuple(argmax_point)

# We define the three different architectures used for comparison
def create_model_depth1():
    model = tf.keras.Sequential([
        keras.layers.Input(shape=(2,)),
        keras.layers.Dense(units=3, activation='relu'),
        keras.layers.Dense(units=1, activation='linear')
    ])
    return model

def create_model_depth2():
    model = tf.keras.Sequential([
        keras.layers.Input(shape=(2,)),
        keras.layers.Dense(units=3, activation='relu'),
        keras.layers.Dense(units=3, activation='relu'),
        keras.layers.Dense(units=1, activation='linear')
    ])
    return model

def create_model_depth8():
    model = tf.keras.Sequential([
        keras.layers.Input(shape=(2,)),
        keras.layers.Dense(units=3, activation='relu'),
        keras.layers.Dense(units=3, activation='relu'),
        keras.layers.Dense(units=3, activation='relu'),
        keras.layers.Dense(units=3, activation='relu'),
        keras.layers.Dense(units=3, activation='relu'),
        keras.layers.Dense(units=3, activation='relu'),
        keras.layers.Dense(units=3, activation='relu'),
        keras.layers.Dense(units=3, activation='relu'),
        keras.layers.Dense(units=1, activation='linear')
    ])
    return model

# Training settings
epochs = 100
batch_size = 4
loss_fn = tf.keras.losses.MeanSquaredError()

# Next we define some empty arrays to save all the relevant data during training

loss_log_grid1 = []
loss_log_grid2 = []
loss_log_grid8 = []

all_params = []

infinity_norms_grid1 = []
infinity_norms_grid2 = []
infinity_norms_grid8 = []

# Our training routine: We train three networks ten times for 50 epochs
for i in range(10):
    model_depth1_grid = create_model_depth1()
    model_depth2_grid = create_model_depth2()
    model_depth8_grid = create_model_depth8()

    model_depth1_grid.compile(optimizer='Adam', loss=loss_fn)
    model_depth2_grid.compile(optimizer='Adam', loss=loss_fn)
    model_depth8_grid.compile(optimizer='Adam', loss=loss_fn)

    # Training each epoch individually to save norm and argmax for each epoch
    for epoch in range(epochs):
        history1 = model_depth1_grid.fit(grid_input, grid_output, epochs=1, batch_size=batch_size, verbose=0)
        history2 = model_depth2_grid.fit(grid_input, grid_output, epochs=1, batch_size=batch_size, verbose=0)
        history3 = model_depth8_grid.fit(grid_input, grid_output, epochs=1, batch_size=batch_size, verbose=0)
        loss1 = history1.history["loss"][0]
        loss2 = history2.history["loss"][0]
        loss3 = history3.history["loss"][0]
        sup_norm1, argmax_point1 = infinity_norm_difference(model_depth1_grid, f, grid_check)
        sup_norm2, argmax_point2 = infinity_norm_difference(model_depth2_grid, f, grid_check)
        sup_norm3, argmax_point3 = infinity_norm_difference(model_depth8_grid, f, grid_check)
        loss_log_grid1.append([i, epoch + 1, loss1, sup_norm1, *argmax_point1])
        loss_log_grid2.append([i, epoch + 1, loss2, sup_norm2, *argmax_point2])
        loss_log_grid8.append([i, epoch + 1, loss3, sup_norm3, *argmax_point3])



    # Extract all parameters into one array

    params1 = model_depth1_grid.get_weights()
    params_flat1 = [p.flatten() for p in params1]
    param_series1 = pd.Series(np.concatenate(params_flat1), name=f'model_{i + 1}_depth1_grid')

    params2 = model_depth2_grid.get_weights()
    params_flat2 = [p.flatten() for p in params2]
    param_series2 = pd.Series(np.concatenate(params_flat2), name=f'model_{i + 1}_depth2_grid')

    params3 = model_depth8_grid.get_weights()
    params_flat3 = [p.flatten() for p in params3]
    param_series3 = pd.Series(np.concatenate(params_flat3), name=f'model_{i + 1}_depth8_grid')

    all_params.append(param_series1)
    all_params.append(param_series2)
    all_params.append(param_series3)

    # Calculate the infinity norm and maximizers after training

    max_diff, argmax_diff = infinity_norm_difference(model_depth1_grid, f, grid_check)
    infinity_norms_grid1.append([i + 1, max_diff, argmax_diff[0], argmax_diff[1]])

    max_diff, argmax_diff = infinity_norm_difference(model_depth2_grid, f, grid_check)
    infinity_norms_grid2.append([i + 1, max_diff, argmax_diff[0], argmax_diff[1]])

    max_diff, argmax_diff = infinity_norm_difference(model_depth8_grid, f, grid_check)
    infinity_norms_grid8.append([i + 1, max_diff, argmax_diff[0], argmax_diff[1]])


# Save loss + norm each epoch

with open("../Loss grid depth 1 width 3 dim 2.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Run", "Epoch", "Loss", "Supnorm", "Argmax_x1", "Argmax_x2"])
    writer.writerows(loss_log_grid1)

with open("../Loss grid depth 2 width 3 dim 2.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Run", "Epoch", "Loss", "Supnorm", "Argmax_x1", "Argmax_x2"])
    writer.writerows(loss_log_grid2)

with open("../Loss grid depth 8 width 3 dim 2.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Run", "Epoch", "Loss", "Supnorm", "Argmax_x1", "Argmax_x2"])
    writer.writerows(loss_log_grid8)



# Save the model parameters into a csv
param_df_final = pd.concat(all_params, axis=1)
param_df_final.to_csv("all_params_width3_dim2.csv", index=False)

# Save infinity norm and maximizers into a csv

infinity_norm_df = pd.DataFrame(infinity_norms_grid1, columns=["Number", "Max Diff", "Argmax X", "Argmax Y"])
infinity_norm_df.to_csv("NormArgmaxGrid1width3_dim2.csv", index=False)

infinity_norm_df = pd.DataFrame(infinity_norms_grid2, columns=["Number", "Max Diff", "Argmax X", "Argmax Y"])
infinity_norm_df.to_csv("NormArgmaxGrid2width3_dim2.csv", index=False)

infinity_norm_df = pd.DataFrame(infinity_norms_grid8, columns=["Number", "Max Diff", "Argmax X", "Argmax Y"])
infinity_norm_df.to_csv("NormArgmaxGrid8width3_dim2.csv", index=False)

print("Training finished and results saved.")