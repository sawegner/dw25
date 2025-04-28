import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow import keras
import pandas as pd
from itertools import product
import csv

TF_ENABLE_ONEDNN_OPTS=0

# Generate random uniformly distributed points in the ball around (1/2, 1/2) with radius 1/2 in dimension 2

# Set the number of points
num_points = 10000
dimension = 2

# Calculate random uniformly distributed points in [0,1]^5 for training data as well as data to calculate the infinity norm later
points1 = np.random.rand(num_points, dimension)
points2 = np.random.rand(num_points*10, dimension)

# Center of the ball
center = np.array([0.5] * dimension)

# Radius of the ball
radius = 0.5

# Calculate euclidean distances
distances1 = np.linalg.norm(points1 - center, axis=1)
distances2 = np.linalg.norm(points2 - center, axis=1)

# Filter out the points inside the ball
filtered_points1 = points1[distances1 <= radius]
filtered_points2 = points2[distances2 <= radius]



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

# To avoid our programm calulating the input and output data new for each compiling, we save the data in the form of numpy files.

Input_path_dim2 = 'Random_Input_dim2.npy'
Input_path_dim2_check = 'Random_Input_dim2_check.npy'

if os.path.exists(Input_path_dim2):

    random_input1 = np.load(Input_path_dim2)
    print("Inputs1 loaded.")

if os.path.exists(Input_path_dim2_check):

    random_input2 = np.load(Input_path_dim2_check)
    print("Inputs2 loaded.")
else:

    random_input1 = np.array(filtered_points1)
    random_input2 = np.array(filtered_points2)
    np.save(Input_path_dim2, random_input1)
    np.save(Input_path_dim2_check, random_input2)
    print("Inputs1+2 generated and saved.")


Output_path_dim2 = 'Random_Output1.npy'
Output_path_dim2_check = 'Random_Output2.npy'

if os.path.exists(Output_path_dim2):

    random_output1 = np.load(Output_path_dim2)
    print("Outputs1 loaded.")

if os.path.exists(Output_path_dim2_check):

    random_output2 = np.load(Output_path_dim2_check)
    print("Outputs2 loaded.")

else:

    random_output1 = np.array(outputgenerator(random_input1))
    random_output2 = np.array(outputgenerator(random_input2))
    np.save(Output_path_dim2, random_output1)
    np.save(Output_path_dim2_check, random_output2)
    print("Outputs1+2 generated and saved.")

# Generate points arranged in a grid and filter those inside the ball
# We do this to see whether there are differences depending on the structure of the training set (random vs. grid)

points_per_dim = 100

grid_values = np.linspace(0, 1, points_per_dim)
grid_points = np.array(list(product(grid_values, repeat=2)))

center = np.array([0.5] * 2)

radius = 0.5

distances = np.linalg.norm(grid_points - center, axis=1)

filtered_points = grid_points[distances <= radius]
print(filtered_points.shape)
grid_input = np.array(filtered_points)
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
        keras.layers.Dense(units=2, activation='relu'),
        keras.layers.Dense(units=1, activation='linear')
    ])
    return model

def create_model_depth2():
    model = tf.keras.Sequential([
        keras.layers.Input(shape=(2,)),
        keras.layers.Dense(units=2, activation='relu'),
        keras.layers.Dense(units=2, activation='relu'),
        keras.layers.Dense(units=1, activation='linear')
    ])
    return model

def create_model_depth8():
    model = tf.keras.Sequential([
        keras.layers.Input(shape=(2,)),
        keras.layers.Dense(units=2, activation='relu'),
        keras.layers.Dense(units=2, activation='relu'),
        keras.layers.Dense(units=2, activation='relu'),
        keras.layers.Dense(units=2, activation='relu'),
        keras.layers.Dense(units=2, activation='relu'),
        keras.layers.Dense(units=2, activation='relu'),
        keras.layers.Dense(units=2, activation='relu'),
        keras.layers.Dense(units=2, activation='relu'),
        keras.layers.Dense(units=1, activation='linear')
    ])
    return model

# Training settings
epochs = 50
batch_size = 1
loss_fn = tf.keras.losses.MeanSquaredError()

# Next we define some empty arrays to save all the relevant data during training
loss_log_rand1 = []
loss_log_rand2 = []
loss_log_rand8 = []
loss_log_grid1 = []
loss_log_grid2 = []
loss_log_grid8 = []

all_params = []

infinity_norms_rand1 = []
infinity_norms_rand2 = []
infinity_norms_rand8 = []
infinity_norms_grid1 = []
infinity_norms_grid2 = []
infinity_norms_grid8 = []

# Our training routine: We train our six networks ten times for 50 epochs
for i in range(10):
    model_depth1_random = create_model_depth1()
    model_depth2_random = create_model_depth2()
    model_depth8_random = create_model_depth8()
    model_depth1_grid = create_model_depth1()
    model_depth2_grid = create_model_depth2()
    model_depth8_grid = create_model_depth8()

    model_depth1_random.compile(optimizer='Adam', loss=loss_fn)
    model_depth2_random.compile(optimizer='Adam', loss=loss_fn)
    model_depth8_random.compile(optimizer='Adam', loss=loss_fn)
    model_depth1_grid.compile(optimizer='Adam', loss=loss_fn)
    model_depth2_grid.compile(optimizer='Adam', loss=loss_fn)
    model_depth8_grid.compile(optimizer='Adam', loss=loss_fn)

    # Training each epoch individually to save norm and argmax for each epoch
    for epoch in range(epochs):
        history1 = model_depth1_random.fit(random_input1, random_output1, epochs=1, batch_size=batch_size, verbose=0)
        history2 = model_depth2_random.fit(random_input1, random_output1, epochs=1, batch_size=batch_size, verbose=0)
        history3 = model_depth8_random.fit(random_input1, random_output1, epochs=1, batch_size=batch_size, verbose=0)
        history4 = model_depth1_grid.fit(grid_input, grid_output, epochs=1, batch_size=batch_size, verbose=0)
        history5 = model_depth2_grid.fit(grid_input, grid_output, epochs=1, batch_size=batch_size, verbose=0)
        history6 = model_depth8_grid.fit(grid_input, grid_output, epochs=1, batch_size=batch_size, verbose=0)
        loss1 = history1.history["loss"][0]
        loss2 = history2.history["loss"][0]
        loss3 = history3.history["loss"][0]
        loss4 = history4.history["loss"][0]
        loss5 = history5.history["loss"][0]
        loss6 = history6.history["loss"][0]
        sup_norm1, argmax_point1 = infinity_norm_difference(model_depth1_random, f, random_input2)
        sup_norm2, argmax_point2 = infinity_norm_difference(model_depth2_random, f, random_input2)
        sup_norm3, argmax_point3 = infinity_norm_difference(model_depth8_random, f, random_input2)
        sup_norm4, argmax_point4 = infinity_norm_difference(model_depth1_grid, f, random_input2)
        sup_norm5, argmax_point5 = infinity_norm_difference(model_depth2_grid, f, random_input2)
        sup_norm6, argmax_point6 = infinity_norm_difference(model_depth8_grid, f, random_input2)
        loss_log_rand1.append([i, epoch + 1, loss1, sup_norm1, *argmax_point1])
        loss_log_rand2.append([i, epoch + 1, loss2, sup_norm2, *argmax_point2])
        loss_log_rand8.append([i, epoch + 1, loss3, sup_norm3, *argmax_point3])
        loss_log_grid1.append([i, epoch + 1, loss4, sup_norm4, *argmax_point4])
        loss_log_grid2.append([i, epoch + 1, loss5, sup_norm5, *argmax_point5])
        loss_log_grid8.append([i, epoch + 1, loss6, sup_norm6, *argmax_point6])


    # Extract all parameters into one array
    params1 = model_depth1_random.get_weights()
    params_flat1 = [p.flatten() for p in params1]
    param_series1 = pd.Series(np.concatenate(params_flat1), name=f'model_{i + 1}_depth1_random')

    params2 = model_depth2_random.get_weights()
    params_flat2 = [p.flatten() for p in params2]
    param_series2 = pd.Series(np.concatenate(params_flat2), name=f'model_{i + 1}_depth2_random')

    params3 = model_depth8_random.get_weights()
    params_flat3 = [p.flatten() for p in params3]
    param_series3 = pd.Series(np.concatenate(params_flat3), name=f'model_{i + 1}_depth8_random')

    params4 = model_depth1_grid.get_weights()
    params_flat4 = [p.flatten() for p in params4]
    param_series4 = pd.Series(np.concatenate(params_flat4), name=f'model_{i + 1}_depth1_grid')

    params5 = model_depth2_grid.get_weights()
    params_flat5 = [p.flatten() for p in params5]
    param_series5 = pd.Series(np.concatenate(params_flat5), name=f'model_{i + 1}_depth2_grid')

    params6 = model_depth8_grid.get_weights()
    params_flat6 = [p.flatten() for p in params6]
    param_series6 = pd.Series(np.concatenate(params_flat6), name=f'model_{i + 1}_depth8_grid')

    all_params.append(param_series1)
    all_params.append(param_series2)
    all_params.append(param_series3)
    all_params.append(param_series4)
    all_params.append(param_series5)
    all_params.append(param_series6)

    # Calculate the infinity norm and maximizers after training
    max_diff, argmax_diff = infinity_norm_difference(model_depth1_random, f, random_input2)
    infinity_norms_rand1.append([i + 1, max_diff, argmax_diff[0], argmax_diff[1]])

    max_diff, argmax_diff = infinity_norm_difference(model_depth2_random, f, random_input2)
    infinity_norms_rand2.append([i + 1, max_diff, argmax_diff[0], argmax_diff[1]])

    max_diff, argmax_diff = infinity_norm_difference(model_depth8_random, f, random_input2)
    infinity_norms_rand8.append([i + 1, max_diff, argmax_diff[0], argmax_diff[1]])

    max_diff, argmax_diff = infinity_norm_difference(model_depth1_grid, f, random_input2)
    infinity_norms_grid1.append([i + 1, max_diff, argmax_diff[0], argmax_diff[1]])

    max_diff, argmax_diff = infinity_norm_difference(model_depth2_grid, f, random_input2)
    infinity_norms_grid2.append([i + 1, max_diff, argmax_diff[0], argmax_diff[1]])

    max_diff, argmax_diff = infinity_norm_difference(model_depth8_grid, f, random_input2)
    infinity_norms_grid8.append([i + 1, max_diff, argmax_diff[0], argmax_diff[1]])


# Save loss + norm each epoch
with open("Loss random depth 1 dim 2.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Run", "Epoch", "Loss", "Supnorm", "Argmax_x1", "Argmax_x2"])
    writer.writerows(loss_log_rand1)

with open("Loss random depth 2 dim 2.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Run", "Epoch", "Loss", "Supnorm", "Argmax_x1", "Argmax_x2"])
        writer.writerows(loss_log_rand2)

with open("Loss random depth 8 dim 2.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Run", "Epoch", "Loss", "Supnorm", "Argmax_x1", "Argmax_x2"])
    writer.writerows(loss_log_rand8)

with open("Loss grid depth 1 dim 2.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Run", "Epoch", "Loss", "Supnorm", "Argmax_x1", "Argmax_x2"])
    writer.writerows(loss_log_grid1)

with open("Loss grid depth 2 dim 2.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Run", "Epoch", "Loss", "Supnorm", "Argmax_x1", "Argmax_x2"])
    writer.writerows(loss_log_grid2)

with open("Loss grid depth 8 dim 2.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Run", "Epoch", "Loss", "Supnorm", "Argmax_x1", "Argmax_x2"])
    writer.writerows(loss_log_grid8)



# Save the model parameters into a csv
param_df_final = pd.concat(all_params, axis=1)
param_df_final.to_csv("all_params_dim2.csv", index=False)

# Save infinity norm and maximizers into a csv
infinity_norm_df = pd.DataFrame(infinity_norms_rand1, columns=["Number", "Max Diff", "Argmax X", "Argmax Y"])
infinity_norm_df.to_csv("NormArgmaxRandom1_dim2.csv", index=False)

infinity_norm_df = pd.DataFrame(infinity_norms_rand2, columns=["Number", "Max Diff", "Argmax X", "Argmax Y"])
infinity_norm_df.to_csv("NormArgmaxRandom2_dim2.csv", index=False)

infinity_norm_df = pd.DataFrame(infinity_norms_rand8, columns=["Number", "Max Diff", "Argmax X", "Argmax Y"])
infinity_norm_df.to_csv("NormArgmaxRandom8_dim2.csv", index=False)

infinity_norm_df = pd.DataFrame(infinity_norms_grid1, columns=["Number", "Max Diff", "Argmax X", "Argmax Y"])
infinity_norm_df.to_csv("NormArgmaxGrid1_dim2.csv", index=False)

infinity_norm_df = pd.DataFrame(infinity_norms_grid2, columns=["Number", "Max Diff", "Argmax X", "Argmax Y"])
infinity_norm_df.to_csv("NormArgmaxGrid2_dim2.csv", index=False)

infinity_norm_df = pd.DataFrame(infinity_norms_grid8, columns=["Number", "Max Diff", "Argmax X", "Argmax Y"])
infinity_norm_df.to_csv("NormArgmaxGrid8_dim2.csv", index=False)

print("Training finished and results saved.")