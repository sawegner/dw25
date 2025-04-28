import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow import keras
import pandas as pd
from itertools import product
import csv

TF_ENABLE_ONEDNN_OPTS=0

# For the case of data in the 5d ball, we need to just the generation process of the points to ensure
# that they are evenly spread across the ball.

def points_in_5d_ball(n_points):
    dim = 5

    # Generate points on the surface of the ball
    normal_deviates = np.random.normal(size=(n_points, dim))
    normal_deviates /= np.linalg.norm(normal_deviates, axis=1)[:, np.newaxis]

    # Generate radii with correct distribution (r^4 for 5D
    radii = np.random.rand(n_points) ** (1 / dim)

    # Scale by radii
    points = normal_deviates * radii[:, np.newaxis]

    # Move the points into the ball
    points = points * 0.5 + 0.5

    return points

random_points = points_in_5d_ball(100000)
random_points_check = points_in_5d_ball(300000)

# Next, we define the function f which we would like to approximate

def f(x1,x2,x3,x4,x5):
    return (x1-0.5) ** 2 + (x2-0.5) ** 2 + (x3-0.5) ** 2 + (x4-0.5) ** 2 + (x5-0.5) ** 2

# The function 'outputgenerator' calculates f on a given input set
def outputgenerator(inputarray):
    outputs = list()
    inputarray_np = np.array(inputarray, dtype=float)
    i=0
    while  i < len(inputarray_np):
        outputs.append(f(inputarray_np[i, 0], inputarray_np[i, 1], inputarray_np[i, 2], inputarray_np[i, 3], inputarray_np[i, 4]))
        i += 1
    return list(outputs)

# To avoid the programm to always calculate new random data, we save the random set once it has been calculated for the first time.

Input_path_dim5 = 'Random_Input_dim5.npy'
Input_path_dim5_check = 'Random_Input_dim5_check.npy'


if os.path.exists(Input_path_dim5):
    random_input = np.load(Input_path_dim5)
    print("Inputs_dim5 loaded.")
if os.path.exists(Input_path_dim5_check):
    random_input_check = np.load(Input_path_dim5_check)
    print("Inputs_dim5_check loaded.")
else:
    random_input = np.array(random_points)
    random_input_check = np.array(random_points_check)
    np.save(Input_path_dim5, random_input)
    np.save(Input_path_dim5_check, random_input_check)
    print("Inputs generated and saved.")


Output_path_dim5 = 'Random_Output1_dim5.npy'

if os.path.exists(Output_path_dim5):
    random_output = np.load(Output_path_dim5)
    print("Outputs1 loaded.")
else:
    random_output = np.array(outputgenerator(random_input))
    np.save(Output_path_dim5, random_output)
    print("Outputs generated and saved.")


# 'infinity_norm_difference' calculates the difference to f of a given network on a given set.
def infinity_norm_difference(network, func, inputs):


    inputs = np.array(inputs)
    x1, x2, x3, x4, x5 = inputs[:, 0], inputs[:, 1], inputs[:, 2], inputs[:, 3], inputs[:, 4]


    network_outputs = network(inputs).numpy()
    if len(network_outputs.shape) > 1:
        network_outputs = network_outputs.flatten()

    func_outputs = func(x1, x2, x3, x4, x5)

    differences = np.abs(network_outputs - func_outputs)

    argmax_index = np.argmax(differences)

    argmax_point = inputs[argmax_index]


    return np.max(differences), tuple(argmax_point)

# We define the three architectures we want to compare
def create_model_depth1():
    model = tf.keras.Sequential([
        keras.layers.Input(shape=(5,)),
        keras.layers.Dense(units=5, activation='relu'),
        keras.layers.Dense(units=1, activation='linear')
    ])
    return model

def create_model_depth10():
    model = tf.keras.Sequential([
        keras.layers.Input(shape=(5,)),
        keras.layers.Dense(units=5, activation='relu'),
        keras.layers.Dense(units=5, activation='relu'),
        keras.layers.Dense(units=5, activation='relu'),
        keras.layers.Dense(units=5, activation='relu'),
        keras.layers.Dense(units=5, activation='relu'),
        keras.layers.Dense(units=5, activation='relu'),
        keras.layers.Dense(units=5, activation='relu'),
        keras.layers.Dense(units=5, activation='relu'),
        keras.layers.Dense(units=5, activation='relu'),
        keras.layers.Dense(units=5, activation='relu'),
        keras.layers.Dense(units=1, activation='linear')
    ])
    return model

def create_model_depth20():
    model = tf.keras.Sequential([
        keras.layers.Input(shape=(5,)),
        keras.layers.Dense(units=5, activation='relu'),
        keras.layers.Dense(units=5, activation='relu'),
        keras.layers.Dense(units=5, activation='relu'),
        keras.layers.Dense(units=5, activation='relu'),
        keras.layers.Dense(units=5, activation='relu'),
        keras.layers.Dense(units=5, activation='relu'),
        keras.layers.Dense(units=5, activation='relu'),
        keras.layers.Dense(units=5, activation='relu'),
        keras.layers.Dense(units=5, activation='relu'),
        keras.layers.Dense(units=5, activation='relu'),
        keras.layers.Dense(units=5, activation='relu'),
        keras.layers.Dense(units=5, activation='relu'),
        keras.layers.Dense(units=5, activation='relu'),
        keras.layers.Dense(units=5, activation='relu'),
        keras.layers.Dense(units=5, activation='relu'),
        keras.layers.Dense(units=5, activation='relu'),
        keras.layers.Dense(units=5, activation='relu'),
        keras.layers.Dense(units=5, activation='relu'),
        keras.layers.Dense(units=5, activation='relu'),
        keras.layers.Dense(units=5, activation='relu'),
        keras.layers.Dense(units=1, activation='linear')
    ])
    return model

# Settings for training
epochs = 50
batch_size = 8
loss_fn = tf.keras.losses.MeanSquaredError()

# Arrays for saving all the relevant training data
loss_log_rand1_dim5 = []
loss_log_rand10_dim5 = []
loss_log_rand20_dim5 = []

all_params = []

infinity_norms_rand1_dim5 = []
infinity_norms_rand10_dim5 = []
infinity_norms_rand20_dim5 = []


# We train our three different networks 10 times and save loss, maximizer and infinity norm for each epoch
for i in range(10):
    model_depth1_random = create_model_depth1()
    model_depth10_random = create_model_depth10()
    model_depth20_random = create_model_depth20()

    model_depth1_random.compile(optimizer='Adam', loss=loss_fn)
    model_depth10_random.compile(optimizer='Adam', loss=loss_fn)
    model_depth20_random.compile(optimizer='Adam', loss=loss_fn)


    # Training
    for epoch in range(epochs):
        history1 = model_depth1_random.fit(random_input, random_output, epochs=1, batch_size=batch_size, verbose=0)
        history2 = model_depth10_random.fit(random_input, random_output, epochs=1, batch_size=batch_size, verbose=0)
        history3 = model_depth20_random.fit(random_input, random_output, epochs=1, batch_size=batch_size, verbose=0)
        loss1 = history1.history["loss"][0]
        loss2 = history2.history["loss"][0]
        loss3 = history3.history["loss"][0]

        sup_norm1, argmax_point1 = infinity_norm_difference(model_depth1_random, f, random_input_check)
        sup_norm2, argmax_point2 = infinity_norm_difference(model_depth10_random, f, random_input_check)
        sup_norm3, argmax_point3 = infinity_norm_difference(model_depth20_random, f, random_input_check)

        loss_log_rand1_dim5.append([i, epoch + 1, loss1, sup_norm1, *argmax_point1])
        loss_log_rand10_dim5.append([i, epoch + 1, loss2, sup_norm2, *argmax_point2])
        loss_log_rand20_dim5.append([i, epoch + 1, loss3, sup_norm3, *argmax_point3])


    # Lastly we create some csv files which include all the relevant data we saved earlier during training

    # Save all the model parameters
    params1 = model_depth1_random.get_weights()
    params_flat1 = [p.flatten() for p in params1]
    param_series1 = pd.Series(np.concatenate(params_flat1), name=f'model_{i + 1}_depth1_random')

    params2 = model_depth10_random.get_weights()
    params_flat2 = [p.flatten() for p in params2]
    param_series2 = pd.Series(np.concatenate(params_flat2), name=f'model_{i + 1}_depth2_random')

    params3 = model_depth20_random.get_weights()
    params_flat3 = [p.flatten() for p in params3]
    param_series3 = pd.Series(np.concatenate(params_flat3), name=f'model_{i + 1}_depth8_random')


    all_params.append(param_series1)
    all_params.append(param_series2)
    all_params.append(param_series3)

    # Calculate norms and maximizers after training is done
    max_diff, argmax_diff = infinity_norm_difference(model_depth1_random, f, random_input_check)
    infinity_norms_rand1_dim5.append([i + 1, max_diff, argmax_diff[0], argmax_diff[1], argmax_diff[2], argmax_diff[3], argmax_diff[4]])

    max_diff, argmax_diff = infinity_norm_difference(model_depth10_random, f, random_input_check)
    infinity_norms_rand10_dim5.append([i + 1, max_diff, argmax_diff[0], argmax_diff[1], argmax_diff[2], argmax_diff[3], argmax_diff[4]])

    max_diff, argmax_diff = infinity_norm_difference(model_depth20_random, f, random_input_check)
    infinity_norms_rand20_dim5.append([i + 1, max_diff, argmax_diff[0], argmax_diff[1], argmax_diff[2], argmax_diff[3], argmax_diff[4]])



# Save loss + norm each epoch
with open("Loss random depth 1 dim 5.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Run", "Epoch", "Loss", "Supnorm", "Argmax_x1", "Argmax_x2", "Argmax_x3", "Argmax_x4", "Argmax_x5"])
    writer.writerows(loss_log_rand1_dim5)

with open("Loss random depth 10 dim 5.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Run", "Epoch", "Loss", "Supnorm", "Argmax_x1", "Argmax_x2", "Argmax_x3", "Argmax_x4", "Argmax_x5"])
        writer.writerows(loss_log_rand10_dim5)

with open("Loss random depth 20 dim 5.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Run", "Epoch", "Loss", "Supnorm", "Argmax_x1", "Argmax_x2", "Argmax_x3", "Argmax_x4", "Argmax_x5"])
    writer.writerows(loss_log_rand20_dim5)


# Final save of parameters
param_df_final = pd.concat(all_params, axis=1)
param_df_final.to_csv("all_params_dim5.csv", index=False)

# Final save of infinity norms and maximizers
infinity_norm_df = pd.DataFrame(infinity_norms_rand1_dim5, columns=["Number", "Max Diff", "Argmax X1", "Argmax X2", "Argmax X3", "Argmax X4", "Argmax X5"])
infinity_norm_df.to_csv("NormArgmaxRandom1_dim5.csv", index=False)

infinity_norm_df = pd.DataFrame(infinity_norms_rand10_dim5, columns=["Number", "Max Diff", "Argmax X", "Argmax Y", "Argmax X3", "Argmax X4", "Argmax X5"])
infinity_norm_df.to_csv("NormArgmaxRandom10_dim5.csv", index=False)

infinity_norm_df = pd.DataFrame(infinity_norms_rand20_dim5, columns=["Number", "Max Diff", "Argmax X", "Argmax Y", "Argmax X3", "Argmax X4", "Argmax X5"])
infinity_norm_df.to_csv("NormArgmaxRandom20_dim5.csv", index=False)


print("Training finished and results saved.")