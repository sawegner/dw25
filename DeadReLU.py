import numpy as np
from itertools import product

# best nets in infinity norm:
#   random1 6
#   random2 43
#   random8 8
#   grid1   21
#   grid2   4
#   grid8   29

params_dim2 = np.loadtxt("all_params_dim2_withzeros.csv", delimiter=",")

points_per_dim = 500

grid_values = np.linspace(0,1, points_per_dim)
grid_points = np.array(list(product(grid_values,repeat=2)))

center = np.array([0.5] * 2)

radius = 0.5

distances = np.linalg.norm(grid_points - center, axis=1)

filtered_points = grid_points[distances <= radius]

def relu(x):
    return np.maximum(0,x)

def percent_dead(x):
    return (round(np.sum(np.all(x == 0, axis=1)) / x.shape[0], 2)) * 100


def percent_dead_node1(x):
    dead = 0
    for i in range(len(x)):
        if x[i, 0] == 0:
            dead += 1
    return round(dead / len(x), 3) * 100


def percent_dead_node2(x):
    dead = 0
    for i in range(len(x)):
        if x[i, 1] == 0:
            dead += 1
    return round(dead / len(x), 3) * 100

for i in range(10):
    def best_random1_layer1(x):
        A = np.array([[params_dim2[1, 6*i], params_dim2[2, 6*i]], [params_dim2[3, 6], params_dim2[4, 6]]])
        b = np.array([[params_dim2[5, 6*i]], [params_dim2[6, 6*i]]])
        return relu(x @ A.T + b.T)


    """def best_random1_out(x):
        O = np.array([params_dim2[7, 6], params_dim2[8, 6]])
        bo = np.array(params_dim2[9, 6])
        print(bo)
        return x @ O + bo"""


    def best_random2_layer1(x):
        A = np.array([[params_dim2[1, 6*i+1], params_dim2[2, 6*i+1]],
                      [params_dim2[3, 6*i+1], params_dim2[4, 6*i+1]]])
        b = np.array([[params_dim2[5, 6*i+1]], [params_dim2[6, 6*i+1]]])
        return relu(x @ A.T + b.T)


    def best_random2_layer2(x):
        A = np.array([[params_dim2[7, 6*i+1], params_dim2[8, 6*i+1]],
                      [params_dim2[9, 6*i+1], params_dim2[10, 6*i+1]]])
        b = np.array([[params_dim2[11, 6*i+1]], [params_dim2[12, 6*i+1]]])
        return relu(x @ A.T + b.T)


    """def best_random2_out(x):
        O = np.array([params_dim2[13, 43], params_dim2[14, 43]])
        bo = np.array(params_dim2[15, 43])
        out = np.array([])
        for element in x:
            np.append(out, np.matmul(O, element))
        return out + bo"""


    def best_random8_layer1(x):
        A = np.array([[params_dim2[1, 6*i+2], params_dim2[2, 6*i+2]],
                      [params_dim2[3, 6*i+2], params_dim2[4, 6*i+2]]])
        b = np.array([[params_dim2[5, 6*i+2]], [params_dim2[6, 6*i+2]]])
        return relu(x @ A.T + b.T)


    def best_random8_layer2(x):
        A = np.array([[params_dim2[7, 6*i+2], params_dim2[8, 6*i+2]],
                      [params_dim2[9, 6*i+2], params_dim2[10, 6*i+2]]])
        b = np.array([[params_dim2[11, 6*i+2]], [params_dim2[12, 6*i+2]]])
        return relu(x @ A.T + b.T)


    def best_random8_layer3(x):
        A = np.array([[params_dim2[13, 6*i+2], params_dim2[14, 6*i+2]],
                      [params_dim2[15, 6*i+2], params_dim2[16, 6*i+2]]])
        b = np.array([[params_dim2[17, 6*i+2]], [params_dim2[18, 6*i+2

]]])
        return relu(x @ A.T + b.T)


    def best_random8_layer4(x):
        A = np.array([[params_dim2[19, 6*i+2], params_dim2[20, 6*i+2]],
                      [params_dim2[21, 6*i+2], params_dim2[22, 6*i+2]]])
        b = np.array([[params_dim2[23, 6*i+2]], [params_dim2[24, 6*i+2]]])
        return relu(x @ A.T + b.T)


    def best_random8_layer5(x):
        A = np.array([[params_dim2[25, 6*i+2], params_dim2[26, 6*i+2]],
                      [params_dim2[27, 6*i+2], params_dim2[28, 6*i+2]]])
        b = np.array([[params_dim2[29, 6*i+2]], [params_dim2[30, 6*i+2]]])
        return relu(x @ A.T + b.T)


    def best_random8_layer6(x):
        A = np.array([[params_dim2[31, 6*i+2], params_dim2[32, 6*i+2]],
                      [params_dim2[33, 6*i+2], params_dim2[34, 6*i+2]]])
        b = np.array([[params_dim2[35, 6*i+2]], [params_dim2[36, 6*i+2]]])
        return relu(x @ A.T + b.T)


    def best_random8_layer7(x):
        A = np.array([[params_dim2[37, 6*i+2], params_dim2[38, 6*i+2]],
                      [params_dim2[39, 6*i+2], params_dim2[40, 6*i+2]]])
        b = np.array([[params_dim2[41, 6*i+2]], [params_dim2[42, 6*i+2]]])
        return relu(x @ A.T + b.T)


    def best_random8_layer8(x):
        A = np.array([[params_dim2[43, 6*i+2], params_dim2[44, 6*i+2]],
                      [params_dim2[45, 6*i+2], params_dim2[46, 6*i+2]]])
        b = np.array([[params_dim2[47, 6*i+2]], [params_dim2[48, 6*i+2]]])
        return relu(x @ A.T + b.T)


    """def best_random8_out(x):
        O = np.array([params_dim2[49, 8], params_dim2[50, 8]])
        bo = np.array(params_dim2[51, 8])
        out = np.array([])
        for element in x:
            np.append(out, np.matmul(O, element))
        return out + bo"""


    def best_grid1_layer1(x):
        A = np.array([[params_dim2[1, 6*i+3], params_dim2[2, 6*i+3]],
                      [params_dim2[3, 6*i+3], params_dim2[4, 6*i+3]]])
        b = np.array([[params_dim2[5, 6*i+3]], [params_dim2[6, 6*i+3]]])
        return relu(x @ A.T + b.T)

    print("Params for grid1, number ",i+1)
    for j in range(9):
        print(params_dim2[j+1, 6*i+3])
    print()

    """def best_grid1_out(x):
        O = np.array([params_dim2[7, 21], params_dim2[8, 21]])
        bo = np.array(params_dim2[9, 21])
        out = np.array([])
        for element in x:
            np.append(out, np.matmul(O, element))
        return out + bo"""


    def best_grid2_layer1(x):
        A = np.array([[params_dim2[1, 6*i+4], params_dim2[2, 6*i+4]],
                      [params_dim2[3, 6*i+4], params_dim2[4, 6*i+4]]])
        b = np.array([[params_dim2[5, 6*i+4]], [params_dim2[6, 6*i+4]]])
        return relu(x @ A.T + b.T)


    def best_grid2_layer2(x):
        A = np.array([[params_dim2[7, 6*i+4], params_dim2[8, 6*i+4]],
                      [params_dim2[9, 6*i+4], params_dim2[10, 6*i+4]]])
        b = np.array([[params_dim2[11, 6*i+4]], [params_dim2[12, 6*i+4]]])
        return relu(x @ A.T + b.T)


    """def best_grid2_out(x):
        O = np.array([params_dim2[13, 4], params_dim2[14, 4]])
        bo = np.array(params_dim2[15, 4])
        out = np.array([])
        for element in x:
            np.append(out, np.matmul(O, element))
        return out + bo"""


    def best_grid8_layer1(x):
        A = np.array([[params_dim2[1, 6*i+5], params_dim2[2, 6*i+5]],
                      [params_dim2[3, 6*i+5], params_dim2[4, 6*i+5]]])
        b = np.array([[params_dim2[5, 6*i+5]], [params_dim2[6, 6*i+5]]])
        return relu(x @ A.T + b.T)


    def best_grid8_layer2(x):
        A = np.array([[params_dim2[7, 6*i+5], params_dim2[8, 6*i+5]],
                      [params_dim2[9, 6*i+5], params_dim2[10, 6*i+5]]])
        b = np.array([[params_dim2[11, 6*i+5]], [params_dim2[12, 6*i+5]]])
        return relu(x @ A.T + b.T)


    def best_grid8_layer3(x):
        A = np.array([[params_dim2[13, 6*i+5], params_dim2[14, 6*i+5]],
                      [params_dim2[15, 6*i+5], params_dim2[16, 6*i+5]]])
        b = np.array([[params_dim2[17, 6*i+5]], [params_dim2[18, 6*i+5]]])
        return relu(x @ A.T + b.T)


    def best_grid8_layer4(x):
        A = np.array([[params_dim2[19, 6*i+5], params_dim2[20, 6*i+5]],
                      [params_dim2[21, 6*i+5], params_dim2[22, 6*i+5]]])
        b = np.array([[params_dim2[23, 6*i+5]], [params_dim2[24, 6*i+5]]])
        return relu(x @ A.T + b.T)


    def best_grid8_layer5(x):
        A = np.array([[params_dim2[25, 6*i+5], params_dim2[26, 6*i+5]],
                      [params_dim2[27, 6*i+5], params_dim2[28, 6*i+5]]])
        b = np.array([[params_dim2[29, 6*i+5]], [params_dim2[30, 6*i+5]]])
        return relu(x @ A.T + b.T)


    def best_grid8_layer6(x):
        A = np.array([[params_dim2[31, 6*i+5], params_dim2[32, 6*i+5]],
                      [params_dim2[33, 6*i+5], params_dim2[34, 6*i+5]]])
        b = np.array([[params_dim2[35, 6*i+5]], [params_dim2[36, 6*i+5]]])
        return relu(x @ A.T + b.T)


    def best_grid8_layer7(x):
        A = np.array([[params_dim2[37, 6*i+5], params_dim2[38, 6*i+5]],
                      [params_dim2[39, 6*i+5], params_dim2[40, 6*i+5]]])
        b = np.array([[params_dim2[41, 6*i+5]], [params_dim2[42, 6*i+5]]])
        return relu(x @ A.T + b.T)


    def best_grid8_layer8(x):
        A = np.array([[params_dim2[43, 6*i+5], params_dim2[44, 6*i+5]],
                      [params_dim2[45, 6*i+5], params_dim2[46, 6*i+5]]])
        b = np.array([[params_dim2[47, 6*i+5]], [params_dim2[48, 6*i+5]]])
        return relu(x @ A.T + b.T)



    print("Number ",i+1," % dead for random 1:\n")
    print("1st layer, first node: ", percent_dead_node1(best_random1_layer1(filtered_points)), "%")
    print("1st layer, second node: ", percent_dead_node2(best_random1_layer1(filtered_points)), "%")
    print()
    print("Number ",i+1," % dead for random 2:\n")
    print("1st layer, first node: ", percent_dead_node1(best_random2_layer1(filtered_points)), "%")
    print("1st layer, second node: ", percent_dead_node2(best_random2_layer1(filtered_points)), "%")
    print("2nd layer, first node: ", percent_dead_node1(best_random2_layer2(best_random2_layer1(filtered_points))), "%")
    print("2nd layer, second node: ", percent_dead_node2(best_random2_layer2(best_random2_layer1(filtered_points))),"%")
    print()
    print("Number ",i+1," % dead for random 8:\n")
    print()
    print("Out Bias:", params_dim2[51, 6 * i + 2])
    print()
    print("1st layer, first node: ", percent_dead_node1(best_random8_layer1(filtered_points)), "%")
    print("1st layer, second node: ", percent_dead_node2(best_random8_layer1(filtered_points)), "%")
    print("2nd layer, first node: ", percent_dead_node1(best_random8_layer2(best_random8_layer1(filtered_points))),"%")
    print("2nd layer, second node: ", percent_dead_node2(best_random8_layer2(best_random8_layer1(filtered_points))),"%")
    print("3rd layer, first node: ",
          percent_dead_node1(best_random8_layer3(best_random8_layer2(best_random8_layer1(filtered_points)))),"%")
    print("3rd layer, second node: ",
          percent_dead_node2(best_random8_layer3(best_random8_layer2(best_random8_layer1(filtered_points)))),"%")
    print("4th layer, first node: ", percent_dead_node1(
        best_random8_layer4(best_random8_layer3(best_random8_layer2(best_random8_layer1(filtered_points))))),"%")
    print("4th layer, second node: ", percent_dead_node2(
        best_random8_layer4(best_random8_layer3(best_random8_layer2(best_random8_layer1(filtered_points))))),"%")
    print("5th layer first node: ", percent_dead_node1(best_random8_layer5(
        best_random8_layer4(best_random8_layer3(best_random8_layer2(best_random8_layer2(filtered_points)))))),"%")
    print("5th layer, second node: ", percent_dead_node2(best_random8_layer5(
        best_random8_layer4(best_random8_layer3(best_random8_layer2(best_random8_layer2(filtered_points)))))),"%")
    print("6th layer, first node: ", percent_dead_node1(best_random8_layer6(best_random8_layer5(
        best_random8_layer4(best_random8_layer3(best_random8_layer2(best_random8_layer1(filtered_points))))))),"%")
    print("6th layer, second node: ", percent_dead_node2(best_random8_layer6(best_random8_layer5(
        best_random8_layer4(best_random8_layer3(best_random8_layer2(best_random8_layer1(filtered_points))))))),"%")
    print("7th layer, first node: ", percent_dead_node1(best_random8_layer7(best_random8_layer6(best_random8_layer5(
        best_random8_layer4(best_random8_layer3(best_random8_layer2(best_random8_layer1(filtered_points)))))))),"%")
    print("7th layer, second node: ", percent_dead_node2(best_random8_layer7(best_random8_layer6(best_random8_layer5(
        best_random8_layer4(best_random8_layer3(best_random8_layer2(best_random8_layer1(filtered_points)))))))),"%")
    print("8th layer, first node: ", percent_dead_node1(best_random8_layer8(best_random8_layer7(best_random8_layer6(
        best_random8_layer5(
            best_random8_layer4(best_random8_layer3(best_random8_layer2(best_random8_layer1(filtered_points))))))))),"%")
    print("8th layer, second node: ", percent_dead_node2(best_random8_layer8(best_random8_layer7(best_random8_layer6(
        best_random8_layer5(
            best_random8_layer4(best_random8_layer3(best_random8_layer2(best_random8_layer1(filtered_points))))))))),"%")
    # print("Output layer:",percent_dead(best_random8_out(filtered_points)))
    print()
    print("Number ",i+1," % dead for grid 1:\n")
    print("1st layer, first node: ", percent_dead_node1(best_grid1_layer1(filtered_points)),"%")
    print("1st layer, second node: ", percent_dead_node2(best_grid1_layer1(filtered_points)),"%")
    # print("Output layer:",percent_dead(best_grid1_out(filtered_points)))
    print()
    print("Number ",i+1," % dead for grid 2:\n")
    print("1st layer, first node: ", percent_dead_node1(best_grid2_layer1(filtered_points)),"%")
    print("1st layer, second node: ", percent_dead_node2(best_grid2_layer1(filtered_points)),"%")
    print("2nd layer, first node: ", percent_dead_node1(best_grid2_layer2(best_grid2_layer1(filtered_points))),"%")
    print("2nd layer, second node: ", percent_dead_node2(best_grid2_layer2(best_grid2_layer1(filtered_points))),"%")
    # print("Output layer:",percent_dead(best_grid2_out(filtered_points)))
    print()
    print("Number ",i+1," % dead for grid 8:\n")
    print()
    print("Out Bias:", params_dim2[51, 6 * i + 5])
    print()
    print("1st layer, first node: ", percent_dead_node1(best_grid8_layer1(filtered_points)), "%")
    print("1st layer, second node: ", percent_dead_node2(best_grid8_layer1(filtered_points)), "%")
    print("2nd layer, first node: ", percent_dead_node1(best_grid8_layer2(best_grid8_layer1(filtered_points))),"%")
    print("2nd layer, second node: ", percent_dead_node2(best_grid8_layer2(best_grid8_layer1(filtered_points))),"%")
    print("3rd layer, first node: ",
          percent_dead_node1(best_grid8_layer3(best_grid8_layer2(best_grid8_layer1(filtered_points)))),"%")
    print("3rd layer, second node: ",
          percent_dead_node2(best_grid8_layer3(best_grid8_layer2(best_grid8_layer1(filtered_points)))),"%")
    print("4th layer, first node: ", percent_dead_node1(
        best_grid8_layer4(best_grid8_layer3(best_grid8_layer2(best_grid8_layer1(filtered_points))))),"%")
    print("4th layer, second node: ", percent_dead_node2(
        best_grid8_layer4(best_grid8_layer3(best_grid8_layer2(best_grid8_layer1(filtered_points))))),"%")
    print("5th layer, first node: ", percent_dead_node1(
        best_grid8_layer5(best_grid8_layer4(best_grid8_layer3(best_grid8_layer2(best_grid8_layer2(filtered_points)))))),"%")
    print("5th layer, second node: ", percent_dead_node2(
        best_grid8_layer5(best_grid8_layer4(best_grid8_layer3(best_grid8_layer2(best_grid8_layer2(filtered_points)))))),"%")
    print("6th layer, first node: ", percent_dead_node1(best_grid8_layer6(best_grid8_layer5(
        best_grid8_layer4(best_grid8_layer3(best_grid8_layer2(best_grid8_layer1(filtered_points))))))),"%")
    print("6th layer, second node: ", percent_dead_node2(best_grid8_layer6(best_grid8_layer5(
        best_grid8_layer4(best_grid8_layer3(best_grid8_layer2(best_grid8_layer1(filtered_points))))))),"%")
    print("7th layer, first node: ", percent_dead_node1(best_grid8_layer7(best_grid8_layer6(best_grid8_layer5(
        best_grid8_layer4(best_grid8_layer3(best_grid8_layer2(best_grid8_layer1(filtered_points)))))))),"%")
    print("7th layer, second node: ", percent_dead_node2(best_grid8_layer7(best_grid8_layer6(best_grid8_layer5(
        best_grid8_layer4(best_grid8_layer3(best_grid8_layer2(best_grid8_layer1(filtered_points)))))))),"%")
    print("8th layer, first node: ", percent_dead_node1(best_grid8_layer8(best_grid8_layer7(best_grid8_layer6(
        best_grid8_layer5(
            best_grid8_layer4(best_grid8_layer3(best_grid8_layer2(best_grid8_layer1(filtered_points))))))))),"%")
    print("8th layer, second node: ", percent_dead_node2(best_grid8_layer8(best_grid8_layer7(best_grid8_layer6(
        best_grid8_layer5(
            best_grid8_layer4(best_grid8_layer3(best_grid8_layer2(best_grid8_layer1(filtered_points))))))))),"%")
    # print("Output layer:",percent_dead(best_grid8_out(filtered_points)))
    print()
    print()
    print()

