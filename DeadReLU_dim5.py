import numpy as np
from itertools import product

params_dim5 = np.loadtxt("all_params_dim5.csv", delimiter=",")

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

def percent_dead_node1(x):
    dead = 0
    for i in range(len(x)):
        if x[i,0] == 0 :
            dead+=1
    return round(dead/len(x),3)*100

def percent_dead_node2(x):
    dead = 0
    for i in range(len(x)):
        if x[i,1] == 0 :
            dead+=1
    return round(dead/len(x),3)*100

def percent_dead_node3(x):
    dead = 0
    for i in range(len(x)):
        if x[i,2] == 0 :
            dead+=1
    return round(dead/len(x),3)*100

def percent_dead_node4(x):
    dead = 0
    for i in range(len(x)):
        if x[i,3] == 0 :
            dead+=1
    return round(dead/len(x),3)*100

def percent_dead_node5(x):
    dead = 0
    for i in range(len(x)):
        if x[i,4] == 0 :
            dead+=1
    return round(dead/len(x),3)*100
def relu(x):
    return np.maximum(0,x)

for i in range(10):
    def best_random20_layer1(x):
        A = np.array(
            [[params_dim5[1, 3*i+2], params_dim5[2, 3*i+2], params_dim5[3, 3*i+2], params_dim5[4, 3*i+2], params_dim5[5, 3*i+2]],
             [params_dim5[6, 3*i+2], params_dim5[7, 3*i+2], params_dim5[8, 3*i+2], params_dim5[9, 3*i+2], params_dim5[10, 3*i+2]],
             [params_dim5[11, 3*i+2], params_dim5[12, 3*i+2], params_dim5[13, 3*i+2], params_dim5[14, 3*i+2], params_dim5[15, 3*i+2]],
             [params_dim5[16, 3*i+2], params_dim5[17, 3*i+2], params_dim5[18, 3*i+2], params_dim5[19, 3*i+2], params_dim5[20, 3*i+2]],
             [params_dim5[21, 3*i+2], params_dim5[22, 3*i+2], params_dim5[23, 3*i+2], params_dim5[24, 3*i+2], params_dim5[25, 3*i+2]]])
        b = np.array([[params_dim5[26, 3*i+2]], [params_dim5[27, 3*i+2]], [params_dim5[28, 3*i+2]], [params_dim5[29, 3*i+2]],
                      [params_dim5[30, 3*i+2]]])
        return relu(x @ A.T + b.T)


    def best_random20_layer2(x):
        A = np.array([
            [params_dim5[31, 3*i+2], params_dim5[32, 3*i+2], params_dim5[33, 3*i+2], params_dim5[34, 3*i+2], params_dim5[35, 3*i+2]],
            [params_dim5[36, 3*i+2], params_dim5[37, 3*i+2], params_dim5[38, 3*i+2], params_dim5[39, 3*i+2], params_dim5[40, 3*i+2]],
            [params_dim5[41, 3*i+2], params_dim5[42, 3*i+2], params_dim5[43, 3*i+2], params_dim5[44, 3*i+2], params_dim5[45, 3*i+2]],
            [params_dim5[46, 3*i+2], params_dim5[47, 3*i+2], params_dim5[48, 3*i+2], params_dim5[49, 3*i+2], params_dim5[50, 3*i+2]],
            [params_dim5[51, 3*i+2], params_dim5[52, 3*i+2], params_dim5[53, 3*i+2], params_dim5[54, 3*i+2], params_dim5[55, 3*i+2]]
        ])
        b = np.array([[params_dim5[56, 3*i+2]], [params_dim5[57, 3*i+2]], [params_dim5[58, 3*i+2]], [params_dim5[59, 3*i+2]],
                      [params_dim5[60, 3*i+2]]])
        return relu(x @ A.T + b.T)


    def best_random20_layer3(x):
        A = np.array(
            [[params_dim5[61, 3*i+2], params_dim5[62, 3*i+2], params_dim5[63, 3*i+2], params_dim5[64, 3*i+2], params_dim5[65, 3*i+2]],
             [params_dim5[66, 3*i+2], params_dim5[67, 3*i+2], params_dim5[68, 3*i+2], params_dim5[69, 3*i+2], params_dim5[70, 3*i+2]],
             [params_dim5[71, 3*i+2], params_dim5[72, 3*i+2], params_dim5[73, 3*i+2], params_dim5[74, 3*i+2], params_dim5[75, 3*i+2]],
             [params_dim5[76, 3*i+2], params_dim5[77, 3*i+2], params_dim5[78, 3*i+2], params_dim5[79, 3*i+2], params_dim5[80, 3*i+2]],
             [params_dim5[81, 3*i+2], params_dim5[82, 3*i+2], params_dim5[83, 3*i+2], params_dim5[84, 3*i+2], params_dim5[85, 3*i+2]]])
        b = np.array([[params_dim5[86, 3*i+2]], [params_dim5[87, 3*i+2]], [params_dim5[88, 3*i+2]], [params_dim5[89, 3*i+2]],
                      [params_dim5[90, 3*i+2]]])
        return relu(x @ A.T + b.T)


    def best_random20_layer4(x):
        A = np.array(
            [[params_dim5[91, 3*i+2], params_dim5[92, 3*i+2], params_dim5[93, 3*i+2], params_dim5[94, 3*i+2], params_dim5[95, 3*i+2]],
             [params_dim5[96, 3*i+2], params_dim5[97, 3*i+2], params_dim5[98, 3*i+2], params_dim5[99, 3*i+2], params_dim5[100, 3*i+2]],
             [params_dim5[101, 3*i+2], params_dim5[102, 3*i+2], params_dim5[103, 3*i+2], params_dim5[104, 3*i+2],
              params_dim5[105, 3*i+2]],
             [params_dim5[106, 3*i+2], params_dim5[107, 3*i+2], params_dim5[108, 3*i+2], params_dim5[109, 3*i+2],
              params_dim5[110, 3*i+2]],
             [params_dim5[111, 3*i+2], params_dim5[112, 3*i+2], params_dim5[113, 3*i+2], params_dim5[114, 3*i+2],
              params_dim5[115, 3*i+2]]])
        b = np.array([[params_dim5[116, 3*i+2]], [params_dim5[117, 3*i+2]], [params_dim5[118, 3*i+2]], [params_dim5[119, 3*i+2]],
                      [params_dim5[120, 3*i+2]]])
        return relu(x @ A.T + b.T)


    def best_random20_layer5(x):
        A = np.array([[params_dim5[121, 3*i+2], params_dim5[122, 3*i+2], params_dim5[123, 3*i+2], params_dim5[124, 3*i+2],
                       params_dim5[125, 3*i+2]],
                      [params_dim5[126, 3*i+2], params_dim5[127, 3*i+2], params_dim5[128, 3*i+2], params_dim5[129, 3*i+2],
                       params_dim5[130, 3*i+2]],
                      [params_dim5[131, 3*i+2], params_dim5[132, 3*i+2], params_dim5[133, 3*i+2], params_dim5[134, 3*i+2],
                       params_dim5[135, 3*i+2]],
                      [params_dim5[136, 3*i+2], params_dim5[137, 3*i+2], params_dim5[138, 3*i+2], params_dim5[139, 3*i+2],
                       params_dim5[140, 3*i+2]],
                      [params_dim5[141, 3*i+2], params_dim5[142, 3*i+2], params_dim5[143, 3*i+2], params_dim5[144, 3*i+2],
                       params_dim5[145, 3*i+2]]])
        b = np.array([[params_dim5[146, 3*i+2]], [params_dim5[147, 3*i+2]], [params_dim5[148, 3*i+2]], [params_dim5[149, 3*i+2]],
                      [params_dim5[150, 3*i+2]]])
        return relu(x @ A.T + b.T)


    def best_random20_layer6(x):
        A = np.array([[params_dim5[151, 3*i+2], params_dim5[152, 3*i+2], params_dim5[153, 3*i+2], params_dim5[154, 3*i+2],
                       params_dim5[155, 3*i+2]],
                      [params_dim5[156, 3*i+2], params_dim5[157, 3*i+2], params_dim5[158, 3*i+2], params_dim5[159, 3*i+2],
                       params_dim5[160, 3*i+2]],
                      [params_dim5[161, 3*i+2], params_dim5[162, 3*i+2], params_dim5[163, 3*i+2], params_dim5[164, 3*i+2],
                       params_dim5[165, 3*i+2]],
                      [params_dim5[166, 3*i+2], params_dim5[167, 3*i+2], params_dim5[168, 3*i+2], params_dim5[169, 3*i+2],
                       params_dim5[170, 3*i+2]],
                      [params_dim5[171, 3*i+2], params_dim5[172, 3*i+2], params_dim5[173, 3*i+2], params_dim5[174, 3*i+2],
                       params_dim5[175, 3*i+2]]])
        b = np.array([[params_dim5[176, 3*i+2]], [params_dim5[177, 3*i+2]], [params_dim5[178, 3*i+2]], [params_dim5[179, 3*i+2]],
                      [params_dim5[180, 3*i+2]]])
        return relu(x @ A.T + b.T)


    def best_random20_layer7(x):
        A = np.array([[params_dim5[181, 3*i+2], params_dim5[182, 3*i+2], params_dim5[183, 3*i+2], params_dim5[184, 3*i+2],
                       params_dim5[185, 3*i+2]],
                      [params_dim5[186, 3*i+2], params_dim5[187, 3*i+2], params_dim5[188, 3*i+2], params_dim5[189, 3*i+2],
                       params_dim5[190, 3*i+2]],
                      [params_dim5[191, 3*i+2], params_dim5[192, 3*i+2], params_dim5[193, 3*i+2], params_dim5[194, 3*i+2],
                       params_dim5[195, 3*i+2]],
                      [params_dim5[196, 3*i+2], params_dim5[197, 3*i+2], params_dim5[198, 3*i+2], params_dim5[199, 3*i+2],
                       params_dim5[200, 3*i+2]],
                      [params_dim5[201, 3*i+2], params_dim5[202, 3*i+2], params_dim5[203, 3*i+2], params_dim5[204, 3*i+2],
                       params_dim5[205, 3*i+2]]])
        b = np.array([[params_dim5[206, 3*i+2]], [params_dim5[207, 3*i+2]], [params_dim5[208, 3*i+2]], [params_dim5[209, 3*i+2]],
                      [params_dim5[210, 3*i+2]]])
        return relu(x @ A.T + b.T)


    def best_random20_layer8(x):
        A = np.array([[params_dim5[211, 3*i+2], params_dim5[212, 3*i+2], params_dim5[213, 3*i+2], params_dim5[214, 3*i+2],
                       params_dim5[215, 3*i+2]],
                      [params_dim5[216, 3*i+2], params_dim5[217, 3*i+2], params_dim5[218, 3*i+2], params_dim5[219, 3*i+2],
                       params_dim5[220, 3*i+2]],
                      [params_dim5[221, 3*i+2], params_dim5[222, 3*i+2], params_dim5[223, 3*i+2], params_dim5[224, 3*i+2],
                       params_dim5[225, 3*i+2]],
                      [params_dim5[226, 3*i+2], params_dim5[227, 3*i+2], params_dim5[228, 3*i+2], params_dim5[229, 3*i+2],
                       params_dim5[230, 3*i+2]],
                      [params_dim5[231, 3*i+2], params_dim5[232, 3*i+2], params_dim5[233, 3*i+2], params_dim5[234, 3*i+2],
                       params_dim5[235, 3*i+2]]])
        b = np.array([[params_dim5[236, 3*i+2]], [params_dim5[237, 3*i+2]], [params_dim5[238, 3*i+2]], [params_dim5[239, 3*i+2]],
                      [params_dim5[240, 3*i+2]]])
        return relu(x @ A.T + b.T)

    def best_random20_layer9(x):
        A = np.array([[params_dim5[241, 3*i+2], params_dim5[242, 3*i+2], params_dim5[243, 3*i+2], params_dim5[244, 3*i+2],
                       params_dim5[245, 3*i+2]],
                      [params_dim5[246, 3*i+2], params_dim5[247, 3*i+2], params_dim5[248, 3*i+2], params_dim5[249, 3*i+2],
                       params_dim5[250, 3*i+2]],
                      [params_dim5[251, 3*i+2], params_dim5[252, 3*i+2], params_dim5[253, 3*i+2], params_dim5[254, 3*i+2],
                       params_dim5[255, 3*i+2]],
                      [params_dim5[256, 3*i+2], params_dim5[257, 3*i+2], params_dim5[258, 3*i+2], params_dim5[259, 3*i+2],
                       params_dim5[260, 3*i+2]],
                      [params_dim5[261, 3*i+2], params_dim5[262, 3*i+2], params_dim5[263, 3*i+2], params_dim5[264, 3*i+2],
                       params_dim5[265, 3*i+2]]])
        b = np.array([[params_dim5[266, 3*i+2]], [params_dim5[267, 3*i+2]], [params_dim5[268, 3*i+2]], [params_dim5[269, 3*i+2]],
                      [params_dim5[270, 3*i+2]]])
        return relu(x @ A.T + b.T)

    def best_random20_layer10(x):
        A = np.array([[params_dim5[271, 3*i+2], params_dim5[272, 3*i+2], params_dim5[273, 3*i+2], params_dim5[274, 3*i+2],
                       params_dim5[275, 3*i+2]],
                      [params_dim5[276, 3*i+2], params_dim5[277, 3*i+2], params_dim5[278, 3*i+2], params_dim5[279, 3*i+2],
                       params_dim5[280, 3*i+2]],
                      [params_dim5[281, 3*i+2], params_dim5[282, 3*i+2], params_dim5[283, 3*i+2], params_dim5[284, 3*i+2],
                       params_dim5[285, 3*i+2]],
                      [params_dim5[286, 3*i+2], params_dim5[287, 3*i+2], params_dim5[288, 3*i+2], params_dim5[289, 3*i+2],
                       params_dim5[290, 3*i+2]],
                      [params_dim5[291, 3*i+2], params_dim5[292, 3*i+2], params_dim5[293, 3*i+2], params_dim5[294, 3*i+2],
                       params_dim5[295, 3*i+2]]])
        b = np.array([[params_dim5[296, 3*i+2]], [params_dim5[297, 3*i+2]], [params_dim5[298, 3*i+2]], [params_dim5[299, 3*i+2]],
                      [params_dim5[300, 3*i+2]]])
        return relu(x @ A.T + b.T)

    def best_random20_layer11(x):
        A = np.array([[params_dim5[301, 3*i+2], params_dim5[302, 3*i+2], params_dim5[303, 3*i+2], params_dim5[304, 3*i+2],
                       params_dim5[305, 3*i+2]],
                      [params_dim5[306, 3*i+2], params_dim5[307, 3*i+2], params_dim5[308, 3*i+2], params_dim5[309, 3*i+2],
                       params_dim5[310, 3*i+2]],
                      [params_dim5[311, 3*i+2], params_dim5[312, 3*i+2], params_dim5[313, 3*i+2], params_dim5[314, 3*i+2],
                       params_dim5[315, 3*i+2]],
                      [params_dim5[316, 3*i+2], params_dim5[317, 3*i+2], params_dim5[318, 3*i+2], params_dim5[319, 3*i+2],
                       params_dim5[320, 3*i+2]],
                      [params_dim5[321, 3*i+2], params_dim5[322, 3*i+2], params_dim5[323, 3*i+2], params_dim5[324, 3*i+2],
                       params_dim5[325, 3*i+2]]])
        b = np.array([[params_dim5[326, 3*i+2]], [params_dim5[327, 3*i+2]], [params_dim5[328, 3*i+2]], [params_dim5[329, 3*i+2]],
                      [params_dim5[330, 3*i+2]]])
        return relu(x @ A.T + b.T)


    def best_random20_layer12(x):
        A = np.array([[params_dim5[331, 3 * i + 2], params_dim5[332, 3 * i + 2], params_dim5[333, 3 * i + 2],
                       params_dim5[334, 3 * i + 2],
                       params_dim5[335, 3 * i + 2]],
                      [params_dim5[336, 3 * i + 2], params_dim5[337, 3 * i + 2], params_dim5[338, 3 * i + 2],
                       params_dim5[339, 3 * i + 2],
                       params_dim5[340, 3 * i + 2]],
                      [params_dim5[341, 3 * i + 2], params_dim5[342, 3 * i + 2], params_dim5[343, 3 * i + 2],
                       params_dim5[344, 3 * i + 2],
                       params_dim5[345, 3 * i + 2]],
                      [params_dim5[346, 3 * i + 2], params_dim5[347, 3 * i + 2], params_dim5[348, 3 * i + 2],
                       params_dim5[349, 3 * i + 2],
                       params_dim5[350, 3 * i + 2]],
                      [params_dim5[351, 3 * i + 2], params_dim5[352, 3 * i + 2], params_dim5[353, 3 * i + 2],
                       params_dim5[354, 3 * i + 2],
                       params_dim5[355, 3 * i + 2]]])
        b = np.array([[params_dim5[356, 3 * i + 2]], [params_dim5[357, 3 * i + 2]], [params_dim5[358, 3 * i + 2]],
                      [params_dim5[359, 3 * i + 2]],
                      [params_dim5[360, 3 * i + 2]]])
        return relu(x @ A.T + b.T)

    def best_random20_layer13(x):
        A = np.array([[params_dim5[361, 3 * i + 2], params_dim5[362, 3 * i + 2], params_dim5[363, 3 * i + 2],
                       params_dim5[364, 3 * i + 2],
                       params_dim5[365, 3 * i + 2]],
                      [params_dim5[366, 3 * i + 2], params_dim5[367, 3 * i + 2], params_dim5[368, 3 * i + 2],
                       params_dim5[369, 3 * i + 2],
                       params_dim5[370, 3 * i + 2]],
                      [params_dim5[371, 3 * i + 2], params_dim5[372, 3 * i + 2], params_dim5[373, 3 * i + 2],
                       params_dim5[374, 3 * i + 2],
                       params_dim5[375, 3 * i + 2]],
                      [params_dim5[376, 3 * i + 2], params_dim5[377, 3 * i + 2], params_dim5[378, 3 * i + 2],
                       params_dim5[379, 3 * i + 2],
                       params_dim5[370, 3 * i + 2]],
                      [params_dim5[381, 3 * i + 2], params_dim5[382, 3 * i + 2], params_dim5[383, 3 * i + 2],
                       params_dim5[384, 3 * i + 2],
                       params_dim5[385, 3 * i + 2]]])
        b = np.array([[params_dim5[386, 3 * i + 2]], [params_dim5[387, 3 * i + 2]], [params_dim5[388, 3 * i + 2]],
                      [params_dim5[389, 3 * i + 2]],
                      [params_dim5[390, 3 * i + 2]]])
        return relu(x @ A.T + b.T)

    def best_random20_layer14(x):
        A = np.array([[params_dim5[391, 3 * i + 2], params_dim5[392, 3 * i + 2], params_dim5[393, 3 * i + 2],
                       params_dim5[394, 3 * i + 2],
                       params_dim5[395, 3 * i + 2]],
                      [params_dim5[396, 3 * i + 2], params_dim5[397, 3 * i + 2], params_dim5[398, 3 * i + 2],
                       params_dim5[399, 3 * i + 2],
                       params_dim5[400, 3 * i + 2]],
                      [params_dim5[401, 3 * i + 2], params_dim5[402, 3 * i + 2], params_dim5[403, 3 * i + 2],
                       params_dim5[404, 3 * i + 2],
                       params_dim5[405, 3 * i + 2]],
                      [params_dim5[406, 3 * i + 2], params_dim5[407, 3 * i + 2], params_dim5[408, 3 * i + 2],
                       params_dim5[409, 3 * i + 2],
                       params_dim5[410, 3 * i + 2]],
                      [params_dim5[411, 3 * i + 2], params_dim5[412, 3 * i + 2], params_dim5[413, 3 * i + 2],
                       params_dim5[414, 3 * i + 2],
                       params_dim5[415, 3 * i + 2]]])
        b = np.array([[params_dim5[416, 3 * i + 2]], [params_dim5[417, 3 * i + 2]], [params_dim5[418, 3 * i + 2]],
                      [params_dim5[419, 3 * i + 2]],
                      [params_dim5[420, 3 * i + 2]]])
        return relu(x @ A.T + b.T)

    def best_random20_layer15(x):
        A = np.array([[params_dim5[421, 3 * i + 2], params_dim5[422, 3 * i + 2], params_dim5[423, 3 * i + 2],
                       params_dim5[424, 3 * i + 2],
                       params_dim5[425, 3 * i + 2]],
                      [params_dim5[426, 3 * i + 2], params_dim5[427, 3 * i + 2], params_dim5[428, 3 * i + 2],
                       params_dim5[429, 3 * i + 2],
                       params_dim5[430, 3 * i + 2]],
                      [params_dim5[431, 3 * i + 2], params_dim5[432, 3 * i + 2], params_dim5[433, 3 * i + 2],
                       params_dim5[434, 3 * i + 2],
                       params_dim5[435, 3 * i + 2]],
                      [params_dim5[436, 3 * i + 2], params_dim5[437, 3 * i + 2], params_dim5[438, 3 * i + 2],
                       params_dim5[439, 3 * i + 2],
                       params_dim5[440, 3 * i + 2]],
                      [params_dim5[441, 3 * i + 2], params_dim5[442, 3 * i + 2], params_dim5[443, 3 * i + 2],
                       params_dim5[444, 3 * i + 2],
                       params_dim5[445, 3 * i + 2]]])
        b = np.array([[params_dim5[446, 3 * i + 2]], [params_dim5[447, 3 * i + 2]], [params_dim5[448, 3 * i + 2]],
                      [params_dim5[449, 3 * i + 2]],
                      [params_dim5[450, 3 * i + 2]]])
        return relu(x @ A.T + b.T)

    def best_random20_layer16(x):
        A = np.array([[params_dim5[451, 3 * i + 2], params_dim5[452, 3 * i + 2], params_dim5[453, 3 * i + 2],
                       params_dim5[454, 3 * i + 2],
                       params_dim5[455, 3 * i + 2]],
                      [params_dim5[456, 3 * i + 2], params_dim5[457, 3 * i + 2], params_dim5[458, 3 * i + 2],
                       params_dim5[459, 3 * i + 2],
                       params_dim5[460, 3 * i + 2]],
                      [params_dim5[461, 3 * i + 2], params_dim5[462, 3 * i + 2], params_dim5[463, 3 * i + 2],
                       params_dim5[464, 3 * i + 2],
                       params_dim5[465, 3 * i + 2]],
                      [params_dim5[466, 3 * i + 2], params_dim5[467, 3 * i + 2], params_dim5[468, 3 * i + 2],
                       params_dim5[469, 3 * i + 2],
                       params_dim5[470, 3 * i + 2]],
                      [params_dim5[471, 3 * i + 2], params_dim5[472, 3 * i + 2], params_dim5[473, 3 * i + 2],
                       params_dim5[474, 3 * i + 2],
                       params_dim5[475, 3 * i + 2]]])
        b = np.array([[params_dim5[476, 3 * i + 2]], [params_dim5[477, 3 * i + 2]], [params_dim5[478, 3 * i + 2]],
                      [params_dim5[479, 3 * i + 2]],
                      [params_dim5[480, 3 * i + 2]]])
        return relu(x @ A.T + b.T)

    def best_random20_layer17(x):
        A = np.array([[params_dim5[481, 3 * i + 2], params_dim5[482, 3 * i + 2], params_dim5[483, 3 * i + 2],
                       params_dim5[484, 3 * i + 2],
                       params_dim5[485, 3 * i + 2]],
                      [params_dim5[486, 3 * i + 2], params_dim5[487, 3 * i + 2], params_dim5[488, 3 * i + 2],
                       params_dim5[489, 3 * i + 2],
                       params_dim5[490, 3 * i + 2]],
                      [params_dim5[491, 3 * i + 2], params_dim5[492, 3 * i + 2], params_dim5[493, 3 * i + 2],
                       params_dim5[494, 3 * i + 2],
                       params_dim5[495, 3 * i + 2]],
                      [params_dim5[496, 3 * i + 2], params_dim5[497, 3 * i + 2], params_dim5[498, 3 * i + 2],
                       params_dim5[499, 3 * i + 2],
                       params_dim5[500, 3 * i + 2]],
                      [params_dim5[501, 3 * i + 2], params_dim5[502, 3 * i + 2], params_dim5[503, 3 * i + 2],
                       params_dim5[504, 3 * i + 2],
                       params_dim5[505, 3 * i + 2]]])
        b = np.array([[params_dim5[506, 3 * i + 2]], [params_dim5[507, 3 * i + 2]], [params_dim5[508, 3 * i + 2]],
                      [params_dim5[509, 3 * i + 2]],
                      [params_dim5[510, 3 * i + 2]]])
        return relu(x @ A.T + b.T)

    def best_random20_layer18(x):
        A = np.array([[params_dim5[511, 3 * i + 2], params_dim5[512, 3 * i + 2], params_dim5[513, 3 * i + 2],
                       params_dim5[514, 3 * i + 2],
                       params_dim5[515, 3 * i + 2]],
                      [params_dim5[516, 3 * i + 2], params_dim5[517, 3 * i + 2], params_dim5[518, 3 * i + 2],
                       params_dim5[519, 3 * i + 2],
                       params_dim5[520, 3 * i + 2]],
                      [params_dim5[521, 3 * i + 2], params_dim5[522, 3 * i + 2], params_dim5[523, 3 * i + 2],
                       params_dim5[524, 3 * i + 2],
                       params_dim5[525, 3 * i + 2]],
                      [params_dim5[526, 3 * i + 2], params_dim5[527, 3 * i + 2], params_dim5[528, 3 * i + 2],
                       params_dim5[529, 3 * i + 2],
                       params_dim5[530, 3 * i + 2]],
                      [params_dim5[531, 3 * i + 2], params_dim5[532, 3 * i + 2], params_dim5[533, 3 * i + 2],
                       params_dim5[534, 3 * i + 2],
                       params_dim5[535, 3 * i + 2]]])
        b = np.array([[params_dim5[536, 3 * i + 2]], [params_dim5[537, 3 * i + 2]], [params_dim5[538, 3 * i + 2]],
                      [params_dim5[539, 3 * i + 2]],
                      [params_dim5[540, 3 * i + 2]]])
        return relu(x @ A.T + b.T)

    def best_random20_layer19(x):
        A = np.array([[params_dim5[541, 3 * i + 2], params_dim5[542, 3 * i + 2], params_dim5[543, 3 * i + 2],
                       params_dim5[544, 3 * i + 2],
                       params_dim5[545, 3 * i + 2]],
                      [params_dim5[546, 3 * i + 2], params_dim5[547, 3 * i + 2], params_dim5[548, 3 * i + 2],
                       params_dim5[549, 3 * i + 2],
                       params_dim5[550, 3 * i + 2]],
                      [params_dim5[551, 3 * i + 2], params_dim5[552, 3 * i + 2], params_dim5[553, 3 * i + 2],
                       params_dim5[554, 3 * i + 2],
                       params_dim5[555, 3 * i + 2]],
                      [params_dim5[556, 3 * i + 2], params_dim5[557, 3 * i + 2], params_dim5[558, 3 * i + 2],
                       params_dim5[559, 3 * i + 2],
                       params_dim5[560, 3 * i + 2]],
                      [params_dim5[561, 3 * i + 2], params_dim5[562, 3 * i + 2], params_dim5[563, 3 * i + 2],
                       params_dim5[564, 3 * i + 2],
                       params_dim5[565, 3 * i + 2]]])
        b = np.array([[params_dim5[566, 3 * i + 2]], [params_dim5[567, 3 * i + 2]], [params_dim5[568, 3 * i + 2]],
                      [params_dim5[569, 3 * i + 2]],
                      [params_dim5[570, 3 * i + 2]]])
        return relu(x @ A.T + b.T)

    def best_random20_layer20(x):
        A = np.array([[params_dim5[571, 3 * i + 2], params_dim5[572, 3 * i + 2], params_dim5[573, 3 * i + 2],
                       params_dim5[574, 3 * i + 2],
                       params_dim5[575, 3 * i + 2]],
                      [params_dim5[576, 3 * i + 2], params_dim5[577, 3 * i + 2], params_dim5[578, 3 * i + 2],
                       params_dim5[579, 3 * i + 2],
                       params_dim5[580, 3 * i + 2]],
                      [params_dim5[581, 3 * i + 2], params_dim5[582, 3 * i + 2], params_dim5[583, 3 * i + 2],
                       params_dim5[584, 3 * i + 2],
                       params_dim5[585, 3 * i + 2]],
                      [params_dim5[586, 3 * i + 2], params_dim5[587, 3 * i + 2], params_dim5[588, 3 * i + 2],
                       params_dim5[589, 3 * i + 2],
                       params_dim5[590, 3 * i + 2]],
                      [params_dim5[591, 3 * i + 2], params_dim5[592, 3 * i + 2], params_dim5[593, 3 * i + 2],
                       params_dim5[594, 3 * i + 2],
                       params_dim5[595, 3 * i + 2]]])
        b = np.array([[params_dim5[596, 3 * i + 2]], [params_dim5[597, 3 * i + 2]], [params_dim5[598, 3 * i + 2]],
                      [params_dim5[599, 3 * i + 2]],
                      [params_dim5[600, 3 * i + 2]]])
        return relu(x @ A.T + b.T)

    layer1 = best_random20_layer1(random_points)
    layer2 = best_random20_layer2(layer1)
    layer3 = best_random20_layer3(layer2)
    layer4 = best_random20_layer4(layer3)
    layer5 = best_random20_layer5(layer4)
    layer6 = best_random20_layer6(layer5)
    layer7 = best_random20_layer7(layer6)
    layer8 = best_random20_layer8(layer7)
    layer9 = best_random20_layer9(layer8)
    layer10 = best_random20_layer10(layer9)
    layer11 = best_random20_layer11(layer10)
    layer12 = best_random20_layer12(layer11)
    layer13 = best_random20_layer13(layer12)
    layer14 = best_random20_layer14(layer13)
    layer15 = best_random20_layer15(layer14)
    layer16 = best_random20_layer16(layer15)
    layer17 = best_random20_layer17(layer16)
    layer18 = best_random20_layer18(layer17)
    layer19 = best_random20_layer19(layer18)
    layer20 = best_random20_layer20(layer19)


    print("Number ",i+1," % dead for random 20:\n")
    print()
    print("Out Bias:",params_dim5[246, 3*i+2])
    print()
    print("1st layer, first node: ", percent_dead_node1(layer1), "%")
    print("1st layer, second node: ", percent_dead_node2(layer1), "%")
    print("1st layer, third node: ", percent_dead_node3(layer1), "%")
    print("1st layer, fourth node: ", percent_dead_node4(layer1), "%")
    print("1st layer, fifth node: ", percent_dead_node5(layer1), "%")
    print()
    print("2nd layer, first node: ", percent_dead_node1(layer2), "%")
    print("2nd layer, second node: ", percent_dead_node2(layer2), "%")
    print("2nd layer, third node: ", percent_dead_node3(layer2), "%")
    print("2nd layer, fourth node: ", percent_dead_node4(layer2), "%")
    print("2nd layer, fifth node: ", percent_dead_node5(layer2), "%")
    print()
    print("3rd layer, first node: ",percent_dead_node1(layer3), "%")
    print("3rd layer, second node: ",percent_dead_node2(layer3), "%")
    print("3rd layer, third node: ",percent_dead_node3(layer3), "%")
    print("3rd layer, fourth node: ",percent_dead_node4(layer3), "%")
    print("3rd layer, fifth node: ",percent_dead_node5(layer3), "%")
    print()
    print("4th layer, first node: ", percent_dead_node1(layer4), "%")
    print("4th layer, second node: ", percent_dead_node2(layer4), "%")
    print("4th layer, third node: ", percent_dead_node3(layer4), "%")
    print("4th layer, fourth node: ", percent_dead_node4( layer4), "%")
    print("4th layer, fifth node: ", percent_dead_node5(layer4), "%")
    print()
    print("5th layer first node: ", percent_dead_node1(layer5), "%")
    print("5th layer, second node: ", percent_dead_node2(layer5), "%")
    print("5th layer, third node: ", percent_dead_node3(layer5), "%")
    print("5th layer, fourth node: ", percent_dead_node4(layer5), "%")
    print("5th layer, fifth node: ", percent_dead_node5(layer5), "%")
    print()
    print("6th layer, first node: ", percent_dead_node1(layer6), "%")
    print("6th layer, second node: ", percent_dead_node2(layer6), "%")
    print("6th layer, third node: ", percent_dead_node3(layer6), "%")
    print("6th layer, fourth node: ", percent_dead_node4(layer6), "%")
    print("6th layer, fifth node: ", percent_dead_node5(layer6), "%")
    print()
    print("7th layer, first node: ", percent_dead_node1(layer7), "%")
    print("7th layer, second node: ", percent_dead_node2(layer7), "%")
    print("7th layer, third node: ", percent_dead_node3(layer7), "%")
    print("7th layer, fourth node: ", percent_dead_node4(layer7), "%")
    print("7th layer, fifth node: ", percent_dead_node5(layer7), "%")
    print()
    print("8th layer, first node: ", percent_dead_node1(layer8), "%")
    print("8th layer, second node: ", percent_dead_node2(layer8), "%")
    print("8th layer, third node: ", percent_dead_node3(layer8), "%")
    print("8th layer, fourth node: ", percent_dead_node4(layer8), "%")
    print("8th layer, fifth node: ", percent_dead_node5(layer8), "%")
    print()
    print("9th layer, first node: ", percent_dead_node1(layer9), "%")
    print("9th layer, second node: ", percent_dead_node2(layer9), "%")
    print("9th layer, third node: ", percent_dead_node3(layer9), "%")
    print("9th layer, fourth node: ", percent_dead_node4(layer9), "%")
    print("9th layer, fifth node: ", percent_dead_node5(layer9), "%")
    print()
    print("10th layer, first node: ",  percent_dead_node1(layer10), "%")
    print("10th layer, second node: ", percent_dead_node2(layer10), "%")
    print("10th layer, third node: ",  percent_dead_node3(layer10), "%")
    print("10th layer, fourth node: ", percent_dead_node4(layer10), "%")
    print("10th layer, fifth node: ",  percent_dead_node5(layer10), "%")
    print()
    print("11th layer, first node: ",  percent_dead_node1(layer11), "%")
    print("11th layer, second node: ", percent_dead_node2(layer11), "%")
    print("11th layer, third node: ",  percent_dead_node3(layer11), "%")
    print("11th layer, fourth node: ", percent_dead_node4(layer11), "%")
    print("11th layer, fifth node: ",  percent_dead_node5(layer11), "%")
    print()
    print("12th layer, first node: ",  percent_dead_node1(layer12), "%")
    print("12th layer, second node: ", percent_dead_node2(layer12), "%")
    print("12th layer, third node: ",  percent_dead_node3(layer12), "%")
    print("12th layer, fourth node: ", percent_dead_node4(layer12), "%")
    print("12th layer, fifth node: ",  percent_dead_node5(layer12), "%")
    print()
    print("13th layer, first node: ",  percent_dead_node1(layer13), "%")
    print("13th layer, second node: ", percent_dead_node2(layer13), "%")
    print("13th layer, third node: ",  percent_dead_node3(layer13), "%")
    print("13th layer, fourth node: ", percent_dead_node4(layer13), "%")
    print("13th layer, fifth node: ",  percent_dead_node5(layer13), "%")
    print()
    print("14th layer, first node: ",  percent_dead_node1(layer14), "%")
    print("14th layer, second node: ", percent_dead_node2(layer14), "%")
    print("14th layer, third node: ",  percent_dead_node3(layer14), "%")
    print("14th layer, fourth node: ", percent_dead_node4(layer14), "%")
    print("14th layer, fifth node: ",  percent_dead_node5(layer14), "%")
    print()
    print("15th layer, first node: ",  percent_dead_node1(layer15), "%")
    print("15th layer, second node: ", percent_dead_node2(layer15), "%")
    print("15th layer, third node: ",  percent_dead_node3(layer15), "%")
    print("15th layer, fourth node: ", percent_dead_node4(layer15), "%")
    print("15th layer, fifth node: ",  percent_dead_node5(layer15), "%")
    print()
    print("16th layer, first node: ",  percent_dead_node1(layer16), "%")
    print("16th layer, second node: ", percent_dead_node2(layer16), "%")
    print("16th layer, third node: ",  percent_dead_node3(layer16), "%")
    print("16th layer, fourth node: ", percent_dead_node4(layer16), "%")
    print("16th layer, fifth node: ",  percent_dead_node5(layer16), "%")
    print()
    print("17th layer, first node: ",  percent_dead_node1(layer17), "%")
    print("17th layer, second node: ", percent_dead_node2(layer17), "%")
    print("17th layer, third node: ",  percent_dead_node3(layer17), "%")
    print("17th layer, fourth node: ", percent_dead_node4(layer17), "%")
    print("17th layer, fifth node: ",  percent_dead_node5(layer17), "%")
    print()
    print("18th layer, first node: ",  percent_dead_node1(layer18), "%")
    print("18th layer, second node: ", percent_dead_node2(layer18), "%")
    print("18th layer, third node: ",  percent_dead_node3(layer18), "%")
    print("18th layer, fourth node: ", percent_dead_node4(layer18), "%")
    print("18th layer, fifth node: ",  percent_dead_node5(layer18), "%")
    print()
    print("19th layer, first node: ",  percent_dead_node1(layer19), "%")
    print("19th layer, second node: ", percent_dead_node2(layer19), "%")
    print("19th layer, third node: ",  percent_dead_node3(layer19), "%")
    print("19th layer, fourth node: ", percent_dead_node4(layer19), "%")
    print("19th layer, fifth node: ",  percent_dead_node5(layer19), "%")
    print()
    print("20th layer, first node: ",  percent_dead_node1(layer20), "%")
    print("20th layer, second node: ", percent_dead_node2(layer20), "%")
    print("20th layer, third node: ",  percent_dead_node3(layer20), "%")
    print("20th layer, fourth node: ", percent_dead_node4(layer20), "%")
    print("20th layer, fifth node: ",  percent_dead_node5(layer20), "%")
    print()
    