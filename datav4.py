#   Introduction
#   The purpose of fist code:
#   Get the simulation data;
#  
#   In this research, the data is stored in TXT file.
#   And the incident muon is uniform and collimated accelerator muon, the incident direction is the z direction,
#   scintillators is the XOY plane and the object is at the origin point.

#   This is the code of format conversion for later PoCA algorithm
#==========================================================================

# import ROOT as r
import numpy as np
from scipy.stats import norm
import sys
# from matplotlib.colors import Normalize
import os

all_stds = []
all_means = []
all_poca = []
all_voxel = []

one_for_all = []
outline = 0
def create_voxel_dataset(path_file):
    #Parameters of Scintillators
    L = 50   # Length: cm
    a = 3    # Base: cm
    h = 1.5  # Height: cm

    # Find the vertex point for one scintillator code
    Z = [[60.0,58.5,58.5,57.0],[30.0,28.5,28.5,27.0],[-30.0,-31.5,-31.5,-33.0],[-60.0,-61.5,-61.5,-63.0]]
    X1 = [-22.5, -19.5, -16.5, -13.5, -10.5, -7.5, -4.5, -1.5, 1.5, 4.5, 7.5, 10.5, 13.5, 16.5, 19.5, 22.5]
    X2 = [-21, -18, -15, -12, -9, -6, -3, 0, 3, 6, 9, 12, 15, 18, 21, 24]
    Y1 = [-23.25, -20.25, -17.25, -14.25, -11.25, -8.25, -5.25, -2.25, 0.75, 3.75, 6.75, 9.75, 12.75, 15.75, 18.75, 21.75]
    Y2 = [-21.75, -18.75, -15.75, -12.75, -9.75, -6.75, -3.75, -0.75, 2.25, 5.25, 8.25, 11.25, 14.25, 17.25, 20.25, 23.25]

    ########## Data_1 taking
    Data_1 = []
    Data_2 = []
    Data_3 = []
    Data_4 = []
    outline = 0   # Limit the number of muon events used for later imaging

    # Path of simulation data on SJTU/INPAC server cluster
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path_1 = os.path.join(base_dir, path_file)
    # path_1 = "C:/Users/salva/PycharmProjects/pythonProject2/TDLI/AI_Auto_{2025-08-04_13-43-28}_{33122323}.log"

    def extract_data(path, Data_number):

        for line in open('{}'.format(path),'r'):      #Read line by line
            a = line.split()    # Read string and split with "      "
            global outline

            p1=[]               # Store center position of hit pixel in Track detector '1'
            p2=[]               # ...... in Track detector '2'
            p3=[]               # ...... in Track detector '3'
            p4=[]               # ...... in Track detector '4'
            data=[]             # Store [p1,p2,p3,p4]
            
            Vol=[]      # Record the det_ID of scintillators
            Edep=[]     # Record energy deposition in related scintillators

            if (len(a) %4 != 0) or (len(a) < 5): continue   # If data is not reasonable, it will be rejected.

            for i in range(len(a)):
                if i % 4 == 0: Vol.append(int(a[i]))        #1/3 is det_ID
                if i % 4 == 2: Edep.append(float(a[i]))     #2/3 is edep_muon 
            
            Vol,Edep = zip(*sorted(zip(Vol, Edep)))     # To sort two related lists to let Vol from small to large
            
            ############## From the VolumeID to position information
            current_layer = 1   # The layer number
            Flag_sublayer = 0   # The Flag from X position to Y position
            
            XE = []     # The X position list in 4 layers
            YE = []
            ZE = []

            xe = 0      # The x in 4 layers using CoG
            ye = 0
            ze = 0
            Ze = []     # To store 2 z position in up and down layer
            subedep=0   # The edep in one sublayer
            old_sublayer=0  # Help to trigger Flag_sublayer

            for i in range(len(Vol)):

                layer = int(Vol[i]/1000)
                layer_index = layer - 1     # The layer information
                sublayer = int(int(Vol[i]/100) % 10)
                sublayer_index = sublayer - 1   # The sublayer information
                code = int(Vol[i] % 100) - 1
                edep = Edep[i]                  # The edep in related scintillator

                if old_sublayer==2 and sublayer==3: # If from X sublayer to Y sublayer, Trigger
                    Flag_sublayer = 1

                if (layer == current_layer) and (Flag_sublayer == 1):   # If from X sublayer to Y sublayer
                    if(subedep==0): continue
                    XE.append(xe/subedep)
                    Ze.append(ze/subedep)
                    
                    xe = 0
                    ye = 0
                    ze = 0
                    subedep = 0
                    Flag_sublayer = 0
                if layer > current_layer:
                    if(subedep==0): continue
                    YE.append(ye/subedep)
                    Ze.append(ze/subedep)                
                    ZE.append(Ze)
                    current_layer += 1
                    
                    Ze = []
                    xe = 0
                    ye = 0
                    ze = 0
                    subedep = 0

                z = Z[layer_index][sublayer_index]
                if sublayer==1:
                    x = X1[code]
                    xe += x*edep
                elif sublayer==2:
                    x = X2[code]
                    xe += x*edep
                elif sublayer==3:
                    y = Y1[code]
                    ye += y*edep
                elif sublayer==4:
                    y = Y2[code]
                    ye += y*edep
                
                ze += z*edep
                subedep += edep

                old_sublayer = sublayer     # Save current sublayer number

            # store 4th y layer position
            if(subedep==0): continue
            YE.append(ye/subedep)
            Ze.append(ze/subedep)                
            ZE.append(Ze)

            if (len(XE)<4 or len(YE)<4 or len(ZE)<4): continue
            y1 = (ZE[0][0]-ZE[0][1])*(YE[1]-YE[0])/(ZE[1][1]-ZE[0][1])+YE[0]    # Using y-z line to get y position in 1st layer
            y2 = (ZE[1][0]-ZE[0][1])*(YE[1]-YE[0])/(ZE[1][1]-ZE[0][1])+YE[0]    # ... 2nd layer
            y3 = (ZE[2][0]-ZE[3][1])*(YE[3]-YE[2])/(ZE[3][1]-ZE[2][1])+YE[2]    # ... 3rd layer
            y4 = (ZE[3][0]-ZE[3][1])*(YE[3]-YE[2])/(ZE[3][1]-ZE[2][1])+YE[2]    # ... 4th layer

            Y_new = [y1,y2,y3,y4]
            
            # Fill the position
            p1.append(XE[0])
            p1.append(Y_new[0])
            p1.append(ZE[0][0])
            
            p2.append(XE[1])
            p2.append(Y_new[1])
            p2.append(ZE[1][0])
            
            p3.append(XE[2])
            p3.append(Y_new[2])
            p3.append(ZE[2][0])
            
            p4.append(XE[3])
            p4.append(Y_new[3])
            p4.append(ZE[3][0])
            
            data.append(p1)
            data.append(p2)
            data.append(p3)
            data.append(p4)

            Data_number.append(data)      # Store the coordinates of each track as a set in a new list. It's for the analysis later

            outline += 1
        # if(outline>100000):break     # Use ... events for imaging
    # print(len(Data_1))                # Make sure the number
    # print(Data_1[:2] )                #See the contents of the list 'Data_1'


    extract_data(path_1, Data_1)
    ########################### starting the new program ########################

    def get_unit_vector(v_1, v_2):
        v1 = np.array(v_1)
        v2 = np.array(v_2)
        direction = v2 - v1
        magnitude = np.linalg.norm(direction)
        
        # unit vector between entrance and exit of dual layer of detectirs
        return direction / magnitude

    def angle_between_unit_vectors(u, v):
        dot_product = np.dot(u, v)
        # avoiding absurd results
        dot_product = np.clip(dot_product, -1.0, 1.0)
        angle_rad = np.arccos(dot_product)
        return angle_rad  # radians




    angle_variation_1 = []

    # com modulação de vetores, nao sei se esta funcionando
    def get_angle_variation(Data_number, angle_variation_number):

        for j in range (len(Data_number)):
                top_unit_vector = get_unit_vector(Data_number[j][0], Data_number[j][1])
                bottom_unit_vector = get_unit_vector(Data_number[j][2], Data_number[j][3])
                
                # print(angle_variation)


                unit_vr = get_unit_vector(Data_number[j][0], Data_number[j][1])
                # print(unit_vr)
                parameter = (Data_number[j][2][2] - Data_number[j][1][2]) / unit_vr[2]
                # print(parameter)
                v_expected = np.array(Data_number[j][1]) + unit_vr * parameter     # this is where the muon should land without scattering
                # print(v_expected)
                v_expected_unit = v_expected / np.linalg.norm(v_expected)

                v_delta =  v_expected - Data_number[j][2]
                # print(v_delta)

                v_delta_unit = v_delta / np.linalg.norm(v_delta)
                # print(v_delta_unit)

                alfa_observed = angle_between_unit_vectors(-unit_vr, v_delta_unit)

                alfa_expected = angle_between_unit_vectors(-v_expected_unit, v_delta_unit)

                # print("Alfa observado: ", alfa_observed)
                # print("Alfa esperado: ",alfa_expected)
                # print("Delta alfa: ", alfa_observed - alfa_expected)
                if (alfa_observed - alfa_expected) > 0:
                    mod = 1
                else:
                    mod = -1
                
                angle_variation_number.append(mod * angle_between_unit_vectors(top_unit_vector, bottom_unit_vector))

    get_angle_variation(Data_1, angle_variation_1)
        


    #################


    def find_intersection(P1, P2, P3, P4):
        """
        Finds the intersection point of two 3D lines defined by P1-P2 and P3-P4.
        Returns None if lines are parallel or do not intersect.
        """
        P1, P2, P3, P4 = np.array(P1), np.array(P2), np.array(P3), np.array(P4)
        
        # Direction vectors
        d1 = P2 - P1  # Direction of Line 1
        d2 = P4 - P3  # Direction of Line 2
        
        # Vector between a point on Line 1 and a point on Line 2
        P3_P1 = P3 - P1
        
        # Solve for t and s in the system: P1 + t*d1 = P3 + s*d2
        # Using least squares for numerical stability
        A = np.column_stack((d1, -d2))
        t_s, _, _, _ = np.linalg.lstsq(A, P3_P1, rcond=None)
        t, s = t_s
        
        # Calculate intersection point
        intersection = P1 + t * d1
        
        return intersection.tolist()

    x_min, x_max = -28, 28 
    y_min, y_max = -28, 28 
    z_min, z_max = -34, 34 
    n_voxels_x = 14 
    n_voxels_y = 14 
    n_voxels_z = 17 

    # Generate edges (boundaries)
    x_edges = np.linspace(x_min, x_max, n_voxels_x + 1)
    y_edges = np.linspace(y_min, y_max, n_voxels_y + 1)
    z_edges = np.linspace(z_min, z_max, n_voxels_z + 1)

    # encontrando os midpoints
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    z_centers = (z_edges[:-1] + z_edges[1:]) / 2

    xx, yy, zz = np.meshgrid(x_centers, y_centers, z_centers, indexing='ij')


    def get_voxel_center(x, y, z):


        x_idx = np.searchsorted(x_edges, x) - 1
        y_idx = np.searchsorted(y_edges, y) - 1
        z_idx = np.searchsorted(z_edges, z) - 1
        
        # Clamp indices to handle edge cases (e.g., x = x_max)
        x_idx = np.clip(x_idx, 0, n_voxels_x - 1)
        y_idx = np.clip(y_idx, 0, n_voxels_y - 1)
        z_idx = np.clip(z_idx, 0, n_voxels_z - 1)
        
        # Return the voxel center
        return (x_centers[x_idx], y_centers[y_idx], z_centers[z_idx])




    temp_inter_point = []
    def_inter_point = []
    acceptable_tracks_full = []


    for n in range(len(Data_1)):
        temp_inter_point.append(find_intersection(Data_1[n][0],Data_1[n][1],Data_1[n][2],Data_1[n][3]))
        if  25 > temp_inter_point[n][0] > -25 and 25 > temp_inter_point[n][1] > -25 and 29 > temp_inter_point[n][2] > -32:
            def_inter_point.append(temp_inter_point[n])

            a = get_voxel_center(temp_inter_point[n][0], temp_inter_point[n][1], temp_inter_point[n][2])
            acceptable_tracks_full.append([angle_variation_1[n], a])
    # print(def_inter_point)
    print("")
    print(len(temp_inter_point))
    print(len(def_inter_point))



    counter = np.column_stack((
        xx.ravel(),
        yy.ravel(),
        zz.ravel(),
        np.zeros_like(xx.ravel()) 
    ))

    counter = np.array(counter)  # se pa muda isso aqui

    print(get_voxel_center(def_inter_point[0][0], def_inter_point[0][1], def_inter_point[0][2]))

    def increment_voxel_counter(target_xyz, voxel_array):
        # Find matching indices (only check first 3 columns)
        matches = np.where(np.all(np.isclose(voxel_array[:, :3], target_xyz, atol=1e-6), axis=1))[0]
        
        if len(matches) > 0:
            voxel_array[matches, 3] += 1
            return voxel_array, True
        return voxel_array, False

    # exemplo
    # target_center = [-24.5, -24.5, -31.5]  # Just x,y,z

    for n in range(len(def_inter_point)):
        target_center = get_voxel_center(def_inter_point[n][0], def_inter_point[n][1], def_inter_point[n][2])
        counter, success = increment_voxel_counter(target_center, counter)





    np.set_printoptions(threshold=sys.maxsize, linewidth=200)
    # print(counter)

    doido = 0
    doido2 = 0
    p_max = 0
    for i in counter:
        if i[3] > 0:
            doido = doido + 1

    for i in counter:
        if i[3] > p_max:
            p_max = i[3]

    print(doido)
    print(len(def_inter_point))

    for i in counter:
        if i[3] == 0:
            doido2 = doido2 + 1
    print("Total com 0 PoCA:", doido2)
    print("Total com mais de 0 PoCA:", doido)
    print("Isso representa ", doido/(n_voxels_x * n_voxels_y * n_voxels_z) * 100 ,"porcento do total" )



    # print(acceptable_tracks_full)
    tolerance = 0.01
    
    # global all_stds
    # global all_means
    # global all_poca
    # global all_voxel
    def rotate(x,y):
        if x == -2 and y == 2:
            return 2,2
        if x == 2 and y == 2:
            return 2,-2
        if x == 2 and y == -2:
            return -2,-2
        if x == -2 and y == -2:
            return -2,2
        
    def identity(x,y):
        return (x,y)

    def voxel_stats(x,y,z, rotation_number):
        media = []
        for i in acceptable_tracks_full:
            if ((np.abs(i[1][0] - (x)) <= tolerance).all() and 
                (np.abs(i[1][1] - (y)) <= tolerance).all() and 
                (np.abs(i[1][2] - (z)) <= tolerance).all()):
                media.append(i[0])
                #
        mu, sigma = norm.fit(media)
        a, b = rotate(x,y)
        c, d = rotate(a,b)
        e, f = rotate(c,d)
        if rotation_number == 0:
            print(f"Voxel: [{x}, {y}, {z}] | PoCA density: {len(media)/ p_max} | Mean: {mu} | STD: {sigma}")
            all_voxel.append([x,y,z])

        if rotation_number ==1:
            print(f"Voxel: [{a}, {b}, {z}] | PoCA density: {len(media)/ p_max} | Mean: {mu} | STD: {sigma}")
            all_voxel.append([a,b,z])
        
        if rotation_number ==2:
            print(f"Voxel: [{c}, {d}, {z}] | PoCA density: {len(media)/ p_max} | Mean: {mu} | STD: {sigma}")
            all_voxel.append([c,d,z])
        
        if rotation_number ==3:
            print(f"Voxel: [{e}, {f}, {z}] | PoCA density: {len(media)/ p_max} | Mean: {mu} | STD: {sigma}")
            all_voxel.append([e,f,z])

            


        all_means.append(mu)
        all_stds.append(sigma)
        all_poca.append(len(media)/p_max)
        


    def execute(n):
        voxel_stats(-2,2,0,n)
        voxel_stats(2,2,0,n)
        voxel_stats(2,-2,0,n)
        voxel_stats(-2,-2,0,n)
        print("")
        print("")
        voxel_stats(-2,2,4,n)
        voxel_stats(2,2,4,n)
        voxel_stats(2,-2,4,n)
        voxel_stats(-2,-2,4,n)
        print("")
    execute(0)
    execute(1)
    execute(2)
    execute(3)
    
    print(all_voxel)

    print("")
    print("")
    print("")
    print("")
    
    

base_dir = os.path.dirname(os.path.abspath(__file__))
log_file_path = os.path.join(base_dir, "log_file_list.txt")

with open(log_file_path, 'r') as file:
    lines = file.read().splitlines()  # Automatically removes '\n'

for n in range(len(lines)):
    create_voxel_dataset(lines[n])

# 
# print(one_for_all[0][0,0,1])

def normalized_data(list):
        percent_99_mean = np.percentile(list, 99.9)
        normalized_list = list / percent_99_mean
        return normalized_list

def min_max_normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))



normal_std = normalized_data(all_stds)
normal_mean = min_max_normalize(all_means)

voxel_data = []
for i in range(len(normal_std)):
    voxel_data.append([all_voxel[i], all_poca[i] , normal_mean[i], normal_std[i]  ])

print(voxel_data)


from collections import defaultdict

def prepare_cnn_data(voxel_data, voxels_per_simulation=8):
    """Properly stacks original and rotated simulations"""
    
    # Calculate total base simulations (before rotations)
    num_base_simulations = len(voxel_data) // (voxels_per_simulation * 2)  # Each has original + rotated
    
    # Position mapping
    pos_to_index = {
        (-2, -2, 0): (0, 0, 0),
        (2, -2, 0): (1, 0, 0),
        (-2, 2, 0): (0, 1, 0),
        (2, 2, 0): (1, 1, 0),
        (-2, -2, 4): (0, 0, 1),
        (2, -2, 4): (1, 0, 1),
        (-2, 2, 4): (0, 1, 1),
        (2, 2, 4): (1, 1, 1)
    }
    
    # Initialize output array (2x num_base_simulations for original + rotated)
    X_sample = np.zeros((num_base_simulations * 2, 2, 2, 2, 3))
    
    for i, voxel in enumerate(voxel_data):
        # Determine simulation number (grouping original + rotated pairs)
        sim_num = i // voxels_per_simulation
        
        # Get physical coordinates
        x, y, z = voxel[0]
        
        # Map to grid indices
        h, w, d = pos_to_index[(x, y, z)]
        
        # Store features
        X_sample[sim_num, h, w, d] = voxel[1:]
    
    return X_sample

# Process data
X_sample = prepare_cnn_data(voxel_data)

# Correct access pattern:
print("Checking rotations:")
print(X_sample[0, 0, 0, 0])  # Original simulation 0, position (0,0,0)
print(X_sample[1, 0, 1, 0])  # Rotated version of simulation 0, position (1,1,1)
print(X_sample[2, 1, 1, 0])  # Original simulation 1
print(X_sample[3, 1, 0, 0])  # Rotated version of simulation 1
print("")
# print(X_sample[4, 0, 0, 0])  # Original simulation 0, position (0,0,0)
# print(X_sample[5, 0, 1, 0])  # Rotated version of simulation 0, position (1,1,1)
# print(X_sample[6, 1, 1, 0])  # Original simulation 1
# print(X_sample[7, 1, 0, 0])  # Rotated version of simulation 1

print("")
print("Checking whole cube configuration, not rotated: ")

print(X_sample[0, 0, 1, 0])  
print(X_sample[0, 1, 1, 0])  
print(X_sample[0, 1, 0, 0]) 
print(X_sample[0, 0, 0, 0]) 



import pickle


path_pickle = os.path.join(base_dir, "test.pkl")

with open(path_pickle, "wb") as f:
    pickle.dump(X_sample, f)

# print(lines)
print(lines[0])
print(lines[0][31])
print(lines[0][32])
print(lines[0][33])
print(lines[0][34])