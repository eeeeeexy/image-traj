import pickle
from typing import Tuple
from matplotlib import image
from math import floor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL.Image import Image
import mpl_toolkits.axes_grid1 as axes_grid1
from collections import defaultdict

DATASET='MTL'
DATASET='Geolife'
num_classes = 4
other_features = False
normalize = True
print(DATASET)

prefix = '/home/xieyuan/Transportation-mode/Traj2Image/'                                                                                      
                                                
# --> trips
filename=prefix+DATASET+'/trips_motion_features_NotFixedLength_woOutliers_xy_traintest_interpolatedLinear_1s_trip20.pickle'
with open(filename, 'rb') as f:
    trip_motion_all_user_with_label_train, trip_motion_all_user_with_label_test,trip_motion_all_user_wo_label = pickle.load(f)


lat_gap = [max(trip[0][-3]) - min(trip[0][-3]) for trip in trip_motion_all_user_with_label_train]
lon_gap = [max(trip[0][-2]) - min(trip[0][-2]) for trip in trip_motion_all_user_with_label_train]
label_train = [trip[1] for trip in trip_motion_all_user_with_label_train]

print('Descriptive statistics for labeled train',  pd.Series(lat_gap).describe(percentiles=[0.05, 0.1, 0.15, 0.25, 0.5, 0.6, 0.65, 
                                                                                            0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]))
print('Descriptive statistics for labeled test',  pd.Series(lon_gap).describe(percentiles=[0.05, 0.1, 0.15, 0.25, 0.5, 0.6, 0.65, 
                                                                                            0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]))

pixel_size = 64
print('pixel size:', pixel_size)

image_center_point = (floor(pixel_size/2)-1, floor(pixel_size/2)-1)
sub_lat = 0.049327/pixel_size
sub_lon = 0.042016/pixel_size

print(image_center_point)


temp_train = [trip_motion_all_user_with_label_train[i] for i in [1, 4]]

def traj_to_image_shift(traj_data):

    image_count = 0
    fig = plt.figure()
    grid_shift = axes_grid1.AxesGrid(
        fig, 111, nrows_ncols=(1, 4), axes_pad = 0.5, cbar_location = "right",
        cbar_mode="each", cbar_size="15%", cbar_pad="5%",)
    fig = plt.figure()

    image_data = []
    label_list = []
    

    for index, trip in enumerate(traj_data):

        shape_array = np.zeros((pixel_size, pixel_size))
        count_array = np.zeros((pixel_size, pixel_size))
        final_array = np.zeros((pixel_size, pixel_size))

        acc_array_final = np.zeros((pixel_size, pixel_size))
        speed_array_final = np.zeros((pixel_size, pixel_size))
        time_array_final = np.zeros((pixel_size, pixel_size))

        acc_array = defaultdict(lambda  : defaultdict(list))
        for i in range(pixel_size):
            for j in range(pixel_size):
                acc_array[i][j] = []

        speed_array = defaultdict(lambda  : defaultdict(list))
        for i in range(pixel_size):
            for j in range(pixel_size):
                speed_array[i][j] = []

        time_array = defaultdict(lambda  : defaultdict(list))
        for i in range(pixel_size):
            for j in range(pixel_size):
                time_array[i][j] = []

        lat = trip[0][-3] 
        lon = trip[0][-2]

        acc = trip[0][-6]
        speed = trip[0][-7]
        jerk = trip[0][-5]
        bearing_rate = trip[0][-4]
        relative_distance = trip[0][0]

        start_point = (min(lat), min(lon))

        # lat_id = [int((point - start_point[0])/sub_lat) for point in lat if int((point - start_point[0])/sub_lat) <= pixel_size - 1]
        # lon_id = [int((point - start_point[1])/sub_lon) for point in lon if int((point - start_point[1])/sub_lon) <= pixel_size - 1]
        # print('-->list')

        lat_id = []
        lon_id = []
        acc_id = []
        speed_id = []
        time_id = []

        time_count = 0

        for point in zip(lat, lon):
            if int((point[0] - start_point[0])/sub_lat) <= pixel_size - 1 and int((point[1] - start_point[1])/sub_lon) <= pixel_size - 1:
                lat_id.append(int((point[0] - start_point[0])/sub_lat))
                lon_id.append(int((point[1] - start_point[1])/sub_lon))
                acc_id.append(acc[lat.tolist().index(point[0])])
                speed_id.append(speed[lat.tolist().index(point[0])])
                time_id.append(time_count)
                time_count += 1
        
        print(index, len(lat_id), len(lon_id), len(acc_id), len(speed_id), len(time_id))

        lat_lon_id = list(zip(lon_id, lat_id, acc_id, speed_id, time_id))
        # lat_lon_id = list(zip(lon_id, lat_id))


        if len(lat_lon_id) == 0:
            continue

        for id in lat_lon_id:
            shape_array[id[0]][id[1]] = 1      
        nonzero_array = np.nonzero(shape_array)

        traj_center_point = (int(nonzero_array[0].sum()/len(nonzero_array[0])), int(nonzero_array[1].sum()/len(nonzero_array[1])))
        lat_shift = image_center_point[0] - traj_center_point[0]
        lon_shift = image_center_point[1] - traj_center_point[1]
        shape_array = np.zeros((pixel_size, pixel_size))

        # print('-->image array')
        if max(nonzero_array[0]) < pixel_size/2 and max(nonzero_array[1]) < pixel_size/2:
            for id in lat_lon_id:
                shape_array[id[0]+lat_shift][id[1]+lon_shift] = 1
                count_array[id[0]+lat_shift][id[1]+lon_shift] += 1
                # print('acc shift array', id[0], id[1]+lon_shift, acc_array[id[0]+lat_shift][id[1]+lon_shift])
                acc_array[id[0]+lat_shift][id[1]+lon_shift].append(id[2])
                speed_array[id[0]+lat_shift][id[1]+lon_shift].append(id[3])
                time_array[id[0]+lat_shift][id[1]+lon_shift].append(id[4])

        elif max(nonzero_array[0]) < pixel_size/2 and max(nonzero_array[1]) > pixel_size/2:
            for id in lat_lon_id:
                shape_array[id[0]+lat_shift][id[1]] = 1
                count_array[id[0]+lat_shift][id[1]] += 1
                acc_array[id[0]+lat_shift][id[1]].append(id[2])
                speed_array[id[0]+lat_shift][id[1]].append(id[3])
                time_array[id[0]+lat_shift][id[1]].append(id[4])

        elif max(nonzero_array[0]) > pixel_size/2 and max(nonzero_array[1]) < pixel_size/2:
            for id in lat_lon_id:
                shape_array[id[0]][id[1]+lon_shift] = 1
                count_array[id[0]][id[1]+lon_shift] += 1
                acc_array[id[0]][id[1]+lon_shift].append(id[2])
                speed_array[id[0]][id[1]+lon_shift].append(id[3])
                time_array[id[0]][id[1]+lon_shift].append(id[4])
        else:
            for id in lat_lon_id:
                shape_array[id[0]][id[1]] = 1
                count_array[id[0]][id[1]] += 1
                acc_array[id[0]][id[1]].append(id[2])
                speed_array[id[0]][id[1]].append(id[3])
                time_array[id[0]][id[1]].append(id[4])

        # print('-->final normalized image')
        for i in range(pixel_size):
            for j in range(pixel_size):
                if len(acc_array[i][j]) == 0:
                    acc_array_final[i][j] = 0
                    speed_array_final[i][j] = 0
                    time_array_final[i][j] = 0   
                    # print(i, j, acc_array[i][j], acc_array_final[i][j])
                else:
                    acc_array_final[i][j] = np.array(acc_array[i][j]).mean()
                    speed_array_final[i][j] = np.array(speed_array[i][j]).mean()
                    time_array_final[i][j] = np.array(time_array[i][j]).mean()
                    # print(i, j, acc_array_final[i][j], speed_array_final[i][j], time_array_final[i][j])
        

        final_array = (count_array - count_array.mean())/count_array.std()
        acc_array_final = (acc_array_final - acc_array_final.mean())/acc_array_final.std()
        speed_array_final = (speed_array_final - speed_array_final.mean())/speed_array_final.std()
        time_array_final = (time_array_final - time_array_final.mean())/time_array_final.std()

        # print(acc_array_final)
        # im0 = grid_shift[image_count].imshow(shape_array, cmap='gray', interpolation='nearest')
        # grid_shift.cbar_axes[0].colorbar(im0)

        # im1 = grid_shift[image_count+1].imshow(time_array_final, cmap='gray', interpolation='nearest')
        # grid_shift.cbar_axes[1].colorbar(im1)

        # im2 = grid_shift[image_count+2].imshow(acc_array_final, cmap='gray', interpolation='nearest')
        # grid_shift.cbar_axes[2].colorbar(im2)

        # im3 = grid_shift[image_count+3].imshow(speed_array_final, cmap='gray', interpolation='nearest')
        # grid_shift.cbar_axes[3].colorbar(im3)

        # break 

        image_data.append([final_array, acc_array_final, speed_array_final, time_array_final])
        # image_data.append([final_array])
        label_list.append(trip[1])
    
    return image_data, label_list


if other_features == True:
    filename = prefix+DATASET+'/trips_traj2image_1s_trip20_shift_4class_pixelsize%d_normalize_acc+speed+time.pickle'%pixel_size
else:
    filename = prefix+DATASET+'/trips_traj2image_1s_trip20_shift_4class_pixelsize%d.pickle'%pixel_size

train_data, train_data_label = traj_to_image_shift(traj_data=trip_motion_all_user_with_label_train)
print(len(train_data), len(train_data_label))

test_data, test_data_label = traj_to_image_shift(traj_data=trip_motion_all_user_with_label_test)
print(len(test_data), len(test_data_label))

with open(filename, 'wb') as f:
    pickle.dump([train_data, train_data_label, test_data, test_data_label], f)

