import pickle
from matplotlib import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL.Image import Image
import mpl_toolkits.axes_grid1 as axes_grid1


DATASET='MTL'
DATASET='Geolife'
interpolation=True
mode='interpolatedLinear_1s'
trip_min=20
print(DATASET,interpolation,trip_min,mode)

prefix = '/home/xieyuan/Transportation-mode/Traj2Image/'

# prefix = '/home/xieyuan/Transportation-mode/TS2Vec/datafiles/'
mode='interpolatedLinear_1s'

# DATASET='Huawei'
# DATASET = 'MTL'
DATASET = 'Geolife'
interpolation=True                                                                                          
trip_min=20     


                                                                                  
# --> trips
filename=prefix+DATASET+'/trips_motion_features_NotFixedLength_woOutliers_xy_traintest_'+mode+'_trip%d.pickle'%trip_min
with open(filename, 'rb') as f:
    trip_motion_all_user_with_label_train, trip_motion_all_user_with_label_test,trip_motion_all_user_wo_label = pickle.load(f)

# trip_length_labeled_train = [trip[0].shape[1] for trip in trip_motion_all_user_with_label_train]
# trip_length_labeled_test =  [trip[0].shape[1] for trip in trip_motion_all_user_with_label_test]
# trip_length_unlabeled = [trip.shape[1] for trip in trip_motion_all_user_wo_label]

# print('all trips:', len(trip_motion_all_user_with_label_train))

# print('Descriptive statistics for labeled train',  pd.Series(trip_length_labeled_train).describe(percentiles=[0.05, 0.1, 0.15,
#                                                                                                       0.25, 0.5, 0.75,
#                                                                                                       0.85, 0.9, 0.95]))
# print('Descriptive statistics for labeled test',  pd.Series(trip_length_labeled_test).describe(percentiles=[0.05, 0.1, 0.15,
#                                                                                                       0.25, 0.5, 0.75,
#                                                                                                       0.85, 0.9, 0.95]))
# print('Descriptive statistics for unlabeled',  pd.Series(trip_length_unlabeled).describe(percentiles=[0.05, 0.1, 0.15,
#                                                                                                       0.25, 0.5, 0.75,
#                                                                                                       0.85, 0.9, 0.95]))

lat_gap = [max(trip[0][-3]) - min(trip[0][-3]) for trip in trip_motion_all_user_with_label_train]
lon_gap = [max(trip[0][-2]) - min(trip[0][-2]) for trip in trip_motion_all_user_with_label_train]
label_train = [trip[1] for trip in trip_motion_all_user_with_label_train]

print('Descriptive statistics for labeled train',  pd.Series(lat_gap).describe(percentiles=[0.05, 0.1, 0.15,
                                                                                            0.25, 0.5, 0.6, 0.65, 0.7, 0.75,
                                                                                            0.8, 0.85, 0.9, 0.95, 0.99]))
print('Descriptive statistics for labeled test',  pd.Series(lon_gap).describe(percentiles=[0.05, 0.1, 0.15,
                                                                                            0.25, 0.5, 0.6, 0.65, 0.7, 0.75,
                                                                                            0.8, 0.85, 0.9, 0.95, 0.99]))

pixel_size = 128
sub_lat = 0.034985/pixel_size
sub_lon = 0.030509/pixel_size

# lable 0: 0, 2     50%     0.009341/0.007467
# lable 1: 3, 6     65%     0.021197/0.017117
# label 2: 33, 8071 95%     0.112131/0.149708
# label 3: 1, 4     75%     0.034985/0.030509

temp_train = [trip_motion_all_user_with_label_train[i] for i in [1, 4]]



image_count = 0
fig = plt.figure()
grid = axes_grid1.AxesGrid(
    fig, 111, nrows_ncols=(2, 2), axes_pad = 0.5, cbar_location = "right",
    cbar_mode="each", cbar_size="15%", cbar_pad="5%",)
fig = plt.figure()


for trip in temp_train:

    shape_array = np.zeros((pixel_size, pixel_size))
    count_array = np.zeros((pixel_size, pixel_size))
    final_array = np.zeros((pixel_size, pixel_size))

    lat = trip[0][-3]
    lon = trip[0][-2]

    print(trip[1])

    start_point = (min(lat), min(lon))

    lat_id = [int((point - start_point[0])/sub_lat) for point in lat]
    lon_id = [int((point - start_point[1])/sub_lon) for point in lon]


    lat_lon_id = list(zip(lon_id, lat_id))
    print(lat_lon_id)
    for id in lat_lon_id:
        shape_array[id[0]][id[1]] = 1
        count_array[id[0]][id[1]] += 1

    print(shape_array)
    print(count_array)
    print(count_array.max())
    final_array = (count_array - count_array.min())/(count_array.max() - count_array.min())
    print(final_array)

    im0 = grid[image_count].imshow(shape_array, cmap='gray', interpolation='nearest')
    grid.cbar_axes[0].colorbar(im0)

    im1 = grid[image_count+1].imshow(final_array, cmap='jet', interpolation='nearest')
    grid.cbar_axes[1].colorbar(im1)



    image_count += 2






# --> segments
# train_full_segments = []

# filename = prefix + DATASET + '/traindata_4class_xy_traintest_interpolatedLinear_1s_trip%d_new_001meters.pickle'%trip_min
# with open(filename, 'rb') as f:
#         kfold_dataset, X_unlabeled = pickle.load(f)
# dataset = kfold_dataset
# # print('dataset:', dataset)
# train_x_geolife = dataset[0]
# train_y_geolife = dataset[1]
# x_unlabeled_geilife = X_unlabeled

# lat_gap = []
# lon_gap = []
# full_seg = 0
# for seg1 in train_x_geolife:
    
#     for seg in seg1:
#         lat = []
#         lon = []
#         lat_count = 0
#         for point in seg:
#             lat.append(point[0])
#             lon.append(point[1])
#             if point[0] != 0 and point[1] != 0:
#                 lat_count += 1      
#         # print(lat_count, len(lat))
#         if lat_count == len(lat):
#             full_seg += 1
#             train_full_segments.append(seg)

#     '''remove all padding 0.0 in the segments'''
#     while 0.0 in lat:
#         lat.remove(0.0)

#     '''only count the lat/lon gap for full segments '''
#     if len(lat) == 248:
#         lat_gap.append(max(lat) - min(lat))
#         lon_gap.append(max(lon) - min(lon))

# print('all segments:', len(train_x_geolife))
# print('full segments:', len(train_full_segments))

# print('Descriptive statistics for labeled train',  pd.Series(lat_gap).describe(percentiles=[0.05, 0.1, 0.15,
#                                                                                                       0.25, 0.5, 0.75, 0.8,
#                                                                                                       0.85, 0.9, 0.95]))
# print('Descriptive statistics for labeled test',  pd.Series(lon_gap).describe(percentiles=[0.05, 0.1, 0.15,
#                                                                                                       0.25, 0.5, 0.75, 0.8,
#                                                                                                       0.85, 0.9, 0.95]))

# image_size = '16x16'


# sub_lat = 0.030765/16
# sub_lon = 0.043949/16

# print('sub length lat:', sub_lat)
# print('sub length lon:', sub_lon)


# shape_array = np.zeros((16, 16))
# count_array = np.zeros((16, 16))

# for seg in train_full_segments:
#     lat = [point[0] for point in seg]
#     lon = [point[1] for point in seg]

#     plt.plot(lat, lon)
#     plt.show()

#     start_point = (min(lat), min(lon))

#     lat_id = [((point[0] - start_point[0]), (point[0] - start_point[0])/sub_lat) for point in seg]
#     lon_id = [(point[1] - start_point[1])/sub_lon for point in seg]
#     break

# for index, i in enumerate(lat_id):
#     print(index, i)

