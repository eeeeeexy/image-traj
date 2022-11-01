import pickle

filename = '/home/xieyuan/Transportation-mode/Traj2Image/Geolife/trips_traj2image_1s_trip20.pickle'
with open(filename, 'rb') as f:
    trip_train_image_with_label, trip_test_image_with_label = pickle.load(f) 