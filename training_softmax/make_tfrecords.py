import tf_utils
import os
import glob

base_dir = '/home/ancora-sirlab/arcade_claw_test/test'
dis_folder = './tfrecords-5k-train'

# crop box for the tray
crop_box = (200, 0, 1280, 1080)
it = 0
len_dir = len(glob.glob(base_dir+"/*04-0*"))

# save the data in src_folder into tfrecord in dis_folder
for data_dir in glob.glob(base_dir+"/*04-0*"):
    it += 1
    print('writing '+data_dir+' ('+str(it)+'/'+str(len_dir)+')')
    tf_utils.tf_writer(data_dir, dis_folder, crop_box)

