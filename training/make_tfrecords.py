import tf_utils
import os

base_dir = '/home/ancora-sirlab/arcade_claw_test/test'
dis_folder = './tfrecords-5k'

# crop box for the tray
crop_box = (200, 0, 1280, 1080)
it = 0
len_dir = len(os.listdir(base_dir))
# save the data in src_folder into tfrecord in dis_folder
for dirs in os.listdir(base_dir):
    if dirs.split(' ')[0].split('-')[0] == '2017':
        data_dir = os.path.join(base_dir, dirs)
        it += 1
        print('writing '+data_dir+' ('+str(it)+'/'+str(len_dir)+')')
        tf_utils.tf_writer(data_dir, dis_folder, crop_box)

