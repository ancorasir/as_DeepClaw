import tf_utils


src_folder = '/home/ancora-sirlab/arcade_claw_test'
dis_folder = './tfrecords'

# crop box for the tray
crop_box = (400, 300, 1040, 840)
# save the data in src_folder into tfrecord in dis_folder
tf_utils.tfwriter(src_folder, dis_folder, crop_box)

