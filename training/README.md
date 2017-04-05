## Network Training Scripts

### data processing

Set the path to the collected data and path to save tfrecord in `make_tfrecords.py`

`src_folder = path/to/collected/data`
`dis_folder = path/to/save/tfrecord`

Set the crop box region in `make_tfrecords.py`

crop_box = (left, up, right, down)

Then to precess data

`python make_tfrecords`

***

### training process

Set all the parameters in the `test_train.py`

Set data_path to the path where you store .tfrecord files, the training script will use all the tfrecord files in the data_path folder.

`python test_train.py` to train grasping network in our dataset

When training finish, the network weights will be save in `checkpoint_path`, it can be automatically loaded in the next training. The summary files will be saved in `summary_path`, it can be used by TensorBoard.

***

### training visualisation

You can use TensorBoard to visualise the training process.

run

`tensorboard --logdir=summary`

to launch tensorboard application (port 6006 by default).

Then you can see the result in the browser. Now it just recording the loss value.

