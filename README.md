README
1. Environment:->

$ git clone https://github.com/codeslake/PVDNet.git
$ cd PVDNet

$ conda create -y --name PVDNet python=3.8 && conda activate PVDNet
# for CUDA10.2
$ sh install_CUDA10.2.sh
# for CUDA11.1
$ sh install_CUDA11.1.sh

2. Datasets

Download and unzip Su et al.'s dataset and Nah et al.'s dataset under [DATASET_ROOT]:

├── [DATASET_ROOT]
│   ├── train_DVD
│   ├── test_DVD
│   ├── train_nah
│   ├── test_nah
Note:

[DATASET_ROOT] is currently set to ./datasets/video_deblur. It can be specified by modifying config.data_offset in ./configs/config.py.

3. Copy the project code



copies the model files and the config files  \\
a) copy deblur.py to model/archs/deblur.py \\
b) copy pixel_volume.py to model/archs/pixel_volume.py \\
c) copy PVDNet.py to model/archs/PVDNet.py (our Video modeling framework) \\
d) copy config_PVDNet.py to config/config_PVDNet.py
e) project model is PVDNet_DVD_00231.pytorch is more than 25MB so not allowed to be copied.

4. To run training:

CUDA_VISIBLE_DEVICES=0,1 python -B -m torch.distributed.launch --nproc_per_node=2 --master_port=9000 run.py             --is_train             --mode PVDNet_DVD             --config config_PVDNet             --trainer trainer             --data DVD             -LRS CA             -b 2             -th 4             -dl             -ss             -dist

5. To test

CUDA_VISIBLE_DEVICES=0 python run.py --mode PVDNet_DVD --config config_PVDNet --data DVD --ckpt_abs_name /tmp/PVDNet_DVD_00231.pytorch

6. project model copied in the project_model folder, say, /tmpPVDNet_DVD_00231.pytorch

7. The model output in the /tmp/logs folder.
