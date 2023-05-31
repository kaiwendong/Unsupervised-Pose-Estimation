# Unsupervised Learning of 3D-2D consistency for 3D pose Estimation from Multi-View Videos 

This branch is the code corresponding to the "3D scene pose estimation scenario".
To get the "3D Human pose estimation" code, switch to Human3D branch.

## Structure of the repository
The code is divided into 2 parts.
The first part is the data generation part. This part allows to (1) generate data from a modified version of the RLBench Simulation, and (2) export the data to a format that can be used to train our 3D pose Estimation model.
The second part is the proposed baseline for Unsupervised 3D pose Estimation.
Note that we provide ready-to-use data for the 5 tasks studied in our paper in the directory "pose_obj_3d/data/RLBench_pose_est_data", under the file "sim_all_formal.npz".

## Install 

```
# create your virtual environment (tested only with python 3.9)
conda create -n scene_pose_estimation python=3.9
conda activate scene_pose_estimation
```
### RLBench install 
We use a cutom version of RLBench, that allows us to acquire keypoints of interest while generating tasks demonstrations.
RLBench is built around PyRep and V-REP. First head to the 
[PyRep github](https://github.com/stepjam/PyRep) page and install.
**If you previously had PyRep installed, you will need to update your installation!**


Hopefully you have now installed PyRep and have run one of the PyRep examples.
Now lets install our custom RLBench:

```bash
cd RLBench
pip install -r requirements.txt
pip install .
```

### Additionnal dependencies for data generation

go back to the root directory and install requirements:
```bash
cd ..
pip install -r data_gen_requirements.txt
```


### Unsupervised 3D pose Estimation install
```bash
pip install -r  pose_est_requirements.txt
```

## Generate data

### Raw data generation
go to the data generation directory and launch dataset_management.py code, 
```bash
cd data_generation/RLBench
python generate_data.py  --path_dataset data/my_expe/ --path_dataset_config configs/dataset_config.json --n_processes 1
```
, where --path_dataset indicates where the generated data will be stored, and where --path_dataset_config describes some features of the data that needs to be collected. 
Here are the most useful features:
n_ex_per_variation : the number of examples to generate for each task.
headless_generation : whether to lauch the simulation GUI during data generation. GUI considerablyy slows down the data generation. We suggest to only use it for visualization/debug.
"train_task_ids": [0, 8, 41, 79]: defines the tasks for which you want to generate data. The task ids are organized following the order defined in "tasks_ids.txt", where the 0-th line of the file, featuring "reach_target" task corresponds to the task 0, and the 94-th line of the file, featuring task "PourFromCupToCup" correpsonds to task 94. Tasks 9, 12, 40, 54 and 79 correpsonds to the tasks used in our paper.

Once the data generation completed, you'll find raw demonstration data (multi-view image observations, point-clouds, robot states, robot actions, trajectories keypoints, ...) under your "path_dataset".
One more step is required to export data to a format that we can use to train the 3D pose estimation models.


### Export data for 3D pose Estimation model
```bash
export $path_dataset=data/my_expe/
export path_dataset_config=configs/dataset_config.json
sh convert_data_to_3D_pose_estimation.sh
```
The converted data will be available in the "sim_all_formal.npz" file.

### Training 3D pose estimation models

```bash
cd pose_obj_3d
```

training supervised model on task 0 : 
```bash
python run_h36m_waypoints.py --cfg ./cfg/submit/config_waypoints_task_0.yaml
```
training unsupervised model on task 0 :
```bash
python run_h36m_waypoints_unsup.py --cfg ./cfg/submit/config_u_waypoints_task_0.yaml
```

training supervised model on task 8 : 
```bash
python run_h36m_waypoints.py --cfg ./cfg/submit/config_waypoints_task_8.yaml
```
training unsupervised model on task 8 :
```bash
python run_h36m_waypoints_unsup.py --cfg ./cfg/submit/config_u_waypoints_task_8.yaml
```

### Monitor training
```bash
cd pose_obj_3d/log/submit/
```

```bash
tensorboard --logdir .  --bind_all
```


