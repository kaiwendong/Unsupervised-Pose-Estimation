# Unsupervised Learning of 3D-2D consistency for 3D pose Estimation from Multi-View Videos 

This repo is the code corresponding to the unsupervised 3D human pose estimation scenario.
To get the supervised 3D Human pose estimation code, follow this link :
https://anonymous.4open.science/r/Unsupervised-Pose-Estimation-60B7/README.md

To get codes corresponding 3D scene pose estimation scenario, follow this link:
https://anonymous.4open.science/r/Unsupervised-Pose-Estimation-177D/README.md


## To get H3.6m data
clone this repository and create a "data/" directory at the root of the project:
```bash
mkdir data/
```
we used the link provided by https://github.com/lelexx/MTF-Transformer : 

https://pan.baidu.com/s/1Wu6XEEuAtQLpttIAYQaE4Q?pwd=i6dd

just go to that link and download all the files to the "data/" directory

## Install Required packages
### To launch unsupervised training on H36M:
```bash
pip install -r requirements.txt
```


## Models training

### To launch unsupervised training on H36M:
```bash
python run_h36m_relative_pose.py --cfg ./cfg/submit/u_P_R.yaml
```
the file common/config.py contains the default training configuration.
the parameters defined in ./cfg/submit/u_P_R.yaml overwrites the  common/config.py parameters.


