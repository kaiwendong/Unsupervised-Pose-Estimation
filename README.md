# Unsupervised Learning of 3D-2D consistency for 3D pose Estimation from Multi-View Videos 

This repo is the code corresponding to the unsupervised 3D human pose estimation scenario.
To get the supervised 3D Human pose estimation code, follow this link :
https://anonymous.4open.science/r/Unsupervised-Pose-Estimation-60B7/README.md

To get codes corresponding 3D scene pose estimation scenario, follow this link:
https://anonymous.4open.science/r/Unsupervised-Pose-Estimation-177D/README.md

## To get H3.6m data :
[multi-view.tar.gz 提取码：ise7](https://pan.baidu.com/s/134vlOJmFKJSH7tiATfA6BQ?pwd=ise7)
```bash
cp multi-view.tar.gz ${path}/anaconda3/envs
cd ${path}/anaconda3/envs
mkdir -p multi-view
tar -xzf multi-view.tar.gz -C multi-view
conda activate multi-view
```
## 数据
[H36M_data 提取码：i6dd ](https://pan.baidu.com/s/1Wu6XEEuAtQLpttIAYQaE4Q?pwd=i6dd)
## 预训练模型
[H36M_checkpoint 提取码：yshz ](https://pan.baidu.com/s/1lvwDJ0K_lHlEzfC06g-InA?pwd=yshz)
## 训练

### To launch unsupervised training on H36M:
```bash
python run_h36m_relative_pose.py --cfg ./cfg/submit/u_P_R.yaml
```
the file common/config.py contains the default training configuration.
the parameters defined in ./cfg/submit/u_P_R.yaml overwrites the  common/config.py parameters.


