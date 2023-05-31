# Supervised Phase of Paper ID 12502
## Environment
```bash
conda env create --file environment.yaml
conda activate enrironment
```
## Dataset Preparation
[2D joints detection results for H36M_data, Extration Code:i6dd ](https://pan.baidu.com/s/1Wu6XEEuAtQLpttIAYQaE4Q?pwd=i6dd)
## Pretrained Model
[Checkpoint](Will be released soom.)
## Training
### Training on H36M with 27 frames and 4 views, batchsize=720
```bash
python -m pdb run_h36m.py --cfg ./cfg/submit/mht_gt_trans_t_27_no_res.yaml --gpu 1,2,3,4,5,6 
```
### Evaluating on H36M with 27 frames and 4 views
```bash
python -m pdb run_h36m.py --cfg ./cfg/submit/mht_gt_trans_t_27_no_res.yaml --eval --checkpoint ./where_you_put_checkpoint/model.bin --gpu 1,2,3,4,5,6 --n_frames 27  --eval_batch_size 360 --eval_n_frames 27
```
### Training on 3 views and test on 4, batchsize=720 
```bash
python -m pdb run_h36m.py --cfg ./cfg/submit/mht_gt_trans_t_7_no_res_3view.yaml --gpu 1,2,3,4,5,6
```
### Evaluating on H36M with 7 frames and 4 views
```bash
python -m pdb run_h36m.py --cfg ./cfg/submit/mht_gt_trans_t_7_no_res_3view.yaml --eval --checkpoint ./where_you_put_checkpoint/model.bin --gpu 0,1,2,3 --n_frames 7  --eval_batch_size 360 --eval_n_frames 7
```

