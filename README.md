# Supervised Phase of Paper ID 12502
## Environment
```bash
pip install -r requirement.txt
```
## Dataset Preparation
clone this repository and create a "data/" directory at the root of the project:
```bash
mkdir data/
```

we used the link provided by https://github.com/lelexx/MTF-Transformer : H36M_data Extration Code：i6dd just go to that link and download all the files to the "data/" directory
## Pretrained Model
[Checkpoint](Will be released soom.)
## Training & Evaluating
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

