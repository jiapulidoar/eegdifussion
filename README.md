# eegdifussion



## Environment setup

Create and activate conda environment named ```dreamdiffusion``` from the ```env.yaml```
```sh
conda env create -f env.yaml
conda activate dreamdiffusion
``` 



## Pre-training on EEG data
To perform the pre-training from scratch with defaults parameters, run 
```sh
python3 code/stageA1_eeg_pretrain.py
``` 

Hyper-parameters can be changed with command line arguments,
```sh
python3 code/stageA1_eeg_pretrain.py --mask_ratio 0.75 --num_epoch 800 --batch_size 2
```

Or the parameters can also be changed in ```code/config.py```

Multiple-GPU (DDP) training is supported, run with 
```sh
python -m torch.distributed.launch --nproc_per_node=NUM_GPUS code/stageA1_eeg_pretrain.py
```



## Finetune the Stable Diffusion with Pre-trained fMRI Encoder
In this stage, the cross-attention heads and pre-trained EEG encoder will be jointly optimized with EEG-image pairs. 

```sh
python3 code/eeg_ldm.py --dataset EEG  --num_epoch 300 --batch_size 4 --pretrain_mbm_path ../dreamdiffusion/pretrains/eeg_pretrain/checkpoint.pth
```


## Generating Images with Trained Checkpoints

```sh
python3 code/gen_eval_eeg.py --dataset EEG --model_path  pretrains/generation/checkpoint.pth
```



![Capture.PNG](https://hackmd.io/_uploads/BJssKIPmp.png)


