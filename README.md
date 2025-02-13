# RSANet: Relative-Sequence Quality Assessment Network for Gait Recognition in the Wild
[Paper](https://doi.org/10.1016/j.patcog.2024.111219) has been accepted in Pattern Recognition. This is the code for it.
# Operating Environments
## Pytorch Environment
* Pytorch=1.11.0
* Python=3.8
# CheckPoints
* The checkpoint for Gait3D [link]().
* The checkpoint for GREW [link]().
# Train and Test
## Train
```
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port=1354 opengait/main.py --cfgs ./configs/RSANet/RSANet_Gait3D.yaml --phase train
```
## Test
```
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port=1354 opengait/main.py --cfgs ./configs/RSANet/RSANet_Gait3D.yaml --phase test
```
* python -m torch.distributed.launch: DDP launch instruction.
* --nproc_per_node: The number of gpus to use, and it must equal the length of CUDA_VISIBLE_DEVICES.
* --cfgs: The path to config file.
* --phase: Specified as train or test.
# Acknowledge
The codebase is based on [OpenGait](https://github.com/ShiqiYu/OpenGait).
# Citation
```
@inproceedings{peng2024glgait,
  title={GLGait: A Global-Local Temporal Receptive Field Network for Gait Recognition in the Wild},
  author={Peng, Guozhen and Wang, Yunhong and Zhao, Yuwei and Zhang, Shaoxiong and Li, Annan},
  booktitle={Proceedings of the 32nd ACM International Conference on Multimedia},
  pages={826--835},
  year={2024}
}
```
