# RSANet: Relative-Sequence Quality Assessment Network for Gait Recognition in the Wild
[Paper](https://doi.org/10.1016/j.patcog.2024.111219) has been accepted in Pattern Recognition. This is the code for it.
* You can download Duke-Gait in [link].
# Operating Environments
## Pytorch Environment
* Pytorch=1.11.0
* Python=3.8
# CheckPoints
* The checkpoint for Gait3D [link](https://pan.baidu.com/s/1nfdcGeCWNOpq2FH62jKSSw?pwd=hxa4).
* The checkpoint for GREW [link](https://pan.baidu.com/s/1-eejS7eiI-NX44KqOWQ8HQ?pwd=3ftb).
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
@article{peng2025rsanet,
  title={RSANet: Relative-sequence quality assessment network for gait recognition in the wild},
  author={Peng, Guozhen and Wang, Yunhong and Zhang, Shaoxiong and Li, Rui and Zhao, Yuwei and Li, Annan},
  journal={Pattern Recognition},
  volume={161},
  pages={111219},
  year={2025},
  publisher={Elsevier}
}
```
