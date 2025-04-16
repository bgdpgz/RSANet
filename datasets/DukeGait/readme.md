# Tutorial for BUAA-Duke-Gait
## Download the BUAA-Duke-Gait dataset
Download the dataset (1x64x64 silhouette shape) from the [link](https://pan.baidu.com/s/1p5b6R_Q3cKqKtgwCv3-i7g?pwd=3yth).
Then you will get BUAA-Duke-Gait formatted as:
```
    DATASET_ROOT/
        0001 (subject)/
            00 (type)/
                cam2_46186_47066 (view)/
                    cam2_46186_47066.pkl (contains all frames)
                ......
            ......
        ......
```
The dataset comes from RealGait, which borrows data from [DukeMTMC-VideoReID](https://github.com/Yu-Wu/DukeMTMC-VideoReID?tab=readme-ov-file).
Please cite the following three papers if this dataset helps your research.
```
@article{zhang2022realgait,
  title={RealGait: Gait recognition for person re-identification},
  author={Zhang, Shaoxiong and Wang, Yunhong and Chai, Tianrui and Li, Annan and Jain, Anil K},
  journal={arXiv preprint arXiv:2201.04806},
  year={2022}
}

@inproceedings{wu2018cvpr_oneshot,
  title = {Exploit the Unknown Gradually: One-Shot Video-Based Person Re-Identification by Stepwise Learning},
  author = {Wu, Yu and Lin, Yutian and Dong, Xuanyi and Yan, Yan and Ouyang, Wanli and Yang, Yi},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2018}
}

@inproceedings{ristani2016MTMC,
  title = {Performance Measures and a Data Set for Multi-Target, Multi-Camera Tracking},
  author = {Ristani, Ergys and Solera, Francesco and Zou, Roger and Cucchiara, Rita and Tomasi, Carlo},
  booktitle = {European Conference on Computer Vision workshop on Benchmarking Multi-Target Tracking},
  year = {2016}
}
```
## License
Please refer to the license file for [DukeMTMC-VideoReID](https://github.com/Yu-Wu/DukeMTMC-VideoReID/blob/master/LICENSES/LICENSE_DukeMTMC-VideoReID.txt) and [DukeMTMC](https://github.com/Yu-Wu/DukeMTMC-VideoReID/blob/master/LICENSES/LICENSE_DukeMTMC.txt).
## Contact
If any questions, pls contact guozhen_peng@buaa.edu.cn, thank you.
