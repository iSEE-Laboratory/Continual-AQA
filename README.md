# Continual-AQA
Official PyTorch implementation of paper "Continual Action Assessment via Task-Consistent Score-Discriminative Feature Distribution Modeling" (TCSVT 2024). 
[[ArXiv]](https://arxiv.org/abs/2309.17105) [[IEEE Trans]](https://ieeexplore.ieee.org/document/10518028)

Please feel free to contact us if you have any question.

**Contact:** yuanmingli527@gmail.com / liym266@mail2.sysu.edu.cn

## News
- [2024.04.27] This work is accepted by **TCSVT-2024** :)
- [2024.05.02] The pre-processed **data**, **checkpoints** and **logs** of the experiments on AQA-7 dataset are available :)
- [2024.05.24] The **code** and running **script** for experiments on the AQA-7 dataset are available :)
- [2024.06.01] The pre-processed **data**, **code**, and running **script** for experiments on the MTL-AQA dataset are available :)


## Requirements
- Python 3.8+
- Pytorch
- torchvision
- numpy
- timm
- scipy

Our experiments can be conducted on 4 Nvidia RTX 1080Ti GPUs.

## Usage
### Data Preparation
- Click [here](https://drive.google.com/drive/folders/1Llnwbn2CO-ktQU1oxkhO46Qj-n-Z7BeM?usp=sharing) to download the preprocessed AQA-7 dataset.
- Click [here](https://drive.google.com/file/d/12a1abdgoSDXNW7iTqOvLYFnQIvIFkgzu/view?usp=share_link) to download the preprocessed MTL-AQA dataset.

### Checkpoints \& logs
Click [here](https://drive.google.com/drive/folders/1QVT0U_HLNdHYZi4GEGZsIAVXz2idtKGa?usp=sharing) to download the checkpoints and logs of our experiments.

### Test with checkpoints
Coming soon.

### Train from scratch
Use the following script to train our model on the AQA-7 dataset.
```
python run_net.py --exp_name your_exp_name \
  --gpu 0,1,2,3 --seed 0 --approach g_e_graph \
  --lambda_distill 9 --lambda_diff 0.7 \
  --replay --replay_method group_replay --memory_size 30 \
  --diff_loss \
  --aug_approach aug-diff --aug_mode fs_aug --num_helpers 7 --aug_scale 0.3\
  --save_graph --g_e_graph --fix_graph_mode no_fix \
  --save_ckpt\
  --optim_mode new_optim --lr_decay --num_epochs 200 --batch-size 16 --alpha 0.8 
```

## Citation
Please cite it if you find this work useful.
```
@ARTICLE{10518028,
  author={Li, Yuan-Ming and Zeng, Ling-An and Meng, Jing-Ke and Zheng, Wei-Shi},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={Continual Action Assessment via Task-Consistent Score-Discriminative Feature Distribution Modeling}, 
  year={2024},
  doi={10.1109/TCSVT.2024.3396692}}
```

## Acknowledgement
The authors thank Jia-Hui Pan for providing the code and pre-proceesed data used in her works:
```
@inproceedings{pan2019action,
  title={Action assessment by joint relation graphs},
  author={Pan, Jia-Hui and Gao, Jibin and Zheng, Wei-Shi},
  booktitle={Proceedings of the IEEE/CVF international conference on computer vision},
  pages={6331--6340},
  year={2019}
}

@article{pan2021adaptive,
  title={Adaptive action assessment},
  author={Pan, Jia-Hui and Gao, Jibin and Zheng, Wei-Shi},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  volume={44},
  number={12},
  pages={8779--8795},
  year={2021},
  publisher={IEEE}
}
```
