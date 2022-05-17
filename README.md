
## Now, the code can reproduce the mAP result of ppyoloe_s on COCO.

## Introduction
An unofficial implementation of Pytorch version PP-YOLOE,based on  Megvii YOLOX training code.
Many codes references from [PP-YOLOE Official implementation](https://github.com/PaddlePaddle/PaddleDetection) and [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX).
[Report on Arxiv](https://arxiv.org/pdf/2203.16250.pdf)

## Updates
* 【2022/05/17】 Exciting! Now, the code can reproduce the mAP on COCO.
* 【2022/05/09】 Fix some bugs. Add the data augmentation methods in Paddle version
* 【2022/04/15】 Initial commit support training, eval, demo for ppyoloe-s/m/l/x.

## Comming soon
- [  ] PP-YOLOE model deploy.
- [√] More pretrained model.
- [  ] More experiments results.
- [  ] Code to convert PP-YOLOE model from PaddlePaddle to Pytorch.

## Model
|Model                                                | size |mAP<sup>val<br>0.5:0.95 |mAP<sup>test<br>0.5:0.95 | Speed V100<br>(ms) | Params<br>(M) |FLOPs<br>(G)| weights | backbone weights |
| ------                                              |:---: | :---:                  | :---:                   |:---:               |:---:          | :---:      | :----:  | :----:  |
|[PP-YOLOE-s](./exps/ppyoloe/default/ppyoloe_s.py)    |640   | 43.1%                  |Training...              | Training...        |7.93           | 17.36      | [baidu pan](https://pan.baidu.com/s/1cwF05pjxQ2PRWqDNfav1_Q) code:qfld | [baidu pan](https://pan.baidu.com/s/1ZtKExb-ElLCmoAWZFYFAmw) code:mwjy |
|[PP-YOLOE-m](./exps/ppyoloe/default/ppyoloe_m.py)    |640   |Training...             |Training...              | Training...        |23.43          | 49.91      | [baidu pan](https://pan.baidu.com/s/1YT_npdeECflXNKN9JSs7ow) code:xgji | [baidu pan](https://pan.baidu.com/s/1tyinbghS_j5l9LYEZ9Mg_w) code:p4gy |
|[PP-YOLOE-l](./exps/ppyoloe/default/ppyoloe_l.py)    |640   |Training...             |Training...              | Training...        |52.20          | 110.07     | [baidu pan](https://pan.baidu.com/s/1KloomVNYwdnumQg6SRLY-g) code:1v82 | [baidu pan](https://pan.baidu.com/s/1Ntgm1ICSrPeCthtMGAA9Mw) code:6kkb |
|[PP-YOLOE-x](./exps/ppyoloe/default/ppyoloe_x.py)    |640   |Training...             |Training...              | Training...        |98.42          | 206.59     | [baidu pan](https://pan.baidu.com/s/1QqqnaE-uPCImTvkoV2adkA) code:liq3 | [baidu pan](https://pan.baidu.com/s/1f9f5lVgZBua3cNiLXCmong) code:izas |

## Quick Start

Unofficial paper interpretation:[PPYOLOE深度解析](https://zhuanlan.zhihu.com/p/505992733)

### Installation
Step1. Install from source.(No difference from the Megvii YOLOX)
```shell
git clone git@github.com:Nioolek/PPYOLOE_pytorch.git
cd PPYOLOE_pytorch
pip3 install -v -e .  # or  python3 setup.py develop
```

### Demo
Step1. Download a pretrained model from the benchmark table.

Step2. Run demo.py
```shell
python tools/demo.py image -f exps/ppyoloe/default/ppyoloe_l.py -c /path/to/your/ppyoloe_l.pth --path assets/dog.jpg --conf 0.25 --nms 0.45 --tsize 640 --save_result --device [cpu/gpu] --ppyoloe --legacy
```
Because of the difference in preprocess, you must use the args '--ppyoloe' and '--legacy' while inference and eval.

### Train

Step1. Prepare COCO dataset
```shell
cd <PPYOLOE_pytorch_HOME>
ln -s /path/to/your/COCO ./datasets/COCO
```

Step2. Reproduce our results on COCO by specifying -f:
```shell
python -m yolox.tools.train -f exps/ppyoloe/default/ppyoloe_l.py -d 8 -b 64 --fp16 -o [--cache]
```

### Evaluation
```shell
python -m yolox.tools.eval -f  exps/ppyoloe/default/ppyoloe_l.py -c ppyoloe_l.pth -b 64 -d 8 --conf 0.001 --legacy --ppyoloe [--fp16] [--fuse]
```
Because of the difference in preprocess, you must use the args '--ppyoloe' and '--legacy' while inference and eval.

More details can find from the docs of [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX).


### (optional) Convert Paddle model to Pytorch
```shell
python paddle2torch.py -f 0 -i weights/ppyoloe_crn_x_300e_coco.pdparams -o weights/ppyoloe_x.pth
python paddle2torch.py -f 1 -i weights/CSPResNetb_x_pretrained.pdparams -o weights/CSPResNetb_x_pretrained.pth
```
-f 0 means convert pretrained detection model.

-f 1 means convert pretrained backbone model.

The download url of Paddle pretrained model can get from [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/ppyoloe)


## More information
* You are welcome to report bugs.

## Reference
https://github.com/Megvii-BaseDetection/YOLOX
```latex
 @article{yolox2021,
  title={YOLOX: Exceeding YOLO Series in 2021},
  author={Ge, Zheng and Liu, Songtao and Wang, Feng and Li, Zeming and Sun, Jian},
  journal={arXiv preprint arXiv:2107.08430},
  year={2021}
}
```
https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/ppyoloe
