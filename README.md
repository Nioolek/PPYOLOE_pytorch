
## warning
!!!Now, the code cannot reproduce the mAp result on coco.Please don`t use it.

## Introduction
An unofficial implementation of Pytorch version PP-YOLOE,based on  Megvii YOLOX training code.
Many codes references from [PP-YOLOE Official implementation](https://github.com/PaddlePaddle/PaddleDetection) and [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX).
[Report on Arxiv](https://arxiv.org/pdf/2203.16250.pdf)

## Updates
* 【2022/05/09】 Fix some bugs. Add the data augmentation methods in Paddle version
* 【2022/04/15】 Initial commit support training, eval, demo for ppyoloe-s/m/l/x.

## Comming soon
- [ ] PP-YOLOE model deploy.
- [ ] More pretrained model.
- [ ] More experiments results.
- [ ] Code to convert PP-YOLOE model from PaddlePaddle to Pytorch.

## Model
|Model |size |mAP<sup>val<br>0.5:0.95 |mAP<sup>test<br>0.5:0.95 | Speed V100<br>(ms) | Params<br>(M) |FLOPs<br>(G)| weights |
| ------        |:---: | :---:    | :---:       |:---:     |:---:  | :---: | :----: |
|[PP-YOLOE-s](./exps/ppyoloe/default/ppyoloe_s.py)    |640  |Training... |Training...      |Training...      |7.93 | 17.36 | Coming soon... |
|[PP-YOLOE-m](./exps/ppyoloe/default/ppyoloe_m.py)    |640  |Training... |Training...      |Training...     |23.43 |49.91| Coming soon... |
|[PP-YOLOE-l](./exps/ppyoloe/default/ppyoloe_l.py)    |640  |Training... |Training...      |Training...     |52.20| 110.07 | Coming soon... |
|[PP-YOLOE-x](./exps/ppyoloe/default/ppyoloe_x.py)   |640   |Training... |Training...  | Training...    |98.42 |206.59 | Coming soon... |

## Quick Start

Now, there are some differences between this code and the official version. I will explain it in detail in zhihu later.

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
```
python -m yolox.tools.eval -f  exps/ppyoloe/default/ppyoloe_l.py -c ppyoloe_l.pth -b 64 -d 8 --conf 0.001 --legacy --ppyoloe [--fp16] [--fuse]
```
Because of the difference in preprocess, you must use the args '--ppyoloe' and '--legacy' while inference and eval.

More details can find from the docs of [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX).

## More information
* Because of lack of GPUs,I am now training the model.I hope the code can reproduce the PP-YOLOE results on COCO.
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
