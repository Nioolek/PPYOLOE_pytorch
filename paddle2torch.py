import argparse
import paddle
import torch
from collections import OrderedDict


def make_parser():
    parser = argparse.ArgumentParser("convert paddle model to pytorch")
    parser.add_argument("-f", "--flag", type=int, default=0, help="flag type.flag '0' means convert detection model, flag '1'means convert backbone model")

    parser.add_argument("-i", "--input_path", type=str, default=None, help="input path")
    parser.add_argument("-o", "--output_path", type=str, default=None, help="output path")
    return parser


def replace_key(key):
    if key.startswith('yolo_head'):
        key = key.replace('yolo_head.', 'head.')
    if key.startswith('backbone.stem'):   # 'backbone.stem.conv1.bn._mean',  --->  'backbone.stem.0.bn.running_mean'
        split_list = key.split('.')
        split_list[2] = str(int(split_list[2].replace('conv', ''))-1)
        out_key = '.'.join(split_list)
        if '.bn.' in out_key:
            out_key = out_key.replace('_mean', 'running_mean').replace('_variance', 'running_var')
            return out_key
        return out_key
    # paddle转torch
    if '.bn.' in key:
        out_key = key.replace('_mean', 'running_mean').replace('_variance', 'running_var')
        return out_key
    return key


def replace_key_backbone(key):
    if key.startswith('stem'):   # 'backbone.stem.conv1.bn._mean',  --->  'backbone.stem.0.bn.running_mean'
        split_list = key.split('.')
        split_list[1] = str(int(split_list[1].replace('conv', ''))-1)
        out_key = '.'.join(split_list)
        if '.bn.' in out_key:
            out_key = out_key.replace('_mean', 'running_mean').replace('_variance', 'running_var')
            return out_key
        return out_key
    # paddle转torch
    if '.bn.' in key:
        out_key = key.replace('_mean', 'running_mean').replace('_variance', 'running_var')
        return out_key
    return key

if __name__ == '__main__':
    args = make_parser().parse_args()
    if args.flag == 0:
        # convert ppyoloe model 转换目标检测模型
        assert args.input_path.endswith('pdparams')
        assert args.output_path.endswith('pth')
        state_dict_paddle = paddle.load(args.input_path)
        state_dict_torch = OrderedDict()
        state_dict_torch['model'] = OrderedDict()
        save_name = args.output_path
        keys_paddle = list(state_dict_paddle.keys())
        for paddle_key in keys_paddle:
            torch_key = replace_key(paddle_key)
            w_paddle = state_dict_paddle[paddle_key]
            w_torch = torch.tensor(w_paddle.numpy(), dtype=torch.float)
            state_dict_torch['model'][torch_key] = w_torch
        torch.save(state_dict_torch, save_name)
    else:
        assert args.input_path.endswith('pdparams')
        assert args.output_path.endswith('pth')
        # convert backbone 转换backbone
        state_dict_paddle = paddle.load(args.input_path)
        state_dict_torch = OrderedDict()
        keys_paddle = list(state_dict_paddle.keys())
        for paddle_key in keys_paddle:
            torch_key = replace_key(paddle_key)
            w_paddle = state_dict_paddle[paddle_key]
            w_torch = torch.tensor(w_paddle.numpy(), dtype=torch.float)
            state_dict_torch[paddle_key] = w_torch

        torch.save(state_dict_torch, args.output_path)

