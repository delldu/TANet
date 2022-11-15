# coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2020-2022(18588220928@163.com), All Rights Reserved.
# ***
# ***    File Author: Dell, 2020年 12月 28日 星期一 14:29:37 CST
# ***
# ************************************************************************************/
#

import pdb
import os
import time
from tqdm import tqdm

import torch
import todos
import image_tanet

SO_B, SO_C, SO_H, SO_W = 1, 3, 224, 224
# NotImplementedError: The following operators are not implemented: ['aten::copy_']


def compile():
    model, device = image_tanet.get_tvm_model()

    todos.data.mkdir("output")
    if not os.path.exists("output/image_tanet.so"):
        input = torch.randn(SO_B, SO_C, SO_H, SO_W)
        todos.tvmod.compile(model, device, input, "output/image_tanet.so")
    todos.model.reset_device()


def predict(input_files):
    # load model
    device = todos.model.get_device()
    tvm_model = todos.tvmod.load("output/image_tanet.so", str(device))

    # load files
    image_filenames = todos.data.load_files(input_files)

    output_scores = []
    # start predict
    mean_time = 0
    progress_bar = tqdm(total=len(image_filenames))
    for filename in image_filenames:
        progress_bar.update(1)

        # orig input
        input_tensor = todos.data.load_tensor(filename)
        # pytorch recommand clone.detach instead of torch.Tensor(input_tensor)
        orig_tensor = input_tensor.clone().detach()

        B, C, H, W = input_tensor.shape
        input_tensor = todos.data.resize_tensor(input_tensor, SO_H, SO_W)

        start_time = time.time()
        predict_tensor = todos.tvmod.forward(tvm_model, input_tensor)
        torch.cuda.synchronize()
        mean_time += time.time() - start_time

        output_scores.append([filename, predict_tensor.item()])

    mean_time = mean_time / len(image_filenames)

    for fs in output_scores:
        print(f"{fs[0]} -- {fs[1]:.4f}")

    print(f"Mean spend {mean_time:0.4f} seconds")
    os.system("nvidia-smi | grep python")

    todos.model.reset_device()


if __name__ == "__main__":
    compile()
    predict("images/*.png")
