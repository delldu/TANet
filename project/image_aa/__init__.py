"""Image/Video Tanet Package."""  # coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2022-2024(18588220928@163.com) All Rights Reserved.
# ***
# ***    File Author: Dell, 2022年 09月 25日 星期日 11:31:05 CST
# ***
# ************************************************************************************/
#

__version__ = "1.0.0"

import os
from tqdm import tqdm
import torch

import todos
from . import tanet

import pdb
from PIL import Image, ImageDraw, ImageFont


def get_model():
    """Create model."""

    model = tanet.TANet()
    device = todos.model.get_device()
    model = model.to(device)
    model.eval()

    print(f"Running on {device} ...")

    # make sure model good for C/C++
    model = torch.jit.script(model)
    # https://github.com/pytorch/pytorch/issues/52286
    torch._C._jit_set_profiling_executor(False)
    # C++ Reference
    # torch::jit::getProfilingMode() = false;                                                                                                             
    # torch::jit::setTensorExprFuserEnabled(false);

    todos.data.mkdir("output")
    if not os.path.exists("output/image_aa.torch"):
        model.save("output/image_aa.torch")

    return model, device



def image_predict(input_files, output_dir):
    # Create directory to store result
    todos.data.mkdir(output_dir)

    # load model
    model, device = get_model()

    # load files
    image_filenames = todos.data.load_files(input_files)

    ttf_path = "/usr/share/fonts/truetype/ubuntu/Ubuntu-C.ttf"
    ttf = ImageFont.truetype(ttf_path, 32)

    # start predict
    output_scores = []
    progress_bar = tqdm(total=len(image_filenames))
    for filename in image_filenames:
        progress_bar.update(1)

        # orig input
        input_tensor = todos.data.load_tensor(filename)

        predict_tensor = todos.model.forward(model, device, input_tensor)
        # if predict_tensor.item() < 6.0:
        #     continue

        output_file = f"{output_dir}/{os.path.basename(filename)}"
        image = Image.open(filename)
        draw = ImageDraw.Draw(image)
        draw.text((10, 10), f"{predict_tensor.item(): .4f}", fill="red", font=ttf)
        image.save(output_file)

        output_scores.append([filename, predict_tensor.item()])
    progress_bar.close()

    for fs in output_scores:
        print(f"{fs[0]} -- {fs[1]:.4f}")

    # images/0001.png -- 6.5177
    # images/0002.png -- 5.6350
    # images/0003.png -- 6.4909
    # images/0004.png -- 4.4555
    # images/0005.png -- 5.1979
    # images/0006.png -- 4.3984

    todos.model.reset_device()
