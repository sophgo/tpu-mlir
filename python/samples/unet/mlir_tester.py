import os
import torch
from torch.autograd import Variable
import torchvision.transforms as standard_transforms
import numpy as np
import pyruntime_bm
from tools.model_runner import mlir_inference

from PIL import Image
import argparse
from time import time
import cv2
from tqdm import tqdm
import pandas as pd

parser = argparse.ArgumentParser(description="Model inference")
parser.add_argument("--img_dir", type=str, default="", help="the path of images")
parser.add_argument("--model", type=str, default="", help="the model path")
parser.add_argument("--out_dir", type=str, default="", help="the path of result to save")
parser.add_argument("--height", type=int, default=0, help="the path of result to save")
parser.add_argument("--width", type=int, default=0, help="the path of result to save")
args = parser.parse_args()


# ref.: https://www.kaggle.com/stainsby/fast-tested-rle
def rle_encode(img):
    """
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    """
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


class Tester:

    def __init__(self) -> None:
        """engine initialization"""
        self.newW, self.newH = args.width, args.height
        self.oldW, self.oldH = 1918, 1280

    def pre_process(self, img_pth):
        """image preprocessing"""
        img = Image.open(img_pth)
        img = img.crop((130, 290, 1780, 1100))  # 1650 * 810 = 2 * 1
        img = img.resize((self.newW, self.newH), resample=Image.NEAREST)
        img = np.asarray(img).transpose(2, 0, 1) / 255.0
        img = (np.expand_dims(img, axis=0)).astype(np.float32)
        img = {"input": img}
        return img

    def post_process(self, out, out_pth):
        name = list(out.keys())[-1]

        out_ = np.array(out[name])
        out_ = np.where(out_[0, 0] > out_[0, 1], 0, 255)
        out_ = np.repeat(np.expand_dims(out_, 0), 3, axis=0).transpose(1, 2, 0)

        img = Image.fromarray(np.uint8(out_)).resize((1780 - 130, 1100 - 290),
                                                     resample=Image.BICUBIC)
        gray = cv2.cvtColor(np.asarray(img), cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, 0)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        areas = [cv2.contourArea(c) for c in contours]
        max_index = np.argmax(areas)
        cnt = contours[max_index]
        img = np.zeros_like(img)
        cv2.drawContours(img, [cnt], 0, (255, 255, 255), cv2.FILLED)

        black_full = np.zeros((1280, 1918, 3))
        black_full[290:290 + img.shape[0], 130:130 + img.shape[1], :] = img
        img = black_full
        cv2.imwrite(out_pth, img)
        return np.array(img)

    def test_all(self):
        """test all images and save results"""
        time_cost = 0.0
        df = pd.DataFrame(columns=["img", "rle_mask"])
        for p in tqdm(os.listdir(args.img_dir)):
            t1 = time()
            img_pth = os.path.join(args.img_dir, p)
            out_pth = os.path.join(args.out_dir, p)
            img_input = self.pre_process(img_pth)
            out = mlir_inference(img_input, args.model)
            img_output = self.post_process(out, out_pth)
            rle = rle_encode(img_output[:, :, 0])
            df = df.append({"img": p, "rle_mask": rle}, ignore_index=True)
            t2 = time()
            time_cost += t2 - t1
        df.sort_values(by="img", inplace=True)
        df.to_csv(
            "test_masks.csv",
            index=False,
        )
        print("time cost {}".format(time_cost))


tester = Tester()
tester.test_all()
