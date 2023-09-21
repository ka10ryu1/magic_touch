#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import time
import argparse

import cv2 as cv
import numpy as np
import onnxruntime

from demo_MagicTouch_onnx import run_inference


def mouse_callback(event, x, y, flags, param):
    global mouse_point
    mouse_point = [x, y]


def main():
    global mouse_point

    parser = argparse.ArgumentParser()

    parser.add_argument("--image", type=str, default='sample.jpg')
    parser.add_argument(
        "--model",
        type=str,
        default='magic_touch.onnx',
    )

    args = parser.parse_args()
    model_path = args.model
    image_path = args.image

    # Load model
    onnx_session = onnxruntime.InferenceSession(
        model_path,
        providers=[
            'CUDAExecutionProvider',
            'CPUExecutionProvider',
        ],
    )

    # Create GUI
    mouse_point = None
    cv.namedWindow('MagicTouch Input')
    cv.setMouseCallback('MagicTouch Input', mouse_callback)

    # Load Image
    image = cv.imread(image_path)

    while True:
        start_time = time.time()

        original_image = copy.deepcopy(image)
        debug_image = copy.deepcopy(image)
        image_height, image_width = image.shape[0], image.shape[1]

        # Inference execution
        result = run_inference(
            onnx_session,
            image,
            mouse_point,
        )

        result = cv.resize(
            result,
            dsize=(image_width, image_height),
        )

        elapsed_time = time.time() - start_time

        # Crop with mask
        black_image = np.zeros(debug_image.shape, dtype=np.uint8)
        mask = np.where(result > 0.5, 1.0, 0.0)
        mask = np.stack((mask, ) * 3, axis=-1).astype('uint8')
        mask_size = np.sum(mask) / 255
        debug_image = np.where(mask, debug_image, black_image)

        # Inference elapsed time
        cv.putText(
            original_image,
            f"Elapsed Time : {elapsed_time * 1000 :.1f} ms / Mask size : {mask_size:.1f}",
            (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1, cv.LINE_AA)

        key = cv.waitKey(1)
        if key == 27:  # ESC
            break
        cv.imshow('MagicTouch Input', original_image)

        # 小さい物体は誤検出として無視する
        if mask_size > 500:
            continue

        # 矩形抽出
        contours, _ = cv.findContours(
            cv.cvtColor(mask, cv.COLOR_RGB2GRAY),
            cv.RETR_EXTERNAL,
            cv.CHAIN_APPROX_NONE
        )
        # 矩形の面積を計算して最大の矩形を抽出
        x, y, w, h = cv.boundingRect(
            max(contours, key=lambda x: cv.contourArea(x))
        )
        # 矩形描画
        cv.rectangle(debug_image, (x, y), (x+w-1, y+h-1), (0, 255, 0), 5)

        cv.imshow('MagicTouch Output', result)
        cv.imshow('MagicTouch Post-Process Image', debug_image)

    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
