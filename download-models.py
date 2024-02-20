#!/usr/bin/env python

from facefusion.download import conditional_download
from facefusion.filesystem import resolve_relative_path


OUT_PATH="../.assets/models"
MODEL_NAMES = [
    "2dfan4.onnx",
    "arcface_simswap.onnx",
    "arcface_w600k_r50.onnx",
    "blendswap_256.onnx",
    "codeformer.onnx",
    "face_occluder.onnx",
    "face_parser.onnx",
    "gender_age.onnx",
    "GFPGANv1.2.onnx",
    "GFPGANv1.3.onnx",
    "GFPGANv1.4.onnx",
    "GFPGANv1.4.pth",
    "gfpgan_1.2.onnx",
    "gfpgan_1.3.onnx",
    "gfpgan_1.4.onnx",
    "GPEN-BFR-512.onnx",
    "gpen_bfr_256.onnx",
    "gpen_bfr_512.onnx",
    "inswapper_128.onnx",
    "inswapper_128_fp16.onnx",
    "open_nsfw.onnx",
    "RealESRGAN_x2plus.pth",
    "RealESRGAN_x4plus.pth",
    "RealESRNet_x4plus.pth",
    "real_esrgan_x2plus.pth",
    "real_esrgan_x4plus.pth",
    "real_esrnet_x4plus.pth",
    "restoreformer.onnx",
    "restoreformer_plus_plus.onnx",
    "retinaface_10g.onnx",
    "simswap_256.onnx",
    "simswap_512_unofficial.onnx",
    "uniface_256.onnx",
    "wav2lip_gan.onnx",
    "yoloface_8n.onnx",
    "yunet_2023mar.onnx",
]


def main():
    model_urls = [
        f"https://github.com/facefusion/facefusion-assets/releases/download/models/{name}" for name in MODEL_NAMES
    ]
    conditional_download(
        resolve_relative_path(OUT_PATH),
        model_urls
    )


if __name__ == '__main__':
    main()

