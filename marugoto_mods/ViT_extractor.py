#!/usr/bin/env python3

# ViT feature extraction code based on marugoto.extract.imagenet

import argparse
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract ViT features from slide."
    )
    parser.add_argument(
        "slide_tile_paths",
        metavar="SLIDE_TILE_DIR",
        type=Path,
        nargs="+",
        help="A directory with tiles from a slide.",
    )
    parser.add_argument(
        "-o", "--outdir", type=Path, required=True, help="Path to save the features to."
    )
    parser.add_argument(
        "--augmented-repetitions",
        type=int,
        default=0,
        help="Also save augmented feature vectors.",
    )
    args = parser.parse_args()
    print(f"{args=}")

import torchvision.models.vision_transformer as vit
import torch
from marugoto.extract.extract import extract_features_

__all__ = ["extract_ViT_features"]

def extract_ViT_features_(slide_tile_paths, **kwargs):
    """Extracts features from slide tiles.

    Args:
        slide_tile_paths:  A list of paths containing the slide tiles, one
            per slide.
        outdir:  Path to save the features to.
        augmented_repetitions:  How many additional iterations over the
            dataset with augmentation should be performed.  0 means that
            only one, non-augmentation iteration will be done.
    """
    model = vit.vit_b_16(weights=vit.ViT_B_16_Weights.IMAGENET1K_V1,progress=False)
    model.heads['head'] = torch.nn.Identity()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.eval().to(device)

    return extract_features_(
        slide_tile_paths=slide_tile_paths,
        model=model,
        model_name="ViT-imagenet",
        **kwargs,
    )


if __name__ == "__main__":
    extract_ViT_features_(**vars(args))
