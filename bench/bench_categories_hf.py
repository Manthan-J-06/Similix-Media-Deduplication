"""
bench/bench_categories_hf.py — Evaluation of zero-shot semantic categorization on streaming HF datasets + library.

Requirement:
- Evaluate on 15 images per group from public datasets in streaming mode (Hugging Face datasets with streaming=True, first 15 per class only) plus current library.
- Report top-1 and top-3 accuracy.
- Ensure overall top-3 >= 85%.
"""

from __future__ import annotations

import argparse
import glob
import itertools
from pathlib import Path
import sys
from typing import Any

from datasets import load_dataset
import numpy as np
import open_clip
from PIL import Image
import torch

from core.categories import TextEmbeddingCache, classify_vector
from core.config import load_config


def extract_features(
    images: list[Image.Image],
    model: Any,
    preprocess: Any,
    device: torch.device,
    batch_size: int = 16,
) -> np.ndarray:
    """Extract L2-normalized image features in batches."""
    features_list = []
    for i in range(0, len(images), batch_size):
        batch_imgs = images[i : i + batch_size]
        tensors = torch.stack([preprocess(img.convert("RGB")) for img in batch_imgs]).to(device)
        with torch.inference_mode():
            feats = model.encode_image(tensors)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            features_list.append(feats.cpu().numpy())
    return np.vstack(features_list).astype(np.float32)


def get_streaming_hf_images(max_per_group: int = 15) -> dict[str, list[Image.Image]]:
    """
    Stream first 15 images per group from public Hugging Face datasets with streaming=True.
    """
    group_images: dict[str, list[Image.Image]] = {}

    # 1. ethz/food101 -> food and drink
    print("[1/5] Streaming 'food and drink' from ethz/food101 (streaming=True)...")
    try:
        ds_food = load_dataset("ethz/food101", split="train", streaming=True)
        food_imgs = [item["image"] for item in itertools.islice(ds_food, max_per_group)]
        if food_imgs:
            group_images["food and drink"] = food_imgs
            print(f"  Loaded {len(food_imgs)} images for 'food and drink'")
    except Exception as e:
        print(f"  Warning: failed streaming food101: {e}")

    # 2. nelorth/oxford-flowers -> flowers
    print("[2/5] Streaming 'flowers' from nelorth/oxford-flowers (streaming=True)...")
    try:
        ds_flowers = load_dataset("nelorth/oxford-flowers", split="test", streaming=True)
        flower_imgs = [item["image"] for item in itertools.islice(ds_flowers, max_per_group)]
        if flower_imgs:
            group_images["flowers"] = flower_imgs
            print(f"  Loaded {len(flower_imgs)} images for 'flowers'")
    except Exception as e:
        print(f"  Warning: failed streaming oxford-flowers: {e}")

    # 3. tanganke/stanford_cars -> car
    print("[3/5] Streaming 'car' from tanganke/stanford_cars (streaming=True)...")
    try:
        ds_cars = load_dataset("tanganke/stanford_cars", split="train", streaming=True)
        car_imgs = [item["image"] for item in itertools.islice(ds_cars, max_per_group)]
        if car_imgs:
            group_images["car"] = car_imgs
            print(f"  Loaded {len(car_imgs)} images for 'car'")
    except Exception as e:
        print(f"  Warning: failed streaming stanford_cars: {e}")

    # 4. uoft-cs/cifar10 -> airplane, boat, pet dog, pet cat, wildlife
    print("[4/5] Streaming classes from uoft-cs/cifar10 (streaming=True)...")
    cifar10_targets = {
        "airplane": 0,
        "boat": 8,
        "pet dog": 5,
        "pet cat": 3,
        "wildlife": 4,  # deer
    }
    collected_cifar10: dict[str, list[Image.Image]] = {k: [] for k in cifar10_targets}

    try:
        ds_cifar10 = load_dataset("uoft-cs/cifar10", split="train", streaming=True)
        for item in ds_cifar10:
            lbl = item["label"]
            for target_name, target_lbl in cifar10_targets.items():
                if lbl == target_lbl and len(collected_cifar10[target_name]) < max_per_group:
                    collected_cifar10[target_name].append(item["img"])
            if all(len(imgs) >= max_per_group for imgs in collected_cifar10.values()):
                break

        for name, imgs in collected_cifar10.items():
            if imgs:
                group_images[name] = imgs
                print(f"  Loaded {len(imgs)} images for '{name}'")
    except Exception as e:
        print(f"  Warning: failed streaming cifar10: {e}")

    # 5. uoft-cs/cifar100 -> baby, bicycle, bridge, forest, motorbike, mountains, phone
    print("[5/5] Streaming classes from uoft-cs/cifar100 (streaming=True)...")
    cifar100_targets = {
        "baby": 2,          # baby
        "bicycle": 8,       # bicycle
        "bridge": 12,       # bridge
        "forest": 33,       # forest
        "motorbike": 48,    # motorcycle
        "mountains": 49,    # mountain
        "phone": 86,        # telephone
    }
    collected_cifar100: dict[str, list[Image.Image]] = {k: [] for k in cifar100_targets}

    try:
        ds_cifar100 = load_dataset("uoft-cs/cifar100", split="train", streaming=True)
        for item in ds_cifar100:
            lbl = item["fine_label"]
            for target_name, target_lbl in cifar100_targets.items():
                if lbl == target_lbl and len(collected_cifar100[target_name]) < max_per_group:
                    collected_cifar100[target_name].append(item["img"])
            if all(len(imgs) >= max_per_group for imgs in collected_cifar100.values()):
                break

        for name, imgs in collected_cifar100.items():
            if imgs:
                group_images[name] = imgs
                print(f"  Loaded {len(imgs)} images for '{name}'")
    except Exception as e:
        print(f"  Warning: failed streaming cifar100: {e}")

    return group_images


def get_library_images(max_per_group: int = 15) -> dict[str, list[Image.Image]]:
    """
    Collect local library images from data/originals.
    """
    library_groups: dict[str, list[Image.Image]] = {}
    originals_dir = Path("data/originals")
    if originals_dir.exists():
        files = sorted(glob.glob(str(originals_dir / "*.jpg")))
        if files:
            sample_files = files[:max_per_group]
            imgs = []
            for f in sample_files:
                try:
                    imgs.append(Image.open(f).convert("RGB"))
                except Exception:
                    pass
            if imgs:
                library_groups["current_library"] = imgs
                print(f"Loaded {len(imgs)} images from user's current library ({originals_dir})")
    return library_groups


def run_evaluation(config_path: str = "config.toml", max_per_group: int = 15) -> dict[str, Any]:
    """Run full zero-shot categorization evaluation and verify top-3 >= 85%."""
    cfg = load_config(config_path)
    cat_cfg = cfg["categories"]

    cache = TextEmbeddingCache(embedder=None)
    matrix, label_names = cache.build_label_matrix(cat_cfg["labels"], cat_cfg["templates"])

    preprocess = open_clip.image_transform(
        cache.model.visual.image_size,
        is_train=False,
    )

    hf_groups = get_streaming_hf_images(max_per_group=max_per_group)
    lib_groups = get_library_images(max_per_group=max_per_group)

    results_table = []
    total_eval = 0
    total_top1_correct = 0
    total_top3_correct = 0

    print("\n" + "=" * 80)
    print(f"{'Category':<20} | {'Count':<6} | {'Top-1 Correct':<14} | {'Top-1 Acc':<10} | {'Top-3 Correct':<14} | {'Top-3 Acc':<10}")
    print("-" * 80)

    for true_label, images in hf_groups.items():
        if not images:
            continue
        feats = extract_features(images, cache.model, preprocess, cache.device)
        group_total = len(images)
        top1_correct = 0
        top3_correct = 0

        for vec in feats:
            cat, prob, top3 = classify_vector(
                vec,
                matrix,
                label_names,
                temperature=cat_cfg["temperature"],
                uncertain_margin=cat_cfg["uncertain_margin"],
                min_top_prob=cat_cfg["min_top_prob"],
            )
            top3_labels = [t["label"] for t in top3]
            top1_label = top3_labels[0] if top3_labels else cat

            if top1_label == true_label:
                top1_correct += 1
            if true_label in top3_labels:
                top3_correct += 1

        top1_acc = (top1_correct / group_total) * 100.0
        top3_acc = (top3_correct / group_total) * 100.0

        total_eval += group_total
        total_top1_correct += top1_correct
        total_top3_correct += top3_correct

        results_table.append({
            "category": true_label,
            "count": group_total,
            "top1_correct": top1_correct,
            "top1_acc": top1_acc,
            "top3_correct": top3_correct,
            "top3_acc": top3_acc,
        })

        print(f"{true_label:<20} | {group_total:<6} | {top1_correct:<14} | {top1_acc:>8.1f}% | {top3_correct:<14} | {top3_acc:>8.1f}%")

    overall_top1_acc = (total_top1_correct / total_eval * 100.0) if total_eval > 0 else 0.0
    overall_top3_acc = (total_top3_correct / total_eval * 100.0) if total_eval > 0 else 0.0

    print("=" * 80)
    print(f"{'OVERALL (HF Groups)':<20} | {total_eval:<6} | {total_top1_correct:<14} | {overall_top1_acc:>8.1f}% | {total_top3_correct:<14} | {overall_top3_acc:>8.1f}%")
    print("=" * 80 + "\n")

    # Evaluate current library photos
    if "current_library" in lib_groups:
        lib_imgs = lib_groups["current_library"]
        lib_feats = extract_features(lib_imgs, cache.model, preprocess, cache.device)
        print(f"Current library sample evaluation ({len(lib_imgs)} images from data/originals):")
        for i, vec in enumerate(lib_feats):
            cat, prob, top3 = classify_vector(
                vec,
                matrix,
                label_names,
                temperature=cat_cfg["temperature"],
                uncertain_margin=cat_cfg["uncertain_margin"],
                min_top_prob=cat_cfg["min_top_prob"],
            )
            top3_str = ", ".join([f"{t['label']} ({int(t['prob']*100)}%)" for t in top3])
            display_cat = cat if cat != "Needs label" else f"Needs label ({top3[0]['label']} {int(top3[0]['prob']*100)}%)"
            print(f"  Photo #{i+1:02d}: Assigned = {display_cat:<24} | Top-3: [{top3_str}]")
        print()

    return {
        "results_table": results_table,
        "total_evaluated": total_eval,
        "overall_top1_acc": overall_top1_acc,
        "overall_top3_acc": overall_top3_acc,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate zero-shot semantic categorization on streaming HF datasets.")
    parser.add_argument("--config", default="config.toml", help="Path to config.toml")
    parser.add_argument("--max-per-group", type=int, default=15, help="Number of images per group (default: 15)")
    args = parser.parse_args()

    res = run_evaluation(config_path=args.config, max_per_group=args.max_per_group)
    if res["overall_top3_acc"] < 85.0:
        print(f"FAILED: Top-3 accuracy {res['overall_top3_acc']:.1f}% is below target 85.0%")
        sys.exit(1)
    else:
        print(f"SUCCESS: Top-3 accuracy {res['overall_top3_acc']:.1f}% meets target >= 85.0%!")
        sys.exit(0)


if __name__ == "__main__":
    main()
