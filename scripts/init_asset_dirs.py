"""Create empty asset directories for the repository's default inference layout."""

import argparse
from pathlib import Path


def path_component(value):
    if not value or value in (".", "..") or any(c in value for c in '/\\<>:"|?*'):
        raise argparse.ArgumentTypeError("Use a single directory name without path separators.")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--town", type=path_component, help="Recording town name, e.g. Town05")
    parser.add_argument("--weather", choices=("sunny", "night"))
    parser.add_argument("--model", type=path_component, help="Exact checkpoint directory name")
    args = parser.parse_args()
    if bool(args.town) != bool(args.weather):
        parser.error("--town and --weather must be supplied together")

    root = Path(__file__).resolve().parents[1]
    paths = [
        "datasets/training_v12_mapillary",
        "pix2pixHD/checkpoints",
        "pix2pixHD/results/mp4/vision_pilot",
        "pix2pixHD/results/mp4/NEW",
        "output/vp_input_1024",
        "output/calibrated",
        "output/gt",
    ]
    if args.model:
        paths.append(f"pix2pixHD/checkpoints/{args.model}")
    if args.town:
        name = f"{args.town}_{args.weather}_inst"
        paths.extend(f"datasets/recorded_{name}/{kind}" for kind in ("rgb", "semantic"))
        channels = ["label", "edge", "depth", "normal", "label_rich"]
        channels.append("chroma" if args.weather == "sunny" else "light")
        paths.extend(f"datasets/training_v12_mapillary/test_{name}_gt_{c}" for c in channels)
        paths.append(f"pix2pixHD/results/mp4/vision_pilot/{args.town.lower()}")
    for relative in paths:
        destination = root / relative
        destination.mkdir(parents=True, exist_ok=True)
        print(destination)


if __name__ == "__main__":
    main()
