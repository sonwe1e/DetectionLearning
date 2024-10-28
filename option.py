import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_wandb", type=bool, default=True)
    parser.add_argument("--project", type=str, default="Test")

    # dataset
    parser.add_argument(
        "--data_path",
        type=str,
        default="/media/hdd/sonwe1e/Detection/Data/Line/train/bbox_data.json",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default="/media/hdd/sonwe1e/Detection/Data/Line/train/images",
    )
    parser.add_argument("-is", "--image_size", type=int, default=512)
    parser.add_argument("--aug_m", type=int, default=2)
    parser.add_argument("--num_classes", type=int, default=2)

    # training setups
    parser.add_argument("-wd", "--weight_decay", type=float, default=5e-2)
    parser.add_argument("-lr", "--learning_rate", type=float, default=4e-4)
    parser.add_argument("-bs", "--batch_size", type=int, default=32)
    parser.add_argument("-e", "--epochs", type=int, default=500)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--rpn_sigma", type=int, default=1)
    parser.add_argument("--roi_sigma", type=int, default=1)

    # experiment
    parser.add_argument("-d", "--devices", type=int, default=0)
    parser.add_argument("-en", "--exp_name", type=str, default="baselinev1")
    parser.add_argument("--precision", type=str, default="bf16-mixed")
    parser.add_argument("--val_check", type=float, default=1.0)
    parser.add_argument("--log_step", type=int, default=20)
    parser.add_argument("--gradient_clip_val", type=int, default=1e6)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)

    return parser.parse_args("")


def get_option():
    opt = parse_args()
    return opt
