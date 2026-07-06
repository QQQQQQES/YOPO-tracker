import os
import torch
import random
import argparse
import numpy as np
from policy.yopo_trainer import YopoTrainer
from config.config import cfg


def configure_random_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", type=int, default=0, help="use pre-trained model?")
    parser.add_argument("--trial", type=int, default=1, help="trial of pre-trained model")
    parser.add_argument("--epoch", type=int, default=50, help="epoch of pre-trained model")
    parser.add_argument("--epochs", type=int, default=50, help="training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="training batch size")
    parser.add_argument("--learning_rate", type=float, default=1.5e-4, help="training learning rate")
    parser.add_argument("--save_interval", type=int, default=5, help="checkpoint save interval in epochs")
    parser.add_argument("--checkpoint_path", type=str, default="", help="explicit checkpoint path")
    parser.add_argument("--dataset_path", type=str, default="", help="override single dataset path")
    parser.add_argument("--dataset_paths", nargs="+", default=None, help="override mixed dataset paths")
    parser.add_argument("--dataset_repeat_factors", nargs="+", type=int, default=None, help="train-only repeat factor per dataset path")
    parser.add_argument("--dataset_eval_modes", nargs="+", default=None, help="per-dataset split mode: both, train_only, or valid_only")
    parser.add_argument("--temporal_split_unit", type=str, default="", choices=["", "sequence", "scene", "unity_scene"], help="override temporal split unit")
    parser.add_argument("--dataloader_num_workers", type=int, default=-1, help="override DataLoader workers; negative keeps config value")
    parser.add_argument("--max_train_steps", type=int, default=0, help="limit train batches per epoch for smoke tests; 0 means no limit")
    parser.add_argument("--max_eval_steps", type=int, default=0, help="limit eval batches per epoch for smoke tests; 0 means no limit")
    parser.add_argument("--train_target_follow_distance", type=float, default=None, help="override tracking label follow distance")
    parser.add_argument("--tracking_score_include_goal_cost", type=int, choices=[0, 1], default=None, help="include positive-grid goal cost in tracking score labels")
    parser.add_argument("--tracking_non_positive_goal_weight", type=float, default=None, help="weak target-progress weight for visible non-positive grids")
    parser.add_argument("--tracking_score_include_non_positive_goal_cost", type=int, choices=[0, 1], default=None, help="include visible non-positive weak goal cost in score labels")
    parser.add_argument("--tracking_non_positive_length_weight", type=float, default=None, help="length-only cost weight for visible non-positive grids")
    parser.add_argument("--tracking_non_positive_length_cap", type=float, default=None, help="max effective terminal length for visible non-positive length-only cost")
    parser.add_argument("--tracking_non_positive_length_xy_only", type=int, choices=[0, 1], default=None, help="use horizontal terminal length only for visible non-positive length cost")
    parser.add_argument("--tracking_score_include_non_positive_length_cost", type=int, choices=[0, 1], default=None, help="include visible non-positive length-only cost in score labels")
    parser.add_argument("--tracking_goal_cost_mode", type=str, default="", choices=["", "progress_cap", "l2_terminal"], help="override tracking goal cost mode")
    parser.add_argument("--tracking_progress_cap", type=float, default=None, help="override positive-grid tracking progress cap")
    parser.add_argument("--tracking_use_lost_forward_cost", type=int, choices=[0, 1], default=None, help="include no-target lost-forward cost in trajectory labels")
    parser.add_argument("--tracking_score_include_lost_forward_cost", type=int, choices=[0, 1], default=None, help="include no-target lost-forward cost in tracking score labels")
    parser.add_argument("--w_lost_forward", type=float, default=None, help="override no-target lost-forward cost weight")
    parser.add_argument("--lambda_lost_traj", type=float, default=None, help="override no-target trajectory loss weight")
    parser.add_argument("--lost_forward_progress_cap", type=float, default=None, help="override no-target forward progress cap")
    parser.add_argument("--w_tracking_goal", type=float, default=None, help="override positive-grid tracking goal cost weight")
    parser.add_argument("--w_target", type=float, default=None, help="override target regression loss weight")
    parser.add_argument("--w_objectness", type=float, default=None, help="override objectness loss weight")
    parser.add_argument("--lambda_negative_obj", type=float, default=None, help="override negative objectness loss weight")
    parser.add_argument("--lambda_hard_negative_obj", type=float, default=None, help="override hard negative objectness loss weight")
    parser.add_argument("--hard_negative_obj_topk", type=int, default=None, help="override hard negative top-k grid count")
    parser.add_argument("--wc", type=float, default=None, help="override collision/safety loss weight")
    parser.add_argument("--d0", type=float, default=None, help="override safety loss obstacle influence distance")
    parser.add_argument("--safety_radius", type=float, default=None, help="override safety loss radius parameter r")
    parser.add_argument("--safety_voxel_size", type=float, default=None, help="override ESDF voxel size")
    return parser


if __name__ == "__main__":
    args = parser().parse_args()
    configure_random_seed(0)    # set random seed

    # save the configuration and other files
    log_dir = os.path.dirname(os.path.abspath(__file__)) + "/saved"
    os.makedirs(log_dir, exist_ok=True)
    checkpoint_path = args.checkpoint_path
    if not checkpoint_path:
        checkpoint_path = log_dir + "/YOPO_{}/epoch{}.pth".format(args.trial, args.epoch) if args.pretrained else ""

    if args.dataset_paths:
        cfg["dataset_paths"] = args.dataset_paths
    elif args.dataset_path:
        cfg["dataset_path"] = args.dataset_path
        cfg["dataset_paths"] = []
    if args.dataset_repeat_factors:
        cfg["dataset_repeat_factors"] = args.dataset_repeat_factors
    if args.dataset_eval_modes:
        cfg["dataset_eval_modes"] = args.dataset_eval_modes
    if args.temporal_split_unit:
        cfg["temporal_split_unit"] = args.temporal_split_unit
    if args.dataloader_num_workers >= 0:
        cfg["dataloader_num_workers"] = args.dataloader_num_workers
    if args.train_target_follow_distance is not None:
        cfg["train_target_follow_distance"] = args.train_target_follow_distance
    if args.tracking_score_include_goal_cost is not None:
        cfg["tracking_score_include_goal_cost"] = bool(args.tracking_score_include_goal_cost)
    if args.tracking_non_positive_goal_weight is not None:
        cfg["tracking_non_positive_goal_weight"] = args.tracking_non_positive_goal_weight
    if args.tracking_score_include_non_positive_goal_cost is not None:
        cfg["tracking_score_include_non_positive_goal_cost"] = bool(args.tracking_score_include_non_positive_goal_cost)
    if args.tracking_non_positive_length_weight is not None:
        cfg["tracking_non_positive_length_weight"] = args.tracking_non_positive_length_weight
    if args.tracking_non_positive_length_cap is not None:
        cfg["tracking_non_positive_length_cap"] = args.tracking_non_positive_length_cap
    if args.tracking_non_positive_length_xy_only is not None:
        cfg["tracking_non_positive_length_xy_only"] = bool(args.tracking_non_positive_length_xy_only)
    if args.tracking_score_include_non_positive_length_cost is not None:
        cfg["tracking_score_include_non_positive_length_cost"] = bool(args.tracking_score_include_non_positive_length_cost)
    if args.tracking_goal_cost_mode:
        cfg["tracking_goal_cost_mode"] = args.tracking_goal_cost_mode
    if args.tracking_progress_cap is not None:
        cfg["tracking_progress_cap"] = args.tracking_progress_cap
    if args.tracking_use_lost_forward_cost is not None:
        cfg["tracking_use_lost_forward_cost"] = bool(args.tracking_use_lost_forward_cost)
    if args.tracking_score_include_lost_forward_cost is not None:
        cfg["tracking_score_include_lost_forward_cost"] = bool(args.tracking_score_include_lost_forward_cost)
    if args.w_lost_forward is not None:
        cfg["w_lost_forward"] = args.w_lost_forward
    if args.lambda_lost_traj is not None:
        cfg["lambda_lost_traj"] = args.lambda_lost_traj
    if args.lost_forward_progress_cap is not None:
        cfg["lost_forward_progress_cap"] = args.lost_forward_progress_cap
    if args.w_tracking_goal is not None:
        cfg["w_tracking_goal"] = args.w_tracking_goal
    if args.w_target is not None:
        cfg["w_target"] = args.w_target
    if args.w_objectness is not None:
        cfg["w_objectness"] = args.w_objectness
    if args.lambda_negative_obj is not None:
        cfg["lambda_negative_obj"] = args.lambda_negative_obj
    if args.lambda_hard_negative_obj is not None:
        cfg["lambda_hard_negative_obj"] = args.lambda_hard_negative_obj
    if args.hard_negative_obj_topk is not None:
        cfg["hard_negative_obj_topk"] = args.hard_negative_obj_topk
    if args.wc is not None:
        cfg["wc"] = args.wc
    if args.d0 is not None:
        cfg["d0"] = args.d0
    if args.safety_radius is not None:
        cfg["r"] = args.safety_radius
    if args.safety_voxel_size is not None:
        cfg["safety_voxel_size"] = args.safety_voxel_size

    trainer = YopoTrainer(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        loss_weight=[1.0, cfg["w_cost"]],
        tensorboard_path=log_dir,
        checkpoint_path=checkpoint_path,
        save_on_exit=True,
        max_train_steps=args.max_train_steps if args.max_train_steps > 0 else None,
        max_eval_steps=args.max_eval_steps if args.max_eval_steps > 0 else None,
    )

    trainer.train(epoch=args.epochs, save_interval=args.save_interval)

    print("Run YOPO Finish!")
