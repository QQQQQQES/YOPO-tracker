from torch.utils.data import DataLoader

from policy.yopov2_dataset import YOPOv2TrackerDataset


if __name__ == "__main__":
    dataset = YOPOv2TrackerDataset(mode="train", max_samples=16)
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    print("rgbd:", batch["rgbd"].shape, batch["rgbd"].dtype, float(batch["rgbd"].min()), float(batch["rgbd"].max()))
    print("state:", batch["state"].shape, float(batch["state"].min()), float(batch["state"].max()))
    print("objectness:", batch["objectness"].shape, "positives=", float(batch["objectness"].sum()))
    print("objectness_mask:", batch["objectness_mask"].shape, "used=", float(batch["objectness_mask"].sum()))
    print("target:", batch["target"].shape, "valid=", float(batch["target_valid"].sum()))
    print("target_body:", batch["target_body"].shape)
    print("first seq:", batch["seq_dir"][0])
