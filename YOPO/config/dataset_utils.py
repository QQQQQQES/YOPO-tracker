import glob
import json
import os

from config.config import cfg


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def resolve_dataset_paths(base_dir):
    paths = _as_list(cfg.get("dataset_paths", []))
    if not paths:
        paths = [cfg["dataset_path"]]
    return [resolve_dataset_path(base_dir, path) for path in paths]


def resolve_dataset_path(base_dir, path_arg):
    path_arg = str(path_arg)
    if os.path.isabs(path_arg):
        return os.path.abspath(path_arg)
    return os.path.abspath(os.path.join(base_dir, "../", path_arg))


def resolve_dataset_repeat_factors(num_paths):
    factors = _as_list(cfg.get("dataset_repeat_factors", []))
    if not factors:
        return [1] * num_paths
    if len(factors) != num_paths:
        raise ValueError(
            f"dataset_repeat_factors length {len(factors)} does not match dataset_paths length {num_paths}"
        )
    return [max(1, int(factor)) for factor in factors]


def resolve_dataset_eval_modes(num_paths):
    modes = _as_list(cfg.get("dataset_eval_modes", []))
    if not modes:
        return ["both"] * num_paths
    if len(modes) != num_paths:
        raise ValueError(
            f"dataset_eval_modes length {len(modes)} does not match dataset_paths length {num_paths}"
        )
    allowed = {"both", "train_only", "valid_only"}
    out = []
    for mode in modes:
        normalized = str(mode).strip().lower()
        if normalized not in allowed:
            raise ValueError(f"dataset_eval_modes entries must be one of {sorted(allowed)}, got {mode}")
        out.append(normalized)
    return out


def resolve_ply_map_entries(path):
    metadata_path = os.path.join(path, "metadata.json")
    if os.path.isfile(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        scene_map = metadata.get("scene_map", [])
        if scene_map:
            entries = []
            for item in sorted(scene_map, key=lambda x: int(x["output_scene_id"])):
                scene_id = int(item["output_scene_id"])
                unity_scene_id = int(item["unity_scene_id"])
                file = os.path.join(path, f"pointcloud-unity-scene-{unity_scene_id}.ply")
                if not os.path.isfile(file):
                    raise FileNotFoundError(file)
                entries.append((scene_id, file))
            return entries

        pointcloud = metadata.get("pointcloud", "")
        if pointcloud:
            file = os.path.join(path, pointcloud)
            if not os.path.isfile(file):
                raise FileNotFoundError(file)
            scene_count = int(metadata.get("num_scenes") or count_scene_dirs(path) or 1)
            return [(scene_id, file) for scene_id in range(scene_count)]

    sorted_files = read_sorted_numeric_pointclouds(path)
    if not sorted_files:
        return []

    scene_count = count_scene_dirs(path)
    if len(sorted_files) == 1 and scene_count > 1:
        return [(scene_id, sorted_files[0]) for scene_id in range(scene_count)]
    return [(idx, file) for idx, file in enumerate(sorted_files)]


def count_scene_dirs(path):
    if not os.path.isdir(path):
        return 0
    return sum(
        1
        for name in os.listdir(path)
        if name.startswith("scene_") and os.path.isdir(os.path.join(path, name))
    )


def read_sorted_numeric_pointclouds(path):
    ply_files = glob.glob(os.path.join(path, "pointcloud-*.ply"))

    def extract_index(filename):
        base = os.path.basename(filename)
        number_part = base.replace("pointcloud-", "").replace(".ply", "")
        return int(number_part)

    ply_files = [
        filename
        for filename in ply_files
        if os.path.basename(filename).replace("pointcloud-", "").replace(".ply", "").isdigit()
    ]
    return sorted(ply_files, key=extract_index)
