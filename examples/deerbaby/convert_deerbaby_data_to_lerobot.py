"""
Script to convert Deerbaby hdf5 data to the LeRobot dataset v2.0 format.

Example usage: uv run examples/deerbaby/convert_deerbaby_data_to_lerobot.py --raw-dir /path/to/raw/data --repo-id <org>/<dataset-name>
--raw-dir input; --repo-id output
"""

import dataclasses
from pathlib import Path
import shutil
from typing import Literal

import h5py
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from merge_datasets import merge_datasets
import numpy as np
import torch
import tqdm
import tyro


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()
CAMERAS = [
    "camera_front",
    "camera_wrist",
    # "camera_high",
    # "camera_low",
]


def create_empty_dataset(
    repo_id: str,
    robot_type: str,
    mode: str = "image", # "video" | "image"
    *,
    has_velocity: bool = False,
    has_effort: bool = False,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> LeRobotDataset:
    motors = [
        "waist",
        "shoulder",
        "elbow",
        "forearm_roll",
        "wrist_angle",
        "wrist_rotate",
        "gripper",
    ]

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        },
        "action": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        },
    }

    if has_velocity:
        features["observation.velocity"] = {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        }

    if has_effort:
        features["observation.effort"] = {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        }

    for cam in CAMERAS:
        features[f"observation.images.{cam}"] = {
            "dtype": mode,
            "shape": (3, 480, 640),
            "names": [
                "channels",
                "height",
                "width",
            ],
        }

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=50,
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )


def get_cameras(hdf5_files: list[Path]) -> list[str]:
    with h5py.File(hdf5_files[0], "r") as ep:
        # ignore depth channel, not currently handled
        return [key for key in ep["/observations/images"].keys() if "depth" not in key]  # noqa: SIM118


def has_velocity(hdf5_files: list[Path]) -> bool:
    with h5py.File(hdf5_files[0], "r") as ep:
        return "/observations/qvel" in ep


def has_effort(hdf5_files: list[Path]) -> bool:
    with h5py.File(hdf5_files[0], "r") as ep:
        return "/observations/effort" in ep


def load_raw_images_per_camera(ep: h5py.File, cameras: list[str]) -> dict[str, np.ndarray]:
    imgs_per_cam = {}
    for camera in cameras:
        uncompressed = ep[f"/observations/images/{camera}"].ndim == 4

        if uncompressed:
            # load all images in RAM
            imgs_array = ep[f"/observations/images/{camera}"][:]
        else:
            import cv2

            # load one compressed image after the other in RAM and uncompress
            imgs_array = []
            for data in ep[f"/observations/images/{camera}"]:
                imgs_array.append(cv2.imdecode(data, 1))
            imgs_array = np.array(imgs_array)

        imgs_per_cam[camera] = imgs_array
    return imgs_per_cam


def load_raw_episode_data(
    ep_path: Path,
) -> tuple[dict[str, np.ndarray], torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None, str]:
    with h5py.File(ep_path, "r") as ep:
        state = torch.from_numpy(ep["/observations/qpos"][:])
        action = torch.from_numpy(ep["/action"][:])

        velocity = None
        if "/observations/qvel" in ep:
            velocity = torch.from_numpy(ep["/observations/qvel"][:])

        effort = None
        if "/observations/effort" in ep:
            effort = torch.from_numpy(ep["/observations/effort"][:])

        imgs_per_cam = load_raw_images_per_camera(ep, CAMERAS)

        prompt = ep.attrs.get("prompt", "")

    return imgs_per_cam, state, action, velocity, effort, prompt


def populate_dataset(
    dataset: LeRobotDataset,
    hdf5_files: list[Path],
    task: str,
    episodes: list[int] | None = None,
) -> LeRobotDataset:
    if episodes is None:
        episodes = range(len(hdf5_files))

    for ep_idx in tqdm.tqdm(episodes):
        ep_path = hdf5_files[ep_idx]

        imgs_per_cam, state, action, velocity, effort, prompt = load_raw_episode_data(ep_path)
        num_frames = state.shape[0]

        for i in range(num_frames):
            frame = {
                "observation.state": state[i],
                "action": action[i],
                "task": prompt or task,
            }

            for camera, img_array in imgs_per_cam.items():
                frame[f"observation.images.{camera}"] = img_array[i]

            if velocity is not None:
                frame["observation.velocity"] = velocity[i]
            if effort is not None:
                frame["observation.effort"] = effort[i]

            dataset.add_frame(frame)

        dataset.save_episode()

    return dataset


def port_deerbaby(
    repo_id: str,
    hdf5_files: list[Path],
    episodes: list[int],
    task: str = "test",
    mode: Literal["video", "image"] = "image",
):
    dataset = create_empty_dataset(
        repo_id,
        robot_type="deerbaby",
        mode=mode,
        has_effort=has_effort(hdf5_files),
        has_velocity=has_velocity(hdf5_files),
        dataset_config=DEFAULT_DATASET_CONFIG,
    )

    dataset = populate_dataset(
        dataset,
        hdf5_files,
        task=task,
        episodes=episodes,
    )
    return dataset


def main(
    raw_dir: Path,
    repo_id: str,
    batch_size: int = 25,
    task: str = "test",
    *,
    ep_range: list[int] | None = None,
    push_to_hub: bool = False,
):
    if (HF_LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(HF_LEROBOT_HOME / repo_id)

    if not raw_dir.exists():
        raise ValueError("input dir does not exist")

    hdf5_files = sorted(raw_dir.glob("episode_*.hdf5"), key=lambda x: int(x.stem.split('_')[1]))

    req_idx = set(range(ep_range[0], ep_range[1]) if ep_range else range(len(hdf5_files)))
    all_idx = set([int(f.stem.split('_')[1]) for f in hdf5_files])
    episodes = sorted(list(req_idx & all_idx))
    batch_count = (len(episodes) + batch_size - 1)// batch_size # round up

    if batch_count == 1:
        dataset = port_deerbaby(repo_id, hdf5_files, episodes, task=task)
        if push_to_hub: dataset.push_to_hub()
        return

    for i in range(batch_count):
        print(f"Start batch {i}/{batch_count}")
        start = i * batch_size
        end = min((i + 1) * batch_size, len(episodes))
        port_deerbaby(
            repo_id=f'tmp/dataset_{i}',
            hdf5_files=hdf5_files,
            episodes=episodes[start:end],
            task=task,
        )
        print(f"Finish batch {i}/{batch_count}")

    merge_datasets(
        source_folders=[f"{HF_LEROBOT_HOME}/tmp/dataset_{i}" for i in range(batch_count)],
        output_folder=f"{HF_LEROBOT_HOME}/{repo_id}",
        max_dim=7,
    )

    for i in range(batch_count):
        shutil.rmtree(HF_LEROBOT_HOME / f"tmp/dataset_{i}")

    if push_to_hub:
        LeRobotDataset(repo_id=repo_id).push_to_hub()


if __name__ == "__main__":
    tyro.cli(main)
