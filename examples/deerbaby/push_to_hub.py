import os
import sys

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


def main(repo_id: str):
    dataset = LeRobotDataset(repo_id=repo_id)
    dataset.push_to_hub()


if __name__ == '__main__':
    main(sys.argv[1])
