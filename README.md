# lerobot-ros2
lerobot 切换到 ros2 

## 规划
- sim（集成 Issac sim）
- benchmark
- lerobot
    - model 训练
        - 使用已有 dataset
    - model 推理
        - 使用设备数据
        - 使用已有 dataset
    - 设备数据抓取
    - 爪子能够挪到小车上？

## 步骤
- [X] lerobot 功能总结
- [X] lerobot 依赖分析
- [ ] lerobot 模块与机理分析
- [ ] lerobot-ros2 规划（考虑泛用性）

# lerobot 
## 功能
1. 下载 dataset 以及预览 - hold
2. 下载训练好的模型 - hold
3. 训练模型
    - 使用已有的 dataset
4. 推理
    - 使用已有的 dataset
    - 使用设备数据 - focus

## 依赖
```python
# python 依赖
from functools import wraps
from huggingface_hub import HfApi
from lerobot.calibrate import CalibrateConfig, calibrate
from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot import available_cameras, available_motors, available_robots
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.tdmpc.modeling_tdmpc import TDMPCPolicy
from lerobot.policies.vqbet.modeling_vqbet import VQBeTPolicy
from lerobot.record import DatasetRecordConfig, RecordConfig, record
from lerobot.replay import DatasetReplayConfig, ReplayConfig, replay
from lerobot.teleoperate import TeleoperateConfig, teleoperate
from lerobot.utils.import_utils import is_package_available
from pathlib import Path
from pprint import pprint
from serial import SerialException
import gymnasium as gym
import gym_pusht  # noqa: F401
import imageio
import importlib
import lerobot
    importlib.import_module(package_name)
import numpy
import os
import platform
import pytest
import torch
import traceback

# python 依赖，但是优先级不高
from tests.fixtures.constants import DUMMY_REPO_ID
from tests.mocks.mock_robot import MockRobotConfig
from tests.mocks.mock_teleop import MockTeleopConfig
from tests.utils import DEVICE
from tests.utils import require_env
```



# troubleShooting:
1. 下载模型时报错 network connection timeout
   解决方法：设置代理镜像：export HF_ENDPOINT=https://hf-mirror.com
            解决不了还可参考：https://github.com/huggingface/transformers/issues/17611

2. 运行 data visualizer 报错：
    ValueError: No valid stream found in input file. Is -1 of the desired media type?
    解决方法：降级 ffmpeg：conda install -c conda-forge ffmpeg=6.1.1 -y
            可能也有其他解决方案，可参考：https://github.com/huggingface/lerobot/issues/970
