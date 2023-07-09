# cyberdog_gym

小米铁蛋强化学习。

[视频演示](https://www.bilibili.com/video/BV1Eg4y1P7KA)

## 环境

* Ubuntu-20.04
* NVIDIA RTX3070-Laptop

## 依赖

### NVIDIA显卡驱动

[NVIDIA显卡驱动](https://www.nvidia.cn/Download/index.aspx?lang=cn)

查看驱动版本`nvidia-smi`

### CUDA

[CUDA](https://developer.nvidia.com/cuda-toolkit-archive)

查看CUDA版本`nvcc -V`

### Conda

[Anaconda](https://www.anaconda.com/download)

创建新的conda环境

```text
conda create -n your_env_name python=3.8
```

激活环境

```python3
conda activate your_env_name
```

### Pytorch

[Pytorch](https://pytorch.org/get-started/locally/)

激活上文的环境，按照网站引导进行安装

## 部署

```bash
git clone --recursive https://github.com/fan-ziqi/cyberdog_gym.git
cd cyberdog_gym
```

### 配置isaacgym

下载[isaacgym](https://developer.nvidia.com/isaac-gym)到cyberdog_gym中，需要注册NVIDIA开发者账号

```bash
pip install -e isaacgym/python
```

测试：

```bash
python isaacgym/python/examples/1080_balls_of_solitude.py
```

### 配置rsl_rl

```bash
pip install -e rsl_rl
```

### 配置legged_gym

```bash
pip install -e legged_gym
```

测试：

```bash
python legged_gym/script/train.py
```

## 报错解决

1. `AttributeError: module 'numpy' has no attribute 'float'.`

   numpy在1.24以后弃用了float类型，将Numpy版本降级到1.23.5

   ```bash
   conda install numpy==1.23.5
   ```

2. `ModuleNotFoundError: No module named 'tensorboard'`

   安装tensorboard

   ```bash
   conda install tensorboard
   ```

## 使用

训练

```bash
python train.py --task=cyberdog_rough --headless --run_name=upstair 
```

回放

```bash
python play.py --task=cyberdog_rough --resume --run_name=upstair
```

其他可选参数：

*  检查点`--checkpoint=2500`

## 更新代码

```bash
git pull
git submodule update --remote --recursive
```

若第二行报错请执行

```bash
git pull --recurse-submodules
```

## 注意事项

如果使用多张显卡进行训练，使用了`--rl_device=cuda:1`选择了除cuda0以外的显卡，需要进行remap。将`rsl_rl/rsl_rl/runners/on_policy_runner.py`中222行`load`函数中的`torch.load`改为如下形式：

```cpp
def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path,map_location={'cuda:2': 'cuda:0','cuda:1': 'cuda:0'})
```



