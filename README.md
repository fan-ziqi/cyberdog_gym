# cyberdog_gym

小米铁蛋强化学习，[视频演示](https://www.bilibili.com/video/BV1Eg4y1P7KA)

环境：

* Ubuntu-20.04
* NVIDIA RTX3070-Laptop

## 下载代码

```bash
git clone --recursive https://github.com/fan-ziqi/cyberdog_gym.git
cd cyberdog_gym
```

下载[isaacgym](https://developer.nvidia.com/isaac-gym)到cyberdog_gym中，需要注册NVIDIA开发者账号

代码如有更新，执行此句更新代码

```bash
git pull --recurse-submodules
```

## 显卡依赖

### NVIDIA显卡驱动

[NVIDIA显卡驱动](https://www.nvidia.cn/Download/index.aspx?lang=cn)

查看驱动版本`nvidia-smi`

注意：若使用WSL2，不需要在Linux中装显卡驱动，只需要在Windows中装好显卡驱动就可以。

### CUDA

[CUDA](https://developer.nvidia.com/cuda-toolkit-archive)

查看CUDA版本`nvcc -V`

## 使用Docker环境

使用docker可以快速部署隔离的、虚拟的、完全相同的开发环境， 不会出现“我的电脑能跑，你的电脑跑不了”的情况。

注意：

* 如果使用的是RTX4090显卡，请修改`docker/Dockerfile`文件中的第一句为：

  ```dockerfile
  nvcr.io/nvidia/pytorch:22.12-py3
  ```

* 如果使用的是RTX3070显卡，则无需修改

构建并运行镜像

```bash
cd docker
bash build.sh
bash run.sh 0
```

在镜像内进行初始化

```bash
bash setup.sh
```

### 查看资源使用情况

镜像中内置了nvitop，新建一个窗口，运行`docker attach isaacgym_container`进入容器，运行`nvitop`查看系统资源使用情况

### 报错解决

执行`bash run.sh 0`的时候若出现如下报错：

```
Error response from daemon: failed to create task for container: failed to create shim task: OCI runtime create failed: runc create failed: unable to start container process: error during container init: error running hook #0: error running hook: exit status 1, stdout: , stderr: Auto-detected mode as 'legacy'
nvidia-container-cli: initialization error: load library failed: libnvidia-ml.so.1: cannot open shared object file: no such file or directory: unknown
```

出现此问题大多是因为没有使用root权限运行容器，以下几种方案均可：

* 在bash前加root
* 切换至root用户
* 将当前用户加入root组

若无法找到构建好的isaacgym镜像，则需重新以root权限构建镜像。

## 使用Conda环境

### 安装环境

#### Anaconda

[Anaconda](https://www.anaconda.com/download)

创建新的conda环境

```text
conda create -n your_env_name python=3.8
```

激活环境

```python3
conda activate your_env_name
```

#### Pytorch

[Pytorch](https://pytorch.org/get-started/locally/)

激活上文的环境，按照网站引导进行安装

#### 安装其他依赖

```bash
pip install tensorboard pybullet pygame lxml transformations opencv-python opencv-contrib-python nvitop
```

### 配置环境

以下命令均在根目录下执行

#### 配置isaacgym

```bash
pip install -e isaacgym/python
```

测试：

```bash
cd isaacgym/python/examples
python 1080_balls_of_solitude.py
```

#### 配置rsl_rl

```bash
pip install -e rsl_rl
```

#### 配置legged_gym

```bash
pip install -e legged_gym
```

测试：

```bash
cd legged_gym/legged_gym/script
python train.py
```

#### 报错解决

`AttributeError: module 'numpy' has no attribute 'float'.`

numpy在1.24以后弃用了float类型，将Numpy版本降级到1.23.5

```bash
conda install numpy==1.23.5
```

**推荐：**修改`isaacgym/python/isaacgym/torch_utils.py`第135行为

```cpp
def get_axis_params(value, axis_idx, x_value=0., dtype=np.float64, n_dims=3):
```

## 使用强化学习

进入脚本所在目录

```bash
cd legged_gym/legged_gym/script
```

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

若需要从某一检查点开始训练：

```bash
python train.py --task=cyberdog_rough --headless --resume --load_run=Aug16_20-41-16_rough --checkpoint=1000
```

## 多卡注意事项

如果使用多张显卡进行训练，使用`--rl_device=cuda:1`选择了除cuda0以外的显卡，需要进行remap。本项目已将`rsl_rl/rsl_rl/runners/on_policy_runner.py`中222行`load`函数中的`torch.load`改为：

```cpp
def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path,map_location={'cuda:2': 'cuda:0','cuda:1': 'cuda:0'})
```



