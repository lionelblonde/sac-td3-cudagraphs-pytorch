# Off-Policy RL Baselines with CUDA Graphs 

An implementation of Soft Actor-Critic (SAC)
and Twin Delayed Deep Deterministic Policy Gradient (TD3) algorithms
using [CUDA Graphs in PyTorch](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/).

## Prerequisites

* __Python Version__: Python 3.8 or higher
* __GPU__: NVIDIA GPU with CUDA support
* __CUDA Version__: CUDA 11.7 or higher (for GPU support)
* __NVIDIA Drivers__: Compatible with your CUDA version
* Libraries like `libglew-dev` and `patchelf` might be required on your system for MuJoCo and
`dm_control` to run. Refer to Gymnasium/DeepMind Control Suite/MuJoCo projects to see what are
the requirements of the current versions of their tools.

## Installation

You can set up the project using either **Conda** or **Docker**.

### Using Conda


#### Step 1: Clone the Repository

```bash
git clone https://github.com/lionelblonde/sac-td3-cudagraphs-pytorch.git
cd sac-td3-cudagraphs-pytorch
```

#### Step 2: Install Conda or Mamba

Download and install
[Anaconda](https://docs.conda.io/projects/conda/en/stable/user-guide/getting-started.html)
or [Miniconda](https://docs.anaconda.com/miniconda/)
or [Miniforge](https://github.com/conda-forge/miniforge)
or [Mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html)
or [Micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html).
Avoid Mambaforge as it is deprecated as of July 2024.

#### Step 3: Create a New Conda Environment

```bash
conda create -n sac_td3_cuda python=3.8
conda activate sac_td3_cuda
```

#### Step 4: Install PyTorch with CUDA Support

Install PyTorch in accordance to your OS. For Linux with CUDA 12.1, run:
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

#### Step 5: Install Other Dependencies

```bash
pip install -r requirements.txt
```

#### Step 6: Set Up MuJoCo

1. Download MuJoCo from [the GitHub releases](https://github.com/google-deepmind/mujoco/releases).
2. Install MuJoCo for your system. For Linux and MuJoCo v3.2.3 (latest at time of release):
```bash
mkdir ~/.mujoco
tar -xvzf mujoco-3.2.3-linux-x86_64.tar.gz -C ~/.mujoco/
```
3. Set environment variables (if you picked MuJoCo v3.2.3):
```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco323/bin
export MUJOCO_PY_MUJOCO_PATH=$HOME/.mujoco/mujoco323
export MUJOCO_GL=egl
```

### Using Docker

#### Step 0: Prerequisites

* __Docker__ installed on your machine
* __NVIDIA Container Toolkit__ for GPU support[Installation guide](
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

#### Step 1: Clone the Repository

```bash
git clone https://github.com/lionelblonde/sac-td3-cudagraphs-pytorch.git
cd sac-td3-cudagraphs-pytorch
```

#### Step 2: Build the Docker Image

```bash
docker build -t sac_td3_cuda .
```

#### Step 3: Run the Docker Container with GPU Support

```bash
docker run --gpus all --rm -it sac_td3_cuda
```

## Usage

### Using Conda

#### Train

```bash
python -O main.py train --cfg="tasks/defaults/sac.yml" --env_id="Hopper-v4" --seed=0
```

Note: `-O` turns the value of the constant `__debug__` to `False`, which turns off `beartype`
decorator, hence reducing the overhead time (although constant). Running the command without the
`-O` option activates `beartype` real-time type-checking functionalities.

#### Evaluate

```bash
python -O main.py evaluate --cfg="tasks/defaults/sac.yml" --env_id="Hopper-v4" --seed=0 --load_ckpt="wandb_run_path"
```
This function evaluates a model by retrieving its best performing checkpoint stored on Weighs & Biases servers.
What should be given as value to the `--load_ckpt` argument is the "Run Path" on the Weighs & Biases overview page
of the run to evaluate. The parameters of the best model are downloaded in a temporary directory.

### Using Docker

#### Train

```bash
docker run --gpus all --rm -it sac_td3_cuda python -O main.py train --cfg="tasks/defaults/sac.yml" --env_id="Hopper-v4" --seed=0
```

Note: `train` is the default command in the Dockerfile, so this previous command could also be run with the shorter:
```bash
docker run --gpus all --rm -it sac_td3_cuda  --env_id="Hopper-v4" --seed=0
```

#### Evaluate

```bash
docker run --gpus all --rm -it sac_td3_cuda python -O main.py evaluate --cfg="tasks/defaults/sac.yml" --env_id="Hopper-v4" --seed=0 --load_ckpt="wandb_run_path"
```

## Advanced: Usage via the Spawner

The `spawner.py` enables the creating (and launch) of an array of experiments on a Slurm cluster,
or locally in a new `tmux` session, with one experiment running per window in the session.

Here is how it can be used:

```bash
python spawner.py --cfg="tasks/defaults/sac.yml" --conda_env="ptfarama" --env_bundle="low" --deployment="slurm" --num_seeds=3 --caliber="long" --deploy_now
```

To create the scripts _without_ deploying them immediately, use `--nodeploy_now` instead of `--deploy_now`.
This logic applies to all the boolean options since we are using [google/python-fire](https://github.com/google/python-fire).

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.
