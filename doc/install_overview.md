# Installing AIDE

## Migration from AIDE v1
If you have [AIDE v1](https://github.com/microsoft/aerial_wildlife_detection/tree/v1) already running and want to upgrade its contents to AIDE v2, see [here](upgrade_from_v1.md).


## New installation

### Windows

_Note:_ these are preliminary instructions and still subject to extensive testing. They currently apply to Windows 11; Windows 10 will follow.


#### With the Installer

__Prerequisites:__ You need Windows 10 or greater and support for [Windows Subsystem for Linux](https://docs.microsoft.com/en-gb/windows/wsl/install-manual) to install and use AIDE on a Windows machine.

1. Install the [Windows Subsystem for Linux (WSL)](https://docs.microsoft.com/en-gb/windows/wsl/install-manual). Select "Ubuntu 20.04.4 LTS" as an image in the Microsoft store.
2. If not running, open a new WSL shell: click "Start", type "wsl", select "wsl - Run command". All the rest below must be issued in this new command window.
3. Install [Conda](https://conda.io/) (important: download the Linux binaries); follow any instructions during the install process:
```bash
cd ${HOME}
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh   # follow instructions prompted

source .bashrc
```
4. Create Python environment and clone AIDE repository:
```bash
conda create -y -n aide python=3.8
conda activate aide

git clone https://github.com/microsoft/aerial_wildlife_detection.git && cd aerial_wildlife_detection
```
5. Launch installer and follow the instructions:
```bash
./install/install_debian.sh
```

#### With Docker

Coming soon.



### Ubuntu / Debian

#### With the Installer

1. Install a Python environment manager, such as [Conda](https://conda.io/) (recommended and used below) or [Virtualenv](https://virtualenv.pypa.io).
2. Create a new Python environment for AIDE:
```bash
    conda create -y -n aide python=3.8
    conda activate aide
```
3. Clone the AIDE repository:
```bash
git clone https://github.com/microsoft/aerial_wildlife_detection.git && cd aerial_wildlife_detection/
```
4. Launch installer and follow the instructions:
```bash
    ./install/install_debian.sh
```

__Tip:__ the installer also supports more advanced configurations; you can check it out by calling `./install/install_debian.sh --help`.



#### With Docker

If you wish to install AIDE in a self-contained environment instead of the host operating system, you can do so with Docker:

1. Download and install [Docker](https://docs.docker.com/engine/install) as well as [Docker Compose](https://docs.docker.com/compose/install)
2. If you want to use a GPU, you have to install the NVIDIA container toolkit:
```bash
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
    sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
    sudo systemctl restart docker
```
3. Clone the AIDE repository: `git clone https://github.com/microsoft/aerial_wildlife_detection.git && cd aerial_wildlife_detection/`
4. **Important:** modify the `docker/settings.ini` file and replace the default super user credentials (section `[Project]`) with new values. Make sure to review and update the other default settings as well, if needed.
5. Install:
    ```bash
        cd docker
        sudo docker compose build
        cd ..
    ```
    _Note:_ for older versions, you might have to issue `sudo docker-compose build` (with a hyphen) instead.
6. Launch:
    * With Docker:
    ```bash
        sudo docker/docker_run_cpu.sh     # for machines without a GPU
        sudo docker/docker_run_gpu.sh     # for AIWorker instances with a CUDA-enabled GPU (strongly recommended for model training)
    ```
    * With Docker Compose (note that Docker Compose currently does not provide support for GPUs):
    ```bash
        cd docker
        sudo docker-compose up
    ```

#### Manual installation

See [here](install_manual.md) for instructions on configuring an instance of AIDE.

After that, see [here](launch_aide.md) for instructions on launching an instance of AIDE.



### Microsoft Windows

Instructions coming soon.