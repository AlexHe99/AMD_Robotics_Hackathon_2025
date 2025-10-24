# Train Lerobot Models on MI300x
This guide walks you through setting up the LeRobot training environment on a DigitalOcean (DO) instance equipped with AMD MI300x GPUs and ROCm.

## Prerequisites
- Access to DO instance AMD Mi300x GPU
- Verify ROCm and GPU availability:
  ``` bash
  rocm-smi
  ```
  Example output:
  ``` bash
  ============================================= ROCm System Management Interface =============================================
  ======================================================= Concise Info =======================================================
  Device  Node  IDs              Temp        Power     Partitions          SCLK     MCLK     Fan  Perf  PwrCap  VRAM%  GPU%
                (DID,     GUID)  (Junction)  (Socket)  (Mem, Compute, ID)
  ============================================================================================================================
  0       1     0x74b5,   21947  67.0°C      737.0W    NPS1, SPX, 0        1280Mhz  1100Mhz  0%   auto  750.0W  49%    100%
  ============================================================================================================================
  =================================================== End of ROCm SMI Log ====================================================
  ```
- Record ~50 episodes of your task (e.g., pick-and-place with different cube positions)
- Upload the dataset to the Hugging Face Hub or store it locally on the DO instance

## Environment Setup
- #### Start Docker Container

    -   **Option 1 (Recommended)**: use the pre-built docker image which includes all the necessary dependencies for training ACT and SmolVLA models. `--volume` is used to set the shared folder between host and container. Datasets and trained models can be transfered through the folder.
        ``` bash
        docker run \
        --device /dev/dri \
        --device /dev/kfd \
        --network host \
        --ipc host \
        --group-add video \
        --cap-add SYS_PTRACE \
        --security-opt seccomp=unconfined \
        --workdir /lerobot-0.3.3 \
        --volume /path/on/host:/path/in/container \
        --privileged \
        -it -d \
        --name lerobot xshan1/pytorch:rocm6.4.4_ubuntu24.04_py3.12_pytorch_release_2.7.1_lerobot_0.3.3
        /bin/bash
        ```

    -  **Option 2**: build environment from official ROCm Docker image. Here are the steps to prepare the environment.
        1. Pull official Docker image for PyTorch 2.7.1 with ROCm backend support. **Note:** The reason to choose Pytorch 2.7.1 is that Lerobot has only been verified on Pytorch 2.7.x. 
            ``` bash
            docker pull rocm/pytorch:rocm6.4.4_ubuntu24.04_py3.12_pytorch_release_2.7.1
            ```
        2. Start the container
        ``` bash
        docker run \
        --device /dev/dri \
        --device /dev/kfd \
        --network host \
        --ipc host \
        --group-add video \
        --cap-add SYS_PTRACE \
        --security-opt seccomp=unconfined \
        --volume /path/on/host:/path/in/container \
        --privileged \
        -it -d \
        --name lerobot rocm/pytorch:rocm6.4.4_ubuntu24.04_py3.12_pytorch_release_2.7.1
        /bin/bash
        ```
        3. Install FFmpeg 7.1.1
        ``` bash
        apt add-apt-repository ppa:ubuntuhandbook1/ffmpeg7 # install PPA which contains ffmpeg 7.1.1
        apt update
        apt install ffmpeg
        ffmpeg -version # verify version
        ```
        4. Download Lerobot v0.3.3
        ``` bash
        wget https://github.com/huggingface/lerobot/releases/download/v0.3.3/lerobot-0.3.3.tar.gz
        tar zxvf lerobot-0.3.3.tar.gz -C /
        ```
        5. Install Lerobot in edit mode
        ``` bash
        cd /lerobot-0.3.3
        pip install -e ".[smolvla]" # install both base dependencies and extra dependencies for smolvla and ACT
        ```
        6. Intall and Configure Wandb (optional)
        ``` bash
        pip install wandb
        wandb login # create a wandb account through https://wandb.ai/signup and login wandb with your token
        ```
- #### Train models
1. Use the lerobot-train CLI
    ``` bash
    lerobot-train \
      --dataset.repo_id=${HF_USER}/${DATASET_NAME} \ # The dataset in Huggingface
      --batch_size=128 \
      --steps=10000 \
      --output_dir=outputs/train/<type>_<dataset>_<tag> \ # eg. act_pickplace_3cube_10ksteps
      --job_name=<type>_<dataset>_<tag> \ eg. act_pickplace_3cube_10ksteps
      --policy.device=cuda \
      --policy.type=act \ # change to smolvla or other models
      --wandb.enable=true # disable it if it is not needed
   ```
   Notes:
   - Replace `<type>` with act, smolvla, etc.
   - Replace `<dataset>` with your task name (e.g., pickplace)
   - Replace `<tag>` with a version or descriptor (e.g., 3cube_10ksteps)
   - If using a local dataset, add `--dataset.root=/path/to/dataset`.
   - Adjust `--batch_size` and `--steps` based on your hardware and dataset.
3. Monitoring & Output
    - Checkpoints and logs saved in `--output_dir`
    - Training progress visualized in your wandb dashboard
