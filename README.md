# Visual Information-Based Grasping of the Robotic Arm Using the Deep Learning Model

## Project Description

This project aimed to leverage a deep learning model to predict the robotic arm grasp poses from a depth image scene. According to the objects' spatial information, the deep learning model can output the grasp position, grasp angle, and gripper width to finish the grasp work. The GGCNN [<sup>1</sup>](#refer-anchor-1) was implemented in this project since it is a simple and widely used model..

The goal of this project is not to create novel models or algorithms but to familiarize yourself with the entire workflow of network training, environment construction, and grasping based on the predicted results in the simulated environment.

<figure>
    <img src="./images/grasp.gif">
</figure>

<div style="text-align:justify;font-size:12px;"><b>Fig 1.</b> The left gif shows the grasping process of the panda robotic arm in the Pybullet simulation environment, and the right gif shows the rendered depth map. The position marked by the red line in the depth map is the grasping position of the gripper.</div>

## Installation

1. Install [anaconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)

2. Create and activate a conda virtual environment with python3.7.

   ```bash
   sudo apt update
   conda create -n env_name python=3.7
   conda activate env_name
   ```

3. Download this repository.

   ```bash
   git clone https://github.com/BoceHu/vision_based_grasp.git
   cd vision_based_grasp
   ```

4. Install [PyTorch](https://pytorch.org/) (Please choose the suitable version according to the CUDA version and the system.)

   ```bash
   conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
   ```

5. Install [OpenCV](https://opencv.org/)

   ```bash
   conda install -c conda-forge opencv
   ```

6. Install other requirement packages

   ```bash
   conda install -r requirements.txt
   ```



## Reference

<div id="refer-anchor-1"></div>

- [1] [Closing the Loop for Robotic Grasping: A Real-time, Generative Grasp Synthesis Approach](https://arxiv.org/abs/1804.05172)
