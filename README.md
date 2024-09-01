# Robotic Manipulator Simulation with PyBullet

This project simulates a Kinova Gen3 robotic manipulator and J2N6S300 gripper in PyBullet. The simulation focuses on controlling a three-finger gripper to grasp and manipulate deformable objects, such as a soft ball, in a simulated environment. The project includes features like PID control, Exponential Moving Average filtering, and dynamic simulation of soft bodies.

## Features
- **Grasping and Manipulation**: Control a three-finger robotic gripper to grasp and manipulate objects.
- **Soft Body Simulation**: Simulate soft body dynamics and interactions with a rigid robotic arm.
- **PID Control**: Implement PID control for fine-tuning the gripper's movements.
- **Exponential Moving Average Filtering**: Apply EMA filtering to smooth out sensor data.

## Installation
To run this simulation, you'll need to have Python installed along with the following packages:

```bash
pip install pybullet numpy

## Usage
The arm's target position is manually set, and it is recommended that the gripper approaches objects from a height of 0.01 meters above the object to ensure optimal grasping performance.
