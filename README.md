# as\_DeepClaw: Deep Learning for Arcade Claw Grasping

as\_DeepClaw is intended to be setup as a robotic grasping platform for deep learning based research and development. The robot is configured to perform similar tasks like the arcade game of claw crane. The goal of the game is for the player to control the crane in the horizontal x-y plane to determine an optimal coordinate to drop the claw in the vertical z axis. While descending, the claw will close while approaching the bottom of a transparent, enclosed cabinet filled with stuffed toys as rewards. If successful, one (or multiple if lucky) reward will be picked up by the closing claw, and deliver the reward to the lucky (or skilled) player by opening the claw above a drop hole. This is a very interesting game that’s been very popular in arcade game studios since the very beginning.

In this project, a robotic system is setup in the same way to perform grasping tasks for deep learning training and validation. In general, such arcade claw grasping task is very similar to the industrial pick-and-place tasks which was the main reason of this setup. The system is currently setup in a similar way as the game but with only one layer of toys at the bottom, and an added rotation angle at the gripper wrist to learn the best angle from the vertical top view for a successful grasp.

Currently, the system has been updated in configuration to the 2nd version. Data collected from the 1st version is used for training using AlexNet to learn the position (x, y) for grasping. Now it runs autonomously to attempt grasping through random angles to learn the rotation by collecting new data with an improved grasping accuracy. A few key index
* V1.0 blind grasp attempts: xxxx in total, with an average success rate of ~25%.
* V1.0 learning result by transfer learning with AlexNet: >90%
* V2.0 guided grasp using learned result from V1.0 in position (x, y) to train grasping objects of different shape for rotation (theta)
* v2.0 a new set of data is to be collected, including toys as well as daily objects, to update the overall learning
* A key limitation of current system is the use of entirely rigid gripper, which could potentially blocked by the objects during grasping and tripper a safety stop of the system
    * One solution is the use of Force-Torque sensor, which is expensive and available in current system
    * Another solution is the use of compliant gripper to overcome such issue, which is currently being developed using Hybrid Robotics technology with potentially much reduced cost and system setup.

Wan, F. & Song, C., 2017, as_DeepClaw: An Arcade Claw Robot for Logical Learning with A Hybrid Neural Network. Github, https://github.com/ancorasir/as_DeepClaw, DOI: 10.5281/zenodo.581803

# Task Decomposition
* Start
    * Preliminary Preparation
    * Safety Check
    * From Idle Position (x\_idle, y\_idle, z\_idle=0.5, theta\_idle, open)
* Operation
    * Guided Grips (~2k/day)
* End
    * Back to Idle Position (x\_idle, y\_idle, z\_idle=0, theta\_idle, open)
    * Learning and Grasping Result Summary

## Start Stage
* System Setup
    * World => Pedestal => Arm => FT Sensor => Gripper
    * World => Desk => Tray => Objects (to be picked)
    * World => Desk => Bin => Objects (to be placed)
    * World => Camera Kinect Albert (right above the Bin, with full view of the Tray)
    * World => Camera Kinect Bernard (right above the Tray, with full view of the Bin)
    * World => Camera LifeCam (auxiliary camera on the side of the system for recording)

Hardware
* Robot System (RS)
    * Arm: UR5 from Universal Robot
    * FT Sensor: FT300 from Robotiq
    * Gripper: Adaptive 3 Finger from Robotiq
    * Camera 1&2: Kinect for X-box One from Microsoft
    * Camera 3: LifeCam from Microsoft
* Learning Computer (LC)
    * OS: Ubuntu Trusty 14.04.5
    * CPU: Intel Core i7 6800K Hex-Core 3.4GHz
    * GPU: Single 12GB NVIDIA TITAN X (Pascal)
    * RAM: 32GB Corsair Dominator Platinum 3000MHz (2 X 16GB)
    * SSD: 1TB Samsung 850 Pro Series

## V2.0: Guide Stage
TBD


## V1.0: Blind Stage
The blind stage starts with Blind Grips and followed by the Blind Learning, aiming at initializing the neural network. Basically, we need to give the robot a chance to explore the world before we benchmark its performance. Specifically, we choose the following strategy to collect some initial test data to better understand the nature of this experiment.

* Start the robot with total blind grasps, i.e. just reach out to any coordinate within the allowed workspace, perform a grasp, and check the results right after.
    * _The advantage would be the simplicity and automation on programming without the involvement of neural network, and_
    * _The disadvantage would be the lack of purpose, which may result in low successful grasp in the beginning, slowing down the learning process._

Based on initial results collected, this process could be improved with the following strategy to boost the learning process.

* Start the robot with human-guided grasps, i.e. send labelled coordinates of potential grasps to the robot and let the robot grasp (this is the part that is interestingly like the claw machine;, possibly making it a joy to play with it.)
    * _The advantage would be the manual speedup of the learning process with purpose and successful grasps, making it a supervised learning problem, and_
    * _The disadvantage would be the uncertainty for an autonomously learned network for robotic grasping with human intervention, would it be a good or bad thing?_

The Blind Grips consist of many Blind Grip Cycles, and each Blind Grip Cycle consists of many Blind Grip Attempts. Note that one can treat the tray as the bin to speed up the data collection. One can also merge the idle and drop position to further simplify the workflow.

* Start from idle position c\_idle
* [Blind] Grip Cycle n = 1 (start of this Learning Cycle)
    * ...
* [Blind] Grip Cycle n (end of last Grip Cycle n-1 = start of this Grip Cycle n)
    * Shot 00
        * [take picture I\_n\_00 without gripper, only the objects in tray]
    * Move 00
        * [move gripper horizontally to c\_n\_0 following a random vector v\_n\_0 = (x\_n\_0, y\_n\_0, **z\_n\_0=0** , st\_n\_0, ct\_n\_0, **open** ) to the **zero** plane]
    * Shot 01
        * [take picture I\_n\_01 with gripper and objects in tray]
  ---------------------------------------------------------------------------
    * Move 1
        * [move gripper to c\_n\_1 following a random vector v\_n\_1 = (x\_n\_1, y\_n\_1, **z\_n\_1=grip** , st\_n\_1, ct\_n\_1, **open** ) to the **grip** plane]
    * Shot 1
        * [take picture I\_n\_1 with gripper and objects in tray]
  ---------------------------------------------------------------------------
    * Pick
        * [close the gripper for a grip attempt]
    * Shot 11
        * [take picture I\_n\_11 of the tray, recording the immediate grip result]
    * Move
        * [move gripper to the drop position c\_drop above the bin]
    * Shot 12
        * [take picture I\_n\_12 of the tray, recording the confirmed results in the tray after grip attempt, the number of objects in tray "-1"]
    * Drop
        * [open the gripper at drop position c\_drop to release object to the bin]
    * Shot 13
        * [5K I\_n\_13 of the bin, recording the confirmed results in the bin after drop, the number of objects in bin "+1"]
* [Blind] Grip Cycle n+1 (end of this Grip Cycle n = start of next Grip Cycle n+1)
    * ...
* [Blind] Grip End (end of this Learning Cycle = start of next Learning Cycle)
    * Move [move gripper back to idle position c\_idle, marking the end of the experiment]

After the Blind Grips, the data collected from Robot System (RS) will be supplied to the Learning Computer (LC) for network training. Data to be input to the network includes

* Camera data
    * I\_n\_00, I\_n\_01
    * all need to be preprocessed with a random crop
    * both rgb-d and rgb camera data could be used for training, taken from two different angles
* Robot data
    * c\_n\_1 - c\_n\_0
    * note that this is vector data representing relative position
* Reward data
    * r\_g\_n [0/1] grasp success marker processed from I\_n\_11 and I\_n\_12

Each grip cycle can be further simplified with a few Integrated Task Flow (ITF) as the following.

* [Blind] Grip Cycle n (end of last Grip Cycle n-1 = start of this Grip Cycle n)
    * ITF-SMS0 (= Shot00 => Move 00 => Shot 01)
    * ITF-MoSo (= Move 1 => Shot 1)
    * ITF-Pick (= Pick => Shot 11 => Move => Shot 12 => Drop => Shot 13)

## End Stage

The End Stage is straight-forward as the following.

* Back to Idle Position (x\_idle, y\_idle, z\_idle=0, theta\_idle, open)
* Learning and Grasping Result Summary

# Data Management

Image Data (I\_n) and Motor Data (v\_n) are the major data to be transmitted and processed. However, there is a potential to include further Motion Data, Grip Data and Sensor Data (TBD)

Efficient data saving (TBD)

Time stamp synchronization (TBD)

The whole learning process consists of a lot (m) of repetitive Learning Cycles

* Each Learning Cycle consists of a lot (n) of repetitive Grip Cycles.
* Each Grip Cycle consists of a series of Integrated Task Flows (ITF)
* Each ITF consists of a series of Actions (Shot, Move, Pick, Drop, etc.)
* Each Action executes/generates certain Data
* Shot Action
  * generates visual Data
* Move Action
  * executes waypoint Data (position and rotation as a vector)
  * generates motion Data (Arm, Gripper, FT Sensor)
* Pick Action
  * executes close grip Data (adaptive grasp mode: Basic or Pinch)
  * generates motion/sensor Data (Gripper, FT Sensor)
* Drop Action
  * executes open grip Data (adaptive grasp mode: Basic or Pinch)
  * generates motion/sensor Data (Gripper, FT Sensor)
