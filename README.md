# as_DeepClaw: Deep Learning for Arcade Claw Grasping

as_DeepClaw is intended to be setup as a robotic grasping platform for deep learning based research and development. The robot is configured to perform similar tasks like the arcade game of claw crane. The goal of the game is for the player to control the crane in the horizontal x-y plane to determine an optimal coordinate to drop the claw in the vertical z axis. While descending, the claw will close while approaching the bottom of a transparent, enclosed cabinet filled with stuffed toys as rewards. If successful, one (or multiple if lucky) reward will be picked up by the closing claw, and deliver the reward to the lucky (or skilled) player by opening the claw above a drop hole. This is a very interesting game that’s been very popular in arcade game studios since the very beginning. In this project, a robotic system is setup in the same way to perform grasping tasks for deep learning training and validation. 

Wan, F. & Song, C., 2017, as_DeepClaw: An Arcade Claw Robot for Logical Learning with A Hybrid Neural Network. Github, https://github.com/ancorasir/as_DeepClaw, DOI: 10.5281/zenodo.581803

# Task Decomposition
* Start
    * Preliminary Preparation
    * Safety Check
    * From Idle Position (x\_idle, y\_idle, z\_idle=0, theta\_idle, open)
* Operation
    * Blind Grips (~5k in 1 week)
    * Blind Learning with all Blind Grips attempted in 1 week
    * Daily Grips (~1k in 1 day)
    * Daily Learning with all Daily Grips attempted in 1 day
    * Repeat Daily Grips Daily Learning for 8 weeks
* End
    * Back to Idle Position (x\_idle, y\_idle, z\_idle=0, theta\_idle, open)
    * Learning and Grasping Result Summary

## Start Stage
* System Setup
    * World =&gt; Pedestal =&gt; Arm =&gt; FT Sensor =&gt; Gripper
    * World =&gt; Desk =&gt; Tray =&gt; Objects (to be picked)
    * World =&gt; Desk =&gt; Bin =&gt; Objects (to be placed)
    * World =&gt; Camera Kinect (main rgd-d camera for training, focusing on the tray)
    * World =&gt; Camera LifeCam (auxiliary rgd camera for training, focusing on the tray)
    * World =&gt; Camera Canon (record the whole experiment, focusing on the whole robot operation)

Coordinate Setup

|   |   | Notes | X | Y | Z | Theta |
| --- | --- | --- | --- | --- | --- | --- |
| World | Measured |  |   |   |   |   |
| Pedestal Foot A | Measured |   |   |   |   |   |
| Pedestal Foot B | Measured |   |   |   |   |   |
| Pedestal Center | Calculated |   |   |   |   |   |
| Arm Base Center | Calculated |   |   |   |   |   |
| Desk Foot A | Measured |   |   |   |   |   |
| Desk Foot B | Measured |   |   |   |   |   |
| Desk Center | Calculated |   |   |   |   |   |
| Tray Foot A | Measured |   |   |   |   |   |
| Tray Foot B | Measured |   |   |   |   |   |
| Bin Center | Calculated |   |   |   |   |   |
| Bin Foot A | Measured |   |   |   |   |   |
| Bin Foot B | Measured |   |   |   |   |   |
| Bin Center | Calculated |   |   |   |   |   |
| Camera Kinect | Measured |   |   |   |   |   |
| Camera LifeCam | Measured |   |   |   |   |   |
| Camera Canon | Measured |   |   |   |   |   |
| Zero Plane | Measured |   |   |   |   |   |
| Hover Plane | Measured |   |   |   |   |   |
| Grip Plane | Measured |   |   |   |   |   |
| Idle Position | Measured |  |   |   |   |   |
| Drop Position | Measured |  |   |   |   |   |

Hardware
* Robot System (RS)
    * Arm: UR5 from Universal Robot
    * FT Sensor: FT300 from Robotiq
    * Gripper: Adaptive 3 Finger from Robotiq
    * Camera 1: Kinect for X-box One from Microsoft
    * Camera 2: LifeCam from Microsoft
    * Camera 3: Canon EOS M3
* Learning Computer (LC)
    * OS: Ubuntu Trusty 14.04.5
    * CPU: Intel Core i7 6800K Hex-Core 3.4GHz
    * GPU: Single 12GB NVIDIA TITAN X (Pascal)
    * RAM: 32GB Corsair Dominator Platinum 3000MHz (2 X 16GB)
    * SSD: 1TB Samsung 850 Pro Series

## Blind Stage
The blind stage starts with Blind Grips and followed by the Blind Learning, aiming at initializing the neural network. Basically, we need to give the robot a chance to explore the world before we benchmark its performance. Specifically, we choose the following strategy to collect some initial test data to better understand the nature of this experiment.

* Start the robot with total blind grasps, i.e. just reach out to any coordinate within the allowed workspace, perform a grasp, and check the results right after.
    * _The advantage would be the simplicity and automation on programming without the involvement of neural network, and_
    * _The disadvantage would be the lack of purpose, which may result in low successful grasp in the beginning, slowing down the learning process._

Based on initial results collected, this process could be improved with the following strategy to boost the learning process.

* Start the robot with human-guided grasps, i.e. send labelled coordinates of potential grasps to the robot and let the robot grasp (this is the part that is interestingly like &quot;claw machine&quot;, possibly making it a joy to &quot;play&quot; with it.)
    * _The advantage would be the &quot;manual&quot; speedup of the learning process with purpose and successful grasps, making it a supervised learning problem, and_
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
  - ---------------------------------------------------------------------------
    * Move 1
        * [move gripper to c\_n\_1 following a random vector v\_n\_1 = (x\_n\_1, y\_n\_1, **z\_n\_1=grip** , st\_n\_1, ct\_n\_1, **open** ) to the **grip** plane]
    * Shot 1
        * [take picture I\_n\_1 with gripper and objects in tray]
  - ---------------------------------------------------------------------------
    * Pick
        * [close the gripper for a grip attempt]
    * Shot 11
        * [take picture I\_n\_11 of the tray, recording the immediate grip result]
    * Move
        * [move gripper to the drop position c\_drop above the bin]
    * Shot 12
        * [take picture I\_n\_12 of the tray, recording the confirmed results in the tray after grip attempt, the number of objects in tray &quot;-1&quot;]
    * Drop
        * [open the gripper at drop position c\_drop to release object to the bin]
    * Shot 13
        * [5K I\_n\_13 of the bin, recording the confirmed results in the bin after drop, the number of objects in bin &quot;+1&quot;]
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
    * ITF-SMS0 (= Shot00 =&gt; Move 00 =&gt; Shot 01)
    * ITF-MoSo (= Move 1 =&gt; Shot 1)
    * ITF-Pick (= Pick =&gt; Shot 11 =&gt; Move =&gt; Shot 12 =&gt; Drop =&gt; Shot 13)

## Daily Stage

Once a set of initial weights are obtained, the data collection and model training process is then carried out daily. For example, the Robot System will collect data during day time and leave the Learning Computer to update the network during night time. The daily stage is expected to last for 2~4 weeks, accumulating a total of 30~50k attempts. The dual camera input could possibly double the training data to 60~80k attempts, but at the risk of over fitting the model.

* Start from idle position c\_idle
* [Daily] Grip Cycle n = 1 (start of today&#39;s Learning Cycle)
    * ...
* [Daily] Grip Cycle n (end of last Grip Cycle n-1 = start of this Grip Cycle n)
    * ITF-SMS0 (= Shot00 =&gt; Move 00 =&gt; Shot 01)
  - ---------------------------------------------------------------------------
    * CEMs (decide whether to Pick, Move, or Raise)
        * [CEMs denote 3 cycles of calculation using Cross-Entropy Method (CEM) to infer a possible new motor command v\_n\_x\* for a new waypoint for picking using the trained model g\_m.
        * A decision is made on whether a successful grasp can be performed based on the ratio (p\_n) between the calculated grasp success possibility at the current waypoint (g\_m(I\_n\_x\_1, close)) and that of at a new waypoint (g\_m(I\_n\_x-1, v\_n\_x)).]
    * ITF-MoSo (if 50% &lt; p\_n =&lt; 90%)
        * Repeat CEMs no more than 10 times
    * ITF-Pick (if p\_n &gt; 90%)
        * Close the gripper for a grip attempt and stop this Grip Cycle
    * ITF-Raise (if p\_n &lt;= 50%)
        * Move the gripper by raising it 10 cm directly and stop this Grip Cycle
  - ---------------------------------------------------------------------------
* [Daily] Grip Cycle n+1 (end of this Grip Cycle n = start of next Grip Cycle n+1)
    * ...
* [Daily] Grip End (end of this Learning Cycle = start of next Learning Cycle)
    * Move [move gripper back to idle position c\_idle, marking the end of the experiment]

## End Stage

The End Stage is straight-forward as the following.

* Back to Idle Position (x\_idle, y\_idle, z\_idle=0, theta\_idle, open)
* Learning and Grasping Result Summary

# Data Management

Image Data (I\_n) and Motor Data (v\_n) are the major data to be transmitted and processed. However, there is a potential to include further Motion Data, Grip Data and Sensor Data (TBD)

Efficient data saving (TBD)

Time stamp synchronization (TBD)

The whole learning process consists of a lot (m) of repetitive Learning Cycles

- Each Learning Cycle consists of a lot (n) of repetitive Grip Cycles.
- Each Grip Cycle consists of a series of Integrated Task Flows (ITF)
- Each ITF consists of a series of Actions (Shot, Move, Pick, Drop, etc.)
- Each Action executes/generates certain Data
- Shot Action
  - generates visual Data
- Move Action
  - executes waypoint Data (position and rotation as a vector)
  - generates motion Data (Arm, Gripper, FT Sensor)
- Pick Action
  - executes close grip Data (adaptive grasp mode: Basic or Pinch)
  - generates motion/sensor Data (Gripper, FT Sensor)
- Drop Action
  - executes open grip Data (adaptive grasp mode: Basic or Pinch)
  - generates motion/sensor Data (Gripper, FT Sensor)
- XXXX Action
  - XXXX
