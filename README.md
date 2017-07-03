# as_urobotiq

<div align="center">
  <img src="http://ancorasir.com/wp-content/uploads/2017/01/URobotiq-ur5a3f-DefaultScene.jpg"><br><br>
</div>

[as_urobotiq](https://github.com/ancorasir/as_urobotiq) is intended to be setup as a general purpose robot platform for autonomous pick-and-place with deep learning. The current robot setup includes a [UR5 arm from Universal Robot](https://www.universal-robots.com/products/ur5-robot/), an [Adaptive 3 Finger gripper from Robotiq](http://robotiq.com/products/industrial-robot-hand/), a [Kinect for X-box One camera from Microsoft](http://www.xbox.com/en-US/xbox-one/accessories/kinect), an [Xtion Pro camera from ASUS](https://www.asus.com/3D-Sensor/Xtion_PRO/), and an [FT300 sensor from Robotiq](http://robotiq.com/products/robotics-force-torque-sensor/), all controlled by a high performance desktop computer with [NVIDIA TITAN X Graphics Card](https://www.nvidia.com/en-us/geforce/products/10series/titan-x-pascal/) running [Tensorflow](https://www.tensorflow.org/) to autonomously learn object grasping using Deep Learning.

This is a project under-development at the Sustainable and Intelligent Robotics ([SIR](http://ancorasir.com)) Group , under the supervision of Dr. Chaoyang Song.

as_urobotiq: A general purpose robot for autonomous pick-and-place with deep learning. Github.com/ancorasir/as_urobotiq, [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.581804.svg)](https://doi.org/10.5281/zenodo.581804)

Current branch: master. To simulate the robots, go to branch: ros-indigo. To test tensorflow, go to branch: tf


# Research Goal

The aim of this project is to explore autonomous and adaptive robotic pick-and-place through deep learning.

[Recent research by Google](https://sites.google.com/site/brainrobotdata/home) has demonstrated the possibility of training a robotic learning system with hand-eye-coordination during grasping task with significantly improved success rate. Yet to be validated, this approach could provide the flexibility on the extensive visual calibration and object recognition during usual robotic pick-and-place tasks.

In human pick-and-place actions, there's another critical component that involves the dexterity of human hand and fingers, which enables the effective manipulation of objects. This is what we aim at exploring in this project, where a similar robotic setup is presented with a more advanced gripper with optional operation modes for adaptive grasping. Currently, a Robotiq Adaptive 3 Finger gripper is configured for this project. Future development involves the integration of custom hybrid grippers to be developed at the AncoraSIR lab.

## Structured digitization of robotic grasping
With specific attentions on the structured digitization of the robotic grasping data collected from ROS and the robot
* Robot System (RS) and Controller Computer (CC)
	* RS: structured data collection from each robotic component
  * CC: systematic data recording for efficient storage
  * RS+CC: effective labelling of robotic data for autonomous learning


## Simulation and implementation of robotic grasping
With specific attentions on setting up the simulation (ROS) and realization of robotic grasping for data collection
* Controller Computer (CC)
	* CC: ROS simulation of the robot platform
	* CC: hardware implementation of the robot platform
	* CC: robot action programming for the grasping task


## Grasping data processing for deep learning
With specific attentions on processing the recorded robot data for model training in tensorflow with efficiency
* Controller Computer (CC) and Learning Computer (LC)
	* CC: selective robot data processing for LC
	* LC: cross-entropy method and servoing mechanism programming
	* CC+LC: efficient data interfacing between CC and LC

## Deep learning modeling for robotic grasping
With specific attentions on the modification of Google’s learned model for our robotic grasping learning
* Learning Computer (LC)
	* LC: minimum network design and modeling
	* LC: optimization and modification on Google's existing learned model
	* LC: introduction of adaptive grasping to the autonomous learning


# Research Team

Dr. SONG, Chaoyang (Chaoyang.Song@Monash.edu)
* Principal Investigator

Dr. WAN, Fang
* Technical Consultant

Mr. CHEN, Yaohui
* Research Assistant

XIA, Tian | HE, Xiaoyi | DENG, Yuntian | ZHANG, Jingwei | JIANG, Will | ONG, Ben
* Team Members


# System Setup

* World => Pedestal => Arm => FT Sensor => Gripper
* World => Desk => Tray 1 => Objects (to be picked)
* World => Desk => Tray 2 => Objects (to be placed)
* World => Camera 1 (main image input on gripper, objects and trays)
* World => Camera 2 (auxiliary image recording for referencing)

## Hardware

* Robot System
	* Arm: [UR5 from Universal Robot](https://www.universal-robots.com/products/ur5-robot/)
	* Gripper: [Adaptive 3 Finger from Robotiq](http://robotiq.com/products/industrial-robot-hand/)
	* Camera 1: [Kinect for X-box One from Microsoft](http://www.xbox.com/en-US/xbox-one/accessories/kinect)
  * Camera 2: [Xtion Pro from ASUS](https://www.asus.com/3D-Sensor/Xtion_PRO/)
  * FT Sensor: [FT300 from Robotiq](http://robotiq.com/products/robotics-force-torque-sensor/)
* Control Computer (CC)
	* Standard PC with Ubuntu Trusty (to be integrated with LC)
	* Sufficiently large hard drive (1TB SSD)
* Learning Computer (LC)
	* CPU: Intel Core i7 6800K Hex-Core 3.4GHz
	* GPU: Single 12GB [NVIDIA TITAN X (Pascal)](https://www.nvidia.com/en-us/geforce/products/10series/titan-x-pascal/)
	* RAM: 32GB Corsair Dominator Platinum 3000MHz (2 X 16GB)
	* SSD: 1TB Samsung 850 Pro Series

## Software

* ROS Indigo on Control Computer
* Tensorflow on Learning Computer


# Task Decomposition

* Start
	* Safety Check and Preliminary Preparation (TBD)
  * From Start Position (TBD)
* Operation
	* Learning Cycle m = 1
  * Network Training based on all Grip Cycles in Learning Cycle m = 1
  * Learning Cycle m = 2
  * Network Training based on all Grip Cycles in Learning Cycle m = 2
  * Learning Cycle m = 3 ~ 5
  * Network Training based on all Grip Cycles in Learning Cycle m = 3 ~ 5
* End
	* Obtain final Learned Model (TBD)
  * Back to Start Position (TBD)
  * Learning and Grasping Result Summary (TBD)
  * Validate final Learned Model (TBD)

## Learning Cycle (m = 1, 2, 3 ~ 5)

During each Learning Cycle, the Robot System is going to accumulate sufficient dataset by repetitively performing pick-and-place tasks, after which a Network Training is going to take place, updating the weights of the neural network (g), aiming at an improving performing at doing the pick-and-place tasks in the next Learning Cycle (or goal application).

One can interpret it as an alien life with vision and grasping capabilities sitting in front of a desk, being asked to pick up objects from Tray 1 and place it in Tray 2. The alien only knows that it can "grasp" (with gripper) and it can "see" (with camera), but has never actually grasped from or seen anything in the trays before (establishing hand-eye-coordination). And it is going to "learn" how to do it, starting with "blindly" (before hand-eye-coordination is established) moving its gripper towards "possible" (CEMs) coordinates above an object and attempts to pick it up. Through the repetition of these attempts with validation from its "eye", a learning process is established (through the updated weights of the neural network after each Learning Cycle).

The 1st Learning Cycle starts with randomized initial weights in the network, which leaves with three possible ways to start the task. But overall, this is something very similar to the "claw machine" picking up toys from the box, either you start by letting the robot to do it randomly, do it blindly, or do it under guidance. (It would be interesting to see an "intelligent" claw machine built following similar structure, which should be much simpler and also interesting.)

1. Start the robot with randomized initial values for the neural network to start with. No matter how bad or good the initialized weights are, the robot is expected to correct itself later through the learning process. But just not necessary. This is just unnecessary unless a special purpose needs to be achieved.
  * advantage would be the reuse of code, and the disadvantage would be a much larger dataset to make sure the learned model can be corrected through learning (potentially too large to be meaningful).

2. Start the robot with total blind grasps, i.e. just reach out to any one of all possible workspace and perform a grasp and check the results right after.
  * advantage would be the simplicity and automation of programming in the 1st Learning Cycle, and the disadvantage would be the lack of purpose, which may result in very low successful grasp in the beginning, slowing down the learning process.

3. Start the robot with human-guided grasps, i.e. send labelled coordinates of potential grasps to the robot and let the robot grasp (this is the part that is interestingly similar to "claw machine", possibly making it a joy to "play" with it.)
  * advantage would be the "manual" speedup of the learning process with purpose and successful grasps, the disadvantage would be not sure if this is a good or bad thing for robotic grasping with deep learning.

Starting from the 2nd Learning Cycle, the weights of the neural network starts updating, indicating its learning results while performing the next cycle of data collection.

Currently, only 5 rounds of Learning Cycle are designed. However, more rounds may be required as there's only one robot performing this learning task in this project.

## Typical Grip Cycles (n = 10,000 ~ 20,000) during Learning Cycles m = 2 ~ 5

A Learning Cycle comprises of a large number of Grip Cycles, collecting data to update the weights of the neural network (g). As mentioned above, a typical Learning Cycle starts from the 2nd until the last, which comprises of a standard structure as the following to be repeated the most times.

__Learning Cycle m = 2~5__:
  * Grip Cycle n = 1 (start of __this__ Learning Cycle 1)
  * ...
  * Grip Cycle n (end of __last__ Grip Cycle n-1 = start of __this__ Grip Cycle n)
      * Shot 00 (without gripper, only the objects in tray)
      * Move 00 (randomly move gripper into target workspace)
      * Shot 01 (with gripper and objects in tray)
      * ----------
      * __CEMs 1 => Move 1 => Shot 1 (prepare to perform Pick)__
      * __CEMs 2 => Move 2 => Shot 2 (prepare to perform Pick)__
      * ...
      * __CEMs x => Move x => Shot x (decides to perform Pick)__
      * ----------
      * Pick (close grip)
      * Shot x1 (grasp moment record on Tray 1)
      * Move (to Tray 2)
      * Shot x2 (grasp success detection on Tray 1, objects in Tray 1 "-1")
      * Drop (open grip)
      * Shot x3 (grasp success detection on Tray 2, objects in Tray 2 "+1")
  * Grip Cycle n+1 (end of __this__ Grip Cycle n = start of __next__ Grip Cycle n+1)
  * ...
  * Grip End (end of __this__ Learning Cycle 1 = start of __next__ Learning Cycle 2)

Each Grip Cycle (n) consists of certain Integrated Task Flows (ITFs), including ITF-SoMo, ITF-CEMs, and ITF-Pick. Each ITF shall be integrated into a script of commands to be called and executed.

* __ITF-SoMo: Shot 00 => Move 00 => Shot 01__
  * input: null
  * processed: v_n_0
  * output: I_n_00, I_n_01
  * This ITF aims at initiate the pick-and-place process with the initial two pictures without and with the gripper. It takes an initial picture with Shot 00 (I_n_00) of the target tray (without gripper), following the motor command (v_n_0) that moves the gripper into a fixed coordinate inside the workspace of the tray waiting to start the pick-and-place task, and take another initial picture with Shot 01 (I_n_01) with gripper inside.

* __ITF-Serv: CEMs => Move x => Shot x (repeat up to 10 times in a Grasp Cycle since start)__
  * input: I_n_x-1 (if x = 1, then I_n_x-1 = I_n_01), g_m
  * processed: p_n_x
  * output: close (if p_n_x > 90%), v_n_x (="raise") (if p_n_x <= 50%), or v_n_x (if 90% >= p_n_x > 50%), I_n_x
  * This ITF aims at applying the network based on the given input and decide (p_n) whether a successful grasp can be performed. p_n refers to the ratio between the calculated grasp success possibility at the current waypoint (g_m(I_n_x_1,close)) and that of at a new waypoint (g_m(I_n_x-1,v_n_x)).
    * if p_n > 90%: meaning the network is confident enough to perform a successful grasp at the current waypoint, comparing to all possible new waypoints it calculated during ITF-CEMs, decides to "close" the gripper, attempting the object grasping.
    * if p_n <= 50%: meaning the network has no confident at all to perform a successful grasp at the current waypoint, therefore decides replace the calculated motor command v_n_x with a command to "raise" the gripper 10 cm higher for safety, and executes this new motor command v_n_x (="raise") to a new waypoint in a new Grip Cycle.
    * else (if 90% >= p_n > 50%): meaning the network has some confident that it can perform a successful grasp in this Grip Cycle at the next waypoint, therefore decides to follow the calculated motor command v_n_x to a new waypoint and continue this Grip Cycle.

* __ITF-CEMs: CEMs = CEM1 => CEM2 => CEM 3 => heuristics rules(repeat up to 10 times in a Grasp Cycle since start)__
  * input: I_n_x-1 (if x = 1, then I_n_x_1 = I_n_01), g_m
  * processed: v_n_x*, p_n_x*
  * output: v_n_x, p_n_x
  * This ITF aims at using the Cross-Entropy Method (CEM) to infer a possible new motor command v_n_x* for a new waypoint. The CEM is repeated three times to forecast the best possible waypoint with the highest confidence of a successful grasp. The resulting motor command v_n_x involves a directional component d_n_x = (x,y,z) and a rotational component r_n_x = (sin(theta), cos(theta)), indicating move along which direction by how far and rotate along the global vertical axis by how much will the gripper arrive at a new waypoint for grasping.

* __ITF-Pick: Pick => Shot x1 => Move => Shot x2 => Drop => Shot x3 (perform once at the end of a Grasp Cycle)__
  * input: close
  * processed: I_n_x1, I_n_x2, I_n_x3
  * output: s_n (to be used later to update g)
  * This ITF aims at attempting to pick up the object and validate the outcome by object detection on the gripper and in the tray. Pick action executes the "close" command. The combination of Shot x1, x2, and x3 generates a decision (s_n) on whether this is a successful Grip Cycle.
    * Shot x1 takes a picture of the gripper and the objects in Tray 1 after Pick.
    * Shot x2 takes a picture of Tray 1 without the gripper after the gripper Move to Tray 2.
    * Shot x3 takes a picture of the gripper and the objects in Tray 2 after Drop.

## 1st Possible Grip Cycles (n = 5,000) during Learning Cycle m = 1 (pass)

The following example demonstrates the 1st possible structure of the 1st Learning Cycle, as introduced earlier and repeated as below. This is a simple repetition of the Typical Grip Cycles with randomized initial values in g_m. This is just unnecessary unless a special purpose needs to be achieved.

```
Start the robot with randomized initial values for the neural network to start with. No matter how bad or good the initialized weights are, the robot is expected to correct itself later through the learning process. But just not necessary. This is just unnecessary unless a special purpose needs to be achieved.
  * advantage would be the reuse of code, and the disadvantage would be a much larger dataset to make sure the learned model can be corrected through learning (potentially too large to be meaningful).
```

## 2nd Possible Grip Cycles (n = 5,000) during Learning Cycle m = 1 (chosen)

The following example demonstrates the 2nd possible structure of the 1st Learning Cycle, as introduced earlier and repeated as below. This simplifies the ITF-Serv process in the 1st Learning Cycle.

```
Start the robot with total blind grasps, i.e. just reach out to any one of all possible workspace and perform a grasp and check the results right after.
  * advantage would be the simplicity and automation of programming in the 1st Learning Cycle, and the disadvantage would be the lack of purpose, which may result in very low successful grasp in the beginning, slowing down the learning process.
```

__Learning Cycle m = 1__:
  * Grip Cycle n = 1 (start of __this__ Learning Cycle 1)
  * ...
  * Grip Cycle n (end of __last__ Grip Cycle n-1 = start of __this__ Grip Cycle n)
      * Shot 00 (without gripper, only the objects in tray)
      * Move 00 (randomly move gripper into target workspace)
      * Shot 01 (with gripper and objects in tray)
      * ----------
      * __Rand (randomly generate a v_n for grasping)__
      * __Move (execute v_n)__
      * __Shot (I_n)__
      * ----------
      * Pick (close grip)
      * Shot x1 (grasp moment record on Tray 1)
      * Move (to Tray 2)
      * Shot x2 (grasp success detection on Tray 1, objects in Tray 1 "-1")
      * Drop (open grip)
      * Shot x3 (grasp success detection on Tray 2, objects in Tray 2 "+1")
  * Grip Cycle n+1 (end of __this__ Grip Cycle n = start of __next__ Grip Cycle n+1)
  * ...
  * Grip End (end of __this__ Learning Cycle 1 = start of __next__ Learning Cycle 2)

* __ITF-RaMo: Rand => Move => Shot__
  * input: null
  * generate: v_n, I_n
  * output: null
  * This ITF aims at randomly selecting a motor command v_n, leading to a new waypoint in workspace of Tray 1, attempting a blind grasp, "hoping" to get lucky and pick up something. One may be able to relate this to a man trying to pick up rocks from a shallow muddy water, who is left with no other options but to blindly reach into the water and try his luck.

## 3rd Possible Grip Cycles (n = 5,000) during Learning Cycle m = 1 (optional for optimization)

The following example demonstrates the 3rd possible structure of the 1st Learning Cycle, as introduced earlier.

```
Start the robot with human-guided grasps, i.e. send labelled coordinates of potential grasps to the robot and let the robot grasp (this is the part that is interestingly similar to "claw machine", possibly making it a joy to "play" with it.)
  * advantage would be the "manual" speedup of the learning process with purpose and successful grasps, the disadvantage would be not sure if this is a good or bad thing for robotic grasping with deep learning.
```

__Learning Cycle m = 1__:
  * ...
  * __Hand Pick a list of v_n to be automatically performed by the Robot System__
  * Grip Cycle n (end of __last__ Grip Cycle n-1 = start of __this__ Grip Cycle n)
      * Shot 00 (without gripper, only the objects in tray)
      * Move 00 (randomly move gripper into target workspace)
      * Shot 01 (with gripper and objects in tray)
      * ----------
      * __Take (pick one from the list of given v_n)__
      * __Move (execute v_n)__
      * __Shot (I_n)__
      * ----------
      * Pick (close grip)
      * Shot (grasp moment record on Tray 1)
      * Move (to Tray 2)
      * Shot (grasp success detection on Tray 1, objects in Tray 1 "-1")
      * Drop (open grip)
      * Shot (grasp success detection on Tray 2, objects in Tray 2 "+1")
  * Grip Cycle n+1 (end of __this__ Grip Cycle n = start of __next__ Grip Cycle n+1)
  * ...

* __ITF-Hand: Take => Move => Shot__
  * input: v_n
  * generate: I_n
  * output: null
  * This ITF aims at following a list of hand picked v_n, leading to waypoints leading to possibly successful object grasping as training data. (just like playing the "claw machine")
  * A possible optimization is introducing this 3rd Grip Cycle into the 2nd Grip Cycle to generate variations in the initial training data.

## Data

Image Data (I_n) and Motor Data (v_n) are the major data to be transmitted and processed. However, there is a potential to include further Motion Data, Grip Data and Sensor Data (TBD)

Efficient data saving (TBD)

Time stamp synchronization (TBD)

* The whole learning process consists of a lot (m) of repetitive Learning Cycles
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
  * XXXX Action
    * XXXX

## Safety (TBD)

* Virtual boundaries of robot workspace in a cubical shape, programmable in UR5 controller.
  * bottom side: align with tray surface
  * top side: 80cm above tray surface
  * robot side: 10cm inside tray, close to robot
  * camera side: 10cm inside left tray, close to camera
  * left side: 10cm inside right tray, on robot left
  * right side: 10cm inside tray, on robot right
* Virtual boundaries of robot Pick/Drop actions workspace
  * similar as above, with respect to the corresponding Pick/Drop tray

* The Robotiq Gripper has a rigid design, meaning that it can not interact with rigid boundaries of objects when collision occurs.
  * object selection
  * soft padding on the bottom of the tray (2cm thickness)
  * FT Sensor integration

* Wrist orientation of the gripper
  * for easy implementation and testing, only vertical pick-and-place will be allowed, meaning the gripper will be set to allow only translation in X, Y, Z direction and rotation about Z axis.
