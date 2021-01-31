# I Hiter Decision Group

This project is the decision group code display of ICRA2020 AI challenge IHiter team

## 1. Introductioon

The software of decision-making part includes **reinforcement learning environment simulation and visualization** and **Deep reinforcement learning network training part**. Among them:
* **simulation and visualization of reinforcement learning environment** is based on the simulation of the game state of the game robot, and the game process is abstracted as Markov decision process, which is written according to the requirements of gym environment. At the same time, the algorithm should be fast enough to make the intensive training process fast.
* **The deep reinforcement learning network training part** is to train a large number of samples by building a deep reinforcement learning network. Taking the observable state of the current field as the input, the score of each action is output, and the action with the highest score is selected as the next action.

## 2. Demonstration

### Demonstration of top view of playing field

This software uses pyglet as graphics rendering library,
The color blocks on the map are as follows:

* The blue square represents our robot
* Red squares represent enemy robots
* Use small black squares to represent obstacles and boundaries
* The red and blue boxes indicate where the robot will start at the start of the game
* The gray box represents the buff area

The color and letter rules for the buff area are shown below

* Red represents the addition area of the red side,
* Blue represents the addition area of the blue square,
* Black represents the forbidden zone.
* The letter *A* stands for ammo,
* The letter *R* stands for blood return,
* The letter *M* means no movement,
* The letter *S* means no shooting.

<p align="center">
  <img style="display: block; margin: 0 auto;" src="./GIF/双机器人.gif" />
</p>  

### Simulation demonstration of competition process

In the process of simulation, robots need to do a lot of calculation,
Therefore, some techniques are used to speed up the calculation in the simulation demonstration of the competition process.

Checking whether the robot strikes an obstacle is a large part of the algorithm.
The software realizes collision detection by checking the geometric relationship between robot and obstacle,
In order to minimize the amount of computation,
The software uses numpy to calculate the matrix to calculate the position relationship between lines,
On this basis,
To calculate the position relationship between the outgoing line and the rectangle,
According to the position relationship between the line and the rectangle, the position relationship between the rectangle and the rectangle is calculated.
In determining the position relationship between line segments,
This paper uses a highly integrated cross product algorithm,
The running time of the algorithm is greatly reduced,
It provides a guarantee for the training of reinforcement learning.

In the buff section,
According to the rules of the game, this software writes a buff detection and operation system,
When the robot enters the buff area, the buff will be triggered,
After a certain period of time, the class buff will be cancelled,
When the buff of the bonus class is triggered, you can add blood and ammunition.
Refresh the buff area after 30s.

In the shooting part,
The environment can detect whether the robot's firing instruction can hit the predetermined target.
It can increase the calorific value,
When the heat value is exceeded, the robot will stop shooting.
If the ammunition runs out,
Robots can't shoot,
And output the above results.

### Robot status display

In order to show how the robot is running,
The software describes the basic state of each robot in the top view.
Including the amount of blood, ammunition, muzzle temperature display,
And use a color bar to represent the size.

## 3. Environment Requirement

* tensorflow-gpu >= 1.4
* tensorboard == 1.14.0
* python 3.6
* pyglet
* Numpy

## 4. How to install

clone code to local repository

```bash
git clone https://github.com/MengXiangBo/ICRA2020_RM_IHiter_Decision.git Decision
```

enter the program directory and install the relevant library

```bash
cd decision
pip install pyglet
pip install numpy
pip install tensorflow-gpu == 1.4
```

visualization of simulation environment

```bash
python env_show.py
```

training model

```bash
python train.py
```

run the following command to check the training effect (10 rounds)

```bash
python eval.py
```

## 5. File directory

```bash
|     debug.py
│     DuelingDQN.py  #Network file
│     env_ show.py  #Visual program display
│     eval.py  #Training effect evaluation
│     README.md  #Instructions
│     train.py  #Training document
│
|- IHiterenv # visualization and simulation environment folder
│     agent.py  #Store some robot classes
│     display.py  #Some visual classes are stored
│     env.py  #Master file of simulation environment
│     map_ element.py  #Some elements of the map
│     parameter.py  #Parameter file
│     policy.py  #Policy document
│     __ init__ .py
│
└─train_data # training data folder
      │   param.txt  #Super parameter record file
      |- checkpointsfile ා weight file
      └─ Summary # eventout file generated by training
```

## 6. Principle and theoretical analysis

### 6.1 Reinforcement learning simulation environment

Decision system simulation environment is the experimental platform of decision system algorithm,
It needs to output a lot of feedback information of state and action.
In order to make the decision system obtain a large number of training samples in a short time,
This group has built an environment system which can be used for robot's movement and shooting.

The experimental platform of simulation environment is round system,
That is, each input action will change the state of the robot,
Between every two input actions,
The state and simulation environment of the robot will not change.

The action of robot is composed of shooting action and moving action,
The movement and shooting of robot are discrete,
The specific implementation is as follows:

* The gun barrel and moving direction of the robot are the same
* The next move of the robot is the eight outer squares in the nine palace grid
* There are two shooting states of robot: shooting and no shooting

Integrate the actions of the two robots of the blue side into the actions of the blue team,
Because there are eight possibilities for movement,
There are two possibilities for shooting,
So for a robot,
The robot's action space is 16.
The action space of a team of two robots is the square of the action space of a single robot,
That's 256.

The state of a single robot consists of the following parts:

* Abscissa of robot
* The ordinate of robot
* Robot barrel angle
* Robot's blood volume
* Robot ammunition
* Gun barrel temperature of robot

The allocation of buff area also consists of six parts,
They are the buff codes of the six buff areas.
The coordinates of the four robots in the field are combined with the array allocated by the buff area,
In this way, an array of 30 numbers is formed,
After normalizing the array, the state of the environment is formed,
Input it into reinforcement learning network for training.

### 6.2 Robot decision system based on value function

In decision making,
We use dueling dqn network based on value function,
The Q value of the corresponding action of the robot in each state is fitted by neural network,
Choose the action corresponding to the maximum Q value to make the decision.
The architecture of the decision-making subsystem will be introduced in detail below.

#### 6.2.1 Dueling DQN network structure

The network structure is composed of three layers,
The first and second layers are full connected networks with 1024 neurons,
The activation function is relu.
The third layer is divided into a value network and an advantage network,
There is only one neuron in the value network,
Output the value of the state,
The advantage network outputs the advantage value of each action,
The direct output of the above two parts,
There is no active function.
Then the average value of the advantage of the action is returned to zero and the value of the value network is added to get the final action Q_ Eval value.

The formula is as follows:

<p align="center">
  <br>
  <img style="display: block; margin: 0 auto;" src="https://latex.codecogs.com/gif.latex?%5C%5B%7BQ_%7Beval%7D%7D%28s%2Ca%3B%5Ctheta%20%2C%5Calpha%20%2C%5Cbeta%20%29%20%3D%20V%28s%3B%5Ctheta%20%2C%5Cbeta%20%29%20&plus;%20%28A%28s%2Ca%3B%5Ctheta%20%2C%5Calpha%20%29%20-%20%5Cmax%20%5C%7B%20A%28s%2Ca%3B%5Ctheta%20%2C%5Calpha%20%29%5C%7D%20%29%5C%5D" />
</p>  

In the process of training,
The error of network is the Q_value of prediction_Eval value and simulated real value Q_Real,
Where Q_ The calculation method of real is based on Bellman principle

<p align="center">
  <br>
  <img style="display: block; margin: 0 auto;" src="https://latex.codecogs.com/gif.latex?%5C%5B%7BQ_%7Breal%7D%7D%28s%2Ca%3B%7B%5Ctheta%20%5E%20-%20%7D%29%20%3D%20reward%20&plus;%20%5Cgamma%20%7BQ_%7Bt%5Carg%20et%7D%7D%28s%27%2C%7Ba_%7B%5Cmax%20%7D%7D%3B%7B%5Ctheta%20%5E%20-%20%7D%29%5C%5D" />
</p>  

Where s' is the next state after action a in state s,
a_ Max is the maximum Q that can be obtained in s' state_ Target value,
γ is the discount factor,
θ - is the parameter of the network.
After the forward propagation, we can get the loss value of the network as follows:

<p align="center">
  <br>
  <img style="display: block; margin: 0 auto;" src="https://latex.codecogs.com/gif.latex?%5C%5Bloss%20%3D%20%7C%7C%7BQ_%7Breal%7D%7D%28s%2Ca%3B%7B%5Ctheta%20%5E%20-%20%7D%29%20-%20%7BQ_%7Beval%7D%7D%28s%2Ca%3B%5Ctheta%20%29%7C%7B%7C%5E2%7D%5C%5D" />
</p>  

By using this method, the predicted value of target network can be used as the approximate true value to calculate the loss value after Bellman processing.
At the same time, after a period of network training,
All weights of the target network are updated to those of the eval network.

#### 6.2.2 ϵ-greedy Exploration strategy

In the process of training, the ϵ grey exploration strategy was adopted,
That is, in the early stage of training, due to the poor decision-making ability of the network,
If the action completely depends on the decision network,
You fall into the trap of unwise behavior,
So we need to explore according to a certain probability of random action.
With the effect of network training getting better and better,
The probability of random action for exploration will also decrease,
The choice of training action is more and more dependent on network choice.

<p align="center">
  <img style="display: block; margin: 0 auto;" src="./PIC/网络结构.png" />
</p>  

## 6.3 环境奖励设定

### 6.3 setting of environmental reward

There are three kinds of environmental rewards,
namely:

* Mobile rewards
* Award for shooting
* Victory Award

During each step,
The environment records the movement and shooting results of each robot,
They are classified into several kinds,
Reward according to different structures,
The size of the move results and rewards are as follows:

* Enter the buff of forbidden class - 100
* Enter the enemy's bonus buff - 100
* Enter our bonus buff + 10
* Didn't hit an obstacle or enter the buff area - 0.1
* Install obstacles - 100

Shooting results:

* Robot autonomous decision not to shoot - 0.1
* Gun barrel temperature too high to shoot - 10
* Buffs with no shooting can't shoot - 10
* Not enough ammo to shoot - 10
* The target is dead and cannot attack - 10
* Unable to shoot to - 10 due to obstacles or target not in shooting range
* Attack success + 10

The final victory is a reward,
Victory will be awarded 1000

### 6.4 training effect

After about 1000 rounds,
60K steps,
The average revenue of each step of the agent has been greatly improved,
Convergence is achieved,
The results are as follows:

<p align="center">
  <img style="display: block; margin: 0 auto;" src="./PIC/训练过程中Reward每步的变化.jpg" />
</p>  

In the process of training,
Every 100 rounds of training will be tested,
To calculate the winning rate,
We can get the following chart:

<p align="center">
  <img style="display: block; margin: 0 auto;" src="./PIC/蓝方胜率.png" />
</p>  

As you can see,
After 600 rounds of training,
The winning rate of the network tends to be stable,
About 80%.
