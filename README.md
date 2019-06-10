# Deep Racer

This Sample Application runs a simulation which trains a reinforcement learning (RL) model to drive a car around a track.

_AWS RoboMaker sample applications include third-party software licensed under open-source licenses and is provided for demonstration purposes only. Incorporation or use of RoboMaker sample applications in connection with your production workloads or a commercial products or devices may affect your legal rights or obligations under the applicable open-source licenses. Source code information can be found [here](https://s3.console.aws.amazon.com/s3/buckets/robomaker-applications-us-east-1-72fc243f9355/deep-racer/?region=us-east-1)._

Keywords: Reinforcement learning, AWS, RoboMaker

![deepracer-hard-track-world.jpg](docs/images/deepracer-hard-track-world.jpg)

## Requirements

- ROS Kinetic (optional) - To run the simulation locally. Other distributions of ROS may work, however they have not been tested
- Gazebo (optional) - To run the simulation locally
- Stable Baselines

## Usage

### Training the model

#### Building the simulation bundle

```bash
cd simulation_ws
rosws update
rosdep install --from-paths src --ignore-src -r -y
colcon build
colcon bundle
```

#### Running the simulation

You can run local training using the roslaunch command

```bash
source simulation_ws/install/setup.sh
roslaunch deepracer_simulation local_training.launch
```

### Evaluating the model

#### Building the simulation bundle

You can reuse the bundle from the training phase again in the simulation phase.

#### Running the simulation

The evaluation phase requires that the same environment variables be set as in the training phase. Once the environment variables are set, you can run
evaluation using the roslaunch command

```bash
source simulation_ws/install/setup.sh
roslaunch deepracer_simulation evaluation.launch
```

## License

Most of this code is licensed under the MIT-0 no-attribution license. However, the sagemaker_rl_agent package is
licensed under Apache 2. See LICENSE.txt for further information.

## How to Contribute

Create issues and pull requests against this Repository on Github
