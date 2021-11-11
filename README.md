### SelfdrivingcarForza

<h4>FynnJammerinc / Jammerinc</h4>

Portfolio: www.fynnjammer.com/projects

github: https://github.com/jimhoggey

## Youtube:
Long video explaining in depth: https://youtu.be/KOZXrtPuaR8 
Shorter video project video goal and output: https://youtu.be/Jl_7BrnvueU
None error video: https://youtu.be/u32axlrp8uU
Connecting to forza problem logbook: https://youtu.be/gbRIlkofcc0 

Self-driving cars will continue to become increasingly popular and will form the future of personal transportation in everyday life. Companies like Tesla, Google and Waymo are already testing and developing this technology, making it an extremely new and unknown field of technology and research. This enticed me to approach this area of emerging technology as my project, but with a twist by using video games to act as a platform to test and develop parts of self-driving technology.   

## Overview
Using a video game (in this case Forza horizon 3) as a test environment. This is place where we can test and adapt the self-driving car model virtually, it also allows us to change specific parameters to evaluate how the self-driving car model adapts and behaves in different conditions, such as with a different car, different location (city, highway, offroad) and different weather conditions.
This research project will explore both the method and the real-world potential of a self-driving car model used in a video game simulation designed to also work in the real world. This project aims to assess the viability of a self-driving car model designed to function in the real world in a simulated environment. To assess this, a prerecorded video of a car being driven in a simulated environment(Video game) will be processed and put through several algorithms designed to extract useful and potentially relevant information that would assist a future full self driving car model to make accurate decisions based on these extractions.

## Aim of this project
Is to accurately extract and identify features, from a prerecorded video in a simulated environment. These features should include active tracking of lane markings, other vehicles, pedestrians, traffic lights and speed signs. 

### For this project I will be using
```
Python
Opencv2
matplotlib
numpy
tensorflow
```
## Packages used:

https://www.udemy.com/course/autonomous-cars-deep-learning-and-computer-vision-in-python/ (udamy course) 
for frist gen line dection system and to learn about the basics (very good to get started)

https://github.com/uppala75/CarND-Advanced-Lane-Lines (Lane Line System)
Really good write up, clear and works effectivly 

https://github.com/Sigil-Wen/YOLO (yolo)
simple, easy to use just have to download the weights, for car and people dection

download data german_traffic sign benchmark, test.p, train.p and validate.p

https://s3.amazonaws.com/udacity-sdc/datasets/german_traffic_sign_benchmark/test.p
https://s3.amazonaws.com/udacity-sdc/datasets/german_traffic_sign_benchmark/train.p
https://s3.amazonaws.com/udacity-sdc/datasets/german_traffic_sign_benchmark/validate.p


## Versions
To see full list of all installed packages go to allinsatlled (https://github.com/jimhoggey/SelfdrivingcarForza/blob/main/allinstalled.txt
```
tensorflow              2.6.0
tensorflow-estimator    2.6.0
jupyter-client          7.0.1
jupyter-core            4.7.1
jupyterlab-pygments     0.1.2
keras                   2.6.0
Keras-Preprocessing     1.1.2
matplotlib              3.3.4
opencv-python           4.0.0.21
Pillow                  8.3.1
pip                     21.2.4
pooch                   1.4.0
pycparser               2.19
pyglet                  1.3.2
pyparsing               2.4.7
python-dateutil         2.8.2
scikit-image            0.17.2
scikit-learn            0.24.2
scipy                   1.5.4
seaborn                 0.11.2

```


# Quick Overview of how the model should work.
![image](https://user-images.githubusercontent.com/31178932/132157438-25501fe5-f405-4192-a834-ca0222e015a3.png)

A visual representation of how my model will interact with the Forza horizon game engine.
![image](https://user-images.githubusercontent.com/31178932/132157516-dd2e1aa5-8c7d-47cb-b9ef-1a010e3af4e2.png)

## Full project: Full self drivning car that also controls the car in Forza Horizon 
Project 1: get lane detection, vehicle detecion to work on a video file feature extraction

Project (future project): use what we learnt in Project 1 and use a live video import captured from game, use Neat neural network to assign rewards for correct driving to train network. Have hard rules such as car crashes -10 rewards and car stay in lane to +1 reward for example.

Goal/ things to do outline:

1. simple laine dection 
2. advanced line dection
3. svm car dection (did't work)
4. yolo car and people dection 
5. set forza game as input for system 
6. control forza with keyboard input
