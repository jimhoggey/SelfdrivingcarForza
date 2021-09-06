# SelfdrivingcarForza
Self-driving cars will continue to become increasingly popular and vital to everyday life. This means it is critical for self-driving cars to be of a high safety rating. In order to ensure this, my project is to create a self-driving car model that can be demonstrated and evaluated in the video game of Forza Horizon(racing game), as Forza provides lifelike factors, realistic roads, road signs, a general environment and accurate physics. The model will be able to drive itself, stay in its lane and avoid crashing.  

This research project will explore both the method and the real-world potential of a self-driving car model trained in a video game. In simple terms, a self-driving car has to read the road markings, road signs, and other vehicles and pedestrians and make decisions based on these factors. There are a couple of different ways to approach the self-driving car problem; the first and most simple is a hard programmed set of rules that the car follows in regards to speed, overtaking and give way rules and so on, but it was quickly determined that this way of approaching the problem would limit the capability the model would work in and the lack for ‘thinking outside the box, and problem-solving” would make this approach inappropriate. So the approach I want to explore is by tackling this problem using a machine learning technique. Which has less confinement and means there might be a higher chance of the self-driving car performing safely in the real world and in untested situations. The fundamental reason and potential benefit of having a self-driving car built and trained in a simulation is the ability to change factors quickly to train for a wide range of scenarios. These factors could be the environment, speed limits and a range of road conditions.

Quick Overview of how the model should work.
![image](https://user-images.githubusercontent.com/31178932/132157438-25501fe5-f405-4192-a834-ca0222e015a3.png)

A visual representation of how my model will interact with the Forza horizon game engine.
![image](https://user-images.githubusercontent.com/31178932/132157516-dd2e1aa5-8c7d-47cb-b9ef-1a010e3af4e2.png)

Full project: Full self drivning car that also controls the car in Forza Horizon 
Project 1: get lane detection, vehicle detecion to work on a video file 
Project 2: use what we learnt in Project 1 and use a live video import captured from game, use Neat neural network to assign rewards for correct driving to train network. Have hard rules such as car crashes -10 rewards and car stay in lane to +1 reward for example.

Goal/ things to do outline:

(currently working on)
1.  Read lane marking from video file.
  
2. Vehicle dection using SVM image recognition

