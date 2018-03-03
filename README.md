## Using Keras and Deep Deterministic Policy Gradient to play TORCS

DDPG algorithm codes are based on Github repository : https://github.com/yanpanlau/DDPG-Keras-Torcs
DDPG algorithm codes are modified to be compatible with Pyhton/TF/Keras version as mentioned in 'Environment setup'.
Minor changes to hyper parameters of the original DDPG codes to reduce computation complexity.

The 'torcs.mp4' file is a video clip capturing a sample racing drive on TORCS after the model having been trained for more than 310K steps.
 

# Environment setup:
* OS: Ubuntu 16.04 LTS
* Python 3.6
* Keras 2.1.4
* Tensorflow 1.5
* Installation of Visual TORCS for RL learning with color vision 'vtorcs-RL-color' as listed in "Gym-torcs" project
  (https://github.com/ugo-nama-kun/gym_torcs)
* Installation of software packages of 'mencoder' (for stiching together sequence of PNG files captured by TORCS to a MPEG4 video) and 'xautomation' on Ubuntu.
* Setting TORCS' config/raceengine.xml file to enable video frame capturing (Press 'c' to start/stop video frame capturing)

# How to Use?
* Training : run 'python ddpg.py'
* Testing : run 'python ddpg_test.py'  (requiring the Actor network model being saved first!)


