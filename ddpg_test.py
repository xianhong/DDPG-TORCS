from gym_torcs import TorcsEnv
import numpy as np
from keras import backend as K
import tensorflow as tf
from ActorNetwork import ActorNetwork


def playGame(train_indicator=1):    #1 means Train, 0 means simply Run
    TAU = 0.001     #Target Network HyperParameters
    LRA = 0.0001    #Learning rate for Actor

    action_dim = 3  #Steering/Acceleration/Brake
    state_dim = 29  #of sensors input


    vision = False

    episode_count = 1
    max_steps = 1000 #100000
    done = False
    step = 0
    #Tensorflow GPU optimization
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    
    K.set_session(sess)

    actor = ActorNetwork(sess, state_dim, action_dim, 1, TAU, LRA)
    # Generate a Torcs environment
    env = TorcsEnv(vision=vision, throttle=True,gear_change=False)

    #Now load the weight
    print("Now we load Actor model's weights")
    try:
        actor.model.load_weights("actormodel.h5")
        print("Weight load successfully")
    except:
        print("Cannot find the weight")

    print("TORCS Experiment Start.")
    for i in range(episode_count):

        if np.mod(i, 3) == 0:
            ob = env.reset(relaunch=True)   #relaunch TORCS every 3 episode because of the memory leak error
        else:
            ob = env.reset()

        s_t = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY,  ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))
     
        total_reward = 0.
        
                
        for j in range(max_steps):
            a_t_original = actor.model.predict(s_t.reshape(1, s_t.shape[0]))
            
            ob, r_t, done, info = env.step(a_t_original[0])

            s_t1 = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))
        

            total_reward += r_t
            s_t = s_t1
            
            if np.mod(j,100) == 0: 
                print("Episode", i, "Step", step, "Action", a_t_original[0], "Reward", r_t)
            
            step += 1
            if done:
                break


        print("TOTAL REWARD @ " + str(i) +"-th Episode  : Reward " + str(total_reward))
        print("Total Step: " + str(step))
        print("")

    env.end()  # This is for shutting down TORCS
    print("Finish.")

if __name__ == "__main__":
    playGame(train_indicator=0)
