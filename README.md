# MarioAgent
Testbed for Deep Reinforcement Learning methods on Super Mario Brothers (1985)

## Dependencies

Python 3 >= 3.5

Install module dependencies via pip, preferably within a virtual environment  

```bash
  pip install git+https://github.com/ppaquette/gym-super-mario  
  pip install git+https://github.com/openai/baselines  
  pip install gym opencv-python python-gflags
```

## Models Implemented

### Deep Q Network  
  explaination of DQN here  
    
  * **Experience Replay**
    
    Upon entering each state, the agent generates an "experience," represented as a tuple (s,a,r,s') containing information on current state, s_t, the action selected via the action selection policy, \alpha, reward yielded by the pair s_t/\alpha, and the resultant next state, s'. The experience generated is then stored in the agent's "experience replay memory."

    Since we are working on reward trajectories with temporal dependence, experiences exhibit high correlation given temporal locality. We wish to decorrelate experience in order to avoid overfitting the agent's action selection.
    
  * **Target Networks**
    
    Since the agent is evaluating the value of each state/action pair via a Bellman equation, the agent is affecting its own network's weight structure upon each learning phase. This means that, effectively, the ground is moving under the agent as it's learning - introducing significant instability into the network. Target networks alleviate this instability by holding a network "frozen in time" separate from the network upon which the learning is happening. This way, intermittent updates happen to the agent asynchronously without introducing the instability of online network updates.
    
  * **Reward Clipping**
    
    This is simply a regularization of points and reward between different game environments
    
  * **Skipping Frames**
    
    Most games run at 60 frames per second. The agent does not need this high of a refresh rate to compute accurate state/action values. By skipping every other frame and including information for the past four frames, we can both lower the computational frequency and introduce some indication of velocity and direction through small scale history.
