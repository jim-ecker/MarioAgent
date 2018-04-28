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

An implementation of the Deep Q Learning neural network introduced by Deepmind in the following Nature article:

[**Human-level control through deep reinforcement learning**](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf), V. Mnih, K. Kavukcuoglu, D. Silver, A. Rusu, J. Veness, M. Bellemare, A. Graves, M. Riedmiller, A. Fidjeland, G. Ostrovski, S. Petersen, C. Beattie, A. Sadik, I. Antonoglou, H. King, D. Kumaran, D. Wierstra, S. Legg, and D. Hassabis. **Nature 518 (7540): 529--533** (February 2015)

<Nature cover here>

The **Deep Q Network (DQN)** is a hybrid Convolutional & Fully Connected neural network that approximates the Q-function for the state-space over which the agent is performing some set of actions using only raw pixel values as input.

<DQN network diagram here>

The DQN achieves stability via four main features:

  * **Experience Replay**
    
    Upon entering each state, the agent selects an action via its action selection policy. It then sends this action to the environment.This generates an "experience," represented as the tuple (s,a,r,s'). This experience gives the agent the information it needs to evaluate its performance: s_t, the action selected via the action selection policy, \alpha, reward yielded by the pair s_t/\alpha, and the resultant next state, s'. The experience generated is then stored in the agent's "experience replay memory," E.

    Experiences exhibit high correlation given temporal locality since we are working with trajectories with temporal reward dependence. One should decorrelate the agent's experience in order to avoid overfitting the agent's action selection. This is achieved by sampling a finite batch of experiences b, representing a subset of E, via the uniform distrubtion. The agent uses then uses b as its data in the learning phase.
    
  * **Target Networks**
    
    Since the agent is evaluating the value of each state/action pair via a Bellman equation, the agent is affecting its own network's weight structure upon each learning phase. This means that, effectively, the ground is moving under the agent as it's learning - introducing significant instability into the network. Target networks alleviate this instability by holding a network "frozen in time" separate from the network upon which the learning is happening. This way, intermittent updates happen to the agent asynchronously without introducing the instability of online network updates.
    
  * **Reward Clipping**
    
    This is simply a regularization of points and reward between different game environments
    
  * **Skipping Frames**
    
    Most games run at 60 frames per second. The agent does not need this high of a refresh rate to compute accurate state/action values. By skipping every other frame and including information for the past four frames, we can both lower the computational frequency and introduce some indication of velocity and direction through small scale history.
    
 In their paper, Deepmind showed that the DQN was able to exhibit generality amongst a set of Atari 2600 games and achieved human level control in most, and superhuman control in some, of the games it was run against.
 
 <Perf chart from paper here>
