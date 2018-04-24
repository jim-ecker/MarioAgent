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

  * ### Deep Q Network  
    explaination of DQN here

  * ### ACKTR
    explaination of ACKTR
    
    ACKTR (pronounced “actor”) — Actor/Critic using Kronecker-factored Trust Region
    * Developed by researchers at the University of Toronto and New York University.
    * Used to learn control policies for 
      * Simulated robots
        * Input: Raw pixels
        * Action Space: continuous
      * Video game agents
        * Input: Raw pixels
        * Action Space: discrete

    ACKTR combines three distinct techniques: **actor-critic methods**, **trust region optimization** for more consistent improvement, and **distributed Kronecker factorization** to improve sample efficiency and scalability.
    
    Complexity within machine learning algorithms is characterized with respect to two metrics: **sample complexity** and **computational complexity**.  
    **Sample complexity** refers to the number of timesteps of interaction between the agent and its environment  
    **Computational complexity** refers to the amount of numerical operations that must be performed.

    ACKTR has better **sample complexity** than first-order methods such as A2C (Advantage Actor Critic) because it takes a step in the **natural gradient** direction , rather than the **gradient** direction (or a rescaled version as in ADAM).  
    The **natural gradient** gives us the direction in parameter space that achieves the largest (instantaneous) improvement in the objective, per unit of change, in the output distribution of the network, as measured using the **KL-divergence** (Kullback-Leibler). By limiting the KL divergence, we ensure that the new policy does not behave radically differently than the old one, which could cause a collapse in performance.

    As for **computational complexity**, the KFAC update used by ACKTR is only 10-25% more expensive per update step than a standard gradient update. This contrasts with methods like **TRPO** (Trust Region Policy Optimization - i.e, Hessian-free optimization), which requires a more expensive **conjugate-gradient** computation.

    In the following video you can see comparisons at different timesteps between agents trained with ACKTR to solve the game Q-Bert and those trained with A2C. ACKTR agents get higher scores than ones trained with A2C.
