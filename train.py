#!/usr/bin/env python
import gym
import os
import sys
import gflags as flags

from baselines.logger import Logger, TensorBoardOutputFormat, HumanOutputFormat

from deepq import deepq
from deepq.models import cnn_to_mlp

import ppaquette_gym_super_mario

from wrappers import MarioActionSpaceWrapper
from wrappers import ProcessFrame84

import datetime

PROJ_DIR = os.path.dirname(os.path.abspath(__file__))

FLAGS = flags.FLAGS
flags.DEFINE_string("log", "stdout", "logging type(stdout, tensorboard)")
flags.DEFINE_string("env", "ppaquette/SuperMarioBros-1-1-v0", "RL environment to train.")
flags.DEFINE_string("algorithm", "deepq", "RL algorithm to use.")
flags.DEFINE_integer("timesteps", 2000000, "Steps to train")
flags.DEFINE_float("exploration_fraction", 0.5, "Exploration Fraction")
flags.DEFINE_boolean("prioritized", False, "prioritized_replay")
flags.DEFINE_boolean("dueling", False, "dueling")
flags.DEFINE_float("lr", 5e-4, "Learning rate")

max_mean_reward = 0
last_filename = ""

start_time = datetime.datetime.now().strftime("%Y%m%d%H%M")


def train_dqn(env_id, num_timesteps):
    """Train a dqn model.

      Parameters
      -------
      env_id: environment to train on
      num_timesteps: int
          number of env steps to optimizer for

      """

    # 1. Create gym environment
    env = gym.make(FLAGS.env)

    # 2. Apply action space wrapper
    env = MarioActionSpaceWrapper(env)

    # 3. Apply observation space wrapper to reduce input size
    env = ProcessFrame84(env)

    # 4. Create a CNN model for Q-Function
    model = cnn_to_mlp(
      convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
      hiddens=[256],
      dueling=FLAGS.dueling
    )

    # 5. Train the model
    act = deepq.learn(
        env,
        q_func=model,
        lr=FLAGS.lr,
        max_timesteps=FLAGS.timesteps,
        buffer_size=10000,
        exploration_fraction=FLAGS.exploration_fraction,
        exploration_final_eps=0.01,
        train_freq=4,
        learning_starts=10000,
        target_network_update_freq=1000,
        gamma=0.99,
        prioritized_replay=FLAGS.prioritized,
        callback=deepq_callback
    )
    act.save("mario_model.pkl")
    env.close()


def deepq_callback(locals):

    global max_mean_reward, last_filename
    if 'done' in locals and locals['done'] == True:
        if 'mean_100ep_reward' in locals and locals['num_episodes'] >= 10 and locals['mean_100ep_reward'] > max_mean_reward:
            print("mean_100ep_reward : %s max_mean_reward : %s" % (locals['mean_100ep_reward'], max_mean_reward))

        if not os.path.exists(os.path.join(PROJ_DIR, 'models/deepq/')):
            try:
              os.mkdir(os.path.join(PROJ_DIR,'models/'))
            except Exception as e:
              print(str(e))
            try:
              os.mkdir(os.path.join(PROJ_DIR,'models/deepq/'))
            except Exception as e:
              print(str(e))
        if last_filename != "":
            os.remove(last_filename)
            print("delete last model file : %s" % last_filename)

        max_mean_reward = locals['mean_100ep_reward']
        act = deepq.ActWrapper(locals['act'], locals['act_params'])

        filename = os.path.join(PROJ_DIR,'models/deepq/mario_reward_%s.pkl' % locals['mean_100ep_reward'])
        act.save(filename)
        print("save best mean_100ep_reward model to %s" % filename)
        last_filename = filename


def main():
    FLAGS(sys.argv)
    logdir = "tensorboard"
    if FLAGS.algorithm == "deepq":
        logdir = "tensorboard/%s/%s_%s_prio%s_duel%s_lr%s/%s" % (FLAGS.algorithm, FLAGS.timesteps, FLAGS.exploration_fraction, FLAGS.prioritized, FLAGS.dueling, FLAGS.lr, start_time)

    if FLAGS.log == "tensorboard":
      Logger.DEFAULT \
        = Logger.CURRENT \
        = Logger(dir=None,
                 output_formats=[TensorBoardOutputFormat(logdir)])
    elif FLAGS.log == "stdout":
      Logger.DEFAULT \
        = Logger.CURRENT \
        = Logger(dir=None,
                 output_formats=[HumanOutputFormat(sys.stdout)])
    print("env : %s" % FLAGS.env)
    print("algorithm : %s" % FLAGS.algorithm)
    print("timesteps : %s" % FLAGS.timesteps)
    print("exploration_fraction : %s" % FLAGS.exploration_fraction)
    print("prioritized : %s" % FLAGS.prioritized)
    print("dueling : %s" % FLAGS.dueling)
    print("lr : %s" % FLAGS.lr)
    # Choose which RL algorithm to train.
    if FLAGS.algorithm == "deepq": # Use DQN
      train_dqn(env_id=FLAGS.env, num_timesteps=FLAGS.timesteps)


if __name__ == '__main__':
  main()
