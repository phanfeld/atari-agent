# -*- coding: utf-8 -*-
import torch
import argparse
import gym
import os, glob
import numpy as np
from collections import deque
from utils.agent import Agent, Action_Scheduler, DQN
from helper.functions import preprocess, frame_to_tensor, clip_reward, generate_gif
import pickle

from copy import deepcopy

def train(n_actions, agent, env, env_name, path, rewards=None,frame_number=0, max_frames=30000000, 
          max_episode_length=18000,start_learning=50000, evaluate_every=200000, eval_length=10000, 
          network_update_every=10000, update_every=4):
    last_lives = 0
    terminal_life_lost = False
    if rewards == None:
        rewards = []
    
    losses = []

    states_for_action = deque(maxlen=4)
    
    try:
        action_scheduler = Action_Scheduler(num_actions=n_actions, max_frames=max_frames, 
                                            replay_memory_start_size=start_learning-frame_number, model=agent.main_dqn)
        
        while frame_number < max_frames:
           
            #training on Deterministic does not lead to reliable agents on random episodes
            env = gym.make('{0}-v4'.format(env_name))
            agent.main_dqn.train()
            epoch_frame = 0
            
            while epoch_frame < evaluate_every:
                terminal_life_lost = False
                state = env.reset()
                states_for_action.append(torch.FloatTensor(preprocess(state)/255.))
                episode_reward_sum = 0
                for _ in range(max_episode_length):
                    action = action_scheduler.get_action(frame_number, torch.stack(list(states_for_action)).to(DEVICE))
                    next_state, reward, terminal, info = env.step(action)

                    if info['ale.lives'] < last_lives:
                        terminal_life_lost = True
                    else:
                        terminal_life_lost = terminal
                    last_lives = info['ale.lives']

                    frame_number += 1
                    epoch_frame += 1
                    episode_reward_sum += reward
                    reward = clip_reward(reward)

                    agent.replay_memory.add_experience(action=action,
                                                       frame=preprocess(next_state),
                                                       reward=reward,
                                                       terminal=terminal_life_lost)
                                                       

                    state = next_state

                    if frame_number % update_every == 0 and frame_number > start_learning:
                        loss = agent.optimize()
                        losses.append(loss.item())

                    if frame_number % network_update_every == 0 and frame_number > start_learning:
                        agent.target_dqn.load_state_dict(agent.main_dqn.state_dict())

                    if terminal:
                        terminal = False
                        break
                rewards.append(episode_reward_sum)

                if len(rewards) % 10 == 0:
                    print(len(rewards), frame_number, np.mean(rewards[-100:]), np.mean(losses[-100:]))
                    with open('{0}/rewards.dat'.format(path), 'a') as reward_file:
                        print(len(rewards), frame_number, np.mean(rewards[-100:]), file=reward_file)


            torch.save(agent.main_dqn.state_dict(), "{0}/checkpoints/{3}_{1}_reward{2}.pth".format(path, frame_number, \
                                                                                np.mean(rewards[-100:]), env_name))
            with open('{}/logs/rewards.log'.format(path),'wb') as f: pickle.dump(rewards, f)
            terminal = True
            gif = True
            frames_for_gif = []
            eval_rewards = []
            evaluation_frame_number = 0

            env = gym.make('{0}-v4'.format(env_name))
            agent.main_dqn.eval()

            for i in range(eval_length):
                if terminal:
                    terminal_life_lost = True
                    state = env.reset()
                    episode_reward_sum = 0
                    terminal = False

                states_for_action.append(torch.FloatTensor(preprocess(state)/255.))
                starting_point = np.random.randint(0, 30)

                
                action = 1 if terminal_life_lost and i < starting_point \
                         else action_scheduler.get_action(frame_number, torch.stack(list(states_for_action)).to(DEVICE), evaluation=True)

                next_state, reward, terminal, info = env.step(action)

                evaluation_frame_number += 1
                episode_reward_sum += reward
                state = next_state
                if gif:
                    frames_for_gif.append(next_state)
                if terminal:
                    eval_rewards.append(episode_reward_sum)
                    gif = False

            print("Evaluation score:\n", np.mean(eval_rewards))
            
            try:
                generate_gif(frame_number, frames_for_gif, eval_rewards[0], '{0}/gifs/'.format(path))
                print("Gif saved.")
            except IndexError:
                print("No evaluation game finished")

            print("Model saved")
            frames_for_gif = []

    except KeyboardInterrupt:
        torch.save(agent.main_dqn.state_dict(), "{3}/checkpoints/{2}_{0}_reward{1}.pth".format(frame_number, int(np.mean(rewards[-100:])), env_name, path))
        with open('{}/logs/rewards.log'.format(path),'wb') as f: pickle.dump(rewards, f)
        print("Training stopped")


def test(agent, runs=1, render=False):
    env = gym.make('{0}-v4'.format(args.env))
    state = env.reset()
    done = False
    transition = deque(maxlen=4)
    eps_reward = 0
    total_reward = []

    i = 0
    while i < runs:
        transition.append(frame_to_tensor(preprocess(state)))

        if len(transition) < 4:
            action = env.action_space.sample()
        else:
            action = agent.predict_action(torch.stack(list(transition)).to(DEVICE).unsqueeze(0))
        next_state, reward, done, _ = env.step(action)

        if render:
            env.render()

        eps_reward += reward
        if done:
            env = gym.make('{0}-v4'.format(args.env))
            state = env.reset()
            total_reward.append(eps_reward)
            print("[{}] Total episode reward: {}".format(len(total_reward), eps_reward))
            eps_reward = 0
            i += 1
            done = False
        else:
            state = next_state
    
    print("Average reward over {} runs: {}+-{}".format(runs, np.average(total_reward), np.std(total_reward)))


def get_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help='commands')
    train_parser = subparsers.add_parser('train', help='Train an Atari agent or load a checkpoint.')
    train_parser.add_argument('--env', default='Pong', type=str, help='Name of the Atari game')
    train_parser.add_argument('--load', default=False, type=bool, help='Load a checkpoint and train from there')
    train_parser.add_argument('--lr', default=0.0000625, type=float, help='Set the learning rate')
    train_parser.add_argument('--start_learning', default=50000, type=float, help='Set the amount of frames after which the agent starts to learn.')
    train_parser.add_argument('--gamma', default=0.99, type=float, help='Set the rewards discount factor')
    train_parser.add_argument('--hidden', default=128, type=int, help='Set the hidden size of the agent')
    train_parser.add_argument('--update_target', default=10000, type=int, help='Set the amount of frames after which the target network is updated.')
    train_parser.add_argument('--batch_size', default=32, type=int, help='Set the batch size')
    train_parser.add_argument('--memory_size', default=1000000, type=int, help='Set the memory size')
    train_parser.add_argument('--transition_size', default=4, type=int, help='Set the amount of stacked frames that form a transition')
    train_parser.add_argument('--eps_initial', default=1.0, type=float, help='Set the initial epsilon for the action scheduler')
    train_parser.add_argument('--eps_final', default=0.1, type=float, help='Set the first final epsilon for the action scheduler')
    train_parser.add_argument('--eps_final_frame', default=0.01, type=float, help='Set the second epsilon for the action scheduler')
    train_parser.add_argument('--eps_annealing_frames', default=1000000, type=int, help='Set for how many frames epsilon should be reduced to the first final epsilon')
    train_parser.add_argument('--replay_memory_start_size', default=50000, type=int, help='Set the amount of frames for which epsilon stays at the initial value')
    train_parser.add_argument('--max_frames', default=30000000, type=int, help='Set the amount overall training frames')
    train_parser.add_argument('--save_path', default='../tests/', type=str, help='Set the saving directory')


    test_parser = subparsers.add_parser('test', help='Test a preveously trained model')
    test_parser.add_argument('--env', default='Pong', type=str, help='Name of the Atari game')
    test_parser.add_argument('-i', default=None, metavar='.pth file', type=argparse.FileType('r'), help='Path to the trained model')
    test_parser.add_argument('--hidden', default=128, type=int, help='Hyperparameter necessary to load the agent')
    test_parser.add_argument('--render', default=False, type=bool, help='Choose to render the game if there\'s a display available.')
    test_parser.add_argument('--runs', default=1, type=int, help='Set the amount of runs the agent is tested for.')
    return parser.parse_args()


if __name__ == "__main__":

    args = get_args()
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if 'lr' in args:
        print("Creating environment...")
        env = gym.make('{0}-v4'.format(args.env))
        rewards = []
        print("Initializing the agent...")
        agent = Agent(n_actions=env.action_space.n, device=DEVICE, hidden=args.hidden, 
                      learning_rate=args.lr, gamma=args.gamma, batch_size=args.batch_size, 
                      agent_history_length=args.transition_size, memory_size=args.memory_size)
        print("Preparing save dirs..")
        path = "{0}{1}".format(args.save_path, args.env)
        os.makedirs(path, exist_ok=True)
        os.makedirs("{0}/checkpoints/".format(path), exist_ok=True)
        os.makedirs("{0}/gifs/".format(path), exist_ok=True)
        os.makedirs("{0}/logs/".format(path), exist_ok=True)
        print("Created directories successfully!")
        if args.load:
            print("Search for model..")
            paths = glob.glob('{0}/checkpoints/*.pth'.format(path))
            if len(paths) > 0:
                latest_model = max(paths, key=os.path.getctime)
                frame_number = latest_model.split('_')[1]
                frame_number = int(frame_number.split('.')[0])
                agent.main_dqn.load_state_dict(torch.load(latest_model))
                agent.target_dqn.load_state_dict(torch.load(latest_model))
                agent.target_dqn.eval()
                args.start_learning = frame_number + args.start_learning
                try:
                    with open('{}/logs/rewards.log'.format(path),'rb') as f: rewards = pickle.load(f)
                except:
                    print("Failed loading rewards from log.")
                    rewards = []
            else:
                "No checkpoint found."
                frame_number = 0
                rewards = []
        else:
            frame_number = 0
        print("Start training at frame number {0}".format(frame_number))
        train(n_actions=env.action_space.n, agent=agent, env=env, env_name=args.env, path=path, frame_number=frame_number, start_learning=args.start_learning, network_update_every=args.update_target, rewards=rewards)

    if 'render' in args:
        print("Creating environment...")
        env = gym.make('{0}-v4'.format(args.env))
        print("Load agent...")
        agent = DQN(n_actions=env.action_space.n, hidden=args.hidden)
        agent.load_state_dict(torch.load(str(args.i.name)))
        agent = agent.to(DEVICE)
        agent.eval()
        print("Start!")
        test(agent=agent, runs=args.runs, render=args.render)