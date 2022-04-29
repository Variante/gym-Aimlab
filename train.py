import copy
import math
import os
import pickle as pkl
import sys
import time
import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from logger import Logger
from replay_buffer import ReplayBuffer
from video import VideoRecorder
from drq import *

from aimlabgym import *

from tkinter import *
import tkinter.font as tkFont
from PIL import Image, ImageDraw, ImageFont, ImageTk


torch.backends.cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser()
    # environment
    parser.add_argument('--pre_image_size_online', default=100, type=int)
    parser.add_argument('--pre_image_size_target', default=100, type=int)

    parser.add_argument('--image_size', default=84, type=int)
    # parser.add_argument('--action_repeat', default=4, type=int)
    parser.add_argument('--frame_stack', default=3, type=int)
    # replay buffer
    parser.add_argument('--replay_buffer_capacity', default=100000, type=int)
    # train
    parser.add_argument('--agent', default='drq', type=str)
    parser.add_argument('--init_steps', default=1000, type=int) # 1000
    parser.add_argument('--num_train_steps', default=200000, type=int)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--hidden_dim', default=1024, type=int)
    # eval
    parser.add_argument('--eval_freq', default=10000, type=int) # 10000
    parser.add_argument('--num_eval_episodes', default=10, type=int) # 10
    # critic
    parser.add_argument('--critic_lr', default=1e-3, type=float)
    parser.add_argument('--critic_beta', default=0.9, type=float)
    parser.add_argument('--critic_tau', default=0.01, type=float) # try 0.05 or 0.1
    parser.add_argument('--critic_target_update_freq', default=2, type=int) # try to change it to 1 and retain 0.01 above
    # actor
    parser.add_argument('--actor_lr', default=1e-3, type=float)
    parser.add_argument('--actor_beta', default=0.9, type=float)
    parser.add_argument('--actor_log_std_min', default=-10, type=float)
    parser.add_argument('--actor_log_std_max', default=2, type=float)
    parser.add_argument('--actor_update_freq', default=2, type=int)
    # encoder
    parser.add_argument('--encoder_feature_dim', default=50, type=int)
    parser.add_argument('--encoder_lr', default=1e-3, type=float)
    parser.add_argument('--encoder_tau', default=0.05, type=float)
    parser.add_argument('--num_layers', default=4, type=int)
    parser.add_argument('--num_filters', default=32, type=int)
    # sac
    parser.add_argument('--discount', default=0.99, type=float)
    parser.add_argument('--init_temperature', default=0.1, type=float)
    parser.add_argument('--alpha_lr', default=1e-4, type=float)
    parser.add_argument('--alpha_beta', default=0.5, type=float)
    # misc
    parser.add_argument('--seed', default=-1, type=int)
    parser.add_argument('--work_dir', default='../main_results/', type=str)
    parser.add_argument('--save_tb', default=True, action='store_true')
    parser.add_argument('--save_buffer', default=True, action='store_true')
    parser.add_argument('--save_video', default=False, action='store_true')
    parser.add_argument('--save_model', default=True, action='store_true')

    parser.add_argument('--log_interval', default=100, type=int)
    parser.add_argument('--device', default='cuda', type=str)

    args = parser.parse_args()
    return args

def make_agent(obs_shape, action_shape, args, device):
    if args.agent == 'drq':
        return DRQAgent(
            obs_shape=obs_shape, 
            action_shape=action_shape, 
            action_range=[-1.0, 1.0], 
            device=device,
            hidden_dim=args.hidden_dim,
            feature_dim=args.encoder_feature_dim,
            discount=args.discount,
            init_temperature=args.init_temperature, 
            lr=args.critic_lr, 
            actor_update_frequency=args.actor_update_freq, 
            critic_tau=args.critic_tau,
            critic_target_update_frequency=args.critic_target_update_freq, 
            batch_size=args.batch_size,
        )
    else:
        assert f'agent is not supported: {args.agent}'


def main():
    
    def flat_obs(obs, next_obs=None):
        num_of_imgs = obs.shape[0] // 3
        imgs = [obs[i * 3: (i + 1) * 3] for i in range(num_of_imgs)]
        if next_obs is not None:
            imgs.append(next_obs[-3:])
        imgs = [np.moveaxis(i, 0, -1)[:,:,::-1] for i in imgs]
        return np.hstack(imgs)
    
    train_step = 0
    last_step = 0
    loss_str = ''
    
    def train():
        nonlocal last_step
        nonlocal train_step
        nonlocal loss_str
        # run training update
        if step >= args.init_steps:
            num_updates = step - last_step
            for i in range(num_updates):
                agent.update(replay_buffer, L, train_step + args.init_steps)
                # GUI
                ldtag.configure(text=f'Agent: {process_title} Mode: Train')
                ldres.configure(text=f'Step:{step} | EP: {episode} | ES:{episode_step} | RB: {replay_buffer.idx} / {len(replay_buffer)}')
                ldres2.configure(text=f'[{last_step}] - ({last_step + i}) -> [{step}]')
                root.update()
                # GUI
                train_step += 1
            last_step = step
            search_done = True

    def evaluate(env, agent, video, num_episodes, L, step, args):
        all_ep_rewards = []

        def run_eval_loop(sample_stochastically=True):
            start_time = time.time()
            prefix = 'stochastic_' if sample_stochastically else ''
            for i in range(num_episodes):
                obs = env.reset()
                video.init(enabled=True)
                done = False
                episode_reward = 0
                while not done:
                    # center crop image
                    with utils.eval_mode(agent):
                        action = agent.act(obs, sample=sample_stochastically)


                    next_obs, reward, done, res = env.step(action)
                    
                    # GUI
                    img = Image.fromarray(flat_obs(obs, next_obs))
                    imgtk = ImageTk.PhotoImage(image=img)
                    lmain.imgtk = imgtk
                    lmain.configure(image=imgtk)
                    ldtag.configure(text=f'Agent: {process_title} Mode: Eval')
                    ldres.configure(text=f'Eval {i + 1}/{num_episodes} | ES:{step} | ER: {episode_reward:.3f}')
                    ldres2.configure(text=f'pts: {res["pts"]} | Action: x: {action[0]:.3f} y: {action[1]:.3f} s: {action[2]:.3f}')
                    root.update()
                    video.record(env)
                    episode_reward += reward
                    obs = next_obs

                video.save(f'Step{step}_T{i:02d}.mp4')
                L.log('eval/' + prefix + 'episode_reward', episode_reward, step)
                all_ep_rewards.append(episode_reward)

            L.log('eval/' + prefix + 'eval_time', time.time()-start_time , step)
            mean_ep_reward = np.mean(all_ep_rewards)
            best_ep_reward = np.max(all_ep_rewards)
            L.log('eval/' + prefix + 'mean_episode_reward', mean_ep_reward, step)
            L.log('eval/' + prefix + 'best_episode_reward', best_ep_reward, step)
            print(f'Evaluation reward: mean: {mean_ep_reward:.1f} std:{np.std(all_ep_rewards):.1f}')
            return mean_ep_reward

        mean_ep_reward = run_eval_loop(sample_stochastically=False)
        L.dump(step)
        return mean_ep_reward


    args = parse_args()

    if args.seed == -1:
        args.seed = np.random.randint(1,1000000)

    utils.set_seed_everywhere(args.seed)

    pre_transform_image_size = max(args.pre_image_size_online, args.pre_image_size_target)

    cfg = load_cfg()
    cfg['image_size'] = pre_transform_image_size
    env = AimlabGym(cfg)
    env.seed(args.seed)
    action_shape = env.action_space.shape
    
    # stack several consecutive frames together
    env = utils.FrameStack(env, k=args.frame_stack)

    # make directory
    ts = time.gmtime()
    ts = time.strftime("%m_%d", ts)
    env_name = f'Aimlab-Grid'

    process_title = 'DrQ-1.3'

    # setproctitle.setproctitle(f"python {process_title}-{env_name}-{ts}")

    def gen_mag_str(mag):
        return f'{mag[0]:02d}-'+ ('-'.join([f'{i:.2f}' for i in mag[1:]]))

    exp_name = f'{env_name}-{ts}-Oim{args.pre_image_size_online}-Tim{args.pre_image_size_target}-b{args.batch_size}-s{args.seed}'

    args.work_dir = f'{args.work_dir}/{process_title}-{exp_name}'

    utils.make_dir(args.work_dir)
    print(args.work_dir)

    if args.save_video:
        video_dir = utils.make_dir(os.path.join(args.work_dir, 'video'))
    if args.save_model:
        model_dir = utils.make_dir(os.path.join(args.work_dir, 'model'))
    if args.save_buffer:
        buffer_dir = utils.make_dir(os.path.join(args.work_dir, 'buffer'))

    video = VideoRecorder(video_dir if args.save_video else None)

    with open(os.path.join(args.work_dir, 'args.json'), 'w') as f:
        arg_json = vars(args)
        json.dump(arg_json, f, sort_keys=True, indent=4)

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(args.device)

    

    obs_shape = env.reset().shape
    print(obs_shape)

    replay_buffer = ReplayBuffer(
        obs_shape=obs_shape,
        action_shape=action_shape,
        capacity=args.replay_buffer_capacity,
        image_pad=0,
        device=device,
    )

    agent = make_agent(
        obs_shape=(obs_shape[0], args.image_size, args.image_size),
        action_shape=action_shape,
        args=args,
        device=device
    )

    L = Logger(args.work_dir, save_tb=args.save_tb)

    episode, episode_reward, done = 0, 0, True
    start_time = time.time()

    print("Start loop")
    
    # Windows
    root = Tk()
    # Create a frame
    app = Frame(root)
    app.pack()

    # Create a label in the frame
    lmain = Label(app)
    lmain.pack()
    
    ldtag = Label(app, width=805, font=tkFont.Font(size=15, weight=tkFont.BOLD))
    ldtag.pack()
    ldres = Message(app, width=805,  font=tkFont.Font(size=15, weight=tkFont.BOLD))
    ldres.pack()
    ldres2 = Message(app, width=805, font=tkFont.Font(size=15, weight=tkFont.NORMAL))
    ldres2.pack()
    
    root.geometry(f"1010x300+{0}+{730}")
    
    search_done = False
    
    for step in range(args.num_train_steps):
        try:
            if step % args.eval_freq == 0 and step != 0:
                L.log('eval/episode', episode, step)
                # env.calibrate()
                train()
                evaluate(env, agent, video, args.num_eval_episodes, L, step,args)
                if args.save_model:
                    agent.save(model_dir, step)
                if args.save_buffer:
                    replay_buffer.save(buffer_dir)
                    
            if done:
                train()
                    
                if step > 0:
                    # if step % args.log_interval == 0:
                    L.log('train/duration', time.time() - start_time, step)
                    L.dump(step)
                    start_time = time.time()
                # if step % args.log_interval == 0:
                L.log('train/episode_reward', episode_reward, step)

                obs = env.reset()
                done = False
                episode_reward = 0
                episode_step = 0
                episode += 1
                # if step % args.log_interval == 0:
                L.log('train/episode', episode, step)
            
            

            # sample action for data collection
            if step < args.init_steps:
                action = env.action_space.sample()
            else:
                if not search_done:
                    search_done = True
                    train()
                    env.reset()
                    continue
                with utils.eval_mode(agent):
                    action = agent.act(obs,sample=True)
            # print("Action: ", action)
            next_obs, reward, done, res = env.step(action)
            
            if done:
                # avoid summary page
                next_obs[-3:] = next_obs[-6:-3]
            
            # GUI
            img = Image.fromarray(flat_obs(obs, next_obs))
            imgtk = ImageTk.PhotoImage(image=img)
            lmain.imgtk = imgtk
            lmain.configure(image=imgtk)
            ldtag.configure(text=f'Agent: {process_title} Mode: Explore')
            ldres.configure(text=f'Step:{step} | EP: {episode} | ES:{episode_step} | RB: {replay_buffer.idx} / {len(replay_buffer)}')
            ldres2.configure(text=f'pts: {res["pts"]} | Action: x: {action[0]:.3f} y: {action[1]:.3f} s: {action[2]:.3f}')
            root.update()

            # allow infinit bootstrap
            done = float(done)
            done_no_max = 0
            episode_reward += reward

            replay_buffer.add(obs, action, reward, next_obs, done,
                                   done_no_max)

            obs = next_obs
            episode_step += 1
            
            
        except KeyboardInterrupt:
            break

    # evaluate and save results
    # res = evaluate(env, agent, video, args.num_eval_episodes, L, step,args)
    agent.save(model_dir, step)
    root.quit()
    return res


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    main()