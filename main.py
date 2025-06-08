import argparse
import datetime
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data as data
from PIL import Image, ImageFile
from tensorboardX import SummaryWriter
from torchvision import transforms
from tqdm import tqdm
from torchvision.utils import save_image

from rl.dataset import InfiniteSamplerWrapper, train_transform, FlatFolderDataset
from rl.function import fix_seed
from rl.agent import Agent
from rl.env import Env
from rl.replay_memory import ReplayMemory
from metrics.extranet import ExtraVGG19


cudnn.benchmark = True
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
# Disable OSError: image file is truncated
ImageFile.LOAD_TRUNCATED_IMAGES = True


def main(args, writer, device):
    extranet = ExtraVGG19(args.vgg).to(device)

    agent = Agent(args, extranet, device)

    content_tf = train_transform()
    style_tf = train_transform()
    content_dataset = FlatFolderDataset(args.content_dir, content_tf)
    style_dataset = FlatFolderDataset(args.style_dir, style_tf)

    content_iter = iter(data.DataLoader(
        content_dataset, batch_size=args.batch_size,
        sampler=InfiniteSamplerWrapper(content_dataset),
        num_workers=args.n_threads))
    style_iter = iter(data.DataLoader(
        style_dataset, batch_size=args.batch_size,
        sampler=InfiniteSamplerWrapper(style_dataset),
        num_workers=args.n_threads))

    env = Env(args, content_iter, style_iter, extranet, device)

    # Memory
    memory = ReplayMemory(args.replay_size, args.seed)

    # training
    total_numsteps = 0
    updates = 0
    start_episode = -1
    for i_episode in tqdm(range(start_episode + 1, args.max_global_episode)):
        episode_reward = 0
        episode_steps = 0
        finish_episode = False
        state, init_score = env.reset()

        tic = time.time()
        while not finish_episode:
            if i_episode % 10 == 0:
                env.save_init(args.process)
            actions = agent.select_action(state)
            if len(memory) > args.sample_batch_size and args.start_episode < i_episode:  
                # Number of updates per step in environment
                for i in range(args.updates_per_step):
                    # Update parameters of all the networks
                    agent.optimize(memory, args.sample_batch_size, updates, writer, alpha=1.0)
                    updates += 1

            next_state, reward, done, _ = env.step(actions['action_moving']) 
            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward

            mask = 1 if episode_steps == env.max_episode_steps else float(not done) 
            finish_episode = True if episode_steps == env.max_episode_steps or done else False

            memory.push(
                state['moving'].detach().cpu().numpy()[0],
                state['style'].detach().cpu().numpy()[0],
                actions['action'].detach().cpu().numpy()[0],
                reward,
                next_state['moving'].detach().cpu().numpy()[0],
                mask)  # Append transition to memory

            if episode_steps <= env.max_episode_steps and i_episode % 10 == 0:
                env.save_process(episode_steps, args.process)
            state = next_state
        ############################################################################
        toc = time.time()
        writer.add_scalar('reward/train', episode_reward, i_episode)
        print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}, time: {}".format(i_episode, total_numsteps,
                                                                                      episode_steps,
                                                                                      round(episode_reward, 2),
                                                                                      round(toc-tic, 2)))

        if (i_episode + 1) % args.save_model_interval == 0 or (i_episode + 1) == args.max_iter:
            state_dict = agent.actor.state_dict()
            for key in state_dict.keys():
                state_dict[key] = state_dict[key].to(torch.device('cpu'))
            torch.save(state_dict, checkpoints_dir /
                       'actor_iter_{:d}.pth.tar'.format(i_episode + 1))

            state_dict = agent.decoder.state_dict()
            for key in state_dict.keys():
                state_dict[key] = state_dict[key].to(torch.device('cpu'))
            torch.save(state_dict, checkpoints_dir /
                       'decoder_iter_{:d}.pth.tar'.format(i_episode + 1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RLMiniStyler")
    parser.add_argument('--gpu_id', type=int, default=0)
    # Basic options
    parser.add_argument('--content_dir', type=str,
                        help='Directory path to a batch of content images',
                        default='../../../../Datasets/MS-COCO-2014/train2014')
    parser.add_argument('--style_dir', type=str,
                        help='Directory path to a batch of style images',
                        default='../../../../Datasets/WikiArt/train')

    parser.add_argument('--vgg', type=str, default='./metrics/vgg_normalised.pth')

    # training options
    parser.add_argument('--exp_dir', default='./exp',
                        help='Directory to save the exp')
    parser.add_argument('--ckpt_dir', default='./exp/ckpt',
                        help='Directory to save the models')
    parser.add_argument('--log_dir', default='./exp/logs',
                        help='Directory to save the log')
    parser.add_argument('--process', default='./exp/process',
                        help='Directory to save the process image')
    parser.add_argument('--sample_path', type=str, default='./exp/samples',
                        help='Derectory to save the intermediate samples')

    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--lr_decay', type=float, default=5e-5)
    parser.add_argument('--max_iter', type=int, default=1600000)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--sample_batch_size', type=int, default=8)

    parser.add_argument('--style_weight', type=float, default=3.0)
    parser.add_argument('--content_weight', type=float, default=10.0)
    parser.add_argument('--SSC_weight', type=float, default=3.0)
    parser.add_argument('--n_threads', type=int, default=8)
    parser.add_argument('--save_model_interval', type=int, default=1000)
    parser.add_argument('--resume', type=bool, default=False, help='train the model from the checkpoint')

    parser.add_argument('--max_episode_steps', type=int, default=10, metavar='N',
                        help='maximum number of steps each episode (default: 10)')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                        help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--replay_size', type=int, default=5000, metavar='N',
                        help='size of replay buffer (default: 1000000)')
    parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                        help='model updates per simulator step (default: 1)')
    parser.add_argument('--seed', type=int, default=123456, metavar='N',
                        help='random seed (default: 123456)')
    parser.add_argument('--score_threshold', type=float, default=0, metavar='N',
                        help='score threshold (default: 0)')
    parser.add_argument('--score_c', type=bool, default=False, metavar='N',
                        help='score content (default: False)')

    parser.add_argument('--max_global_episode', type=int, default=1000001, metavar='N',
                        help='maximum number of episode (default: 1000000)')
    parser.add_argument('--start_episode', type=int, default=10, metavar='N',
                        help='Steps sampling random actions (default: 5)')

    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--actor_update_interval', type=int, default=4, metavar='N',
                        help='actor update per no. of updates per step (default: 16)')
    parser.add_argument('--vae_update_interval', type=int, default=1, metavar='N',
                        help='VAE update per no. of updates per step (default: 1)')
    args = parser.parse_args()

    fix_seed(args.seed)

    device = torch.device('cuda:%d' % args.gpu_id)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(exist_ok=True, parents=True)
    checkpoints_dir = Path(args.ckpt_dir)
    checkpoints_dir.mkdir(exist_ok=True, parents=True)
    process_dir = Path(args.process)
    process_dir.mkdir(exist_ok=True, parents=True)

    # Tensorboard
    writer = SummaryWriter(log_dir='{}/{}'.format(
        args.log_dir,
        datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    ))

    main(args, writer, device)
        
    writer.close()

