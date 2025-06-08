import os
import torch
import torch.nn as nn
import torchvision.utils as vutils
from .function import calc_style_loss


class Env(object):

    def __init__(self, args, content_iter, style_iter, extranet, device='cpu'):
        self.max_episode_steps = args.max_episode_steps
        self.args = args
        self.device = device
        self.generator = self._generator(content_iter, style_iter)
        self.extranet = extranet
        self.reset()

    def _generator(self, content_iter, style_iter):
        self.epoch = 0
        while True:
            self.epoch += 1

            for content, style in zip(content_iter, style_iter):
                yield content, style

    def reset(self):
        content, style = next(self.generator)
        self.content = content.to(self.device)
        self.style = style.to(self.device)
        self.moving = torch.clone(self.content)
        self.style_feats = self.extranet(self.style)
        self.global_step = 1
        self.prev_score = self.score()

        return self.state(), self.prev_score

    def state(self):
        s = {'moving': self.moving,
             'style': self.style,
             }
        return s

    def score(self):
        self.moving_feats = self.extranet(self.moving)
        loss = calc_style_loss(self.moving_feats[0], self.style_feats[0])
        for i in range(1, 4):
            loss += calc_style_loss(self.moving_feats[i], self.style_feats[i])
        if self.args.score_c == True:
            loss += calc_style_loss(self.moving_feats[-2], self.style_feats[-2])

        return -loss.cpu().detach().item()

    def done(self):
        return self.score() > self.args.score_threshold

    def step(self, prediction):
        self.moving = prediction
        score = self.score()

        image_score = self.score()
        reward = score

        if self.done():
            reward += 10
        return self.state(), reward, self.done(), image_score

    def save_init(self, dir):
        # print(self.source.data.shape)
        if not os.path.exists(dir):
            os.makedirs(dir)
        vutils.save_image(self.content, dir + '/content.png')
        vutils.save_image(self.style, dir + '/style.png')

    def save_process(self, idx, dir):
        if not os.path.exists(dir):
            os.makedirs(dir)
        vutils.save_image(self.moving, dir + '/moving-{}.png'.format(str(idx).zfill(2)))
