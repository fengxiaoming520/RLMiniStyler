from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

from . import nets
from .function import adjust_learning_rate, calc_mean_std, hard_update, soft_update
from . import AutomaticWeightedLoss


class Agent(object):
    def __init__(self, args, extranet, device):
        self.args = args
        self.device = device
        self.tau = args.tau
        self.gamma = args.gamma
        self.target_entropy = -torch.prod(torch.Tensor([64, 64, 64]).to(self.device)).item()
        self.scale = 1.  
        self.log_alpha = torch.tensor(0.0).to(device).requires_grad_(True)
        self.mse_loss = nn.MSELoss().to(device)
        self.awl = AutomaticWeightedLoss(3)

        self.extranet = extranet
        self.actor = nets.Actor().to(device)
        self.decoder = nets.Decoder().to(device)

        self.critic1 = nets.QNetwork().to(device)
        self.critic2 = nets.QNetwork().to(device)
        self.critic1_target = nets.QNetwork().to(device)
        self.critic2_target = nets.QNetwork().to(device)

        self.eval(self.critic1_target)
        self.eval(self.critic2_target)
        hard_update(self.critic1_target, self.critic1)
        hard_update(self.critic2_target, self.critic2)

        self.optim_alpha = torch.optim.Adam([self.log_alpha], lr=args.lr, betas=(0.5, 0.9))
        self.optim_actor = torch.optim.Adam([
            {'params': self.actor.parameters()},
        ], lr=args.lr/10., betas=(0.5, 0.9))

        self.optim_decoder = torch.optim.Adam([
            {'params': self.decoder.parameters()},
        ], lr=args.lr, betas=(0.5, 0.9))

        self.optim_awl = torch.optim.Adam([
            {'params': self.awl.parameters(), 'weight_decay': 0}
        ], lr=args.lr, betas=(0.5, 0.9))

        self.optim_critic1 = torch.optim.Adam([
            {'params': self.critic1.parameters()},
        ], lr=args.lr, betas=(0.5, 0.9))
        self.optim_critic2 = torch.optim.Adam([
            {'params': self.critic2.parameters()},
        ], lr=args.lr, betas=(0.5, 0.9))

        pactor = sum(p.numel() for p in self.actor.parameters() if p.requires_grad)
        pexecutor = sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)
        pcritic = sum(p.numel() for p in self.critic1.parameters() if p.requires_grad)
        print('total parameters: {}'.format(pactor + pexecutor))
        print('total critic parameters: {}'.format(pcritic))

    @property
    def alpha(self):
        return torch.exp(self.log_alpha)

    def get_value(self, state, action):
        self.critic1.eval()
        self.critic2.eval()
        v1 = self.critic1(state, action)
        v2 = self.critic2(state, action)
        return torch.min(v1, v2).mean().item()

    def select_action(self, state, evaluate=False):
        self.actor.eval()
        self.decoder.eval()
        action, log_pi, mean, encs = self.actor.sample(state['moving'], state['style'])
        if evaluate:
            action = mean
        action_moving = self.decoder(action, encs)
        actions = {'action': action.detach(),
                   'action_moving': action_moving.detach(),
                   }

        return actions

    def optimize(self, memory, batch_size, updates, writer, alpha=1.0):
        self.actor.train()
        self.decoder.train()
        self.critic1.train()
        self.critic2.train()

        self._loss(memory, batch_size, updates, writer, alpha)

        soft_update(self.critic1_target, self.critic1, self.tau)
        soft_update(self.critic2_target, self.critic2, self.tau)

    def _loss(self, memory, batch_size, updates, writer, alpha):

        state_m_batch, state_s_batch, action_batch, reward_batch, next_state_m_batch, mask_batch = memory.sample(
            batch_size=batch_size)
        state_m_batch = torch.FloatTensor(state_m_batch).to(self.device)
        state_s_batch = torch.FloatTensor(state_s_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        next_state_m_batch = torch.FloatTensor(next_state_m_batch).to(self.device)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        state_m_batch_feats_vgg = self.extranet(state_m_batch)
        state_s_batch_feats_vgg = self.extranet(state_s_batch)

        # ######################--------------- update critic -----------#################################
        with torch.no_grad():
            next_moving_action, next_moving_action_log_pi, _, _ = self.actor.sample(next_state_m_batch, state_s_batch)

            qf1_next_target = self.critic1_target(next_state_m_batch, next_moving_action)
            qf2_next_target = self.critic2_target(next_state_m_batch, next_moving_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_moving_action_log_pi
            next_q_value = reward_batch.unsqueeze(-1).unsqueeze(-1) + mask_batch.unsqueeze(-1).unsqueeze(-1) * self.gamma * min_qf_next_target

        qf1 = self.critic1(state_m_batch, action_batch)  
        qf2 = self.critic2(state_m_batch, action_batch)  
        qf1_loss = F.mse_loss(qf1, next_q_value)  
        qf2_loss = F.mse_loss(qf2, next_q_value)  

        self.optim_critic1.zero_grad()
        qf1_loss.backward()
        self.optim_critic1.step()
        self.optim_critic2.zero_grad()
        qf2_loss.backward()
        self.optim_critic2.step()
        writer.add_scalar('loss/critic_1', qf1_loss.item(), updates + 1)
        writer.add_scalar('loss/critic_2', qf2_loss.item(), updates + 1)

        # #################------------ update actor -----------#########################
        if updates % self.args.actor_update_interval == 0:
            action, log_pi, _, _ = self.actor.sample(state_m_batch, state_s_batch)
            # ------------------------- update alpha -----------------------------------
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.optim_alpha.zero_grad()
            alpha_loss.backward()
            self.optim_alpha.step()

            alpha_tlogs = self.alpha.clone()  # For TensorboardX logs
            writer.add_scalar('loss/entropy_loss', alpha_loss.item(), updates)
            writer.add_scalar('entropy_temperature/alpha', alpha_tlogs.item(), updates)

            # #############---------- actor ----------###################
            qf1_pi = self.critic1(state_m_batch, action)
            qf2_pi = self.critic2(state_m_batch, action)
            actor_Q = torch.min(qf1_pi, qf2_pi)
            actor_loss = (self.alpha * log_pi * self.scale - actor_Q).mean()
            self.optim_actor.zero_grad()
            actor_loss.backward()
            self.optim_actor.step()

            writer.add_scalar('loss/actor', actor_loss.item(), updates)

        # #################------------ update vae -----------#######################
        if updates % self.args.vae_update_interval == 0:
            latent, log_pi, _, encs = self.actor.sample(state_m_batch, state_s_batch)
            action_moving = self.decoder(latent, encs, alpha)
            next_state_moving = action_moving
            
            next_state_moving_vgg = self.extranet(next_state_moving)
            
            loss_c = self.calc_content_loss(next_state_moving_vgg[-2], state_m_batch_feats_vgg[-2])
            loss_s = self.calc_style_loss(next_state_moving_vgg[0], state_s_batch_feats_vgg[0])
            for i in range(1, 4):
                loss_s = loss_s + self.calc_style_loss(next_state_moving_vgg[i], state_s_batch_feats_vgg[i])

            res_style_feats = self.actor.encoder(next_state_moving, style_mode=True)
            res_filter_weights, res_filter_biases = self.actor.modulator(res_style_feats)
            style_feats, filter_weights, filter_biases = encs[1], encs[2], encs[3]

            # style signal contrastive loss
            loss_contrastive = 0.
            for i in range(int(state_s_batch.size(0))):
                pos_loss = 0.
                neg_loss = 0.

                for j in range(int(state_s_batch.size(0))):
                    if j == i:
                        FeatMod_loss = self.calc_style_loss(res_style_feats[0][i].unsqueeze(0), style_feats[0][j].unsqueeze(0)) + \
                                       self.calc_style_loss(res_style_feats[1][i].unsqueeze(0), style_feats[1][j].unsqueeze(0)) + \
                                       self.calc_style_loss(res_style_feats[2][i].unsqueeze(0), style_feats[2][j].unsqueeze(0)) + \
                                       self.calc_style_loss(res_style_feats[3][i].unsqueeze(0), style_feats[3][j].unsqueeze(0))
                        FilterMod_loss = self.calc_content_loss(res_filter_weights[0][i], filter_weights[0][j]) + \
                                         self.calc_content_loss(res_filter_weights[1][i], filter_weights[1][j]) + \
                                         self.calc_content_loss(res_filter_biases[0][i], filter_biases[0][j]) + \
                                         self.calc_content_loss(res_filter_biases[1][i], filter_biases[1][j]) + \
                                         self.calc_content_loss(res_style_feats[4][i], style_feats[4][j]) + \
                                         self.calc_content_loss(res_style_feats[5][i], style_feats[5][j]) + \
                                         self.calc_content_loss(res_style_feats[6][i], style_feats[6][j]) + \
                                         self.calc_content_loss(res_style_feats[7][i], style_feats[7][j])
                        pos_loss = FeatMod_loss + FilterMod_loss
                    else:
                        FeatMod_loss = self.calc_style_loss(res_style_feats[0][i].unsqueeze(0),
                                                            style_feats[0][j].unsqueeze(0)) + \
                                       self.calc_style_loss(res_style_feats[1][i].unsqueeze(0),
                                                            style_feats[1][j].unsqueeze(0)) + \
                                       self.calc_style_loss(res_style_feats[2][i].unsqueeze(0),
                                                            style_feats[2][j].unsqueeze(0)) + \
                                       self.calc_style_loss(res_style_feats[3][i].unsqueeze(0),
                                                            style_feats[3][j].unsqueeze(0))
                        FilterMod_loss = self.calc_content_loss(res_filter_weights[0][i], filter_weights[0][j]) + \
                                         self.calc_content_loss(res_filter_weights[1][i], filter_weights[1][j]) + \
                                         self.calc_content_loss(res_filter_biases[0][i], filter_biases[0][j]) + \
                                         self.calc_content_loss(res_filter_biases[1][i], filter_biases[1][j]) + \
                                         self.calc_content_loss(res_style_feats[4][i], style_feats[4][j]) + \
                                         self.calc_content_loss(res_style_feats[5][i], style_feats[5][j]) + \
                                         self.calc_content_loss(res_style_feats[6][i], style_feats[6][j]) + \
                                         self.calc_content_loss(res_style_feats[7][i], style_feats[7][j])
                        neg_loss = neg_loss + FeatMod_loss + FilterMod_loss

                loss_contrastive = loss_contrastive + pos_loss / neg_loss

            loss = self.awl(loss_c, loss_s, loss_contrastive)

            self.optim_awl.zero_grad()
            self.optim_actor.zero_grad()
            self.optim_decoder.zero_grad()
            loss.backward()
            self.optim_actor.step()
            self.optim_decoder.step()
            self.optim_awl.step()

            writer.add_scalar('loss/loss_content', loss_c.item(), updates + 1)
            writer.add_scalar('loss/loss_style', loss_s.item(), updates + 1)
            writer.add_scalar('loss/loss_contrast', loss_contrastive.item(), updates + 1)
            writer.add_scalar('param/content_weight', self.awl.params[0].item(), updates + 1)
            writer.add_scalar('param/style_weight', self.awl.params[1].item(), updates + 1)
            writer.add_scalar('param/contrast_weight', self.awl.params[2].item(), updates + 1)

            # save intermediate samples
            output_dir = Path(self.args.sample_path)
            output_dir.mkdir(exist_ok=True, parents=True)
            print('updates:', updates + 1, 'vae loss:', loss.item())
            if (updates + 1) % 50 == 0:
                visualized_imgs = torch.cat([state_m_batch, state_s_batch, next_state_moving])

                output_name = output_dir / 'output{}.jpg'.format(str(updates + 1).zfill(6))
                save_image(visualized_imgs, str(output_name), nrow=self.args.sample_batch_size)
                print('[%d/%d] loss_content:%.4f, loss_style:%.4f, loss_contrastive:%.4f' \
                       % (updates+1, self.args.max_iter, loss_c.item(), loss_s.item(), loss_contrastive.item()))
                ############################################################################

    def eval(self, model):
        for param in model.parameters():
            param.requires_grad = False

    def train(self, model):
        for param in model.parameters():
            param.requires_grad = True

    def calc_content_loss(self, input, target):
        assert (input.size() == target.size())
        return self.mse_loss(input, target)

    def calc_style_loss(self, input, target):
        assert (input.size() == target.size())
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)

