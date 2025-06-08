import argparse
from pathlib import Path

import time

import thop
import torch
from PIL import Image
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image

import numpy as np
import traceback

import rl.nets as net


class TestNet(nn.Module):
    def __init__(self, actor, decoder):
        super(TestNet, self).__init__()

        self.actor = actor
        self.decoder = decoder

    def forward(self, content, style, alpha=1.0):
        assert 0 <= alpha <= 1

        action, log_pi, mean, encs = self.actor.sample(content, style)
        res = self.decoder(action, encs)

        return res


def test_transform(size, crop=True):
    transform_list = []

    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))

    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform


parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content', type=str, default=r"RLMiniStyler\figures\1K\imgs-eval\content\building_clock_chapel_1228225_3840x2160.jpg",
                    help='File path to the content image')
parser.add_argument('--style', type=str, default=r"RLMiniStyler\figures\1K\imgs-eval\style\1920x1200-882036-picture-the-urban-landscape-Claude-Monet-Bridge.jpg",
                    help='File path to the style image')
parser.add_argument('--content_dir', type=str, default=None,
                    help='Directory path to a batch of content images')
parser.add_argument('--style_dir', type=str, default=None,
                    help='Directory path to a batch of style images')
parser.add_argument('--actor', type=str, default=r'RLMiniStyler\exp\ckpt\actor.pth.tar')
parser.add_argument('--decoder', type=str, default=r'RLMiniStyler\exp\ckpt\builder.pth.tar')
parser.add_argument('--output', type=str, default=r'RLMiniStyler\outputs\eval\1K',
                    help='Directory to save the output image(s)')
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--steps', type=int, default=10)

# Additional options
parser.add_argument('--content_size', type=int, default=0,
                    help='New (minimum) size for the content image, \
                    keeping the original size if set to 0')
parser.add_argument('--style_size', type=int, default=0,
                    help='New (minimum) size for the style image, \
                    keeping the original size if set to 0')
parser.add_argument('--crop', default=False,
                    help='do center crop to create squared image')
parser.add_argument('--save_ext', default='.jpg',
                    help='The extension name of the output image')

# Advanced options
parser.add_argument('--alpha', type=float, default=1.0,
                    help='The weight that controls the degree of \
                             stylization. Should be between 0 and 1')

args = parser.parse_args()

if args.gpu_id != -1:
    device = torch.device('cuda:%d' % args.gpu_id)
else:
    device = torch.device('cpu')

output_dir = Path(args.output)
output_dir.mkdir(exist_ok=True, parents=True)

# Either --content or --contentDir should be given.
assert (args.content or args.content_dir)
if args.content:
    content_paths = [Path(args.content)]
else:
    content_dir = Path(args.content_dir)
    content_paths = [f for f in content_dir.glob('*')]

# Either --style or --styleDir should be given.
assert (args.style or args.style_dir)
if args.style:
    style_paths = [Path(args.style)]
else:
    style_dir = Path(args.style_dir)
    style_paths = [f for f in style_dir.glob('*')]

actor = net.Actor()
decoder = net.Decoder()

actor.load_state_dict(torch.load(args.actor))
decoder.load_state_dict(torch.load(args.decoder))

actor.eval().to(device)
decoder.eval().to(device)

network = TestNet(actor, decoder)
network.to(device)

content_tf = test_transform(args.content_size, args.crop)
style_tf = test_transform(args.style_size, args.crop)

times = []
steps = args.steps
for content_path in content_paths:
    for style_path in style_paths:
        try:
            content = content_tf(Image.open(str(content_path)).convert('RGB')).to(device).unsqueeze(0)
            style = style_tf(Image.open(str(style_path)).convert('RGB')).to(device).unsqueeze(0)

            c_name = output_dir / '{:s}{:s}'.format(content_path.stem, args.save_ext)
            save_image(content, str(c_name))
            s_name = output_dir / '{:s}{:s}'.format(style_path.stem, args.save_ext)
            save_image(style, str(s_name))
            tic = time.time()

            with torch.no_grad():
                flops, params = thop.profile(network, inputs=(content, style, args.alpha))
                print("GFLOPS: %.4f, Params: %.4f" % (flops / 1e9, params / 1e6))
                for i in range(steps):
                    content = network(content, style)

                    toc = time.time()
                    times.append(toc-tic)
                    print("Elapsed time: %.4f seconds" % (toc - tic))
                    print("step: ", i)
                    output = content.cpu()

                    output_name = output_dir / '{:s}_stylized_{:s}{:s}{:s}'.format(
                        content_path.stem, style_path.stem, str(i), args.save_ext)
                    save_image(output, str(output_name))
            # output_name = output_dir / '{:s}_stylized_{:s}{:s}'.format(
            #     content_path.stem, style_path.stem, args.save_ext)
            # save_image(output, str(output_name))
        except:
            traceback.print_exc()
