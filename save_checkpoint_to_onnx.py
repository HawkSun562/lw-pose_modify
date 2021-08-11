import argparse
import torch
import torch.onnx
from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.load_state import load_state

def main(args):
    net = PoseEstimationWithMobileNet(args.num_refinement_stages)
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    load_state(net, checkpoint)

    dummy_input = torch.randn(1, 3, args.size[0], args.size[1])
    torch.onnx.export(net, dummy_input, args.output_name, opset_version=11)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-refinement-stages', type=int, default=1, help='refine stages number')
    parser.add_argument('--size', type=int, required=True, nargs='+', help='size of input [height width]')
    parser.add_argument('--checkpoint-path', type=str, required=True, help='path to the checkpoint')
    parser.add_argument('--output-name', type=str, default='lw_pose.onnx', help='name of output model in ONNX format')
    args = parser.parse_args()
    main(args)