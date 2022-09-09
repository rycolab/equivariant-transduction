import argparse
import torch
from dataset import DataSet
from groups import CyclicPermutationGroup
from model import EquivariantHardAlignmentModel

def evaluate_on_split(model, data_gen, device, split='test', beam_size=3, 
    max_length=50, use_max=False):
    set_size = len(data_gen.data[split]['in'])
    total_correct = 0
    for i in range(set_size):
        x, y = data_gen.get_batch(1, split=split)
        y_idxs = data_gen.out_vocab.batch_tensor_to_sent(y).to(device)
        y_pred = model.decode(x.to(device), beam_size=beam_size, 
            max_length=max_length, use_max=use_max)
        if y_idxs.shape[1] == y_pred.shape[0]:
            if torch.all(torch.eq(y_idxs.squeeze(0), y_pred)):
                total_correct += 1
        if (i + 1) % 100 == 0:
            print(i,"/",set_size)
            print("Accuracy so far:", total_correct / i)
    print("Accuracy", total_correct / set_size)
    return total_correct / set_size

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help="Model path")
    parser.add_argument('--scan_split', type=str, choices=['simple', 'add_jump',
     'length_generalization', 'around_right', 'simple_p1', 'simple_p2', 
     'simple_p4', 'simple_p8','simple_p16', 'simple_p32', 'simple_p64'], 
        help="SCAN split to train on", default='simple')
    parser.add_argument('--scan_equi', type=str, 
        choices=['verbs', 'directions'], help="Equivariance to use", 
        default='verbs')
    parser.add_argument('--scan_dir', type=str, default='./SCAN/')
    parser.add_argument('--dev_percent', type=float, default=0.1, 
        help="Proportion of training data to use as dev set")
    parser.add_argument('--split', type=str, choices=["train", "test", "dev"], 
        default="test")
    parser.add_argument('--beam_size', type=int, help="Beam size", default=3)
    parser.add_argument('--use_max', default=False, action='store_true',
                    help="Use max in decoding instead of sum.")
    args = parser.parse_args()

    if args.scan_equi == "verbs":
        in_equivariances = ["jump", "run", "walk", "look"]
        out_equivariances = ["I_JUMP", "I_RUN", "I_WALK", "I_LOOK"]
    elif args.scan_equi == "directions":
        in_equivariances = ["right", "left"]
        out_equivariances = ["I_TURN_RIGHT", "I_TURN_LEFT"]
    dir_path = args.scan_dir

    data_gen = DataSet(dir_path, in_equivariances, out_equivariances, 
        args.dev_percent, args.scan_split)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(args.model_path).to(device)
    test_acc = evaluate_on_split(model, data_gen, device, split=args.split, 
        beam_size=args.beam_size, max_length=50, use_max=args.use_max)