import argparse
import random
import os
import torch
from dataset import DataSet
from groups import CyclicPermutationGroup
from model import EquivariantHardAlignmentModel
from test import evaluate_on_split

def config_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', type=str, default='./models/', 
    help="Location to save models")
parser.add_argument('--scan_split', type=str, choices=['simple', 'add_jump', 
    'length_generalization', 'around_right', 'simple_p1', 'simple_p2', 
    'simple_p4', 'simple_p8', 'simple_p16', 'simple_p32', 'simple_p64'], 
    help="SCAN split to train on", default='simple')
parser.add_argument('--scan_equi', type=str, choices=['verbs', 'directions'], 
    help="Equivariance to use", default='verbs')
parser.add_argument('--scan_dir', type=str, default='./SCAN/')
parser.add_argument('--K', type=int, help="Dimension of G-Embedding", default=6)
parser.add_argument('--num_filters', type=int, 
    help="Number of filters in G-Convolutions", default=13)
parser.add_argument('--hidden_size', type=int, 
    help="Size of hidden state in LSTM", default=13)
parser.add_argument('--embed_dim', type=int, 
    help="Dimension of embedding", default=67)
parser.add_argument('--batch_size', type=int,
    help="Batch size", default=8)
parser.add_argument('--nonlin', type=str, choices=['tanh'], 
    default="tanh", help="Selection of non-linearity in g-equivariant portion")
parser.add_argument('--epochs', type=int, default=1000, 
    help="Number of epochs")
parser.add_argument('--lr', type=float, default=1e-3, help="Learning Rate")
parser.add_argument('--dev_percent', type=float, default=0.1, 
    help="Proportion of training data to use as dev set")
parser.add_argument('--seed', type=int, help="Random seed")
parser.add_argument('--training_max', default=False, action='store_true',
                    help="Use max in training instead of sum.")
parser.add_argument('--annealed', default=False, action='store_true',
                    help="Anneal max during training.")
parser.add_argument('--best_hyperparams', choices=['simple', 'add_jump', 
    'length_generalization', 'around_right'], default='', 
                    help="Use hyperparams for best-performing model on this"
                    " split")
args = parser.parse_args()

if args.best_hyperparams == '':
    K = args.K
    num_filters = args.num_filters
    hidden_size = args.hidden_size
    scan_equi = args.scan_equi
    embed_dim = args.embed_dim
    nonlin = args.nonlin
    training_max = args.training_max
    annealed = args.annealed
    scan_split = args.scan_split
    batch_size = args.batch_size
    learning_rate = args.lr
    seed = args.seed
else:
    best_hyperparam_dict = {
        'simple': {'K':6, 'num_filters':13, 'hidden_size':13, 
                    'embed_dim':67, 'batch_size':8, 'seed':40779, 
                    'scan_equi':'verbs'},
        'add_jump': {'K':122, 'num_filters':7, 'hidden_size':223, 
                    'embed_dim':67, 'batch_size':8, 'seed':627395, 
                    'scan_equi':'verbs'},
        'around_right': {'K':20, 'num_filters':20, 'hidden_size':9, 
                    'embed_dim':55, 'batch_size':8, 'seed':162950, 
                    'scan_equi':'directions'},
        'length_generalization': {'K':45, 'num_filters':24, 'hidden_size':11, 
                    'embed_dim':149, 'batch_size':32, 'seed':954865, 
                    'scan_equi':'verbs'}
        }
    hyperparams = best_hyperparam_dict[args.best_hyperparams]
    K = hyperparams['K']
    num_filters = hyperparams['num_filters']
    hidden_size = hyperparams['hidden_size']
    scan_equi = hyperparams['scan_equi']
    embed_dim = hyperparams['embed_dim']
    batch_size = hyperparams['batch_size']
    seed = hyperparams['seed']
    learning_rate = 1e-3
    scan_split = args.best_hyperparams
    nonlin = 'tanh'
    training_max = False
    annealed = False

if seed is not None:
    config_seed(seed)

if not os.path.exists(args.model_dir):
    os.makedirs(args.model_dir)

if scan_equi == "verbs":
    in_equivariances = ["jump", "run", "walk", "look"]
    out_equivariances = ["I_JUMP", "I_RUN", "I_WALK", "I_LOOK"]
elif scan_equi == "directions":
    in_equivariances = ["right", "left"]
    out_equivariances = ["I_TURN_RIGHT", "I_TURN_LEFT"]
dir_path = args.scan_dir

data_gen = DataSet(dir_path, in_equivariances, out_equivariances, 
    args.dev_percent, scan_split)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

G = CyclicPermutationGroup(n=len(data_gen.in_vocab), 
    p=len(in_equivariances), device=device)

model = EquivariantHardAlignmentModel(data_gen.in_vocab, data_gen.out_vocab, G, 
    K, num_filters, hidden_size, 1, embed_dim, embed_dim, device, nonlin, 
    training_max, annealed).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
batches = len(data_gen.data['train']['in'])//batch_size
dev_batches = len(data_gen.data['dev']['in'])//batch_size
best_p = float('inf')
model_name = '__'.join([scan_split, str(K), str(num_filters), 
        str(hidden_size), str(embed_dim), 
        str(batch_size), str(learning_rate), ".model"])

T = 0.5
for j in range(args.epochs):
    max_norm=0
    print("EPOCH", j)
    epoch_loss = 0
    for i in range(batches):
        x, y = data_gen.get_batch(batch_size)
        optimizer.zero_grad()
        p = model(x.to(device),y.to(device), T)
        epoch_loss += float(p)
        
        p.backward()

        optimizer.step()

        # Updating annealing parameter
        n_iters = (j*batches) + i
        if (n_iters + 1) % 2000 == 0:
            if args.annealed:
                T *= 0.5
                if T < 0.0001:
                    T = 0.0001
            
    print("EPOCH LOSS AVERAGE", (epoch_loss/batches)/batch_size)
    dev_loss = 0
    for i in range(dev_batches):
        x, y = data_gen.get_batch(batch_size, split='dev')
        dev_p = model(x.to(device),y.to(device), T)
        dev_loss += float(dev_p)
    print("DEV LOSS AVERAGE", (dev_loss/dev_batches)/batch_size, "best", best_p)
    if (dev_loss/dev_batches)/args.batch_size < best_p:
        torch.save(model, os.path.join(args.model_dir, model_name))
        best_p = (dev_loss/dev_batches)/batch_size
        print("Saved new best", best_p)

model = torch.load(os.path.join(args.model_dir, model_name)).to(device)
test_acc = evaluate_on_split(model, data_gen, device, split='test', beam_size=3, 
    max_length=50,use_max=training_max)

print("Accuracy on test set:", test_acc)
full_name = '___'.join([str(test_acc), model_name])
os.rename(os.path.join(args.model_dir, model_name),os.path.join(args.model_dir, 
    full_name))
