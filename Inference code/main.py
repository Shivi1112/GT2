import os
import torch
import argparse
import numpy as np
from joblib import load
from utils import *
from data.Task import *
from models.Model import *
from models.baselines import *
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--dev', type=int, default=7)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--dataset', type=str, default="mimic3", choices=['mimic3', 'mimic4', 'ccae'])
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--pe_dim', type=int, default=4, help='dimensions of spatial encoding')
args = parser.parse_args()

# Set random seeds
torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.cuda.manual_seed(args.seed)

# Set device
cudaid = "cuda:" + str(args.dev)
device = torch.device(cudaid if torch.cuda.is_available() else "cpu")
print(f"Running inference on {args.dataset} with device {device}")

# Dataset loading
fileroot = {
    'mimic3': 'data path of mimic3',
    'mimic4': 'data path of mimic4',
    'ccae': './data/processed_dip.pkl'
}
if args.dataset == 'mimic4':
    task_dataset = load_dataset(args.dataset, root=fileroot[args.dataset], task_fn=diag_prediction_mimic4_fn)
elif args.dataset == 'mimic3':
    task_dataset = load_dataset(args.dataset, root=fileroot[args.dataset], task_fn=diag_prediction_mimic3_fn)
else:
    task_dataset = load_dataset(args.dataset, root=fileroot[args.dataset])

# Tokenizer
Tokenizers = get_init_tokenizers(task_dataset)
label_tokenizer = Tokenizer(tokens=task_dataset.get_all_tokens('conditions'))

# Load processed data if available
data_path = f'./logs/{args.dataset}_{args.pe_dim}.pkl'
if os.path.exists(data_path):
    mdataset = load(data_path)
else:
    mdataset = MMDataset(task_dataset, Tokenizers, dim=128, device=device, trans_dim=args.pe_dim)
    from joblib import dump
    dump(mdataset, data_path)

_, _, testset = split_dataset(mdataset)
_, _, test_loader = mm_dataloader(None, None, testset, batch_size=args.batch_size)

# Build and load model
model = TRANS(Tokenizers, 128, len(task_dataset.get_all_tokens('conditions')),
              device, graph_meta=graph_meta, pe=args.pe_dim)
ckptpath = f'./logs/trained_TRANS_{args.dataset}.ckpt'

# Load trained model
model.load_state_dict(torch.load(ckptpath, map_location=device))
model = model.to(device)
model.eval()


# Run inference
with torch.no_grad():
    y_true, y_prob, sample_ids = test(test_loader, model, label_tokenizer)

# Save predictions to CSV
output_csv_path = f'./logs/predictions_{args.dataset}.csv'

df = pd.DataFrame({
    'id': sample_ids,
    'y_true': y_true,
    'y_prob': [list(map(float, prob)) for prob in y_prob],  # convert tensors to list if needed
})
df.to_csv(output_csv_path, index=False)

print(f"\n Inference results saved to: {output_csv_path}")


