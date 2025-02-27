import os
import time
import numpy as np

import torch
from torch_geometric.explain import GNNExplainer, Explainer

from loader import *
from model import *
from utils import *


if __name__ == "__main__":
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	weight_path = "../results/exp1/checkpoint"
	output_path = f"../results/explainer"
	data_path = "../data"
	if not os.path.exists(f"{output_path}/genXaiOutput_all"):
		os.mkdir(f"{output_path}/genXaiOutput_all")
  
	# just use one model from each subject
	sub_exp_id = 0

	for sub_id in range(10):
		
		print(f"sub_id: {sub_id}")
		config = np.load(f"{weight_path}/{sub_exp_id}_{sub_id}_config.npy", allow_pickle=True).item()

		model = XaiModel(config).to(device)
		checkpoint = torch.load(f"{weight_path}/{sub_exp_id}_{sub_id}.pt")  
		model.load_state_dict(checkpoint['model_state_dict'])

		explainer = Explainer(
			model=model,
			algorithm=GNNExplainer(epochs=200),
			explanation_type='model',
			node_mask_type=None,
			edge_mask_type='object',
			model_config=dict(
				mode='multiclass_classification',
				task_level='graph',
				return_type='log_probs',
			),
		)

		data = np.load(f"{output_path}/genXaiInput_all/{sub_id}_XaiInput.npy", allow_pickle=True).item()
		x = torch.tensor(data['x']).to(torch.float32).to(device)
		y_true = torch.LongTensor(data['y_true'])
		y_pred = torch.LongTensor(data['y_pred'])
		edge_index = torch.tensor(data['edge_index']).to(torch.long).to(device)
		batch = torch.tensor(data['batch']).to(torch.long).to(device)

		n_class = 6
		counter, pin_time = 0, time.time()
		edge_weights = [np.zeros((124,124), dtype=np.float32) for _ in range(n_class)]
		explanations = [[] for _ in range(n_class)]
		for _x, _y_true, _y_pred, _batch in zip(x, y_true, y_pred, batch):
			counter += 1
			if counter % 100 == 0:
				print(f"{counter}, duration: {time.time() - pin_time}")
				pin_time = time.time()

			explanation = explainer(_x.reshape(1,-1), edge_index=edge_index, batch=_batch.reshape(-1))
			sub = explanation.get_explanation_subgraph()
			edge_weight = sub['edge_mask'].cpu().detach().numpy()
			sub_edge_index = sub['edge_index'].cpu().detach().numpy()

			# add corretly predicted sample
			if _y_true == _y_pred:
				explanations[_y_true].append((edge_weight, sub_edge_index))
				for i, (f, t) in enumerate(sub_edge_index.T):
					edge_weights[_y_true][f,t] += edge_weight[i]

		np.save(f"{output_path}/genXaiOutput_all/{sub_id}_XaiOutput.npy", {'explanations': explanations, 'edge_weights': edge_weights})


		
