import os
import csv
import math
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchmetrics
import torch.optim as optim
from torchvision import transforms
from torch_scatter import scatter_add
from torch.utils.data import Dataset, random_split
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops
from torch_geometric.nn import SuperGATConv, GCNConv, GATConv, TopKPooling, global_mean_pool, global_max_pool, MessagePassing

from loader import generate_loader
from model import Model, NN_Model

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import seaborn as sns
import matplotlib.pyplot as plt


if __name__ == "__main__":

	torch.cuda.set_device(0)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


	weight_path = "../results/exp1/checkpoint"
	output_path = f"../results/explainer"
	data_path = "../data"
	if not os.path.exists(f"{output_path}/genXaiInput_all"):
		os.mkdir(f"{output_path}/genXaiInput_all")

	# just use one model from each subject
	sub_exp_id = 0
 
	for sub_id in range(10):
		
		config = np.load(f"{weight_path}/{sub_exp_id}_{sub_id}_config.npy", allow_pickle=True).item()
		config["batch_size"] = 1				# just to be save, we generate process each sample one by one for GNNexplainer

		train_loader = generate_loader(config, mode="train")
		model = Model(config).to(device)
		checkpoint = torch.load(f"{weight_path}/{sub_exp_id}_{sub_id}.pt")     
		model.load_state_dict(checkpoint['model_state_dict'])

		model.eval()
		with torch.no_grad():
			record_target_label = torch.zeros(1).to(device)
			record_predict_label = torch.zeros(1, 6).to(device)
			record_x = torch.zeros(1, 124*32).to(device)
			record_batch = torch.zeros(1, 124).to(device)

			for i, batch in enumerate(train_loader):

				x = batch.x.to(device)
				y = batch.y.to(device)
				edge_index = batch.edge_index.to(device)
				batch = batch.batch.to(device)

				pred = model(x, edge_index, batch)

				record_target_label = torch.cat((record_target_label, y), 0)
				record_predict_label = torch.cat((record_predict_label, pred), 0)    
				record_x = torch.cat((record_x, x.reshape(-1, 124*32)), 0)  
				record_batch = torch.cat((record_batch, batch.reshape(-1, 124)), 0)  
				record_edge_index = edge_index				# all graph have the same edge_index

			y_true = record_target_label[1:].cpu().detach().numpy()
			y_pred = record_predict_label[1:].cpu().detach().numpy()
			y_pred = np.argmax(y_pred, axis=1)
			record_x = record_x[1:].cpu().detach().numpy()
			record_batch = record_batch[1:].cpu().detach().numpy()
			record_edge_index = record_edge_index.cpu().detach().numpy()

			accuracy = accuracy_score(y_true, y_pred)
			print("Accuracy:", accuracy)

			np.save(f"{output_path}/genXaiInput_all/{sub_id}_XaiInput.npy", 
				{'x': record_x, 
				'batch': record_batch,
				'edge_index': record_edge_index,
				'y_true': y_true, 
				'y_pred': y_pred})

		

