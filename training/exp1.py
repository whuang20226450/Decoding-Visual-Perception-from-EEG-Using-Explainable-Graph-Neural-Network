import os
import csv
import math
import time
import random
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
from model import Model

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import seaborn as sns
import matplotlib.pyplot as plt


def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues):
	# y_true = [0, 1, 1, 0, 1, 0]
	# y_pred = [0, 1, 0, 0, 1, 1]

	cm = confusion_matrix(y_true, y_pred)
	
	plt.figure(figsize=(8, 6))
	sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
	plt.title('Confusion Matrix')
	plt.ylabel('True Label')
	plt.xlabel('Predicted Label')

	# # Tick labels can be set as needed
	# plt.xticks([0.5, 1.5], ['Negative (0)', 'Positive (1)'])
	# plt.yticks([0.5, 1.5], ['Negative (0)', 'Positive (1)'])

	plt.show()
	return

def set_seed(seed):
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	random.seed(seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

def evaluate(model, data_loader):
	model.eval()
	running_loss = 0.0
	total = 0

	with torch.no_grad():
		record_target_label = torch.zeros(1).to(device)
		record_predict_label = torch.zeros(1, 6).to(device)

		for i, batch in enumerate(data_loader):

			x = batch.x.to(device)
			y = batch.y.to(device).squeeze(-1)
			edge_index = batch.edge_index.to(device)

			pred = model(x, edge_index, batch.batch.to(device))
			loss = criterion(pred, y, reduction='mean')

			cur_bs = (batch.batch[-1] + 1).cpu().detach().numpy()
			running_loss += loss.item() * cur_bs
			total += cur_bs

			record_target_label = torch.cat((record_target_label, y), 0)
			record_predict_label = torch.cat((record_predict_label, pred), 0)      
			
		y_true = record_target_label[1::].cpu().detach().numpy()
		y_pred = record_predict_label[1::].cpu().detach().numpy()
		y_pred = np.argmax(y_pred, axis=1)

		accuracy = accuracy_score(y_true, y_pred)
		print("Accuracy:", accuracy)

		macro_score = f1_score(y_true, y_pred, average='macro')
		micro_score = f1_score(y_true, y_pred, average='micro')
		print(f"Macro F1 score: {macro_score}")
		print(f"Micro F1 score: {micro_score}\n")

	return accuracy, macro_score, micro_score, running_loss, total



if __name__ == "__main__":
	set_seed(0)
	torch.cuda.set_device(0)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	output_path = f"../results/exp1"
	data_path = "../data"
	if not os.path.exists(output_path):
		os.mkdir(output_path)
		os.mkdir(f"{output_path}/checkpoint")

	best_val_acc_list = [[] for i in range(9)]
	train_loader, val_loader = None, None

	edge_mode_list = ["distance_knn", "distance_knn", "complete_graph"]
	k_list = [5, 10, None]
	n_layers_list = [1, 2, 3]
	sub_exp_id = -1
	for edge_mode, k in zip(edge_mode_list, k_list):

		for n_layers in n_layers_list:
			sub_exp_id += 1
			for fold in range(10):
				config = {}
				config["n_class"] = 6
				config["n_sample"] = 32
				config["n_channel"] = 124

				config["fold"] = fold
				config["data_path"] = data_path
				config["add_self_loops"] = True

				config["batch_size"] = 10
				config['epochs'] = 20
				config["lr"] = 5e-4
				config["num_workers"] = 4
				config["weight_decay"] = 1e-5

				config["input_dim"] = 32
				config["n_heads"] = 16
				config["dropout"] = 0.5
				config["edge_sample_ratio"] = 1
				config["neg_sample_ratio"] = 0.5
				config["inter_connect"] = True

				# exp content
				config["sub_id"] = 1
				config["n_layers"] = n_layers
				config["edge_mode"] = edge_mode
				config["k"] = k
				np.save(f"{output_path}/{sub_exp_id}_{fold}_config.npy", config)

				train_loader, val_loader = generate_loader(config, fold=config["fold"])

				model = Model(config).to(device)
				opt = optim.Adam(model.parameters(), lr=config['lr'], weight_decay = config['weight_decay'])
				scaler = torch.cuda.amp.GradScaler()

				criterion = F.cross_entropy
				
				total, max_acc = 0, 0
				train_loss_list, train_acc_list = [], []
				val_loss_list, val_acc_list = [], []

				for epoch in range(config['epochs']):
					model.train()
					start_time, time_pin = time.time(), time.time()
					count, running_loss = 0, 0.0
					record_target_label = torch.zeros(1).to(device)
					record_predict_label = torch.zeros(1, 6).to(device)
					logs = []
					
					for i, batch in enumerate(train_loader):
						
						model.zero_grad()
						opt.zero_grad()

						x = batch.x.to(device)
						y = batch.y.to(device).squeeze(-1)
						edge_index = batch.edge_index.to(device)
						
						pred = model(x, edge_index, batch.batch.to(device))
						loss = criterion(pred, y, reduction='mean')
						
						record_target_label = torch.cat((record_target_label, y), 0)
						record_predict_label = torch.cat((record_predict_label, pred), 0)      
						
						scaler.scale(loss).backward()
						scaler.step(opt)
						scaler.update()
						
						cur_bs = (batch.batch[-1] + 1).cpu().detach().numpy()
						running_loss += loss.item() * cur_bs
						count += cur_bs
						
						if count % 1000 == 0:
							if total == 0:
								print(f"epoch {epoch}: {count}/unknown | train loss: {running_loss / count:.4f} | duration: {time.time() - time_pin:.2f} seconds")
								logs.append("epoch "+ str(epoch) +": "+ str(count) +"/unknown | train loss: "+ str(round(running_loss / count, 4)) +" | duration: "+ str(round(time.time() - time_pin)) +" seconds")
							elif total != 0:
								print(f"epoch {epoch}: {count}/{total} {count/total*100:.2f}% | train loss: {running_loss / count:.4f} | duration: {time.time() - time_pin:.2f} seconds")
								logs.append("epoch "+ str(epoch) +": "+ str(count) +"/unknown | train loss: "+ str(round(running_loss / count, 4)) +" | duration: "+ str(round(time.time() - time_pin)) +" seconds")
							time_pin = time.time()
									
					total = count

					y_true = record_target_label[1::].cpu().detach().numpy()
					y_pred = record_predict_label[1::].cpu().detach().numpy()
					y_pred = np.argmax(y_pred, axis=1)
					train_acc_list.append(accuracy_score(y_true, y_pred))
					train_loss_list.append(running_loss / total)
					
					accuracy, macro_score, micro_score, val_running_loss, val_total = evaluate(model, val_loader)
					val_acc_list.append(accuracy)
					val_loss_list.append(val_running_loss / val_total)

					if accuracy > max_acc:
						max_acc = accuracy
						torch.save({
							'model_state_dict': model.state_dict(),
							'optimizer_state_dict': opt.state_dict(),
						}, f"{output_path}/checkpoint/{sub_exp_id}_{fold}.pt")

					duration = time.time() - start_time
					with open(f"{output_path}/{sub_exp_id}_{fold}_output.txt", 'a') as file:
						for log in logs:
							file.write(f"{log}\n")
						file.write(f"epoch {epoch}: validation acc: {accuracy} | macrof1: {macro_score} | microf1: {micro_score} | loss: {val_running_loss / val_total} | duration: {duration}\n")

					with open(f"{output_path}/{sub_exp_id}_{fold}_result.txt", 'a') as file:
						file.write(f"epoch {epoch}: validation acc: {accuracy} | macrof1: {macro_score} | microf1: {micro_score} | loss: {val_running_loss / val_total} | duration: {duration}\n")

				best_val_acc_list[sub_exp_id].append(max_acc)
				
				# Create a new figure and a subplot with its own y-axis
				fig, ax1 = plt.subplots(figsize=(10, 6))

				# Plot training and validation accuracy on the first y-axis (left)
				ax1.plot(train_acc_list, 'b-', label='Training Accuracy')  # Solid blue line
				ax1.plot(val_acc_list, 'b--', label='Validation Accuracy')  # Dashed blue line
				ax1.set_xlabel('Epochs')
				ax1.set_ylabel('Accuracy', color='b')
				ax1.tick_params('y', colors='b')

				# Create a second y-axis for the same x-axis
				ax2 = ax1.twinx()

				# Plot training and validation loss on the second y-axis (right)
				ax2.plot(train_loss_list, 'r-', label='Training Loss')  # Solid red line
				ax2.plot(val_loss_list, 'r--', label='Validation Loss')  # Dashed red line
				ax2.set_ylabel('Loss', color='r')
				ax2.tick_params('y', colors='r')

				# Adding title and customizing layout
				plt.title('Training/Validation Accuracy and Loss')

				# Adding legends
				lines, labels = ax1.get_legend_handles_labels()
				lines2, labels2 = ax2.get_legend_handles_labels()
				ax2.legend(lines + lines2, labels + labels2, loc='best')

				# plt.show()
				plt.savefig(f'{output_path}/{sub_exp_id}_{fold}_training.png', dpi=300)
			
	acclist = np.array(best_val_acc_list)
	np.save(f"{output_path}/best_val_acc.npy", acclist)
	for i in range(9):
		print(f"sub exp id {i}: {round(np.mean(acclist[i]), 4)}")
		print(f"sub exp id {i}: {round(np.std(acclist[i]), 4)}\n")

	# best_val_acc = np.array(best_val_acc_list)
	# with open(f"{output_path}/all_result.txt", 'a') as file:
	# 	file.write(f"Best val acc, mean: {np.mean(best_val_acc, axis=1)[1:]} | std: {np.std(best_val_acc, axis=1)[1:]}\n")
	# 	for i in range(1,10):
	# 		file.write(f"sub_exp_id {i}: {best_val_acc[i]}\n")


	# best_val_acc = np.array(best_val_acc_list)
	# np.save(f"{output_path}/best_val_acc.npy", best_val_acc)
	# with open(f"{output_path}/a_result.txt", 'a') as file:
	# 	for i in range(9):
	# 		file.write(f"sub_id {i}: {best_val_acc[i]}\n")
	# 	file.write(f"Best val acc, mean: {np.mean(best_val_acc, axis=1)} | std: {np.std(best_val_acc, axis=1)}\n")