import os
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

from loader import *
from model import *
from utils import plot2


if __name__ == "__main__":

	weight_path = "../results/exp1/checkpoint"
	output_path = f"../results/explainer"
	data_path = "../data"
	if not os.path.exists(f"{output_path}/plotXai"):
		os.mkdir(f"{output_path}/plotXai")

	sub_id = 6
	data = np.load(f"{output_path}/genXaiInput_all/{sub_id}_XaiInput.npy", allow_pickle=True).item()
	edge_index = data['edge_index']

	data = np.load(f"{output_path}/genXaiOutput_all/{sub_id}_XaiOutput.npy", allow_pickle=True).item()
	explanations = data["explanations"]
	edge_weights = data["edge_weights"]

	n_class = 6
	n_channel = 124
	class_names = ['Human Body', 'Human Face', 'Animal Body', 'Animal Face', 'Fruit Vegetable', 'Inanimate Object']
	class_count = [len(explanations[i]) for i in range(n_class)]

	fig, axs = plt.subplots(2, 3, figsize=(15, 8))
	for i, class_name in enumerate(class_names):
		mean_edge_weight = edge_weights[i] / class_count[i]
		print(f"mean_edge_weight: {mean_edge_weight.shape}")

		ax = axs[i//3, i%3]
		threshold = 0.35

		# the plot2 function is defined in utils.py. 
  		# To plot the head outline, I cropped related part from source code to plot a more customized plot, so that part is pretty messy.
		_, pos = plot2(ax=ax)
		for f, t in zip(edge_index[0], edge_index[1]):

				x, y = [], []
				x.append(pos[f][0])
				x.append(pos[t][0])
				y.append(pos[f][1])
				y.append(pos[t][1])

			w = mean_edge_weight[f, t]
   
			if w > threshold:
				w *= 1.3		# the scaling is to make the plot more clear

				cmap = mpl.cm.coolwarm
				norm = mpl.colors.Normalize(vmin= 0, vmax=1)
				w = mpl.cm.ScalarMappable(norm=norm, cmap=cmap).to_rgba([w])
				ax.plot(x, y, linewidth=2, color=w[0])
				ax.set_title(class_name, fontsize=20)

			sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
			c_bar = fig.colorbar(sm, ax=ax)



	fig.tight_layout()
	fig.savefig(f'S{sub_id}.png', dpi=1200)

	plt.show()

		

