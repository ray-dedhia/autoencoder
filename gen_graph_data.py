"""
generate graph data based on tensorflow csv files
"""

import numpy as np
import os.path
import matplotlib.pyplot as plt

class make_graphs:

	def __init__(self, nn=0, lr=0.01): 
		"""
		nn: neural network number
		lr: initial learning rate
		"""
		self.nn = nn
		self.lr = str(lr)[2:]	
		self.tr_amts = [100, 250, 500, 1000, 2000, 4000, 8000]

	def gen_arr(self, tr, num, loss):
		"""
		tr: amt of training data
		num: file number
		loss: training, validation or testing loss (0, 1 or 2)
		"""
		graphs = ["training", "validation", "test"]

		fn = "csv_files/run_tr_{}_nn_{}_lr_{}_{},tag_{}-loss.csv".format(tr, self.nn, self.lr, num, graphs[loss])

		if os.path.isfile(fn):
			return np.genfromtxt(fn, delimiter=",", skip_header=1, usecols=(1,2)).T	

		else:
			return None

	def all_arr(self, tr, loss):
		arrs = list(self.gen_arr(tr, i+1, loss) for i in range(5))
		rem_nones = []
		for arr in arrs:
			if arr is not None:
				rem_nones.append(arr)
		arrs = rem_nones
		if arrs==[]:
			return None
		sizes = list(len(arr[0]) for arr in arrs)
		l = min(sizes)
		for i in range(len(arrs)):
			arrs[i] = np.array([ arrs[i][0][:l], arrs[i][1][:l] ])
		return arrs

	def all_arr_test(self, tr):
		arrs = list(self.gen_arr(tr, i+1, 2) for i in range(5))
		rem_nones = []
		for arr in arrs:
			if arr is not None:
				rem_nones.append(np.mean(arr[1]))
		arrs = rem_nones
		if arrs==[]:
			return None
		return arrs

	def ave_arr(self, tr, loss):
		arrs = self.all_arr(tr, loss)
		if arrs is None:
			return None
		ave_arr = sum(arrs) / len(arrs)
		return np.array(ave_arr)

	def gen_test_data(self):
		test_arrs = []
		using_trs = []
		for tr_amt in self.tr_amts:
			all_arr = self.all_arr_test(tr_amt)
			if all_arr is not None:
				test_arrs.append(all_arr)
				using_trs.append(tr_amt)
		if test_arrs==[]:
			return (None, None)
		test_data = {'min':[], 'max':[], 'ave':[]}
		for arr in test_arrs:
			test_data['min'].append(min(arr))
			test_data['max'].append(max(arr))
			test_data['ave'].append(np.mean(arr))
		test_data['min'] = np.array(test_data['min'])
		test_data['max'] = np.array(test_data['max'])
		test_data['ave'] = np.array(test_data['ave'])
		return (test_data, using_trs)

	def graph_test_data(self):
		nn_names={0:'nn0', 1:'nn1,D,B', 2:'nn2,D,B', 3:'nn3,D,B', 4:'nn4', 5:'nn0,D,B', 6:'nn0,D', 7:'nn0,B', 8:'nn2,B', 9:'nn2,D', 10:'nn5', 11:'nn1,D', 12:'nn1,B'} 

		graphs_data = [[]]

		neural_nets = range(13)
		first = True

		for lr in ['01', '05', '001', '1']:
			self.lr = lr
			for nn in neural_nets: 
				self.nn = nn
				arr, trn_amts = self.gen_test_data()
				if arr is not None:
					label = "{},lr{}".format(nn_names[nn], lr)
					if (nn==7 and lr=='001') or (nn==11 and lr=='001') or (nn==2 and lr=='001') or (nn==3 and lr=='001') or (nn==4):
						graphs_data[0].append((trn_amts, arr['ave'], [arr['ave'] - arr['min'], arr['max'] - arr['ave']], label ))

		fig = plt.figure(figsize=(8,8))
		subplots = [111]

		for i,graph in enumerate(graphs_data):
			ax = plt.subplot(subplots[i])
			for data in graph:
				plt.errorbar(data[0], data[1], data[2], capsize=4, lw=1.8, fmt='.-', label=data[3])
			plt.ylabel("loss")
			plt.xlabel("amount of training data (number of images)")
			plt.yticks(np.arange(0, 0.005, 0.0005))
			plt.title("best run of each nn: test loss vs. amount of training data")
			ax.grid()
			ax.legend() 

		plt.savefig("third_graph.png")

		plt.show()

	def fold_ex(self):
		self.nn = 5
		arrs = self.all_arr(2000, 1)
		return arrs

x = make_graphs()
x.graph_test_data()
