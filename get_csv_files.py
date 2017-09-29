"""
get tensorflow csv files from autoencoder runs
"""

import ipdb
import os.path
import urllib as ulb

def open_csv(tr, nn, lr, run, tag):
	file_name = "csv_files/run_tr_{}_nn_{}_lr_{}_{},tag_{}-loss.csv".format(tr, nn, lr, run, tag)

	url = "http://devbox:6006/data/scalars?run=tr_{}_nn_{}_lr_{}/{}&tag={}-loss&format=csv".format(tr, nn, lr, run, tag)

	if os.path.isfile(file_name):
		return True

	dwnld = ulb.URLopener()

	try:
		dwnld.retrieve(url, file_name)
		return True
	except:
		return False

if __name__ == "__main__":
	tr_amts = [100, 250, 500, 1000, 2000, 4000, 8000]
	tags = ["training", "validation", "test"]
	for nn in range(13):
		for tr in tr_amts:
			for tag in tags:
				for run in range(1,6):
					for lr in ['1', '01', '001', '05', '005', '0005']:
						open_csv(tr, nn, lr, run, tag)

