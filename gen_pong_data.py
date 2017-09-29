"""
takes random actions in the open ai gym atari 2600 game pong-v0
takes images of the screen, converts them to black and white images, and saves them as bin files
"""

from PIL import Image
import random
import gym
import bitarray
import numpy as np
from scipy.misc import imsave

def binarize_image(input_image):
	"""Binarize an image."""
	mc = input_image.convert('L')  # convert image to monochrome
	arr = np.array(mc)
	data = binarize_array(arr)
	return data.flatten().tolist()		

def binarize_array(array):
	"""Binarize an array."""
	for i in range(len(array)):
		for j in range(len(array[0])):
			if array[i][j] > 100:
				array[i][j] = 1 
			else:
				array[i][j] = 0
	return array


def do_nothing(n, env):
	for k in range(n):
		env.step(0)

def run(l, amt):
	env = gym.make('Pong-v0')
	env.reset()
	do_nothing(20, env)
	f = open("/home/rdedhia/Documents/pongv0/%s/%s.txt"%(l,l), "w")
	
	for _ in range(amt+2):
		env.render()
	
		if (0 != random.randint(0,4) or _ == 0): # change old x val with 4/5 probability
			x = random.randint(0,5)
	
		reward, done = env.step(x)[1:3]
		
		f.write("%d %d \n" %(x,reward))

		fn = "/home/rdedhia/Documents/pongv0/%s/image.png"%l
		env.env.ale.saveScreenPNG(fn)
		img = Image.open(fn)
		area = (0,34,320,194)
		cropped = img.crop(area)
		bw = binarize_image(cropped) # convert to black and white array
		bits = bitarray.bitarray(bw) # convert to bitarray	

		# save to file
		bin_fn = "/home/rdedhia/Documents/pongv0/%s/%s%04d.bin"%(l,l,_)
		with open(bin_fn,'wb') as fh:
			bits.tofile(fh)
	
		if done:
			env.reset()
			do_nothing(20, env)

	f.close()
	env.close()

run("J", 4000)
