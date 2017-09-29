"""
play openai gym breakout game to generate screenshot data
"""

import conv
import bitarray
import gym
from PIL import Image
import numpy as np
import time
import curses

env=gym.make('Breakout-v0')
env.reset()
env.render()

U = curses.KEY_UP
D = curses.KEY_DOWN
R = curses.KEY_RIGHT
L = curses.KEY_LEFT

iters = -1
prev_lives = 5

def move(step, descr):
	global iters, prev_lives

	iters += 1
	print(descr)
	lives = env.step(step)[3]['ale.lives']
	env.render()

	done = (lives==0)

	if prev_lives > lives:
		env.step(1)
		prev_lives = lives

	env.env.ale.saveScreenPNG("img.png")
	img = Image.open('img.png')
	cropped = img.crop((16,57,304,193))
	bw = conv.binarize_image(cropped) # convert to black and white array
	bits = bitarray.bitarray(bw) # convert to bitarray  
	
	if iters%50==0:
		print("iters", iters)		

	if done: 
		prev_lives = 5
		env.reset()
		env.step(1)
		print('reset')

	# save to file
	bin_fn = "/home/rdedhia/Documents/breakoutv0/bins_3/%05d.bin"%iters
	with open(bin_fn,'wb') as fh:
		bits.tofile(fh)

def pause(screen):
	while True:
		char=screen.getch()
		if char==ord(' '):
			print("restart")
			return False # don't stop the program
		elif char==curses.KEY_BACKSPACE:
			print("stop")
			screen.nodelay(0)
			screen.keypad(0)
			curses.nocbreak()
			curses.echo()
			curses.endwin()
			return True # stop the program

def play(N, screen):
	while True:
		char=screen.getch()

		if char==curses.KEY_BACKSPACE or iters==N:
			print("stop")
			screen.nodelay(0)
			screen.keypad(0)
			curses.nocbreak()
			curses.echo()
			curses.endwin()
			break
		elif char==U:
			move(1, "launch")
		elif char==R:
			move(2, "right")
		elif char==L:
			move(3, "left")
		elif char==ord(' '):
			print("pause")
			if pause(screen):
				break
		else:
			move(0, "do nothing")

def run(N):
	while True:
		screen=curses.initscr()
		curses.noecho()
		curses.cbreak()
		screen.keypad(1)
		screen.nodelay(1)
		char=screen.getch()

		if char==curses.KEY_BACKSPACE or iters==N:
			print("stop")
			screen.nodelay(0)
			screen.keypad(0)
			curses.nocbreak()
			curses.echo()
			curses.endwin()
			break
		elif char==U:
			move(1, "launch")
			play(N, screen)
			break
	
run(10002)
