import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt

class Main:
	def __init__(self):
		pass

	def Execute(self):
		fig, axes = plt.subplots(1, 3, layout = "constrained")
		fig.set_figwidth(12)
		fig.set_figheight(4)
		
		self.ShowBigO_N(axes[0])
		self.ShowBigO_B(axes[1])
		self.ShowBigO_T(axes[2])
		
		axes[0].set_title("a)")
		axes[1].set_title("b)")
		axes[2].set_title("c)")
		fig.show()
		plt.show()
	
	def ShowBigO_T(self, ax):
		n = 6
		b = 7
		h = 256
		
		ax.cla()
		
		x = np.linspace(10, 100)
		y_nn = 2 * ((n * h) + h) 
		y_nn = np.array([y_nn]).repeat(x.shape[0])
		ax.plot(x, y_nn, label = 'nn256', color = 'b')
		
		t = np.linspace(10, 100)
		y_itsim = t * ((n * (3 * b + 4)) + b)
		ax.plot(t, y_itsim, label = 'itsim', color = 'g')
		
		ax.set(xlabel = "time steps", ylabel = "computational complexity")
		ax.set_xlim([10, 100])
		ax.set_ylim([0, 10000])
		ax.set_xticks(np.linspace(10, 100, 10))
		ax.set_yticks([])
		
		leg = ax.legend(loc = "upper left")	
	
	def ShowBigO_B(self, ax):
		n = 6
		h = 256
		p = 10
		t = 40
		
		ax.cla()
		
		x = np.linspace(0, 10)
		y_nn = 2 * ((n * h) + h) 
		y_nn = np.array([y_nn]).repeat(x.shape[0])
		ax.plot(x, y_nn, label = 'nn256', color = 'b')
		
		b = np.linspace(0, 10)
		y_itsim = t * ((n * (3 * b + 4)) + b)
		ax.plot(b, y_itsim, label = 'itsim', color = 'g')
		
		ax.set(xlabel = "obstacle count", ylabel = "computational complexity")
		ax.set_xlim([0, 10])
		ax.set_ylim([0, 10000])
		ax.set_xticks(np.linspace(0, 10, 11))
		ax.set_yticks([])
		
		leg = ax.legend()
		
	def ShowBigO_N(self, ax):
		h = 256
		b = 7
		p = 10
		t = 40
		
		ax.cla()
		
		x = np.linspace(0, 10)
		y_nn = 2 * ((x * h) + h) 
		ax.plot(x, y_nn, label = 'nn256', color = 'b')
		
		y_itsim = t * ((x * (3 * b + 4)) + b)
		ax.plot(x, y_itsim, label = 'itsim', color = 'g')
		
		y_pdb = x * np.power(p, x) 
		ax.plot(x, y_pdb, label = 'pdb', color = 'r')
		
		ax.set(xlabel = "input size", ylabel = "computational complexity")
		ax.set_xlim([0, 10])
		ax.set_ylim([0, 10000])
		ax.set_xticks(np.linspace(0, 10, 11))
		ax.set_yticks([])
		
		leg = ax.legend()
		
main = Main()
main.Execute()