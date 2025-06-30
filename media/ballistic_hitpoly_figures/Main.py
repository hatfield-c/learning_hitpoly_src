import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt

class Main:
	def __init__(self):
		pass

	def Execute(self):
		low_p = torch.FloatTensor([0, 0, 0])
		high_p = torch.FloatTensor([10, 10, 10])
		res_p = torch.FloatTensor([0.5, 0.5, 0.5])
		
		low_v = torch.FloatTensor([-5, -5, -5])
		high_v = torch.FloatTensor([5, 5, 5])
		res_v = torch.FloatTensor([1, 1, 1])
		
		steps_p = ((high_p - low_p + 1) / res_p).int()
		steps_v = ((high_v - low_v + 1) / res_v).int()
		
		x0_vals = torch.linspace(low_p[0], high_p[0], steps_p[0])
		x1_vals = torch.linspace(low_p[1], high_p[1], steps_p[1])
		x2_vals = torch.linspace(low_p[2], high_p[2], steps_p[2])
		
		positions, velocities = self.DenseGrid(low_p, high_p, low_v, high_v, res_p, res_v)
		acceleration = torch.FloatTensor([0, 0, -9.8]).reshape(1, 1, 1, 1, 3)
		
		positions = positions.cuda()
		velocities = velocities.cuda()
		acceleration = acceleration.cuda()
		
		delta_t = 1 / 20
		steps = 10
		
		temporal_positions = positions[None, :, :, :, :, :]
		temporal_positions = temporal_positions.repeat((steps, 1, 1, 1, 1, 1))
		for i in range(1, steps):
			temporal_positions[i] = temporal_positions[i - 1] + (velocities * delta_t)
			velocities = velocities + (acceleration * delta_t)
			
		temporal_positions = temporal_positions.reshape(steps, x0_vals.shape[0], x1_vals.shape[0], x2_vals.shape[0], velocities.shape[1], velocities.shape[2], velocities.shape[3], 3)
		print(temporal_positions.shape)
		print(velocities.shape)
		
		target = torch.FloatTensor([7, 0, 3]).cuda()
		
		#t = temporal_positions[:, 10, 10, 10, 5, 6, 7, :]
		#t_delta = t - target.reshape(1, 3)
		
		tp_delta = temporal_positions - target.reshape(1, 1, 1, 1, 1, 1, 1, 3)
		tp_delta = torch.norm(tp_delta, dim = -1)
		print(tp_delta.shape)

		for i in range(1):		
			tp_delta = tp_delta[:, :, 0, :, 7, 5, 7]
			
			tp0 = temporal_positions[0, :, 0, :, 7, 5, 7, [0, 2]]
			tp_delta, indices = torch.min(tp_delta, dim = 0)
			tp_positive = tp0[tp_delta < 2]
			
			fig, ax = plt.subplots(1, 1)
			fig.set_figwidth(7)
			fig.set_figheight(7)
			ax.cla()
			
			x = tp_positive[:, 0].cpu().numpy()
			y = tp_positive[:, 1].cpu().numpy()
			ax.scatter(x, y)
			
			x = target.cpu().numpy()[0]
			y = target.cpu().numpy()[2]
			circle = plt.Circle((x, y), 1, color = 'r', alpha = 0.2)
			ax.add_patch(circle)
			
			ax.set_xlim([0, 10])
			ax.set_ylim([0, 10])
			
			fig.show()
			plt.show()
		
		print("show")
	
	def DenseGrid(self, low_p, high_p, low_v, high_v, res_p, res_v):
		steps_p = ((high_p - low_p + 1) / res_p).int()
		steps_v = ((high_v - low_v + 1) / res_v).int()
		
		x0_vals = torch.linspace(low_p[0], high_p[0], steps_p[0])
		x1_vals = torch.linspace(low_p[1], high_p[1], steps_p[1])
		x2_vals = torch.linspace(low_p[2], high_p[2], steps_p[2])
		
		v0_vals = torch.linspace(low_v[0], high_v[0], steps_v[0])
		v1_vals = torch.linspace(low_v[1], high_v[1], steps_v[1])
		v2_vals = torch.linspace(low_v[2], high_v[2], steps_v[2])
		
		positions = []
		
		for i in range(x0_vals.shape[0]):
			for j in range(x1_vals.shape[0]):
				for k in range(x2_vals.shape[0]):
					x0 = x0_vals[i]
					x1 = x1_vals[j]
					x2 = x2_vals[k]
					
					position = torch.FloatTensor([x0, x1, x2])
					positions.append(position)
		
		positions = torch.stack(positions)
		positions = positions.reshape((-1, 1, 1, 1, 3))
		positions = positions.repeat(1, steps_v[0], steps_v[1], steps_v[2], 1)

		velocities = torch.zeros((1, v0_vals.shape[0], v0_vals.shape[0], v0_vals.shape[0], 3))
		for i in range(v0_vals.shape[0]):
			for j in range(v1_vals.shape[0]):
				for k in range(v2_vals.shape[0]):
					v0 = v0_vals[i]
					v1 = v1_vals[j]
					v2 = v2_vals[k]
					
					velocities[0, i, j, k] = torch.FloatTensor([v0, v1, v2])
		
		return positions, velocities
	
main = Main()
main.Execute()