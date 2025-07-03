import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt

class Main:
	def __init__(self):
		pass

	def Execute(self):
		low_p = torch.FloatTensor([0, 0])
		high_p = torch.FloatTensor([10, 10])
		res_p = torch.FloatTensor([0.75, 0.75])
		
		low_v = torch.FloatTensor([0, 0])
		high_v = torch.FloatTensor([10, 10])
		res_v = torch.FloatTensor([1, 1])
		
		steps_p = ((high_p - low_p + 1) / res_p).int()
		steps_v = ((high_v - low_v + 1) / res_v).int()
		
		x0_vals = torch.linspace(low_p[0], high_p[0], steps_p[0])
		x1_vals = torch.linspace(low_p[1], high_p[1], steps_p[1])
		
		print("Buiilding matrices...")
		positions, velocities = self.DenseGrid(low_p, high_p, low_v, high_v, res_p, res_v)
		acceleration = torch.FloatTensor([0, -9.8]).reshape(1, 1, 1, 2)
		print("    Done!")
		
		positions = positions.cuda()
		velocities = velocities.cuda()
		acceleration = acceleration.cuda()
		
		delta_t = 1 / 20
		steps = 40

		temporal_positions = positions[None, :, :, :, :]
		temporal_positions = temporal_positions.repeat((steps, 1, 1, 1, 1))
		for i in range(1, steps):
			temporal_positions[i] = temporal_positions[i - 1] + (velocities * delta_t)
			velocities = velocities + (acceleration * delta_t)
			
		temporal_positions = temporal_positions.reshape(steps, x0_vals.shape[0], x1_vals.shape[0], velocities.shape[1], velocities.shape[2], 2)

		target = torch.FloatTensor([8, 2]).cuda()
		
		#t = temporal_positions[:, 10, 10, 10, 5, 6, 7, :]
		#t_delta = t - target.reshape(1, 3)
		
		target_delta = temporal_positions - target.reshape(1, 1, 1, 1, 2)
		target_delta = torch.norm(target_delta, dim = -1)
		
		fig, axes = plt.subplots(3, 3, layout = "constrained")
		fig.set_figwidth(7.4)
		fig.set_figheight(7)
		
		self.GraphRow1(axes, temporal_positions, target_delta, target)
		self.GraphRow2(axes, temporal_positions, target_delta, target)
		self.GraphRow3(axes, temporal_positions, target_delta, target)
		
		fig.show()
		plt.show()
	
	def GraphRow3(self, axes, t_positions, target_delta, target):
		temporal_indices = ((torch.arange(20) * 2).int())
		
		tp_delta = target_delta[[20]]
		tp_delta = tp_delta[:, :, :, [6]]
		tp_delta = tp_delta[:, :, :, :, [1]]
		self.ShowRow1Graph(axes[2, 0], t_positions, tp_delta, target)
		self.ShowTrajectory(axes[2, 0], t_positions, [3, 8], [5, 0], temporal_indices[:7])
		circle = plt.Circle((5, 4), 1, color = 'k')
		axes[2, 0].add_patch(circle)
		circle = plt.Circle((4, 3), 1, color = 'k')
		axes[2, 0].add_patch(circle)
		circle = plt.Circle((4, 2), 1, color = 'k')
		axes[2, 0].add_patch(circle)
		circle = plt.Circle((4, 1), 1, color = 'k')
		axes[2, 0].add_patch(circle)
		circle = plt.Circle((5, 5), 1, color = 'k')
		axes[2, 0].add_patch(circle)
		circle = plt.Circle((8, 6), 1, color = 'k')
		axes[2, 0].add_patch(circle)
		circle = plt.Circle((9, 6), 1, color = 'k')
		axes[2, 0].add_patch(circle)
		axes[2, 0].set_title("g)")
		
		tp_delta = target_delta[[20]]
		tp_delta = tp_delta[:, :, :, [6]]
		tp_delta = tp_delta[:, :, :, :, [1]]
		self.ShowRow1Graph(axes[2, 1], t_positions, tp_delta, target)
		self.ShowTrajectory(axes[2, 1], t_positions, [3, 8], [3, 8], temporal_indices[:18])
		circle = plt.Circle((5, 4), 1, color = 'k')
		axes[2, 1].add_patch(circle)
		circle = plt.Circle((4, 3), 1, color = 'k')
		axes[2, 1].add_patch(circle)
		circle = plt.Circle((4, 2), 1, color = 'k')
		axes[2, 1].add_patch(circle)
		circle = plt.Circle((4, 1), 1, color = 'k')
		axes[2, 1].add_patch(circle)
		circle = plt.Circle((5, 5), 1, color = 'k')
		axes[2, 1].add_patch(circle)
		circle = plt.Circle((8, 6), 1, color = 'k')
		axes[2, 1].add_patch(circle)
		circle = plt.Circle((9, 6), 1, color = 'k')
		axes[2, 1].add_patch(circle)
		axes[2, 1].set_title("h)")
		
		tp_delta = target_delta[[20]]
		tp_delta = tp_delta[:, :, :, [6]]
		tp_delta = tp_delta[:, :, :, :, [1]]
		self.ShowRow1Graph(axes[2, 2], t_positions, tp_delta, target)
		self.ShowTrajectory(axes[2, 2], t_positions, [3, 8], [3, 6], temporal_indices)
		circle = plt.Circle((5, 4), 1, color = 'k')
		axes[2, 2].add_patch(circle)
		circle = plt.Circle((4, 3), 1, color = 'k')
		axes[2, 2].add_patch(circle)
		circle = plt.Circle((4, 2), 1, color = 'k')
		axes[2, 2].add_patch(circle)
		circle = plt.Circle((4, 1), 1, color = 'k')
		axes[2, 2].add_patch(circle)
		circle = plt.Circle((5, 5), 1, color = 'k')
		axes[2, 2].add_patch(circle)
		circle = plt.Circle((8, 6), 1, color = 'k')
		axes[2, 2].add_patch(circle)
		circle = plt.Circle((9, 6), 1, color = 'k')
		axes[2, 2].add_patch(circle)
		axes[2, 2].set_title("i)")
	
	def GraphRow2(self, axes, t_positions, target_delta, target):
		temporal_indices = ((torch.arange(20) * 2).int())
		
		tp_delta = target_delta[[20]]
		tp_delta = tp_delta[:, :, :, [6]]
		tp_delta = tp_delta[:, :, :, :, [1]]
		self.ShowRow1Graph(axes[1, 0], t_positions, tp_delta, target)
		self.ShowTrajectory(axes[1, 0], t_positions, [3, 8], [6, 1], temporal_indices)
		axes[1, 0].set_title("d)")
		
		tp_delta = target_delta[[20]]
		tp_delta = tp_delta[:, :, :, [6]]
		tp_delta = tp_delta[:, :, :, :, [1]]
		self.ShowRow1Graph(axes[1, 1], t_positions, tp_delta, target)
		self.ShowTrajectory(axes[1, 1], t_positions, [3, 8], [3, 6], temporal_indices)
		axes[1, 1].set_title("e)")
		
		tp_delta = target_delta[[20]]
		tp_delta = tp_delta[:, :, :, [6]]
		tp_delta = tp_delta[:, :, :, :, [1]]
		self.ShowRow1Graph(axes[1, 2], t_positions, tp_delta, target)
		self.ShowTrajectory(axes[1, 2], t_positions, [3, 8], [1, 6], temporal_indices)
		axes[1, 2].set_title("f)")
	
	def ShowTrajectory(self, ax, temporal_positions, start_position, start_velocity, temporal_indices):
		path = temporal_positions[:, start_position[0]]
		path = path[:, start_position[1]]
		path = path[:, start_velocity[0]]
		path = path[:, start_velocity[1]]
		path = path[temporal_indices]
		
		x = path[:, 0].cpu().numpy()
		y = path[:, 1].cpu().numpy()
		ax.scatter(x, y, label = 'path sample', color = 'g')
	
	def GraphRow1(self, axes, t_positions, target_delta, target):
		print("Row One Graphs")
		
		tp_delta = target_delta[[20]]
		tp_delta = tp_delta[:, :, :, [6]]
		tp_delta = tp_delta[:, :, :, :, [1]]
		self.ShowRow1Graph(axes[0, 0], t_positions, tp_delta, target)
		axes[0, 0].set_title("a)")
		
		tp_delta = target_delta[:]
		tp_delta = tp_delta[:, :, :, [6]]
		tp_delta = tp_delta[:, :, :, :, [1]]
		self.ShowRow1Graph(axes[0, 1], t_positions, tp_delta, target)
		axes[0, 1].set_title("b)")
		
		tp_delta = target_delta[[20]]
		tp_delta = tp_delta[:, :, :, [3, 4, 5]]
		tp_delta = tp_delta[:, :, :, :, [1, 2, 3]]
		self.ShowRow1Graph(axes[0, 2], t_positions, tp_delta, target)
		axes[0, 2].set_title("c)")
		
	def ShowRow1Graph(self, ax, t_positions, tp_delta, target):
		tp0 = t_positions[0, :, :, 0, 0, :]
		
		tp_delta, indices = torch.min(tp_delta, dim = -1)
		tp_delta, indices = torch.min(tp_delta, dim = -1)
		tp_delta, indices = torch.min(tp_delta, dim = 0)
		tp_positive = tp0[tp_delta < 1]
		
		ax.cla()
		
		x = tp_positive[:, 0].cpu().numpy()
		y = tp_positive[:, 1].cpu().numpy()
		ax.scatter(x, y, label = 'positive sample')
		
		x = target.cpu().numpy()[0]
		y = target.cpu().numpy()[1]
		circle = plt.Circle((x, y), 1, color = 'r', alpha = 0.2)
		ax.add_patch(circle)
		
		ax.set_xlim([0, 10])
		ax.set_ylim([0, 10])
		
		ax.set_yticks([0, 5, 10])
		ax.set_xticks([0, 5, 10])
		
		ax.set(xlabel = "x meters", ylabel = "y meters")
		
	def DenseGrid(self, low_p, high_p, low_v, high_v, res_p, res_v):
		steps_p = ((high_p - low_p + 1) / res_p).int()
		steps_v = ((high_v - low_v + 1) / res_v).int()
		
		x0_vals = torch.linspace(low_p[0], high_p[0], steps_p[0])
		x1_vals = torch.linspace(low_p[1], high_p[1], steps_p[1])
		
		v0_vals = torch.linspace(low_v[0], high_v[0], steps_v[0])
		v1_vals = torch.linspace(low_v[1], high_v[1], steps_v[1])
		
		positions = []
		
		for i in range(x0_vals.shape[0]):
			for j in range(x1_vals.shape[0]):
					x0 = x0_vals[i]
					x1 = x1_vals[j]
					
					position = torch.FloatTensor([x0, x1])
					positions.append(position)
		
		positions = torch.stack(positions)
		positions = positions.reshape((-1, 1, 1, 2))
		positions = positions.repeat(1, steps_v[0], steps_v[1], 1)

		velocities = torch.zeros((1, v0_vals.shape[0], v0_vals.shape[0], 2))
		for i in range(v0_vals.shape[0]):
			for j in range(v1_vals.shape[0]):
					v0 = v0_vals[i]
					v1 = v1_vals[j]
					
					velocities[0, i, j] = torch.FloatTensor([v0, v1])
		
		return positions, velocities
	
main = Main()
main.Execute()