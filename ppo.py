from dataset import *

test_set = Ped_data(filename = 'test.h5', flag = 'Validation')

value, vel, ang, acc = test_set.data_structure()

print(value)
print(vel)
print(ang)
print(acc)


import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from Components import mdn

class PPO(object):
	def __init__(self, 
				 drive_net,		# neural network
				 optimizer,		# optimizer, default: Adam
				 clip,			# clip parameter, default: 0.05
				 gamma,			# discount factor, default 0.99
				 lambd,			# GAE lambda parameter, default: 0.95
				 value_coef,	# value loss parameter, default: 1.
				 entropy_coef,  # policy entropy loss, default: 0.01
				 epoch,			# 
				 horizon		#
				 minibatch_num	#
				 ):
		self.drive_net = drive_net
		self.optimizer = optimizer
		self.clip = clip
		self.gamma = gamma
		self.lambd = lambd
		self.value_coef = value_coef
		self.entropy_coef = entropy_coef
		self.epoch = epoch
		self.horizon = horizon
		self.minibatch_num = minibatch_num
		self.minibatch_horizon = self.horizon / self.minibatch_num

	def updata(self, rollouts):

		advantages, _ = gae(rollouts['rewards'], rollouts['masks'], rollouts['values'], rollouts['gamma'], rollouts['lambd'])

		for e in range(self.epoch):

			for minibatch in range(0, self.horizon, self.minibatch_horizon):

				input_data_minibatch = rollouts['input_data'][:, minibatch: minibatch + self.minibatch_horizon]

				value, acc_pi, acc_mu, acc_sigma, ang, _, _, _, _, _ = self.drive_net.forward(input_data_minibatch, config)

				mdn_accuracy(pi = acc_pi, sigma = acc_sigma, mu = acc_mu, target)


		ratio = torch.exp()








def gae(rewards, masks, values, gamma, lambd):

	T, N, _ = rewards.size()

	advantages = torch.zeros(T, N, 1)
	advantage_t = torch.zeros(N, 1)

	for t in reveresed(range(T)):
		delta = rewards[t] = values[t + 1].data * gamma * masks[t] - values[t].data
		advantage_t = delta + advantage_t * gamma * lambd * masks[t]
		advantages[t] = advantage_t

	return advantages, values[: T].data + advantages
