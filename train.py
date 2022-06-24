import os
import sys
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)

import argparse
from nerf.utils import Config, get_root_logger
import time
import taichi as ti
import numpy as np
import pickle as pkl


arch = ti.vulkan if ti._lib.core.with_vulkan() else ti.cuda
ti.init(arch=arch, random_seed=5) # device_memory_fraction=0.8

dtype_f_np = np.float32
real = ti.f32
scalar = lambda: ti.field(dtype=real)

@ti.data_oriented
class SGD:
	def __init__(self, params, lr):
		self.params = params
		self.lr = lr

	def step(self):
		for w in self.params:
			self._step(w)

		# 如果是

	@ti.kernel
	def _step(self, w: ti.template()):
		for I in ti.grouped(w):
			w[I] -= w.grad[I] * self.lr

	def zero_grad(self):
		for w in self.params:
			w.grad.fill(0.0)


@ti.data_oriented
class Linear:
	def __init__(self,
				 batch_size,
				 n_input,
				 n_output,
				 needs_grad=False, 
				 activation=False):
		self.batch_size = batch_size
		self.n_input = n_input
		self.n_output = n_output
		self.activation = activation

		self.hidden = scalar()
		self.output = scalar()

		ti.root.dense(ti.ij, (self.batch_size, self.n_output)).place(self.hidden) # [batch_size, n_hidden]
		ti.root.dense(ti.ij, (self.batch_size, self.n_output)).place(self.output) # [batch_size, n_output]

		self.weights1 = scalar()
		self.bias1 = scalar()

		self.n_hidden_node = ti.root.dense(ti.i, self.n_output)
		self.weights1_node = self.n_hidden_node.dense(ti.j, self.n_input) # hidden , input

		self.n_hidden_node.place(self.bias1) # [hidden, ]
		self.weights1_node.place(self.weights1) # [hidden, input]
		
		if needs_grad:
			ti.root.lazy_grad()

	def parameters(self):
		return [self.weights, self.bias]

	@ti.kernel
	def weights_init(self):
		q1 = ti.sqrt(6 / self.n_input) * 0.01
		for i, j in ti.ndrange(self.n_output, self.n_input):
			self.weights[i, j] = (ti.random() * 2 - 1) * q1

	@ti.kernel
	def _forward(self, t: ti.i32, nn_input: ti.template()):
		for k, i, j in ti.ndrange(self.batch_size, self.n_output, self.n_input):
			self.hidden[k, i] += self.weights[i,j] * nn_input[t, k, j]
		
		if ti.static(self.activation):
			for k, i in ti.ndrange(self.batch_size, self.n_output):
				self.output[k, i] = ti.max(self.hidden[k, i] + self.bias1[i], 0)
		else:
			for k, i in ti.ndrange(self.batch_size, self.n_output):
				self.output[k, i] = self.hidden[k, i] + self.bias1[i]

	@ti.kernel
	def clear(self):
		for I in ti.grouped(self.hidden):
			self.hidden[I] = 0.
		for I in ti.grouped(self.output):
			self.output[I] = 0.

	def forward(self, t, nn_input):
		self._forward(t, nn_input)

	def dump_weights(self, name="save.pkl"):
		w_val = []
		for w in self.parameters():
			w = w.to_numpy()
			w_val.append(w[0])
		with open(name, "wb") as f:
			pkl.dump(w_val, f)

	def load_weights(self, name="save.pkl"):
		with open(name, 'rb') as f:
			w_val = pkl.load(f)
		self.load_weights_from_value(w_val)

	def load_weights_from_value(self, w_val):
		for w, val in zip(self.parameters(), w_val):
			if val.shape[0] == 1:
				val = val[0]
			self.copy_from_numpy(w, val)

	@staticmethod
	@ti.kernel
	def copy_from_numpy(dst: ti.template(), src: ti.ext_arr()):
		for I in ti.grouped(src):
			dst[I] = src[I]

def init_data():
	pass

def init_nn_model():
	global BATCH_SIZE, rays_o, rays_d, rays_dir, pts, target, fc1, fc2, list_hash_table
	global loss
	global optimizer

	n_input = 3
	n_output = 16
	n_output_act = 3
	learning_rate = 1e-2
	
	BATCH_SIZE = 4096
	rays_o = ti.field(float, shape=(BATCH_SIZE, 3))
	rays_d = ti.field(float, shape=(BATCH_SIZE, 3))
	rays_dir = ti.field(float, shape=(BATCH_SIZE, 3))
	pts = ti.field(float, shape=(BATCH_SIZE, 128, 3))
	target = ti.field(float, shape=(BATCH_SIZE, 3))
	loss = ti.field(float, shape=(), needs_grad=True)


	fc1 = Linear(batch_size=BATCH_SIZE,
					n_input=n_input,
					n_output=n_output,
					needs_grad=True,
					activation=False)
	fc2 = Linear(batch_size=BATCH_SIZE,
					n_input=n_output,
					n_output=n_output_act,
					needs_grad=True,
					activation=True)
	fc1.weights_init()
	fc2.weights_init()
	NNs = [fc1, fc2]
	parameters = []
	for layer in NNs:
		parameters.extend(layer.parameters())
	optimizer = SGD(params=parameters, lr=learning_rate)

	# Training data generation
	sample_num = BATCH_SIZE * 25

	def targets_generation(num, x_range_, y_range_, z_range_):
		low = np.array([x_range_[0], y_range_[0], z_range_[0]])
		high = np.array([x_range_[1], y_range_[1], z_range_[1]])
		return np.array(
			[np.random.uniform(low=low, high=high) for _ in range(num)])

	np.random.seed(0)
	all_data = targets_generation(sample_num, x_range, y_range, z_range)
	training_sample_num = BATCH_SIZE * 4
	training_data = all_data[:training_sample_num, :]
	test_data = all_data[training_sample_num:, :]
	

init_nn_model()

def parse_args():
	parser = argparse.ArgumentParser(description="Train a nerf")
	parser.add_argument("config", help="train config file path")
	parser.add_argument("--work_dir", help="the dir to save logs and models")
	args = parser.parse_args()
	return args

def main(timestamp):
	args = parse_args()
	cfg = Config.fromfile(args.config)

	if args.work_dir is not None:
		cfg.work_dir = args.work_dir

	cfg.work_dir = os.path.join(cfg.work_dir, timestamp)

	
	backup_dir = os.path.join(cfg.work_dir, "configs")
	os.makedirs(backup_dir, exist_ok=True)
	try:
		os.system("cp %s %s/" % (args.config, backup_dir))
	except:
		pass

	logger = get_root_logger(cfg.log_level)
	logger.info(f"Backup config file to {cfg.work_dir}")

	# read all training data to memory

	# start to train
	losses = []
	losses_epoch_avg = []
	epochs = 20
	for epoch in range(epochs):
		loss_epoch = 0.0
		cnt = 0
		for current_data_offset in range(0, training_sample_num, BATCH_SIZE):
			# fill rays_o rays_d

			# fill target

			# cal viewdirs

			# get z_vals

			# cal points for each ray

			# hash points to index to get training features and corresponding indexs

			# fill encoding for viewdirs

			# use features and viewdirs to get weights

			# use weights to get fine net's points

			# hash points to index to get training features (mlp's input) and corresponding indexs

			# fill encoding for viewdirs

			

			fc1.clear()
			fc2.clear()
			with ti.ad.Tape(loss=loss):
				# use features and viewdirs to get weights

				# use weights to get rgb
				
				# compute_loss()
			
			optimizer.step() # mlp 

			# deal the hashed features's grad by hand according to index, update it

			print(
				f"current opt progress: {current_data_offset + BATCH_SIZE}/{training_sample_num}, loss: {loss[None]}"
			)
			losses.append(loss[None])
			loss_epoch += loss[None]
			cnt += 1
		print(
			f'opt iter {opt_iter} done. Average loss: {loss_epoch / cnt}')
		losses_epoch_avg.append(loss_epoch / cnt)


if __name__ == "__main__":
	timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
	main(timestamp)
