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
from xml.dom.minidom import parse
from progress.bar import Bar
import cv2
import glob
import math

arch = ti.vulkan if ti._lib.core.with_vulkan() else ti.cuda
# arch = ti.cpu
ti.init(arch=arch, random_seed=5) # device_memory_fraction=0.8

dtype_f_np = np.float32
real = ti.f32
scalar = lambda: ti.field(dtype=real)

def rt_inverse(R, T):
	# input: rt pose
	# w2c->c2w or c2w->w2c
	R = R.transpose(1, 0)
	T = -R @ T
	return R, T

def get_rays_np(H, W, K, c2w):
	"""
		不考虑相机畸变
		K: intrinstic of camera [fu fv cx cy]
		c2w: camera to world transformation
	"""
	i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
	dirs = np.stack([(i-K[2])/K[0], (j-K[3])/K[1], np.ones_like(i), np.ones_like(i)], -1)  #  [h,w,4]
	dirs = np.reshape(dirs, (H*W, 4, 1))

	# Rotate ray directions from camera frame to the world frame
	c2w = np.concatenate([c2w, [[0,0,0,1]]], axis=0) # [4,4]
	c2w = c2w.reshape(1, 4, 4)

	rays_d = np.matmul(c2w, dirs)[:, :, 0]
	rays_d = rays_d.reshape(H, W, 4)[:, :, :3]
	# Translate camera frame's origin to the world frame. It is the origin of all rays.
	rays_o = np.broadcast_to(c2w[0, :3, -1], np.shape(rays_d))
	rays_d = rays_d - rays_o
	return rays_o, rays_d # [h,w,3]  [h,w,3]

def get_pose_from_xml(xml_path):
	pose_xml = parse(xml_path)
	node_0, node_1 = pose_xml.getElementsByTagName("data")
	R = list(map(float, node_0.childNodes[0].data.strip().split()))
	R = np.asarray(R, dtype=np.float32).reshape((3,3))
	
	T = list(map(float, node_1.childNodes[0].data.strip().split()))
	T = np.asarray(T, dtype=np.float32).reshape((3,1))

	# inverse R,T
	R,T = rt_inverse(R, T)
	return np.concatenate((R,T), axis=-1)

def get_intrinsic_from_xml(xml_path):
	intrinsic_xml = parse(xml_path)
	node_0, node_1 = intrinsic_xml.getElementsByTagName("data")
	K = list(map(float, node_0.childNodes[0].data.strip().split()))
	K = np.asarray([K[0], K[4], K[2], K[5]], dtype=np.float32).reshape((4, ))

	D = list(map(float, node_1.childNodes[0].data.strip().split()))
	D = np.asarray(D, dtype=np.float32).reshape((5, ))
	return K, D


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

		self.output = scalar()

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
		return [self.weights1, self.bias1]

	@ti.kernel
	def weights_init(self):
		q1 = ti.sqrt(6 / self.n_input) * 0.01
		for i, j in ti.ndrange(self.n_output, self.n_input):
			self.weights1[i, j] = (ti.random() * 2 - 1) * q1

	@ti.kernel
	def _forward(self, nn_input: ti.template()):
		for k, i, j in ti.ndrange(self.batch_size // 4, self.n_output, self.n_input):
			for l in ti.static(range(4)):
				base = 4*k
				self.output[base + l, i] += self.weights1[i,j] * nn_input[base + l, j]
		
		if ti.static(self.activation):
			for k, i in ti.ndrange(self.batch_size, self.n_output):
				self.output[k, i] = ti.max(self.output[k, i] + self.bias1[i], 0)
		else:
			for k, i in ti.ndrange(self.batch_size, self.n_output):
				self.output[k, i] = self.output[k, i] + self.bias1[i]

	@ti.kernel
	def clear(self):
		for I in ti.grouped(self.output):
			self.output[I] = 0.

	def forward(self, nn_input):
		self._forward(nn_input)

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


def init_data(root_path, para_path):
	global training_data
	root_path = root_path.replace("\\", '/')
	scene_name = root_path.split("/")[-2]
	para_path = os.path.join(para_path, scene_name)

	image_path = sorted(glob.glob(root_path +  "/*.jpg"))
	n_frames = len(image_path)
	
	poses = []
	intrinsics = []

	# read poses and intrinsics
	for img in image_path:
		# get cam id
		cam_id = img.split(".")[-2]
		cam_id = str(int(cam_id[3:5]))
		pose_path = os.path.join(para_path, cam_id, 'extrinsics.xml')
		pose = get_pose_from_xml(pose_path) # ndarray
		poses.append(pose)

		intrinsic_path = os.path.join(para_path, cam_id, 'intrinsic.xml')
		K,D = get_intrinsic_from_xml(intrinsic_path)
		# only use k now
		intrinsics.append(K)

	rays = []

	with Bar('getting all images rays', max=n_frames) as bar:
		for idx, path in enumerate(image_path):
			img = cv2.imread(path)
			img = img / 255.
			H, W, _ = img.shape
			rays_o, rays_d = get_rays_np(H, W, K = intrinsics[idx], c2w = poses[idx]) # [h,w,3]  [h,w,3]
			ray = np.stack([rays_o, rays_d, img], axis=2) # [H, W, ro+rd+rgb, 3]
			rays.append(ray.reshape((H*W, 3, 3)).astype(np.float32))
			bar.next()
			break
	training_data = np.concatenate(rays, axis=0) # [N*H*W, 3, 3]
	

@ti.kernel
def init_embeddings():
	for i,j in ti.ndrange(n_levels, 2 ** log2_hashmap_size):
		embeddings[i, j] = (ti.random(dtype=float) * 0.0002 - 0.0001)

def init_nn_model(config): 
	global BATCH_SIZE, rays_o, rays_d, viewdirs, pts, target
	global loss, primes_1
	global optimizer
	global bounding_box, logspace, N_samples, z_vals
	global n_levels, n_features_per_level, base_resolution, log2_hashmap_size, finest_resolution
	global resolutions, embeddings, sigma_input, color_input, grid_size, box_offsets
	global sigma_output, last_color
	global sigma1, sigma2, color1, color2, color3

	learning_rate = 1e-2
	N_samples = 128
	hidden = 64


	n_levels = 16
	n_features_per_level = 2
	base_resolution = 16
	log2_hashmap_size = 19
	finest_resolution = 1024
	hash_b = math.exp((math.log(finest_resolution) - math.log(base_resolution)) / (n_levels-1))
	

	BATCH_SIZE = 4096
	bounding_box = ti.Vector.field(3, dtype=float, shape=2)
	logspace = ti.field(float, shape=N_samples+1)
	resolutions = ti.field(float, shape=n_levels)
	grid_size = ti.Vector.field(3, dtype=float, shape=n_levels)
	box_offsets = ti.Vector.field(3, dtype=int, shape=8)
	
	rays_o = ti.Vector.field(3, dtype=float, shape=BATCH_SIZE)
	rays_d = ti.Vector.field(3, dtype=float, shape=BATCH_SIZE)
	viewdirs = ti.Vector.field(3, dtype=float, shape=BATCH_SIZE)
	
	z_vals = ti.field(float, shape=(BATCH_SIZE, N_samples))
	pts = ti.Vector.field(3, dtype=float, shape=(BATCH_SIZE * N_samples))
	
	sigma_input = ti.field(float, shape=(BATCH_SIZE * N_samples, n_features_per_level * n_levels)) # 32
	color_input = ti.field(float, shape=(BATCH_SIZE * N_samples, 31)) # 15 + 16
	sigma_output = ti.field(float, shape=(BATCH_SIZE * N_samples))

	embeddings = ti.Vector.field(n_features_per_level, dtype=float, shape=(n_levels, 2 ** log2_hashmap_size), needs_grad=True)

	last_color = ti.Vector.field(3, dtype=float, shape=BATCH_SIZE)
	target = ti.Vector.field(3, dtype=float, shape=BATCH_SIZE)
	loss = ti.field(float, shape=(), needs_grad=True)

	primes_1 = ti.field(ti.i64, shape=())
	primes_1[None] = 2654435761

	bounding_box_np = np.array([[-5, -5, -5], [5, 5, 1]], dtype=np.float32)
	bounding_box.from_numpy(bounding_box_np)

	logspace_np = np.linspace(0., 1., num=N_samples+1, dtype=np.float32)
	logspace.from_numpy(logspace_np)
	
	resolutions_np = np.array([np.floor(base_resolution * (hash_b **i)) for i in range(n_levels)], dtype=np.float32)
	resolutions.from_numpy(resolutions_np)

	grid_size_np = (bounding_box_np[1:] - bounding_box_np[0:1]) / np.expand_dims(resolutions_np, axis=1) # 16,3
	grid_size.from_numpy(grid_size_np)

	box_offsets_np = np.array([[[i,j,k] for i in [0, 1] for j in [0, 1] for k in [0, 1]]], dtype = np.int32)
	box_offsets.from_numpy(box_offsets_np)

	# init embeddings
	# uniform_
	init_embeddings()

	sigma1 = Linear(batch_size=BATCH_SIZE * N_samples,
					n_input = n_features_per_level * n_levels,
					n_output = hidden,
					needs_grad=True,
					activation=True)
	sigma2 = Linear(batch_size = BATCH_SIZE * N_samples,
					n_input = hidden,
					n_output= 16,
					needs_grad=True,
					activation=False)
	color1 = Linear(batch_size=BATCH_SIZE * N_samples,
					n_input= 15 + 16,
					n_output = hidden,
					needs_grad=True,
					activation=True)
	color2 = Linear(batch_size=BATCH_SIZE * N_samples,
					n_input = hidden,
					n_output = hidden,
					needs_grad=True,
					activation=True)
	color3 = Linear(batch_size=BATCH_SIZE * N_samples,
					n_input= hidden,
					n_output = 3,
					needs_grad=True,
					activation=False)

	sigma1.weights_init()
	sigma2.weights_init()
	color1.weights_init()
	color2.weights_init()
	color3.weights_init()

	NNs = [sigma1, sigma2, color1, color2, color3]
	
	parameters = []
	for layer in NNs:
		parameters.extend(layer.parameters())

	parameters.append(embeddings)

	optimizer = SGD(params=parameters, lr=learning_rate)


@ti.kernel
def fill_rays(data: ti.types.ndarray()):
	for i in range(BATCH_SIZE):
		for j in ti.static(range(3)):
			rays_o[i][j] = data[i, 0, j]
			rays_d[i][j] = data[i, 1, j]
			target[i][j] = data[i, 2, j]


@ti.kernel
def cal_viewdirs():
	for i in range(BATCH_SIZE):
		viewdirs[i] =  rays_d[i].normalized(eps = 1e-6)


@ti.func
def lindisp(idx, near, gap):
	"""
		from logspace
	"""
	duan = 1. / N_samples
	for i in range(N_samples):
		# i~i+1 random a value
		z_vals[idx, i] = near + gap * (logspace[i] + ti.random(dtype=float) * duan)
		

@ti.kernel
def get_z_vals():
	for i in range(BATCH_SIZE):
		tmin = (bounding_box[0] - rays_o[i]) / rays_d[i]
		tmax = (bounding_box[1] - rays_o[i]) / rays_d[i]
		far = min(max(tmin[0], tmax[0]), max(tmin[1], tmax[1]), max(tmin[2], tmax[2]))
		near = 0.05
		lindisp(i, near, far - near)
		

@ti.func
def get_pts_func(idx):
	for i in range(N_samples):
		pts[idx * N_samples + i] = rays_o[idx] + rays_d[idx] * z_vals[idx, i]


@ti.kernel
def get_pts():
	for idx in range(BATCH_SIZE):
		get_pts_func(idx)


@ti.kernel
def tail_deal_sigma():
	for idx in range(BATCH_SIZE * N_samples):
		sigma_output[idx] = sigma2.output[idx, 0]

		for i in range(1, 16):
			color_input[idx, i-1] = sigma2.output[idx, i]

primes_0 = 1
primes_2 = 805459861

@ti.func
def hash(v):
	xor_result = v[0] ^ (v[1] * primes_1[None])
	xor_result = xor_result ^ (v[2] * primes_2)
	return ti.cast(((1<<log2_hashmap_size) - 1) & xor_result, ti.i32)


@ti.func
def per_level_fill(idx, level):
	# cal 8 hash index
	bottom_left_idx = ti.cast(ti.floor((pts[idx] - bounding_box[0]) / grid_size[level]), ti.i32)
	
	# hash 0
	hash_0 = hash(bottom_left_idx) # int

	# 1
	hash_1 = hash(bottom_left_idx + box_offsets[1])

	# 2
	hash_2 = hash(bottom_left_idx + box_offsets[2])

	# 3
	hash_3 = hash(bottom_left_idx + box_offsets[3])

	# 4
	hash_4 = hash(bottom_left_idx + box_offsets[4])

	# 5
	hash_5 = hash(bottom_left_idx + box_offsets[5])

	# 6
	hash_6 = hash(bottom_left_idx + box_offsets[6])

	# 7
	hash_7 = hash(bottom_left_idx + box_offsets[7])

	voxel_min_vertex = bottom_left_idx * grid_size[level] + bounding_box[0] # 3

	# interpolate according to indexes
	weights = (pts[idx] - voxel_min_vertex) / grid_size[level]

	# step1
	c00 = embeddings[level, hash_0] * (1-weights[0]) + embeddings[level, hash_4] * weights[0]
	c01 = embeddings[level, hash_1] * (1-weights[0]) + embeddings[level, hash_5] * weights[0]
	c10 = embeddings[level, hash_2] * (1-weights[0]) + embeddings[level, hash_6] * weights[0]
	c11 = embeddings[level, hash_3] * (1-weights[0]) + embeddings[level, hash_7] * weights[0]

    # step 2
	c0 = c00 * (1-weights[1]) + c10 * weights[1]
	c1 = c01 * (1-weights[1]) + c11 * weights[1]

	# step 3
	c = c0 * (1-weights[2]) + c1 * weights[2]

	return c


@ti.kernel
def fill_inputs():
	tot = N_samples * BATCH_SIZE * n_levels
	for i in range(tot):
		level = i % n_levels
		idx = i // n_levels
		c = per_level_fill(idx, level)
		# write to input
		base = level * n_features_per_level
		for j in ti.static(range(n_features_per_level)):
			sigma_input[idx, base + j] = c[j]


C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
	1.0925484305920792,
	-1.0925484305920792,
	0.31539156525252005,
	-1.0925484305920792,
	0.5462742152960396
]
C3 = [
	-0.5900435899266435,
	2.890611442640554,
	-0.4570457994644658,
	0.3731763325901154,
	-0.4570457994644658,
	1.445305721320277,
	-0.5900435899266435
]

@ti.kernel
def fill_views():
	base = 15
	for idx in range(N_samples * BATCH_SIZE):
		batch_idx = idx // N_samples
		x = viewdirs[batch_idx][0]
		y = viewdirs[batch_idx][1]
		z = viewdirs[batch_idx][2]

		color_input[idx, base] = C0
		color_input[idx, base + 1] = -C1 * y
		color_input[idx, base + 2] = C1 * z
		color_input[idx, base + 3] = -C1 * x
		xx, yy, zz = x * x, y * y, z * z
		xy, yz, xz = x * y, y * z, x * z
		color_input[idx, base + 4] = C2[0] * xy
		color_input[idx, base + 5] = C2[1] * yz
		color_input[idx, base + 6] = C2[2] * (2.0 * zz - xx - yy)
		color_input[idx, base + 7] = C2[3] * xz
		color_input[idx, base + 8] = C2[4] * (xx - yy)
		color_input[idx, base + 9] = C3[0] * y * (3 * xx - yy)
		color_input[idx, base + 10] = C3[1] * xy * z
		color_input[idx, base + 11] = C3[2] * y * (4 * zz - xx - yy)
		color_input[idx, base + 12] = C3[3] * z * (2 * zz - 3 * xx - 3 * yy)
		color_input[idx, base + 13] = C3[4] * x * (4 * zz - xx - yy)
		color_input[idx, base + 14] = C3[5] * z * (xx - yy)
		color_input[idx, base + 15] = C3[6] * x * (xx - 3 * yy)


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
	init_data(root_path=cfg.root_path, para_path=cfg.para_path)

	# init all model and mid variables
	init_nn_model(cfg)

	# start to train
	losses = []
	losses_epoch_avg = []
	epochs = 20
	for epoch in range(epochs):
		loss_epoch = 0.0
		cnt = 0
		for current_data_offset in range(0, len(training_data), BATCH_SIZE):
			# fill rays_o rays_d target
			fill_rays(training_data[current_data_offset:current_data_offset + BATCH_SIZE])

			# cal viewdirs
			cal_viewdirs()

			# get z_vals according box
			get_z_vals()

			# cal points for each ray
			get_pts()

			# get index and interpolate to fill
			fill_inputs()

			# fill viewdirs encoding for mlp
			fill_views()

			# cal weights using linears
			sigma1.forward(sigma_input)
			sigma2.forward(sigma1.output)

			tail_deal_sigma() # sigma_output: [bn, ]

			color1.forward(color_input)
			color2.forward(color1.output)
			color3.forward(color2.output)
			# color3.output:  [bn, 3]

			with ti.Tape(loss=loss):
				# use features and viewdirs to get weights

				# use weights to get rgb
				
				# compute_loss()
				pass
			
			optimizer.step() # mlp 

			# deal the hashed features's grad by hand according to index, update it

			print(
				f"current epoch: {epoch},  progress: {current_data_offset + BATCH_SIZE}/{len(training_data)}, loss: {loss[None]}"
			)
			losses.append(loss[None])
			loss_epoch += loss[None]
			cnt += 1
		
		print(f'epoch {epoch} done. Average loss: {loss_epoch / cnt}')
		losses_epoch_avg.append(loss_epoch / cnt)


if __name__ == "__main__":
	timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
	main(timestamp)
