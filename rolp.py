import cv2, os
import numpy as np
from matplotlib import pyplot as plt
import img_plotter

global kernel
def initialize_kernel(a):
	global kernel
	w_1d = np.array([0.25 - a/2.0, 0.25, a, 0.25, 0.25 - a/2.0])
	kernel = np.outer(w_1d, w_1d)

def main():
	one = readImage('tank_viz.tif')
	two = readImage('tank_thm.tif')

	initialize_kernel(0.4)
	fused = fuse(one, two, levels=3)

	imgs = [one, two, fused]
	titles = ['one', 'two', 'fused']
	img_plotter.plot_images(imgs=imgs, titles=titles, suptitle='ROLP', cols=3)

dataset = './'
def readImage(fn):
	file_path = os.path.join(dataset, fn)
	return cv2.imread(file_path, 0)

def fuse(im_one, im_two, levels=3):
	rows, cols = im_one.shape

	im_rz_one = resize_img(im_one, levels=levels)
	im_rz_two = resize_img(im_two, levels=levels)

	one = normalize(im_rz_one)
	two = normalize(im_rz_two)

	gp_one = build_gaussian_pyramid(one, levels=levels)
	gp_two = build_gaussian_pyramid(two, levels=levels)

	rp_one = build_rolp_pyramid(gp_one)
	rp_two = build_rolp_pyramid(gp_two)

	rp_fus = apply_fusion_rules(rp_one, rp_two)

	fused = reconstruct_rolp(rp_fus)
	fused = np.rint(255*fused).astype(np.uint8)
	return cv2.resize(fused, (cols, rows), interpolation=cv2.INTER_CUBIC)

def resize_img(img, levels):
	rows, cols = img.shape
	rows = getNextValidDimension(rows, D=levels)
	cols = getNextValidDimension(cols, D=levels)
	return cv2.resize(img, (cols, rows), interpolation=cv2.INTER_CUBIC)

def getNextValidDimension(x, D):
	while x % 2**D != 0:
		x *= 2
	return x

def normalize(img):
	return np.true_divide(img, 255)

def build_gaussian_pyramid(img, levels=1):
	gp = [img]
	for i in range(levels):
		G = REDUCE( gp[-1] )
		gp.append(G)
	return gp

def REDUCE(G):
	filtered = cv2.filter2D(G, -1, kernel)
	return filtered[::2, ::2]

def build_rolp_pyramid(gp):
	rp = []
	for i in range( len(gp) - 1 ):
		G = EXPAND( gp[i+1] )
		R = np.true_divide(gp[i], G)
		rp.append(R)
	rp += [  gp[-1]  ]
	return rp

def EXPAND(G):
	shape = (2*G.shape[0], 2*G.shape[1])
	im = np.zeros(shape, dtype=np.float64)
	im[::2, ::2] = G
	return 4*cv2.filter2D(im, -1, kernel)

def apply_fusion_rules(rpA, rpB):
	rpC = []
	for RA, RB in zip(rpA, rpB):
		RC = RB.copy()
		RA_mag, RB_mag = np.abs(RA - 1), np.abs(RB - 1)
		
		indexes = np.where(RA_mag > RB_mag)
		RC[indexes] = RA[indexes]
		
		rpC.append(RC)
	
	return rpC

def reconstruct_rolp(rp):
	G = rp[-1]
	for i in range( -2, -len(rp)-1, -1 ):
		G0 = EXPAND(G)
		G = np.multiply(rp[i], G0)
	return G

if __name__ == '__main__':
	main()

