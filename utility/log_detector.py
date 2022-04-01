# from skimage.feature.blob import *
# from skimage.feature.blob import _blob_overlap, _prune_blobs
# from skimage.feature import hessian_matrix
# from skimage.filters import threshold_niblack
# from sklearn.neighbors import KDTree
# import cv2
#
# def elimination_overlap(blob, overlap):
# 	remove_mask = []
# 	pts = blob[:, :2]
# 	tree = KDTree(pts, leaf_size=200)
# 	sets = tree.query_radius(pts, r=blob[:, 2].max() * np.sqrt(2))
# 	for key, set in enumerate(sets):
# 		if len(set) > 1:
# 			blob1 = blob[key]
# 			for s in set:
# 				if s != key:
# 					blob2 = blob[s]
# 					if _blob_overlap(blob1, blob2) > overlap and blob1[2] <= blob2[2]:
# 						remove_mask += [key]
# 						break
# 	return np.delete(blob, remove_mask, axis=0)
#
#
# def get_adaptive_threshold(img_gray, landmarks, __range = [600, 650]):
# 	cheek = get_cheek(img_gray, landmarks)
#
# 	## bypass binary search
# 	blobs = log_detect(cheek, 0, eliminate=False)
# 	if len(blobs) < __range[0]:
# 		threshold = 0
# 	else:
# 		## binary search
# 		threshold_max = 1.0
# 		threshold_min = 0.
# 		threshold = (threshold_max - threshold_min) / 2
# 		num_pore = 0.
# 		while not (__range[0] <= num_pore <= __range[1]):
# 			blobs = log_detect(cheek, threshold, eliminate=False)
# 			num_pore_new = len(blobs)
# 			if num_pore_new == 0:
# 				threshold_max = threshold
# 				threshold = threshold_min + (threshold_max - threshold_min) / 2
# 			else:
# 				if num_pore_new < __range[0]:
# 					threshold_max = threshold
# 					threshold = threshold_min + (threshold_max - threshold_min) / 2
# 				elif num_pore_new > __range[1]:
# 					threshold_min = threshold
# 					threshold = threshold_min + (threshold_max - threshold_min) / 2
# 				num_pore = num_pore_new
#
# 	return threshold, num_pore
#
#
# def get_cheek(img_gray, landmarks):
# 	d = np.linalg.norm((landmarks[1] - landmarks[3]), ord=2)
# 	cheek_pos = landmarks[3] + [0.3 * d, -0.5 * d]
# 	return cv2.getRectSubPix(img_gray, (int(0.6 * d), int(0.6 * d)), tuple(cheek_pos))
#
#
# def log_detect(image_gray, threshold=0.006, overlap=.5, eliminate=True):
# 	# Modify from https://github.com/scikit-image/scikit-image/blob/master/skimage/feature/blob.py#L214
# 	__parameters = {"min_sigma": 1.0 / np.sqrt(2),
# 	                "max_sigma": 5.5 / np.sqrt(2),
# 	                "num_sigma": 10,
# 	                "th_offset": 25,
# 	                "ratio": 5
# 	                }
#
# 	min_sigma = __parameters["min_sigma"]
# 	max_sigma = __parameters["max_sigma"]
# 	num_sigma = __parameters["num_sigma"]
# 	th_offset = __parameters["th_offset"]
# 	ratio = __parameters["ratio"]
#
# 	assert_nD(image_gray, 2)
#
# 	image = img_as_float(image_gray)
# 	sigma_list = np.linspace(min_sigma, max_sigma, num_sigma)
#
# 	# computing gaussian laplace
# 	# s**2 provides scale invariance
# 	gl_images = [gaussian_laplace(image, s) * s ** 2 for s in sigma_list]
# 	image_cube = np.dstack(gl_images)
#
# 	local_maxima = peak_local_max(image_cube, threshold_abs=threshold,
# 	                              footprint=np.ones((3, 3, 3)),
# 	                              threshold_rel=0.0,
# 	                              exclude_border=True)
# 	# Catch no peaks
# 	if local_maxima.size == 0:
# 		return np.empty((0, 3))
#
# 	# Convert local_maxima to float32
# 	lm = local_maxima.astype(np.float32)
# 	# Convert the last index to its corresponding scale value
# 	lm[:, 2] = sigma_list[local_maxima[:, 2]]
#
# 	if eliminate:
# 		th = np.mean(image_gray) - th_offset
# 		binary_image = threshold_niblack(image_gray, window_size=31, k=0) > th
# 		hessian_matrices = [hessian_matrix(image, sigma=s, order="xy") for s in sigma_list]
# 		mask = []
#
# 		# for key, (_y, _x, _layer) in enumerate(lm):
# 		# 	x, y, layer = np.int32(_x), np.int32(_y), np.int32(_layer)
# 		# 	if binary_image[y][x] == True:
# 		# 		if ratio is not None:
# 		# 			D = hessian_matrices[layer]
# 		# 			Dxx, Dxy, Dyy = D[0][y][x], D[1][y][x], D[2][y][x]
# 		# 			TrH = Dxx + Dyy
# 		# 			DetH = Dxx * Dyy - Dxy ** 2
# 		# 			if TrH ** 2 / DetH < ratio:
# 		# 				mask += [key]
# 		# 		else:
# 		# 			mask += [key]
# 		# local_maxima = lm[mask]
#
# 		for key, (_y, _x, _layer) in enumerate(lm):
# 			x, y, layer = np.int32(_x), np.int32(_y), np.int32(_layer)
# 			if binary_image[y][x] == True:
# 				if ratio is not None:
# 					D = hessian_matrices[layer]
# 					Dxx, Dxy, Dyy = D[0][y][x], D[1][y][x], D[2][y][x]
# 					TrH = Dxx + Dyy
# 					DetH = Dxx * Dyy - Dxy ** 2
# 					if TrH ** 2 / DetH < ratio:
# 						mask += [key]
# 				else:
# 					mask += [key]
# 			if ratio is not None:
# 				D = hessian_matrices[layer]
# 				Dxx, Dxy, Dyy = D[0][y][x], D[1][y][x], D[2][y][x]
# 				TrH = Dxx + Dyy
# 				DetH = Dxx * Dyy - Dxy ** 2
# 				if TrH ** 2 / DetH < ratio:
# 					mask += [key]
# 		local_maxima = lm[mask]
#
# 	# yx -> xy
# 	local_maxima[:, [0, 1]] = local_maxima[:, [1, 0]]
#
# 	return elimination_overlap(local_maxima, overlap)


from skimage.feature.blob import *
from skimage.feature.blob import _blob_overlap, _prune_blobs
from skimage.feature import hessian_matrix
from skimage.filters import threshold_niblack
from sklearn.neighbors import KDTree


def elimination_overlap(blob, overlap):
	remove_mask = []
	pts = blob[:, :2]
	tree = KDTree(pts, leaf_size=200)
	sets = tree.query_radius(pts, r=blob[:, 2].max() * np.sqrt(2))
	for key, set in enumerate(sets):
		if len(set) > 1:
			blob1 = blob[key]
			for s in set:
				if s != key:
					blob2 = blob[s]
					if _blob_overlap(blob1, blob2) > overlap and blob1[2] <= blob2[2]:
						remove_mask += [key]
						break
	return np.delete(blob, remove_mask, axis=0)


# Modify from https://github.com/scikit-image/scikit-image/blob/master/skimage/feature/blob.py#L214


def log_detect(image_gray, threshold=0.006, overlap=.5, eliminate=True):
	__parameters = {"min_sigma": 1.0 / np.sqrt(2),
	                "max_sigma": 5.5 / np.sqrt(2),
	                "num_sigma": 10,
	                "threshold": 0.006,
	                "th_offset": 25,
	                "ratio": 5
	                }
	min_sigma = __parameters["min_sigma"]
	max_sigma = __parameters["max_sigma"]
	num_sigma = __parameters["num_sigma"]
	threshold = threshold
	th_offset = __parameters["th_offset"]
	ratio = __parameters["ratio"]

	th = np.mean(image_gray) - th_offset
	binary_image = threshold_niblack(image_gray, window_size=31, k=0) > th

	image = img_as_float(image_gray)
	sigma_list = np.linspace(min_sigma, max_sigma, num_sigma)

	# computing gaussian laplace
	# s**2 provides scale invariance
	gl_images = [gaussian_laplace(image, s) * s ** 2 for s in sigma_list]
	hessian_matrices = [hessian_matrix(image, sigma=s, order="xy") for s in sigma_list]
	image_cube = np.dstack(gl_images)

	local_maxima = peak_local_max(image_cube, threshold_abs=threshold,
	                              footprint=np.ones((3, 3, 3)),
	                              threshold_rel=0.0,
	                              exclude_border=True)
	# Catch no peaks
	if local_maxima.size == 0:
		return np.empty((0, 3))
	# Convert local_maxima to float64s
	lm = local_maxima.astype(np.float64)
	# Convert the last index to its corresponding scale value

	mask = []
	for key, (_y, _x, _layer) in enumerate(lm):
		x, y, layer = np.int32(_x), np.int32(_y), np.int32(_layer)
		# if binary_image[y][x] == True:
		if ratio is not None:
			D = hessian_matrices[layer]
			Dxx, Dxy, Dyy = D[0][y][x], D[1][y][x], D[2][y][x]
			TrH = Dxx + Dyy
			DetH = Dxx * Dyy - Dxy ** 2
			if TrH ** 2 / DetH < ratio:
				mask += [key]
		else:
			mask += [key]

	lm[:, 2] = sigma_list[local_maxima[:, 2]]
	local_maxima = lm[mask]
	local_maxima[:, [0, 1]] = local_maxima[:, [1, 0]]

	return elimination_overlap(local_maxima, overlap)
