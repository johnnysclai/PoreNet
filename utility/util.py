def normalize_points(pts, size, npts):
	# size in 'x, y' order
	for pt in pts:
		npts.append((pt[0] / size[0], pt[1] / size[1]))