import os

import numpy as np
import cv2 as cv
import glob


def get_camera_parameters():
	# Perform camera calibration to get intrinsic parameters

	# Arrays to store object points and image points from all the images.
	obj_points = []  # 3d point in real world space
	img_points = []  # 2d points in image plane

	# Termination criteria for the corner subpixel refinement
	criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

	# Prepare object points, (0,0,0), (1,0,0), (2,0,0), ..., (7,5,0)
	obj_point_template = np.zeros((8 * 6, 3), np.float32)
	# (2, 8, 6) -> (6, 8, 2) -> (6 * 8, 2)
	obj_point_template[:, :2] = np.mgrid[0:8, 0:6].T.reshape(-1, 2)

	# Declare variable outside loop so my IDE doesn't cry
	gray = None

	# Get paths to all images in the calibration input directory
	base_path = os.path.dirname(os.path.abspath(__file__))
	input_path = os.path.join(base_path, 'data/calibration/*.png')
	# We use the sort to prevent string sorting of 10 before 2 etc.
	# Although order doesn't matter here
	img_paths = glob.glob(input_path)

	# Iterate over all input images
	for i, path in enumerate(img_paths):
		img = cv.imread(path)
		gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

		# Find the chess board corners
		ret, corners = cv.findChessboardCorners(gray, (8, 6), None)

		# If no corners found, skip this image
		if not ret:
			continue

		# If found, we refine the image points
		corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

		# And add the image points along with another set of object points
		obj_points.append(obj_point_template)
		img_points.append(corners)

		# Draw the corners and save the image
		cv.drawChessboardCorners(img, (8, 6), corners2, patternWasFound=True)
		img_name = os.path.basename(path)
		out_path = os.path.join(base_path, f'out/calibration/out_{img_name}')
		cv.imwrite(out_path, img)

	# Return: RMS Re-projection error, _, _, list of estimated rvecs and tvecs
	# Error is about 0.346 for this set of images
	# Between 0.1 and 1.0 is usually good ~ StackOverflow
	_, camera_matrix, distortion_params, _, _ = cv.calibrateCamera(
		obj_points, img_points, gray.shape[::-1], None, None
	)
	return camera_matrix, distortion_params


def main():
	# REVIEW: Save as a json/npy file and read from it?
	mtx, dist = get_camera_parameters()
	print(mtx)
	print(dist)


if __name__ == '__main__':
	main()
