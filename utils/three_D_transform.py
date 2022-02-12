##
import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
from scipy import io


# ## get key point
# root = '/media/khw11044/Samsung_T5/Humandataset/mpi_inf_3dhp/S1/Seq2/'
# mat = io.loadmat(root + 'annot.mat')

output_dtype = float

##

def world_to_camera_frame(P,R,T):

    X_cam = R.dot(P - T)

    return X_cam.T

# x_0 = mat['univ_annot3'][0][0][0][0]
# y_0 = mat['univ_annot3'][0][0][0][1]
# z_0 = mat['univ_annot3'][0][0][0][2]
# u_0 = np.array([[x_0],[y_0],[z_0]])

# x_2 = mat['univ_annot3'][2][0][0][0]
# y_2 = mat['univ_annot3'][2][0][0][1]
# z_2 = mat['univ_annot3'][2][0][0][2]
# u_2 = np.array([[x_2],[y_2],[z_2]])

# R_0 = np.array([[0.9650164, 0.00488022, 0.262144],
#        [-0.004488356, -0.9993728, 0.0351275],
#        [0.262151, -0.03507521, -0.9643893]])

# T_0 = np.array([[-562.8666],
#        [1398.138],
#        [3852.623]])

# R_2 = np.array([[ -0.3608179, -0.009492658, 0.932588],
#        [-0.0585942, -0.9977421, -0.03282591],
#        [0.9307939, -0.06648842, 0.359447]])

# T_2 = np.array([[57.25702],
#        [1307.287],
#        [2799.822]])

# dd = world_to_camera_frame(u_0, R_0, T_0)
# ee = world_to_camera_frame(u_2, R_2, T_2)

## 3d 변환
def linear_eigen_triangulation(u1, P1, u2, P2, max_coordinate_value=1.e16):  # two_2.T, p_nom, two_0.T, proj
    x = cv2.triangulatePoints(P1[0:3, 0:4], P2[0:3, 0:4], u1.T, u2.T)  # OpenCV's Linear-Eigen triangl

    # x[0:3, :] /= x[3:4, :]  # normalize coordinates
    x_status = (np.max(abs(x[0:3, :]), axis=0) <= max_coordinate_value)  # NaN or Inf will receive status False

    x = x[0:3, :].T
    return x, x_status

# # ex_mat_0 = np.concatenate((R_0,T_0),axis=1)
# #
# # ex_mat_2 = np.concatenate((R_2,T_2),axis=1)
# # pro_0 = np.dot(intri_0,ex_mat_0)
# # pro_2 = np.dot(intri_2,ex_mat_2)
# x2_2 = np.array(mat['annot2'][2][0][0][0])
# y2_2 = np.array(mat['annot2'][2][0][0][1])
# u2_2 = np.array([x2_2, y2_2])
# x2_0 = np.array(mat['annot2'][0][0][0][0])
# y2_0 = np.array(mat['annot2'][0][0][0][1])
# u2_0 = np.array([x2_0, y2_0])
# #
# # sss, _ =linear_eigen_triangulation(u2_2,pro_2,u2_0,pro_0)
# # print(sss)


def estimate_relative_pose_from_correspondence(pts1, pts2, K1, K2): # K : 내부 파라미터 
    f_avg = (K1[0, 0] + K2[0, 0]) / 2
    pts1, pts2 = np.ascontiguousarray(pts1, np.float32), np.ascontiguousarray(pts2, np.float32) # 메모리에 연속 배열 (ndim> = 1)을 반환

    pts_l_norm = cv2.undistortPoints(np.expand_dims(pts1, axis=1), cameraMatrix=K1, distCoeffs=None)    # 왜곡을 없애는 함수
    pts_r_norm = cv2.undistortPoints(np.expand_dims(pts2, axis=1), cameraMatrix=K2, distCoeffs=None)

    E, mask = cv2.findEssentialMat(pts_l_norm, pts_r_norm, focal=1.0, pp=(0., 0.),                      # https://www.programcreek.com/python/example/110761/cv2.findEssentialMat
                                   method=cv2.RANSAC, prob=0.999, threshold=3.0 / f_avg)    # E : EssentialMatrix

    points, R_est, t_est, mask_pose = cv2.recoverPose(E, pts_l_norm, pts_r_norm)
    return mask[:, 0].astype(bool), R_est, t_est


# pts_2 = np.array([[mat['annot2'][2][0][0][0], mat['annot2'][2][0][0][1]]])
# pts_0 = np.array([[mat['annot2'][0][0][0][0], mat['annot2'][0][0][0][1]]])
# for i in range(6):
#     pts_2 = np.concatenate((pts_2 , np.array([[mat['annot2'][2][0][0][0+2*i], mat['annot2'][2][0][0][1+2*i]]])),axis=0)
#     pts_0 = np.concatenate((pts_0 , np.array([[mat['annot2'][0][0][0][0+2*i], mat['annot2'][0][0][0][1+2*i]]])),axis=0)
# intri_2 = np.array([[1495.587, 0, 983.8873],
#                     [0, 1497.828, 987.5902],
#                     [0, 0, 1]])
# intri_0 = np.array([[1497.693, 0, 1024.704],
#                     [0, 1497.103, 1051.394],
#                     [0, 0, 1]])
# _, R_est, T_est = estimate_relative_pose_from_correspondence(pts_2,pts_0,intri_2,intri_0)



# p_nom = np.dot(intri_2,np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]]))
# proj = np.dot(intri_0,np.concatenate((R_est,T_est),axis=1))

# new_3d_0, _ = linear_eigen_triangulation(u2_2, p_nom, u2_0, proj)
# new_3d_1, _ = linear_eigen_triangulation(u2_2, proj,u2_0, p_nom)


def rotation_xz(a_,b_,c_):
    a = np.array([a_[9], b_[9], c_[9]])
    for i in range(len(a_)):
        point = np.array([a_[i], b_[i], c_[i]])
        len_xy = (a[0] ** 2 + a[1] ** 2) ** 0.5
        ro_z = np.array([[a[1] / len_xy, -a[0] / len_xy, 0],
                [a[0] / len_xy, a[1] / len_xy, 0],
                [0, 0, 1]])
        T = np.array([[-a_[14]],
                      [-b_[14]],
                      -c_[14]])
        world_to_camera_frame(point,ro_z,T)

# Initialize consts to be used in linear_LS_triangulation()
linear_LS_triangulation_C = -np.eye(2, 3)


def linear_LS_triangulation(u1, P1, u2, P2):
	"""
	Linear Least Squares based triangulation.
	Relative speed: 0.1

	(u1, P1) is the reference pair containing normalized image coordinates (x, y) and the corresponding camera matrix.
	(u2, P2) is the second pair.

	u1 and u2 are matrices: amount of points equals #rows and should be equal for u1 and u2.

	The status-vector will be True for all points.
	"""
	A = np.zeros((4, 3))
	b = np.zeros((4, 1))

	# Create array of triangulated points
	x = np.zeros((3, len(u1)))

	# Initialize C matrices
	C1 = np.array(linear_LS_triangulation_C)
	C2 = np.array(linear_LS_triangulation_C)

	for i in range(len(u1)):
		# Derivation of matrices A and b:
		# for each camera following equations hold in case of perfect point matches:
		#     u.x * (P[2,:] * x)     =     P[0,:] * x
		#     u.y * (P[2,:] * x)     =     P[1,:] * x
		# and imposing the constraint:
		#     x = [x.x, x.y, x.z, 1]^T
		# yields:
		#     (u.x * P[2, 0:3] - P[0, 0:3]) * [x.x, x.y, x.z]^T     +     (u.x * P[2, 3] - P[0, 3]) * 1     =     0
		#     (u.y * P[2, 0:3] - P[1, 0:3]) * [x.x, x.y, x.z]^T     +     (u.y * P[2, 3] - P[1, 3]) * 1     =     0
		# and since we have to do this for 2 cameras, and since we imposed the constraint,
		# we have to solve 4 equations in 3 unknowns (in LS sense).

		# Build C matrices, to construct A and b in a concise way
		C1[:, 2] = u1[i, :]
		C2[:, 2] = u2[i, :]

		# Build A matrix:
		# [
		#     [ u1.x * P1[2,0] - P1[0,0],    u1.x * P1[2,1] - P1[0,1],    u1.x * P1[2,2] - P1[0,2] ],
		#     [ u1.y * P1[2,0] - P1[1,0],    u1.y * P1[2,1] - P1[1,1],    u1.y * P1[2,2] - P1[1,2] ],
		#     [ u2.x * P2[2,0] - P2[0,0],    u2.x * P2[2,1] - P2[0,1],    u2.x * P2[2,2] - P2[0,2] ],
		#     [ u2.y * P2[2,0] - P2[1,0],    u2.y * P2[2,1] - P2[1,1],    u2.y * P2[2,2] - P2[1,2] ]
		# ]
		A[0:2, :] = C1.dot(P1[0:3, 0:3])  # C1 * R1
		A[2:4, :] = C2.dot(P2[0:3, 0:3])  # C2 * R2

		# Build b vector:
		# [
		#     [ -(u1.x * P1[2,3] - P1[0,3]) ],
		#     [ -(u1.y * P1[2,3] - P1[1,3]) ],
		#     [ -(u2.x * P2[2,3] - P2[0,3]) ],
		#     [ -(u2.y * P2[2,3] - P2[1,3]) ]
		# ]
		b[0:2, :] = C1.dot(P1[0:3, 3:4])  # C1 * t1
		b[2:4, :] = C2.dot(P2[0:3, 3:4])  # C2 * t2
		b *= -1

		# Solve for x vector
		cv2.solve(A, b, x[:, i:i + 1], cv2.DECOMP_SVD)

	return x.T.astype(output_dtype), np.ones(len(u1), dtype=bool)
    # return x.T.astype(output_dtype), np.ones(len(u1), dtype=bool)


# Initialize consts to be used in iterative_LS_triangulation()
iterative_LS_triangulation_C = -np.eye(2, 3)


def iterative_LS_triangulation(u1, P1, u2, P2, tolerance=3.e-5):
	"""
	Iterative (Linear) Least Squares based triangulation.
	From "Triangulation", Hartley, R.I. and Sturm, P., Computer vision and image understanding, 1997.
	Relative speed: 0.025

	(u1, P1) is the reference pair containing normalized image coordinates (x, y) and the corresponding camera matrix.
	(u2, P2) is the second pair.
	"tolerance" is the depth convergence tolerance.

	Additionally returns a status-vector to indicate outliers:
		1: inlier, and in front of both cameras
		0: outlier, but in front of both cameras
		-1: only in front of second camera
		-2: only in front of first camera
		-3: not in front of any camera
	Outliers are selected based on non-convergence of depth, and on negativity of depths (=> behind camera(s)).

	u1 and u2 are matrices: amount of points equals #rows and should be equal for u1 and u2.
	"""
	A = np.zeros((4, 3))
	b = np.zeros((4, 1))

	# Create array of triangulated points
	x = np.empty((4, len(u1)))
	x[3, :].fill(1)  # create empty array of homogenous 3D coordinates
	x_status = np.empty(len(u1), dtype=int)

	# Initialize C matrices
	C1 = np.array(iterative_LS_triangulation_C)
	C2 = np.array(iterative_LS_triangulation_C)

	for xi in range(len(u1)):
		# Build C matrices, to construct A and b in a concise way
		C1[:, 2] = u1[xi, :]
		C2[:, 2] = u2[xi, :]

		# Build A matrix
		A[0:2, :] = C1.dot(P1[0:3, 0:3])  # C1 * R1
		A[2:4, :] = C2.dot(P2[0:3, 0:3])  # C2 * R2

		# Build b vector
		b[0:2, :] = C1.dot(P1[0:3, 3:4])  # C1 * t1
		b[2:4, :] = C2.dot(P2[0:3, 3:4])  # C2 * t2
		b *= -1

		# Init depths
		d1 = d2 = 1.

		for i in range(10):  # Hartley suggests 10 iterations at most
			# Solve for x vector
			cv2.solve(A, b, x[0:3, xi:xi + 1], cv2.DECOMP_SVD)

			# Calculate new depths
			d1_new = P1[2, :].dot(x[:, xi])
			d2_new = P2[2, :].dot(x[:, xi])

			if abs(d1_new - d1) <= tolerance and \
							abs(d2_new - d2) <= tolerance:
				break

			# Re-weight A matrix and b vector with the new depths
			A[0:2, :] *= 1 / d1_new
			A[2:4, :] *= 1 / d2_new
			b[0:2, :] *= 1 / d1_new
			b[2:4, :] *= 1 / d2_new

			# Update depths
			d1 = d1_new
			d2 = d2_new

		# Set status
		x_status[xi] = (i < 10 and  # points should have converged by now
		                (d1_new > 0 and d2_new > 0))  # points should be in front of both cameras
		if d1_new <= 0: x_status[xi] -= 1
		if d2_new <= 0: x_status[xi] -= 2

	return x[0:3, :].T.astype(output_dtype), x_status


def polynomial_triangulation(u1, P1, u2, P2):
	"""
	Polynomial (Optimal) triangulation.
	Uses Linear-Eigen for final triangulation.
	Relative speed: 0.1

	(u1, P1) is the reference pair containing normalized image coordinates (x, y) and the corresponding camera matrix.
	(u2, P2) is the second pair.

	u1 and u2 are matrices: amount of points equals #rows and should be equal for u1 and u2.

	The status-vector is based on the assumption that all 3D points have finite coordinates.
	"""
	P1_full = np.eye(4)
	P1_full[0:3, :] = P1[0:3, :]  # convert to 4x4
	P2_full = np.eye(4)
	P2_full[0:3, :] = P2[0:3, :]  # convert to 4x4
	P_canon = P2_full.dot(cv2.invert(P1_full)[1])  # find canonical P which satisfies P2 = P_canon * P1

	# "F = [t]_cross * R" [HZ 9.2.4]; transpose is needed for numpy
	F = np.cross(P_canon[0:3, 3], P_canon[0:3, 0:3], axisb=0).T

	# Other way of calculating "F" [HZ (9.2)]
	# op1 = (P2[0:3, 3:4] - P2[0:3, 0:3] .dot (cv2.invert(P1[0:3, 0:3])[1]) .dot (P1[0:3, 3:4]))
	# op2 = P2[0:3, 0:4] .dot (cv2.invert(P1_full)[1][0:4, 0:3])
	# F = np.cross(op1.reshape(-1), op2, axisb=0).T

	# Project 2D matches to closest pair of epipolar lines
	u1_new, u2_new = cv2.correctMatches(F, u1.reshape(1, len(u1), 2), u2.reshape(1, len(u1), 2))

	# For a purely sideways trajectory of 2nd cam, correctMatches() returns NaN for all possible points!
	if np.isnan(u1_new).all() or np.isnan(u2_new).all():
		F = cv2.findFundamentalMat(u1, u2, cv2.FM_8POINT)[0]  # so use a noisy version of the fund mat
		u1_new, u2_new = cv2.correctMatches(F, u1.reshape(1, len(u1), 2), u2.reshape(1, len(u1), 2))

	# Triangulate using the refined image points
	return linear_eigen_triangulation(u1_new[0], P1, u2_new[0], P2)