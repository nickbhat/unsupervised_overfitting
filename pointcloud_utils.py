import ot 
import numpy as np 


"""
Finds optimal Euclidean transformation to align point clouds P and Q.

This method uses SVD of the correlation matrix between two
sets of points P and Q to find the optimal orthogonal
transformation and translation to align them. 

Returns the alignment of P with respect to Q. 

NOTE: Assumes that the rows are individual data points (t-SNE default). 

""" 

def align(P, Q): 
	# Compute means
	p_mean = np.mean(P, axis = 0)
	q_mean = np.mean(Q, axis = 0)

	# Center
	X = P - p_mean 
	Y = Q - q_mean 

	# Form covariance matrix 
	A = np.matmul(np.transpose(X), Y)

	# Get SVD 
	U, S, VH = np.linalg.svd(A, full_matrices = True)

	# Get Translation and final vector 
	R = np.transpose(U@VH)
	t = q_mean - np.dot(R, p_mean)

	# Return aligned set of P
	return np.dot(P,R) + t

"""

Rescales point cloud P to be on the same scale as Q. Just rescales
the x and y range of values marginally, nothing fancy. 

"""

def rescale(P, Q): 
	# Rescale x axis
	p_xrange = max(P[:,0]) - min(P[:,0])
	q_xrange = max(Q[:,0]) - min(Q[:,0])

	P[:,0] *= (q_xrange/p_xrange)

	# Rescale y axis 
	p_yrange = max(P[:,1]) - min(P[:,1])
	q_yrange = max(Q[:,1]) - min(Q[:,1])

	P[:,1] *= (q_yrange/p_yrange)

	return P

"""

Computes the distance between two point clouds P and Q. Currently
just supports Earth-Mover's distance with uniform mass on both clouds
and euclidean distance. The mass and distance can be changed easily
using the POT package functionality. 

"""

def dist(P,Q): 
	n = P.shape[0]
	m = Q.shape[0]

	# Get distance matrix
	M = ot.dist(P,Q)
	M /= M.max()

	# Declare uniform masses 
	a,b = np.ones((n,))/n, np.ones((m,))/m

	# Get optimal transport matrix
	G0 = ot.emd(a, b, M)

	# Return the distance 
	return np.multiply(G0, M).sum() / G0.sum() 