from dolfin import *
from fenics import *
import numpy as np
from scipy import sparse

"""
FEniCS tutorial demo program: Poisson equation with Dirichlet conditions.
  -div(a * nabla u) = f    in the unit square
                  u = u_D  on the boundary
                u_D = 0.0
                  f = 10.0
"""

def boundary(x, on_boundary):
    return on_boundary

def matrix_sqrt(mat):
	'''Caculate matrix square root of SPD matrix via eigendecomposition'''

	w, V = np.linalg.eigh(mat)

	if np.min(w) < 0:
		print('Smallest eigenvalue: ', np.min(w))
		print('Warning negative eigenvalues set to Zero')
		diag_mat = np.diag(np.sqrt(np.maximum(0, w)))
	else:
		diag_mat = np.diag(np.sqrt(w))

	mat_sqrt = np.linalg.multi_dot([V, diag_mat, V.T])

	return mat_sqrt

def gram_matrix(V):
	'''Calculate the Gram matrix'''

	u = TrialFunction(V)
	v = TestFunction(V)
	G1 = assemble(u*v*dx).array()
	G2 = assemble(dot(grad(u), grad(v))*dx).array()

	G = G1 + G2

	return G.astype(np.float32)

def sqrt_gram_matrix(V):
	'''Calculate square root of Gram matrix'''

	G = gram_matrix(V)

	return matrix_sqrt(G.astype(np.float32))

def gen_dirichlet_G(V):
    return sparse.csr_matrix(gram_matrix(V))

def gen_dirichlet_data(z, V, u_D, fenics_params):
    # specify boundary conditions
    bc = DirichletBC(V, u_D, boundary)

    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)

    # Define the right hand side and diffusion coefficient
    f = Expression(fenics_params['right_hand_side'], degree = fenics_params['inter_deg'])

    if fenics_params['example'] == 'logKL_expansion':
        pi = 3.14159265359
        pi_s = str(pi)
        L_c = 1.0/4.0
        L_p = np.max([1.0, 2.0*L_c])
        L_c_s = str(L_c)
        L_p_s = str(L_p)
        L = L_c/L_p
        L_s = str(L)

        string = '1.0+sqrt(sqrt(' + pi_s + ')*' + L_s + '/2.0)*' + str(z[0])
        for j in range(2, fenics_params['input_dim']):
            term = str(z[j-1]) + '*sqrt(sqrt(' + pi_s + ')*' + L_s + ')*exp(-pow(floor(' 
            term = term + str(j) + '/2.0)*' + pi_s + '*' + L_s + ',2.0)/8.0)'
            if j % 2 == 0:
                term = term + '*sin(floor(' + str(j) + '/2.0)*' + pi_s + '*x[0]/' + L_p_s + ')'
            else:
                term = term + '*cos(floor(' + str(j) + '/2.0)*' + pi_s + '*x[0]/' + L_p_s + ')'

            string = string + '+' + term
        string = 'exp(' + string + ')'

    elif fenics_params['example'] == 'log_affine':
        string = '5.0+exp(1.0*' + str(z[0]) + '*x[0]+1.0*' + str(z[1]) + '*x[1])'

    elif fenics_params['example'] == 'affine': 
        string = '3.0+' + str(z[0]) + '*x[0]+' + str(z[1]) + '*x[1]'

    else:
        print('unknown example')

    g = Expression(string, degree = fenics_params['inter_deg'])

    # Define a and L
    a = dot(g*grad(u), grad(v))*dx
    L = f*v*dx

    # Compute solution
    u = Function(V)
    solve(a == L, u, bc)

    # Get the FE basis coefficients and store them
    u_coefs = np.array(u.vector().get_local())

    return u_coefs
