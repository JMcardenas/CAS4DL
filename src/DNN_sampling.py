import time, os, argparse, io, shutil, sys, math, socket 
import tensorflow.keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation,BatchNormalization, Dropout, Input
from tensorflow.keras import optimizers
import tensorflow.keras.backend as K
import tensorflow as tf
import random
import hdf5storage
from callbacks import EarlyStoppingPredictHistory
from example_functions import func_test
from dolfin import *
from fenics import *
from PDE_data import gen_dirichlet_data, gen_dirichlet_G

#import matplotlib.pyplot as plt
#import pandas as pd
import numpy as np
import scipy.io as sio

import Tasmanian

# All parameters
default_parameters = {
	'basis_fct': 'polynomial', #'trigonometric',
	'mu': 0.1, # shift hyper-parameter 
	'mode': 'x', # only required for cookies	
	'sigma': 0, # only required for trigonometric polnomials
	'num_basis': 2, # actual number of basis functions may vary
	'finite_element': 'P', # see http://femtable.org/ 
	'inter_deg': 1, # degree of interpolation
	'right_hand_side': '10.0', #'4+2*x[0]-x[1]', 
	'part': 1,
	'part_max': 5,
}

if __name__ == '__main__': 

    print('Running tensorflow with version:')
    print(tf.__version__)
     

    # depending on where running, change scratch/project directories
    # TODO: change these when installed on your local machine!!!
    if socket.gethostname() == 'jmlaptop':
        scratchdir = '/home/juanma/Documents/scratch/DNN_sampling'
        projectdir = '/home/juanma/Documents/scratch/DNN_sampling'

    elif socket.gethostname() == 'ubuntu-dev':
        scratchdir = '/home/nick/scratch/DNN_sampling'
        projectdir = '/home/nick/scratch/DNN_sampling'

    elif socket.gethostbyname(socket.gethostname()) == '172.28.0.2': 
        scratchdir = '/content/drive/My Drive/scratch' 

    elif 'cedar.computecanada.ca' in socket.gethostname():
        scratchdir = '/home/juanma/scratch/DNN_sampling'
        projectdir = '/home/juanma/project/def-adcockb/juanma'

    else:
        print('wrong host name')

    print(scratchdir)

    timestamp  = str(int(time.time()));
    start_time = time.time()

    # parse the arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--nb_layers", default = 10, type = int, help = "Number of hidden layers")
    parser.add_argument("--nb_nodes_per_layer", default = 100, type = int, help = "Number of nodes per hidden layer")
    parser.add_argument("--nb_train_points", default = 1050, type = int, help = "Number of points to use in training")
    parser.add_argument("--train_pointset", default = 'uniform_random', type = str, help = "Type of points to use in training")
    parser.add_argument("--nb_test_points", default = 1, type = int, help = "Number of points to use in testing")
    parser.add_argument("--test_pointset", default = 'CC_sparse_grid', type = str, help = "Type of points to use in testing")
    parser.add_argument("--nb_epochs", default = 10000, type = int, help = "Number of epochs for training")
    parser.add_argument("--batch_size", default = 1000, type = int, help = "Number of training samples per batch")
    parser.add_argument("--nb_trials", default = 20, type = int, help = "Number of trials to run for averaging results")
    parser.add_argument("--train", default = 0, type = int, help = "Switch for training or testing")
    parser.add_argument("--make_plots", default = 0, type = int, help = "Switch for making plots")
    parser.add_argument("--run_ID", type = str, help = "String for naming batch of trials in this run")
    parser.add_argument("--blocktype", default = 'default', type = str, help = "Type of building block for hidden layers, e.g., ResNet vs. default")
    parser.add_argument("--activation", default = 'relu', type = str, help = "Type of activation function to use")
    parser.add_argument("--example", type = int, help = "Example function to approximate (a number 1-5)")
    parser.add_argument("--optimizer", default = 'Adam', type = str, help = "Optimizer to use in minimizing the loss")
    parser.add_argument("--initializer", default = 'he_normal', type = str, help = "Initializer to use for the weights and biases")
    parser.add_argument("--quiet", default = 0, type = int, help = "Switch for verbose output")
    parser.add_argument("--input_dim", default = 1, type = int, help = "Dimension of the input")
    parser.add_argument("--output_dim", default = 1, type = int, help = "Dimension of the output")
    parser.add_argument("--MATLAB_data", default = 1, type = int, help = "Switch for using MATLAB input data points")
    parser.add_argument("--PDE_data", default = 0, type = int, help = "Switch for using PDE example input data points")
    parser.add_argument("--trial_num", default = 0, type = int, help = "Number for the trial to run")
    parser.add_argument("--precision", default = 'single', type = str, help = "Switch for double vs. single precision")
    parser.add_argument("--use_regularizer", default = 0, type = int, help = "Switch for using regularizer")
    parser.add_argument("--SG_level", default = 1, type = int, help = "The level of the sparse grid rule for testing")
    parser.add_argument("--reg_lambda", default = "1e-3", type = str, help = "Regularization parameter lambda")
    parser.add_argument("--loss_function", default = "MSE", type = str, help = "Loss function to minimize with optimizer set by arguments")
    parser.add_argument("--error_tol", default = "5e-7", type = str, help = "Stopping tolerance for the solvers")
    parser.add_argument("--sigma", default = "1e-1", type = str, help = "Standard deviation for normal initializer, max and min for uniform symmetric initializer, constant for constant initializer")
    parser.add_argument("--training_method", default = "MC", type = str, help = "Method to use for training/sampling strategy (MC = Monte Carlo sampling with retraining, ASGD = Adaptive sampling with retraining, MC2 = MC without retraining")
    parser.add_argument("--training_steps", default = 10, type = int, help = "Number of steps to use if training with a multi-step procedure (e.g., 10 steps)")
    parser.add_argument("--nb_epochs_per_iter", default = 5000, type = int, help = "Number of epochs per iteration of multi-step procedures")
    parser.add_argument("--lrn_rate_schedule", default = "exp_decay", type = str, help = "Learning rate schedule, e.g. exp_decay")
    parser.add_argument("--mesh_size", default = 32, type = int, help = "Number of nodes on one side of mesh of square domain (default 32)")
    parser.add_argument("--PDE_example", default = "affine", type = str, help = "Coefficient to use in the elliptic PDE example (default affine)")
    args = parser.parse_args()

    print('using ' + args.optimizer + ' optimizer')

    if args.train:
        print('batching with ' + str(args.batch_size) + ' out of ' + 
              str(args.nb_train_points) + ' ' + args.train_pointset + 
              ' training points')

    # set the standard deviation for initializing the DNN weights and biases
    if args.initializer == 'normal' or args.initializer == 'he_normal':
        sigma = float(args.sigma)
        print('initializing (W,b) with N(0, ' + str(sigma) + '^2)')
    elif args.initializer == 'uniform':
        sigma = float(args.sigma)
        print('initializing (W,b) with U(-' + str(sigma) + ', ' + str(sigma) + ')')
    elif args.initializer == 'constant':
        sigma = float(args.sigma)
        print('initializing (W,b) as constant ' + str(sigma))

    # set the precision variable to initialize weights and biases in either double or single precision
    if args.precision == 'double':
        print('Using double precision') 
        precision = tf.float64
        error_tol = float(args.error_tol)
        tf.keras.backend.set_floatx('float64')
    elif args.precision == 'single':
        print('Using single precision')
        precision = tf.float32
        error_tol = float(args.error_tol)
        tf.keras.backend.set_floatx('float32')


    # set the unique run ID used in many places, e.g., directory names for output
    if args.run_ID is None:
        unique_run_ID = timestamp
    else:
        unique_run_ID = args.run_ID

    # set the seeds for numpy and tensorflow to ensure all initializations are the same
    np_seed = 0
    tf_seed = 0

    # record the trial number
    trial = args.trial_num

    fenics_params = default_parameters
    mesh = UnitSquareMesh(args.mesh_size, args.mesh_size)
    V = FunctionSpace(mesh, fenics_params['finite_element'], fenics_params['inter_deg'])
    G = gen_dirichlet_G(V)
    # define homogeneous Dirichlet boundary
    u_D = Expression('0', degree = fenics_params['inter_deg'])

    K_FEM = args.output_dim
    fenics_params['mesh'] = mesh
    fenics_params['V'] = V
    fenics_params['G'] = G
    fenics_params['u_D'] = u_D
    fenics_params['example'] = args.PDE_example
    fenics_params['input_dim'] = args.input_dim

    # unique key for naming results
    key = unique_run_ID + '_' + args.activation + '_' + args.blocktype + '_' + str(args.nb_layers) + 'x' + \
            str(args.nb_nodes_per_layer) + '_' + str(args.nb_train_points).zfill(6) + \
            '_pnts_' + str(error_tol) + '_tol_' + args.optimizer + '_opt_' +\
            'example_' + str(args.example) + '_dim_' + str(args.input_dim) +\
            '_training_method_' + str(args.training_method)

    print('using key:', key)

    # the results and scratch directory can be individually specified (however for now they are the same)
    result_folder  = scratchdir + '/' + key + '/trial_' + str(trial)
    scratch_folder = scratchdir + '/' + key + '/trial_' + str(trial)
    run_root_folder = scratchdir + '/' + key

    # create the result folder if it doesn't exist yet
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    # create the scratch folder if it doesn't exist yet
    if not os.path.exists(scratch_folder):
        os.makedirs(scratch_folder)

    print('Saving results to:', result_folder)


    # loading the data from MATLAB 
    if args.MATLAB_data: 

        # the training data is in MATLAB files with names in the form:  
        # training_data_example_(nb_example)_dim_(input_dim).mat
        training_data_filename = scratchdir + '/data/' + 'example_' + str(args.example) + \
                                '_dim_' + str(args.input_dim) + \
                                '/train_data_example_' + \
                                str(args.example) + '_dim_' + \
                                str(args.input_dim) + '.mat' 

        print('Loading training data from: ' + training_data_filename) 

        # load the MATLAB -v7.3 hdf5-based format mat file 
        training_data = hdf5storage.loadmat(training_data_filename) 

        # depending on the training model (either using deterministic or random points)  
        # set up the seeds based on the trial number or just use 0 seed in the case of  
        # random points 
        if args.train_pointset == 'linspace': 
            print('Using uniformly spaced points from MATLAB') 

            # every trial uses same points (trials based on seed of np & tf) 
            x_train_data = training_data['X'] 
            y_train_data = training_data['Y'] 

            # for deterministic data, there is only one data set  
            # (don't need to split based on trial number as below) 
            x_train_data_trials = x_train_data 

            # Since there are no different trial datasets, use the trial numbers to 
            # initialize np and tf random seeds  
            print('Using trial_num = ' + str(trial) + ' as seed for tensorflow') 
            print('Using trial_num = ' + str(trial) + ' as seed for numpy') 
            np.random.seed(trial) 
            python_random.seed(trial) 
            tf.random.set_seed(trial) 

        elif args.train_pointset == 'CC_sparse_grid': 
            print('Using uniformly spaced points from MATLAB') 

            # every trial uses same points (trials based on seed of np & tf) 
            x_train_data = training_data['X'] 
            y_train_data = training_data['Y'] 

            # for deterministic data, there is only one data set  
            # (don't need to split based on trial number as below) 
            x_train_data_trials = x_train_data 

            # record the quadrature weights if using SG-quadrature-regularized training 
            w_quadrature_weights_train = training_data['W'] 

            # initialize np and tf random seeds to 0
            print('Using trial_num = ' + str(trial) + ' as seed for tensorflow')
            print('Using trial_num = ' + str(trial) + ' as seed for numpy')
            np.random.seed(trial)
            python_random.seed(trial)
            tf.random.set_seed(trial)

        elif args.train_pointset == 'uniform_random':
            # every trial uses same np & tf seed (trials based on sets of random points)
            print('Using uniform random points (trial ' + str(trial) + ') from MATLAB')

            # take the point data for this trial
            x_train_data_trials = training_data['X']
            x_train_data = x_train_data_trials[:,:,trial]

            # take the function data for this trial
            y_train_data_trials = training_data['Y']
            y_train_data = y_train_data_trials[:,:,trial]

            # these are both initialized above (TODO: change this to specify the seeds at command line)
            print('Using ' + str(tf_seed) + ' as seed for tensorflow')
            print('Using ' + str(np_seed) + ' as seed for numpy')
            np.random.seed(np_seed)
            random.seed(np_seed)                #python_random.seed(np_seed)
            tf.random.set_seed(tf_seed)

        # testing data filename has same structure as training data filename:
        # testing_data_example_(nb_example)_dim_(input_dim).mat
        testing_data_filename = scratchdir + '/data/' + 'example_' + str(args.example) + \
                                '_dim_' + str(args.input_dim) + \
                                '/test_data_example_' + \
                                str(args.example) + '_dim_' + \
                                str(args.input_dim) + '.mat' 
        
        print('Loading testing data from: ' + testing_data_filename)
        testing_data = hdf5storage.loadmat(testing_data_filename)

        # set the testing data (often smaller than the final testing data, for 
        # outputting the errors while training)
        x_test_data = testing_data['X']
        y_test_data = testing_data['Y']

        # quadrature weights used in reporting the error while training
        w_quadrature_weights_test = np.transpose(testing_data['W'])

        nb_test_points = len(x_test_data)

        print('Using ' + args.test_pointset + ' point set from MATLAB for testing with ' + str(len(x_test_data)) + ' points')
        #print(x_test_data)
        #print(y_test_data)
        #print(w_quadrature_weights_test)

        print('')
        print('TRAIN DATA SUMMARY:')
        print('-----------------------')
        print('x_train_data shape:')
        print(x_train_data.shape)
        print('y_train_data shape:')
        print(y_train_data.shape)
        print('')

        print('TEST DATA SUMMARY:')
        print('-----------------------')
        print('x_test_data shape:')
        print(x_test_data.shape)
        print('y_test_data shape:')
        print(y_test_data.shape)
        print('w_quadrature_weights_test shape:')
        print(w_quadrature_weights_test.shape)
        print('')

    
    else:        
        # Generate training points
        np.random.seed(trial);
        x_train_data = np.random.rand(args.nb_train_points, args.input_dim)*2-1

        # Generate testing points
        if args.test_pointset == 'random':
            np.random.seed(np_seed);  
            x_test_data  = np.random.rand(args.nb_test_points,args.input_dim)*2-1
        
        elif args.test_pointset == 'CC_sparse_grid':
            print('using TASMANIAN to generate the testing points')
            grid_test = Tasmanian.SparseGrid()
            grid_test.makeGlobalGrid(args.input_dim, 0, args.SG_level, "level", "clenshaw-curtis") 
            
            x_test_data   = np.array(grid_test.getPoints())
            w_quadrature_weights_test = np.array(grid_test.getQuadratureWeights())

            #print('WARNING: Using TSG points, overwriting argument')
            nb_test_points = x_test_data.shape[0]
            nb_test_points_check = nb_test_points
            
            print('Using TASMANIAN sparse grid rule with', x_test_data.shape, 'data points')
            print('Using TASMANIAN sparse grid rule of size', w_quadrature_weights_test.shape)
        
        else:
            print('unknown pointset passed in:', args.test_pointset)

        plotting_FE_solns = 0
        exact_FE_soln = 0
        save_vtk_files = 0
        plotting_points = 0

        # generate function data
        if args.PDE_data:
            test_data_filename = run_root_folder + '/' + str(nb_test_points).zfill(8) + '_' + args.test_pointset + '_pts_test_data.mat'
            print('Testing the solutions with data from:', test_data_filename)
            train_data_filename = run_root_folder + '/trial_' + str(trial) + '_train_data.mat'

            # check if the FEM testing solutions exist and load them
            if os.path.exists(test_data_filename):

                print('Found FEM test_data file:', test_data_filename)
                test_data = hdf5storage.loadmat(test_data_filename)
                nb_test_points       = test_data['nb_test_points'][0,0]
                x_test_data  = test_data['x_test_data']
                y_test_data  = test_data['y_test_data']
                K_FEM = y_test_data.shape[1]

                if nb_test_points != nb_test_points_check:
                    errstr = ('Testing data does not match command line arguments nb_test_points from file is %d but nb_test_points from SG_level is %d, aborting.' % (nb_test_points, nb_test_points_check))
                    sys.exit(errstr)

            # otherwise generate new testing solutions
            else:

                print('Generating the testing data')

                U_test = []

                for i in range(nb_test_points):

                    # record start time
                    t_start = time.time()

                    # get the training data inputs 
                    z = x_test_data[i,:]

                    # Get the FE basis coefficients and store them
                    u_coefs = gen_dirichlet_data(z, V, u_D, fenics_params)

                    U_test.append(u_coefs)

                    # Compute the H1 norm of the solution
                    GRAM     = G
                    sd2      = (GRAM).dot(u_coefs)
                    H1normsq = np.sqrt(sd2.dot(u_coefs))

                    # Plot solution and mesh
                    if plotting_FE_solns:
                        plot(u)
                        plot(mesh)
                        plt.show()
                    
                    # Save solution to file in VTK format
                    if save_vtk_files:
                        vtkfile = File('poisson/test_solution_i.pvd')
                        vtkfile << u

                    if exact_FE_soln:
                        # Compute error in L2 norm
                        error_L2 = errornorm(u_D, u, 'L2')
                        error_H1 = errornorm(u_D, u, 'H1')
                        
                        # Compute maximum error at vertices
                        vertex_values_u_D = u_D.compute_vertex_values(mesh)
                        vertex_values_u   = u.compute_vertex_values(mesh)
                        error_max         = np.max(np.abs(vertex_values_u_D - vertex_values_u))

                        # Print errors
                        print('i =', i, ' ', z,
                            'error_L2  =', error_L2,
                            'error_H1  =', error_H1,
                            'error_max =', error_max,
                            'H1 norm of the approximation =', H1normsq,
                            'took', time.time() - t_start, 'seconds to compute')
                    else:
                        print('i =', i, ' ', z,
                            'H1 norm of the approximation =', H1normsq,
                            'took', time.time() - t_start, 'seconds to compute')


                    if i % 1000 == 0 and not i == 0:
                        y_test_data = np.array(U_test)

                        print('saving intermediate file')
                        K_FEM = y_test_data.shape[1]
                        test_data = {}
                        test_data['y_test_data']    = y_test_data
                        test_data['x_test_data']    = x_test_data
                        test_data['nb_test_points']         = nb_test_points
                        test_data['w_quadrature_weights_test'] = w_quadrature_weights_test
                        if not os.path.exists(run_root_folder):
                            try:
                                os.makedirs(run_root_folder)    
                            except FileExistsError:
                                print('ERROR: ' + run_root_folder + ' exists, trying saving')

                        sio.savemat(test_data_filename, test_data)

                y_test_data = np.array(U_test)
                x_test_data = x_test_data

                print('Generated inputs of size: ', x_test_data.shape)
                print('Generated outputs of size: ', y_test_data.shape)

                K_FEM = y_test_data.shape[1]
                test_data = {}
                test_data['y_test_data']    = y_test_data
                test_data['x_test_data']    = x_test_data
                test_data['nb_test_points']         = nb_test_points
                test_data['w_quadrature_weights_test'] = w_quadrature_weights_test
                sio.savemat(test_data_filename, test_data)
                print('test results generation and save finished after %4.2f' %(time.time() - start_time), 'seconds')
            
            # check if the FEM training solutions exist and load them
            if os.path.exists(train_data_filename):

                print('Found FEM train_data file:', train_data_filename)
                train_data = hdf5storage.loadmat(train_data_filename)
                K_FEM            = train_data['K_FEM'][0,0]
                args.output_dim = K_FEM
                nb_train_points_file        = train_data['nb_train_points'][0,0]
                x_train_data = train_data['x_train_data']
                y_train_data = train_data['y_train_data']

                print('loaded data correctly')

            # otherwise generate new solutions
            else:
                
                print('Did not find FEM train_data file:', train_data_filename + ', Generating new data')

                x_train_data = np.transpose(np.random.uniform(-1.0,1.0,(args.nb_train_points,args.input_dim)))
                U = []

                print('Using uniform random training points with nb_train_points =', args.nb_train_points)

                # scatter plot the points
                if plotting_points:
                    plt.scatter(x_train_data[0,:], x_train_data[1,:])
                    plt.show()

                print('Generating the training data')
                # Generate the training data
                for i in range(args.nb_train_points):

                    # record start time
                    t_start = time.time()

                    # get the training data inputs 
                    z = x_train_data[:,i]

                    # get the coefficients of the solution
                    u_coefs = gen_dirichlet_data(z, V, u_D, fenics_params)

                    U.append(u_coefs)

                    # Record the number of FE degrees of freedom
                    if i == 0:
                        K_FEM = len(U[0])
                        print('FE degrees of freedom K_FEM = ', K_FEM)

                    # Plot solution and mesh
                    if plotting_FE_solns:
                        plot(u)
                        plot(mesh)
                        plt.show()
                    
                    # Save solution to file in VTK format
                    if save_vtk_files:
                        vtkfile = File('poisson/train_solution_i.pvd')
                        vtkfile << u

                    # Compute the H1 norm of the solution
                    GRAM     = G
                    sd2      = (GRAM).dot(u_coefs)
                    H1normsq = np.sqrt(sd2.dot(u_coefs))
                    
                    if exact_FE_soln:
                        # Compute error in L2 norm
                        error_L2 = errornorm(u_D, u, 'L2')
                        error_H1 = errornorm(u_D, u, 'H1')
                        
                        # Compute maximum error at vertices
                        vertex_values_u_D = u_D.compute_vertex_values(mesh)
                        vertex_values_u   = u.compute_vertex_values(mesh)
                        error_max         = np.max(np.abs(vertex_values_u_D - vertex_values_u))

                        # Print errors
                        print('i =', i, ' ', x_train_data[:,i],
                            'error_L2  =', error_L2,
                            'error_H1  =', error_H1,
                            'error_max =', error_max,
                            'H1 norm of the approximation =', H1normsq,
                            'took', time.time() - t_start, 'seconds to compute')
                    else:
                        print('i =', i, ' ', x_train_data[:,i],
                            'H1 norm of the approximation =', H1normsq,
                            'took', time.time() - t_start, 'seconds to compute')

                y_train_data = np.array(U)
                x_train_data = x_train_data.T

                print('Generated inputs of size: ', x_train_data.shape)
                print('Generated outputs of size: ', y_train_data.shape)

                train_data = {}
                train_data['input_dim']                = args.input_dim
                train_data['K_FEM']                    = K_FEM
                train_data['G']                        = G
                train_data['nb_train_points']          = args.nb_train_points
                train_data['y_train_data']             = y_train_data
                train_data['x_train_data']             = x_train_data
                train_data['fenics_mesh_coords']       = np.array(mesh.coordinates())
                train_data['fenics_mesh_cells']        = np.array(mesh.cells())
                train_data['fenics_mesh_num_cells']    = np.array(mesh.num_cells())
                train_data['fenics_mesh_num_edges']    = np.array(mesh.num_edges())
                train_data['fenics_mesh_num_vertices'] = np.array(mesh.num_vertices())
                train_data['fenics_mesh_hmax']         = np.array(mesh.hmax())
                train_data['fenics_mesh_hmin']         = np.array(mesh.hmin())
                train_data['fenics_mesh_rmax']         = np.array(mesh.rmax())
                train_data['fenics_mesh_rmin']         = np.array(mesh.rmin())
                sio.savemat(train_data_filename, train_data)
                args.output_dim = K_FEM

        else:
            print('using function data for training/testing')
            y_train_data = func_test(x_train_data, args.input_dim, args.example)
            y_test_data  = func_test(x_test_data, args.input_dim, args.example)

        print('Training points shape: ' + str(x_train_data.shape))
        print('Testing points shape:' + str(x_test_data.shape)) 
        print('y_train data shape: ' + str(y_train_data.shape))
        print('y_test data shape:' + str(y_test_data.shape))

    

    #------------------------------------------------------------------------#
    # Plot function
    #------------------------------------------------------------------------#
    
    """
    if args.input_dim == 1:
        # 1D
        fig = plt.figure()
        plt.plot(x_test_data,y_test_data)
        plt.show()
    elif args.input_dim == 2:
        # 2D
        fig = plt.figure()
        ax  = fig.add_subplot(111, projection='3d',title = 'Test function')   
        ax.scatter3D(x_test_data[:,0],x_test_data[:,1],y_test_data)    
    elif args.input_dim > 2:
        print('dimension bigger than 2')
    """
    ######## END OF DATA GENERATION #########

    ######## SETUP RUN DATA ###############
    best_loss    = 10
    
    # Sampling parameters
    
    N_max = args.nb_nodes_per_layer

    #Check loop
    if args.input_dim < 4:
        if 1000 <= args.nb_train_points:
            print('Input dim:', str(args.input_dim), ' | Grid size: ', str(args.nb_train_points), ' | Correct grid size for dimension')
        else:
            print('increase grid size')
    elif 4 <= args.input_dim < 8:
        if 2000 <= args.nb_train_points:
            print('Input dim:', str(args.input_dim), ' | Grid size: ', str(args.nb_train_points), ' | Correct grid size for dimension')
        else:
            print('increase grid size')
    elif 8 <= args.input_dim:
        if 5000 <= args.nb_train_points:
             print('Input dim: ', str(args.input_dim), ' | Grid size: ', str(args.nb_train_points),' | Correct grid size for dimension')
        else:
            print('increase grid size')        

    # Compute M values (M_i = k_i N_max)

    if args.input_dim < 4:
        min_M_value = 100
    elif 4 <= args.input_dim < 8:
        min_M_value = 300
    elif 8 <= args.input_dim:
        min_M_value = 700

    k_init  = np.round(min_M_value/N_max)
    k_final = np.round(args.nb_train_points/(2*N_max))

    print('k init value: ', str(k_init), 'k final value: ', str(k_final))
    k_values = np.round(np.linspace(k_init,k_final,args.training_steps -1)).astype(int)
    M_values = (k_values*N_max).astype(int)

    epochs_so_far   = 0
    
    print('N_max: ' + str(N_max))
    print('k values: ' + str(k_values))
    print('M values: ' + str(M_values))

    # sampling parameters 
    #N_max        = args.nb_nodes_per_layer

    #old_k_values = np.round(np.linspace(10,5*args.training_steps,args.training_steps-1)).astype(int) 
    #old_M_values = 10*old_k_values 
    #print('old_k values: ' + str(old_k_values))
    #print('old_M values: ' + str(old_M_values))

    #k_values     = (old_M_values/args.nb_nodes_per_layer).astype(int)
    #M_values     = (k_values*args.nb_nodes_per_layer).astype(int)

    #epochs_so_far   = 0

    #print('N_max: ' + str(N_max))
    #print('k values: ' + str(k_values))
    #print('M values: ' + str(M_values))

    # save relevant run data for loading into MATLAB, some items will 
    # be updated by callback
    run_data = {}
    run_data['lrn_rate_schedule'] = args.lrn_rate_schedule
    #if args.lrn_rate_schedule == 'exp_decay':
    #    run_data['base'] = base
    #    run_data['decay_steps'] = decay_steps

    run_data['training_method'] = args.training_method
    run_data['training_steps'] = args.training_steps
    run_data['k_values']   = k_values
    run_data['M_values']   = M_values
    run_data['optimizer']  = args.optimizer  
    run_data['error_tol']  = error_tol
    run_data['activation'] = args.activation
    run_data['nb_layers']  = args.nb_layers
    run_data['nb_nodes_per_layer'] = args.nb_nodes_per_layer
    run_data['nb_train_points'] = args.nb_train_points
    run_data['nb_test_points']  = nb_test_points
    run_data['nb_epochs_per_iter'] = args.nb_epochs_per_iter
    run_data['nb_epochs'] = args.nb_epochs
    run_data['nb_trials'] = args.nb_trials 
    run_data['blocktype'] = args.blocktype
    run_data['initializer'] = args.initializer
    run_data['PDE_data'] = args.PDE_data
    run_data['example']    = args.example
    run_data['start_time'] = start_time
    run_data['timestamp']  = timestamp 
    run_data['tf_version'] = tf.__version__
    run_data['input_dim']  = args.input_dim
    run_data['output_dim'] = args.output_dim
    run_data['x_train_data'] = x_train_data
    run_data['y_train_data'] = y_train_data 
    run_data['x_test_data']  = x_test_data
    run_data['y_test_data']  = y_test_data 
    run_data['w_quadrature_weights_test'] = w_quadrature_weights_test
    run_data['test_pointset'] = args.test_pointset
    run_data['sigma'] = sigma
    run_data['update_ratio'] = 0.0625
    run_data['quiet'] = args.quiet
    run_data['patience'] = 1e16
    run_data['best_loss'] = best_loss
    run_data['LSGD'] = False
    run_data['model_save_folder'] = 'final'    #args.training_method + '_model_save'
    run_data['run_data_filename'] = 'run_data.mat'  #args.training_method + '_run_data.mat'
    run_data['trial']         = trial
    run_data['result_folder'] = result_folder
    run_data['opt_final_epoch'] = epochs_so_far + args.nb_epochs_per_iter
    
    loss_per_iter_global      = np.array([])
    all_last_epochs_per_iter  = np.array([])
    all_first_epochs_per_iter = np.array([])
    all_lrn_rates_per_iter    = np.array([])
    steps_per_iter_global     = np.array([])

    #print('red flag')
    if (args.training_method == 'MC') or (args.training_method == 'ASGD'):

        #------------------------------------------------------------------------------#
        # set up the initializer for the weight and biase
        #------------------------------------------------------------------------------#
        if args.initializer == 'normal':
            weight_bias_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev= sigma, seed = trial)
        elif args.initializer == 'uniform':
            weight_bias_initializer = tf.keras.initializers.RandomUniform(minval= -sigma, maxval= sigma, seed= trial)
        elif args.initializer == 'constant':
            weight_bias_initializer = tf.keras.initializers.Constant(value= sigma)
        elif args.initializer == 'he_normal':
            weight_bias_initializer = tf.keras.initializers.HeNormal(seed= trial)
        elif args.initializer == 'xavier_normal':
            weight_bias_initializer = tf.keras.initializers.GlorotNormal(seed= trial)
        elif args.initializer == 'xavier_uniform':
            weight_bias_initializer = tf.keras.initializers.GlorotUniform(seed= trial)
        else:
            sys.exit('initializer must be one of the supported types, e.g., normal, uniform, etc.')

        #------------------------------------------------------------------------------#
        # set up learning rate schedule from either exp_decay linear, or constant
        #------------------------------------------------------------------------------#
        if args.lrn_rate_schedule == 'exp_decay':
            init_rate = 1e-3
            lrn_rate  = init_rate 

            # decay steps specifies how many epochs until the learning rate is 
            # decayed by specific amount
            decay_steps = 1e3

            #WARNING: Edit the parameter final stepsize for best results
            final_learning_rate = error_tol

            # calculate the base so that the learning rate schedule with exponential 
            # decay follows (init_rate)*(base)^(current_epoch/decay_step)
            base = np.exp(decay_steps/args.nb_epochs*(np.log(final_learning_rate)-np.log(init_rate)))

            # base on the above, the final learning rate is (init_rate)*(base)^(total_epoch/decay_step)
            print('based on init rate = ' + str(init_rate)
                + ', decay_steps = ' + str(decay_steps)
                + ', calculated base = ' + str(base)
                + ', so that after ' + str(args.nb_epochs)
                + ' epochs we have final learning rate = '
                + str(init_rate*base**(args.nb_epochs/decay_steps)))
            
            decay_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                init_rate, decay_steps, base, staircase= False, name= None)
        

        elif args.lrn_rate_schedule == 'linear':
            # only need to specify the init rate for linear
            init_rate = 1e-3

            # decay steps specifies how many epochs until the learning rate is 
            # decayed by a specific amount 
            decay_steps = 1e3 
            
            print('using a linear learning rate schedule')

            decay_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
                init_rate, decay_steps, end_learning_rate=error_tol, power=1.0
            )

        elif args.lrn_rate_schedule == 'constant':
            # only need to specify the init rate for constant (stays the same)
            init_rate = 1e-3

            decay_schedule = init_rate

            print('using a constant learning rate')
        else:
            print('incorrect learning rate schedule selected')    

        #------------------------------------------------------------------------------#
        # set up optimizers for training
        #------------------------------------------------------------------------------#
        if args.optimizer == 'SGD':
            standard_opt = tf.keras.optimizers.SGD(
                learning_rate= 1e-4,
                name= 'SGD'
            )
        elif args.optimizer == 'Adam':
            standard_opt = tf.keras.optimizers.Adam(
                learning_rate= decay_schedule,
                beta_1= 0.9, beta_2= 0.999, epsilon= 1e-07,
                name= 'Adam'
            )    
        else:
            sys.exit('optimizer must be one of the preset optimizers: SGD, Adam, etc.')    

        #print('blue flag')
        #------------------------------------------------------------------------------#
        # Create neural network model
        #------------------------------------------------------------------------------#
        all_layers = []

        u       = tf.keras.models.Sequential()
        input_u = tf.keras.layers.InputLayer(input_shape=(args.input_dim,), name='input_u')
        u.add(input_u)

        # add 2nd to last layer
        for layer in range(args.nb_layers + 1):
            if layer == args.nb_layers:
                layer_name = 'second_to_last_u'
            else:
                layer_name = 'dense_u' + str(layer)

            print('adding layer', layer_name)
            this_layer = tf.keras.layers.Dense(args.nb_nodes_per_layer, 
                                                activation=args.activation,
                                                name=layer_name,
                                                kernel_initializer= weight_bias_initializer,
                                                bias_initializer= weight_bias_initializer,
                                                dtype=tf.float32)
            all_layers.append(this_layer)
            u.add(this_layer)
        
        # output layer 
        output_layer = tf.keras.layers.Dense(args.output_dim,
                                             activation= tf.keras.activations.linear,
                                             trainable=True,
                                             use_bias=False,
                                             name='output_u',
                                             kernel_initializer=weight_bias_initializer,
                                             dtype=tf.float32)

        # add the output layer
        u.add(output_layer)

        # Create B basis neural network model 
        B = tf.keras.Model(inputs=u.input, outputs=u.get_layer('second_to_last_u').output) 

        # loss function 
        mse = tf.keras.losses.MeanSquaredError()

        # compile model
        u.compile(loss=mse, optimizer=standard_opt, metrics=['accuracy'])

        # number of variables
        model_num_trainable_variables = np.sum([np.prod(v.get_shape().as_list()) for v in u.trainable_variables])
        if not args.quiet:
            print('This model has {} trainable variables'.format(model_num_trainable_variables))
            u.summary()
            B.summary()

        run_data['tf_trainable_vars'] = model_num_trainable_variables
        run_data['loss_function'] = args.loss_function

        #------------------------------------------------------------------------------#
        # Training 
        #------------------------------------------------------------------------------#

        # Error variables 
        l2_error_data = []
        L2_error_data = []
        H1_error_data = []
        L2_u = 0.0
        H1_u = 0.0
        time_intervals = []
        constant_C_from_B_data = np.array([])
        constant_C_from_Q_data = np.array([])

        Chris_function = np.array([])
        Prob_dist      = np.array([])
        r_vals         = np.array([])
        r_Phi_vals     = np.array([])
        
        # Indices 
        I_ad  = np.array([])          
        I_new = np.array([]) 
        I     = np.array([])
        I_i   = np.arange(0,args.nb_train_points)

        print('-----------------------------------------------------------------------')
        print('Training DNN: ' + str(key))
        print('Trial number: ' + str(trial))
        print('-----------------------------------------------------------------------')

        for l in range(args.training_steps-1):
            print('----  Iteration: ' + str(l))

            N = args.nb_nodes_per_layer
            M = M_values[l]

            # Compute SVD and rank 
            if l == 0:
                # Matrix B
                B_matrix = np.asmatrix(B(x_train_data)/np.sqrt(args.nb_train_points))
                # SVD 
                U_matrix, S_matrix, V_matrix = np.linalg.svd(B_matrix, full_matrices=False)

                # find rank r
                for i in range(N):
                    if (S_matrix[i]/S_matrix[0]) > 1e-6:
                        r = i

            else:
                # Use the basis and rank computed from last SVD
                U_matrix = Phi_basis
                r        = r_phi 
            
            print('r value: ', r)
            r_vals = np.append(r_vals, [r])

            if N == r:
                print('rank is equal to N')
                N_max       = r 
                k_min_ratio = 0

                if l == 0:
                    k_max_ratio = (k_values[l]).astype(int)
                else:
                    k_max_ratio = (k_values[l]-k_values[l-1]).astype(int)
            else:
                print('rank is less to N: ', r)
                N_max       = r
                k_min_ratio = 1

                if l == 0:
                    k_max_ratio = np.floor(M_values[l]/r).astype(int)
                    s_l = (M_values[l] - k_max_ratio*r).astype(int)
                else:
                    k_max_ratio = np.floor((M_values[l]-M_values[l-1])/r).astype(int)
                    s_l = ((M_values[l]-M_values[l-1])-k_max_ratio*r).astype(int)

            print('k_max_ratio: ', k_max_ratio, 'k_min_ratio: ', k_min_ratio)

            #----------------------------------------------------------------------#
            # Samling 
            if args.training_method == 'ASGD':
                # ASGD - Probability Distribution
                mu  = abs(np.square(U_matrix[:,0:r]))  
                print('size mu: ' + str(mu.shape) + ' | Number of points: '+ str(args.nb_train_points) )

                if l == 0:
                    # Draw points from current sampling measures
                    for j in range(N_max):
                        mu_j  = mu[:,j] 
                        I_a   = random.choices(I_i, weights = mu_j, k = k_max_ratio)
                        I_new = np.append(I_new,I_a)

                    print('size I new: ' + str((I_new).astype(int).shape[0]))
                    if r < N:
                        print('Draw k_min samples from mu_i')
                        count_pts = 0
                        samp_num  = 0
                        while count_pts < s_l:
                            mu_j  = mu[:,samp_num]
                            I_a   = random.choices(I_i, weights= mu_j, k= k_min_ratio)
                            I_new = np.append(I_new, I_a)

                            count_pts = count_pts + 1
                            if samp_num == (r-1):
                                samp_num = 0
                            else:
                                samp_num = samp_num + 1
                    
                    print('size I new: ' + str((I_new).astype(int).shape[0]))
                else:  
                    # Adding points from old sampling measures
                    I_ad = np.array([]) 
                    
                    print('k ad:' + str(k_max_ratio))

                    for j in range(N_max):
                        mu_j    = mu[:,j]
                        I_ad_ax = random.choices(I_i, weights= mu_j, k= k_max_ratio) 
                        I_ad    = np.append(I_ad, I_ad_ax)

                    if r < N:
                        print('Draw k_min samples from mu_i')
                        count_pts = 0
                        samp_num  = 0
                        while count_pts < s_l:
                            mu_j    = mu[:,samp_num]
                            I_ad_ax = random.choices(I_i, weights= mu_j, k= k_min_ratio)
                            I_ad    = np.append(I_ad, I_ad_ax)

                            count_pts = count_pts + 1
                            if samp_num == (r-1):
                                samp_num = 0
                            else:
                                samp_num = samp_num + 1
                
                I_ad = np.array(I_ad, dtype=np.float32).astype(int)
                print('size I ad:' + str(I_ad.shape[0]))

                if l == 0:
                    I = I_new
                else: 
                    I = np.append(I, I_ad)

                # Convert I
                I = I.astype(int)   
            elif args.training_method == 'MC':   
                # Uniform Sampling 
                if l == 0:
                    I = np.arange(0,M_values[l])
                else:
                    I_ad = np.arange(M_values[l-1], M_values[l])
                    I    = np.append(I, I_ad)
                # Convert I
                I = np.array(I, dtype=np.float32).astype(int)

            print('Number of points in I set: ' + str(I.shape[0]))    
            
            #------------------------------------------------------------------------#
            # Compute constant C

            if args.training_method == 'ASGD':
                #Uncomment these lines if l2-weighted loss function is being used
                sum_mu         = np.sum(mu[I,:], axis= 1)/r
                sample_weights = np.concatenate((np.divide(1,sum_mu)).tolist())
                matrix_weights = np.asmatrix(np.diag(np.sqrt(sample_weights)))
                
                # weigths
                sample_weights_method = np.sqrt(sample_weights)     
                #sample_weights_method = None
                #print('sample weights CAS size: ', sample_weights_method.shape)

                # 1st SVD
                U_matrix_on_M = matrix_weights*U_matrix[I,0:r]/np.sqrt(M)
                #U_matrix_on_M = U_matrix[I,0:r]/np.sqrt(M)
                
                # 2nd SVD
                U_matrix_from_B, S_matrix_from_B, V_matrix_from_B = np.linalg.svd(U_matrix_on_M, full_matrices=False)

                constant_C_from_Q = 1/min(S_matrix_from_B)
                print('Constant C:', constant_C_from_Q)

            elif args.training_method == 'MC':
                # weights
                sample_weights_method = None
                
                # 1st SVD
                U_matrix_on_M = U_matrix[I,0:r]/np.sqrt(M)
                # 2nd SVD
                U_matrix_from_B, S_matrix_from_B, V_matrix_from_B = np.linalg.svd(U_matrix_on_M, full_matrices=False)
                
                constant_C_from_Q = 1/min(S_matrix_from_B)
                print('Constant C:', constant_C_from_Q)
            
            else:
                print('wrong training method')

            constant_C_from_Q_data = np.append(constant_C_from_Q_data, [constant_C_from_Q])

            #----------------------------------------------------------------------#
            # Fit model
            run_data['x_train_data_orig'] = x_train_data
            run_data['y_train_data_orig'] = y_train_data
            run_data['x_train_data_M_values'] = x_train_data[I,:]
            run_data['y_train_data_M_values'] = y_train_data[I,:]    
            run_data['fenics_params'] = fenics_params
            run_data['I_set']        = I
            run_data['M_values']     = M_values
            run_data['k_values']     = k_values

            early_stopping_prediction_history = EarlyStoppingPredictHistory(run_data)

            u.fit(x_train_data[I,:], y_train_data[I,:], initial_epoch= epochs_so_far, batch_size= M_values[l], 
                verbose=0, shuffle= True, epochs= run_data['opt_final_epoch'], sample_weight= sample_weights_method,
                callbacks= [early_stopping_prediction_history]
                )

            last_epoch_from_callback = early_stopping_prediction_history.steps[-1] - early_stopping_prediction_history.steps[0]
            print('callback says it ran for', last_epoch_from_callback, 'epochs')

            epochs_so_far = epochs_so_far + last_epoch_from_callback
            print('epochs_so_far: ', epochs_so_far)
            run_data['opt_final_epoch'] = epochs_so_far + args.nb_epochs_per_iter
            
            #loss per iter global 
            loss_per_iter_global  = np.append(loss_per_iter_global, [early_stopping_prediction_history.losses])
            steps_per_iter_global = np.append(steps_per_iter_global, [early_stopping_prediction_history.steps])
            
            # epochs per iter 
            last_epoch_per_iter  = early_stopping_prediction_history.steps[-1]
            first_epoch_per_iter = early_stopping_prediction_history.steps[0]

            print('last epoch per iter:', last_epoch_per_iter)
            print('first epoch per iter:', first_epoch_per_iter)

            all_last_epochs_per_iter  = np.append(all_last_epochs_per_iter, [last_epoch_per_iter]) 
            all_first_epochs_per_iter = np.append(all_first_epochs_per_iter, [first_epoch_per_iter]) 
            all_lrn_rates_per_iter    = np.append(all_lrn_rates_per_iter, [early_stopping_prediction_history.lrn_rates])

            # Testing error
            y_DNN_pred    = u(x_test_data)
            absdiff       = abs(y_test_data - y_DNN_pred)
            
            if args.PDE_data:
                test_start_time = time.time()

                # the L2 and H1 errors we compute
                L2_err = 0.0
                H1_err = 0.0

                error = Function(V)
                relative_error_denom = Function(V)

                for i in range(run_data['nb_test_points']):
                    error.vector().set_local(absdiff[i,:])
                    relative_error_denom.vector().set_local(y_test_data[i,:])
                    relative_error_denom_L2 = norm(relative_error_denom, 'L2')
                    relative_error_denom_H1 = norm(relative_error_denom, 'H1')
                    error_L2 = norm(error, 'L2')
                    error_H1 = norm(error, 'H1')

                    L2_err = L2_err + np.abs(error_L2)**(2.0)*run_data['w_quadrature_weights_test'][i]
                    H1_err = H1_err + np.abs(error_H1)**(2.0)*run_data['w_quadrature_weights_test'][i]
                    L2_u = L2_u + np.abs(relative_error_denom_L2)**(2.0)*run_data['w_quadrature_weights_test'][i]
                    H1_u = H1_u + np.abs(relative_error_denom_H1)**(2.0)*run_data['w_quadrature_weights_test'][i]

                L2_err = np.sqrt(np.abs(L2_err*2**(-1.0*run_data['input_dim'])))
                H1_err = np.sqrt(np.abs(H1_err*2**(-1.0*run_data['input_dim'])))
                L2_u = np.sqrt(np.abs(L2_u*2**(-1.0*run_data['input_dim'])))
                H1_u = np.sqrt(np.abs(H1_u*2**(-1.0*run_data['input_dim'])))
                L2_error_data.append(L2_err)
                H1_error_data.append(H1_err)
                time_intervals.append(time.time() - run_data['start_time'])
                test_time = time.time() - test_start_time
                print('completed computing test errors in', str(test_time), 'seconds')
            else:
                L2err_diff    = np.sqrt(abs(np.sum(np.square(absdiff)*run_data['w_quadrature_weights_test']*2.0**(-1.0*run_data['input_dim']))))
                L2err_f       = np.sqrt(abs(np.sum(np.square(abs(y_test_data))*run_data['w_quadrature_weights_test']*2.0**(-1.0*run_data['input_dim']))))
                L2err         = L2err_diff/L2err_f
                L2_error_data = np.append(L2_error_data, [L2err])

                l2err_dif     = np.sqrt(np.sum(np.square(absdiff))/args.nb_test_points) 
                l2err_f       = np.sqrt(np.sum(np.square(y_test_data))/args.nb_test_points) 
                l2err         = l2err_dif/l2err_f
                l2_error_data = np.append(l2_error_data, [l2err])

                print('-----------------------------------------------------------------------')
                print('L2 error at the end of epochs training:' + str(L2err)) 
                print('l2 error at the end of epochs training:' + str(l2err))     

            #------------------------------------------------------------------#
            # Christoffel function

            # SVD 
            Psi_basis =  np.asmatrix(B(x_train_data)/np.sqrt(args.nb_train_points)) 
            Phi_basis, S_Phi_matrix, V_Phi_basis = np.linalg.svd(Psi_basis, full_matrices=False)

            # find rank r
            for i in range(N):
                if (S_Phi_matrix[i]/S_Phi_matrix[0]) > 1e-6:
                    r     = i
                    r_phi = i

            print('Phi r value: ', r)
            r_Phi_vals = np.append(r_Phi_vals, [r])
            
            Phi_basis_sum = np.sum(np.square(np.abs(Phi_basis[:,0:r])), axis= 1)
            
            print('Phi basis sum size: ', Phi_basis_sum.shape)
            
            Chris_function_temp = np.divide(1,Phi_basis_sum)
            Prob_dist_temp      = Phi_basis_sum/r
            
            print('Size of Prob distribution: ', Prob_dist_temp.shape)
            print('Size of Christoffel func:', Chris_function_temp.shape)

            if l == 0:
                Chris_function = Chris_function_temp
                Prob_dist      = Prob_dist_temp
            else:
                Chris_function = np.append(Chris_function, Chris_function_temp, axis= 1)
                Prob_dist      = np.append(Prob_dist, Prob_dist_temp, axis= 1)

            print('Christoffel function append size:', str(Chris_function.shape), 'Probability distribution append size:', str(Prob_dist.shape))
            
        """
        #----------------------------------------------------------------------#
        # Evaluate the main basis-functions
        if args.input_dim == 1: # or args.input_dim == 2:
            # 0. Compute l2 norm of basis
            B_matrix_train = np.asmatrix(B(x_train_data))
            basis_l2_norm  = np.transpose(np.sqrt(np.sum(np.square(np.abs(B_matrix_train))/len(x_train_data), axis= 0)))
            print('size l2 norm basis: ', str(basis_l2_norm.shape))

            # 1. Pick the largest indices
            final_layer_weights = u.get_layer('output_u').get_weights()[0]
            final_layer_vector  = np.reshape(final_layer_weights, -1)   
            nb_main_coefs   = 6
            final_layer_weights_abs = np.expand_dims(np.abs(final_layer_vector), axis= 1)
            mult_final_layer = np.concatenate(np.multiply(final_layer_weights_abs, basis_l2_norm).tolist())
            largest_indices = np.argsort(-1*mult_final_layer)[:nb_main_coefs]
            
            # 2. Evaluate basis functions
            basis_on_largest_coef_train = np.asmatrix(B(x_train_data))[:,largest_indices]
            
            # 3. Save data
            run_data['basis_on_largest_coef_train'] = basis_on_largest_coef_train
            run_data['largest_indices_coeff'] = largest_indices
            run_data['largest_coeff'] = final_layer_weights[largest_indices]
        """

        #----------------------------------------------------------------------#
        # Compute the best approx to f from P_phi
        B_matrix   = B(x_test_data)
        best_coeff = tf.linalg.lstsq(B_matrix,run_data['y_test_data'],fast=False)

        # Least-square testing error
        p_DNN_pred  = tf.matmul(B_matrix,best_coeff)
        absdiff     = abs(y_test_data - p_DNN_pred)            
        if args.PDE_data:
            test_start_time = time.time()

            # the L2 and H1 errors we compute
            L2_err = 0.0
            H1_err = 0.0

            error = Function(V)
            relative_error_denom = Function(V)

            for i in range(run_data['nb_test_points']):
                error.vector().set_local(absdiff[i,:])
                relative_error_denom.vector().set_local(y_test_data[i,:])
                relative_error_denom_L2 = norm(relative_error_denom, 'L2')
                relative_error_denom_H1 = norm(relative_error_denom, 'H1')
                error_L2 = norm(error, 'L2')
                error_H1 = norm(error, 'H1')

                L2_err = L2_err + np.abs(error_L2)**(2.0)*run_data['w_quadrature_weights_test'][i]
                H1_err = H1_err + np.abs(error_H1)**(2.0)*run_data['w_quadrature_weights_test'][i]
                L2_u = L2_u + np.abs(relative_error_denom_L2)**(2.0)*run_data['w_quadrature_weights_test'][i]
                H1_u = H1_u + np.abs(relative_error_denom_H1)**(2.0)*run_data['w_quadrature_weights_test'][i]

            L2_err = np.sqrt(np.abs(L2_err*2**(-1.0*run_data['input_dim'])))
            H1_err = np.sqrt(np.abs(H1_err*2**(-1.0*run_data['input_dim'])))
            L2_u = np.sqrt(np.abs(L2_u*2**(-1.0*run_data['input_dim'])))
            H1_u = np.sqrt(np.abs(H1_u*2**(-1.0*run_data['input_dim'])))
            L2_error_data.append(L2_err)
            H1_error_data.append(H1_err)
            time_intervals.append(time.time() - run_data['start_time'])
            test_time = time.time() - test_start_time
        else:
            L2err_diff  = np.sqrt(abs(np.sum(np.square(absdiff)*run_data['w_quadrature_weights_test']*2.0**(-1.0*run_data['input_dim']))))
            L2err_f     = np.sqrt(abs(np.sum(np.square(abs(y_test_data))*run_data['w_quadrature_weights_test']*2.0**(-1.0*run_data['input_dim']))))
            L2_error_LS = L2err_diff/L2err_f
            print('L2 least square error: ', L2_error_LS )
            l2err_dif   = np.sqrt(np.sum(np.square(absdiff))/args.nb_test_points) 
            l2err_f     = np.sqrt(np.sum(np.square(y_test_data))/args.nb_test_points) 
            l2_error_LS = l2err_dif/l2err_f
            print('l2 least square error: ', l2_error_LS )

            # Save data
            run_data['L2_error_least_squares'] = L2_error_LS
            run_data['l2_error_least_squares'] = l2_error_LS
            #----------------------------------------------------------------------#

        # Save data 
        run_data['iterations'] = steps_per_iter_global
        run_data['loss_per_iteration'] = loss_per_iter_global
        run_data['all_last_epochs_per_iter'] = all_last_epochs_per_iter
        run_data['all_first_epochs_per_iter'] = all_first_epochs_per_iter
        run_data['lrn_rates'] = all_lrn_rates_per_iter
        run_data['constant_C_from_Q_data'] = constant_C_from_Q_data
        run_data['y_DNN_pred']        = u(x_test_data)
        run_data['L2_error_data']     = L2_error_data  
        run_data['H1_error_data']     = H1_error_data  
        run_data['L2_u']              = L2_u
        run_data['H1_u']              = H1_u
        run_data['l2_error_data']     = l2_error_data  
        run_data['Chris_func_vals']   = Chris_function
        run_data['Prob_dist_vals']    = Prob_dist
        run_data['r_values']          = r_vals
        run_data['r_Phi_values']      = r_Phi_vals
        run_data['run_data_filename'] = 'run_data.mat' 

        # save the resulting mat file with scipy.io
        sio.savemat(run_data['result_folder'] + '/' + run_data['run_data_filename'], run_data)
        print('save to:',run_data['result_folder'] + '/' + run_data['run_data_filename'])

    elif args.training_method == 'MC2':

        # Error variables
        L2_error_data = np.array([])
        l2_error_data = np.array([])

        # Indices 
        #I_i = np.linspace(0, args.nb_train_points-1, args.nb_train_points).astype(int)
        I_i = np.arange(0,args.nb_train_points)

        for l in range(args.training_steps-1):
                                  
            #------------------------------------------------------------------------------#
            # set up the initializer for the weight and biase
            #------------------------------------------------------------------------------#
            if args.initializer == 'normal':
                weight_bias_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev= sigma, seed = trial)
            elif args.initializer == 'uniform':
                weight_bias_initializer = tf.keras.initializers.RandomUniform(minval= -sigma, maxval= sigma, seed= trial)
            elif args.initializer == 'constant':
                weight_bias_initializer = tf.keras.initializers.Constant(value= sigma)
            elif args.initializer == 'he_normal':
                weight_bias_initializer = tf.keras.initializers.HeNormal(seed= trial)
            elif args.initializer == 'xavier_normal':
                weight_bias_initializer = tf.keras.initializers.GlorotNormal(seed= trial)
            elif args.initializer == 'xavier_uniform':
                weight_bias_initializer = tf.keras.initializers.GlorotUniform(seed= trial)
            else:
                sys.exit('initializer must be one of the supported types, e.g., normal, uniform, etc.')

            #------------------------------------------------------------------------------#
            # set up learning rate schedule from either exp_decay linear, or constant
            #------------------------------------------------------------------------------#
            if args.lrn_rate_schedule == 'exp_decay':
                init_rate = 1e-3
                lrn_rate  = init_rate 

                # decay steps specifies how many epochs until the learning rate is 
                # decayed by specific amount
                decay_steps = 1e3

                #WARNING: Edit the parameter final stepsize for best results
                final_learning_rate = error_tol

                # calculate the base so that the learning rate schedule with exponential 
                # decay follows (init_rate)*(base)^(current_epoch/decay_step)
                base = np.exp(decay_steps/args.nb_epochs*(np.log(final_learning_rate)-np.log(init_rate)))

                # base on the above, the final learning rate is (init_rate)*(base)^(total_epoch/decay_step)
                print('based on init rate = ' + str(init_rate)
                    + ', decay_steps = ' + str(decay_steps)
                    + ', calculated base = ' + str(base)
                    + ', so that after ' + str(args.nb_epochs)
                    + ' epochs we have final learning rate = '
                    + str(init_rate*base**(args.nb_epochs/decay_steps)))
                
                decay_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                    init_rate, decay_steps, base, staircase= False, name= None)
            else:
                print('incorrect learning rate schedule selected')    

            #------------------------------------------------------------------------------#
            # set up optimizers for training
            #------------------------------------------------------------------------------#
            if args.optimizer == 'SGD':
                standard_opt = tf.keras.optimizers.SGD(
                    learning_rate= 1e-4,
                    name= 'SGD'
                )
            elif args.optimizer == 'Adam':
                standard_opt = tf.keras.optimizers.Adam(
                    learning_rate= decay_schedule,
                    beta_1= 0.9, beta_2= 0.999, epsilon= 1e-07,
                    name= 'Adam'
                )    
            else:
                sys.exit('optimizer must be one of the preset optimizers: SGD, Adam, etc.')    

            #------------------------------------------------------------------------------#
            # Create neural network model
            #------------------------------------------------------------------------------#
            all_layers = []

            u       = tf.keras.models.Sequential()
            input_u = tf.keras.layers.InputLayer(input_shape=(args.input_dim ,), name='input_u')
            u.add(input_u)

            # add 2nd to last layer
            for layer in range(args.nb_layers + 1):
                if layer == args.nb_layers:
                    layer_name = 'second_to_last_u'
                else:
                    layer_name = 'dense_u' + str(layer)

                this_layer = tf.keras.layers.Dense(args.nb_nodes_per_layer, 
                                                    activation=args.activation,
                                                    name=layer_name,
                                                    kernel_initializer= weight_bias_initializer,
                                                    bias_initializer= weight_bias_initializer,
                                                    dtype=tf.float32)
                all_layers.append(this_layer)
                u.add(this_layer)
            
            # output layer 
            output_layer = tf.keras.layers.Dense(args.output_dim,
                                                    activation= tf.keras.activations.linear,
                                                    trainable=True,
                                                    use_bias=False,
                                                    name='output_u',
                                                    kernel_initializer=weight_bias_initializer,
                                                    dtype=tf.float32)

            # add the output layer
            u.add(output_layer)

            # loss function 
            mse = tf.keras.losses.MeanSquaredError()

            # compile model
            u.compile(loss=mse, optimizer=standard_opt, metrics=['accuracy'])
        
            
            # number of variables
            model_num_trainable_variables = np.sum([np.prod(v.get_shape().as_list()) for v in u.trainable_variables])
            if not args.quiet:
                print('This model has {} trainable variables'.format(model_num_trainable_variables))
                u.summary()

            run_data['tf_trainable_vars'] = model_num_trainable_variables
            run_data['loss_function'] = args.loss_function
            
            #----------------------------------------------------------------------#
            # Uniform Sampling 
            M = M_values[l]
            # FIXME: the points are already randomly generated, why do we need to randomly subsample them???
            # to fix this, I just made it the first M indices 
            #I = random.choices(I_i, k= M)
            I = np.arange(0,M)

            # Convert I
            #I = np.array(I, dtype=np.float32).astype(int)
            #print(I_i)
            #print(I_i.shape)
            #print(I)
            #print(I.shape)
            #print(M_values)

            print('Number of points in I set: ' + str(I.shape[0]))    
            #----------------------------------------------------------------------#
            # Fit model
            run_data['x_train_data'] = x_train_data[I,:]
            run_data['y_train_data'] = y_train_data[I,:]    
            run_data['I_set']        = I
            run_data['M_values']     = M_values
            run_data['k_values']     = k_values
            
            print('here1')

            early_stopping_prediction_history = EarlyStoppingPredictHistory(run_data)

            u.fit(x_train_data[I,:], y_train_data[I,:], batch_size= M_values[l],
                    epochs= args.nb_epochs, verbose= 0, shuffle=True, 
                    callbacks=[early_stopping_prediction_history]
                    )

            print('here2')

            last_epoch_from_callback = early_stopping_prediction_history.steps[-1] - early_stopping_prediction_history.steps[0]
            print('callback says it ran for', last_epoch_from_callback, 'epochs')

            epochs_so_far = epochs_so_far + last_epoch_from_callback
            print('epochs_so_far: ', epochs_so_far)
            run_data['opt_final_epoch'] = epochs_so_far + args.nb_epochs_per_iter

            # Testing error
            y_DNN_pred    = u(x_test_data)
            absdiff       = abs(y_test_data - y_DNN_pred)
            
            L2err_diff    = np.sqrt(abs(np.sum(np.square(absdiff)*run_data['w_quadrature_weights_test']*2.0**(-1.0*run_data['input_dim']))))
            L2err_f       = np.sqrt(abs(np.sum(np.square(abs(y_test_data))*run_data['w_quadrature_weights_test']*2.0**(-1.0*run_data['input_dim']))))
            L2err         = L2err_diff/L2err_f
            L2_error_data = np.append(L2_error_data, [L2err])
            
            l2err_dif     = np.sqrt(np.sum(np.square(absdiff))/args.nb_test_points) 
            l2err_f       = np.sqrt(np.sum(np.square(y_test_data))/args.nb_test_points) 
            l2err         = l2err_dif/l2err_f
            l2_error_data = np.append(l2_error_data, [l2err])

            print('-----------------------------------------------------------------------')
            print('L2 error at the end of epochs training:' + str(L2err)) 
            print('L2 error at the end of epochs training:' + str(l2err))      
            
        
        # Save data 
        run_data['y_DNN_pred']        = u(x_test_data)
        run_data['L2_error_data']     = L2_error_data  
        run_data['l2_error_data']     = l2_error_data  
        run_data['run_data_filename'] = 'run_data.mat'      

        # save the resulting mat file with scipy.io
        sio.savemat(run_data['result_folder'] + '/' + run_data['run_data_filename'], run_data)

