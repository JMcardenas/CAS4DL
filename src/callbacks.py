import os, time
import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
from dolfin import *
from fenics import *
import scipy.io as sio

class EarlyStoppingPredictHistory(tf.keras.callbacks.Callback):
    """Stop training when the loss is at its min, i.e. the loss stops decreasing. 
    Also tests the model while training, using an update ratio to prevent testing too often. 
    Note: the best weights for this checkpoint are held in memory, and restored after training 
    if the loss for the checkpoint is better than the current loss.

  Arguments:
        run_data: dictionary of options for the callback and running
  """

    def __init__(self, run_data):
        super(EarlyStoppingPredictHistory, self).__init__()
        self.run_data = run_data
        self.fenics_params = self.run_data['fenics_params']
        self.best_loss = 1e12
        self.best_loss_epoch = -1
        # best_weights to store the weights at which the minimum loss occurs.
        self.best_weights = None

    def on_train_begin(self, logs={}):
        self.predictions = []
        self.steps = []
        self.losses = []
        self.L2_test_errs = []
        self.H1_test_errs = []
        self.time_intervals = []
        self.lrn_rates = np.array([])
        self.last_ckpt_loss = np.Inf
        self.stopped_epoch = -1
        if self.run_data['LSGD']:
            self.x = tf.constant(self.run_data['x_train_data'], dtype=tf.float32)
            self.B = tf.keras.Model(inputs=self.model.input, 
                outputs=self.model.get_layer('dense_' 
                                            + str(self.run_data['nb_layers'])).output)
            self.B(self.x)

        # variable holding percentage of testing points with error above 10^{-k} for various thresholds k
        self.percs = []
        self.num_perc = np.array([])

        # keep track of the minimum loss (corresponding to the last save)
        self.last_output_loss = 10
        self.last_output_epoch = -1

    def on_epoch_end(self, epoch, logs=None):

        if self.run_data['LSGD']:
            final_layer_weights = self.model.get_layer('output').get_weights()
            c = tf.linalg.lstsq(self.B(self.x),self.run_data['y_train_data'],fast=False)
            final_layer_weights = [c]
            self.model.get_layer('output').set_weights(final_layer_weights)

        current_loss = logs.get("loss")
        current_learning_rate = K.eval(self.model.optimizer.lr.__call__(epoch))

        self.losses = np.append(self.losses, [logs["loss"]])
        self.lrn_rates = np.append(self.lrn_rates, [current_learning_rate])

        self.steps.append(epoch)

        if (current_loss < self.best_loss):

            # record the best loss
            self.best_loss = current_loss
            self.best_loss_epoch = epoch

            # start the epoch timer for waiting if using patience
            self.wait = 0

            # Record the best weights if current results is better (less).
            self.best_weights = self.model.get_weights()

            # check if the loss has decreased enough (or more than 10k epochs have gone by) 
            # to evaluate the model on the testing data (just for command line output purposes)
            if (current_loss/self.last_output_loss < self.run_data['update_ratio']) or (epoch - self.last_output_epoch > 1e4):

                # update the last output epoch for checking if 10k have gone by
                self.last_output_epoch = epoch

                # update the last output loss for checking if loss decreased enough
                if (current_loss/self.last_output_loss < self.run_data['update_ratio']):
                    self.last_output_loss = current_loss

                # predict on the x_test_data
                y_DNN_pred = self.model.predict(self.run_data['x_test_data'])

                # compute the absolute difference between the trained model 
                # and the true data
                absdiff = abs(self.run_data['y_test_data'] - y_DNN_pred)

                # compute percentages of points with errors above thresholds
                perc_e1 = (1.0*(absdiff > 10)).sum()/self.run_data['nb_test_points']*100.0
                perc_e0 = (1.0*(absdiff > 1)).sum()/self.run_data['nb_test_points']*100.0
                perc_em1 = (1.0*(absdiff > 1e-1)).sum()/self.run_data['nb_test_points']*100.0
                perc_em2 = (1.0*(absdiff > 1e-2)).sum()/self.run_data['nb_test_points']*100.0
                perc_em3 = (1.0*(absdiff > 1e-3)).sum()/self.run_data['nb_test_points']*100.0
                perc_em4 = (1.0*(absdiff > 1e-4)).sum()/self.run_data['nb_test_points']*100.0
                perc_em5 = (1.0*(absdiff > 1e-5)).sum()/self.run_data['nb_test_points']*100.0
                perc_em6 = (1.0*(absdiff > 1e-6)).sum()/self.run_data['nb_test_points']*100.0

                # the l-infinity error over the test points
                linferr = np.amax(absdiff)

                # the l2err over the test points
                l2err = np.sqrt(sum(np.square(absdiff)))

                if self.run_data['test_pointset'] == 'CC_sparse_grid':
                    # the L2 error on the testing points computed with a SG quadrature rule
                    if self.run_data['PDE_data']:
                        test_start_time = time.time()

                        # the L2 and H1 errors we compute
                        L2_err = 0.0
                        H1_err = 0.0

                        error = Function(self.fenics_params['V'])

                        for i in range(self.run_data['nb_test_points']):
                            error.vector().set_local(absdiff[i,:])
                            error_L2 = norm(error, 'L2')
                            error_H1 = norm(error, 'H1')

                            L2_err = L2_err + np.abs(error_L2)**(2.0)*self.run_data['w_quadrature_weights_test'][i]
                            H1_err = H1_err + np.abs(error_H1)**(2.0)*self.run_data['w_quadrature_weights_test'][i]

                        L2_err = np.sqrt(np.abs(L2_err*2**(-1.0*self.run_data['input_dim'])))
                        H1_err = np.sqrt(np.abs(H1_err*2**(-1.0*self.run_data['input_dim'])))
                        self.L2_test_errs.append(L2_err)
                        self.H1_test_errs.append(H1_err)
                        self.time_intervals.append(time.time() - self.run_data['start_time'])
                        test_time = time.time() - test_start_time
                    else:
                        L2_err = np.sqrt(abs(np.sum(np.square(absdiff)*self.run_data['w_quadrature_weights_test']*2.0**(-1.0*self.run_data['input_dim']))))
                else:
                    L2_err = -1.0

                # assign to running stats
                if (epoch == 0) or (len(self.percs) == 0):
                    self.num_perc = np.append(self.num_perc, epoch)
                    self.percs = [perc_e1, perc_e0, 
                            perc_em1, perc_em2, 
                            perc_em3, perc_em4, 
                            perc_em5, perc_em6]
                else:
                    self.num_perc = np.append(self.num_perc, epoch)
                    self.percs = np.vstack([self.percs, [perc_e1, perc_e0, 
                                            perc_em1, perc_em2, 
                                            perc_em3, perc_em4, 
                                            perc_em5, perc_em6]])

                if not self.run_data['quiet']:
                    print("error_cond: %s, loss: %8.4e, error: linf = %6.4f, L2 = %6.4f, %%>1ek: " \
                        "0 = %6.4f, -1 = %6.4f, -2 = %6.4f, -3 = %6.4f, -4 = %6.4f, -5 = %6.4f, -6 = %6.4f" \
                        "  lrn_rate = %6.4e"
                        % (str(epoch).zfill(8), logs["loss"], linferr, L2_err, perc_e0, perc_em1, perc_em2, perc_em3, 
                            perc_em4, perc_em5, perc_em6, current_learning_rate))

        else:
            self.wait += 1

            # With patience large, this will never happen. However, if a small value of patience is used, 
            # then the model weights will be replaced with the best weights seen so far according to the loss
            if self.wait >= self.run_data['patience']:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print("Restoring model weights from the end of the best epoch.")
                self.model.set_weights(self.best_weights)

        # if the model has converge or run out of epochs of training, or if 1000 epochs have passed
        if (epoch == 0) or (current_loss <= self.run_data['error_tol']) or (epoch == self.run_data['nb_epochs'] - 1) or (epoch == self.run_data['opt_final_epoch'] - 1) or self.model.stop_training:
            y_DNN_pred = self.model.predict(self.run_data['x_test_data'])

            # compute the absolute difference between the trained model 
            # and the true data
            absdiff = abs(self.run_data['y_test_data'] - y_DNN_pred)

            # compute percentages of points with errors above thresholds
            perc_e1 = (1.0*(absdiff > 10)).sum()/self.run_data['nb_test_points']*100.0
            perc_e0 = (1.0*(absdiff > 1)).sum()/self.run_data['nb_test_points']*100.0
            perc_em1 = (1.0*(absdiff > 1e-1)).sum()/self.run_data['nb_test_points']*100.0
            perc_em2 = (1.0*(absdiff > 1e-2)).sum()/self.run_data['nb_test_points']*100.0
            perc_em3 = (1.0*(absdiff > 1e-3)).sum()/self.run_data['nb_test_points']*100.0
            perc_em4 = (1.0*(absdiff > 1e-4)).sum()/self.run_data['nb_test_points']*100.0
            perc_em5 = (1.0*(absdiff > 1e-5)).sum()/self.run_data['nb_test_points']*100.0
            perc_em6 = (1.0*(absdiff > 1e-6)).sum()/self.run_data['nb_test_points']*100.0

            # the l-infinity error over the test points
            linferr = np.amax(absdiff)

            # the l2err over the test points
            l2err = np.sqrt(sum(np.square(absdiff)))

            if self.run_data['test_pointset'] == 'CC_sparse_grid':
                # the L2 error on the testing points computed with a SG quadrature rule
                if self.run_data['PDE_data']:
                    test_start_time = time.time()

                    # the L2 and H1 errors we compute
                    L2_err = 0.0
                    H1_err = 0.0

                    error = Function(self.fenics_params['V'])

                    for i in range(self.run_data['nb_test_points']):
                        error.vector().set_local(absdiff[i,:])
                        error_L2 = norm(error, 'L2')
                        error_H1 = norm(error, 'H1')

                        L2_err = L2_err + np.abs(error_L2)**(2.0)*self.run_data['w_quadrature_weights_test'][i]
                        H1_err = H1_err + np.abs(error_H1)**(2.0)*self.run_data['w_quadrature_weights_test'][i]

                    L2_err = np.sqrt(np.abs(L2_err*2**(-1.0*self.run_data['input_dim'])))
                    H1_err = np.sqrt(np.abs(H1_err*2**(-1.0*self.run_data['input_dim'])))
                    self.L2_test_errs.append(L2_err)
                    self.H1_test_errs.append(H1_err)
                    self.time_intervals.append(time.time() - self.run_data['start_time'])
                    test_time = time.time() - test_start_time
                else:
                    L2_err = np.sqrt(abs(np.sum(np.square(absdiff)*self.run_data['w_quadrature_weights_test']*2.0**(-1.0*self.run_data['input_dim']))))
            else:
                L2_err = -1.0

            # assign to running stats
            if (epoch == 0) or (len(self.percs) == 0):
                self.num_perc = np.append(self.num_perc, epoch)
                self.percs = [perc_e1, perc_e0, 
                        perc_em1, perc_em2, 
                        perc_em3, perc_em4, 
                        perc_em5, perc_em6]
            else:
                self.num_perc = np.append(self.num_perc, epoch)
                self.percs = np.vstack([self.percs, [perc_e1, perc_e0, 
                                        perc_em1, perc_em2, 
                                        perc_em3, perc_em4, 
                                        perc_em5, perc_em6]])

            self.run_data['run_time'] = time.time() - self.run_data['start_time']
            self.run_data['percentiles_at_save'] = self.percs
            self.run_data['percentiles_save_iters'] = self.num_perc
            self.run_data['y_DNN_pred'] = y_DNN_pred
            self.run_data['iterations'] = self.steps
            self.run_data['loss_per_iteration'] = self.losses
            self.run_data['lrn_rates'] = self.lrn_rates

            # save the resulting mat file with scipy.io
            sio.savemat(self.run_data['result_folder'] + '/' + self.run_data['run_data_filename'], self.run_data)

            # if we've converged to the error tolerance in the loss, or run
            # into the maximum number of epochs, stop training and save
            if (current_loss <= self.run_data['error_tol']) or (epoch == self.run_data['nb_epochs'] - 1) or (epoch == self.run_data['opt_final_epoch'] - 1) or self.model.stop_training:
                # output the final checkpoint loss and statistics
                if not self.run_data['quiet']:
                    print("final: %s, loss: %8.4e, error: linf = %6.4f, L2 = %6.4f, %%>1ek: " \
                        "0 = %6.4f, -1 = %6.4f, -2 = %6.4f, -3 = %6.4f, -4 = %6.4f, -5 = %6.4f, -6 = %6.4f" \
                        "  lrn_rate = %6.4e"
                        % (str(epoch).zfill(8), logs["loss"], linferr, L2_err, perc_e0, perc_em1, perc_em2, perc_em3,
                            perc_em4, perc_em5, perc_em6, current_learning_rate))

                print('Current loss at epoch %s:   %8.12e' % (str(epoch).zfill(8), current_loss))
                print('Best loss at epoch    %s:   %8.12e' % (str(self.best_loss_epoch).zfill(8), self.best_loss))
                if current_loss <= self.best_loss:
                    print("Saving model with current loss.")
                    self.stopped_epoch = epoch
                else:
                    print("Restoring model weights from the end of the best epoch.")
                    self.stopped_epoch = self.best_loss_epoch
                    self.model.set_weights(self.best_weights)

                self.model.save(self.run_data['result_folder'] + '/' + self.run_data['model_save_folder'])

                self.run_data['run_time'] = time.time() - self.run_data['start_time']
                self.run_data['percentiles_at_save'] = self.percs
                self.run_data['percentiles_save_iters'] = self.num_perc
                self.run_data['y_DNN_pred'] = y_DNN_pred
                self.run_data['iterations'] = self.steps
                self.run_data['loss_per_iteration'] = self.losses
                self.run_data['lrn_rates'] = self.lrn_rates
                self.run_data['stopped_epoch'] = self.stopped_epoch
                self.run_data['best_loss'] = self.best_loss
                self.run_data['best_loss_epoch'] = self.best_loss_epoch

                # save the resulting mat file with scipy.io
                sio.savemat(self.run_data['result_folder'] + '/' + self.run_data['run_data_filename'], self.run_data)

                self.model.stop_training = True

        if (epoch % 1000 == 0):
            if not self.run_data['quiet']:
                print('epoch: ' + str(epoch).zfill(8) + ', loss: %8.4e, lrn_rate: %4.4e, seconds: %8.2f ' \
                    % (logs["loss"], current_learning_rate, time.time() - self.run_data['start_time']))


    def on_train_end(self, logs=None):
        if (self.stopped_epoch > 0) and (self.stopped_epoch < self.run_data['nb_epochs'] - 1):
            print("Epoch %05d: early stopping" % (self.stopped_epoch))

