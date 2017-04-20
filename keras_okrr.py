#   --------------------------------------
#   Copyright & Disclaimer
#   --------------------------------------
#
#   The programs contained in this package are granted free of charge for
#   research and education purposes only. Scientific results produced using
#   the software provided shall acknowledge the use of this implementation
#   provided by us. If you plan to use it for non-scientific purposes,
#   don't hesitate to contact us. Because the programs are licensed free of
#   charge, there is no warranty for the program, to the extent permitted
#   by applicable law. except when otherwise stated in writing the
#   copyright holders and/or other parties provide the program "as is"
#   without warranty of any kind, either expressed or implied, including,
#   but not limited to, the implied warranties of merchantability and
#   fitness for a particular purpose. the entire risk as to the quality and
#   performance of the program is with you. should the program prove
#   defective, you assume the cost of all necessary servicing, repair or
#   correction. In no event unless required by applicable law or agreed to
#   in writing will any copyright holder, or any other party who may modify
#   and/or redistribute the program, be liable to you for damages,
#   including any general, special, incidental or consequential damages
#   arising out of the use or inability to use the program (including but
#   not limited to loss of data or data being rendered inaccurate or losses
#   sustained by you or third parties or a failure of the program to
#   operate with any other programs), even if such holder or other party
#   has been advised of the possibility of such damages.
#
#   NOTE: This is just a demo providing a default initialization. Training
#   is not at all optimized. Other initializations, optimization techniques,
#   and training strategies may be of course better suited to achieve improved
#   results in this or other problems.
#
# Copyright (c) 2017 by Gonzalo Mateo-Garcia
# gonzalo.mateo-garcia@uv.es
# http://isp.uv.es/
#

from keras import backend as K
from keras.engine.topology import Layer
from keras.models import Model
from keras.layers import Dense,Input
from keras.optimizers import Adam
from keras import regularizers
from keras import metrics
from keras.callbacks import Callback

from scipy.spatial.distance import pdist
import numpy as np
import logging
from time import time

import log_conf as log

LOSS = "mse"
K.set_floatx("float64")

logger = logging.getLogger(__name__)
log.screen_logger(logger)


def init_gamma_numpy(x):
    D = pdist(x)
    return 1/(2*np.median(D[D>0]))


def k_test_numpy(x,x_test,gamma):
    K_2 = -np.sum((x - x_test[:,np.newaxis])**2,axis=-1)
    K_2=np.exp(gamma*K_2)
    return K_2


def krr_numpy(x,y,gamma,lda=.1):
    K_2 =  k_test_numpy(x,x,gamma)
    alpha = np.linalg.solve(K_2+lda*np.eye(K_2.shape[0]),y)
    err = np.sum((K_2.dot(alpha)-y)**2)
    return K_2,alpha,err


def k_test_keras(x,mu,gamma):
    d = K.sum(x**2,axis=1,keepdims=True)
    d_test = K.sum(mu**2,axis=1,keepdims=True)
    K_test = 2*K.dot(x,K.transpose(mu)) - d - K.transpose(d_test)
    
    #K_test_tf = -tf.reduce_sum((datos_entrada-tf.expand_dims(datos_entrada_test,axis=1))**2,
    #                           reduction_indices=2)
    
    return K.exp(gamma*K_test)


def alpha_keras(k_test,y,lmbda=0.,N_centers=None):
    if N_centers is None:
        shape_k = K.int_shape(k_test)
        N_centers = shape_k[1]
    
    y_proj = K.dot(K.transpose(k_test),y)
    k_to_inv = K.dot(K.transpose(k_test),k_test)
    k_to_inv +=  lmbda*K.eye(N_centers)
    if K.backend() == "theano":
        import theano.tensor as T
        new_val = K.dot(T.nlinalg.matrix_inverse(k_to_inv),
                        y_proj)

        #new_val = T.nlinalg.lstsq()(k_test,y,rcond=0.0001)
        #new_val = (new_val[0],new_val[3])
    else:
        import tensorflow as tf
        new_val = tf.matrix_solve(k_to_inv,y_proj)

    return new_val


class KernelLayer(Layer):
    def __init__(self, N_centers,optimize_gamma=False, optimize_mu=True, **kwargs):
        self.N_centers = N_centers
        self.optimize_gamma = optimize_gamma
        self.optimize_mu = optimize_mu
        super(KernelLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.mu = self.add_weight((self.N_centers,input_shape[1]),
                                   initializer='glorot_uniform',
                                   name="mu",trainable=self.optimize_mu)
        
        self.gamma = K.variable(np.random.rand(1)[0]*10+.001,
                                name="gamma")
        if self.optimize_gamma:
            self.trainable_weights += [self.gamma]
        else:
            self.non_trainable_weights += [self.gamma]
        
        super(KernelLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x, mask=None):
        return k_test_keras(x, self.mu, self.gamma)
    
    def compute_output_shape(self, input_shape):
        return input_shape[0], self.N_centers
    

def get_model(input_shape,output_shape,N_centers):
    if len(output_shape)== 1:
        output_shape=(output_shape[0],1)
    
    layer_alpha = Dense(output_shape[1],
                    name="alpha",use_bias=False)
    
    layer_k = KernelLayer(N_centers,name="k_test")
    input_l = Input(shape=input_shape)
    sal = layer_alpha(layer_k(input_l))
    model = Model(inputs=input_l,outputs=sal)   
    
    return model


def save_model(model,filename,extra_save={}):
    model.save_weights(filename)
    import h5py
    with h5py.File(filename, "r+") as f:
        f.attrs["N_centers"] = model.get_layer("k_test").output_shape[1]
        f.attrs["lmbda_init"] = model.lmbda_init
        f.attrs["N_input"] = model.get_layer("k_test").input_shape[1]
        f.attrs["N_output"] = model.get_layer("alpha").output_shape[1]
        f.attrs.update(extra_save)
        if isinstance(model, ModelCordinateDescentAlpha):
            f.attrs["ModelCordinateDescentAlpha"] = True


def load_model(filename):
    import h5py
    with h5py.File(filename, "r+") as f:
        N_centers = f.attrs["N_centers"]
        lmbda_init = f.attrs["lmbda_init"]
        N_input = f.attrs["N_input"]
        N_output = f.attrs["N_output"]
        cgd = False
        if "ModelCordinateDescentAlpha" in f.attrs:
            cgd = True
    
    model =  get_model((N_input,),(N_output,),N_centers=N_centers)
    model.load_weights(filename)
    if cgd:
        model = ModelCordinateDescentAlpha(model=model,
                                           lmbda_init=lmbda_init)
    else:
        setattr(model,"lmbda_init",lmbda_init)
    
    return model


def initialize_model(init_mu,init_y,model=None,
                     init_gamma=None,lmbda_init=.1,
                     regularize_alpha_layer=False):
    if len(init_y.shape) == 1:
        init_y = init_y[:,np.newaxis]
    
    if model is None:
        model = get_model(init_mu.shape[1:], init_y.shape[1:], init_mu.shape[0])
    
    layer_alpha = model.get_layer("alpha")
    layer_k = model.get_layer("k_test")
    
    if regularize_alpha_layer:
        kernel_regularizer=regularizers.l2(lmbda_init)
        layer_alpha.add_loss(kernel_regularizer(layer_alpha.kernel))
    
    if init_gamma is None:
        init_gamma = init_gamma_numpy(init_mu)
        logger.info("gamma not provided. Initialized to: %.4f"%init_gamma)
    
    layer_k.set_weights([init_mu, init_gamma])
            
    _,alpha_init,err = krr_numpy(init_mu,
                                 init_y, init_gamma,lmbda_init)
    
    layer_alpha.set_weights([alpha_init])
    mse = err/init_mu.shape[0]
    logger.info("MSE subset start: %.4f"%mse)
    #logger.info("Alpha initial norm: %.4f"%(np.sqrt(np.sum(alpha_init**2))))
    
    return model


def model_all_gd(init_mu,init_y,init_gamma=None,optimizer=Adam(),
                 lmbda_init=.1,regularize_alpha_layer=False):
    
    model = initialize_model(init_mu=init_mu,
                             init_y=init_y,
                             init_gamma=init_gamma,
                             lmbda_init=lmbda_init,
                             regularize_alpha_layer=regularize_alpha_layer)
    model.compile(optimizer, LOSS,
                  metrics=[metrics.mean_squared_error])
    setattr(model,"lmbda_init",lmbda_init)
    return model


def model_coordinate_descent_alpha(init_mu,init_y,init_gamma=None,
                                   use_next_value=True,optimizer=Adam(),
                                   lmbda_init=.1):
    model = ModelCordinateDescentAlpha(init_mu=init_mu,init_y=init_y,
                                       init_gamma=init_gamma,
                                       use_next_value=use_next_value,
                                       lmbda_init=lmbda_init)
    
    model.compile(optimizer, LOSS,
                  metrics=[metrics.mean_squared_error])
    return model


class ModelCordinateDescentAlpha(Model):
    def __init__(self,init_mu=None,init_y=None,
                 model=None,init_gamma=None,
                 use_next_value=True, lmbda_init=.1):
        
        if model is None:
            model = initialize_model(init_mu=init_mu,
                                     init_y=init_y,
                                     init_gamma=init_gamma,
                                     lmbda_init=lmbda_init)
        
        super(ModelCordinateDescentAlpha,self).__init__(inputs=model.inputs,
                                                        outputs=model.outputs)
        
        self.layer_alpha = self.get_layer("alpha")
        self.layer_k = self.get_layer("k_test")

        self.layer_alpha.trainable = False
        self.use_next_value = use_next_value   
        self.lmbda_init = lmbda_init
        
    
    def _make_train_function(self):     
        if not hasattr(self, 'train_function'):
            raise RuntimeError('You must compile your model before using it.')
        if self.train_function is None:
            inputs = self._feed_inputs + self._feed_targets + self._feed_sample_weights
            if self.uses_learning_phase and not isinstance(K.learning_phase(), int):
                inputs += [K.learning_phase()]

            training_updates = self.optimizer.get_updates(self._collected_trainable_weights,
                                                          self.constraints,
                                                          self.total_loss)
            pos = len(training_updates)//len(self._collected_trainable_weights) -1
            if self.use_next_value:           
                new_val_mu = training_updates[pos][1] if K.backend() == "theano" else training_updates[pos]
                
                K_test = k_test_keras(self.layer_k.input,
                                      new_val_mu,
                                      self.layer_k.gamma)
            else:
                K_test = self.layer_k.output
            
            new_val = alpha_keras(K_test,
                                        self.targets[0], 
                                        lmbda=self.lmbda_init,
                                        N_centers=self.layer_k.output_shape[1])

            new_val = K.cast(new_val,K.dtype(self.layer_alpha.kernel))
            training_updates.append(K.update(self.layer_alpha.kernel, new_val))
            
            updates = self.updates + training_updates

            # returns loss and metrics. Updates weights at each call.
            self.train_function = K.function(inputs,
                                             [self.total_loss] + self.metrics_tensors,
                                             updates=updates,
                                             **self._function_kwargs)


class TimeLogger(Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.time = 0 
        self.start = 0        
    
    def on_batch_begin(self, batch, logs=None):
        self.start = time()
        
    def on_batch_end(self, batch, logs=None):
        self.time += time() - self.start
    
    def on_epoch_end(self, epoch, logs=None):
        logs["time"] = self.time


def fn_to_abr(fn, batch_size):
    if fn == model_all_gd.__name__:
        abr = "GD"
    elif fn == model_coordinate_descent_alpha.__name__:
        abr = "OA"
    if batch_size != 0:
        abr = "MB-%s"%abr
    return abr
