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
# Copyright (c) 2017 by Gonzalo Mateo-García, Jordi Muñoz-Marí, Valero Laparra, Luis Gómez-Chova
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

import numpy as np
import logging

#import log_conf as log

LOSS = "mse"
K.set_floatx("float64")

logger = logging.getLogger(__name__)
#log.screen_logger(logger)


def init_gamma_numpy(x):
    from scipy.spatial.distance import pdist
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


class OKRRReg(regularizers.Regularizer):
    """Regularizer for L1 and L2 regularization.

    # Arguments
        l1: Float; L1 regularization factor.
        l2: Float; L2 regularization factor.
    """

    def __init__(self,kuu, l2=0.):
        self.l2 = K.cast_to_floatx(l2)
        self.kuu = kuu

    def __call__(self, x):
        regularization = 0.
        if self.l2:
            regularization +=self.l2* K.squeeze(K.squeeze(K.dot(K.dot(K.transpose(x),self.kuu),x),axis=0),
                                                axis=0)

        return regularization


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
        logger.info("Regularize alpha: %.4f" % lmbda_init)
        kuu = k_test_keras(layer_k.mu, layer_k.mu, layer_k.gamma)
        kernel_regularizer=OKRRReg(kuu,lmbda_init)
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
