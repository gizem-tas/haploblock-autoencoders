"""autoencoder - a module for creating and wrapping autoencoder 
models using the tensorflow API.
=============================================================
# Created By  : T. Westerdijk
=============================================================
Autoencoders are neural networks which attempt to reconstruct
the input layer as their output layer while constraining the 
flow of information through the use of a bottleneck.

This module contains functions to create a 
tensorflow.keras.Sequential autoencoder model, as well as a 
wrapper API class to use the autoencoder model with other
libraries, like scikit-learn.
"""

import os
import copy
import types
import logging

import numpy as np

from tensorflow import int64
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation, Dense, LeakyReLU
from tensorflow.keras.regularizers import L1

from tensorflow.keras import backend as K
from tensorflow.keras.backend import tanh
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.optimizers import Adam

from tensorflow.python.keras.utils.generic_utils import has_arg

from tensorflow.python.ops.math_ops import cast, count_nonzero_v2
from tensorflow.python.ops.array_ops import size_v2
from tensorflow.python.ops.gen_math_ops import rint


class BaseWrapper():
    # Adapted from `tensorflow.keras.wrappers.scikit_learn.py`.
    """Base class for the Keras scikit-learn wrapper.
    
    Arguments:
        build_fn: callable function or class instance
        **sk_params: model parameters & fitting parameters
    """
    
    def __init__(self, build_fn=None, **sk_params):
        self.build_fn = build_fn
        self.sk_params = sk_params
        
    def check_params(self, params):
        """Checks for user typos in `params`.
        """

        legal_params_fns = [
            Sequential.fit, Sequential.predict, Sequential.evaluate
        ]
        if self.build_fn is None:
            legal_params_fns.append(self.__call__)
        elif (not isinstance(self.build_fn, types.FunctionType) and
              not isinstance(self.build_fn, types.MethodType)):
            legal_params_fns.append(self.build_fn.__call__)
        else:
            legal_params_fns.append(self.build_fn)

        for params_name in params:
            for fn in legal_params_fns:
                if has_arg(fn, params_name):
                    break
            else:
                if params_name != 'nb_epoch':
                    raise ValueError('{} is not a legal parameter'.format(params_name))
    
    def get_params(self, **params):
        """Gets parameters for this estimator.
        """
        
        res = copy.deepcopy(self.sk_params)
        res.update({"build_fn": self.build_fn})
        return res
    
    def set_params(self, **params):
        """Sets the parameters of this estimator.
        """
        
        self.check_params(params)
        self.sk_params.update(params)
        return self
    
    def filter_sk_params(self, fn, override=None):
        """Filters `sk_params` and returns those in `fn`'s arguments.
        """
        override = override or {}
        res = {}
        for name, value in self.sk_params.items():
            if has_arg(fn, name):
                res.update({name: value})
        res.update(override)
        return res
    
    def fit(self, x, y, **kwargs):
        """Constructs a new model with `build_fn` & fit the model to `(x, y)`.
        """
        
        if self.build_fn is None:
            self.model = self.__call__(**self.filter_sk_params(self.__call__))
        elif (not isinstance(self.build_fn, types.FunctionType) and
              not isinstance(self.build_fn, types.MethodType)):
            self.model = self.build_fn(**self.filter_sk_params(self.build_fn.__call__))
        else:
            self.model = self.build_fn(**self.filter_sk_params(self.build_fn))
        
        fit_args = copy.deepcopy(self.filter_sk_params(Sequential.fit))
        fit_args.update(kwargs)
        
        history = self.model.fit(x, y, **fit_args)
        
        return history

    
class Classifier(BaseWrapper):
    # Classifier class is based on the KerasClassifier class from
    # `tensorflow.keras.wrappers.scikit_learn.py` and customized to
    # allow `Classifier.score()` to accept any compiled scoring 
    # metric of the model, including loss.
    """Implementation of the scikit-learn classifier API for Keras.
    Compatible with the keras Sequential model returned by
    `autoencoder.create_model()`.

    Arguments:
        build_fn: callable function or class instance
        **sk_params: model parameters & fitting parameters
    """
    
    def fit(self, x, y, **kwargs):
        """Constructs a new model with `build_fn` & fit the model to `(x, y)`.
        """
        
        y = np.array(y)
        if len(y.shape) == 2 and y.shape[1] > 1:
            self.classes_ = np.unique(y)
            y = np.searchsorted(self.classes_, y)
        else:
            raise ValueError("Invalid shape for y: " + str(y.shape))
        self.n_classes_ = len(self.classes_)
        return super(Classifier, self).fit(x, y, **kwargs)
    
    def predict(self, x, **kwargs):
        """Returns the predictions for the given test data.
        """
        
        kwargs = self.filter_sk_params(Sequential.predict, kwargs)
        preds = self.model.predict(x, **kwargs)
        return preds
    
    def score(self, x, y, scoring = "snp_accuracy", greater_is_better = True, **kwargs):
        """Returns the passed `scoring` metric on the given test data and labels.
        If the returned value should be minimized, set the `greater_is_better` 
        parameter to `False`.
        """
        
        eval_kwargs = self.filter_sk_params(Sequential.evaluate, kwargs)
        score_kwargs = self.filter_sk_params(self.score, kwargs)
        
        if "scoring" in score_kwargs.keys():
            scoring = score_kwargs["scoring"]
        if "greater_is_better" in score_kwargs.keys():
            greater_is_better = score_kwargs["greater_is_better"]
        
        outputs = self.model.evaluate(x, y, **eval_kwargs, verbose = 0)
        if not isinstance(outputs, list):
            outputs = [outputs]
        for name, output in zip(self.model.metrics_names, outputs):
            if name == scoring:
                return output if greater_is_better else -output
        raise ValueError(f"The model is not configured to compute {scoring}. "
                         f"You should pass `metrics=[{scoring}]` to " 
                         "the `model.compile()` method.")
        
        
def snp_accuracy(y_true, y_pred):
    """Custom tensorflow metrics function to calculate snp accuracy.
    
    Arguments:
        y_true: target array (tensorflow.tensor)
        y_pred: prediction array (tensorflow.tensor)
        
    Returns:
        tensorflow.tensor accuracy value
    """
    comp = (rint(y_pred) == y_true)
    nonz = cast(count_nonzero_v2(comp), dtype=int64)
    size = cast(size_v2(comp), dtype=int64)
    acc = nonz/size
    return acc

def snp_accuracy_numpy(y_true, y_pred):
    """Custom tensorflow metrics function to calculate snp accuracy.
        Use for scoring in cross_validate.
    
    Arguments:
        y_true: target array (tensorflow.tensor)
        y_pred: prediction array (tensorflow.tensor)
        
    Returns:
        tensorflow.tensor accuracy value converted to numpy for sklearn compatibility
    """
    comp = (rint(y_pred) == y_true)
    nonz = cast(count_nonzero_v2(comp), dtype=int64)
    size = cast(size_v2(comp), dtype=int64)
    acc = nonz/size
    acc = acc.numpy()
    return acc

def additive(x):
    """Custom activation function [tanh(x)+1] with an output range from 0 to 2.
    """
    return tanh(x)+1
get_custom_objects().update({'additive': Activation(additive)})

def var_acc(model, data):
    prediction = np.rint(model.predict(data))
    
    acc_dict = {}
    for variant in [0, 1, 2]:
        var_loc = np.where(data==variant, data, None)
        correct = np.where(prediction==var_loc, True, False)
        var_acc = np.count_nonzero(correct)/np.count_nonzero(var_loc==variant)

        acc_dict[variant] = var_acc
            
    return acc_dict

def model_shape(inputs, shape="sqrt", hl=3, bn=3):
    """Function for determining the number of nodes for the hidden layers.
    
    Arguments:
        inputs: number of input nodes (int)
        shape: name of the ae shape (str: `sqrt`|`quadr`|`lin`|`block`), default `sqrt`
        hl: number of hidden layers (int), default 3
        bn: number of bottleneck nodes (int), default 3
        
    Returns:
        tuple(encoder list, decoder list) 
    """
    
    shape_dict={"sqrt": .5, "quadr": 2, "lin": 1, "block": 0}
    if shape not in shape_dict:
        raise ValueError(f"`shape` should be either `sqrt`, `quadr`, `lin`, or `block`, `{shape}` is an invalid option.")
        
    power   = shape_dict[shape]
    y       = [i**power for i in range(hl+2)]
    
    ratio   = (inputs-bn)/y[-1] 
    decoder = [round(i*ratio + bn) for i in y[1:-1]]
    encoder = decoder[::-1]
    
    return encoder, decoder


def create_model(inputs, shape="sqrt", hl=4, bn=3, activation=LeakyReLU(), **kwargs):
    """Function for creating and compiling a keras Sequential autoencoder model.
    
    Arguments:
        inputs: number of input nodes (int)
        shape: name of the ae shape (str: `sqrt`|`quadr`|`lin`|`block`), default `sqrt`
        hl: number of hidden layers (int), default 3
        bn: number of bottleneck nodes (int), default 3
        activation: activation function of hidden layers, default LeakyReLU
    
    Returns:
        tensorflow.keras.Sequential autoencoder model
    """
    K.clear_session()
    
    ae = Sequential(name="ae")
    ae.add(Dense(inputs, activation=activation, input_shape=(inputs,), name="inputs"))
    
    encoder, decoder = model_shape(inputs=inputs, shape=shape, hl=hl, bn=bn)
    
    for nodes in encoder:
        ae.add(Dense(nodes, activation=activation, kernel_initializer='he_uniform'))
    ae.add(Dense(bn, activation=activation, kernel_initializer='he_uniform', name="bottleneck")) #activity regularizer comes here when the autoencoder is sparse
    for nodes in decoder:
        ae.add(Dense(nodes, activation=activation, kernel_initializer='he_uniform'))
    ae.add(Dense(inputs, activation="additive", name="output"))
    
    ae.compile(optimizer=Adam(learning_rate=0.0001), loss="mse", metrics=[snp_accuracy])
    
    return ae
    
def create_sparse_model(inputs, shape="sqrt", hl=3, bn=3, activation=LeakyReLU(), **kwargs):
    """Function for creating and compiling a keras Sequential autoencoder model.
    
    Arguments:
        inputs: number of input nodes (int)
        shape: name of the ae shape (str: `sqrt`|`quadr`|`lin`|`block`), default `sqrt`
        hl: number of hidden layers (int), default 3
        bn: number of bottleneck nodes (int), default 3
        activation: activation function of hidden layers, default LeakyReLU
    
    Returns:
        tensorflow.keras.Sequential autoencoder model
    """
    K.clear_session()
    
    ae = Sequential(name="ae")
    ae.add(Dense(inputs, activation=activation, input_shape=(inputs,), name="inputs"))
    
    encoder, decoder = model_shape(inputs=inputs, shape=shape, hl=hl, bn=bn)
    
    for nodes in encoder:
        ae.add(Dense(nodes, activation=activation))
    ae.add(Dense(bn, activation=activation, activity_regularizer=L1(0.01), name="bottleneck")) #activity regularizer comes here when the autoencoder is sparse
    for nodes in decoder:
        ae.add(Dense(nodes, activation=activation))
    ae.add(Dense(inputs, activation="additive", name="output"))
    
    ae.compile(optimizer="adam", loss="mse", metrics=[snp_accuracy])
    
    return ae

def extract_encoder(model):
    """Function to extract encoder part of a (trained) keras 
    Sequential autoencoder model.

    Arguments:
        model: keras Sequential autoencoder model

    Returns:
        keras Sequential encoder model
    """
    return Model(inputs=model.inputs, outputs=model.get_layer("bottleneck").output)

def rounding_diffs(model, data):
    """Function that returns a list with the differences
    between the predicted rounded outputs and the 
    non-rounded outputs.

    Arguments:
        model: keras Sequential autoencoder model
        data: data to evaluate

    Returns:
        list with rounding differences
    """
    pred = model.predict(data)
    rpred = np.rint(model.predict(data))
    np_list = list((pred - rpred).flatten())
    return list(map(float, np_list))



def silence_tensorflow():
    """Silence all warnings from tensorflow."""
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    os.environ["KMP_AFFINITY"] = "noverbose"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    try:
        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')
        tf.autograph.set_verbosity(3)
    except ModuleNotFoundError:
        pass
