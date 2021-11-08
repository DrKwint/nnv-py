"""
DNN wrapper for a standard internal representation
"""
# pylint: disable=import-outside-toplevel, import-error

import numpy as np
import tree

from nnv_py.nnv_py import PyDNN  # pylint: disable=no-name-in-module


class DNN:
    """
    DNN class provides a standard interface to a deep neural network originally constructed in
    Python with any of the many libraries used for that purpose.
    """
    def __init__(self, network):
        """Initialize a DNN object (should only be called once, unless architecture changes)

        Args:
            network: Python object with the DNN architecture to build the internal network
                     representation. A TF2 network can be passed in directly, but a TF1 network
                     should use a session.run to retrieve the network parameters to pass in. If
                     you're having trouble with NotImplementedError exceptions, contact us and
                     we'll implement for your use case.
        """
        self.dnn = PyDNN()
        type_ = type(network)
        if 'tensorflow' in str(type_):  # TF2
            import tensorflow as tf
            assert isinstance(network, tf.Module)
            self._build_from_tensorflow_module(network)
        elif type_ == list:  # TF1
            # Assume that inputs are tuples (weights, bias) and that each laye
            if not isinstance(network[0], tuple):
                network = list(zip(network[::2], network[1::2]))
            assert isinstance(network[0], tuple)
            assert isinstance(network[0][0], np.ndarray)
            self._build_from_tensorflow_params(network)
        elif 'torch.nn' in str(type_):
            import torch
            if isinstance(network, torch.nn.Sequential):
                self._build_from_torch_sequential(network)
            else:
                raise NotImplementedError(str(type_))
        else:
            raise NotImplementedError(str(type_))

    def input_shape(self):
        """Suffix of the expected input shape to the network"""
        return self.dnn.input_shape()

    def deeppoly_bounds(self, lower: np.ndarray, upper: np.ndarray):
        """Rapidly calculate DeepPoly bounds on the output

        Args:
            lower (ndarray): Lower bound on each input dimension
            upper (ndarray): Upper bound on each input dimension

        Returns:
            (ndarray, ndarray): tuple of output lower and output bounds
        """
        return self.dnn.deeppoly_output_bounds(lower, upper)

    def _build_from_torch_sequential(self, sequential):
        import torch
        for i, layer in enumerate(sequential):
            if isinstance(layer, torch.nn.Linear):
                weight = layer.weight.detach().numpy()
                bias = layer.bias.detach().numpy()
                self.dnn.add_dense(weight, bias)
            elif isinstance(layer, torch.nn.ReLU):
                self.dnn.add_relu(bias.shape[-1])
            else:
                raise NotImplementedError("Unrecognized PyTorch layer!")

    def _build_from_tensorflow_params(self, affine_list):
        nlayers = len(affine_list)
        for i, aff in enumerate(affine_list):
            # Add dense
            assert len(aff[0].shape) == 2, aff
            self.dnn.add_dense(aff[0].T, aff[1])
            # Add relu
            if i != nlayers - 1:
                self.dnn.add_relu(aff[1].shape[0])

    def _build_from_tensorflow_module(self, network):
        import tensorflow as tf
        from tensorflow.keras.layers import (Conv2D, Dense, Flatten,
                                             InputLayer, MaxPooling2D)

        submodules = network.layers
        submodules = tree.flatten(submodules)
        while any(x.submodules for x in submodules):
            submodules = [
                x.submodules if x.submodules else x for x in submodules
            ]
            submodules = tree.flatten(submodules)
        for layer in submodules:
            # Call the appropriate rust-code defined function
            if isinstance(layer, InputLayer):
                pass  # we don't need to do anything in this case
            elif isinstance(layer, Dense):
                weights = layer.get_weights()
                self.dnn.add_dense(weights[0].T, weights[1])
                if layer.activation == tf.nn.relu:
                    self.dnn.add_relu(len(weights[1]))
            elif isinstance(layer, Conv2D):
                weights = layer.get_weights()
                self.dnn.add_conv(weights[0], weights[1])
            elif isinstance(layer, MaxPooling2D):
                pool_size = layer.pool_size[0]
                assert layer.pool_size[0] == layer.pool_size[1]
                self.dnn.add_maxpool(pool_size)
            elif isinstance(layer, Flatten):
                self.dnn.add_flatten()
            else:
                print('Unknown layer', layer)
                raise NotImplementedError()

    def __str__(self):
        return str(self.dnn)
