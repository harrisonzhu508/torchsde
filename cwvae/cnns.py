import tensorflow.compat.v1 as tf
from tensorflow.keras import layers as tfkl
import numpy as np

import tools


class Encoder(tools.Module):
    """
    Multi-level Video Encoder.
    1. Extracts hierarchical features from a sequence of observations.
    2. Encodes observations using Conv layers, uses them directly for the bottom-most level.
    3. Uses dense features for each level of the hierarchy above the bottom-most level.
    """

    def __init__(
        self,
        levels,
        tmp_abs_factor,
        dense_layers=3,
        embed_size=100,
        channels_mult=1,
        var_scope="encoder_convdense",
    ):
        """
        Arguments:
            obs : Tensor
                Flattened/Non-flattened observations of shape (batch size, timesteps, [dim(s)])
            levels : int
                Number of levels in the hierarchy
            tmp_abs_factor : int
                Temporal abstraction factor used at each level
            dense_layers : int
                Number of dense hidden layers at each level
            embed_size : int
                Size of dense hidden embeddings
            channels_mult: int
                Multiplier for the number of channels in the conv encoder
        """
        super().__init__()
        self._levels = levels
        self._tmp_abs_factor = tmp_abs_factor
        self._dense_layers = dense_layers
        self._embed_size = embed_size
        self._channels_mult = channels_mult
        self._activation = tf.nn.leaky_relu
        self._kwargs = dict(strides=2, activation=self._activation, use_bias=True)
        self._var_scope = var_scope

        assert levels >= 1, "levels should be >=1, found {}".format(levels)
        assert tmp_abs_factor >= 1, "tmp_abs_factor should be >=1, found {}".format(
            tmp_abs_factor
        )
        assert (
            not dense_layers or embed_size
        ), "embed_size={} invalid for Dense layer".format(embed_size)

    def __call__(self, obs):
        """
        Arguments:
            obs : Tensor
                Un-flattened observations (videos) of shape (batch size, timesteps, [dim1, dim2, dim3])
        """
        with tf.name_scope(self._var_scope):
            # Squeezing batch and time dimensions.
            # only 1 Conv extractor needed in order to do dimensionality reduction
            # the rest is handled by dense
            # merge batchxT -> batch*T
            hidden = tf.reshape(obs, tf.concat([[-1], tf.shape(obs)[2:]], -1))

            filters = 32
            hidden = self.get(
                "h1_conv", tfkl.Conv2D, self._channels_mult * filters, 4, **self._kwargs
            )(hidden)
            hidden = self.get(
                "h2_conv",
                tfkl.Conv2D,
                self._channels_mult * filters * 2,
                4,
                **self._kwargs
            )(hidden)
            hidden = self.get(
                "h3_conv",
                tfkl.Conv2D,
                self._channels_mult * filters * 4,
                4,
                **self._kwargs
            )(hidden)
            hidden = self.get(
                "h4_conv",
                tfkl.Conv2D,
                self._channels_mult * filters * 8,
                4,
                **self._kwargs
            )(hidden)
            # No dense here. Just flatten conv2d outputs
            hidden = tf.layers.flatten(hidden)  # shape: (BxT, :)
            hidden = tf.reshape(
                hidden, tf.concat([tf.shape(obs)[:2], [hidden.shape.as_list()[-1]]], -1)
            )  # shape: (B, T, :)
            layer = hidden

            layers = list([])
            layers.append(layer)
            print("Input shape at level {}: {}".format(0, layer.shape))

            feat_size = layer.shape[-1]

            # now move the conved outputs up the hierarchy
            for level in range(1, self._levels):
                for i_dl in range(self._dense_layers - 1):
                    # create dense + relu layer and apply to conved stuff
                    hidden = self.get(
                        "h{}_dense".format(5 + (level - 1) * self._dense_layers + i_dl),
                        tfkl.Dense,
                        self._embed_size,
                        activation=tf.nn.relu,
                    )(hidden)
                if self._dense_layers > 0:
                    hidden = self.get(
                        "h{}_dense".format(4 + level * self._dense_layers),
                        tfkl.Dense,
                        feat_size,
                        activation=None,
                    )(hidden)
                layer = hidden

                # now we have to remove some encoded outputs
                # based on the resolution of the slow variables
                # this whole thing:
                # k^l (l = l-1 in python convention)
                # k^l - (T mod k^l) mod k^l
                # T mod k^l to see if T is divisible by k^l
                # e.g. T=100, k=2, l=2-> T mod 4 = 0
                # [d,...,d....,Tth value,...[in-between regions to be padded with 0]]
                timesteps_to_merge = np.power(self._tmp_abs_factor, level)
                # Padding the time dimension.
                timesteps_to_pad = tf.mod(
                    timesteps_to_merge - tf.mod(tf.shape(layer)[1], timesteps_to_merge),
                    timesteps_to_merge,
                )
                paddings = tf.convert_to_tensor([[0, 0], [0, timesteps_to_pad], [0, 0]])
                # paddings: shape [rank, 2]
                # since layer: [b, T, hidden_dim] -> only T-padding is needed
                # [how many to add before tensor, how many to add after tensor]
                # [0, timesteps_to_pad] means we add timesteps_to_pad
                # now becomes dimension [b, T+extra, hidden_dim]
                layer = tf.pad(layer, paddings, mode="CONSTANT", constant_values=0)
                # Reshaping and merging in time.
                # now becomes dimension [b, T+extra, hidden_dim]
                layer = tf.reshape(
                    layer,
                    [
                        tf.shape(layer)[0], # batch
                        -1, # 
                        timesteps_to_merge, # extra bits
                        layer.shape.as_list()[2], # 
                    ],
                )
                # reduce sum gets rid of the extra states
                # basically sum over e_t,...,t_{t+k^l-1}
                # the extra ends are 0, so doesn't contribute! Only contributions
                # up to T matter
                layer = tf.reduce_sum(layer, axis=2)
                layers.append(layer)
                print("Input shape at level {}: {}".format(level, layer.shape))

        return layers


class Decoder(tools.Module):
    """ States to Images Decoder. """

    def __init__(self, out_channels, channels_mult=1, var_scope="decoder_conv"):
        """
        Arguments:
            out_channels : int
                Number of channels in the output video
            channels_mult : int
                Multiplier for the number of channels in the conv encoder
        """
        super().__init__()
        self._out_channels = out_channels
        self._channels_mult = channels_mult
        self._out_activation = tf.nn.tanh
        self._kwargs = dict(strides=2, activation=tf.nn.leaky_relu, use_bias=True)
        self._var_scope = var_scope

    def __call__(self, states):
        """
        Arguments:
            states : Tensor
                State tensor of shape (batch_size, timesteps, feature_dim)

        Returns:
            out : Tensor
                Output video of shape (batch_size, timesteps, 64, 64, out_channels)
        """
        with tf.name_scope(self._var_scope):
            hidden = self.get("h1", tfkl.Dense, self._channels_mult * 1024, None)(
                states
            )  # (B, T, 1024)

            # Squeezing batch and time dimensions, and expanding two extra dims.
            hidden = tf.reshape(
                hidden, [-1, 1, 1, hidden.shape[-1].value]
            )  # (BxT, 1, 1, 1024)

            filters = 32
            hidden = self.get(
                "h2",
                tfkl.Conv2DTranspose,
                self._channels_mult * filters * 4,
                5,
                **self._kwargs
            )(
                hidden
            )  # (BxT, 5, 5, 128)
            hidden = self.get(
                "h3",
                tfkl.Conv2DTranspose,
                self._channels_mult * filters * 2,
                5,
                **self._kwargs
            )(
                hidden
            )  # (BxT, 13, 13, 64)
            hidden = self.get(
                "h4",
                tfkl.Conv2DTranspose,
                self._channels_mult * filters,
                6,
                **self._kwargs
            )(
                hidden
            )  # (BxT, 30, 30, 32)
            out = self.get(
                "out",
                tfkl.Conv2DTranspose,
                self._out_channels,
                6,
                strides=2,
                activation=self._out_activation,
            )(
                hidden
            )  # (BxT, 64, 64, out_channels)
            out = tf.reshape(
                out, tf.concat([tf.shape(states)[:2], tf.shape(out)[1:]], -1)
            )  # (B, T, 64, 64, out_channels)
        return out
