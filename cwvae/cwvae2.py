"""Rewrite of cwvae by https://github.com/vaibhavsaxena11/cwvae
"""

import tensorflow as tf
from tensorflow_probability import distributions as tfd
import tensorflow.compat.v1 as tf_v1
import numpy as np
import tools

class Encoder3D(tf.keras.Model):
    """
    Multi-level data Encoder
    1. Extracts hierarchical features from a sequence of observations.
    2. Encodes observations using Conv layers, uses them directly for the bottom-most level.
    3. Uses dense features for each level of the hierarchy above the bottom-most level.
    """

    def __init__(
        self,
        levels,
        tmp_abs_factor,
        layers,
    ):
        """
        Arguments:
            obs : Tensor
                Flattened/Non-flattened observations of shape (batch size, timesteps, [dim(s)])
            levels : int
                Number of levels in the hierarchy
            tmp_abs_factor : int
                Temporal abstraction factor used at each level
        """
        super().__init__()
        self._levels = levels
        self._tmp_abs_factor = tmp_abs_factor

        self.layers_custom = layers
        assert levels >= 1, "levels should be >=1, found {}".format(levels)
        assert tmp_abs_factor >= 1, "tmp_abs_factor should be >=1, found {}".format(
            tmp_abs_factor
        )

    def __call__(self, obs):
        """
        Arguments:
            obs : Tensor
                Un-flattened observations (videos) of shape (batch size, timesteps, D_x])
        """
        # Squeezing batch and time dimensions.
        # merge batchxT -> batch*T
        data_shape = tf.shape(obs)
        outputs = []
        obs = tf.reshape(obs, tf.concat([[-1], data_shape[2:]], -1))
        hidden = self.layers_custom[0](obs)

        # shape: (B, T, :)
        outputs.append(tf.reshape(hidden, (data_shape[0], data_shape[1], -1)))

        # now move the conved outputs up the hierarchy
        for level in range(1, self._levels):
            hidden = self.layers_custom[level](hidden)
            hidden = tf.reshape(hidden, (data_shape[0], data_shape[1], -1))
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
            timesteps_to_pad = tf_v1.mod(
                timesteps_to_merge - tf_v1.mod(tf.shape(hidden)[1], timesteps_to_merge),
                timesteps_to_merge,
            )
            paddings = tf.convert_to_tensor([[0, 0], [0, timesteps_to_pad], [0, 0]])
            # paddings: shape [rank, 2]
            # since layer: [b, T, hidden_dim] -> only T-padding is needed
            # [how many to add before tensor, how many to add after tensor]
            # [0, timesteps_to_pad] means we add timesteps_to_pad
            # now becomes dimension [b, T+extra, hidden_dim]
            output = tf.pad(hidden, paddings, mode="CONSTANT", constant_values=0)
            # Reshaping and merging in time.
            # now becomes dimension [b, T+extra, hidden_dim]
            # now becomes dimension [b, (T+extra) / k^l, k^l, hidden_dim]
            output = tf.reshape(
                output,
                [
                    tf.shape(output)[0], # batch
                    -1, # 
                    timesteps_to_merge, # extra bits
                    tf.shape(output)[-1], # 
                ],
            )
            # reduce sum gets rid of the extra states
            # basically sum over e_t,...,t_{t+k^l-1}
            # the extra ends are 0, so doesn't contribute! Only contributions
            # up to T matter
            output = tf.reduce_sum(output, axis=2)
            outputs.append(output)

        return outputs


class RSSMCell(tf.keras.Model):
    def __init__(
        self,
        state_size,
        detstate_size,
        embed_size,
        reset_states=False,
        min_stddev=0.0001,
        mean_only=False,
    ):
        super().__init__()
        self._state_size = state_size
        self._detstate_size = detstate_size
        self._embed_size = embed_size
        self._min_stddev = min_stddev
        self._mean_only = mean_only
        self._reset_states = reset_states  # whether or not to reset states as per the reset_state tensor passed to __call__() (VTA-like behavior)

        self._cell = tf.keras.layers.GRUCell(units=self._detstate_size)
        self.prior_net1 = tf.keras.layers.Dense(self._embed_size, dtype=tf.float32, activation="relu")
        self.prior_net2 = tf.keras.layers.Dense(self._embed_size, dtype=tf.float32, activation="relu")
        self.det_to_priormu = tf.keras.layers.Dense(self._state_size, dtype=tf.float32)
        self.det_to_priorstd = tf.keras.layers.Dense(self._state_size, dtype=tf.float32, activation="softplus")

        self.posterior_net1 = tf.keras.layers.Dense(self._embed_size, dtype=tf.float32, activation="relu")
        self.posterior_net2 = tf.keras.layers.Dense(self._embed_size, dtype=tf.float32, activation="relu")
        self.det_to_posteriormu = tf.keras.layers.Dense(self._state_size, dtype=tf.float32)
        self.det_to_posteriorstd = tf.keras.layers.Dense(self._state_size, dtype=tf.float32, activation="softplus")

    def _prior(self, prev_state, context):
        inputs = tf.concat([prev_state["sample"], context], -1)
        hl = self.prior_net1(inputs)
        det_out, det_state = self._cell(hl, (prev_state["det_state"],))
        det_state = det_state[0]
        hl = det_out
        hl = self.prior_net2(hl)
        mean = self.det_to_priormu(hl)
        stddev = self.det_to_priorstd(hl + 0.54)
        stddev += self._min_stddev
        if self._mean_only:
            sample = mean
        else:
            sample = tfd.MultivariateNormalDiag(mean, stddev).sample()
        return {
            "mean": mean,
            "stddev": stddev,
            "sample": sample,
            "det_out": det_out,
            "det_state": det_state,
            "output": tf.concat([sample, det_out], -1),
        }

    def _posterior(self, obs_inputs, prev_state, context):
        """
        obs_inputs: observations
        prev_state: temporal context
        context: top-down context

        For the top layer, context=0 i.e. no top-down context
        """
        # obs_inputs = time series data
        prior = self._prior(prev_state, context)
        inputs = tf.concat([prior["det_out"], obs_inputs], -1)
        hl = self.posterior_net1(inputs)
        det_out, det_state = self._cell(hl, (prev_state["det_state"],))
        det_state = det_state[0]
        hl = det_out
        hl = self.posterior_net2(hl)
        mean = self.det_to_posteriormu(hl)
        stddev = self.det_to_posteriorstd(hl + 0.54)
        stddev += self._min_stddev

        if self._mean_only:
            sample = mean
        else:
            sample = tfd.MultivariateNormalDiag(mean, stddev).sample()
        return {
            "mean": mean,
            "stddev": stddev,
            "sample": sample,
            "det_out": prior["det_out"],
            "det_state": prior["det_state"],
            "output": tf.concat([sample, prior["det_out"]], -1),
        }

    @property
    def state_size(self):
        return {
            "mean": self._state_size,
            "stddev": self._state_size,
            "sample": self._state_size,
            "det_out": self._detstate_size,
            "det_state": self._detstate_size,
            "output": self._state_size + self._detstate_size,
        }

    @property
    def out_state_size(self):
        return {"out": (self.state_size, self.state_size), "state": self.state_size}

    def zero_state(self, batch_size, dtype=tf.float32):
        return dict(
            [
                (k, tf.zeros([batch_size, v], dtype=dtype))
                for k, v in self.state_size.items()
            ]
        )

    def zero_out_state(self, batch_size, dtype=tf.float32):
        zero_st = self.zero_state(batch_size, dtype)
        return {"out": (zero_st, zero_st), "state": zero_st}

    def __call__(self, prev_out, inputs, use_obs):
        """
        Arguments:
            prev_out : dict
                output of this __call__ at the previous time-step during unroll.
            inputs : dict
                dict of context and other inputs (including observations).
                obs_input will remain unused during test phase when the posterior is not computed.
            use_obs : bool
        Returns:
            dict
                'out': (prior, posterior) --> cell out
                'state': (posterior) --> cell state
        """
        prev_state = prev_out["state"]
        obs_input, context, reset_state = inputs
        if not self._reset_states:
            reset_state = tf.ones_like(reset_state)
        prev_state["sample"] = tf.multiply(prev_state["sample"], reset_state)

        prior = self._prior(prev_state, context)
        if use_obs:
            posterior = self._posterior(obs_input, prev_state, context)
        else:
            posterior = prior

        return {"out": (prior, posterior), "state": posterior}



class CWVAE2(tf.keras.Model):
    def __init__(
        self,
        levels,
        tmp_abs_factor,
        state_sizes,
        embed_size,
        cell_type,
        lr,
        min_stddev,
        encoder,
        decoder,
        mean_only_cell=False,
        reset_states=False,
    ):  
        super().__init__()
        self.cell_type = cell_type
        self._levels = levels
        self._state_size = state_sizes["stoch"]
        self._detstate_size = state_sizes["deter"]
        self._embed_size = embed_size
        self.lr = lr
        self._min_stddev = min_stddev
        self._tmp_abs_factor = tmp_abs_factor
        self._reset_states = reset_states
        self.encoder = encoder
        self.decoder = decoder

        self.cells = []
        for _ in range(self._levels):
            if self.cell_type == "RSSMCell":
                assert (
                    self._detstate_size
                ), "deter state size should be non-zero int, found {}".format(
                    self._detstate_size
                )

                # The RNN cell for each layer
                # each layer outputs the prior and posterior
                # mean and standard deviations
                # how does it handle x_{t:t+k-1}?
                cell = RSSMCell(
                    self._state_size, # use same state size 
                    self._detstate_size,
                    self._embed_size,
                    reset_states=self._reset_states,
                    min_stddev=self._min_stddev,
                    mean_only=mean_only_cell,
                )
            else:
                raise ValueError("Cell type {} not supported".format(self.cell_type))
            self.cells.append(cell)

    def hierarchical_unroll(
        self, inputs, actions=None, use_observations=None, initial_state=None
    ):
        """
        Used to unroll a list of recurrent cells.

        Outputs list of priors and posteriors

        Arguments:
            cells
            inputs : list of encoded observations. list of [b, number_of_states, hidden_dim]
                Number of nodes at every level in 'inputs' is the exact number of nodes to be unrolled
            actions
            use_observations : bool or list[bool]
            initial_state : list of cell states
        """
        if use_observations is None:
            use_observations = self._levels * [True]
        elif not isinstance(use_observations, list):
            use_observations = self._levels * [use_observations]

        if initial_state is None:
            initial_state = self._levels * [None]

        level_top = self._levels - 1
        inputs_top = inputs[level_top]

        level = level_top

        # Feeding in zeros as context to the top level.
        # because there are no top-down contexts for the top layer
        context = tf.zeros(
            shape=tf.concat(
                [tf.shape(inputs_top)[0:2], [self.cells[-1].state_size["output"]]], -1
            )
        )

        # Init reset_state: alternate zeros and ones, of the size of input[level=-2]
        if level_top >= 1:
            inputs_top_ = inputs[level_top - 1]
            temp_zeros = tf.zeros(
                tf.concat([tf.shape(inputs_top_)[:2], [tf.constant(1)]], -1)
            )
            temp_ones = tf.ones_like(temp_zeros)
            _reset_state = tf.concat([temp_zeros, temp_ones], -1)
            _reset_state = tf.reshape(_reset_state, [tf.shape(_reset_state)[0], -1, 1])
            _reset_state = tf.slice(
                _reset_state,
                [0, 0, 0],
                tf.concat([tf.shape(inputs_top_)[:2], [tf.constant(1)]], -1),
            )
        else:
            _reset_state = tf.no_op()

        # bottom level to top level 0...L-1
        prior_list = []  # Stored bot to top.
        posterior_list = []  # Stored bot to top.

        last_state_all_levels = list([])

        for level in range(level_top, -1, -1):
            # obs_inputs: b x C x H x W
            obs_inputs = inputs[level]
            # print("Input shape in CWVAE level {}: {}".format(level, obs_inputs.shape))
            if level == level_top:
                reset_state, reset_state_next = (
                    tf.ones(shape=tf.concat([tf.shape(obs_inputs)[0:2], [1]], -1)),
                    _reset_state,
                )
            else:
                reset_state, reset_state_next = (
                    reset_state,
                    tf.tile(reset_state, [1, self._tmp_abs_factor, 1]),
                )

            # Pruning reset_state, context from previous layer, to match the num of nodes reqd as in inputs[level]
            reset_state = tf.slice(
                reset_state,
                len(context.shape.as_list()) * [0],
                tf.concat([tf.shape(obs_inputs)[:2], [1]], -1),
            )
            # slice the data so that the resolution matches the required
            # resolution of the level
            context = tf.slice(
                context,
                len(context.shape.as_list()) * [0],
                tf.concat([tf.shape(obs_inputs)[:2], context.shape.as_list()[2:]], -1),
            )

            # Concatenating actions (if provided) to the context at the bottom-most level.
            if level == 0 and actions is not None:
                context = tf.concat([context, actions], axis=-1)

            # Dynamic unroll of RNN cell.
            initial = self.cells[level].zero_out_state(tf.shape(obs_inputs)[0])
            if initial_state[level] is not None:
                initial["state"] = initial_state[level]

            # loop and output prior and posterior
            (prior, posterior), posterior_last_step = tools.scan(
                self.cells[level],
                (obs_inputs, context, reset_state),
                use_observations[level],
                initial,
            )

            last_state_all_levels.insert(0, posterior_last_step)
            context = posterior["output"]

            prior_list.insert(0, prior)
            posterior_list.insert(0, posterior)

            # Tiling context by a factor of tmp_abs_factor for use at the level below.
            if level != 0:
                context = tf.expand_dims(context, axis=2)
                # repeat so that we pass on the data to the higher
                # resolution level
                context = tf.tile(
                    context,
                    [1, 1, self._tmp_abs_factor]
                    + (len(context.shape.as_list()) - 3) * [1],
                )
                context = tf.reshape(
                    context,
                    [tf.shape(context)[0], tf.reduce_prod(tf.shape(context)[1:3])]
                    + context.shape.as_list()[3:],
                )

            reset_state = reset_state_next
        output_bot_level = context

        return output_bot_level, last_state_all_levels, prior_list, posterior_list

    def open_loop_unroll(self, inputs, ctx_len, actions=None, use_observations=None):
        if use_observations is None:
            use_observations = self._levels * [True]
        ctx_len_backup = ctx_len
        pre_inputs = []
        post_inputs = []
        for lvl in range(self._levels):
            pre_inputs.append(inputs[lvl][:, :ctx_len, ...])
            post_inputs.append(tf.zeros_like(inputs[lvl][:, ctx_len:, ...]))
            ctx_len = int(ctx_len / self._tmp_abs_factor)
        ctx_len = ctx_len_backup
        actions_pre = actions_post = None
        if actions is not None:
            actions_pre = actions[:, :ctx_len, :]
            actions_post = actions[:, ctx_len:, :]

        (
            _,
            pre_last_state_all_levels,
            pre_priors,
            pre_posteriors,
        ) = self.hierarchical_unroll(
            pre_inputs, actions=actions_pre, use_observations=use_observations
        )
        _, _, post_priors, _ = self.hierarchical_unroll(
            post_inputs,
            actions=actions_post,
            use_observations=self._levels * [False],
            initial_state=pre_last_state_all_levels,
        )

        return pre_posteriors, pre_priors, post_priors

    def _gaussian_KLD(self, dist1, dist2):
        """
        Computes KL(dist1 || dist2)

        Arguments:
            dist1 : dict containing 'mean' and 'stddev' for multivariate normal distributions
                shape of mean/stddev: (batch size, timesteps, [dim1, dim2, ...])
            dist2 : (same as dist1)
        """
        if len(dist1["mean"].shape[2:]) > 1:
            new_shape = tf.concat([tf.shape(dist1["mean"])[:2], [-1]], -1)
            dist1["mean"] = tf.reshape(dist1["mean"], new_shape)
            dist1["stddev"] = tf.reshape(dist1["stddev"], new_shape)
            dist2["mean"] = tf.reshape(dist2["mean"], new_shape)
            dist2["stddev"] = tf.reshape(dist2["stddev"], new_shape)
        mvn1 = tfd.MultivariateNormalDiag(loc=dist1["mean"], scale_diag=dist1["stddev"])
        mvn2 = tfd.MultivariateNormalDiag(loc=dist2["mean"], scale_diag=dist2["stddev"])
        return mvn1.kl_divergence(mvn2)

    def _log_prob_obs(self, samples, mean, stddev):
        """
        Returns prob density of samples in the given distribution
        The last dim of the samples is the one taken sum over.
        """
        if len(samples.shape[2:]) > 1:
            new_shape = tf.concat([tf.shape(samples)[:2], [-1]], -1)
            samples = tf.reshape(samples, new_shape)
            mean = tf.reshape(mean, new_shape)
            if isinstance(stddev, tf.Tensor):
                stddev = tf.reshape(stddev, new_shape)
        dist = tfd.Independent(tfd.Normal(mean, stddev), reinterpreted_batch_ndims=1)
        return dist.log_prob(samples)

    def _stop_grad_dist(self, dist):
        dist["mean"] = tf.stop_gradient(dist["mean"])
        dist["stddev"] = tf.stop_gradient(dist["stddev"])
        return dist

    def compute_losses(
        self,
        obs,
        obs_decoded,
        priors,
        posteriors,
        dec_stddev=0.1,
        kl_grad_post_perc=None,
        free_nats=None,
        beta=None,
    ):
        """
        Computes ELBO.

        Arguments:
            obs : Placeholder
                Observed video
            obs_decoded : Tensor
                Decoded video
            priors : list[dict]
                each dict holds the priors at all timesteps for a particular level in the model
            posteriors : list[dict]
                each dict holds the posteriors at all timesteps for a particular level in the model
        """
        nll_term = -self._log_prob_obs(obs, obs_decoded, dec_stddev)  # shape: (B,T)
        nll_term = tf.reduce_mean(tf.reduce_sum(nll_term, axis=1), 0)
        assert len(nll_term.shape) == 0, nll_term.shape

        # Computing KLs between priors and posteriors
        self.kld_all_levels = list([])
        kl_term = tf.constant(0.0)
        for i in range(self._levels):
            kld_level = self._gaussian_KLD(posteriors[i], priors[i])
            if kl_grad_post_perc is None:
                kld_level_total = tf.reduce_mean(tf.reduce_sum(kld_level, axis=1))
            else:
                # Scaling gradient between prior and posterior.
                kld_level_p = (1 - kl_grad_post_perc) * self._gaussian_KLD(
                    self._stop_grad_dist(posteriors[i]), priors[i]
                )  # shape: (B,T)
                kld_level_q = kl_grad_post_perc * self._gaussian_KLD(
                    posteriors[i], self._stop_grad_dist(priors[i])
                )  # shape: (B,T)
                kld_level_total_p = tf.reduce_mean(tf.reduce_sum(kld_level_p, axis=1))
                kld_level_total_q = tf.reduce_mean(tf.reduce_sum(kld_level_q, axis=1))
                # Giving 1 free nat to the posterior.
                kld_level_total_q = tf.maximum(1.0, kld_level_total_q)
                kld_level_total = kld_level_total_p + kld_level_total_q

            if free_nats is None:
                if beta is None:
                    kl_term += kld_level_total
                else:
                    if isinstance(beta, list):
                        kl_term += beta[i] * kld_level_total
                    else:
                        kl_term += beta * kld_level_total
            else:
                if beta is None:
                    kl_term += tf.maximum(0.0, kld_level_total - free_nats)
                else:
                    if isinstance(beta, list):
                        kl_term += beta[i] * tf.maximum(
                            0.0, kld_level_total - free_nats
                        )
                    else:
                        kl_term += beta * tf.maximum(0.0, kld_level_total - free_nats)
            self.kld_all_levels.insert(i, kld_level)

        neg_elbo = nll_term + kl_term

        num_timesteps_obs = tf.cast(tf.shape(obs)[1], tf.float32)
        self.loss = neg_elbo / num_timesteps_obs
        self._kl_term = kl_term / num_timesteps_obs
        self._nll_term = nll_term / num_timesteps_obs

        return self.loss