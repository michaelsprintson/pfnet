import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import activations
import numpy as np

class BasicRNNCell(tf.compat.v1.nn.rnn_cell.RNNCell):

    def __init__(
        self,
        num_units,
        activation=None,
        reuse=None,
        name=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__(_reuse=reuse, name=name, dtype=dtype, **kwargs)

        self._num_units = num_units
        if activation:
            self._activation = activations.get(activation)
        else:
            self._activation = tf.tanh

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def build(self, inputs_shape):
        print("build called")
        if inputs_shape[-1] is None:
            raise ValueError(
                "Expected inputs.shape[-1] to be known, "
                f"received shape: {inputs_shape}"
            )

        input_depth = inputs_shape[-1]
        self._kernel = self.add_weight(
            "kernel",
            shape=[input_depth + self._num_units, self._num_units],
        )
        self._bias = self.add_weight(
            "bias",
            shape=[self._num_units],
            initializer=tf.compat.v1.zeros_initializer(dtype=self.dtype),
        )

        self.built = True

    def call(self, inputs, state):
        """Most basic RNN: output = new_state = act(W * input + U * state +
        B)."""

        gate_inputs = tf.matmul(tf.concat([inputs, state], 1), self._kernel)
        gate_inputs = tf.nn.bias_add(gate_inputs, self._bias)
        output = self._activation(gate_inputs)
        return output, output

    def get_config(self):
        config = {
            "num_units": self._num_units,
            "activation": activations.serialize(self._activation),
            "reuse": self._reuse,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
    

class PFRNNCell(tf.compat.v1.nn.rnn_cell.RNNCell):

    def __init__(self, params, batch_size, num_particles):
        super().__init__()
        self.batch_size = batch_size
        self.params = params
        self.num_particles = num_particles

        self.states_shape = (batch_size, num_particles, 3)
        self.weights_shape = (batch_size, num_particles, )

    @property
    def state_size(self):
        return (tf.TensorShape(self.states_shape[1:]), tf.TensorShape(self.weights_shape[1:])) #1: strips out batch

    @property
    def output_size(self):
        return (tf.TensorShape(self.states_shape[1:]), tf.TensorShape(self.weights_shape[1:]))
    
    def __call__(self, inputs, state, scope = None):

        particle_states, particle_weights = state
        observation, prev_window = inputs #can be additional inputs, passed in like odometry?

        lik = self.observation_model(particle_states, observation, prev_window) # additional inputs can be passed in here
        particle_weights += lik  # unnormalized
        
        particle_states, particle_weights = self.resample(
                particle_states, particle_weights, alpha=self.params['alpha_resample_ratio'])
        
        outputs = particle_states, particle_weights

        particle_states = self.transition_model(particle_states) # or here

        state = particle_states, particle_weights

        return outputs, state
    
    def transition_model(self, particle_states):
        pos_std = self.params['transition_std'][0] # position std
        vel_std = self.params['transition_std'][1]  # velocity std
        acc_std = self.params['transition_std'][1]  # acceleration std

        part_y, part_v, part_a = tf.unstack(particle_states, axis=-1, num=3)

        noise_y = tf.random.normal(part_y.get_shape(), mean=0.0, stddev=1.0) * pos_std
        noise_v = tf.random.normal(part_v.get_shape(), mean=0.0, stddev=1.0) * vel_std
        noise_a = tf.random.normal(part_a.get_shape(), mean=0.0, stddev=1.0) * acc_std

        delta_y = part_y + (part_v * self.params['time_step']) + (0.5 * part_a * self.params['time_step']**2) + noise_y
        delta_v = part_v + (part_a * self.params['time_step']) + noise_v
        delta_a = part_a + noise_a
        
        return tf.stack([delta_y, delta_v, delta_a], axis=-1)
    
    def observation_model(self, particle_states, observation, prev_window):

        # print("in obs model")

        p_flatten = tf.keras.layers.Flatten()(particle_states)
        x = tf.keras.layers.Concatenate()((p_flatten, observation, prev_window))
        x = tf.keras.layers.Dense(self.params['num_particles'], "relu", input_shape=x.shape)(x)
        x = tf.keras.layers.Dense(self.params['num_particles'] * 2, "relu", input_shape=x.shape)(x)
        x = tf.keras.layers.Dense(self.params['num_particles'], "relu", input_shape=x.shape)(x)
        return x
        #plan
        # add previous window num points, 0 padded if dont exist
        # append particle states with obsv with previous window points
        # locallyconnected and dense and relu it 
        
        

    @staticmethod
    def resample(particle_states, particle_weights, alpha):
        """
        Implements (soft)-resampling of particles.
        :param particle_states: tf op (batch, K, 3), particle states
        :param particle_weights: tf op (batch, K), unnormalized particle weights in log space
        :param alpha: float, trade-off parameter for soft-resampling. alpha == 1 corresponds to standard,
        hard-resampling. alpha == 0 corresponds to sampling particles uniformly, ignoring their weights.
        :return: particle_states, particle_weights
        """
        
        assert 0.0 < alpha <= 1.0
        batch_size, num_particles = particle_states.get_shape().as_list()[:2]

        # normalize
        particle_weights = particle_weights - tf.math.reduce_logsumexp(particle_weights, axis=-1, keepdims=True)

        uniform_weights = tf.constant(-np.log(num_particles), shape=(batch_size, num_particles), dtype=tf.float32)

        # build sampling distribution, q(s), and update particle weights
        if alpha < 1.0:
            # soft resampling
            q_weights = tf.stack([particle_weights + np.log(alpha), uniform_weights + np.log(1.0-alpha)], axis=-1)
            q_weights = tf.math.reduce_logsumexp(q_weights, axis=-1, keepdims=False)
            q_weights = q_weights - tf.math.reduce_logsumexp(q_weights, axis=-1, keepdims=True)  # normalized

            particle_weights = particle_weights - q_weights  # this is unnormalized
        else:
            # hard resampling. this will produce zero gradients
            q_weights = particle_weights
            particle_weights = uniform_weights

        # sample particle indices according to q(s)
        # indices = tf.cast(tf.compat.v1.multinomial(q_weights, num_particles), tf.int32)  # shape: (batch_size, num_particles)
        indices = tf.cast(tfp.distributions.Multinomial(100, logits=q_weights).sample(()), tf.int32)  # shape: (batch_size, num_particles)
        # print(indices.shape)

        # index into particles
        helper = tf.range(0, batch_size*num_particles, delta=num_particles, dtype=tf.int32)  # (batch, )
        indices = indices + tf.expand_dims(helper, axis=1)

        particle_states = tf.reshape(particle_states, (batch_size * num_particles, 3))
        particle_states = tf.gather(particle_states, indices=indices, axis=0)  # (batch_size, num_particles, 3)

        particle_weights = tf.reshape(particle_weights, (batch_size * num_particles, ))
        particle_weights = tf.gather(particle_weights, indices=indices, axis=0)  # (batch_size, num_particles,)

        return particle_states, particle_weights
    
class PFNET(object):
    def __init__(self, params, input_shapes, labels_shapes, is_training = True):
        self.params = params
        self.hidden_states = []

        self.train_loss_op = None
        self.valid_loss_op = None

        self.global_step_op = None
        self.learning_rate_op = None
        self.train_op = None
        self.update_state_op = tf.constant(0)

        self.outputs = self.build_rnn(input_shapes)

        p_states, p_weights = self.outputs
        self.build_loss_op(p_states, p_weights, true_state_shape = labels_shapes)

        if is_training:
            self.build_train_op()
    
    def build_rnn(self, input_shapes):
        init_particle_state_shape, observation_shape, prev_window_shape = input_shapes
        batch_size = observation_shape[0]
        
        obs_in = tf.keras.Input(dtype = tf.float32, shape = observation_shape[1:], batch_size= observation_shape[0], name = "X")
        init_particle_states = tf.keras.Input(dtype = tf.float32, shape = init_particle_state_shape[1:], batch_size= init_particle_state_shape[0], name = "initial_state")
        init_particle_weights = tf.keras.Input(dtype = tf.float32, shape = init_particle_state_shape[1:-1], batch_size= init_particle_state_shape[0], name = "initial_weight")
        prev_window = tf.keras.Input(dtype = tf.float32, shape = prev_window_shape[1:], batch_size= prev_window_shape[0], name = "prev_window")
        self.inputs = [obs_in, init_particle_states, init_particle_weights, prev_window]
        
        num_particles = init_particle_states.shape.as_list()[1]

        # init_particle_weights = tf.constant(np.log(1.0/float(num_particles)),
        #                                     shape=(batch_size, num_particles), dtype=tf.float32)
        
        self.hidden_states = [ #potential issue with hardcoded tf float32
            tf.Variable(tf.constant_initializer(0)(shape=init_particle_states.get_shape(), dtype=tf.float32), trainable = False, name = "particle_states"),
            tf.Variable(tf.constant_initializer(np.log(1.0/float(num_particles)))(shape=(batch_size, num_particles), dtype=tf.float32), trainable = False, name = "particle_weights"),
            ]

        state = (init_particle_states, init_particle_weights)

        # print(state[0].shape)
        # print("yo")
        # print(state[1].shape)
        
        cell_func = PFRNNCell(params=self.params, batch_size=batch_size,
                                num_particles=num_particles)
        
        rnn = tf.keras.layers.RNN(cell = cell_func, time_major=False, return_sequences=True, return_state=True)
        (particle_states, particle_weights), states, weights = rnn(inputs = (obs_in,prev_window), initial_state = state)
        state = [states,weights]

        with tf.control_dependencies([particle_states, particle_weights]):
            self.update_state_op = tf.group(
                *(self.hidden_states[i].assign(state[i]) for i in range(len(self.hidden_states))))
            
        return particle_states, particle_weights
    
    def build_loss_op(self, p_states, p_weights, true_state_shape):
        lin_weights = tf.nn.softmax(p_weights, axis=-1)

        true_pos = tf.keras.Input(dtype = tf.float32, shape = true_state_shape[1:], batch_size= true_state_shape[0], name = "y")
        self.inputs.append(true_pos)

        mean_pos = tf.reduce_sum(tf.multiply(p_states[:,:,:,:1], lin_weights[:,:,:,None]), axis = 2)
        pos_diffs = true_pos - mean_pos
        
        loss_pred = tf.reduce_mean(tf.square(pos_diffs), name='prediction_loss')

        loss_reg = tf.multiply(tf.compat.v1.losses.get_regularization_loss(), self.params['l2scale'], name='l2')
        loss_total = tf.add_n([loss_pred, loss_reg], name="training_loss")

        self.valid_loss_op = loss_pred
        self.train_loss_op = loss_total

        return loss_total
    
    def build_train_op(self):
        self.global_step_op = tf.Variable(initial_value = 0.0, shape = (), trainable=False, name = "global_step")
        self.learning_rate_op = tf.compat.v1.train.exponential_decay(
            self.params['learningrate'], self.global_step_op, decay_steps=1, decay_rate=self.params['decayrate'],
            staircase=True, name="learning_rate")
        optimizer = tf.compat.v1.train.RMSPropOptimizer(self.learning_rate_op, decay=0.9)
        # optimizer = tf.keras.optimizers.RMSprop(self.learning_rate_op, rho=0.9) #can't do this because of the tape thing

        with tf.compat.v1.control_dependencies(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)): #dont need this
            self.train_op = optimizer.minimize(self.train_loss_op, global_step=None, var_list=tf.compat.v1.trainable_variables())
            # self.train_op = optimizer.minimize(loss = self.train_loss_op, var_list=tf.compat.v1.trainable_variables(), tape=tf.GradientTape())

        return self.train_op