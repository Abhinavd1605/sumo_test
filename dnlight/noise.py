"""
NoisyNet Layer for DNLight.

Implements factorized Gaussian noise for structured exploration,
replacing standard epsilon-greedy with learned noise parameters.
Compatible with TensorFlow 2.x / Keras.
"""
import tensorflow as tf
import numpy as np


class NoisyLinear(tf.keras.layers.Layer):
    """
    Noisy Linear layer with factorized Gaussian noise.

    y = (mu_w + sigma_w * eps_w) * x + (mu_b + sigma_b * eps_b)

    The noise is factorized: eps_w = f(eps_i) * f(eps_j)^T
    where f(x) = sign(x) * sqrt(|x|)
    """

    def __init__(self, units, noisy_std=0.1, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.noisy_std = noisy_std

    def build(self, input_shape):
        in_features = int(input_shape[-1])
        self.in_features = in_features

        # Mu (deterministic) weights - Kaiming uniform init
        limit = 1.0 / np.sqrt(in_features)
        self.mu_w = self.add_weight(
            name="mu_w",
            shape=(in_features, self.units),
            initializer=tf.keras.initializers.RandomUniform(-limit, limit),
            trainable=True,
        )
        self.mu_b = self.add_weight(
            name="mu_b",
            shape=(self.units,),
            initializer=tf.keras.initializers.RandomUniform(-limit, limit),
            trainable=True,
        )

        # Sigma (noise scale) weights
        sigma_init = self.noisy_std / np.sqrt(in_features)
        self.sigma_w = self.add_weight(
            name="sigma_w",
            shape=(in_features, self.units),
            initializer=tf.keras.initializers.Constant(sigma_init),
            trainable=True,
        )
        self.sigma_b = self.add_weight(
            name="sigma_b",
            shape=(self.units,),
            initializer=tf.keras.initializers.Constant(sigma_init),
            trainable=True,
        )

        # Noise tensors (not trainable)
        self.eps_w = tf.Variable(
            tf.zeros((in_features, self.units)),
            trainable=False, name="eps_w"
        )
        self.eps_b = tf.Variable(
            tf.zeros((self.units,)),
            trainable=False, name="eps_b"
        )

        # Global noise scale (decayed externally)
        self.noise_scale = tf.Variable(
            1.0, trainable=False, name="noise_scale"
        )

        self.reset_noise()
        super().build(input_shape)

    @staticmethod
    def _factorized_noise(size):
        """Generate factorized noise: f(x) = sign(x) * sqrt(|x|)."""
        x = tf.random.normal((size,))
        return tf.sign(x) * tf.sqrt(tf.abs(x))

    def reset_noise(self):
        """Regenerate factorized noise vectors."""
        eps_i = self._factorized_noise(self.in_features)
        eps_j = self._factorized_noise(self.units)

        # Outer product for weight noise
        self.eps_w.assign(tf.tensordot(eps_i, eps_j, axes=0))
        self.eps_b.assign(eps_j)

    def call(self, inputs, training=None):
        if training:
            # Local per-batch scaling: Uniform(0.5, 1.5)
            local_scale = tf.random.uniform((), 0.5, 1.5)
            effective_scale = self.noise_scale * local_scale

            weight = self.mu_w + self.sigma_w * self.eps_w * effective_scale
            bias = self.mu_b + self.sigma_b * self.eps_b * effective_scale
        else:
            weight = self.mu_w
            bias = self.mu_b

        return tf.matmul(inputs, weight) + bias

    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
            "noisy_std": self.noisy_std,
        })
        return config
