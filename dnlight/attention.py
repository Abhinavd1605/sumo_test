"""
Multi-Agent Attention Communication for DNLight.

Implements the attention mechanism that allows fog nodes (units)
to exchange local state information and construct collaborative policies.
"""
import tensorflow as tf


class AttentionComm(tf.keras.layers.Layer):
    """
    Attention-based inter-agent communication layer.

    For unit i receiving messages from neighbor units j:
        Message:    m_j = tanh(W_m * s_j + b_m)
        Attention:  alpha_{ij} = softmax(s_i^T * W_a * m_j)
        Aggregate:  c_i = sum_j alpha_{ij} * m_j

    The aggregated signal c_i is appended to unit i's state
    before passing through the DQN.
    """

    def __init__(self, state_dim, message_dim=64, **kwargs):
        super().__init__(**kwargs)
        self.state_dim = state_dim
        self.message_dim = message_dim

    def build(self, input_shape):
        # Message projection: s_j -> m_j
        self.W_m = self.add_weight(
            name="W_m",
            shape=(self.state_dim, self.message_dim),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.b_m = self.add_weight(
            name="b_m",
            shape=(self.message_dim,),
            initializer="zeros",
            trainable=True,
        )

        # Attention projection: for computing compatibility
        self.W_a = self.add_weight(
            name="W_a",
            shape=(self.state_dim, self.message_dim),
            initializer="glorot_uniform",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, own_state, neighbor_states):
        """
        Compute attention-weighted aggregation of neighbor messages.

        Args:
            own_state: Tensor of shape (batch, state_dim) - unit i's state.
            neighbor_states: Tensor of shape (batch, n_neighbors, state_dim)
                           - states from neighboring units.

        Returns:
            aggregated: Tensor of shape (batch, message_dim)
                       - attention-weighted neighbor information.
        """
        # If no neighbors, return zeros
        if neighbor_states is None or tf.shape(neighbor_states)[1] == 0:
            batch_size = tf.shape(own_state)[0]
            return tf.zeros((batch_size, self.message_dim))

        # Compute messages: m_j = tanh(s_j @ W_m + b_m)
        # neighbor_states: (batch, n_neighbors, state_dim)
        messages = tf.tanh(
            tf.einsum('bns,sm->bnm', neighbor_states, self.W_m) + self.b_m
        )  # (batch, n_neighbors, message_dim)

        # Compute attention scores: s_i^T @ W_a @ m_j
        # query: s_i @ W_a -> (batch, message_dim)
        query = tf.matmul(own_state, self.W_a)  # (batch, message_dim)

        # Scores: (batch, n_neighbors)
        scores = tf.einsum('bm,bnm->bn', query, messages)

        # Softmax attention weights
        alpha = tf.nn.softmax(scores, axis=-1)  # (batch, n_neighbors)

        # Weighted aggregation: c_i = sum_j alpha_ij * m_j
        alpha_expanded = tf.expand_dims(alpha, -1)  # (batch, n_neighbors, 1)
        aggregated = tf.reduce_sum(
            alpha_expanded * messages, axis=1
        )  # (batch, message_dim)

        return aggregated

    def get_config(self):
        config = super().get_config()
        config.update({
            "state_dim": self.state_dim,
            "message_dim": self.message_dim,
        })
        return config
