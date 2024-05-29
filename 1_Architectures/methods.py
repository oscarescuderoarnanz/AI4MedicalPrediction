import numpy as np
import tensorflow as tf

concat = tf.keras.backend.concatenate
stack = tf.keras.backend.stack
K = tf.keras.backend
Dense = tf.keras.layers.Dense
Add = tf.keras.layers.Add
LayerNorm = tf.keras.layers.LayerNormalization
Multiply = tf.keras.layers.Multiply
Dropout = tf.keras.layers.Dropout
Activation = tf.keras.layers.Activation
Lambda = tf.keras.layers.Lambda

tf.config.experimental_run_functions_eagerly(True)
class MyCustomConv1DLayer(tf.keras.layers.Layer):
    """Convolutional layer with mask support.

    Convolutional layers with simple implementation of masks type A and B for
    autoregressive models.

    Arguments:
    filters: Integer, the dimensionality of the output space
        (i.e. the number of output filters in the convolution).
    kernel_size: An integer, specifying the height and width of the 2D convolution window.
        Can be a single integer to specify the same value for all spatial dimensions.
    kernel_initializer: Initializer for the `kernel` weights matrix.
    """
    def __init__(self, 
                 filters, 
                 kernel_size=3, 
                 kernel_initializer='glorot_uniform',
                 **kwargs):
        super(MyCustomConv1DLayer, self).__init__(**kwargs)
        # Signal that the layer is safe for mask propagation
        self.supports_masking = True
        self.filters = filters
        self.kernel_size = kernel_size
    
    def call(self, input, mask=None):
        # using 'mask' you can access the mask passed from the previous layer
        
        #Perform the convolution
        convolved_input = tf.keras.layers.Conv1D(
            self.filters,
            self.kernel_size
        )(input)
        
        # Prepare the mask to be used with the 3d tensor
        mask_matrix = tf.transpose(
            (tf.ones([convolved_input.shape[2], 1, 1]) * tf.cast(mask, "float")), 
            perm=[1, 2, 0]
        )
        mask_matrix = (mask_matrix == 1)
        
        # Apply masking
        output = tf.where(mask_matrix, convolved_input,  tf.zeros_like(convolved_input))
        
        return output
    

def standard_attention(qs, ks, vs, hidden_layer_size, dropout_rate, n_head):
    """Apply the standard attention layer.
        Returns:
        Processed tensor outputs.
    """    
    class ScaledDotProductAttention():
        """Defines scaled dot product attention layer.
          Attributes:
            dropout: Dropout rate to use
            activation: Normalisation function for scaled dot product attention (e.g.
              softmax by default)
        """
        def __init__(self, attn_dropout=0.0):
            self.supports_masking = True
            self.dropout = Dropout(attn_dropout)
            self.activation = Activation('softmax')

        def __call__(self, q, k, v, mask=None):
            """Applies scaled dot product attention.
            Args:
              q: Queries
              k: Keys
              v: Values
              mask: Masking if required -- sets softmax to very large value

            Returns:
              Tuple of (layer outputs, attention weights)
            """
            attn = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[2, 2]))([q, k])  # shape=(batch, q, k)
            temper = tf.sqrt(tf.cast(tf.shape(k)[-1], dtype='float32'))
            attn = attn / temper  # shape=(batch, q, k)
            attn = self.activation(attn)
            output = Lambda(lambda x: K.batch_dot(x[0], x[1]))([attn, v])
            return output, attn


    heads = []
    attns = []
    d_k = 5 // n_head
    d_v = hidden_layer_size // n_head
    for i in range(n_head):
        qs = tf.keras.layers.TimeDistributed(Dense(d_k, use_bias=False))(qs)
        ks = tf.keras.layers.TimeDistributed(Dense(d_k, use_bias=False))(ks)
        vs = tf.keras.layers.TimeDistributed(Dense(d_v, use_bias=False))(vs)
        head, attn = ScaledDotProductAttention()(qs, ks, vs)
    return head, attn



# Layer utility functions.
def linear_layer(size,
                 activation=None,
                 use_time_distributed=False,
                 use_bias=True):
    """Returns simple Keras linear layer.
      Args:
        size: Output size
        activation: Activation function to apply if required
        use_time_distributed: Whether to apply layer across time
        use_bias: Whether bias should be included in layer
    """
    linear = tf.keras.layers.Dense(size, activation=activation, use_bias=use_bias)
    if use_time_distributed:
        linear = tf.keras.layers.TimeDistributed(linear)
    return linear


def add_and_norm(x_list):
    """Applies skip connection followed by layer normalisation.

  Args:
    x_list: List of inputs to sum for skip connection

  Returns:
    Tensor output from layer.
  """
    tmp = Add()(x_list)
    tmp = LayerNorm()(tmp)
    return tmp

def apply_gating_layer(x,
                       hidden_layer_size,
                       dropout_rate=None,
                       use_time_distributed=True,
                       activation=None):
    """Applies a Gated Linear Unit (GLU) to an input.

  Args:
    x: Input to gating layer
    hidden_layer_size: Dimension of GLU
    dropout_rate: Dropout rate to apply if any
    use_time_distributed: Whether to apply across time
    activation: Activation function to apply to the linear feature transform if
      necessary

  Returns:
    Tuple of tensors for: (GLU output, gate)
  """

    if dropout_rate is not None:
        x = tf.keras.layers.Dropout(dropout_rate)(x)

    if use_time_distributed:
        activation_layer = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(hidden_layer_size, activation=activation))(
            x)
        gated_layer = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(hidden_layer_size, activation='sigmoid'))(
            x)
    else:
        activation_layer = tf.keras.layers.Dense(
            hidden_layer_size, activation=activation)(
            x)
        gated_layer = tf.keras.layers.Dense(
            hidden_layer_size, activation='sigmoid')(
            x)

    return tf.keras.layers.Multiply()([activation_layer,
                                       gated_layer]), gated_layer


def gated_residual_network(x,
                           hidden_layer_size,
                           output_size=None,
                           dropout_rate=None,
                           use_time_distributed=True,
                           additional_context=None,
                           return_gate=False):
    """Applies the gated residual network (GRN) as defined in paper.

      Args:
        x: Network inputs
        hidden_layer_size: Internal state size
        output_size: Size of output layer
        dropout_rate: Dropout rate if dropout is applied
        use_time_distributed: Whether to apply network across time dimension
        additional_context: Additional context vector to use if relevant
        return_gate: Whether to return GLU gate for diagnostic purposes

      Returns:
        Tuple of tensors for: (GRN output, GLU gate)
    """
    # Setup skip connection
    if output_size is None:
        output_size = hidden_layer_size
        skip = x
    else:
        linear = Dense(output_size)
        if use_time_distributed:
            linear = tf.keras.layers.TimeDistributed(linear)
        skip = linear(x)

    # Apply feedforward network
    hidden = linear_layer(
        hidden_layer_size,
        activation=None,
        use_time_distributed=use_time_distributed)(
        x)
    if additional_context is not None:
        hidden = hidden + linear_layer(
            hidden_layer_size,
            activation=None,
            use_time_distributed=use_time_distributed,
            use_bias=False)(
            additional_context)
    hidden = tf.keras.layers.Activation('elu')(hidden)
    hidden = linear_layer(
        hidden_layer_size,
        activation=None,
        use_time_distributed=use_time_distributed)(
        hidden)

    gating_layer, gate = apply_gating_layer(
        hidden,
        output_size,
        dropout_rate=dropout_rate,
        use_time_distributed=use_time_distributed,
        activation=None)

    if return_gate:
        return add_and_norm([skip, gating_layer]), gate
    else:
        return add_and_norm([skip, gating_layer])

    
def hadamard_attention(input_layer, hidden_layer_size, dropout):
    """Apply temporal variable selection networks.
        Returns:
        Processed tensor outputs.
    """
    
    
    _, time_steps, num_inputs = input_layer.shape

    # Variable selection weights
    mlp_outputs, static_gate = gated_residual_network(
        input_layer,
        hidden_layer_size,
        output_size=num_inputs,
        dropout_rate=dropout,
        use_time_distributed=True,
        additional_context=None,
        return_gate=True)
    sparse_weights = tf.keras.layers.Activation('softmax', name="softmax_dyn")(mlp_outputs)

    # Score multiplication
    combined = tf.keras.layers.Multiply()([sparse_weights, input_layer])

    return combined, sparse_weights