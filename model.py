import inspect
from typing import List
import tensorflow as tf
from tensorflow.keras import backend as K, Model, Input, optimizers
from tensorflow.keras import layers
from tensorflow.keras.layers import Activation, SpatialDropout1D, Lambda
from tensorflow.keras.layers import Layer, Conv1D, Dense, BatchNormalization, LayerNormalization, GlobalAveragePooling1D
from tensorflow.keras.regularizers import l2
import tensorflow.keras as keras
import tensorflow_addons as tfa
import math


def is_power_of_two(num: int):
    return num != 0 and ((num & (num - 1)) == 0)


def adjust_dilations(dilations: list):
    if all([is_power_of_two(i) for i in dilations]):
        return dilations
    else:
        new_dilations = [2 ** i for i in dilations]
        return new_dilations


def abs_backend(inputs):
    return K.abs(inputs)


def expand_dim(inputs):
    return K.expand_dims(inputs, 1)


def sign_backend(inputs):
    return K.sign(inputs)


class DRSNBlock(Layer):

    def __init__(self,
                 dilation_rate: int,
                 nb_filters: int,
                 kernel_size: int,
                 padding: str,
                 activation: str = 'relu',
                 dropout_rate: float = 0,
                 kernel_initializer: str = 'he_normal',
                 use_batch_norm: bool = False,
                 use_layer_norm: bool = False,
                 use_weight_norm: bool = False,
                 channels=200,
                 **kwargs):
        """Defines the residual block for the WaveNet TCN
        Args:
            x: The previous layer in the model
            training: boolean indicating whether the layer should behave in training mode or in inference mode
            dilation_rate: The dilation power of 2 we are using for this residual block
            nb_filters: The number of convolutional filters to use in this block
            kernel_size: The size of the convolutional kernel
            padding: The padding used in the convolutional layers, 'same' or 'causal'.
            activation: The final activation used in o = Activation(x + F(x))
            dropout_rate: Float between 0 and 1. Fraction of the input units to drop.
            kernel_initializer: Initializer for the kernel weights matrix (Conv1D).
            use_batch_norm: Whether to use batch normalization in the residual layers or not.
            use_layer_norm: Whether to use layer normalization in the residual layers or not.
            use_weight_norm: Whether to use weight normalization in the residual layers or not.
            kwargs: Any initializers for Layer class.
        """

        self.dilation_rate = dilation_rate
        self.nb_filters = nb_filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        self.use_weight_norm = use_weight_norm
        self.kernel_initializer = kernel_initializer
        self.layers = []
        self.layers_outputs = []
        self.shape_match_conv = None
        self.res_output_shape = None
        self.final_activation = None

        self.dense1 = Dense(channels, activation=None, kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))
        self.dense2 = Dense(channels, activation='sigmoid', kernel_regularizer=l2(1e-4))


        self.normal = BatchNormalization()
        # if self.use_batch_norm:
        #     self.normal = BatchNormalization()
        # elif self.use_layer_norm:
        #     self.normal = LayerNormalization()
        # elif self.use_weight_norm:
        #     from tensorflow_addons.layers import WeightNormalization
        #     self.normal = WeightNormalization()
        self.act = Activation('relu')
        self.gap = GlobalAveragePooling1D()

        super(DRSNBlock, self).__init__(**kwargs)

    def _build_layer(self, layer):
        """Helper function for building layer
        Args:
            layer: Appends layer to internal layer list and builds it based on the current output
                   shape of ResidualBlocK. Updates current output shape.
        """
        self.layers.append(layer)
        self.layers[-1].build(self.res_output_shape)
        self.res_output_shape = self.layers[-1].compute_output_shape(self.res_output_shape)

    def build(self, input_shape):

        with K.name_scope(self.name):  # name scope used to make sure weights get unique names
            self.layers = []
            self.res_output_shape = input_shape

            for k in range(2):
                name = 'conv1D_{}'.format(k)
                with K.name_scope(name):  # name scope used to make sure weights get unique names
                    conv = Conv1D(
                        filters=self.nb_filters,
                        kernel_size=self.kernel_size,
                        dilation_rate=self.dilation_rate,
                        padding=self.padding,
                        name=name,
                        kernel_initializer=self.kernel_initializer
                    )
                    if self.use_weight_norm:
                        from tensorflow_addons.layers import WeightNormalization
                        # wrap it. WeightNormalization API is different than BatchNormalization or LayerNormalization.
                        with K.name_scope('norm_{}'.format(k)):
                            conv = WeightNormalization(conv)
                    self._build_layer(conv)

                with K.name_scope('norm_{}'.format(k)):
                    if self.use_batch_norm:
                        self._build_layer(BatchNormalization())
                    elif self.use_layer_norm:
                        self._build_layer(LayerNormalization())
                    elif self.use_weight_norm:
                        pass  # done above.

                self._build_layer(Activation(self.activation))
                self._build_layer(SpatialDropout1D(rate=self.dropout_rate))

            if self.nb_filters != input_shape[-1]:
                # 1x1 conv to match the shapes (channel dimension).
                name = 'matching_conv1D'
                with K.name_scope(name):
                    # make and build this layer separately because it directly uses input_shape
                    self.shape_match_conv = Conv1D(filters=self.nb_filters,
                                                   kernel_size=1,
                                                   padding='same',
                                                   name=name,
                                                   kernel_initializer=self.kernel_initializer)
            else:
                name = 'matching_identity'
                self.shape_match_conv = Lambda(lambda x: x, name=name)

            with K.name_scope(name):
                self.shape_match_conv.build(input_shape)
                self.res_output_shape = self.shape_match_conv.compute_output_shape(input_shape)

            self._build_layer(Activation(self.activation))
            self.final_activation = Activation(self.activation)
            self.final_activation.build(self.res_output_shape)  # probably isn't necessary

            # this is done to force Keras to add the layers in the list to self._layers
            for layer in self.layers:
                self.__setattr__(layer.name, layer)
            self.__setattr__(self.shape_match_conv.name, self.shape_match_conv)
            self.__setattr__(self.final_activation.name, self.final_activation)

            super(DRSNBlock, self).build(input_shape)  # done to make sure self.built is set True

    def call(self, inputs, training=None):
        """
        Returns: A tuple where the first element is the residual model tensor, and the second
                 is the skip connection tensor.
        """
        x = inputs
        self.layers_outputs = [x]
        for layer in self.layers:
            training_flag = 'training' in dict(inspect.signature(layer.call).parameters)
            x = layer(x, training=training) if training_flag else layer(x)
            self.layers_outputs.append(x)
        x2 = self.shape_match_conv(inputs)
        self.layers_outputs.append(x2)

        residual = x
        residual_abs = Lambda(abs_backend)(residual)
        abs_mean = self.gap(residual_abs)

        # print(abs_mean)

        # channels = x.get_shape().as_list()[-1]

        scales = self.dense1(abs_mean)
        scales = self.normal(scales)
        scales = self.act(scales)
        scales = self.dense2(scales)

        # print(scales, abs_mean)

        thres = keras.layers.multiply([abs_mean, scales])

        sub = keras.layers.subtract([residual_abs, thres])
        zeros = keras.layers.subtract([sub, sub])
        n_sub = keras.layers.maximum([sub, zeros])
        residual = keras.layers.multiply([Lambda(sign_backend)(residual), n_sub])
        x = residual

        # print(Lambda(sign_backend)(residual))

        res_x = layers.add([x2, x])
        self.layers_outputs.append(res_x)

        res_act_x = self.final_activation(res_x)
        self.layers_outputs.append(res_act_x)

        return [res_act_x, x]

    def compute_output_shape(self, input_shape):
        return [self.res_output_shape, self.res_output_shape]

class ResidualBlock(Layer):
    def __init__(self,
                 dilation_rate: int,
                 nb_filters: int,
                 kernel_size: int,
                 padding: str,
                 activation: str = 'relu',
                 dropout_rate: float = 0,
                 kernel_initializer: str = 'he_normal',
                 use_batch_norm: bool = False,
                 use_layer_norm: bool = False,
                 use_weight_norm: bool = False,
                 **kwargs):
        """Defines the residual block for the WaveNet TCN
        Args:
            x: The previous layer in the model
            training: boolean indicating whether the layer should behave in training mode or in inference mode
            dilation_rate: The dilation power of 2 we are using for this residual block
            nb_filters: The number of convolutional filters to use in this block
            kernel_size: The size of the convolutional kernel
            padding: The padding used in the convolutional layers, 'same' or 'causal'.
            activation: The final activation used in o = Activation(x + F(x))
            dropout_rate: Float between 0 and 1. Fraction of the input units to drop.
            kernel_initializer: Initializer for the kernel weights matrix (Conv1D).
            use_batch_norm: Whether to use batch normalization in the residual layers or not.
            use_layer_norm: Whether to use layer normalization in the residual layers or not.
            use_weight_norm: Whether to use weight normalization in the residual layers or not.
            kwargs: Any initializers for Layer class.
        """

        self.dilation_rate = dilation_rate
        self.nb_filters = nb_filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        self.use_weight_norm = use_weight_norm
        self.kernel_initializer = kernel_initializer
        self.layers = []
        self.shape_match_conv = None
        self.res_output_shape = None
        self.final_activation = None

        super(ResidualBlock, self).__init__(**kwargs)

    def _build_layer(self, layer):
        """Helper function for building layer
        Args:
            layer: Appends layer to internal layer list and builds it based on the current output
                   shape of ResidualBlocK. Updates current output shape.
        """
        self.layers.append(layer)
        self.layers[-1].build(self.res_output_shape)
        self.res_output_shape = self.layers[-1].compute_output_shape(self.res_output_shape)

    def build(self, input_shape):

        with K.name_scope(self.name):  # name scope used to make sure weights get unique names
            self.layers = []
            self.res_output_shape = input_shape

            for k in range(2):  # dilated conv block.
                name = 'conv1D_{}'.format(k)
                with K.name_scope(name):  # name scope used to make sure weights get unique names
                    conv = Conv1D(
                        filters=self.nb_filters,
                        kernel_size=self.kernel_size,
                        dilation_rate=self.dilation_rate,
                        padding=self.padding,
                        name=name,
                        kernel_initializer=self.kernel_initializer
                    )
                    if self.use_weight_norm:
                        from tensorflow_addons.layers import WeightNormalization
                        # wrap it. WeightNormalization API is different than BatchNormalization or LayerNormalization.
                        with K.name_scope('norm_{}'.format(k)):
                            conv = WeightNormalization(conv)
                    self._build_layer(conv)

                with K.name_scope('norm_{}'.format(k)):
                    if self.use_batch_norm:
                        self._build_layer(BatchNormalization())
                    elif self.use_layer_norm:
                        self._build_layer(LayerNormalization())
                    elif self.use_weight_norm:
                        pass  # done above.

                with K.name_scope('act_and_dropout_{}'.format(k)):
                    self._build_layer(Activation(self.activation, name='Act_Conv1D_{}'.format(k)))
                    self._build_layer(SpatialDropout1D(rate=self.dropout_rate, name='SDropout_{}'.format(k)))

            if self.nb_filters != input_shape[-1]:
                # 1x1 conv to match the shapes (channel dimension).
                name = 'matching_conv1D'
                with K.name_scope(name):
                    # make and build this layer separately because it directly uses input_shape.
                    # 1x1 conv.
                    self.shape_match_conv = Conv1D(
                        filters=self.nb_filters,
                        kernel_size=1,
                        padding='same',
                        name=name,
                        kernel_initializer=self.kernel_initializer
                    )
            else:
                name = 'matching_identity'
                self.shape_match_conv = Lambda(lambda x: x, name=name)

            with K.name_scope(name):
                self.shape_match_conv.build(input_shape)
                self.res_output_shape = self.shape_match_conv.compute_output_shape(input_shape)

            self._build_layer(Activation(self.activation, name='Act_Conv_Blocks'))
            self.final_activation = Activation(self.activation, name='Act_Res_Block')
            self.final_activation.build(self.res_output_shape)  # probably isn't necessary

            # this is done to force Keras to add the layers in the list to self._layers
            for layer in self.layers:
                self.__setattr__(layer.name, layer)
            self.__setattr__(self.shape_match_conv.name, self.shape_match_conv)
            self.__setattr__(self.final_activation.name, self.final_activation)

            super(ResidualBlock, self).build(input_shape)  # done to make sure self.built is set True

    def call(self, inputs, training=None, **kwargs):
        """
        Returns: A tuple where the first element is the residual model tensor, and the second
                 is the skip connection tensor.
        """
        # https://arxiv.org/pdf/1803.01271.pdf  page 4, Figure 1 (b).
        # x1: Dilated Conv -> Norm -> Dropout (x2).
        # x2: Residual (1x1 matching conv - optional).
        # Output: x1 + x2.
        # x1 -> connected to skip connections.
        # x1 + x2 -> connected to the next block.
        #       input
        #     x1      x2
        #   conv1D    1x1 Conv1D (optional)
        #    ...
        #   conv1D
        #    ...
        #       x1 + x2
        x1 = inputs
        for layer in self.layers:
            training_flag = 'training' in dict(inspect.signature(layer.call).parameters)
            x1 = layer(x1, training=training) if training_flag else layer(x1)
        x2 = self.shape_match_conv(inputs)
        x1_x2 = self.final_activation(layers.add([x2, x1], name='Add_Res'))
        return [x1_x2, x1]

    def compute_output_shape(self, input_shape):
        return [self.res_output_shape, self.res_output_shape]

class TCN(Layer):
    """Creates a TCN layer.
        Input shape:
            A tensor of shape (batch_size, timesteps, input_dim).
        Args:
            nb_filters: The number of filters to use in the convolutional layers. Can be a list.
            kernel_size: The size of the kernel to use in each convolutional layer.
            dilations: The list of the dilations. Example is: [1, 2, 4, 8, 16, 32, 64].
            nb_stacks : The number of stacks of residual blocks to use.
            padding: The padding to use in the convolutional layers, 'causal' or 'same'.
            use_skip_connections: Boolean. If we want to add skip connections from input to each residual blocK.
            return_sequences: Boolean. Whether to return the last output in the output sequence, or the full sequence.
            activation: The activation used in the residual blocks o = Activation(x + F(x)).
            dropout_rate: Float between 0 and 1. Fraction of the input units to drop.
            kernel_initializer: Initializer for the kernel weights matrix (Conv1D).
            use_batch_norm: Whether to use batch normalization in the residual layers or not.
            use_layer_norm: Whether to use layer normalization in the residual layers or not.
            use_weight_norm: Whether to use weight normalization in the residual layers or not.
            kwargs: Any other arguments for configuring parent class Layer. For example "name=str", Name of the model.
                    Use unique names when using multiple TCN.
        Returns:
            A TCN layer.
        """

    def __init__(self,
                 nb_filters=64,
                 kernel_size=3,
                 nb_stacks=1,
                 dilations=(1, 2, 4, 8, 16, 32),
                 padding='causal',
                 use_skip_connections=True,
                 dropout_rate=0.0,
                 return_sequences=False,
                 activation='relu',
                 kernel_initializer='he_normal',
                 use_batch_norm=False,
                 use_layer_norm=False,
                 use_weight_norm=False,
                 use_drsn=False,
                 **kwargs):

        self.return_sequences = return_sequences
        self.dropout_rate = dropout_rate
        self.use_skip_connections = use_skip_connections
        self.dilations = dilations
        self.nb_stacks = nb_stacks
        self.kernel_size = kernel_size
        self.nb_filters = nb_filters
        self.activation = activation
        self.padding = padding
        self.kernel_initializer = kernel_initializer
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        self.use_weight_norm = use_weight_norm
        self.skip_connections = []
        self.residual_blocks = []
        self.layers_outputs = []
        self.build_output_shape = None
        self.slicer_layer = None  # in case return_sequence=False
        self.output_slice_index = None  # in case return_sequence=False
        self.padding_same_and_time_dim_unknown = False  # edge case if padding='same' and time_dim = None

        self.use_drsn = use_drsn

        if self.use_batch_norm + self.use_layer_norm + self.use_weight_norm > 1:
            raise ValueError('Only one normalization can be specified at once.')

        if isinstance(self.nb_filters, list):
            assert len(self.nb_filters) == len(self.dilations)

        if padding != 'causal' and padding != 'same':
            raise ValueError("Only 'causal' or 'same' padding are compatible for this layer.")

        # initialize parent class
        super(TCN, self).__init__(**kwargs)

    @property
    def receptive_field(self):
        return 1 + 2 * (self.kernel_size - 1) * self.nb_stacks * sum(self.dilations)

    def build(self, input_shape):

        # member to hold current output shape of the layer for building purposes
        self.build_output_shape = input_shape

        # list to hold all the member ResidualBlocks
        self.residual_blocks = []
        total_num_blocks = self.nb_stacks * len(self.dilations)
        if not self.use_skip_connections:
            total_num_blocks += 1  # cheap way to do a false case for below

        for s in range(self.nb_stacks):
            for i, d in enumerate(self.dilations):
                res_block_filters = self.nb_filters[i] if isinstance(self.nb_filters, list) else self.nb_filters
                if self.use_drsn:
                    self.residual_blocks.append(DRSNBlock(dilation_rate=d,
                                                            nb_filters=res_block_filters,
                                                            kernel_size=self.kernel_size,
                                                            padding=self.padding,
                                                            activation=self.activation,
                                                            dropout_rate=self.dropout_rate,
                                                            use_batch_norm=self.use_batch_norm,
                                                            use_layer_norm=self.use_layer_norm,
                                                            use_weight_norm=self.use_weight_norm,
                                                            kernel_initializer=self.kernel_initializer,
                                                            channels=res_block_filters,
                                                            name='residual_block_{}'.format(len(self.residual_blocks))))
                else:
                    self.residual_blocks.append(ResidualBlock(dilation_rate=d,
                                                            nb_filters=res_block_filters,
                                                            kernel_size=self.kernel_size,
                                                            padding=self.padding,
                                                            activation=self.activation,
                                                            dropout_rate=self.dropout_rate,
                                                            use_batch_norm=self.use_batch_norm,
                                                            use_layer_norm=self.use_layer_norm,
                                                            use_weight_norm=self.use_weight_norm,
                                                            kernel_initializer=self.kernel_initializer,
                                                            name='residual_block_{}'.format(len(self.residual_blocks))))
                # build newest residual block
                self.residual_blocks[-1].build(self.build_output_shape)
                self.build_output_shape = self.residual_blocks[-1].res_output_shape

        # this is done to force keras to add the layers in the list to self._layers
        for layer in self.residual_blocks:
            self.__setattr__(layer.name, layer)

        self.output_slice_index = None
        if self.padding == 'same':
            time = self.build_output_shape.as_list()[1]
            if time is not None:  # if time dimension is defined. e.g. shape = (bs, 500, input_dim).
                self.output_slice_index = int(self.build_output_shape.as_list()[1] / 2)
            else:
                # It will known at call time. c.f. self.call.
                self.padding_same_and_time_dim_unknown = True

        else:
            self.output_slice_index = -1  # causal case.
        self.slicer_layer = Lambda(lambda tt: tt[:, self.output_slice_index, :])

    def compute_output_shape(self, input_shape):
        """
        Overridden in case keras uses it somewhere... no idea. Just trying to avoid future errors.
        """
        if not self.built:
            self.build(input_shape)
        if not self.return_sequences:
            batch_size = self.build_output_shape[0]
            batch_size = batch_size.value if hasattr(batch_size, 'value') else batch_size
            nb_filters = self.build_output_shape[-1]
            return [batch_size, nb_filters]
        else:
            # Compatibility tensorflow 1.x
            return [v.value if hasattr(v, 'value') else v for v in self.build_output_shape]

    def call(self, inputs, training=None):
        x = inputs
        self.layers_outputs = [x]
        self.skip_connections = []
        for layer in self.residual_blocks:
            try:
                x, skip_out = layer(x, training=training)
            except TypeError:  # compatibility with tensorflow 1.x
                x, skip_out = layer(K.cast(x, 'float32'), training=training)
            self.skip_connections.append(skip_out)
            self.layers_outputs.append(x)

        if self.use_skip_connections:
            x = layers.add(self.skip_connections)
            self.layers_outputs.append(x)

        if not self.return_sequences:
            # case: time dimension is unknown. e.g. (bs, None, input_dim).
            if self.padding_same_and_time_dim_unknown:
                self.output_slice_index = K.shape(self.layers_outputs[-1])[1] // 2
            x = self.slicer_layer(x)
            self.layers_outputs.append(x)
        return x

    def get_config(self):
        """
        Returns the config of a the layer. This is used for saving and loading from a model
        :return: python dictionary with specs to rebuild layer
        """
        config = super(TCN, self).get_config()
        config['nb_filters'] = self.nb_filters
        config['kernel_size'] = self.kernel_size
        config['nb_stacks'] = self.nb_stacks
        config['dilations'] = self.dilations
        config['padding'] = self.padding
        config['use_skip_connections'] = self.use_skip_connections
        config['dropout_rate'] = self.dropout_rate
        config['return_sequences'] = self.return_sequences
        config['activation'] = self.activation
        config['use_batch_norm'] = self.use_batch_norm
        config['use_layer_norm'] = self.use_layer_norm
        config['use_weight_norm'] = self.use_weight_norm
        config['kernel_initializer'] = self.kernel_initializer
        return config

def compiled_tcn(num_feat,  # type: int
                 num_classes,  # type: int
                 nb_filters,  # type: int
                 kernel_size,  # type: int
                 dilations,  # type: List[int]
                 nb_stacks,  # type: int
                 max_len,  # type: int
                 output_len=1,  # type: int
                 padding='causal',  # type: str
                 use_skip_connections=False,  # type: bool
                 return_sequences=True,
                 regression=False,  # type: bool
                 dropout_rate=0.05,  # type: float
                 name='tcn',  # type: str,
                 kernel_initializer='he_normal',  # type: str,
                 activation='relu',  # type:str,
                 opt='adam',
                 lr=0.002,
                 use_batch_norm=False,
                 use_layer_norm=False,
                 use_weight_norm=False):
    # type: (...) -> Model
    """Creates a compiled TCN model for a given task (i.e. regression or classification).
    Classification uses a sparse categorical loss. Please input class ids and not one-hot encodings.
    Args:
        num_feat: The number of features of your input, i.e. the last dimension of: (batch_size, timesteps, input_dim).
        num_classes: The size of the final dense layer, how many classes we are predicting.
        nb_filters: The number of filters to use in the convolutional layers.
        kernel_size: The size of the kernel to use in each convolutional layer.
        dilations: The list of the dilations. Example is: [1, 2, 4, 8, 16, 32, 64].
        nb_stacks : The number of stacks of residual blocks to use.
        max_len: The maximum sequence length, use None if the sequence length is dynamic.
        padding: The padding to use in the convolutional layers.
        use_skip_connections: Boolean. If we want to add skip connections from input to each residual blocK.
        return_sequences: Boolean. Whether to return the last output in the output sequence, or the full sequence.
        regression: Whether the output should be continuous or discrete.
        dropout_rate: Float between 0 and 1. Fraction of the input units to drop.
        activation: The activation used in the residual blocks o = Activation(x + F(x)).
        name: Name of the model. Useful when having multiple TCN.
        kernel_initializer: Initializer for the kernel weights matrix (Conv1D).
        opt: Optimizer name.
        lr: Learning rate.
        use_batch_norm: Whether to use batch normalization in the residual layers or not.
        use_layer_norm: Whether to use layer normalization in the residual layers or not.
        use_weight_norm: Whether to use weight normalization in the residual layers or not.
    Returns:
        A compiled keras TCN.
    """

    dilations = adjust_dilations(dilations)

    input_layer = Input(shape=(max_len, num_feat))

    x = TCN(nb_filters, kernel_size, nb_stacks, dilations, padding,
            use_skip_connections, dropout_rate, return_sequences,
            activation, kernel_initializer, use_batch_norm, use_layer_norm,
            use_weight_norm, name=name)(input_layer)

    def get_opt():
        if opt == 'adam':
            return optimizers.Adam(lr=lr, clipnorm=1.)
        elif opt == 'rmsprop':
            return optimizers.RMSprop(lr=lr, clipnorm=1.)
        else:
            raise Exception('Only Adam and RMSProp are available here')

    if not regression:
        # classification
        x = Dense(num_classes)(x)
        x = Activation('softmax')(x)
        output_layer = x
        model = Model(input_layer, output_layer)

        # https://github.com/keras-team/keras/pull/11373
        # It's now in Keras@master but still not available with pip.
        # TODO remove later.
        def accuracy(y_true, y_pred):
            # reshape in case it's in shape (num_samples, 1) instead of (num_samples,)
            if K.ndim(y_true) == K.ndim(y_pred):
                y_true = K.squeeze(y_true, -1)
            # convert dense predictions to labels
            y_pred_labels = K.argmax(y_pred, axis=-1)
            y_pred_labels = K.cast(y_pred_labels, K.floatx())
            return K.cast(K.equal(y_true, y_pred_labels), K.floatx())

        model.compile(get_opt(), loss='sparse_categorical_crossentropy', metrics=[accuracy])
    else:
        # regression
        x = Dense(output_len)(x)
        x = Activation('linear')(x)
        output_layer = x
        model = Model(input_layer, output_layer)
        model.compile(get_opt(), loss='mean_squared_error')
    print('model.x = {}'.format(input_layer.shape))
    print('model.y = {}'.format(output_layer.shape))
    return model

def tcn_full_summary(model: Model, expand_residual_blocks=True):
    layers = model._layers.copy()  # store existing layers
    model._layers.clear()  # clear layers

    for i in range(len(layers)):
        if isinstance(layers[i], TCN):
            for layer in layers[i]._layers:
                if not isinstance(layer, ResidualBlock):
                    if not hasattr(layer, '__iter__'):
                        model._layers.append(layer)
                else:
                    if expand_residual_blocks:
                        for lyr in layer._layers:
                            if not hasattr(lyr, '__iter__'):
                                model._layers.append(lyr)
                    else:
                        model._layers.append(layer)
        else:
            model._layers.append(layers[i])

    model.summary()  # print summary

    # restore original layers
    model._layers.clear()
    [model._layers.append(lyr) for lyr in layers]

class ExposureTCN(tf.keras.Model):
    def __init__(self, num_items, use_layer_norm=False, use_weight_norm=False, use_batch_norm=False, kernel_size=2, emb_dim=100, tcn_emb_dim=64, lr=1e-3, initializer=None, drop_out=0.5, use_drsn=True, name="TCN"):
        super(ExposureTCN, self).__init__(name=name)
        self.tcn = TCN(nb_filters=tcn_emb_dim,
                       return_sequences=True,
                       kernel_size=kernel_size,
                       nb_stacks=1,
                       use_skip_connections=False,
                       dropout_rate=drop_out,
                       use_layer_norm=use_layer_norm,
                       use_weight_norm=use_weight_norm,
                       use_batch_norm=use_batch_norm,
                       activation='relu',
                       use_drsn=use_drsn,
                       kernel_initializer=initializer)
        self.dense = tf.keras.layers.Dense(emb_dim, activation='linear', kernel_initializer=initializer)
        self.optimizer = tfa.optimizers.AdamW(learning_rate=lr, weight_decay=0)
        self.emb_dim = emb_dim
        self.tcn_emb_dim = tcn_emb_dim

    def call(self, data, return_all=False):
        last_element_index = tf.compat.v1.count_nonzero(tf.reduce_sum(data, axis=2), axis=1)
        x = self.tcn(data)

        if not return_all:
            # You already subtract 1 when creating safe_index, so just gather directly with safe_index:
            safe_index = tf.clip_by_value(tf.cast(last_element_index, tf.int32) - 1, 0, tf.shape(x)[1] - 1)
            x = tf.gather(x, tf.reshape(safe_index, (-1, 1)), batch_dims=1)  # no second '-1'
            x = tf.reshape(x, (-1, self.tcn_emb_dim))
        else:
            pass
        return self.dense(x)

class SOUP(tf.keras.Model):
    def __init__(self, num_items, use_layer_norm=False, use_weight_norm=False, use_batch_norm=True, use_position_emb=True, use_drsn=True, drop_out=0.5, emb_dim=100, kernel_size=2, tcn_emb_dim=200, session_max_length=46, lr=0.001, name="SOUP"):
        super(SOUP, self).__init__(name=name)

        self.emb_dim = emb_dim
        self.session_max_length = session_max_length
        self.use_position_emb = use_position_emb

        stdv = 1.0 / math.sqrt(emb_dim)
        # initializer = tf.keras.initializers.RandomUniform(-stdv, stdv, seed=2023)
        initializer = tf.keras.initializers.RandomUniform(-0.05, 0.05, seed=42)

        self.tcn = ExposureTCN(num_items, use_layer_norm=use_layer_norm, use_weight_norm=use_weight_norm, drop_out=drop_out, use_batch_norm=use_batch_norm, kernel_size=kernel_size, emb_dim=emb_dim, tcn_emb_dim=tcn_emb_dim, initializer=tf.keras.initializers.RandomUniform(-0.05, 0.05, seed=42), use_drsn=use_drsn)


        self.exposure = tf.Variable(initial_value=initializer(shape=(num_items, )),
                                shape=(num_items, ), trainable=False, name='exposure')

        self.item_embeddings = tf.keras.layers.Embedding(num_items + 1, emb_dim, embeddings_initializer=initializer, )
        self.position_embedding = tf.keras.layers.Embedding(session_max_length, emb_dim, embeddings_initializer=initializer)

        self.w3 = tf.Variable(initial_value=initializer(shape=(emb_dim * 2, emb_dim)), shape=(emb_dim * 2, emb_dim), trainable=True, name='w3')

        self.q = tf.Variable(initial_value=initializer(shape=(emb_dim, 1)), shape=(emb_dim, 1), trainable=True, name='q')

        self.glu1 = tf.keras.layers.Dense(emb_dim, kernel_initializer=initializer, trainable=True, name='glu1')
        self.glu2 = tf.keras.layers.Dense(emb_dim, use_bias=False, kernel_initializer=initializer, trainable=True, name='glu2')

        self.num_items = num_items
        self.emb_dim = emb_dim

        self.optimizer = tfa.optimizers.AdamW(learning_rate=lr, weight_decay=0)
        self.dense = tf.keras.layers.Dense(emb_dim, activation='linear', kernel_initializer=initializer, trainable=True, name='dense')


    def safe_gather(self, embeddings, indices):
        # Create a mask where indices == -1, replace them with a large valid index (e.g., embeddings.shape[0])
        safe_indices = tf.where(indices < 0, tf.zeros_like(indices), indices)  # Replace -1 with 0
        gathered = tf.gather(embeddings, safe_indices)

        # Zero out the results where the original indices were -1
        mask = tf.cast(indices >= 0, embeddings.dtype)
        return gathered * tf.expand_dims(mask, -1)  # Keep correct shape


    def feedforward(self, data, training=True):
        alias_inputs, adjacency_matrix, items, mask, targets, seq = data

        ieb = self.item_embeddings(tf.keras.backend.arange(1, self.num_items + 1))
        # h_asc = tf.gather(ieb, seq - 1)
        h_asc = self.safe_gather(ieb, seq - 1)

        h_mean1 = self.tcn(h_asc, training=training)
        h_mean = self.dense(h_mean1)

        if self.use_position_emb:
            h_mean = tf.expand_dims(h_mean, -2)
            reversed_seq = tf.reverse_sequence(seq + 1, seq_lengths=tf.cast(tf.reduce_sum(mask, 1), tf.int64), seq_axis=1) - 2
            # h_des = tf.gather(ieb, reversed_seq)
            h_des = self.safe_gather(ieb, reversed_seq)


            pos = self.position_embedding(tf.keras.backend.arange(0, self.session_max_length))
            pos = tf.tile(tf.expand_dims(pos, 0), [tf.shape(h_des)[0], 1, 1])
            concat = tf.concat([pos, h_des], axis=-1)
            z = tf.nn.tanh(tf.matmul(concat, self.w3))

            # eq (13)
            beta = tf.matmul(tf.nn.sigmoid(self.glu1(z) + self.glu2(h_mean)), self.q)
            beta *= tf.expand_dims(mask, axis=-1)

            session_rep = tf.reduce_sum(beta * h_des, axis=1)

        else:

            session_rep = h_mean

        # session_rep = h_mean
        all_items_embedding = self.item_embeddings(tf.keras.backend.arange(1, self.num_items + 1))
        # all_items_embedding += pert
        predictions = tf.matmul(session_rep, tf.transpose(all_items_embedding, perm=[1, 0]))

        return session_rep, predictions
    

class SOFA(tf.keras.Model):
    def __init__(self, num_items, emb_dim, fair_tcn_emb_dim, tcn_emb_dim, session_max_length, use_position_emb=True, soda_drsn=True, use_drsn=True, epi=1, soup_drop_out=0.7, soda_drop_out=0.7, threshold=0.6, fair_kernel_size=2, train_fair=True, soup_steps=1, soda_lr=5e-3, soda_steps=3, lambda_=1e-3, soup_lr=0.001, fairness_steps=1, name="SOFA"):
        super(SOFA, self).__init__(name=name)

        self.num_items = num_items
        self.emb_dim = emb_dim
        self.session_max_length = session_max_length

        self.soup = SOUP(num_items, use_layer_norm=True, use_weight_norm=False, use_batch_norm=False, use_position_emb=use_position_emb, emb_dim=emb_dim, tcn_emb_dim=tcn_emb_dim, session_max_length=session_max_length, lr=soup_lr, drop_out=soup_drop_out, use_drsn=use_drsn)
        self.soda_steps = soda_steps
        self.soup_steps = soup_steps
        self.fairness_steps = fairness_steps
        self.lambda_ = lambda_
        self.train_fair = train_fair
        self.epi = epi
        self.threshold = threshold

        stdv = 1.0 / math.sqrt(emb_dim)
        initializer = tf.keras.initializers.RandomUniform(-0.05, 0.05, seed=42)

        self.soda = ExposureTCN(num_items, use_layer_norm=True, use_weight_norm=False, use_batch_norm=False, drop_out=soda_drop_out, kernel_size=fair_kernel_size, emb_dim=num_items, tcn_emb_dim=fair_tcn_emb_dim, lr=soda_lr, initializer=initializer, use_drsn=soda_drsn)
        self.soda.build(input_shape=(None, session_max_length, num_items))
        self.soda.save_weights("soda.h5")

    def gini(self, data, axis=None, eps=1e-8):
        '''Calculate the Gini coefficient of a numpy array.
        Parameters
        ----------
        data : array_like
            Array to compute the Gini index of along axis.
        axis : None or int, optional
            If axis=None, data is flattened before Gini index is computed.
            If axis is int, Gini index will be computed along the
            specified axis.
        eps : float, optional
            Small, positive number to make sure we don't divide by 0.
        Returns
        -------
        res : array_like
            The Gini coefficients of numpy array data.
        Notes
        -----
        Based on bottom eq on [2]_.
        References
        ----------
        .. [2]_ http://www.statsdirect.com/help/
                default.htm#nonparametric_methods/gini.htm
        '''

        # Move gini index axis up front
        data = tf.cast(tf.experimental.numpy.moveaxis(data, axis, 0), tf.float32)

        # Reshape so we only have two axes to deal with:
        N = tf.cast(tf.shape(data)[0], tf.float32)
        sh_orig = tf.shape(data)[1]
        data = tf.reshape(data, (N, -1))
        idx = tf.keras.backend.arange(0, N) + 1
        idx = tf.reshape(idx, (-1, 1))

        # Values cannot be negative
        # minval = tf.experimental.numpy.amin(tf.reshape(data, (-1, )))
        # data = data - tf.cast(tf.less(minval, 0.0), tf.float32) * minval
        # if minval < 0:
        # if tf.less(minval, 0.0):
        #     data -= minval

        # Values must be nonzero
        data += eps

        # Values must be sorted
        data = tf.sort(data, axis=0)

        # Calculate Gini coefficient
        num = tf.reduce_sum(tf.cast(2 * idx - N - 1, tf.float32) * data, axis=0)
        den = N * tf.reduce_sum(data, axis=0)
        res = num/den

        return tf.reshape(res, (-1, ))

    def test_step(self, data):
        alias_inputs, adjacency_matrix, items, nz_items, mask, targets, non_zeros, dids, history, is_last, seq = data

        # filter data
        index = tf.where(non_zeros)

        alias_inputs = tf.gather_nd(alias_inputs, index)
        adjacency_matrix = tf.gather_nd(adjacency_matrix, index)
        items = tf.gather_nd(items, index)
        nz_items = tf.gather_nd(nz_items, index)
        mask = tf.gather_nd(mask, index)
        is_last = tf.gather_nd(is_last, index)
        targets = tf.gather_nd(targets, index)
        seq = tf.gather_nd(seq, index)
        dids = tf.gather_nd(dids, index)
        history = tf.gather_nd(history, index)
        mask = tf.cast(mask, tf.float32)
        is_last = tf.cast(is_last, tf.int64)

        # set shape info
        alias_inputs = tf.reshape(alias_inputs, (tf.shape(index)[0], self.session_max_length))
        adjacency_matrix = tf.reshape(adjacency_matrix, (tf.shape(index)[0], self.session_max_length, self.session_max_length))
        items = tf.reshape(items, (tf.shape(index)[0], self.session_max_length))
        nz_items = tf.reshape(nz_items, (tf.shape(index)[0], self.session_max_length))
        mask = tf.reshape(mask, (tf.shape(index)[0], self.session_max_length))
        dids = tf.reshape(dids, (tf.shape(index)[0], ))
        history = tf.reshape(history, (tf.shape(index)[0], self.session_max_length))
        seq = tf.reshape(seq, (tf.shape(index)[0], self.session_max_length))
        targets = tf.reshape(targets, (tf.shape(index)[0], ))
        # targets = tf.one_hot(targets - 1, self.num_items)

        all_index = tf.where(tf.equal(tf.reshape(history, (-1, tf.shape(history)[1], 1)), dids))
        index = tf.gather(all_index, [0, 1], axis=1)
        v = tf.gather(all_index, [2], axis=1)

        x = tf.scatter_nd(index, tf.reshape(v, (-1, )), tf.cast(tf.shape(history), tf.int64))
        x = tf.where(tf.equal(history, tf.constant(-1, dtype=tf.int64)), history, x)

        sr, predictions = self.soup.feedforward((alias_inputs, adjacency_matrix, items, mask, targets, seq), training=False)

        # Update the metrics.
        for m in self.metrics:
            if m.name.startswith('SIED'):
                m.update_state(tf.one_hot(targets - 1, self.num_items), (predictions, dids, history))
            else:
                m.update_state(tf.one_hot(targets - 1, self.num_items), predictions)
        # self.compiled_metrics.update_state(tf.one_hot(targets - 1, self.num_items), predictions)

        # Return a dict mapping metric names to current value.
        return {m.name: m.result() for m in self.metrics}
    
    def safe_gather(self, embeddings, indices):
        """
        Safely gather embeddings at specified indices. Handles out-of-bound indices by returning zeros.

        Args:
            embeddings: Tensor of shape (num_items, embedding_dim).
            indices: Tensor of shape (...), where each value should be in the range [0, num_items - 1].

        Returns:
            A tensor of gathered embeddings, with zeros where indices were out-of-bounds.
        """
        # Get the valid index range
        num_items = tf.shape(embeddings)[0]

        # Clamp indices to be within valid bounds [0, num_items - 1]
        clamped_indices = tf.clip_by_value(tf.cast(indices, tf.int32), 0, num_items - 1)

        # Gather the embeddings using clamped indices
        gathered = tf.gather(embeddings, clamped_indices)

        # Mask to identify where the original indices were out of bounds
        mask = tf.cast((tf.cast(indices, tf.int32) >= 0) & (tf.cast(indices, tf.int32) < num_items), embeddings.dtype)

        # Return gathered embeddings with zeros for out-of-bounds indices
        return gathered * tf.expand_dims(mask, -1)

    def train_step(self, data):
        alias_inputs, adjacency_matrix, items, nz_items, mask, targets, non_zeros, dids, history, is_last, seq = data

        # filter data
        index = tf.where(non_zeros)

        alias_inputs = tf.gather_nd(alias_inputs, index)
        adjacency_matrix = tf.gather_nd(adjacency_matrix, index)
        items = tf.gather_nd(items, index)
        nz_items = tf.gather_nd(nz_items, index)
        mask = tf.gather_nd(mask, index)
        is_last = tf.gather_nd(is_last, index)
        targets = tf.gather_nd(targets, index)
        seq = tf.gather_nd(seq, index)
        dids = tf.gather_nd(dids, index)
        history = tf.gather_nd(history, index)
        mask = tf.cast(mask, tf.float32)
        is_last = tf.cast(is_last, tf.float32)
        # is_last = tf.reshape(is_last, (-1, 1))

        # set shape info
        alias_inputs = tf.reshape(alias_inputs, (tf.shape(index)[0], self.session_max_length))
        adjacency_matrix = tf.reshape(adjacency_matrix, (tf.shape(index)[0], self.session_max_length, self.session_max_length))
        items = tf.reshape(items, (tf.shape(index)[0], self.session_max_length))
        nz_items = tf.reshape(nz_items, (tf.shape(index)[0], self.session_max_length))
        mask = tf.reshape(mask, (tf.shape(index)[0], self.session_max_length))
        dids = tf.reshape(dids, (tf.shape(index)[0], ))
        history = tf.reshape(history, (tf.shape(index)[0], self.session_max_length))
        seq = tf.reshape(seq, (tf.shape(index)[0], self.session_max_length))
        targets_ = tf.reshape(targets, (tf.shape(index)[0], ))
        targets = tf.one_hot(targets_ - 1, self.num_items)

        all_index = tf.where(tf.equal(tf.reshape(history, (-1, tf.shape(history)[1], 1)), dids))
        index = tf.gather(all_index, [0, 1], axis=1)
        v = tf.gather(all_index, [2], axis=1)

        x = tf.scatter_nd(index, tf.reshape(v, (-1, )), tf.cast(tf.shape(history), tf.int64))
        x = tf.where(tf.equal(history, tf.constant(-1, dtype=tf.int64)), history, x)

        def each_exposure(ipt):
            _, exposure = ipt
            loss = self.gini(exposure)
            return [loss, exposure]

        #========================================== Train SOUP ==========================================#
        with tf.GradientTape() as tape:
            _, predictions = self.soup.feedforward((alias_inputs, adjacency_matrix, items, mask, targets, seq))

            loss_func = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
            # loss_func = tf.keras.losses.CategoricalHinge()
            loss = loss_func(targets, predictions)
            loss = tf.reduce_mean(loss)

        gradient = tape.gradient(loss, self.soup.trainable_variables)
        self.soup.optimizer.apply_gradients(zip(gradient, self.soup.trainable_variables))

        if self.train_fair:

            #========================================== Train Fair ==========================================#

            self.soda.load_weights("soda.h5")
            _, predictions = self.soup.feedforward((alias_inputs, adjacency_matrix, items, mask, targets, seq))
            position = tf.cast(tf.argsort(tf.argsort(predictions, direction='DESCENDING')), tf.float32) + 1.
            exposure = 1. / tf.math.log(1. + position)

            sum = tf.reduce_sum(self.safe_gather(exposure, x), axis=1)
            num = tf.reshape(tf.reduce_sum(tf.cast(tf.greater_equal(x, tf.constant(0, dtype=tf.int64)), tf.float32), 1), (-1, 1))
            exposure_ground_truth = sum / num

            for _ in range(self.soda_steps):
                with tf.GradientTape() as tape:
                    session_exposure = self.soda(self.safe_gather(predictions, x), training=True)
                    loss = tf.keras.losses.KLD(tf.nn.softmax(exposure_ground_truth), tf.nn.softmax(session_exposure))
                    loss = tf.reduce_mean(loss)

                gradient = tape.gradient(loss, self.soda.trainable_variables)
                self.soda.optimizer.apply_gradients(zip(gradient, self.soda.trainable_variables))

            for _ in range(self.fairness_steps):

                with tf.GradientTape() as tape:
                    _, predictions = self.soup.feedforward((alias_inputs, adjacency_matrix, items, mask, targets, seq))
                    session_exposure = self.soda(self.safe_gather(predictions, x), training=False, return_all=False)

                    loss = tf.reduce_mean(self.gini(tf.abs(session_exposure), axis=1))
                    loss = self.lambda_ * tf.reduce_mean(loss)

                gradient = tape.gradient(loss, self.soup.trainable_variables)
                self.soup.optimizer.apply_gradients(zip(gradient, self.soup.trainable_variables))

        return {}

    @property
    def metrics(self):
        return self.compiled_metrics._metrics if self.compiled_metrics else []