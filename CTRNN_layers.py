from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.ops import array_ops
from tensorflow.python.util import nest
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training.tracking import base as trackable
from tensorflow.python.training.tracking import data_structures
from tensorflow.python.util.tf_export import tf_export
from tensorflow.python.keras.layers.recurrent \
    import _generate_dropout_mask, _generate_zero_filled_state_for_cell, RNN,\
    _config_for_enable_caching_device
from tensorflow.tools.docs import doc_controls

_ALMOST_ONE = 0.999999

RECURRENT_DROPOUT_WARNING_MSG = (
    'RNN `implementation=2` is not supported when `recurrent_dropout` is set. '
    'Using `implementation=1`.')

@doc_controls.do_not_generate_docs
class DropoutRNNCellMixin(object):
  """Object that hold dropout related fields for RNN Cell.

  This class is not a standalone RNN cell. It suppose to be used with a RNN cell
  by multiple inheritance. Any cell that mix with class should have following
  fields:
    dropout: a float number within range [0, 1). The ratio that the input
      tensor need to dropout.
    recurrent_dropout: a float number within range [0, 1). The ratio that the
      recurrent state weights need to dropout.
  This object will create and cache created dropout masks, and reuse them for
  the incoming data, so that the same mask is used for every batch input.
  """

  def __init__(self, *args, **kwargs):
    self._create_non_trackable_mask_cache()
    super(DropoutRNNCellMixin, self).__init__(*args, **kwargs)

  @trackable.no_automatic_dependency_tracking
  def _create_non_trackable_mask_cache(self):
    """Create the cache for dropout and recurrent dropout mask.

    Note that the following two masks will be used in "graph function" mode,
    e.g. these masks are symbolic tensors. In eager mode, the `eager_*_mask`
    tensors will be generated differently than in the "graph function" case,
    and they will be cached.

    Also note that in graph mode, we still cache those masks only because the
    RNN could be created with `unroll=True`. In that case, the `cell.call()`
    function will be invoked multiple times, and we want to ensure same mask
    is used every time.

    Also the caches are created without tracking. Since they are not picklable
    by python when deepcopy, we don't want `layer._obj_reference_counts_dict`
    to track it by default.
    """
    self._dropout_mask_cache = K.ContextValueCache(self._create_dropout_mask)
    self._recurrent_dropout_mask_cache = K.ContextValueCache(
        self._create_recurrent_dropout_mask)

  def reset_dropout_mask(self):
    """Reset the cached dropout masks if any.

    This is important for the RNN layer to invoke this in it `call()` method so
    that the cached mask is cleared before calling the `cell.call()`. The mask
    should be cached across the timestep within the same batch, but shouldn't
    be cached between batches. Otherwise it will introduce unreasonable bias
    against certain index of data within the batch.
    """
    self._dropout_mask_cache.clear()

  def reset_recurrent_dropout_mask(self):
    """Reset the cached recurrent dropout masks if any.

    This is important for the RNN layer to invoke this in it call() method so
    that the cached mask is cleared before calling the cell.call(). The mask
    should be cached across the timestep within the same batch, but shouldn't
    be cached between batches. Otherwise it will introduce unreasonable bias
    against certain index of data within the batch.
    """
    self._recurrent_dropout_mask_cache.clear()

  def _create_dropout_mask(self, inputs, training, count=1):
    return _generate_dropout_mask(
        array_ops.ones_like(inputs),
        self.dropout,
        training=training,
        count=count)

  def _create_recurrent_dropout_mask(self, inputs, training, count=1):
    return _generate_dropout_mask(
        array_ops.ones_like(inputs),
        self.recurrent_dropout,
        training=training,
        count=count)

  def get_dropout_mask_for_cell(self, inputs, training, count=1):
    """Get the dropout mask for RNN cell's input.

    It will create mask based on context if there isn't any existing cached
    mask. If a new mask is generated, it will update the cache in the cell.

    Args:
      inputs: The input tensor whose shape will be used to generate dropout
        mask.
      training: Boolean tensor, whether its in training mode, dropout will be
        ignored in non-training mode.
      count: Int, how many dropout mask will be generated. It is useful for cell
        that has internal weights fused together.
    Returns:
      List of mask tensor, generated or cached mask based on context.
    """
    if self.dropout == 0:
      return None
    init_kwargs = dict(inputs=inputs, training=training, count=count)
    return self._dropout_mask_cache.setdefault(kwargs=init_kwargs)

  def get_recurrent_dropout_mask_for_cell(self, inputs, training, count=1):
    """Get the recurrent dropout mask for RNN cell.

    It will create mask based on context if there isn't any existing cached
    mask. If a new mask is generated, it will update the cache in the cell.

    Args:
      inputs: The input tensor whose shape will be used to generate dropout
        mask.
      training: Boolean tensor, whether its in training mode, dropout will be
        ignored in non-training mode.
      count: Int, how many dropout mask will be generated. It is useful for cell
        that has internal weights fused together.
    Returns:
      List of mask tensor, generated or cached mask based on context.
    """
    if self.recurrent_dropout == 0:
      return None
    init_kwargs = dict(inputs=inputs, training=training, count=count)
    return self._recurrent_dropout_mask_cache.setdefault(kwargs=init_kwargs)

  def __getstate__(self):
    # Used for deepcopy. The caching can't be pickled by python, since it will
    # contain tensor and graph.
    state = super(DropoutRNNCellMixin, self).__getstate__()
    state.pop('_dropout_mask_cache', None)
    state.pop('_recurrent_dropout_mask_cache', None)
    return state

  def __setstate__(self, state):
    state['_dropout_mask_cache'] = K.ContextValueCache(
        self._create_dropout_mask)
    state['_recurrent_dropout_mask_cache'] = K.ContextValueCache(
        self._create_recurrent_dropout_mask)
    super(DropoutRNNCellMixin, self).__setstate__(state)


@tf_export('keras.layers.SimpleCTRNNCell')
class SimpleCTRNNCell(DropoutRNNCellMixin, Layer):
    """Cell class for SimpleCTRNNCell.

    Arguments:
        units: Positive integer, dimensionality of the output space.
        t: Positive float >= 1, timescale.
            Unit-dependent time constant of leakage.
        activation: Activation function to use.
            Default: hyperbolic tangent (`tanh`).
            If you pass `None`, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs.
        recurrent_initializer: Initializer for the `recurrent_kernel`
            weights matrix,
            used for the linear transformation of the recurrent state.
        bias_initializer: Initializer for the bias vector.
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix.
        recurrent_regularizer: Regularizer function applied to
            the `recurrent_kernel` weights matrix.
        bias_regularizer: Regularizer function applied to the bias vector.
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix.
        recurrent_constraint: Constraint function applied to
            the `recurrent_kernel` weights matrix.
        bias_constraint: Constraint function applied to the bias vector.
        dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the inputs.
        recurrent_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the recurrent state.
            
  Call arguments:
    inputs: A 2D tensor.
    states: List of state tensors corresponding to the previous timestep.
    training: Python boolean indicating whether the layer should behave in
      training mode or in inference mode. Only relevant when `dropout` or
      `recurrent_dropout` is used.
    """
    def __init__(self,
               units,
               t=1,
               activation='tanh',
               use_bias=True,
               kernel_initializer='glorot_uniform',
               recurrent_initializer='orthogonal',
               bias_initializer='zeros',
               kernel_regularizer=None,
               recurrent_regularizer=None,
               bias_regularizer=None,
               kernel_constraint=None,
               recurrent_constraint=None,
               bias_constraint=None,
               dropout=0.,
               recurrent_dropout=0.,
               **kwargs):
        if ops.executing_eagerly_outside_functions():
          self._enable_caching_device = kwargs.pop('enable_caching_device', True)
        else:
          self._enable_caching_device = kwargs.pop('enable_caching_device', False)
        super(SimpleCTRNNCell, self).__init__(**kwargs)
        self.units = units
        self.t = t
        self.activation = activations.get(activation)
        self.use_bias = use_bias
    
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
    
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
    
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
    
        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        implementation = kwargs.pop('implementation', 1)
        if self.recurrent_dropout != 0 and implementation != 1:
          logging.debug(RECURRENT_DROPOUT_WARNING_MSG)
          self.implementation = 1
        else:
          self.implementation = implementation
        self.state_size = data_structures.NoDependency([self.units, self.units])
        self.output_size = self.units
        
    @tf_utils.shape_type_conversion
    def build(self, input_shape):
      self.kernel = self.add_weight(
          shape=(input_shape[-1], self.units),
          name='kernel',
          initializer=self.kernel_initializer,
          regularizer=self.kernel_regularizer,
          constraint=self.kernel_constraint)
      self.recurrent_kernel = self.add_weight(
          shape=(self.units, self.units),
          name='recurrent_kernel',
          initializer=self.recurrent_initializer,
          regularizer=self.recurrent_regularizer,
          constraint=self.recurrent_constraint)
      if self.use_bias:
        self.bias = self.add_weight(
            shape=(self.units,),
            name='bias',
            initializer=self.bias_initializer,
            regularizer=self.bias_regularizer,
            constraint=self.bias_constraint)
      else:
        self.bias = None
      self.built = True
    
    def call(self, inputs, states, training=None):
        prev_output = states[0] if nest.is_nested(states) else states
        prev_z = states[1] if nest.is_nested(states) else states
        dp_mask = self.get_dropout_mask_for_cell(inputs, training)
        rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(
            prev_output, training)
        
        if dp_mask is not None:
          h = K.dot(inputs * dp_mask, self.kernel)
        else:
          h = K.dot(inputs, self.kernel)
        if self.bias is not None:
          h = K.bias_add(h, self.bias)
    
        if rec_dp_mask is not None:
          prev_output = prev_output * rec_dp_mask
        y = h + K.dot(prev_output, self.recurrent_kernel)
        
        z = (1. - 1. / self.t) * prev_z + (1. / self.t) * y
        
        if self.activation is not None:
            output = self.activation(z)
        else:
            output = z
        
        return output, [output, z]
      
    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return _generate_zero_filled_state_for_cell(self, inputs, batch_size, dtype)
    
    def get_config(self):
        config = {
            'units':
                self.units,
            't':
                self.t,
            'activation':
                activations.serialize(self.activation),
            'use_bias':
                self.use_bias,
            'kernel_initializer':
                initializers.serialize(self.kernel_initializer),
            'recurrent_initializer':
                initializers.serialize(self.recurrent_initializer),
            'bias_initializer':
                initializers.serialize(self.bias_initializer),
            'kernel_regularizer':
                regularizers.serialize(self.kernel_regularizer),
            'recurrent_regularizer':
                regularizers.serialize(self.recurrent_regularizer),
            'bias_regularizer':
                regularizers.serialize(self.bias_regularizer),
            'kernel_constraint':
                constraints.serialize(self.kernel_constraint),
            'recurrent_constraint':
                constraints.serialize(self.recurrent_constraint),
            'bias_constraint':
                constraints.serialize(self.bias_constraint),
            'dropout':
                self.dropout,
            'recurrent_dropout':
                self.recurrent_dropout
        }
        config.update(_config_for_enable_caching_device(self))
        base_config = super(SimpleCTRNNCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


@tf_export('keras.layers.SimpleCTRNN')
class SimpleCTRNN(RNN):
    """Fully-connected RNN where the output is to be fed back to input.
    
    Arguments:
      units: Positive integer, dimensionality of the output space.
      t: Positive float >= 1, timescale.
              Unit-dependent time constant of leakage.
      activation: Activation function to use.
        Default: hyperbolic tangent (`tanh`).
        If you pass None, no activation is applied
        (ie. "linear" activation: `a(x) = x`).
      use_bias: Boolean, whether the layer uses a bias vector.
      kernel_initializer: Initializer for the `kernel` weights matrix,
        used for the linear transformation of the inputs.
      recurrent_initializer: Initializer for the `recurrent_kernel`
        weights matrix,
        used for the linear transformation of the recurrent state.
      bias_initializer: Initializer for the bias vector.
      kernel_regularizer: Regularizer function applied to
        the `kernel` weights matrix.
      recurrent_regularizer: Regularizer function applied to
        the `recurrent_kernel` weights matrix.
      bias_regularizer: Regularizer function applied to the bias vector.
      activity_regularizer: Regularizer function applied to
        the output of the layer (its "activation")..
      kernel_constraint: Constraint function applied to
        the `kernel` weights matrix.
      recurrent_constraint: Constraint function applied to
        the `recurrent_kernel` weights matrix.
      bias_constraint: Constraint function applied to the bias vector.
      dropout: Float between 0 and 1.
        Fraction of the units to drop for
        the linear transformation of the inputs.
      recurrent_dropout: Float between 0 and 1.
        Fraction of the units to drop for
        the linear transformation of the recurrent state.
      return_sequences: Boolean. Whether to return the last output
        in the output sequence, or the full sequence.
      return_state: Boolean. Whether to return the last state
        in addition to the output.
      go_backwards: Boolean (default False).
        If True, process the input sequence backwards and return the
        reversed sequence.
      stateful: Boolean (default False). If True, the last state
        for each sample at index i in a batch will be used as initial
        state for the sample of index i in the following batch.
      unroll: Boolean (default False).
        If True, the network will be unrolled,
        else a symbolic loop will be used.
        Unrolling can speed-up a RNN,
        although it tends to be more memory-intensive.
        Unrolling is only suitable for short sequences.
    
    Call arguments:
      inputs: A 3D tensor.
      mask: Binary tensor of shape `(samples, timesteps)` indicating whether
        a given timestep should be masked.
      training: Python boolean indicating whether the layer should behave in
        training mode or in inference mode. This argument is passed to the cell
        when calling it. This is only relevant if `dropout` or
        `recurrent_dropout` is used.
      initial_state: List of initial state tensors to be passed to the first
        call of the cell.
    """
    def __init__(self,
                units,
                t=1,
                activation='tanh',
                use_bias=True,
                kernel_initializer='glorot_uniform',
                recurrent_initializer='orthogonal',
                bias_initializer='zeros',
                kernel_regularizer=None,
                recurrent_regularizer=None,
                bias_regularizer=None,
                activity_regularizer=None,
                kernel_constraint=None,
                recurrent_constraint=None,
                bias_constraint=None,
                dropout=0.,
                recurrent_dropout=0.,
                return_sequences=False,
                return_state=False,
                go_backwards=False,
                stateful=False,
                unroll=False,
                **kwargs):
        if 'implementation' in kwargs:
            kwargs.pop('implementation')
            logging.warning('The `implementation` argument '
                            'in `SimpleCTRNN` has been deprecated. '
                            'Please remove it from your layer call.')
        if 'enable_caching_device' in kwargs:
            cell_kwargs = {'enable_caching_device':
                         kwargs.pop('enable_caching_device')}
        else:
            cell_kwargs = {}
        cell = SimpleCTRNNCell(
            units,
            t,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            recurrent_initializer=recurrent_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            recurrent_regularizer=recurrent_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_constraint=kernel_constraint,
            recurrent_constraint=recurrent_constraint,
            bias_constraint=bias_constraint,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            dtype=kwargs.get('dtype'),
            trainable=kwargs.get('trainable', True),
            **cell_kwargs)
        super(SimpleCTRNN, self).__init__(
            cell,
            return_sequences=return_sequences,
            return_state=return_state,
            go_backwards=go_backwards,
            stateful=stateful,
            unroll=unroll,
            **kwargs)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.input_spec = [InputSpec(ndim=3)]
        
    def call(self, inputs, mask=None, training=None, initial_state=None):
        self._maybe_reset_cell_dropout_mask(self.cell)
        return super(SimpleCTRNN, self).call(
            inputs, mask=mask, training=training, initial_state=initial_state)
    
    @property
    def units(self):
      return self.cell.units
  
    @property
    def t(self):
        return self.cell.t

    @property
    def activation(self):
      return self.cell.activation

    @property
    def use_bias(self):
      return self.cell.use_bias

    @property
    def kernel_initializer(self):
      return self.cell.kernel_initializer
    
    @property
    def recurrent_initializer(self):
      return self.cell.recurrent_initializer
    
    @property
    def bias_initializer(self):
      return self.cell.bias_initializer
    
    @property
    def kernel_regularizer(self):
      return self.cell.kernel_regularizer
    
    @property
    def recurrent_regularizer(self):
      return self.cell.recurrent_regularizer
    
    @property
    def bias_regularizer(self):
      return self.cell.bias_regularizer
    
    @property
    def kernel_constraint(self):
      return self.cell.kernel_constraint
    
    @property
    def recurrent_constraint(self):
      return self.cell.recurrent_constraint
    
    @property
    def bias_constraint(self):
      return self.cell.bias_constraint
    
    @property
    def dropout(self):
      return self.cell.dropout
    
    @property
    def recurrent_dropout(self):
      return self.cell.recurrent_dropout

    def get_config(self):
      config = {
          'units':
              self.units,
          't':
              self.t,
          'activation':
              activations.serialize(self.activation),
          'use_bias':
              self.use_bias,
          'kernel_initializer':
              initializers.serialize(self.kernel_initializer),
          'recurrent_initializer':
              initializers.serialize(self.recurrent_initializer),
          'bias_initializer':
              initializers.serialize(self.bias_initializer),
          'kernel_regularizer':
              regularizers.serialize(self.kernel_regularizer),
          'recurrent_regularizer':
              regularizers.serialize(self.recurrent_regularizer),
          'bias_regularizer':
              regularizers.serialize(self.bias_regularizer),
          'activity_regularizer':
              regularizers.serialize(self.activity_regularizer),
          'kernel_constraint':
              constraints.serialize(self.kernel_constraint),
          'recurrent_constraint':
              constraints.serialize(self.recurrent_constraint),
          'bias_constraint':
              constraints.serialize(self.bias_constraint),
          'dropout':
              self.dropout,
          'recurrent_dropout':
              self.recurrent_dropout
      }
      base_config = super(SimpleCTRNN, self).get_config()
      config.update(_config_for_enable_caching_device(self.cell))
      del base_config['cell']
      return dict(list(base_config.items()) + list(config.items()))
    
    @classmethod
    def from_config(cls, config):
      if 'implementation' in config:
        config.pop('implementation')
      return cls(**config)


@tf_export('keras.layers.ACTRNNCell')
class ACTRNNCell(DropoutRNNCellMixin, Layer):
    """Cell class for ACTRNNCell.

    Arguments:
        units_vec: Positive integer, dimensionality of the output space.
        modules: Positive integer, number of modules.
            The dimensionality of the outputspace is a concatenation of
            all modules k with the respective units_vec[k] size.
            Default: depends on size of units_vec or 1 in case of units_vec
            being a scalar.
        t_vec: Positive float >= 1, timescale.
            Unit-dependent time constant of leakage.
        connectivity: connection scheme in case of more than one modules
            Default: `dense`
            Other options are `partitioned`, `clocked`, and `adjacent`
        activation: Activation function to use.
            Default: hyperbolic tangent (`tanh`).
            If you pass `None`, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs.
        recurrent_initializer: Initializer for the `recurrent_kernel`
            weights matrix,
            used for the linear transformation of the recurrent state.
        bias_initializer: Initializer for the bias vector.
        w_t_initializer: Initializer for the w_tau vector.
        kernel_regularizer: Regularizer function applied to the `kernel` weights matrix.
        recurrent_regularizer: Regularizer function applied to
            the `recurrent_kernel` weights matrix.
        bias_regularizer: Regularizer function applied to the bias vector.
        w_t_regularizer: Regularizer function applied to the w_tau vector.
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix.
        recurrent_constraint: Constraint function applied to
            the `recurrent_kernel` weights matrix.
        bias_constraint: Constraint function applied to the bias vector.
        w_t_constraint: Constraint function applied to the w_tau vector.
        dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the inputs.
        recurrent_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the recurrent state.
            
  Call arguments:
    inputs: A 2D tensor.
    states: List of state tensors corresponding to the previous timestep.
    training: Python boolean indicating whether the layer should behave in
      training mode or in inference mode. Only relevant when `dropout` or
      `recurrent_dropout` is used.
    """
    def __init__(self,
               units_vec,
               modules=None,
               t_vec=1.,
               connectivity='dense',
               activation='tanh',
               use_bias=True,
               kernel_initializer='glorot_uniform',
               recurrent_initializer='orthogonal',
               bias_initializer='zeros',
               w_t_initializer='zeros',
               kernel_regularizer=None,
               recurrent_regularizer=None,
               bias_regularizer=None,
               w_t_regularizer=None,
               kernel_constraint=None,
               recurrent_constraint=None,
               bias_constraint=None,
               w_t_constraint=None,
               dropout=0.,
               recurrent_dropout=0.,
               **kwargs):
        if ops.executing_eagerly_outside_functions():
          self._enable_caching_device = kwargs.pop('enable_caching_device', True)
        else:
          self._enable_caching_device = kwargs.pop('enable_caching_device', False)
        super(ACTRNNCell, self).__init__(**kwargs)
        
        self.connectivity = connectivity
        if isinstance(units_vec, list):
            self.units_vec = units_vec
            self.modules = len(units_vec)
            self.units = 0
            for k in range(self.modules):
                self.units += units_vec[k]
        else:
            self.units = units_vec
            if modules is not None and modules > 1:
                self.modules = int(modules)
                self.units_vec = [units_vec//modules for k in range(self.modules)]
            else:
                self.modules = 1
                self.units_vec = [units_vec]
                self.connectivity = 'dense'
        
        if isinstance(t_vec, list):
            if len(t_vec) != self.modules:
                raise ValueError("vector of tau must be of same size as "
                                 "modules or size of vector of units_vec")
            self.t_vec = t_vec
            self.ts = array_ops.constant(
                [[max(1., float(t_vec[k]))] for k in range(self.modules)
                 for n in range(self.units_vec[k])],
                dtype=self.dtype, shape=[self.units], name="ts")
        else:
            if self.modules > 1:
                self.t_vec = [max(1., float(t_vec)) for k in range(self.modules)]
            else:
                self.t_vec = [max(1., float(t_vec))]
            self.ts = array_ops.constant(
                max(1., t_vec), dtype=self.dtype, shape=[self.units], name="ts")

        self.activation = activations.get(activation)
        self.use_bias = use_bias
    
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.w_t_initializer = initializers.get(w_t_initializer)
    
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.w_t_regularizer = regularizers.get(w_t_regularizer)
    
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.w_t_constraint = constraints.get(w_t_constraint)
    
        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        implementation = kwargs.pop('implementation', 1)
        if self.recurrent_dropout != 0 and implementation != 1:
          logging.debug(RECURRENT_DROPOUT_WARNING_MSG)
          self.implementation = 1
        else:
          self.implementation = implementation
        self.state_size = data_structures.NoDependency([self.units, self.units])
        self.output_size = self.units
        
    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.units),
            name='kernel',
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint)
      
        if self.connectivity == 'partitioned':
            self.recurrent_kernel_vec = []
            for k in range(self.modules):
                self.recurrent_kernel_vec += [self.add_weight(
                    shape=(self.units_vec[k], self.units_vec[k]),
                    name='recurrent_kernel' + str(k),
                    initializer=self.recurrent_initializer,
                    regularizer=self.recurrent_regularizer,
                    constraint=self.recurrent_constraint)]
        elif self.connectivity == 'clocked':
            self.recurrent_kernel_vec = []
            for k in range(self.modules):
                self.recurrent_kernel_vec += [self.add_weight(
                    shape=(sum(self.units_vec[k:self.modules]),
                           self.units_vec[k]),
                    name='recurrent_kernel' + str(k),
                    initializer=self.recurrent_initializer,
                    regularizer=self.recurrent_regularizer,
                    constraint=self.recurrent_constraint)]
        elif self.connectivity == 'adjacent':
            self.recurrent_kernel_vec = []
            for k in range(self.modules):
                self.recurrent_kernel_vec += [self.add_weight(
                    shape=(sum(self.units_vec[
                           max(0, k - 1):min(self.modules, k + 1 + 1)]),
                           self.units_vec[k]),
                    name='recurrent_kernel' + str(k),
                    initializer=self.recurrent_initializer,
                    regularizer=self.recurrent_regularizer,
                    constraint=self.recurrent_constraint)]
        else:
            self.recurrent_kernel_vec = [self.add_weight(
                    shape=(self.units, self.units),
                    name='recurrent_kernel',
                    initializer=self.recurrent_initializer,
                    regularizer=self.recurrent_regularizer,
                    constraint=self.recurrent_constraint)]

        if self.use_bias:
          self.bias = self.add_weight(
              shape=(self.units,),
              name='bias',
              initializer=self.bias_initializer,
              regularizer=self.bias_regularizer,
              constraint=self.bias_constraint)
        else:
          self.bias = None
          
        self.a = self.add_weight(
            shape=(self.units,),
            name='wtimescales',
            initializer=self.w_t_initializer,
            regularizer=self.w_t_regularizer,
            constraint=self.w_t_constraint)
        self.t0 = K.log(self.ts - _ALMOST_ONE)
        self.built = True
    
    def call(self, inputs, states, training=None):
        prev_output = states[0]
        prev_z = states[1]
        dp_mask = self.get_dropout_mask_for_cell(inputs, training)
        rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(
            prev_output, training)
        
        if dp_mask is not None:
          h = K.dot(inputs * dp_mask, self.kernel)
        else:
          h = K.dot(inputs, self.kernel)
        
        if self.bias is not None:
          h = K.bias_add(h, self.bias)
    
        if rec_dp_mask is not None:
          prev_output = prev_output * rec_dp_mask
        
        prev_y_vec = array_ops.split(prev_output, self.units_vec, axis=1)
        if self.connectivity == 'partitioned':
            r = array_ops.concat([K.dot(prev_y_vec[k], self.recurrent_kernel_vec[k])
                 for k in range(self.modules)], 1)
        elif self.connectivity == 'clocked':
            r = array_ops.concat([K.dot(array_ops.concat(prev_y_vec[k:self.modules], 1),
                    self.recurrent_kernel_vec[k])
                 for k in range(self.modules)], 1)
        elif self.connectivity == 'adjacent':
            r = array_ops.concat([K.dot(array_ops.concat(
                    prev_y_vec[max(0, k - 1):min(self.modules, k + 1 + 1)], 1),
                    self.recurrent_kernel_vec[k])
                 for k in range(self.modules)], 1)
        else:
            r = K.dot(prev_output, self.recurrent_kernel_vec[0])
        
        ts_act = K.exp(self.a + self.t0) + _ALMOST_ONE
        
        z = (1. - 1. / ts_act) * prev_z + (1. / ts_act) * (h + r)
        
        if self.activation is not None:
            output = self.activation(z)
        else:
            output = z
        
        return output, [output, z]
      
    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return list(_generate_zero_filled_state_for_cell(
            self, inputs, batch_size, dtype))
    
    def get_ts(self):
        return self.ts if self.built else None
    
    def get_config(self):
        config = {
            'units_vec':
                self.units_vec,
            'modules':
                self.modules,
            't_vec':
                self.t_vec,
            'activation':
                activations.serialize(self.activation),
            'use_bias':
                self.use_bias,
            'kernel_initializer':
                initializers.serialize(self.kernel_initializer),
            'recurrent_initializer':
                initializers.serialize(self.recurrent_initializer),
            'bias_initializer':
                initializers.serialize(self.bias_initializer),
            'w_t_initializer':
                initializers.serialize(self.w_t_initializer),
            'kernel_regularizer':
                regularizers.serialize(self.kernel_regularizer),
            'recurrent_regularizer':
                regularizers.serialize(self.recurrent_regularizer),
            'bias_regularizer':
                regularizers.serialize(self.bias_regularizer),
            'w_t_regularizer':
                regularizers.serialize(self.w_t_regularizer),
            'kernel_constraint':
                constraints.serialize(self.kernel_constraint),
            'recurrent_constraint':
                constraints.serialize(self.recurrent_constraint),
            'bias_constraint':
                constraints.serialize(self.bias_constraint),
            'w_t_constraint':
                constraints.serialize(self.w_t_constraint),
            'dropout':
                self.dropout,
            'recurrent_dropout':
                self.recurrent_dropout
        }
        config.update(_config_for_enable_caching_device(self))
        base_config = super(ACTRNNCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    

@tf_export('keras.layers.ACTRNN')
class ACTRNN(RNN):

    """Adaptive Continuous Time RNN that can have several modules
       where the output is to be fed back to input.
    
    Arguments:
      units_vec: Positive integer or vector of positive integer,
            dimensionality of the output space.
      modules: Positive integer, number of modules.
          The dimensionality of the outputspace is a concatenation of
          all modules k with the respective units_vec[k] size.
          Default: depends on size of units_vec or 1 in case of units_vec
          being a scalar.
      t: Positive float >= 1, timescale.
          Unit-dependent time constant of leakage.
      connectivity: connection scheme in case of more than one modules
          Default: `dense`
          Other options are `partitioned`, `clocked`, and `adjacent`
      activation: Activation function to use.
        Default: hyperbolic tangent (`tanh`).
        If you pass None, no activation is applied
        (ie. "linear" activation: `a(x) = x`).
      use_bias: Boolean, whether the layer uses a bias vector.
      kernel_initializer: Initializer for the `kernel` weights matrix,
        used for the linear transformation of the inputs.
      recurrent_initializer: Initializer for the `recurrent_kernel`
        weights matrix,
        used for the linear transformation of the recurrent state.
      bias_initializer: Initializer for the bias vector.
      w_t_initializer: Initializer for the w_t vector.
      kernel_regularizer: Regularizer function applied to
        the `kernel` weights matrix.
      recurrent_regularizer: Regularizer function applied to
        the `recurrent_kernel` weights matrix.
      bias_regularizer: Regularizer function applied to the bias vector.
      w_t_regularizer: Regularizer function applied to the w_t vector.
      activity_regularizer: Regularizer function applied to
        the output of the layer (its "activation")..
      kernel_constraint: Constraint function applied to
        the `kernel` weights matrix.
      recurrent_constraint: Constraint function applied to
        the `recurrent_kernel` weights matrix.
      bias_constraint: Constraint function applied to the bias vector.
      w_t_constraint: Constraint function applied to the w_t vector.
      dropout: Float between 0 and 1.
        Fraction of the units to drop for
        the linear transformation of the inputs.
      recurrent_dropout: Float between 0 and 1.
        Fraction of the units to drop for
        the linear transformation of the recurrent state.
      return_sequences: Boolean. Whether to return the last output
        in the output sequence, or the full sequence.
      return_state: Boolean. Whether to return the last state
        in addition to the output.
      go_backwards: Boolean (default False).
        If True, process the input sequence backwards and return the
        reversed sequence.
      stateful: Boolean (default False). If True, the last state
        for each sample at index i in a batch will be used as initial
        state for the sample of index i in the following batch.
      unroll: Boolean (default False).
        If True, the network will be unrolled,
        else a symbolic loop will be used.
        Unrolling can speed-up a RNN,
        although it tends to be more memory-intensive.
        Unrolling is only suitable for short sequences.
    
    Call arguments:
      inputs: A 3D tensor.
      mask: Binary tensor of shape `(samples, timesteps)` indicating whether
        a given timestep should be masked.
      training: Python boolean indicating whether the layer should behave in
        training mode or in inference mode. This argument is passed to the cell
        when calling it. This is only relevant if `dropout` or
        `recurrent_dropout` is used.
      initial_state: List of initial state tensors to be passed to the first
        call of the cell.
    """
    def __init__(self,
                units_vec,
                modules=None,
                t_vec=1.,
                connectivity='dense',
                activation='tanh',
                use_bias=True,
                kernel_initializer='glorot_uniform',
                recurrent_initializer='orthogonal',
                bias_initializer='zeros',
                w_t_initializer='zeros',
                kernel_regularizer=None,
                recurrent_regularizer=None,
                bias_regularizer=None,
                w_t_regularizer=None,
                activity_regularizer=None,
                kernel_constraint=None,
                recurrent_constraint=None,
                bias_constraint=None,
                w_t_constraint=None,
                dropout=0.,
                recurrent_dropout=0.,
                return_sequences=False,
                return_state=False,
                go_backwards=False,
                stateful=False,
                unroll=False,
                **kwargs):
        if 'implementation' in kwargs:
            kwargs.pop('implementation')
            logging.warning('The `implementation` argument '
                            'in `SimpleCTRNN` has been deprecated. '
                            'Please remove it from your layer call.')
        if 'enable_caching_device' in kwargs:
            cell_kwargs = {'enable_caching_device':
                         kwargs.pop('enable_caching_device')}
        else:
            cell_kwargs = {}
        cell = ACTRNNCell(
            units_vec,
            modules=modules,
            t_vec=t_vec,
            connectivity=connectivity,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            recurrent_initializer=recurrent_initializer,
            bias_initializer=bias_initializer,
            w_t_initializer=w_t_initializer,
            kernel_regularizer=kernel_regularizer,
            recurrent_regularizer=recurrent_regularizer,
            bias_regularizer=bias_regularizer,
            w_t_regularizer=w_t_regularizer,
            kernel_constraint=kernel_constraint,
            recurrent_constraint=recurrent_constraint,
            bias_constraint=bias_constraint,
            w_t_constraint=w_t_constraint,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            dtype=kwargs.get('dtype'),
            trainable=kwargs.get('trainable', True),
            **cell_kwargs)
        super(ACTRNN, self).__init__(
            cell,
            return_sequences=return_sequences,
            return_state=return_state,
            go_backwards=go_backwards,
            stateful=stateful,
            unroll=unroll,
            **kwargs)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        
    def call(self, inputs, mask=None, training=None, initial_state=None):
        self._maybe_reset_cell_dropout_mask(self.cell)
        return super(ACTRNN, self).call(
            inputs, mask=mask, training=training, initial_state=initial_state)
    
    @property
    def units(self):
        return self.cell.units

    @property
    def units_vec(self):
        return self.cell.units_vec

    @property
    def modules(self):
        return self.cell.modules

    @property
    def t_vec(self):
        return self.cell.t_vec

    @property
    def activation(self):
      return self.cell.activation

    @property
    def use_bias(self):
      return self.cell.use_bias

    @property
    def kernel_initializer(self):
      return self.cell.kernel_initializer
    
    @property
    def recurrent_initializer(self):
      return self.cell.recurrent_initializer
    
    @property
    def bias_initializer(self):
      return self.cell.bias_initializer
    
    @property
    def kernel_regularizer(self):
      return self.cell.kernel_regularizer
  
    @property
    def w_t_initializer(self):
        return self.cell.w_t_initializer
    
    @property
    def recurrent_regularizer(self):
      return self.cell.recurrent_regularizer
    
    @property
    def bias_regularizer(self):
      return self.cell.bias_regularizer
  
    @property
    def w_t_regularizer(self):
        return self.cell.w_t_regularizer
    
    @property
    def kernel_constraint(self):
      return self.cell.kernel_constraint
    
    @property
    def recurrent_constraint(self):
      return self.cell.recurrent_constraint
    
    @property
    def bias_constraint(self):
      return self.cell.bias_constraint
  
    @property
    def w_t_constraint(self):
        return self.cell.w_t_constraint
    
    @property
    def dropout(self):
      return self.cell.dropout
    
    @property
    def recurrent_dropout(self):
      return self.cell.recurrent_dropout
  
    @property
    def get_ts(self):
        return self.cell.get_ts()

    def get_config(self):
      config = {
          'units':
              self.units,
          'units_vec':
              self.units_vec,
          'modules':
              self.modules,
          't_vec':
              self.t_vec,
          'activation':
              activations.serialize(self.activation),
          'use_bias':
              self.use_bias,
          'kernel_initializer':
              initializers.serialize(self.kernel_initializer),
          'recurrent_initializer':
              initializers.serialize(self.recurrent_initializer),
          'bias_initializer':
              initializers.serialize(self.bias_initializer),
          'w_t_initializer':
                initializers.serialize(self.w_t_initializer),
          'kernel_regularizer':
              regularizers.serialize(self.kernel_regularizer),
          'recurrent_regularizer':
              regularizers.serialize(self.recurrent_regularizer),
          'bias_regularizer':
              regularizers.serialize(self.bias_regularizer),
          'w_t_regularizer':
                regularizers.serialize(self.w_t_regularizer),
          'activity_regularizer':
              regularizers.serialize(self.activity_regularizer),
          'kernel_constraint':
              constraints.serialize(self.kernel_constraint),
          'recurrent_constraint':
              constraints.serialize(self.recurrent_constraint),
          'bias_constraint':
              constraints.serialize(self.bias_constraint),
          'w_t_constraint':
                constraints.serialize(self.w_t_constraint),
          'dropout':
              self.dropout,
          'recurrent_dropout':
              self.recurrent_dropout
      }
      base_config = super(ACTRNN, self).get_config()
      config.update(_config_for_enable_caching_device(self.cell))
      del base_config['cell']
      return dict(list(base_config.items()) + list(config.items()))
    
    @classmethod
    def from_config(cls, config):
      if 'implementation' in config:
        config.pop('implementation')
      return cls(**config)


@tf_export('keras.layers.CTRNNCell')
class CTRNNCell(DropoutRNNCellMixin, Layer):
    """Cell class for CTRNNCell.

    Arguments:
        units_vec: Positive integer, dimensionality of the output space.
        modules: Positive integer, number of modules.
            The dimensionality of the outputspace is a concatenation of
            all modules k with the respective units_vec[k] size.
            Default: depends on size of units_vec or 1 in case of units_vec
            being a scalar.
        t_vec: Positive float >= 1, timescale.
            Unit-dependent time constant of leakage.
        connectivity: connection scheme in case of more than one modules
            Default: `dense`
            Other options are `partitioned`, `clocked`, and `adjacent`
        activation: Activation function to use.
            Default: hyperbolic tangent (`tanh`).
            If you pass `None`, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs.
        recurrent_initializer: Initializer for the `recurrent_kernel`
            weights matrix,
            used for the linear transformation of the recurrent state.
        bias_initializer: Initializer for the bias vector.
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix.
        recurrent_regularizer: Regularizer function applied to
            the `recurrent_kernel` weights matrix.
        bias_regularizer: Regularizer function applied to the bias vector.
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix.
        recurrent_constraint: Constraint function applied to
            the `recurrent_kernel` weights matrix.
        bias_constraint: Constraint function applied to the bias vector.
        dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the inputs.
        recurrent_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the recurrent state.
            
  Call arguments:
    inputs: A 2D tensor.
    states: List of state tensors corresponding to the previous timestep.
    training: Python boolean indicating whether the layer should behave in
      training mode or in inference mode. Only relevant when `dropout` or
      `recurrent_dropout` is used.
    """
    def __init__(self,
               units_vec,
               modules=None,
               t_vec=1.,
               connectivity='dense',
               activation='tanh',
               use_bias=True,
               kernel_initializer='glorot_uniform',
               recurrent_initializer='orthogonal',
               bias_initializer='zeros',
               kernel_regularizer=None,
               recurrent_regularizer=None,
               bias_regularizer=None,
               kernel_constraint=None,
               recurrent_constraint=None,
               bias_constraint=None,
               dropout=0.,
               recurrent_dropout=0.,
               **kwargs):
        if ops.executing_eagerly_outside_functions():
          self._enable_caching_device = kwargs.pop('enable_caching_device', True)
        else:
          self._enable_caching_device = kwargs.pop('enable_caching_device', False)
        super(CTRNNCell, self).__init__(**kwargs)
        
        self.connectivity = connectivity
        if isinstance(units_vec, list):
            self.units_vec = units_vec
            self.modules = len(units_vec)
            self.units = 0
            for k in range(self.modules):
                self.units += units_vec[k]
        else:
            self.units = units_vec
            if modules is not None and modules > 1:
                self.modules = int(modules)
                self.units_vec = [units_vec//modules for k in range(self.modules)]
            else:
                self.modules = 1
                self.units_vec = [units_vec]
                self.connectivity = 'dense'
        
        if isinstance(t_vec, list):
            if len(t_vec) != self.modules:
                raise ValueError("vector of tau must be of same size as "
                                 "modules or size of vector of units_vec")
            self.t_vec = t_vec
            self.ts = array_ops.constant(
                [[max(1., float(t_vec[k]))] for k in range(self.modules)
                 for n in range(self.units_vec[k])],
                dtype=self.dtype, shape=[self.units], name="ts")
        else:
            if self.modules > 1:
                self.t_vec = [max(1., float(t_vec)) for k in range(self.modules)]
            else:
                self.t_vec = [max(1., float(t_vec))]
            self.ts = array_ops.constant(
                max(1., t_vec), dtype=self.dtype, shape=[self.units], name="ts")

        self.activation = activations.get(activation)
        self.use_bias = use_bias
    
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
    
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
    
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
    
        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        implementation = kwargs.pop('implementation', 1)
        if self.recurrent_dropout != 0 and implementation != 1:
          logging.debug(RECURRENT_DROPOUT_WARNING_MSG)
          self.implementation = 1
        else:
          self.implementation = implementation
        self.state_size = data_structures.NoDependency([self.units, self.units])
        self.output_size = self.units
        
    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.units),
            name='kernel',
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint)
      
        if self.connectivity == 'partitioned':
            self.recurrent_kernel_vec = []
            for k in range(self.modules):
                self.recurrent_kernel_vec += [self.add_weight(
                    shape=(self.units_vec[k], self.units_vec[k]),
                    name='recurrent_kernel' + str(k),
                    initializer=self.recurrent_initializer,
                    regularizer=self.recurrent_regularizer,
                    constraint=self.recurrent_constraint)]
        elif self.connectivity == 'clocked':
            self.recurrent_kernel_vec = []
            for k in range(self.modules):
                self.recurrent_kernel_vec += [self.add_weight(
                    shape=(sum(self.units_vec[k:self.modules]),
                           self.units_vec[k]),
                    name='recurrent_kernel' + str(k),
                    initializer=self.recurrent_initializer,
                    regularizer=self.recurrent_regularizer,
                    constraint=self.recurrent_constraint)]
        elif self.connectivity == 'adjacent':
            self.recurrent_kernel_vec = []
            for k in range(self.modules):
                self.recurrent_kernel_vec += [self.add_weight(
                    shape=(sum(self.units_vec[
                           max(0, k - 1):min(self.modules, k + 1 + 1)]),
                           self.units_vec[k]),
                    name='recurrent_kernel' + str(k),
                    initializer=self.recurrent_initializer,
                    regularizer=self.recurrent_regularizer,
                    constraint=self.recurrent_constraint)]
        else:
            self.recurrent_kernel_vec = [self.add_weight(
                    shape=(self.units, self.units),
                    name='recurrent_kernel',
                    initializer=self.recurrent_initializer,
                    regularizer=self.recurrent_regularizer,
                    constraint=self.recurrent_constraint)]

        if self.use_bias:
          self.bias = self.add_weight(
              shape=(self.units,),
              name='bias',
              initializer=self.bias_initializer,
              regularizer=self.bias_regularizer,
              constraint=self.bias_constraint)
        else:
          self.bias = None
        self.built = True
    
    def call(self, inputs, states, training=None):
        prev_output = states[0]
        prev_z = states[1]
        dp_mask = self.get_dropout_mask_for_cell(inputs, training)
        rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(
            prev_output, training)
        
        if dp_mask is not None:
          h = K.dot(inputs * dp_mask, self.kernel)
        else:
          h = K.dot(inputs, self.kernel)
        
        if self.bias is not None:
          h = K.bias_add(h, self.bias)
    
        if rec_dp_mask is not None:
          prev_output = prev_output * rec_dp_mask
        
        prev_y_vec = array_ops.split(prev_output, self.units_vec, axis=1)
        if self.connectivity == 'partitioned':
            r = array_ops.concat([K.dot(prev_y_vec[k], self.recurrent_kernel_vec[k])
                 for k in range(self.modules)], 1)
        elif self.connectivity == 'clocked':
            r = array_ops.concat([K.dot(array_ops.concat(prev_y_vec[k:self.modules], 1),
                    self.recurrent_kernel_vec[k])
                 for k in range(self.modules)], 1)
        elif self.connectivity == 'adjacent':
            r = array_ops.concat([K.dot(array_ops.concat(
                    prev_y_vec[max(0, k - 1):min(self.modules, k + 1 + 1)], 1),
                    self.recurrent_kernel_vec[k])
                 for k in range(self.modules)], 1)
        else:
            r = K.dot(prev_output, self.recurrent_kernel_vec[0])
        
        z = (1. - 1. / self.ts) * prev_z + (1. / self.ts) * (h + r)
        
        if self.activation is not None:
            output = self.activation(z)
        else:
            output = z
        
        return output, [output, z]
      
    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return list(_generate_zero_filled_state_for_cell(
            self, inputs, batch_size, dtype))
    
    def get_ts(self):
        return self.ts if self.built else None
    
    def get_config(self):
        config = {
            'units_vec':
                self.units_vec,
            'modules':
                self.modules,
            't_vec':
                self.t_vec,
            'activation':
                activations.serialize(self.activation),
            'use_bias':
                self.use_bias,
            'kernel_initializer':
                initializers.serialize(self.kernel_initializer),
            'recurrent_initializer':
                initializers.serialize(self.recurrent_initializer),
            'bias_initializer':
                initializers.serialize(self.bias_initializer),
            'kernel_regularizer':
                regularizers.serialize(self.kernel_regularizer),
            'recurrent_regularizer':
                regularizers.serialize(self.recurrent_regularizer),
            'bias_regularizer':
                regularizers.serialize(self.bias_regularizer),
            'kernel_constraint':
                constraints.serialize(self.kernel_constraint),
            'recurrent_constraint':
                constraints.serialize(self.recurrent_constraint),
            'bias_constraint':
                constraints.serialize(self.bias_constraint),
            'dropout':
                self.dropout,
            'recurrent_dropout':
                self.recurrent_dropout
        }
        config.update(_config_for_enable_caching_device(self))
        base_config = super(CTRNNCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


@tf_export('keras.layers.CTRNN')
class CTRNN(RNN):
    """Fully-connected RNN where the output is to be fed back to input.
    
    Arguments:
      units_vec: Positive integer or vector of positive integer,
            dimensionality of the output space.
      modules: Positive integer, number of modules.
          The dimensionality of the outputspace is a concatenation of
          all modules k with the respective units_vec[k] size.
          Default: depends on size of units_vec or 1 in case of units_vec
          being a scalar.
      t: Positive float >= 1, timescale.
          Unit-dependent time constant of leakage.
      connectivity: connection scheme in case of more than one modules
          Default: `dense`
          Other options are `partitioned`, `clocked`, and `adjacent`
      activation: Activation function to use.
        Default: hyperbolic tangent (`tanh`).
        If you pass None, no activation is applied
        (ie. "linear" activation: `a(x) = x`).
      use_bias: Boolean, whether the layer uses a bias vector.
      kernel_initializer: Initializer for the `kernel` weights matrix,
        used for the linear transformation of the inputs.
      recurrent_initializer: Initializer for the `recurrent_kernel`
        weights matrix,
        used for the linear transformation of the recurrent state.
      bias_initializer: Initializer for the bias vector.
      kernel_regularizer: Regularizer function applied to
        the `kernel` weights matrix.
      recurrent_regularizer: Regularizer function applied to
        the `recurrent_kernel` weights matrix.
      bias_regularizer: Regularizer function applied to the bias vector.
      activity_regularizer: Regularizer function applied to
        the output of the layer (its "activation")..
      kernel_constraint: Constraint function applied to
        the `kernel` weights matrix.
      recurrent_constraint: Constraint function applied to
        the `recurrent_kernel` weights matrix.
      bias_constraint: Constraint function applied to the bias vector.
      dropout: Float between 0 and 1.
        Fraction of the units to drop for
        the linear transformation of the inputs.
      recurrent_dropout: Float between 0 and 1.
        Fraction of the units to drop for
        the linear transformation of the recurrent state.
      return_sequences: Boolean. Whether to return the last output
        in the output sequence, or the full sequence.
      return_state: Boolean. Whether to return the last state
        in addition to the output.
      go_backwards: Boolean (default False).
        If True, process the input sequence backwards and return the
        reversed sequence.
      stateful: Boolean (default False). If True, the last state
        for each sample at index i in a batch will be used as initial
        state for the sample of index i in the following batch.
      unroll: Boolean (default False).
        If True, the network will be unrolled,
        else a symbolic loop will be used.
        Unrolling can speed-up a RNN,
        although it tends to be more memory-intensive.
        Unrolling is only suitable for short sequences.
    
    Call arguments:
      inputs: A 3D tensor.
      mask: Binary tensor of shape `(samples, timesteps)` indicating whether
        a given timestep should be masked.
      training: Python boolean indicating whether the layer should behave in
        training mode or in inference mode. This argument is passed to the cell
        when calling it. This is only relevant if `dropout` or
        `recurrent_dropout` is used.
      initial_state: List of initial state tensors to be passed to the first
        call of the cell.
    """
    def __init__(self,
                units_vec,
                modules=None,
                t_vec=1.,
                connectivity='dense',
                activation='tanh',
                use_bias=True,
                kernel_initializer='glorot_uniform',
                recurrent_initializer='orthogonal',
                bias_initializer='zeros',
                kernel_regularizer=None,
                recurrent_regularizer=None,
                bias_regularizer=None,
                activity_regularizer=None,
                kernel_constraint=None,
                recurrent_constraint=None,
                bias_constraint=None,
                dropout=0.,
                recurrent_dropout=0.,
                return_sequences=False,
                return_state=False,
                go_backwards=False,
                stateful=False,
                unroll=False,
                **kwargs):
        if 'implementation' in kwargs:
            kwargs.pop('implementation')
            logging.warning('The `implementation` argument '
                            'in `SimpleCTRNN` has been deprecated. '
                            'Please remove it from your layer call.')
        if 'enable_caching_device' in kwargs:
            cell_kwargs = {'enable_caching_device':
                         kwargs.pop('enable_caching_device')}
        else:
            cell_kwargs = {}
        cell = CTRNNCell(
            units_vec,
            modules=modules,
            t_vec=t_vec,
            connectivity=connectivity,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            recurrent_initializer=recurrent_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            recurrent_regularizer=recurrent_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_constraint=kernel_constraint,
            recurrent_constraint=recurrent_constraint,
            bias_constraint=bias_constraint,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            dtype=kwargs.get('dtype'),
            trainable=kwargs.get('trainable', True),
            **cell_kwargs)
        super(CTRNN, self).__init__(
            cell,
            return_sequences=return_sequences,
            return_state=return_state,
            go_backwards=go_backwards,
            stateful=stateful,
            unroll=unroll,
            **kwargs)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        
    def call(self, inputs, mask=None, training=None, initial_state=None):
        self._maybe_reset_cell_dropout_mask(self.cell)
        return super(CTRNN, self).call(
            inputs, mask=mask, training=training, initial_state=initial_state)
    
    @property
    def units(self):
        return self.cell.units

    @property
    def units_vec(self):
        return self.cell.units_vec

    @property
    def modules(self):
        return self.cell.modules

    @property
    def t_vec(self):
        return self.cell.t_vec

    @property
    def activation(self):
      return self.cell.activation

    @property
    def use_bias(self):
      return self.cell.use_bias

    @property
    def kernel_initializer(self):
      return self.cell.kernel_initializer
    
    @property
    def recurrent_initializer(self):
      return self.cell.recurrent_initializer
    
    @property
    def bias_initializer(self):
      return self.cell.bias_initializer
    
    @property
    def kernel_regularizer(self):
      return self.cell.kernel_regularizer
    
    @property
    def recurrent_regularizer(self):
      return self.cell.recurrent_regularizer
    
    @property
    def bias_regularizer(self):
      return self.cell.bias_regularizer
    
    @property
    def kernel_constraint(self):
      return self.cell.kernel_constraint
    
    @property
    def recurrent_constraint(self):
      return self.cell.recurrent_constraint
    
    @property
    def bias_constraint(self):
      return self.cell.bias_constraint
    
    @property
    def dropout(self):
      return self.cell.dropout
    
    @property
    def recurrent_dropout(self):
      return self.cell.recurrent_dropout
  
    @property
    def get_ts(self):
        return self.cell.get_ts()

    def get_config(self):
      config = {
          'units':
              self.units,
          'units_vec':
              self.units_vec,
          'modules':
              self.modules,
          't_vec':
              self.t_vec,
          'activation':
              activations.serialize(self.activation),
          'use_bias':
              self.use_bias,
          'kernel_initializer':
              initializers.serialize(self.kernel_initializer),
          'recurrent_initializer':
              initializers.serialize(self.recurrent_initializer),
          'bias_initializer':
              initializers.serialize(self.bias_initializer),
          'kernel_regularizer':
              regularizers.serialize(self.kernel_regularizer),
          'recurrent_regularizer':
              regularizers.serialize(self.recurrent_regularizer),
          'bias_regularizer':
              regularizers.serialize(self.bias_regularizer),
          'activity_regularizer':
              regularizers.serialize(self.activity_regularizer),
          'kernel_constraint':
              constraints.serialize(self.kernel_constraint),
          'recurrent_constraint':
              constraints.serialize(self.recurrent_constraint),
          'bias_constraint':
              constraints.serialize(self.bias_constraint),
          'dropout':
              self.dropout,
          'recurrent_dropout':
              self.recurrent_dropout
      }
      base_config = super(CTRNN, self).get_config()
      config.update(_config_for_enable_caching_device(self.cell))
      del base_config['cell']
      return dict(list(base_config.items()) + list(config.items()))
    
    @classmethod
    def from_config(cls, config):
      if 'implementation' in config:
        config.pop('implementation')
      return cls(**config)


@tf_export('keras.layers.GACTRNNCell')
class GACTRNNCell(DropoutRNNCellMixin, Layer):
    """Cell class for GACTRNN layer.

    Arguments:
        units_vec: Positive integer, dimensionality of the output space.
        modules: Positive integer, number of modules.
            The dimensionality of the outputspace is a concatenation of
            all modules k with the respective units_vec[k] size.
            Default: depends on size of units_vec or 1 in case of units_vec
            being a scalar.
        t_vec: Positive float >= 1, timescale.
            Unit-dependent time constant of leakage.
        connectivity: connection scheme in case of more than one modules
            Default: `dense`
            Other options are `partitioned`, `clocked`, and `adjacent`
        activation: Activation function to use.
            Default: hyperbolic tangent (`tanh`).
            If you pass `None`, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs.
        recurrent_initializer: Initializer for the `recurrent_kernel`
            weights matrix,
            used for the linear transformation of the recurrent state.
        bias_initializer: Initializer for the bias vector.
        t_kernel_initializer: Initializer for the t_kernel vector.
        t_recurrent_initializer: Initializer for the t_recurrent vector.
        t_bias_initializer: Initializer for the t_bias vector.
        kernel_regularizer: Regularizer function applied to the `kernel` weights matrix.
        recurrent_regularizer: Regularizer function applied to
            the `recurrent_kernel` weights matrix.
        bias_regularizer: Regularizer function applied to the bias vector.
        t_kernel_regularizer: Regularizer function applied to the t_kernel vector.
        t_recurrent_regularizer: Regularizer function applied to the t_recurrent vector.
        t_bias_regularizer: Regularizer function applied to the t_bias vector.
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix.
        recurrent_constraint: Constraint function applied to
            the `recurrent_kernel` weights matrix.
        bias_constraint: Constraint function applied to the bias vector.
        t_kernel_constraint: Constraint function applied to the t_kernel vector.
        t_recurrent_constraint: Constraint function applied to the t_recurrent vector.
        t_bias_constraint: Constraint function applied to the t_bias vector.
        dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the inputs.
        recurrent_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the recurrent state.
            
  Call arguments:
    inputs: A 2D tensor.
    states: List of state tensors corresponding to the previous timestep.
    training: Python boolean indicating whether the layer should behave in
      training mode or in inference mode. Only relevant when `dropout` or
      `recurrent_dropout` is used.
    """
    def __init__(self,
               units_vec,
               modules=None,
               t_vec=1.,
               connectivity='dense',
               activation='tanh',
               use_bias=True,
               kernel_initializer='glorot_uniform',
               recurrent_initializer='orthogonal',
               bias_initializer='zeros',
               t_kernel_initializer='zeros',
               t_recurrent_initializer='zeros',
               t_bias_initializer='zeros',
               kernel_regularizer=None,
               recurrent_regularizer=None,
               bias_regularizer=None,
               t_kernel_regularizer=None,
               t_recurrent_regularizer=None,
               t_bias_regularizer=None,
               kernel_constraint=None,
               recurrent_constraint=None,
               bias_constraint=None,
               t_kernel_constraint=None,
               t_recurrent_constraint=None,
               t_bias_constraint=None,
               dropout=0.,
               recurrent_dropout=0.,
               **kwargs):
        if ops.executing_eagerly_outside_functions():
          self._enable_caching_device = kwargs.pop('enable_caching_device', True)
        else:
          self._enable_caching_device = kwargs.pop('enable_caching_device', False)
        super(GACTRNNCell, self).__init__(**kwargs)
        
        self.connectivity = connectivity
        if isinstance(units_vec, list):
            self.units_vec = units_vec
            self.modules = len(units_vec)
            self.units = 0
            for k in range(self.modules):
                self.units += units_vec[k]
        else:
            self.units = units_vec
            if modules is not None and modules > 1:
                self.modules = int(modules)
                self.units_vec = [units_vec//modules for k in range(self.modules)]
            else:
                self.modules = 1
                self.units_vec = [units_vec]
                self.connectivity = 'dense'
        
        if isinstance(t_vec, list):
            if len(t_vec) != self.modules:
                raise ValueError("vector of tau must be of same size as "
                                 "modules or size of vector of units_vec")
            self.t_vec = t_vec
            self.ts = array_ops.constant(
                [[max(1., float(t_vec[k]))] for k in range(self.modules)
                 for n in range(self.units_vec[k])],
                dtype=self.dtype, shape=[self.units], name="ts")
        else:
            if self.modules > 1:
                self.t_vec = [max(1., float(t_vec)) for k in range(self.modules)]
            else:
                self.t_vec = [max(1., float(t_vec))]
            self.ts = array_ops.constant(
                max(1., t_vec), dtype=self.dtype, shape=[self.units], name="ts")

        self.activation = activations.get(activation)
        self.use_bias = use_bias
    
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.t_kernel_initializer = initializers.get(t_kernel_initializer)
        self.t_recurrent_initializer = initializers.get(t_recurrent_initializer)
        self.t_bias_initializer = initializers.get(t_bias_initializer)
    
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.t_kernel_regularizer = regularizers.get(t_kernel_regularizer)
        self.t_recurrent_regularizer = regularizers.get(t_recurrent_regularizer)
        self.t_bias_regularizer = regularizers.get(t_bias_regularizer)
    
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.t_kernel_constraint = constraints.get(t_kernel_constraint)
        self.t_recurrent_constraint = constraints.get(t_recurrent_constraint)
        self.t_bias_constraint = constraints.get(t_bias_constraint)
    
        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        implementation = kwargs.pop('implementation', 1)
        if self.recurrent_dropout != 0 and implementation != 1:
          logging.debug(RECURRENT_DROPOUT_WARNING_MSG)
          self.implementation = 1
        else:
          self.implementation = implementation
        self.state_size = data_structures.NoDependency([self.units, self.units])
        self.output_size = self.units
        
    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.units),
            name='kernel',
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint)
      
        if self.connectivity == 'partitioned':
            self.recurrent_kernel_vec = []
            self.t_recurrent = []
            for k in range(self.modules):
                self.recurrent_kernel_vec += [self.add_weight(
                    shape=(self.units_vec[k], self.units_vec[k]),
                    name='recurrent_kernel' + str(k),
                    initializer=self.recurrent_initializer,
                    regularizer=self.recurrent_regularizer,
                    constraint=self.recurrent_constraint)]
                self.t_recurrent += [self.add_weight(
                    shape=(self.units_vec[k], self.units_vec[k]),
                    name='tau_recurrent' + str(k),
                    initializer=self.t_recurrent_initializer,
                    regularizer=self.t_recurrent_regularizer,
                    constraint=self.t_recurrent_constraint)]
        elif self.connectivity == 'clocked':
            self.recurrent_kernel_vec = []
            self.t_recurrent = []
            for k in range(self.modules):
                self.recurrent_kernel_vec += [self.add_weight(
                    shape=(sum(self.units_vec[k:self.modules]),
                           self.units_vec[k]),
                    name='recurrent_kernel' + str(k),
                    initializer=self.recurrent_initializer,
                    regularizer=self.recurrent_regularizer,
                    constraint=self.recurrent_constraint)]
                self.t_recurrent += [self.add_weight(
                    shape=(sum(self.units_vec[k:self.modules]),
                           self.units_vec[k]),
                    name='tau_recurrent' + str(k),
                    initializer=self.t_recurrent_initializer,
                    regularizer=self.t_recurrent_regularizer,
                    constraint=self.t_recurrent_constraint)]
        elif self.connectivity == 'adjacent':
            self.recurrent_kernel_vec = []
            self.t_recurrent = []
            for k in range(self.modules):
                self.recurrent_kernel_vec += [self.add_weight(
                    shape=(sum(self.units_vec[
                           max(0, k - 1):min(self.modules, k + 1 + 1)]),
                           self.units_vec[k]),
                    name='recurrent_kernel' + str(k),
                    initializer=self.recurrent_initializer,
                    regularizer=self.recurrent_regularizer,
                    constraint=self.recurrent_constraint)]
                self.t_recurrent += [self.add_weight(
                    shape=(sum(self.units_vec[
                               max(0, k - 1):min(self.modules, k + 1 + 1)]),
                           self.units_vec[k]),
                    name='tau_recurrent' + str(k),
                    initializer=self.t_recurrent_initializer,
                    regularizer=self.t_recurrent_regularizer,
                    constraint=self.t_recurrent_constraint)]
        else:
            self.recurrent_kernel_vec = [self.add_weight(
                    shape=(self.units, self.units),
                    name='recurrent_kernel',
                    initializer=self.recurrent_initializer,
                    regularizer=self.recurrent_regularizer,
                    constraint=self.recurrent_constraint)]
            self.t_recurrent = [self.add_weight(
                    shape=(self.units, self.units),
                    name='tau_recurrent',
                    initializer=self.t_recurrent_initializer,
                    regularizer=self.t_recurrent_regularizer,
                    constraint=self.t_recurrent_constraint)]

        if self.use_bias:
          self.bias = self.add_weight(
              shape=(self.units,),
              name='bias',
              initializer=self.bias_initializer,
              regularizer=self.bias_regularizer,
              constraint=self.bias_constraint)
        else:
          self.bias = None
          
        self.t_kernel = self.add_weight(
            shape=(input_shape[-1], self.units),
            name='tau_kernel',
            initializer=self.t_kernel_initializer,
            regularizer=self.t_kernel_regularizer,
            constraint=self.t_kernel_constraint)
        
        self.t_bias = self.add_weight(
            shape=(self.units,),
            name='tau_bias',
            initializer=self.t_bias_initializer,
            regularizer=self.t_bias_regularizer,
            constraint=self.t_bias_constraint)
        
        self.t0 = K.log(self.ts - _ALMOST_ONE)
        self.built = True
    
    def call(self, inputs, states, training=None):
        prev_output = states[0]
        prev_z = states[1]
        
        dp_mask = self.get_dropout_mask_for_cell(inputs, training)
        rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(
            prev_output, training)
        
        if dp_mask is not None:
          h = K.dot(inputs * dp_mask, self.kernel)
        else:
          h = K.dot(inputs, self.kernel)
        
        if self.bias is not None:
          h = K.bias_add(h, self.bias)
    
        if rec_dp_mask is not None:
          prev_output = prev_output * rec_dp_mask
        
        prev_y_vec = array_ops.split(prev_output, self.units_vec, axis=1)
        if self.connectivity == 'partitioned':
            r = array_ops.concat([K.dot(prev_y_vec[k], self.recurrent_kernel_vec[k])
                 for k in range(self.modules)], 1)
            t_r = array_ops.concat(
                [K.dot(prev_y_vec[k], self.t_recurrent[k])
                 for k in range(self.modules)], 1)
        elif self.connectivity == 'clocked':
            r = array_ops.concat([K.dot(array_ops.concat(prev_y_vec[k:self.modules], 1),
                    self.recurrent_kernel_vec[k])
                 for k in range(self.modules)], 1)
            t_r = array_ops.concat(
                [K.dot(array_ops.concat(
                    prev_y_vec[k:self.modules], 1),
                    self.t_recurrent[k])
                    for k in range(self.modules)], 1)
        elif self.connectivity == 'adjacent':
            r = array_ops.concat([K.dot(array_ops.concat(
                    prev_y_vec[max(0, k - 1):min(self.modules, k + 1 + 1)], 1),
                    self.recurrent_kernel_vec[k])
                 for k in range(self.modules)], 1)
            t_r = array_ops.concat(
                [K.dot(array_ops.concat(
                    prev_y_vec[max(0, k - 1):min(self.modules, k + 1 + 1)], 1),
                    self.t_recurrent[k])
                    for k in range(self.modules)], 1)
        else:
            r = K.dot(prev_output, self.recurrent_kernel_vec[0])
            t_r = K.dot(prev_output, self.t_recurrent[0])
        
        ts_act = K.exp(self.t_bias + K.dot(inputs, self.t_kernel) + t_r + self.t0) + _ALMOST_ONE
        z = (1. - 1. / ts_act) * prev_z + (1. / ts_act) * (h + r)
        
        if self.activation is not None:
            output = self.activation(z)
        else:
            output = z
        
        return output, [output, z]
      
    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return list(_generate_zero_filled_state_for_cell(
            self, inputs, batch_size, dtype))
    
    def get_ts(self):
        return self.ts if self.built else None
    
    def get_config(self):
        config = {
            'units_vec':
                self.units_vec,
            'modules':
                self.modules,
            't_vec':
                self.t_vec,
            'activation':
                activations.serialize(self.activation),
            'use_bias':
                self.use_bias,
            'kernel_initializer':
                initializers.serialize(self.kernel_initializer),
            'recurrent_initializer':
                initializers.serialize(self.recurrent_initializer),
            'bias_initializer':
                initializers.serialize(self.bias_initializer),
            't_kernel_initializer':
                initializers.serialize(self.t_kernel_initializer),
            't_recurrent_initializer':
                initializers.serialize(self.t_recurrent_initializer),
            't_bias_initializer':
                initializers.serialize(self.t_bias_initializer),
            'kernel_regularizer':
                regularizers.serialize(self.kernel_regularizer),
            'recurrent_regularizer':
                regularizers.serialize(self.recurrent_regularizer),
            'bias_regularizer':
                regularizers.serialize(self.bias_regularizer),
            't_kernel_regularizer':
                regularizers.serialize(self.t_kernel_regularizer),
            't_recurrent_regularizer':
                regularizers.serialize(self.t_recurrent_regularizer),
            't_bias_regularizer':
                regularizers.serialize(self.t_bias_regularizer),
            'kernel_constraint':
                constraints.serialize(self.kernel_constraint),
            'recurrent_constraint':
                constraints.serialize(self.recurrent_constraint),
            'bias_constraint':
                constraints.serialize(self.bias_constraint),
            't_kernel_constraint':
                constraints.serialize(self.t_kernel_constraint),
            't_recurrent_constraint':
                constraints.serialize(self.t_recurrent_constraint),
            't_bias_constraint':
                constraints.serialize(self.t_bias_constraint),
            'dropout':
                self.dropout,
            'recurrent_dropout':
                self.recurrent_dropout
        }
        config.update(_config_for_enable_caching_device(self))
        base_config = super(GACTRNNCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


@tf_export('keras.layers.GACTRNN')
class GACTRNN(RNN):
    """Timescale Gated Continuous Time RNN that can have several modules
       where the output is to be fed back to input.
    
    Arguments:
      units_vec: Positive integer or vector of positive integer,
            dimensionality of the output space.
      modules: Positive integer, number of modules.
          The dimensionality of the outputspace is a concatenation of
          all modules k with the respective units_vec[k] size.
          Default: depends on size of units_vec or 1 in case of units_vec
          being a scalar.
      t: Positive float >= 1, timescale.
          Unit-dependent time constant of leakage.
      connectivity: connection scheme in case of more than one modules
          Default: `dense`
          Other options are `partitioned`, `clocked`, and `adjacent`
      activation: Activation function to use.
        Default: hyperbolic tangent (`tanh`).
        If you pass None, no activation is applied
        (ie. "linear" activation: `a(x) = x`).
      use_bias: Boolean, whether the layer uses a bias vector.
      kernel_initializer: Initializer for the `kernel` weights matrix,
        used for the linear transformation of the inputs.
      recurrent_initializer: Initializer for the `recurrent_kernel`
        weights matrix,
        used for the linear transformation of the recurrent state.
      bias_initializer: Initializer for the bias vector.
      t_kernel_initializer: Initializer for the tau_kernel vector.
      t_recurrent_initializer: Initializer for the tau_recurrent vector.
      t_bias_initializer: Initializer for the t_bias vector.
      kernel_regularizer: Regularizer function applied to
        the `kernel` weights matrix.
      recurrent_regularizer: Regularizer function applied to
        the `recurrent_kernel` weights matrix.
      bias_regularizer: Regularizer function applied to the bias vector.
      t_kernel_regularizer: Regularizer function applied to the tau_kernel vector.
      t_recurrent_regularizer: Regularizer function applied to the tau_recurrent vector.
      t_bias_regularizer: Regularizer function applied to the t_bias vector.
      activity_regularizer: Regularizer function applied to
        the output of the layer (its "activation")..
      kernel_constraint: Constraint function applied to
        the `kernel` weights matrix.
      recurrent_constraint: Constraint function applied to
        the `recurrent_kernel` weights matrix.
      bias_constraint: Constraint function applied to the bias vector.
      t_kernel_constraint: Constraint function applied to the tau_kernel vector.
      t_recurrent_constraint: Constraint function applied to the tau_recurrent vector.
      t_bias_constraint: Constraint function applied to the t_bias vector.
      dropout: Float between 0 and 1.
        Fraction of the units to drop for
        the linear transformation of the inputs.
      recurrent_dropout: Float between 0 and 1.
        Fraction of the units to drop for
        the linear transformation of the recurrent state.
      return_sequences: Boolean. Whether to return the last output
        in the output sequence, or the full sequence.
      return_state: Boolean. Whether to return the last state
        in addition to the output.
      go_backwards: Boolean (default False).
        If True, process the input sequence backwards and return the
        reversed sequence.
      stateful: Boolean (default False). If True, the last state
        for each sample at index i in a batch will be used as initial
        state for the sample of index i in the following batch.
      unroll: Boolean (default False).
        If True, the network will be unrolled,
        else a symbolic loop will be used.
        Unrolling can speed-up a RNN,
        although it tends to be more memory-intensive.
        Unrolling is only suitable for short sequences.
    
    Call arguments:
      inputs: A 3D tensor.
      mask: Binary tensor of shape `(samples, timesteps)` indicating whether
        a given timestep should be masked.
      training: Python boolean indicating whether the layer should behave in
        training mode or in inference mode. This argument is passed to the cell
        when calling it. This is only relevant if `dropout` or
        `recurrent_dropout` is used.
      initial_state: List of initial state tensors to be passed to the first
        call of the cell.
    """
    def __init__(self,
                units_vec,
                modules=1,
                t_vec=1.,
                connectivity='dense',
                activation='tanh',
                use_bias=True,
                kernel_initializer='glorot_uniform',
                recurrent_initializer='orthogonal',
                bias_initializer='zeros',
                t_bias_initializer='zeros',
                t_kernel_initializer='zeros',
                t_recurrent_initializer='zeros',
                kernel_regularizer=None,
                recurrent_regularizer=None,
                bias_regularizer=None,
                t_kernel_regularizer=None,
                t_recurrent_regularizer=None,
                t_bias_regularizer=None,
                activity_regularizer=None,
                kernel_constraint=None,
                recurrent_constraint=None,
                bias_constraint=None,
                t_kernel_constraint=None,
                t_recurrent_constraint=None,
                t_bias_constraint=None,
                dropout=0.,
                recurrent_dropout=0.,
                return_sequences=False,
                return_state=False,
                go_backwards=False,
                stateful=False,
                unroll=False,
                **kwargs):
        implementation = kwargs.pop('implementation', 1)
        if implementation == 0:
          logging.warning('`implementation=0` has been deprecated, '
                          'and now defaults to `implementation=1`.'
                          'Please update your layer call.')
        if 'enable_caching_device' in kwargs:
          cell_kwargs = {'enable_caching_device':
                         kwargs.pop('enable_caching_device')}
        else:
            cell_kwargs = {}
        cell = GACTRNNCell(
            units_vec,
            modules=modules,
            t_vec=t_vec,
            connectivity=connectivity,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            recurrent_initializer=recurrent_initializer,
            t_bias_initializer=t_bias_initializer,
            bias_initializer=bias_initializer,
            t_kernel_initializer=t_kernel_initializer,
            t_recurrent_initializer=t_recurrent_initializer,
            kernel_regularizer=kernel_regularizer,
            recurrent_regularizer=recurrent_regularizer,
            bias_regularizer=bias_regularizer,
            t_kernel_regularizer=t_kernel_regularizer,
            t_recurrent_regularizer=t_recurrent_regularizer,
            t_bias_regularizer=t_bias_regularizer,
            kernel_constraint=kernel_constraint,
            recurrent_constraint=recurrent_constraint,
            bias_constraint=bias_constraint,
            t_kernel_constraint=t_kernel_constraint,
            t_recurrent_constraint=t_recurrent_constraint,
            t_bias_constraint=t_bias_constraint,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            dtype=kwargs.get('dtype'),
            trainable=kwargs.get('trainable', True),
            **cell_kwargs)
        super(GACTRNN, self).__init__(
            cell,
            return_sequences=return_sequences,
            return_state=return_state,
            go_backwards=go_backwards,
            stateful=stateful,
            unroll=unroll,
            **kwargs)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.input_spec = [InputSpec(ndim=3)]
        
    def call(self, inputs, mask=None, training=None, initial_state=None):
        self._maybe_reset_cell_dropout_mask(self.cell)
        return super(GACTRNN, self).call(
            inputs, mask=mask, training=training, initial_state=initial_state)
    
    @property
    def units(self):
        return self.cell.units

    @property
    def units_vec(self):
        return self.cell.units_vec

    @property
    def modules(self):
        return self.cell.modules

    @property
    def t_vec(self):
        return self.cell.t_vec

    @property
    def activation(self):
      return self.cell.activation

    @property
    def use_bias(self):
      return self.cell.use_bias

    @property
    def kernel_initializer(self):
      return self.cell.kernel_initializer
    
    @property
    def recurrent_initializer(self):
      return self.cell.recurrent_initializer
    
    @property
    def bias_initializer(self):
      return self.cell.bias_initializer
    
    @property
    def kernel_regularizer(self):
      return self.cell.kernel_regularizer
  
    @property
    def t_kernel_initializer(self):
        return self.cell.t_kernel_initializer

    @property
    def t_recurrent_initializer(self):
        return self.cell.t_recurrent_initializer
    
    @property
    def t_bias_initializer(self):
        return self.cell.t_bias_initializer
    
    @property
    def recurrent_regularizer(self):
      return self.cell.recurrent_regularizer
    
    @property
    def bias_regularizer(self):
      return self.cell.bias_regularizer
  
    @property
    def t_kernel_regularizer(self):
        return self.cell.t_kernel_regularizer

    @property
    def t_recurrent_regularizer(self):
        return self.cell.t_recurrent_regularizer
    
    @property
    def t_bias_regularizer(self):
        return self.cell.t_bias_regularizer
    
    @property
    def kernel_constraint(self):
      return self.cell.kernel_constraint
    
    @property
    def recurrent_constraint(self):
      return self.cell.recurrent_constraint
    
    @property
    def bias_constraint(self):
      return self.cell.bias_constraint
  
    @property
    def t_kernel_constraint(self):
        return self.cell.t_kernel_constraint

    @property
    def t_recurrent_constraint(self):
        return self.cell.t_recurrent_constraint
    
    @property
    def t_bias_constraint(self):
        return self.cell.t_bias_constraint
    
    @property
    def dropout(self):
      return self.cell.dropout
    
    @property
    def recurrent_dropout(self):
      return self.cell.recurrent_dropout
  
    @property
    def get_ts(self):
        return self.cell.get_ts()

    def get_config(self):
      config = {
          'units':
              self.units,
          'units_vec':
              self.units_vec,
          'modules':
              self.modules,
          't_vec':
              self.t_vec,
          'activation':
              activations.serialize(self.activation),
          'use_bias':
              self.use_bias,
          'kernel_initializer':
              initializers.serialize(self.kernel_initializer),
          'recurrent_initializer':
              initializers.serialize(self.recurrent_initializer),
          'bias_initializer':
              initializers.serialize(self.bias_initializer),
          't_kernel_initializer':
              initializers.serialize(self.t_kernel_initializer),
          't_recurrent_initializer':
              initializers.serialize(self.t_recurrent_initializer),
          't_bias_initializer':
                initializers.serialize(self.t_bias_initializer),
          'kernel_regularizer':
              regularizers.serialize(self.kernel_regularizer),
          'recurrent_regularizer':
              regularizers.serialize(self.recurrent_regularizer),
          'bias_regularizer':
              regularizers.serialize(self.bias_regularizer),
          't_bias_regularizer':
                regularizers.serialize(self.t_bias_regularizer),
          't_kernels_regularizer':
              regularizers.serialize(self.t_kernel_regularizer),
          't_recurrent_regularizer':
              regularizers.serialize(self.t_recurrent_regularizer),
          'activity_regularizer':
              regularizers.serialize(self.activity_regularizer),
          'kernel_constraint':
              constraints.serialize(self.kernel_constraint),
          'recurrent_constraint':
              constraints.serialize(self.recurrent_constraint),
          'bias_constraint':
              constraints.serialize(self.bias_constraint),
          't_kernel_constraint':
              constraints.serialize(self.t_kernel_constraint),
          't_recurrents_constraint':
              constraints.serialize(self.t_recurrent_constraint),
          't_bias_constraint':
                constraints.serialize(self.t_bias_constraint),
          'dropout':
              self.dropout,
          'recurrent_dropout':
              self.recurrent_dropout
      }
      config.update(_config_for_enable_caching_device(self.cell))
      base_config = super(GACTRNN, self).get_config()
      del base_config['cell']
      return dict(list(base_config.items()) + list(config.items()))
    
    @classmethod
    def from_config(cls, config):
      if 'implementation' in config:
        config.pop('implementation')
      return cls(**config)


@tf_export('keras.layers.GCTRNNCell')
class GCTRNNCell(DropoutRNNCellMixin, Layer):
    """Cell class for GCTRNNCell.

    Arguments:
        units_vec: Positive integer, dimensionality of the output space.
        modules: Positive integer, number of modules.
            The dimensionality of the outputspace is a concatenation of
            all modules k with the respective units_vec[k] size.
            Default: depends on size of units_vec or 1 in case of units_vec
            being a scalar.
        t_vec: Positive float >= 1, timescale.
            Unit-dependent time constant of leakage.
        connectivity: connection scheme in case of more than one modules
            Default: `dense`
            Other options are `partitioned`, `clocked`, and `adjacent`
        activation: Activation function to use.
            Default: hyperbolic tangent (`tanh`).
            If you pass `None`, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs.
        recurrent_initializer: Initializer for the `recurrent_kernel`
            weights matrix,
            used for the linear transformation of the recurrent state.
        bias_initializer: Initializer for the bias vector.
        t_kernel_initializer: Initializer for the t_kernel vector.
        t_recurrent_initializer: Initializer for the t_recurrent vector.
        kernel_regularizer: Regularizer function applied to the `kernel` weights matrix.
        recurrent_regularizer: Regularizer function applied to
            the `recurrent_kernel` weights matrix.
        bias_regularizer: Regularizer function applied to the bias vector.
        t_kernel_regularizer: Regularizer function applied to the t_kernel vector.
        t_recurrent_regularizer: Regularizer function applied to the t_recurrent vector.
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix.
        recurrent_constraint: Constraint function applied to
            the `recurrent_kernel` weights matrix.
        bias_constraint: Constraint function applied to the bias vector.
        t_kernel_constraint: Constraint function applied to the t_kernel vector.
        t_recurrent_constraint: Constraint function applied to the t_recurrent vector.
        dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the inputs.
        recurrent_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the recurrent state.
            
  Call arguments:
    inputs: A 2D tensor.
    states: List of state tensors corresponding to the previous timestep.
    training: Python boolean indicating whether the layer should behave in
      training mode or in inference mode. Only relevant when `dropout` or
      `recurrent_dropout` is used.
    """
    def __init__(self,
               units_vec,
               modules=None,
               t_vec=1.,
               connectivity='dense',
               activation='tanh',
               use_bias=True,
               kernel_initializer='glorot_uniform',
               recurrent_initializer='orthogonal',
               bias_initializer='zeros',
               t_kernel_initializer='zeros',
               t_recurrent_initializer='zeros',
               kernel_regularizer=None,
               recurrent_regularizer=None,
               bias_regularizer=None,
               t_kernel_regularizer=None,
               t_recurrent_regularizer=None,
               kernel_constraint=None,
               recurrent_constraint=None,
               bias_constraint=None,
               t_kernel_constraint=None,
               t_recurrent_constraint=None,
               dropout=0.,
               recurrent_dropout=0.,
               **kwargs):
        if ops.executing_eagerly_outside_functions():
          self._enable_caching_device = kwargs.pop('enable_caching_device', True)
        else:
          self._enable_caching_device = kwargs.pop('enable_caching_device', False)
        super(GCTRNNCell, self).__init__(**kwargs)
        
        self.connectivity = connectivity
        if isinstance(units_vec, list):
            self.units_vec = units_vec
            self.modules = len(units_vec)
            self.units = 0
            for k in range(self.modules):
                self.units += units_vec[k]
        else:
            self.units = units_vec
            if modules is not None and modules > 1:
                self.modules = int(modules)
                self.units_vec = [units_vec//modules for k in range(self.modules)]
            else:
                self.modules = 1
                self.units_vec = [units_vec]
                self.connectivity = 'dense'
        
        if isinstance(t_vec, list):
            if len(t_vec) != self.modules:
                raise ValueError("vector of tau must be of same size as "
                                 "modules or size of vector of units_vec")
            self.t_vec = t_vec
            self.ts = array_ops.constant(
                [[max(1., float(t_vec[k]))] for k in range(self.modules)
                 for n in range(self.units_vec[k])],
                dtype=self.dtype, shape=[self.units], name="ts")
        else:
            if self.modules > 1:
                self.t_vec = [max(1., float(t_vec)) for k in range(self.modules)]
            else:
                self.t_vec = [max(1., float(t_vec))]
            self.ts = array_ops.constant(
                max(1., t_vec), dtype=self.dtype, shape=[self.units], name="ts")

        self.activation = activations.get(activation)
        self.use_bias = use_bias
    
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.t_kernel_initializer = initializers.get(t_kernel_initializer)
        self.t_recurrent_initializer = initializers.get(t_recurrent_initializer)
    
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.t_kernel_regularizer = regularizers.get(t_kernel_regularizer)
        self.t_recurrent_regularizer = regularizers.get(t_recurrent_regularizer)
    
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.t_kernel_constraint = constraints.get(t_kernel_constraint)
        self.t_recurrent_constraint = constraints.get(t_recurrent_constraint)
    
        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        implementation = kwargs.pop('implementation', 1)
        if self.recurrent_dropout != 0 and implementation != 1:
          logging.debug(RECURRENT_DROPOUT_WARNING_MSG)
          self.implementation = 1
        else:
          self.implementation = implementation
        self.state_size = data_structures.NoDependency([self.units, self.units])
        self.output_size = self.units
        
    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.units),
            name='kernel',
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint)
      
        if self.connectivity == 'partitioned':
            self.recurrent_kernel_vec = []
            self.t_recurrent = []
            for k in range(self.modules):
                self.recurrent_kernel_vec += [self.add_weight(
                    shape=(self.units_vec[k], self.units_vec[k]),
                    name='recurrent_kernel' + str(k),
                    initializer=self.recurrent_initializer,
                    regularizer=self.recurrent_regularizer,
                    constraint=self.recurrent_constraint)]
                self.t_recurrent += [self.add_weight(
                    shape=(self.units_vec[k], self.units_vec[k]),
                    name='tau_recurrent' + str(k),
                    initializer=self.t_recurrent_initializer,
                    regularizer=self.t_recurrent_regularizer,
                    constraint=self.t_recurrent_constraint)]
        elif self.connectivity == 'clocked':
            self.recurrent_kernel_vec = []
            self.t_recurrent = []
            for k in range(self.modules):
                self.recurrent_kernel_vec += [self.add_weight(
                    shape=(sum(self.units_vec[k:self.modules]),
                           self.units_vec[k]),
                    name='recurrent_kernel' + str(k),
                    initializer=self.recurrent_initializer,
                    regularizer=self.recurrent_regularizer,
                    constraint=self.recurrent_constraint)]
                self.t_recurrent += [self.add_weight(
                    shape=(sum(self.units_vec[k:self.modules]),
                           self.units_vec[k]),
                    name='tau_recurrent' + str(k),
                    initializer=self.t_recurrent_initializer,
                    regularizer=self.t_recurrent_regularizer,
                    constraint=self.t_recurrent_constraint)]
        elif self.connectivity == 'adjacent':
            self.recurrent_kernel_vec = []
            self.t_recurrent = []
            for k in range(self.modules):
                self.recurrent_kernel_vec += [self.add_weight(
                    shape=(sum(self.units_vec[
                           max(0, k - 1):min(self.modules, k + 1 + 1)]),
                           self.units_vec[k]),
                    name='recurrent_kernel' + str(k),
                    initializer=self.recurrent_initializer,
                    regularizer=self.recurrent_regularizer,
                    constraint=self.recurrent_constraint)]
                self.t_recurrent += [self.add_weight(
                    shape=(sum(self.units_vec[
                               max(0, k - 1):min(self.modules, k + 1 + 1)]),
                           self.units_vec[k]),
                    name='tau_recurrent' + str(k),
                    initializer=self.t_recurrent_initializer,
                    regularizer=self.t_recurrent_regularizer,
                    constraint=self.t_recurrent_constraint)]
        else:
            self.recurrent_kernel_vec = [self.add_weight(
                    shape=(self.units, self.units),
                    name='recurrent_kernel',
                    initializer=self.recurrent_initializer,
                    regularizer=self.recurrent_regularizer,
                    constraint=self.recurrent_constraint)]
            self.t_recurrent = [self.add_weight(
                    shape=(self.units, self.units),
                    name='tau_recurrent',
                    initializer=self.t_recurrent_initializer,
                    regularizer=self.t_recurrent_regularizer,
                    constraint=self.t_recurrent_constraint)]

        if self.use_bias:
          self.bias = self.add_weight(
              shape=(self.units,),
              name='bias',
              initializer=self.bias_initializer,
              regularizer=self.bias_regularizer,
              constraint=self.bias_constraint)
        else:
          self.bias = None
          
        self.t_kernel = self.add_weight(
            shape=(input_shape[-1], self.units),
            name='tau_kernel',
            initializer=self.t_kernel_initializer,
            regularizer=self.t_kernel_regularizer,
            constraint=self.t_kernel_constraint)
        
        self.t0 = K.log(self.ts - _ALMOST_ONE)
        self.built = True
    
    def call(self, inputs, states, training=None):
        prev_output = states[0]
        prev_z = states[1]
        dp_mask = self.get_dropout_mask_for_cell(inputs, training)
        rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(
            prev_output, training)
        
        if dp_mask is not None:
          h = K.dot(inputs * dp_mask, self.kernel)
        else:
          h = K.dot(inputs, self.kernel)
        
        if self.bias is not None:
          h = K.bias_add(h, self.bias)
    
        if rec_dp_mask is not None:
          prev_output = prev_output * rec_dp_mask
        
        prev_y_vec = array_ops.split(prev_output, self.units_vec, axis=1)
        if self.connectivity == 'partitioned':
            r = array_ops.concat([K.dot(prev_y_vec[k], self.recurrent_kernel_vec[k])
                 for k in range(self.modules)], 1)
            t_r = array_ops.concat(
                [K.dot(prev_y_vec[k], self.t_recurrent[k])
                 for k in range(self.modules)], 1)
        elif self.connectivity == 'clocked':
            r = array_ops.concat([K.dot(array_ops.concat(prev_y_vec[k:self.modules], 1),
                    self.recurrent_kernel_vec[k])
                 for k in range(self.modules)], 1)
            t_r = array_ops.concat(
                [K.dot(array_ops.concat(
                    prev_y_vec[k:self.modules], 1),
                    self.t_recurrent[k])
                    for k in range(self.modules)], 1)
        elif self.connectivity == 'adjacent':
            r = array_ops.concat([K.dot(array_ops.concat(
                    prev_y_vec[max(0, k - 1):min(self.modules, k + 1 + 1)], 1),
                    self.recurrent_kernel_vec[k])
                 for k in range(self.modules)], 1)
            t_r = array_ops.concat(
                [K.dot(array_ops.concat(
                    prev_y_vec[max(0, k - 1):min(self.modules, k + 1 + 1)], 1),
                    self.t_recurrent[k])
                    for k in range(self.modules)], 1)
        else:
            r = K.dot(prev_output, self.recurrent_kernel_vec[0])
            t_r = K.dot(prev_output, self.t_recurrent[0])
        
        ts_act = K.exp(K.dot(inputs, self.t_kernel) + t_r + self.t0) + _ALMOST_ONE
        z = (1. - 1. / ts_act) * prev_z + (1. / ts_act) * (h + r)
        
        if self.activation is not None:
            output = self.activation(z)
        else:
            output = z
        
        return output, [output, z]
      
    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return list(_generate_zero_filled_state_for_cell(
            self, inputs, batch_size, dtype))
    
    def get_ts(self):
        return self.ts if self.built else None
    
    def get_config(self):
        config = {
            'units_vec':
                self.units_vec,
            'modules':
                self.modules,
            't_vec':
                self.t_vec,
            'activation':
                activations.serialize(self.activation),
            'use_bias':
                self.use_bias,
            'kernel_initializer':
                initializers.serialize(self.kernel_initializer),
            'recurrent_initializer':
                initializers.serialize(self.recurrent_initializer),
            'bias_initializer':
                initializers.serialize(self.bias_initializer),
            't_kernel_initializer':
                initializers.serialize(self.t_kernel_initializer),
            't_recurrent_initializer':
                initializers.serialize(self.t_recurrent_initializer),
            'kernel_regularizer':
                regularizers.serialize(self.kernel_regularizer),
            'recurrent_regularizer':
                regularizers.serialize(self.recurrent_regularizer),
            'bias_regularizer':
                regularizers.serialize(self.bias_regularizer),
            't_kernel_regularizer':
                regularizers.serialize(self.t_kernel_regularizer),
            't_recurrent_regularizer':
                regularizers.serialize(self.t_recurrent_regularizer),
            'kernel_constraint':
                constraints.serialize(self.kernel_constraint),
            'recurrent_constraint':
                constraints.serialize(self.recurrent_constraint),
            'bias_constraint':
                constraints.serialize(self.bias_constraint),
            't_kernel_constraint':
                constraints.serialize(self.t_kernel_constraint),
            't_recurrent_constraint':
                constraints.serialize(self.t_recurrent_constraint),
            'dropout':
                self.dropout,
            'recurrent_dropout':
                self.recurrent_dropout
        }
        config.update(_config_for_enable_caching_device(self))
        base_config = super(GCTRNNCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    

@tf_export('keras.layers.GCTRNN')
class GCTRNN(RNN):
    """Timescale Gated Continuous Time RNN that can have several modules
       where the output is to be fed back to input.
    
    Arguments:
      units_vec: Positive integer or vector of positive integer,
            dimensionality of the output space.
      modules: Positive integer, number of modules.
          The dimensionality of the outputspace is a concatenation of
          all modules k with the respective units_vec[k] size.
          Default: depends on size of units_vec or 1 in case of units_vec
          being a scalar.
      t: Positive float >= 1, timescale.
          Unit-dependent time constant of leakage.
      connectivity: connection scheme in case of more than one modules
          Default: `dense`
          Other options are `partitioned`, `clocked`, and `adjacent`
      activation: Activation function to use.
        Default: hyperbolic tangent (`tanh`).
        If you pass None, no activation is applied
        (ie. "linear" activation: `a(x) = x`).
      use_bias: Boolean, whether the layer uses a bias vector.
      kernel_initializer: Initializer for the `kernel` weights matrix,
        used for the linear transformation of the inputs.
      recurrent_initializer: Initializer for the `recurrent_kernel`
        weights matrix,
        used for the linear transformation of the recurrent state.
      bias_initializer: Initializer for the bias vector.
      t_kernel_initializer: Initializer for the tau_kernel vector.
      t_recurrent_initializer: Initializer for the tau_recurrent vector.
      kernel_regularizer: Regularizer function applied to
        the `kernel` weights matrix.
      recurrent_regularizer: Regularizer function applied to
        the `recurrent_kernel` weights matrix.
      bias_regularizer: Regularizer function applied to the bias vector.
      t_kernel_regularizer: Regularizer function applied to the tau_kernel vector.
      t_recurrent_regularizer: Regularizer function applied to the tau_recurrent vector.
      activity_regularizer: Regularizer function applied to
        the output of the layer (its "activation")..
      kernel_constraint: Constraint function applied to
        the `kernel` weights matrix.
      recurrent_constraint: Constraint function applied to
        the `recurrent_kernel` weights matrix.
      bias_constraint: Constraint function applied to the bias vector.
      t_kernel_constraint: Constraint function applied to the tau_kernel vector.
      t_recurrent_constraint: Constraint function applied to the tau_recurrent vector.
      dropout: Float between 0 and 1.
        Fraction of the units to drop for
        the linear transformation of the inputs.
      recurrent_dropout: Float between 0 and 1.
        Fraction of the units to drop for
        the linear transformation of the recurrent state.
      return_sequences: Boolean. Whether to return the last output
        in the output sequence, or the full sequence.
      return_state: Boolean. Whether to return the last state
        in addition to the output.
      go_backwards: Boolean (default False).
        If True, process the input sequence backwards and return the
        reversed sequence.
      stateful: Boolean (default False). If True, the last state
        for each sample at index i in a batch will be used as initial
        state for the sample of index i in the following batch.
      unroll: Boolean (default False).
        If True, the network will be unrolled,
        else a symbolic loop will be used.
        Unrolling can speed-up a RNN,
        although it tends to be more memory-intensive.
        Unrolling is only suitable for short sequences.
    
    Call arguments:
      inputs: A 3D tensor.
      mask: Binary tensor of shape `(samples, timesteps)` indicating whether
        a given timestep should be masked.
      training: Python boolean indicating whether the layer should behave in
        training mode or in inference mode. This argument is passed to the cell
        when calling it. This is only relevant if `dropout` or
        `recurrent_dropout` is used.
      initial_state: List of initial state tensors to be passed to the first
        call of the cell.
    """
    def __init__(self,
                units_vec,
                modules=None,
                t_vec=1.,
                connectivity='dense',
                activation='tanh',
                use_bias=True,
                kernel_initializer='glorot_uniform',
                recurrent_initializer='orthogonal',
                bias_initializer='zeros',
                t_kernel_initializer='zeros',
                t_recurrent_initializer='zeros',
                kernel_regularizer=None,
                recurrent_regularizer=None,
                bias_regularizer=None,
                t_kernel_regularizer=None,
                t_recurrent_regularizer=None,
                activity_regularizer=None,
                kernel_constraint=None,
                recurrent_constraint=None,
                bias_constraint=None,
                t_kernel_constraint=None,
                t_recurrent_constraint=None,
                dropout=0.,
                recurrent_dropout=0.,
                return_sequences=False,
                return_state=False,
                go_backwards=False,
                stateful=False,
                unroll=False,
                **kwargs):
        if 'implementation' in kwargs:
            kwargs.pop('implementation')
            logging.warning('The `implementation` argument '
                            'in `SimpleCTRNN` has been deprecated. '
                            'Please remove it from your layer call.')
        if 'enable_caching_device' in kwargs:
            cell_kwargs = {'enable_caching_device':
                         kwargs.pop('enable_caching_device')}
        else:
            cell_kwargs = {}
        cell = GCTRNNCell(
            units_vec,
            modules=modules,
            t_vec=t_vec,
            connectivity=connectivity,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            recurrent_initializer=recurrent_initializer,
            bias_initializer=bias_initializer,
            t_kernel_initializer=t_kernel_initializer,
            t_recurrent_initializer=t_recurrent_initializer,
            kernel_regularizer=kernel_regularizer,
            recurrent_regularizer=recurrent_regularizer,
            bias_regularizer=bias_regularizer,
            t_kernel_regularizer=t_kernel_regularizer,
            t_recurrent_regularizer=t_recurrent_regularizer,
            kernel_constraint=kernel_constraint,
            recurrent_constraint=recurrent_constraint,
            bias_constraint=bias_constraint,
            t_kernel_constraint=t_kernel_constraint,
            t_recurrent_constraint=t_recurrent_constraint,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            dtype=kwargs.get('dtype'),
            trainable=kwargs.get('trainable', True),
            **cell_kwargs)
        super(GCTRNN, self).__init__(
            cell,
            return_sequences=return_sequences,
            return_state=return_state,
            go_backwards=go_backwards,
            stateful=stateful,
            unroll=unroll,
            **kwargs)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        
    def call(self, inputs, mask=None, training=None, initial_state=None):
        self._maybe_reset_cell_dropout_mask(self.cell)
        return super(GCTRNN, self).call(
            inputs, mask=mask, training=training, initial_state=initial_state)
    
    @property
    def units(self):
        return self.cell.units

    @property
    def units_vec(self):
        return self.cell.units_vec

    @property
    def modules(self):
        return self.cell.modules

    @property
    def t_vec(self):
        return self.cell.t_vec

    @property
    def activation(self):
      return self.cell.activation

    @property
    def use_bias(self):
      return self.cell.use_bias

    @property
    def kernel_initializer(self):
      return self.cell.kernel_initializer
    
    @property
    def recurrent_initializer(self):
      return self.cell.recurrent_initializer
    
    @property
    def bias_initializer(self):
      return self.cell.bias_initializer
    
    @property
    def kernel_regularizer(self):
      return self.cell.kernel_regularizer
  
    @property
    def t_kernel_initializer(self):
        return self.cell.t_kernel_initializer

    @property
    def t_recurrent_initializer(self):
        return self.cell.t_recurrent_initializer
    
    @property
    def recurrent_regularizer(self):
      return self.cell.recurrent_regularizer
    
    @property
    def bias_regularizer(self):
      return self.cell.bias_regularizer
  
    @property
    def t_kernel_regularizer(self):
        return self.cell.t_kernel_regularizer

    @property
    def t_recurrent_regularizer(self):
        return self.cell.t_recurrent_regularizer
    
    @property
    def kernel_constraint(self):
      return self.cell.kernel_constraint
    
    @property
    def recurrent_constraint(self):
      return self.cell.recurrent_constraint
    
    @property
    def bias_constraint(self):
      return self.cell.bias_constraint
  
    @property
    def t_kernel_constraint(self):
        return self.cell.t_kernel_constraint

    @property
    def t_recurrent_constraint(self):
        return self.cell.t_recurrent_constraint
    
    @property
    def dropout(self):
      return self.cell.dropout
    
    @property
    def recurrent_dropout(self):
      return self.cell.recurrent_dropout
  
    @property
    def get_ts(self):
        return self.cell.get_ts()

    def get_config(self):
      config = {
          'units':
              self.units,
          'units_vec':
              self.units_vec,
          'modules':
              self.modules,
          't_vec':
              self.t_vec,
          'activation':
              activations.serialize(self.activation),
          'use_bias':
              self.use_bias,
          'kernel_initializer':
              initializers.serialize(self.kernel_initializer),
          'recurrent_initializer':
              initializers.serialize(self.recurrent_initializer),
          'bias_initializer':
              initializers.serialize(self.bias_initializer),
          't_kernel_initializer':
              initializers.serialize(self.t_kernel_initializer),
          't_recurrent_initializer':
              initializers.serialize(self.t_recurrent_initializer),
          'kernel_regularizer':
              regularizers.serialize(self.kernel_regularizer),
          'recurrent_regularizer':
              regularizers.serialize(self.recurrent_regularizer),
          'bias_regularizer':
              regularizers.serialize(self.bias_regularizer),
          't_kernels_regularizer':
              regularizers.serialize(self.t_kernel_regularizer),
          't_recurrent_regularizer':
              regularizers.serialize(self.t_recurrent_regularizer),
          'activity_regularizer':
              regularizers.serialize(self.activity_regularizer),
          'kernel_constraint':
              constraints.serialize(self.kernel_constraint),
          'recurrent_constraint':
              constraints.serialize(self.recurrent_constraint),
          'bias_constraint':
              constraints.serialize(self.bias_constraint),
          't_kernel_constraint':
              constraints.serialize(self.t_kernel_constraint),
          't_recurrents_constraint':
              constraints.serialize(self.t_recurrent_constraint),
          'dropout':
              self.dropout,
          'recurrent_dropout':
              self.recurrent_dropout
      }
      base_config = super(GCTRNN, self).get_config()
      config.update(_config_for_enable_caching_device(self.cell))
      del base_config['cell']
      return dict(list(base_config.items()) + list(config.items()))
    
    @classmethod
    def from_config(cls, config):
      if 'implementation' in config:
        config.pop('implementation')
      return cls(**config)

