import tensorflow as tf

class IED(tf.keras.metrics.Metric):

    def __init__(self, num_items, k, name='ied', **kwargs):
        super(IED, self).__init__(name=name, **kwargs)
        # initializer = tf.keras.initializers.Constant(0.0)
        self.exposure = self.add_weight(name='exposure', initializer='zeros', shape=(num_items, ))
        # self.exposure = tf.zeros((num_items, ))
        # self.exposure = tf.Variable(initial_value=tf.zeros((num_items, )), trainable=False)
        # self.exposure = tf.Variable(initial_value=tf.zeros((num_items, )), trainable=False)
        self.num_items = num_items
        self.k = k

    # @tf.function
    def compute_exposure(self, y_pred, sample_weight=None):
        position = tf.cast(tf.argsort(tf.argsort(y_pred, direction='DESCENDING')), tf.float32) + 1.
        mask = tf.cast(position <= self.k, tf.float32)
        bias = 1. / tf.math.log(1. + position)
        return tf.reduce_sum(bias * mask, axis=0)

    def update_state(self, y_true, y_pred, sample_weight=None):
        expo = self.compute_exposure(y_pred)
        self.exposure.assign_add(expo)
        

    def gini(self, array):
        """Calculate the Gini coefficient of a numpy array."""
        # based on bottom eq:
        # http://www.statsdirect.com/help/generatedimages/equations/equation154.svg
        # from:
        # http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
        # All values are treated equally, arrays must be 1d:
        array = tf.cast(array, tf.float32)
        array = tf.math.abs(array)
        # Values cannot be 0:
        array += 0.0000001
        # Values must be sorted:
        array = tf.sort(array)
        # Index per array element:
        index = tf.cast(tf.keras.backend.arange(1, tf.shape(array)[0] + 1), tf.float32)
        # Number of array elements:
        n = tf.cast(tf.shape(array)[0], tf.float32)
        # Gini coefficient:
        return ((tf.reduce_sum((2 * index - n - 1) * array)) / (n * tf.reduce_sum(array)))

    def result(self):
        return self.gini(self.exposure)

    def reset_state(self):
        self.exposure.assign(tf.zeros((self.num_items, )))


class SessionIED(tf.keras.metrics.Metric):

    def __init__(self, num_items, k, name='session_ied', **kwargs):
        super(SessionIED, self).__init__(name=name, **kwargs)
        self.gini_sum = self.add_weight(name='gini_sum', initializer='zeros')
        self.size = self.add_weight(name='size', initializer='zeros')
        self.num_items = num_items
        self.k = k

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


    def update_state(self, y_true, y_pred, sample_weight=None):
        predictions, dids, history = y_pred

        position = tf.cast(tf.argsort(tf.argsort(predictions, direction='DESCENDING')), tf.float32) + 1.
        mask = tf.cast(position <= self.k, tf.float32)
        exposure = 1. / tf.math.log(1. + position)
        exposure = exposure * mask

        all_index = tf.where(tf.equal(tf.reshape(history, (-1, tf.shape(history)[1], 1)), dids))
        index = tf.gather(all_index, [0, 1], axis=1)
        v = tf.gather(all_index, [2], axis=1)

        x = tf.scatter_nd(index, tf.reshape(v, (-1, )), tf.cast(tf.shape(history), tf.int64))
        x = tf.where(tf.equal(history, tf.constant(-1, dtype=tf.int64)), history, x)
        # exposure_ground_truth = tf.reduce_sum(tf.gather(exposure, x), axis=1)

        # sum = tf.reduce_sum(tf.gather(exposure, x), axis=1)
        sum = tf.reduce_sum(self.safe_gather(exposure, x), axis=1)
        num = tf.reshape(tf.reduce_sum(tf.cast(tf.greater_equal(x, tf.constant(0, dtype=tf.int64)), tf.float32), 1), (-1, 1))
        exposure_ground_truth = sum / num

        def each_exposure(ipt):
            _, exposure = ipt
            loss = self.gini(exposure)
            return [loss, exposure]

        gini, _ = tf.map_fn(each_exposure, [tf.zeros((tf.shape(exposure_ground_truth)[0], )), exposure_ground_truth])
        self.gini_sum.assign_add(tf.reduce_sum(gini))
        self.size.assign_add(tf.cast(tf.shape(gini)[0], tf.float32))

    def gini(self, array):
        """Calculate the Gini coefficient of a numpy array."""
        # based on bottom eq:
        # http://www.statsdirect.com/help/generatedimages/equations/equation154.svg
        # from:
        # http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
        # All values are treated equally, arrays must be 1d:
        array = tf.cast(array, tf.float32)
        array = tf.math.abs(array)
        # Values cannot be 0:
        array += 0.0000001
        # Values must be sorted:
        array = tf.sort(array)
        # Index per array element:
        index = tf.cast(tf.keras.backend.arange(1, tf.shape(array)[0] + 1), tf.float32)
        # Number of array elements:
        n = tf.cast(tf.shape(array)[0], tf.float32)
        # Gini coefficient:
        return ((tf.reduce_sum((2 * index - n - 1) * array)) / (n * tf.reduce_sum(array)))

    def result(self):
        return self.gini_sum / self.size

    def reset_state(self):
        self.gini_sum.assign(0.0)
        self.size.assign(0.0)