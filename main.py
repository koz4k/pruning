"""Investigation of weight and neuron pruning schemes of neural networks."""


import functools
import time

from tensorflow.keras import (
    activations,
    datasets,
    initializers,
    layers,
    losses,
    models,
    optimizers,
    utils,
)

from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf


BATCH_SIZE = 64
PRUNING_FRACTIONS = (0, 0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99)
LAYER_SIZES = (1000, 1000, 500, 200)  # Layer sizes without the output one.
N_EPOCHS = 4
N_TRIALS = 5  # Number of trials in inference time measurement.


class Sparse(layers.Layer):
    """Sparse layer without bias.

    Weight matrix (attribute kernel_t) is stored in transposed form, as
    tf.sparse.matmul is sparse x dense, not the other way around.
    
    Args:
        output_dim (int): Output dimensionality.
        n_nonzero (int): Number of non-zero entries in the weight matrix.
        activation: Activation function.
        indices_initializer: Initializer for the indices of the transposed
            weight matrix - int array of shape (n_nonzero, 2). Should be in
            lexicographical order.
        values_initializer: Initializer for the nonzero entries - float array
            of shape (n_nonzero,). Should be in the same order as the indices.
    """

    def __init__(
        self,
        output_dim,
        n_nonzero,
        activation,
        indices_initializer,
        values_initializer,
        **kwargs
    ):
        self.output_dim = output_dim
        self.n_nonzero = n_nonzero
        self.activation = activations.get(activation)
        self.indices_initializer = initializers.get(indices_initializer)
        self.values_initializer = initializers.get(values_initializer)
        super().__init__(**kwargs)

    def build(self, input_shape):
        (_, input_dim) = input_shape
        # Indices of nonzero entries.
        self.indices = self.add_weight(
            name="indices",
            shape=(self.n_nonzero, 2),
            dtype=tf.int64,
            initializer=self.indices_initializer,
            trainable=False,
        )
        # Values of nonzero entries.
        self.values = self.add_weight(
            name="values",
            shape=(self.n_nonzero,),
            initializer=self.values_initializer,
            trainable=True,
        )
        # Sparse weight tensor (transposed).
        self.kernel_t = tf.SparseTensor(
            self.indices, self.values, dense_shape=(self.output_dim, input_dim)
        )
        super().build(input_shape)

    def call(self, inputs):
        # TensorFlow supports only sparse-to-dense multiplication and we want to
        # do dense-to-sparse (inputs * kernel). We do this according to
        # B * A = (A^T * B^T)^T.
        output = tf.transpose(tf.sparse.matmul(self.kernel_t, tf.transpose(inputs)))
        if self.activation:
            output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        (_, input_dim) = input_shape
        return (input_dim, self.output_dim)


def preprocess_dataset(x, y):
    """Performs preprocessing of the dataset before feeding it to the NN."""
    # Add the channel dimension.
    x = np.expand_dims(x, axis=-1)
    # Rescale to [-1, 1].
    x = x.astype(np.float32)
    x /= 128
    x -= 0.5
    # One-hot encode the labels.
    y = utils.to_categorical(y)
    return (x, y)


def dense_kwargs(weight_matrix):
    """Defines kwargs needed for the Dense layer to initialize its weights."""
    return {"kernel_initializer": initializers.Constant(weight_matrix)}


def sparse_kwargs(weight_matrix):
    """Defines kwargs needed for the Sparse layer to initialize its weights."""
    weight_matrix_t = np.transpose(weight_matrix)
    nonzero_arrays = np.nonzero(weight_matrix_t)
    indices = np.transpose(nonzero_arrays)
    values = weight_matrix_t[nonzero_arrays]
    return {
        "n_nonzero": len(values),
        "indices_initializer": initializers.Constant(indices),
        "values_initializer": initializers.Constant(values),
    }


def create_model(layer_sizes, n_classes, layer_fn, layer_kwargs_fn, weights=None):
    """Creates the model, optionally initializing its weights.

    Args:
        layer_sizes (list): Numbers of neurons in consecutive layers, excluding
            the output layer.
        n_classes (int): Number of classification classes.
        layer_fn: Factory function for hidden layers. Should accept at least
            output_dim, activation and everything returned from layer_kwargs_fn.
        layer_kwargs_fn: Function weights -> kwargs defining kwargs needed
            for the given layer_fn to initialize it with the given weights.
        weights (list or None): List of weight matrices to initialize the model
            with. Defaults to None - use random initialization.

    Returns:
        A compiled model.
    """
    model = models.Sequential()
    model.add(layers.Flatten())

    def layer_kwargs(layer_kwargs_fn, i):
        if weights is not None:
            return layer_kwargs_fn(weights[i])
        else:
            return {}

    # Build the hidden layers using the provided layer_fn.
    for (i, layer_size) in enumerate(layer_sizes):
        model.add(
            layer_fn(layer_size, activation="relu", **layer_kwargs(layer_kwargs_fn, i))
        )
    # We don't prune the output layer so it's always Dense.
    model.add(
        layers.Dense(
            n_classes,
            activation="softmax",
            use_bias=False,
            **layer_kwargs(dense_kwargs, len(layer_sizes))
        )
    )

    model.compile(
        loss=losses.categorical_crossentropy,
        optimizer=optimizers.Adam(),
        metrics=["accuracy"],
    )

    return model


def prune_weights(model, fraction):
    """Prunes a fraction of model weights."""
    weights = model.get_weights()

    def prune_weight_matrix(weight_matrix):
        # Copy the weights so we don't modify the original network.
        weight_matrix = np.copy(weight_matrix)
        flat_weight_matrix = np.reshape(weight_matrix, (-1,))
        kth = int(len(flat_weight_matrix) * fraction)
        # Determine the k least relevant weights using np.argpartition.
        indices = np.argpartition(np.abs(flat_weight_matrix), kth)
        # Prune them.
        flat_weight_matrix[indices[:kth]] = 0
        weight_matrix = np.reshape(flat_weight_matrix, weight_matrix.shape)
        return weight_matrix

    weights[:-1] = list(map(prune_weight_matrix, weights[:-1]))

    (_, n_classes) = weights[-1].shape
    # Create a pruned model.
    return create_model(
        LAYER_SIZES,
        n_classes,
        layer_fn=Sparse,
        layer_kwargs_fn=sparse_kwargs,
        weights=weights,
    )


def prune_neurons(model, fraction):
    """Prunes a fraction of model neurons."""
    weights = model.get_weights()

    def nonzero_indices(weight_matrix):
        neuron_norms = np.linalg.norm(weight_matrix, axis=0)
        kth = int(len(neuron_norms) * fraction)
        # Determine the k least relevant neurons using np.argpartition.
        return np.argpartition(neuron_norms, kth)[kth:]

    (n_inputs, _) = weights[0].shape
    # Remember which neurons we left in the last layer - we'll need to know that
    # to prune the next one. At first it's all of the inputs as we don't prune
    # them.
    last_indices = np.arange(n_inputs)
    layer_sizes = []
    for (i, weight_matrix) in enumerate(weights[:-1]):
        indices = nonzero_indices(weight_matrix)
        layer_sizes.append(len(indices))
        # Take a subset of both rows and columns.
        weights[i] = weight_matrix[last_indices, :][:, indices]
        last_indices = indices
    # Take a subset of rows for the last layer - we don't prune the outputs.
    weights[-1] = weights[-1][last_indices, :]

    (_, n_classes) = weights[-1].shape
    # Create a pruned model.
    return create_model(
        layer_sizes,
        n_classes,
        layer_fn=functools.partial(layers.Dense, use_bias=False),
        layer_kwargs_fn=dense_kwargs,
        weights=weights,
    )


def evaluate_fraction(pruning_fn, model, dataset, fraction):
    """Evaluates a pruning fraction on a given model.

    Args:
        pruning_fn: Function (model, fraction) -> model.
        model: Keras model.
        dataset: Pair (inputs, labels).
        fraction (float): A fraction of the model to prune.

    Returns:
        Pair (accuracy, inference time).
    """
    # Run the model on CPU to avoid copying between CPU and GPU that would
    # dominate the time cost.
    with tf.device("/cpu:0"):
        model = pruning_fn(model, fraction)
        # Measure accuracy on the test set.
        (_, accuracy) = model.evaluate(*dataset)

        (inputs, _) = dataset
        start_time = time.time()
        # Measure inference time by repeatedly running prediction on the test
        # set. Feed the whole dataset at once to remove the impact of batching.
        for _ in range(N_TRIALS):
            model.predict_on_batch(inputs)
        trial_time = (time.time() - start_time) / N_TRIALS

        return (accuracy, trial_time)


def main():
    """Evaluates the two pruning methods and plots the results."""
    # Load and preprocess the data.
    ((x_train, y_train), (x_test, y_test)) = (
        preprocess_dataset(x, y) for (x, y) in datasets.mnist.load_data()
    )
    (_, n_classes) = y_train.shape

    # Create and fit the original model.
    model = create_model(
        LAYER_SIZES,
        n_classes,
        layer_fn=functools.partial(layers.Dense, use_bias=False),
        layer_kwargs_fn=dense_kwargs,
    )
    model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=N_EPOCHS)

    # Evaluate the pruning methods.
    pruning_methods = [("weight", prune_weights), ("neuron", prune_neurons)]
    for (method_name, pruning_fn) in pruning_methods:
        # Compute the evaluation curves.
        (accuracy_curve, time_curve) = zip(
            *[
                evaluate_fraction(pruning_fn, model, (x_test, y_test), fraction)
                for fraction in PRUNING_FRACTIONS
            ]
        )
        # Plot them.
        for (subplot, curve) in [(211, accuracy_curve), (212, time_curve)]:
            plt.subplot(subplot)
            plt.plot(PRUNING_FRACTIONS, curve, label=method_name)

            if method_name == "neuron":
                # The first evaluated model in neuron pruning is the unpruned
                # one. Use it as a baseline.
                (baseline, *_) = curve
                plt.plot((0, 1), (baseline, baseline), "--", label="unpruned")

    # Add some labels to the plots.
    for (subplot, y_label) in [(211, "accuracy"), (212, "inference time")]:
        plt.subplot(subplot)
        plt.xlabel("pruning fraction")
        plt.ylabel(y_label)
        plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
