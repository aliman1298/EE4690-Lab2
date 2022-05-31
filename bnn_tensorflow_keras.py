import sys
import time
import tensorflow as tf
import matplotlib.pyplot as plt

show, testing = True, True


def get_data():
    """
    Function to input the data from mnist set through tf keras
    :return:
    """
    return tf.keras.datasets.mnist


def test():
    """
    Test function for the bnn
    :return:
    """
# Basic inputs from tutorials and stuff
# https://www.tensorflow.org/tutorials/quickstart/beginner
# https://neptune.ai/blog/binarized-neural-network-bnn-and-its-implementation-in-ml
# Tutorial on logic of implementation
# https://towardsdatascience.com/convolutional-layers-vs-fully-connected-layers-364f05ab460b
#

    # mnist is a basic dataset for recognizing symbols
    mnist = get_data()

    if show:
        print("-Show data set stuff-")

    # Split dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if show:
        print(len(x_train), "-", len(x_train[0]), "-", len(x_train[0][0]))

    # Convert to floating point numbers
    x_train, x_test = x_train / 255.0, x_test / 255.0
    input_size = (len(x_train[0]), len(x_train[0][0]))

    # Basic model from tutorial
    # Get some idea for the actual implementation later
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(input_size[0], input_size[1])),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10)
    ])

    sizes = [input_size[0]*input_size[1], 0, 0, 10]
    for i in range(1, len(sizes)):
        sizes[i] = int(sizes[i-1] * 0.24)
    if show:
        print(sizes)

    # =SOME DATA FOR LAYER STUFF=
    # Explanation on layers: https://www.tensorflow.org/tutorials/customization/custom_layers
    # More specifically into layers: https://www.tensorflow.org/api_docs/python/tf/keras/layers
    # Flatten layer: defined for input, puts it in shape given to function
    # Fully Connected Layer = Dense
    #   Dense layer: activation inserts an additional layer, an activation layer after the dense layer
    # Dropout layer: Applies Dropout to the input.
    # The Dropout layer randomly sets input units to 0 with a frequency of rate at each step during training time, which helps prevent overfitting
    # Conv2d layer: 2D convolution layer (e.g. spatial convolution over images).
    # AvgPool2D layer: Average pooling operation for spatial data.

    # Build a model #2
    # Basic implementation with Dense (FC) layers, but no BN
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(input_size[0], input_size[1])),
        # tf.keras.layers.Dense(sizes[1], activation='relu'),
        # tf.keras.layers.Dense(sizes[2], activation='relu'),
        tf.keras.layers.Dense(sizes[1]),
        tf.keras.layers.Dense(sizes[2]),
        tf.keras.layers.Dense(sizes[3])
    ])

    # Build a model #3: BNN UNIT
    # (1 - rate) * norm + rate * norm.clip(min_value, max_value)
    # norm.clip ->
    constraints = tf.keras.constraints.MinMaxNorm(
        min_value=0.0, max_value=1.0, rate=1.0, axis=0
    )

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(input_size[0], input_size[1])),
        # tf.keras.layers.Dense(sizes[1], activation='relu'),
        # tf.keras.layers.Dense(sizes[2], activation='relu'),
        tf.keras.layers.Dense(sizes[1], use_bias=False, kernel_constraint=constraints),
        tf.keras.layers.Dense(sizes[2], use_bias=False, kernel_constraint=constraints),
        tf.keras.layers.Dense(sizes[3], use_bias=False, kernel_constraint=constraints)
    ])

    # Make predictions
    predictions = model(x_train[:1]).numpy()

    # Convert said predictions
    tf.nn.softmax(predictions).numpy()

    # Make a loss function
    # Other function:    binary_crossentropy
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # Make some function
    loss_fn(y_train[:1], predictions).numpy()

    model.compile(optimizer='adam',
                  loss=loss_fn,
                  metrics=['accuracy'])

    # Fit model to the training data in epochs amount of iterations
    model.fit(x_train, y_train, epochs=5)

    # Evaluate the model on specified test set
    model.evaluate(x_test, y_test, verbose=2)

    #Summary of model implementation
    model.summary()
    # View weights
    print("- Weights: -\n")
    for i in range(1, 4):
        print("\n- Layer: ", i, " -\n")
        print("W:", model.layers[i].weights)
        # print("Bi: ", model.layers[i].bias.numpy())
        # print("Bi_i: ", model.layers[i].bias_initializer)

    print("-End of the test-")


def main():
    test()


if __name__ == '__main__':
    main()
