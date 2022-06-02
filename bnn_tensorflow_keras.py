import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import larq as lq

show, testing = True, True
filename = 'model_weights'


class BinaryConstraints(tf.keras.constraints.Constraint):
    def __init__(self, minimum, maximum, rate):
        self.min = minimum
        self.max = maximum
        self.rate = rate
        self.length = 100

    def __call__(self, w):
        # Inputs is:        <tf.Variable 'dense_5/kernel:0' shape=(784, 188) dtype=float32>
        # Output should be a similar variable, or just a check in the full value for max and min values

        # CURRENTLY:
        # Check the tf.variable and with that how to access the weights, check them and put them into max/min amounts

        w_new = w
        for i in range(len(w_new)):
            for j in range(len(w_new[i])):
                w_new[i][j] = int(w_new[i][j])      # Works more easily if it shortens correctly
                # if w_new[i][j] > 0.5*(self.max-self.min):
                #     w_new[i][j] = maximum
                # else:
                #     w_new[i][j] = minimum
        print(w_new)

        # Example of a constraint function
        # w_shape = w.shape
        # rep = w_shape[0]/self.length
        # w_new = (np.arange(1,rep+1)/np.arange(1,rep+1).sum()).astype('float32')
        # w_new = tf.reshape(tf.repeat(tf.constant(w_new), self.length*w_shape[1]), [w_shape[0],w_shape[1]])

        return w_new


# def binary_weights(shape, dtype=tf.float32):
def binary_weights(shape, dtype=tf.int8):
    """This function generates weights of random 0s and 1s based on the provided shape"""
    # build logits matrix:
    logits = tf.fill((shape[0], 2), 0.5)
    # uniformly pick the class.
    return tf.cast(tf.random.categorical(tf.math.log(logits), shape[1]), dtype=dtype)


@tf.custom_gradient
def identity_sign(x):
    def grad(dy):
        return dy

    return tf.sign(x), grad


def get_data():
    """
    Function to input the data from mnist set through tf keras
    :return:
    """
    return tf.keras.datasets.mnist


def view_weights(model, weights):
    # model_weights = model.get_weights()  # get binary weights

    if weights == "":
        print("-- Model Weights: --")
        # print("First is flatten, then ", len(sizes) - 1, " layers of FC+Batch Normalisation")

        for i in range(0, len(model.layers)):       # for layer in model.layers:
            print("\n- Layer: ", i, " -")  # i*2+1 is normal
            print("W: ", model.layers[i].weights)
            # print("Bi: ", model.layers[i].bias.numpy())
            # print("Bi_i: ", model.layers[i].bias_initializer)
    else:
        print("-- Weights --")
        for i in range(len(weights)):
            print("\n- Layer: ", i, " -")  # i*2+1 is normal
            print("W: ", weights[i])

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

    # Load model
    # my_saved_model = keras.models.load_model(filename)

    if show:
        print("-Show data set stuff-")

    # Split dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if show:
        print(len(x_train), "-", len(x_train[0]), "-", len(x_train[0][0]))

    # Convert to floating point numbers
    x_train, x_test = x_train / 255.0, x_test / 255.0
    input_size = (len(x_train[0]), len(x_train[0][0]))

    # Define consistent decreasing sizes per layer
    sizes = [input_size[0]*input_size[1], 0, 0, 10]
    for i in range(1, len(sizes)):
        sizes[i] = int(sizes[i-1] * 0.24)
    if show:
        print(sizes)

    sizes = [input_size[0]*input_size[1], 256, 64, 10]

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

    # Build a model #3: BNN UNIT
    # Initializer define the initial value of kernel
    initializer = 'glorot_uniform'      # Standard value
    initializer = binary_weights
    # Constraints define a min or max norm, but not that it requires to be on the specified values, so wont work
    constraints = tf.keras.constraints.MinMaxNorm(
        min_value=0.0, max_value=1.0, rate=1.0, axis=0
    )
    constraints = BinaryConstraints(0, 1, 1)        # Attempt with a custom constraints

    model3 = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(input_size[0], input_size[1])),
        # tf.keras.layers.Dense(sizes[1], activation='relu'),
        # tf.keras.layers.Dense(sizes[2], activation='relu'),
        tf.keras.layers.Dense(sizes[1], use_bias=False, kernel_initializer=initializer, kernel_constraint=constraints),
        tf.keras.layers.Dense(sizes[2], use_bias=False, kernel_initializer=initializer, kernel_constraint=constraints),
        tf.keras.layers.Dense(sizes[3], use_bias=False, kernel_initializer=initializer, kernel_constraint=constraints)
    ])

    # Attempt #4
    # Define variables to use in model construction
    quantizer = "ste_sign"          # basic unit
    quantizer = identity_sign       # custom unit
    kwargs = dict(input_quantizer=quantizer,
                  kernel_quantizer=quantizer,
                  kernel_constraint="weight_clip",
                  # activation="softmax",
                  use_bias=False)
    # LARQ model
    larq_model =  tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(input_size[0], input_size[1]))
        ])
    # lq.layers.QuantDense(sizes[1], kernel_initializer="ste_sign", kernel_quantizer="ste_sign", kernel_constraint="weight_clip", use_bias=False, input_shape=(input_size[0], input_size[1])),

    for i in range(1, len(sizes)):
        larq_model.add(lq.layers.QuantDense(sizes[i], **kwargs))
        # larq_model.add(tf.keras.layers.BatchNormalization(momentum=0.999, scale=False))         # Seems to have some generic constants to multiply/divide with
    # Adds an extra activation function
    # larq_model.add(tf.keras.layers.Activation("softmax"))         # Softmax is a mathematical function that converts a vector of numbers into a vector of probabilities, where the probabilities of each value are proportional to the relative scale of each value in the vector.

    model = larq_model


    # Make predictions
    predictions = model(x_train[:1]).numpy()

    # Convert said predictions
    tf.nn.softmax(predictions).numpy()

    # Make a loss function
    # Other function:    binary_crossentropy
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # Make some function
    loss_fn(y_train[:1], predictions).numpy()
    # ERROR

    model.compile(optimizer='adam',
                  loss=loss_fn,
                  metrics=['accuracy'])

    # Fit model to the training data in epochs amount of iterations
    model.fit(x_train, y_train, epochs=5)

    print("[+] Evaluate, FP first, then binary")
    # Evaluate the model on specified test set
    model.evaluate(x_test, y_test, verbose=2)
    with lq.context.quantized_scope(True):
        model.evaluate(x_test, y_test, verbose=2)

    print("[+] Summary")
    # Summary of model implementation
    model.summary()

    return 0

    print("[+] Saving models, first full precision, then binary")

    # Save models
    model.save(filename+"_full_precision.h5")  # save full precision latent weights

    if show: # View weights
        print("-- Full prec weights --")
        view_weights(model, "")

    with lq.context.quantized_scope(True):
        model.save(filename+"_binary.h5")  # save binary weights

        if show: # View weights
            print("-- Binary weights --")
            view_weights(model, "")

    # ------------
    # That is to say, for each channel being normalized, the layer returns gamma * (batch - mean(batch)) / sqrt(var(batch) + epsilon) + beta

    print("-End of the test-")


def main():
    test()


if __name__ == '__main__':
    main()
