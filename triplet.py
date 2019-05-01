"""
Triplet neural network module.
"""

import random
import numpy as np

from keras.layers import Input
from keras.models import Model


class TripletNetwork:
    """
    A simple and lightweight triplet neural network implementation.

    The TripletNetwork class requires the base and head model to be defined via the constructor. The class exposes
    public methods that allow it to behave similarly to a regular Keras model by passing kwargs through to the
    underlying keras model object where possible. This allows Keras features like callbacks and metrics to be used.
    """
    def __init__(self, base_model, head_model, num_classes):
        """
        Construct the triplet model class with the following structure.

        -------------------------------------------------------------------
        alternating_input -> base_model |
                                        |
        anchor_input      -> base_model --> head_model --> 2 binary outputs
                                        |
        alternating_input -> base_model |
        -------------------------------------------------------------------

        :param base_model: The embedding model.
        * Input shape must be equal to that of data.
        * Must have a single output of any shape.
        :param head_model: The discriminator model.
        * Must accept a list of 3 inputs
        * Input shape must be equal to that of base model output.
        * Output shape must be equal to 2.
        :param num_classes: The number of classes in the data
        """
        # Set essential parameters
        self.base_model = base_model
        self.head_model = head_model
        self.num_classes = num_classes

        # Get input shape from base model
        self.input_shape = self.base_model.input_shape[1:]

        # Initialize triplet model
        self.triplet_model = None
        self.__initialize_triplet_model()

    def compile(self, *args, **kwargs):
        """
        Configures the model for training.

        Passes all arguments to the underlying Keras model compile function.
        """
        self.triplet_model.compile(*args, **kwargs)

    def fit(self, *args, **kwargs):
        """
        Trains the model on data generated batch-by-batch using the triplet network generator function.

        Redirects arguments to the fit_generator function.
        """
        x_train = args[0]
        y_train = args[1]
        x_test, y_test = kwargs.pop('validation_data')
        batch_size = kwargs.pop('batch_size')

        train_generator = self.__triplet_generator(x_train, y_train, batch_size)
        train_steps = len(x_train) / batch_size
        test_generator = self.__triplet_generator(x_test, y_test, batch_size)
        test_steps = len(x_test) / batch_size
        self.triplet_model.fit_generator(train_generator,
                                         steps_per_epoch=train_steps,
                                         validation_data=test_generator,
                                         validation_steps=test_steps, **kwargs)

    def fit_generator(self, x_train, y_train, x_test, y_test, batch_size, *args, **kwargs):
        """
        Trains the model on data generated batch-by-batch using the triplet network generator function.

        :param x_train: Training input data.
        :param y_train: Training output data.
        :param x_test: Validation input data.
        :param y_test: Validation output data.
        :param batch_size: Number of triplets to generate per batch.
        """
        train_generator = self.__triplet_generator(x_train, y_train, batch_size)
        train_steps = len(x_train) / batch_size
        test_generator = self.__triplet_generator(x_test, y_test, batch_size)
        test_steps = len(x_test) / batch_size
        self.triplet_model.fit_generator(train_generator,
                                         steps_per_epoch=train_steps,
                                         validation_data=test_generator,
                                         validation_steps=test_steps,
                                         *args, **kwargs)

    def load_weights(self, checkpoint_path):
        """
        Load triplet model weights. This also affects the reference to the base and head models.

        :param checkpoint_path: Path to the checkpoint file.
        """
        self.triplet_model.load_weights(checkpoint_path)

    def evaluate(self, *args, **kwargs):
        """
        Evaluate the triplet network with the same generator that is used to train it. Passes arguments through to the
        underlying Keras function so that callbacks etc can be used.

        Redirects arguments to the evaluate_generator function.

        :return: A tuple of scores
        """
        x = args[0]
        y = args[1]
        batch_size = kwargs.pop('batch_size')

        generator = self.__triplet_generator(x, y, batch_size)
        steps = len(x) / batch_size
        return self.triplet_model.evaluate_generator(generator, steps=steps, **kwargs)

    def evaluate_generator(self, x, y, batch_size, *args, **kwargs):
        """
        Evaluate the triplet network with the same generator that is used to train it. Passes arguments through to the
        underlying Keras function so that callbacks etc can be used.

        :param x: Input data
        :param y: Class labels
        :param batch_size: Number of triplets to generate per batch.
        :return: A tuple of scores
        """
        generator = self.__triplet_generator(x, y, batch_size)
        steps = len(x) / batch_size
        return self.triplet_model.evaluate_generator(generator, steps=steps, *args, **kwargs)

    def __initialize_triplet_model(self):
        """
        Create the triplet model structure using the supplied base and head model.
        """
        input_1 = Input(shape=self.input_shape)
        input_anchor = Input(shape=self.input_shape)
        input_2 = Input(shape=self.input_shape)

        alternating_1 = self.base_model(input_1)
        anchor = self.base_model(input_anchor)
        alternating_2 = self.base_model(input_2)

        head = self.head_model([alternating_1, anchor, alternating_2])
        self.triplet_model = Model([input_1, input_anchor, input_2], head)

    def __create_triplets(self, x, class_indices, num_triplets):
        """
        Create a numpy array of triplets and the associated labels.

        :param x: Input data
        :param class_indices: A python list of lists that contains each of the indices in the input data that belong
        to each class. It is used to find and access elements in the input data that belong to a desired class.
        * Example usage:
        * element_index = class_indices[class][index]
        * element = x[element_index]
        :param num_triplets: The number of triplets to create.
        :return: A tuple of (Numpy array of pairs, Numpy array of labels)
        """
        triplets = []
        labels = []
        for sample in range(num_triplets):
            anchor_class, negative_class = self.__randint_unequal(0, self.num_classes - 1)

            num_positive = len(class_indices[anchor_class])
            num_negative = len(class_indices[negative_class])

            anchor_index, positive_index = self.__randint_unequal(0, num_positive - 1)
            negative_index = random.randint(0, num_negative - 1)

            anchor = x[class_indices[anchor_class][anchor_index]]
            positive = x[class_indices[anchor_class][positive_index]]
            negative = x[class_indices[negative_class][negative_index]]

            # Alternate which output is used as the positive and negative leg of the network.
            if random.randint(0, 1) == 0:
                triplets.append([positive, anchor, negative])
                labels.append([0.0, 1.0])
            else:
                triplets.append([negative, anchor, positive])
                labels.append([1.0, 0.0])

        return np.array(triplets), np.array(labels)

    def __triplet_generator(self, x, y, batch_size):
        """
        Creates a python generator that produces triplets from the original input data.
        :param x: Input data
        :param y: Integer class labels
        :param batch_size: The number of triplet samples to create per batch.
        :return:
        """
        class_indices = self.__get_class_indices(y)
        while True:
            triplets, labels = self.__create_triplets(x, class_indices, batch_size)

            # The triplet network expects three inputs and one output. Split the triplets into a list of inputs.
            yield [triplets[:, 0], triplets[:, 1], triplets[:, 2]], labels

    def __get_class_indices(self, y):
        """
        Create a python list of lists that contains each of the indices in the input data that belong
        to each class. It is used to find and access elements in the input data that belong to a desired class.
        * Example usage:
        * element_index = class_indices[class][index]
        * element = x[element_index]
        :param y: Integer class labels
        :return: Python list of lists
        """
        return [np.where(y == i)[0] for i in range(self.num_classes)]

    @staticmethod
    def __randint_unequal(lower, upper):
        """
        Get two random integers that are not equal.

        Note: In some cases (such as there being only one sample of a class) there may be an endless loop here. This
        will only happen on fairly exotic datasets though. May have to address in future.
        :param lower: Lower limit inclusive of the random integer.
        :param upper: Upper limit inclusive of the random integer. Need to use -1 for random indices.
        :return: Tuple of (integer, integer)
        """
        int_1 = random.randint(lower, upper)
        int_2 = random.randint(lower, upper)
        while int_1 == int_2:
            int_1 = random.randint(lower, upper)
            int_2 = random.randint(lower, upper)
        return int_1, int_2
