import os

import keras.backend as K
from keras.models import Model, Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Input, Subtract, Lambda
from keras.optimizers import Adam, SGD
from keras.regularizers import l2
import keras.backend as K
from keras.callbacks import LearningRateScheduler

import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from modified_sgd import Modified_SGD


class SiameseNetwork:
    """Class that constructs the Siamese Net for training

    This Class was constructed to create the siamese net and train it.

    Attributes:
        input_shape: image size
        model: current siamese model
        learning_rate: SGD learning rate
        summary_writer: tensorflow writer to store the logs
    """

    def __init__(self,  learning_rate, batch_size, epochs,
                 learning_rate_multipliers, l2_regularization_penalization, tensorboard_log_path):
        """Inits SiameseNetwork with the provided values for the attributes.

        It also constructs the siamese network architecture, creates a dataset 
        loader and opens the log file.

        Arguments:
            dataset_path: path of Omniglot dataset    
            learning_rate: SGD learning rate
            batch_size: size of the batch to be used in training
            use_augmentation: boolean that allows us to select if data augmentation 
                is used or not
            learning_rate_multipliers: learning-rate multipliers (relative to the learning_rate
                chosen) that will be applied to each fo the conv and dense layers
                for example:
                    # Setting the Learning rate multipliers
                    LR_mult_dict = {}
                    LR_mult_dict['conv1']=1
                    LR_mult_dict['conv2']=1
                    LR_mult_dict['dense1']=2
                    LR_mult_dict['dense2']=2
            l2_regularization_penalization: l2 penalization for each layer.
                for example:
                    # Setting the Learning rate multipliers
                    L2_dictionary = {}
                    L2_dictionary['conv1']=0.1
                    L2_dictionary['conv2']=0.001
                    L2_dictionary['dense1']=0.001
                    L2_dictionary['dense2']=0.01
            tensorboard_log_path: path to store the logs                
        """
        self.input_shape = (256, 256, 3)  # Size of images  !!!
        self.model = []
        self.batch_size = batch_size       # !!!
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.summary_writer = tf.summary.FileWriter(tensorboard_log_path)
        self.__construct_siamese_architecture(learning_rate_multipliers,
                                              l2_regularization_penalization)

    def __construct_siamese_architecture(self, learning_rate_multipliers,
                                         l2_regularization_penalization):
        """ Constructs the siamese architecture and stores it in the class

        Arguments:
            learning_rate_multipliers
            l2_regularization_penalization
        """

        # Let's define the cnn architecture
        convolutional_net = Sequential()
        convolutional_net.add(Conv2D(filters=64, kernel_size=(10, 10),
                                     activation='relu',
                                     input_shape=self.input_shape,
                                     kernel_regularizer=l2(
                                         l2_regularization_penalization['Conv1']),
                                     name='Conv1'))
        convolutional_net.add(MaxPool2D())

        convolutional_net.add(Conv2D(filters=128, kernel_size=(7, 7),
                                     activation='relu',
                                     kernel_regularizer=l2(
                                         l2_regularization_penalization['Conv2']),
                                     name='Conv2'))
        convolutional_net.add(MaxPool2D())

        convolutional_net.add(Conv2D(filters=128, kernel_size=(4, 4),
                                     activation='relu',
                                     kernel_regularizer=l2(
                                         l2_regularization_penalization['Conv3']),
                                     name='Conv3'))
        convolutional_net.add(MaxPool2D())

        convolutional_net.add(Conv2D(filters=256, kernel_size=(4, 4),
                                     activation='relu',
                                     kernel_regularizer=l2(
                                         l2_regularization_penalization['Conv4']),
                                     name='Conv4'))
        convolutional_net.add(MaxPool2D())

        convolutional_net.add(Flatten())
        convolutional_net.add(
            Dense(units=4096, activation='sigmoid',
                  kernel_regularizer=l2(
                      l2_regularization_penalization['Dense1']),
                  name='Dense1'))

        # Now the pairs of images
        input_image_1 = Input(self.input_shape)
        input_image_2 = Input(self.input_shape)

        encoded_image_1 = convolutional_net(input_image_1)
        encoded_image_2 = convolutional_net(input_image_2)

        # L1 distance layer between the two encoded outputs
        # One could use Subtract from Keras, but we want the absolute value
        l1_distance_layer = Lambda(
            lambda tensors: K.abs(tensors[0] - tensors[1]))
        l1_distance = l1_distance_layer([encoded_image_1, encoded_image_2])

        # Same class or not prediction
        prediction = Dense(units=1, activation='sigmoid')(l1_distance)
        self.model = Model(
            inputs=[input_image_1, input_image_2], outputs=prediction)

        # Define the optimizer and compile the model
        optimizer = Modified_SGD(
            lr=self.learning_rate,
            lr_multipliers=learning_rate_multipliers,
            momentum=0.5)

        self.model.compile(loss='binary_crossentropy', metrics=['binary_accuracy'],
                           optimizer=optimizer)

    def __write_logs_to_tensorboard(self, current_iteration, train_losses,
                                    train_accuracies, validation_accuracy,
                                    evaluate_each):
        """ Writes the logs to a tensorflow log file

        This allows us to see the loss curves and the metrics in tensorboard.
        If we wrote every iteration, the training process would be slow, so 
        instead we write the logs every evaluate_each iteration.

        Arguments:
            current_iteration: iteration to be written in the log file
            train_losses: contains the train losses from the last evaluate_each
                iterations.
            train_accuracies: the same as train_losses but with the accuracies
                in the training set.
            validation_accuracy: accuracy in the current one-shot task in the 
                validation set
            evaluate each: number of iterations defined to evaluate the one-shot
                tasks.
        """

        summary = tf.Summary()

        # Write to log file the values from the last evaluate_every iterations
        for index in range(0, evaluate_each):
            value = summary.value.add()
            value.simple_value = train_losses[index]
            value.tag = 'Train Loss'

            value = summary.value.add()
            value.simple_value = train_accuracies[index]
            value.tag = 'Train Accuracy'

            if index == (evaluate_each - 1):
                value = summary.value.add()
                value.simple_value = validation_accuracy
                value.tag = 'One-Shot Validation Accuracy'

            self.summary_writer.add_summary(
                summary, current_iteration - evaluate_each + index + 1)
            self.summary_writer.flush()

    def scheduler(self,epoch):
        if epoch % 500 == 0 and epoch != 0:
            lr = K.get_value(self.model.optimizer.lr)
            K.set_value(self.model.optimizer.lr, lr * 0.1)
            print("lr changed to {}".format(lr * 0.1))
        return K.get_value(self.model.optimizer.lr)

    def train_siamese_network(self, model_name, train_data, train_label, val_data, val_label):

        reduce_lr = LearningRateScheduler(self.scheduler)
        self.model.fit(train_data, train_label, batch_size=self.batch_size, epochs=self.epochs, verbose=1,
                        validation_data=(val_data, val_label))
        # callbacks = [reduce_lr],

        model_json = self.model.to_json()

        if not os.path.exists('./models'):
            os.makedirs('./models')
        with open('models/' + model_name + '.json', "w") as json_file:
            json_file.write(model_json)
        self.model.save_weights('models/' + model_name + '.h5')

        # print(self.model.predict([train_data[0][1:10, :, :, :], train_data[1][1:10, :, :, :]]))
        # print(self.model.predict([train_data[0][10000:10010, :, :, :], train_data[1][10000:10010, :, :, :]]))
