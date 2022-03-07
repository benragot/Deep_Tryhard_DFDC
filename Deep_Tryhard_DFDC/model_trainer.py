'''
This module is meant to train several models of 'simple' CNN based on an architecture
found here :
https://www.sciencedirect.com/science/article/pii/S2667096821000471
The aim is to classify faces as deep fake or real.
'''
import joblib
from tqdm import tqdm
from tensorflow.data import AUTOTUNE
from tensorflow.keras import layers, models
from tensorflow.keras.metrics import Recall, Precision, AUC
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import os

class TurboModel():
    def __init__(self,
                 path_to_train_dataset,
                 path_to_val_dataset,
                 path_to_test_set,
                 folder_to_store_results,
                 model_name = 'Simple_Model_CNN_16_32_64'):
        """
            path_to_train_val_dataset is the path to the train val dataset where there should be two folders:
            FAKE and REAL.
            path_to_test_set is the path to the test dataset where there should be two folders:
            FAKE and REAL.
            folder_to_store_results is the name of the folder where you want the results to be
            stored. No need to create it, it will be created by the module.
            Don't forget the '/' at the end !
            model_name must be explicit and different each time, it will be
            used to name the saved model.
        """
        self.model_name = model_name
        self.path_to_train_dataset = path_to_train_dataset
        self.path_to_val_dataset = path_to_val_dataset
        self.path_to_test_set = path_to_test_set
        self.folder_to_store_results = folder_to_store_results
        self.model_hyperparams = {}
        self.model = None

    def initialize_model(self,
                        kernel_size_Conv2d = (3,3),
                        max_pool_size = (2,2),
                        numbers_of_filters = [16,32,64],
                        dense_layers = [32],
                        dropout = 0.2):
        '''
        This function initializes a CNN model with this architecture :
        len(numbers_of_filters) * Conv2D layers/BatchNormalization/MaxPool2D
        with parameters kernel_size_Conv2d,numbers_of_filters and max_pool_size.
        Then, len(dense_layers) of dense layers/BatchNormalization/Dropout
        with dense_layers[i] of neurons and specific dropout.
        '''
        self.model_hyperparams['kernel_size_Conv2d'] = kernel_size_Conv2d
        self.model_hyperparams['max_pool_size'] = max_pool_size
        self.model_hyperparams['numbers_of_filters'] = numbers_of_filters
        self.model_hyperparams['dense_layers'] = dense_layers
        self.model_hyperparams['dropout'] = dropout

        self.model = models.Sequential()
        ### First convolution & max-pooling
        for numbers_of_filter in numbers_of_filters:
            self.model.add(layers.Conv2D(numbers_of_filter, kernel_size_Conv2d,input_shape=(224, 224, 3), activation="relu"))
            self.model.add(layers.BatchNormalization())
            self.model.add(layers.MaxPool2D(pool_size=max_pool_size))
        ### Flattening
        self.model.add(layers.Flatten())
        ### fully connected
        for dense_layer in dense_layers:
            self.model.add(layers.Dense(dense_layer, activation='relu'))
            self.model.add(layers.BatchNormalization())
            self.model.add(layers.Dropout(rate=dropout))
        ### Last layer (A classification with 1 output)
        self.model.add(layers.Dense(1, activation='sigmoid'))
        return self
    def compile_model(self):
        '''
        Simple compilation of the model.
        Uses Adam with no standard parameters and a binary crossentropy loss function.
        Aka LogLoss.
        '''
        ### Model compilation
        self.model.compile(loss='binary_crossentropy',
                            optimizer='adam',
                            metrics=['accuracy',Recall(),Precision(),AUC()])
        return self
    def create_train_set(self, batch_size = 16,
                              validation_split=0.2):
        '''
        Creates a train set to do batch per batch trainings.
        The path to the dataset used is the one specified in the __init__.
        You can specify a batch_size and a validation split.
        '''
        self.model_hyperparams['batch_size'] = batch_size
        self.model_hyperparams['validation_split'] = validation_split
        self.train_ds = image_dataset_from_directory(
                                self.path_to_train_val_dataset,
                                validation_split=validation_split,
                                subset="training",
                                seed=123,
                                image_size=(224, 224),
                                batch_size=batch_size)
        self.train_ds = self.train_ds.prefetch(buffer_size=AUTOTUNE)
        return self
    def create_val_set(self, batch_size = 8,
                              validation_split=0.05):
        '''
        Creates a tensorflow batch per batch val set.
        The path to the dataset used is the one specified in the __init__.
        You can specify a batch_size and a validation split.
        '''
        self.val_ds = image_dataset_from_directory(
                                self.path_to_val_dataset,
                                validation_split=validation_split,
                                subset="training",
                                seed=123,
                                image_size=(224, 224),
                                batch_size=batch_size)
        self.val_ds = self.val_ds.prefetch(buffer_size=AUTOTUNE)
        return self
    def create_test_set(self, batch_size = 8,
                              validation_split=0.1):
        '''
        Creates a tensorflow batch per batch test set.
        The path to the dataset used is the one specified in the __init__.
        You can specify a batch_size and a validation split.
        '''
        self.test_ds = image_dataset_from_directory(
                                self.path_to_test_set,
                                validation_split=validation_split,
                                subset="training",
                                seed=123,
                                image_size=(224, 224),
                                batch_size=batch_size)
        self.test_ds = self.test_ds.prefetch(buffer_size=AUTOTUNE)
        return self
    def train_model(self, epochs = 20,
                    patience = 5,
                    verbose = 1):
        '''
        Trains the model based on epochs and patience.
        '''
        self.model_hyperparams['epochs'] = epochs
        self.model_hyperparams['patience'] = patience
        self.model_hyperparams['verbose'] = verbose
        es = EarlyStopping(patience=patience, restore_best_weights=True)
        self.history = self.model.fit(self.train_ds,
                epochs=epochs,
                validation_data=self.val_ds,
                verbose=verbose,
                callbacks = [es])
        return self
    def save_curves(self):
        '''
        Saves results (curves + model.evaluate in the folder specified in the init.)
        '''
        #special treatment for the history dict because the keys can vary.
        for key, value in self.history.history.items():
            if 'val' in key:
                #it's validation set value.
                plt.close()
                plt.plot(value)
                plt.title(f'Validation set : {key}')
                plt.ylabel(key)
                plt.xlabel('Epoch')
                plt.savefig(self.folder_to_store_results + f'val_set_{key}.pdf')
            else:
                #it's train set value.
                plt.close()
                plt.plot(value)
                plt.title(f'Train set : {key}')
                plt.ylabel(key)
                plt.xlabel('Epoch')
                plt.savefig(self.folder_to_store_results + f'train_set_{key}.pdf')

    def save_evaluate(self):
        '''
        Saves results (curves + model.evaluate in the folder specified in the init.)
        '''
        scores = self.model.evaluate(self.test_ds)
        text_file = open(self.folder_to_store_results + "model_evaluate.txt", "w")
        n = text_file.write(f'''Test loss : {scores[0]}\
                            \nTest accuracy : {scores[1]}\
                            \nTest recall : {scores[2]}\
                            \nTest precision : {scores[3]}\
                            \nTest AUC : {scores[4]}''')
        text_file.close()
    def save_hyperparams(self):
        '''
        Saves the hyperparams in the folder specified in the init.
        '''
        text_file = open(self.folder_to_store_results + "hyperparams.txt", "w")
        for key, value in self.model_hyperparams.items():
            text_file.write(f'{key} : {value}\n')
        text_file.close()
    def save_model(self):
        '''
        Saves the model in the folder specified in the init.
        '''
        joblib.dump(self.model, self.folder_to_store_results + self.model_name + '.joblib')
    def save_summary(self):
        '''
        Saves the model in the folder specified in the init.
        '''
        def myprint(s):
            with open(self.folder_to_store_results + 'model_summary.txt','w+') as f:
                print(s, file=f)
        self.model.summary(print_fn=myprint)
    def save_results(self):
        '''
        Creates the folder specified in the init.
        Then saves results (the model.joblib + hyperparams + curves + model.evaluate in the folder
        specified in the init.)
        '''
        os.mkdir(self.folder_to_store_results)
        self.save_model()
        self.save_evaluate()
        self.save_curves()
        self.save_hyperparams()
        self.save_summary()
