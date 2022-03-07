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
import pickle
from contextlib import redirect_stdout

class TurboModel():
    def __init__(self,
                 path_to_train_dataset,
                 path_to_val_dataset,
                 path_to_test_set,
                 folder_to_store_results,
                 model_name = 'Simple_Model_CNN_16_32_64'):
        """
            path_to_train_dataset is the path to the train dataset where there should be two folders:
            fake and real.
            path_to_val_dataset is the path to the val dataset where there should be two folders:
            fake and real.
            path_to_test_set is the path to the test dataset where there should be two folders:
            fake and real.
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
                                self.path_to_train_dataset,
                                validation_split=validation_split,
                                subset="training",
                                seed=123,
                                image_size=(224, 224),
                                batch_size=batch_size)
        self.class_names_train_ds = self.train_ds.class_names
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
        self.class_names_val_ds = self.val_ds.class_names
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
        self.class_names_test_ds = self.test_ds.class_names
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
        Saves results (curves and model.evaluate in the folder specified in the init.)
        '''
        metrics_per_pair = {'loss':[],
                         'accuracy':[],
                         'recall':[],
                         'precision':[],
                         'auc':[]
        }
        #special treatment for the history dict because the keys can vary.
        for key, value in self.history.history.items():
            for key_per_pair in metrics_per_pair.keys():
                #if the key is matching the key_per_pair
                if key_per_pair in key:
                    #it will first append the train values in index 0.
                    #it will then append the val values in index 1.
                    metrics_per_pair[key_per_pair].append(value)
        # i is useful for index in the .evaluate.
        scores = self.model.evaluate(self.test_ds)
        i = 0
        #creating all the graphs and saving them as pdf.
        for key, value in metrics_per_pair.items():
            plt.close()
            plt.plot([i + 1 for i in range(len(value[0]))],value[0], '-b', label='Train')
            plt.plot([i + 1 for i in range(len(value[0]))],value[1], '-r', label='Val')
            plt.title(f'Graph of {key}. Value on the test set : {round(scores[i],3)}')
            plt.ylabel(key)
            plt.xlabel('Epoch')
            plt.legend()
            plt.savefig(self.folder_to_store_results + f'{key}.pdf')
            i += 1
        #actually saving the .evaluate.
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
    def save_history_in_pickle(self):
        '''
        Saves the history in a .pickle file in the folder specified in the init.
        '''
        with open(self.folder_to_store_results + 'history.pickle', 'wb') as handle:
            pickle.dump(self.history, handle, protocol=pickle.HIGHEST_PROTOCOL)
    def save_summary(self):
        '''
        Saves the summary of the model in a .txt file in the folder specified in the init.
        '''
        with open(self.folder_to_store_results + 'model_summary.txt', 'w') as f:
            with redirect_stdout(f):
                self.model.summary()
    def save_classes_in_pickle(self):
        '''
        Saves the classes of the train, val and test sets in a .pickle file in the folder specified in the init.
        '''
        with open(self.folder_to_store_results + 'train_ds_classes.pickle', 'wb') as handle:
            pickle.dump(self.class_names_train_ds, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(self.folder_to_store_results + 'val_ds_classes.pickle', 'wb') as handle:
            pickle.dump(self.class_names_val_ds, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(self.folder_to_store_results + 'test_ds_classes.pickle', 'wb') as handle:
            pickle.dump(self.class_names_test_ds, handle, protocol=pickle.HIGHEST_PROTOCOL)
    def save_results(self):
        '''
        Creates the folder specified in the init.
        Then saves results (the model.joblib + hyperparams + curves + model.evaluate in the folder
        specified in the init.)
        '''
        os.mkdir(self.folder_to_store_results)
        self.save_model()
        self.save_curves()
        self.save_hyperparams()
        self.save_history_in_pickle()
        self.save_summary()
        self.save_classes_in_pickle()

# ls Data_test/fake_test | wc 3 830
# ls Data_test/real_test | wc 3 830
# ls Data_train/fake/ | wc 10 727
# ls Data_train/real/ | wc 10 727
# ls Data_val/fake_val | wc 4 597
# ls Data_test/real_val | wc 4 597
#somme = 38 308
