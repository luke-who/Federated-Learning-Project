# Commented out IPython magic to ensure Python compatibility.
#@test {"skip": true}

# tensorflow_federated_nightly also bring in tf_nightly, which
# can causes a duplicate tensorboard install, leading to errors.
# !pip uninstall --yes tensorboard tb-nightly

# !pip install --quiet --upgrade tensorflow-federated
# !pip install --quiet --upgrade nest-asyncio
# !pip install --quiet --upgrade tensorboard  # or tb-nightly, but not both


import nest_asyncio
nest_asyncio.apply()

# %load_ext tensorboard

import collections

import numpy as np
import random
import time
import math
import json, ast
import pickle
import tensorflow as tf
import tensorflow_federated as tff

tff.federated_computation(lambda: 'Hello, World!')()


#*************************************#
#**********Define parameters**********#
#*************************************#
NUM_CLIENTS = 34
NUM_ROUNDS = 150
NUM_EPOCHS = 5
BATCH_SIZE = 20
SHUFFLE_BUFFER = 100
PREFETCH_BUFFER = 10




##############################################################
#### Create functions to preprocess & make federated data ####
##############################################################
def preprocess(dataset):

  def batch_format_fn(element):
    """Flatten a batch `pixels` and return the features as an `OrderedDict`."""
    return collections.OrderedDict(
        x=tf.reshape(element['pixels'], [-1, 784]),
        y=tf.reshape(element['label'], [-1, 1]))

  return dataset.repeat(NUM_EPOCHS).shuffle(SHUFFLE_BUFFER).batch(
      BATCH_SIZE).map(batch_format_fn).prefetch(PREFETCH_BUFFER)

def make_federated_train_data(client_data, client_ids):
  return [
      preprocess(client_data.create_tf_dataset_for_client(x))
      for x in client_ids
  ]

#########################################################################
#### Split the FEMNIST dataset into train set(90%) and test set(10%) ####
#########################################################################
emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()



##########################################################
#### Figuring out how many classes there are in total ####
##########################################################
def all_classes(dataset):
  dataset_client = dataset.create_tf_dataset_for_client(
    dataset.client_ids[0])
  num_data_client = len(dataset_client)
  client_dataset = []
  for n in range(num_data_client):
    client_dataset.append(list(dataset_client.as_numpy_iterator())[n]['label'])
  all_classes = np.unique(client_dataset)
  return all_classes

def train_data_client(client_id):
    train_data_client = emnist_train.create_tf_dataset_for_client(client_id)
    # num_data_client = len(train_data_client)
    return train_data_client

#################################################################
#### Make federeated train data for randomly sampled clients ####
#################################################################
# def make_federated_train_data(emnist_train, sample_clients):
#   emnist_train_selected_clients = []

#   for i,client_id in enumerate(sample_clients):
#     for j,c in enumerate(all_classes(emnist_train)):
#       #Gather data with class labels 0-9 separately
#       class_dataset = train_data_client(client_id).filter(lambda data: data['label']==c)
#       #Shuffle them
#       class_dataset = class_dataset.shuffle(len(train_data_client(client_id)))
#       # Gather datasets
#       if j==0:
#         emnist_train_client = class_dataset
#       elif j > 0:
#         emnist_train_client = emnist_train_client.concatenate(class_dataset).shuffle(len(train_data_client(client_id)))

#     # print(f'client {client_id} | total_num_data: {len(train_data_client(client_id))}')

#     #***********Transform/preprocess the train of all clients into federated type x(where x is an int represents the value of digit) and y(where y is a 1D 784 pixel values for this digit)***********#
#     emnist_train_selected_clients.append(preprocess(emnist_train_client))

#   return emnist_train_selected_clients


########################################################
#### Combine the 10% test data into 1 single client ####
########################################################
def make_federated_test_data(emnist_test):
  #iterate over all 3383 clients, then store them into list according to the order of clients respectively
  emnist_test_all_clients = emnist_test.create_tf_dataset_from_all_clients() #produces a dataset that contains all examples from a single client in order
  num_test_data = len(list(emnist_test_all_clients.as_numpy_iterator()))
  #Shuffle them
  emnist_test_all_clients = emnist_test_all_clients.shuffle(num_test_data)
  print(f'all clients => signle client | num_test_data: {num_test_data}')
    
  #***********Transform/preprocess the test data of all clients into x(where x is an int represents the value of digit) and y(where y is a 1D 784 pixel values for this digit)***********#
  emnist_test_all_clients = list(preprocess(emnist_test_all_clients))

  return emnist_test_all_clients


#*******************************************************************************#
#**********Select random sample clients and make federated traing data**********#
#*******************************************************************************#
##########################################################################################################################
#### Create extremely unbalanced data to compare the difference between NUM_EXAMPLES and UNIFORM for clients weighting####
##########################################################################################################################

#function to calculate total amount of data per client#
def total_data_client(client_index):
  if client_index >= 0 and client_index < len(emnist_train.client_ids):
    client_dataset = emnist_train.create_tf_dataset_for_client(emnist_train.client_ids[client_index])
    return len(client_dataset)
  else:
    print("client_index out of range!")

client_data_counted_list=[total_data_client(x) for x in range(len(emnist_train.client_ids))] #a list of the number of data in each client correspond to its index 

def make_unbalanced_federated_clients(num_clients):
  sample_clients_num_examples_vs_uniform = []

  for i, client_data_num in enumerate(client_data_counted_list):
      if client_data_num > 115:
          if len(sample_clients_num_examples_vs_uniform) < math.ceil(num_clients*0.1):
              sample_clients_num_examples_vs_uniform.append(emnist_train.client_ids[i])
  for i, client_data_num in enumerate(client_data_counted_list):
      if 25 < client_data_num < 52:
          if len(sample_clients_num_examples_vs_uniform) < num_clients:
              sample_clients_num_examples_vs_uniform.append(emnist_train.client_ids[i])
  # print(len(sample_clients_num_examples_vs_uniform))
  # print(sample_clients_num_examples_vs_uniform)
  random.shuffle(sample_clients_num_examples_vs_uniform)
  # print(sample_clients_num_examples_vs_uniform)
  return sample_clients_num_examples_vs_uniform

##################################
#### Make federated test data ####
##################################
emnist_test_all_clients = make_federated_test_data(emnist_test)



"""### Creating a model with Keras(`tf.keras.Model`)

"""

def create_keras_model():
  return tf.keras.models.Sequential([
      tf.keras.layers.Input(shape=(784,)),
      tf.keras.layers.Dense(10, kernel_initializer='zeros'),
      tf.keras.layers.Softmax(),
  ])

def model_fn():
  # We _must_ create a new model here, and _not_ capture it from an external
  # scope. TFF will call this within different graph contexts.
  keras_model = create_keras_model()
  return tff.learning.from_keras_model(
      keras_model,
      input_spec=emnist_train[0].element_spec,
      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

"""### Creating & Customizing the model implementation with `tff.learning.Model `

#### Defining model variables, forward pass, and metrics
"""

MnistVariables = collections.namedtuple(
    'MnistVariables', 'weights bias num_examples loss_sum accuracy_sum')

def create_mnist_variables():
  return MnistVariables(
      weights=tf.Variable(
          lambda: tf.zeros(dtype=tf.float32, shape=(784, 10)),
          name='weights',
          trainable=True),
      bias=tf.Variable(
          lambda: tf.zeros(dtype=tf.float32, shape=(10)),
          name='bias',
          trainable=True),
      num_examples=tf.Variable(0.0, name='num_examples', trainable=False),
      loss_sum=tf.Variable(0.0, name='loss_sum', trainable=False),
      accuracy_sum=tf.Variable(0.0, name='accuracy_sum', trainable=False))

def mnist_forward_pass(variables, batch):
  y = tf.nn.softmax(tf.matmul(batch['x'], variables.weights) + variables.bias)
  predictions = tf.cast(tf.argmax(y, 1), tf.int32)

  flat_labels = tf.reshape(batch['y'], [-1])
  loss = -tf.reduce_mean(
      tf.reduce_sum(tf.one_hot(flat_labels, 10) * tf.math.log(y), axis=[1]))
  accuracy = tf.reduce_mean(
      tf.cast(tf.equal(predictions, flat_labels), tf.float32))

  num_examples = tf.cast(tf.size(batch['y']), tf.float32)

  variables.num_examples.assign_add(num_examples)
  variables.loss_sum.assign_add(loss * num_examples)
  variables.accuracy_sum.assign_add(accuracy * num_examples)

  return loss, predictions

def get_local_mnist_metrics(variables):
  return collections.OrderedDict(
      num_examples=variables.num_examples,
      loss=variables.loss_sum / variables.num_examples,
      accuracy=variables.accuracy_sum / variables.num_examples)

@tff.federated_computation
def aggregate_mnist_metrics_across_clients(metrics):
  return collections.OrderedDict(
      num_examples=tff.federated_sum(metrics.num_examples),
      loss=tff.federated_mean(metrics.loss, metrics.num_examples),
      accuracy=tff.federated_mean(metrics.accuracy, metrics.num_examples))

"""#### Constructing an instance of `tff.learning.Model`"""

class MnistModel(tff.learning.Model):

  def __init__(self):
    self._variables = create_mnist_variables()

  @property
  def trainable_variables(self):
    return [self._variables.weights, self._variables.bias]

  @property
  def non_trainable_variables(self):
    return []

  @property
  def local_variables(self):
    return [
        self._variables.num_examples, self._variables.loss_sum,
        self._variables.accuracy_sum
    ]

  @property
  def input_spec(self):
    return collections.OrderedDict(
        x=tf.TensorSpec([None, 784], tf.float32),
        y=tf.TensorSpec([None, 1], tf.int32))

  @tf.function
  def forward_pass(self, batch, training=True):
    del training
    loss, predictions = mnist_forward_pass(self._variables, batch)
    num_exmaples = tf.shape(batch['x'])[0]
    return tff.learning.BatchOutput(
        loss=loss, predictions=predictions, num_examples=num_exmaples)

  @tf.function
  def report_local_outputs(self):
    return get_local_mnist_metrics(self._variables)

  @property
  def federated_output_computation(self):
    return aggregate_mnist_metrics_across_clients


#################################################
#### Federated learning with the Mnist model ####
#################################################
iterative_process = tff.learning.build_federated_averaging_process(
    MnistModel,
    client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.02),
    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1),
    client_weighting=tff.learning.ClientWeighting.UNIFORM) # change this line to client_weighting=tff.learning.ClientWeighting.NUM_EXAMPLES to try different weighting strategy

global_accuracy = []
all_sampled_clients = []
start_time = time.time()

# sample_clients = np.random.choice(emnist_train.client_ids, NUM_CLIENTS)
sample_clients = make_unbalanced_federated_clients(NUM_CLIENTS)
all_sampled_clients.append(sample_clients)
emnist_train_selected_clients = make_federated_train_data(emnist_train,sample_clients)

state = iterative_process.initialize()
state, metrics = iterative_process.next(state, emnist_train_selected_clients)
# model_weights = iterative_process.get_model_weights(state)

evaluation = tff.learning.build_federated_evaluation(MnistModel)
global_validation_metrics = evaluation(state.model, [emnist_test_all_clients])
global_accuracy.append(global_validation_metrics['accuracy']) 
print('round  1, global_accuracy={}'.format(global_validation_metrics['accuracy']))

for round_num in range(2, NUM_ROUNDS+1):
  # sample_clients = np.random.choice(emnist_train.client_ids, NUM_CLIENTS)
  # all_sampled_clients.append(sample_clients)
  emnist_train_selected_clients = make_federated_train_data(emnist_train,sample_clients)

  state, metrics = iterative_process.next(state, emnist_train_selected_clients)

  global_validation_metrics = evaluation(state.model, [emnist_test_all_clients])
  global_accuracy.append(global_validation_metrics['accuracy']) 
  print('round {:2d}, global_accuracy={}'.format(round_num, global_validation_metrics['accuracy']))



stop_time = time.time()
print(f'finished training in {stop_time - start_time}s')


########################################################
#### Store data into files after Federated Learning ####
########################################################
# -----get sampled_clients stored in a .json file-----#
with open(f"metrics/num_examples_vs_uniform/{NUM_CLIENTS}_clients_{NUM_ROUNDS}_rounds_{NUM_EPOCHS}_epochs.json", 'w') as f:
    # indent=2 is not needed but makes the file human-readable
    json.dump(str(all_sampled_clients), f, indent=2) 
# with open(f"metrics/num_examples_vs_uniform/{NUM_CLIENTS}_clients_{NUM_ROUNDS}_rounds_{NUM_EPOCHS}_epochs.json", 'r') as f:
#     all_sampled_clients = json.load(f)
#     all_sampled_clients = all_sampled_clients.replace(" ",",")
#     all_sampled_clients = ast.literal_eval(all_sampled_clients)
    
#-----training NUM_ROUNDS to get gloal accuracy list stored in a .txt file-----#
with open(f"metrics/num_examples_vs_uniform/{NUM_CLIENTS}_clients_uniform_weights_{NUM_ROUNDS}_rounds_{NUM_EPOCHS}_epochs_accuracy_global.txt","wb") as fp: #pickling
    pickle.dump(global_accuracy,fp)
# with open(f"metrics/num_examples_vs_uniform/{NUM_CLIENTS}_clients_uniform_weights_{NUM_ROUNDS}_rounds_{NUM_EPOCHS}_epochs_accuracy_global.txt","rb") as fp: #unpickling
#     global_accuracy = pickle.load(fp)



