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
import math
import time
import json, ast
import pickle
import argparse
import tensorflow as tf
import tensorflow_federated as tff

tff.federated_computation(lambda: 'Hello, World!')()


#*************************************#
#**********Define parameters**********#
#*************************************#
NUM_CLIENTS = 338
orignal_num_clients = NUM_CLIENTS
NUM_ROUNDS = 150
NUM_EPOCHS = 5
BATCH_SIZE = 20
SHUFFLE_BUFFER = 100
PREFETCH_BUFFER = 10


MODES = ["constant","exponential","linear","sigmoid","reciprocal"]

##############################################################
#### Create functions to preprocess & make federated data ####
##############################################################
def preprocess(dataset):
  """
  Preprocess the dataset.
  
  Args:
      dataset : FMNIST dataset
  Returns:
      preprocessed FMNIST data
  """
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

TOTAL_NUM_CLIENTS = len(emnist_train.client_ids)

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


def vary_num_clients(mode,max_num_clients,min_num_clients,num_rounds):
  # if mode == "constant" and max_num_clients == min_num_clients:
  if mode == "constant":
    num_clients_head = [max_num_clients]*num_rounds
  elif mode == "exponential":
    num_clients_head = [int(-np.exp((x-1)/10.3)+max_num_clients) for x in range(61)]
  elif mode == "linear":
    num_clients_head = [int(-5.065*x+max_num_clients) for x in range(61)]
  elif mode == "sigmoid":
    num_clients_head = [int(-304/(1+np.exp(-0.26*(x-20)))+max_num_clients) for x in range(61)]
    # num_clients_head = [int(-304/(1+np.exp(-0.3*(x-30)))+max_num_clients) for x in range(61)]
  elif mode == "reciprocal":
    num_clients_head = [int(50/x+min_num_clients) for x in [0.164] + list(np.arange(1,61))]

  num_clients_tail = [min_num_clients]*(num_rounds-len(num_clients_head))
  num_clients = num_clients_head + num_clients_tail

  #turn it into a generator/iteratorprint("num_updates:{}".format(NUM_ROUNDS*TOTAL_NUM_CLIENTS-sum(num_clients)))
  # num_clients = (n for n in num_clients)  #generator
  # num_clients = iter(num_clients) #iterator
  return num_clients



STOP_ROUND = 0
#################################################
#### Federated learning with the Mnist model ####
#################################################
def fl_iterative(num_clients,mode):
  iterative_process = tff.learning.build_federated_averaging_process(
      MnistModel,
      client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.02),
      server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1))
      

  global_accuracy = []
  all_sampled_clients = []
  start_time = time.time()

  sample_clients = np.random.choice(emnist_train.client_ids, num_clients[0])
  all_sampled_clients.append(sample_clients)
  emnist_train_selected_clients = make_federated_train_data(emnist_train,sample_clients)

  state = iterative_process.initialize()
  state, metrics = iterative_process.next(state, emnist_train_selected_clients)
  # model_weights = iterative_process.get_model_weights(state)

  evaluation = tff.learning.build_federated_evaluation(MnistModel)
  global_validation_metrics = evaluation(state.model, [emnist_test_all_clients])
  global_accuracy.append(global_validation_metrics['accuracy']) 
  print('round  1, global_accuracy={}, num_clients={}'.format(global_validation_metrics['accuracy'],num_clients[0]))

  for round_num in range(2, NUM_ROUNDS+1):
      sample_clients = np.random.choice(emnist_train.client_ids, num_clients[round_num-1])
      all_sampled_clients.append(sample_clients)
      emnist_train_selected_clients = make_federated_train_data(emnist_train,sample_clients)

      state, metrics = iterative_process.next(state, emnist_train_selected_clients)

      global_validation_metrics = evaluation(state.model, [emnist_test_all_clients])
      global_accuracy.append(global_validation_metrics['accuracy']) 
      print('round {:2d}, global_accuracy={}, num_clients={}'.format(round_num, global_validation_metrics['accuracy'],num_clients[round_num-1]))

      if math.floor(global_validation_metrics['accuracy']*100)/100 == 0.8:
          STOP_ROUND = round_num
          break
      else:
          continue



  stop_time = time.time()
  training_time = stop_time - start_time
  final_num_clients = num_clients[STOP_ROUND-1]
  print(f'num_clients {mode} reductioin:  {orignal_num_clients} -> {final_num_clients} clients')
  print(f'finished training in {training_time}s')


  return all_sampled_clients, global_accuracy, final_num_clients, training_time


#############################################################################################################################################################
############################################### Store data into files after Federated Learning ##############################################################
#############################################################################################################################################################
def store_results(mode, num_clients, all_sampled_clients, global_accuracy, final_num_clients,training_time):
  #*******************************************************************************************#
  #**********************get all_sampled_clients stored in a .json file***********************#
  #*******************************************************************************************#
  with open(f"metrics/vary_num_clients/{orignal_num_clients} -> {final_num_clients} clients_{mode}.json", 'w') as f:
      # indent=2 is not needed but makes the file human-readable
      json.dump(str(all_sampled_clients), f, indent=2) 
  with open(f"metrics/vary_num_clients/{orignal_num_clients} -> {final_num_clients} clients_{mode}.json", 'r') as f:
      all_sampled_clients = json.load(f)
      all_sampled_clients = all_sampled_clients.replace(" ",",")
      all_sampled_clients = ast.literal_eval(all_sampled_clients)

  #*****************************************************************************************************************#
  #**********************training NUM_ROUNDS to get gloal accuracy list stored in a .txt file***********************#
  #*****************************************************************************************************************#
  with open(f"metrics/vary_num_clients_and_rounds/{orignal_num_clients} -> {final_num_clients} clients_{mode}_accuracy_global.txt","wb") as fp: #pickling
      pickle.dump(global_accuracy,fp)
  with open(f"metrics/vary_num_clients_and_rounds/{orignal_num_clients} -> {final_num_clients} clients_{mode}_accuracy_global.txt","rb") as fp: #unpickling
      global_accuracy = pickle.load(fp)



  #####################################################################################
  ############## Calculate selected mode's model updates & percentage #################
  #####################################################################################

  #***********************************************************************************************************************************#
  #**********************store pushed_model_updates_percentage & pushed_model_updates_percentage in a .txt file***********************#
  #***********************************************************************************************************************************#

  f = open(f"metrics/vary_num_clients_and_rounds/pushed_model_updates.json", 'r')
  pushed_model_updates = json.load(f)
  f.close()
  pushed_model_updates[f"{mode}"] = sum(num_clients[:STOP_ROUND])
  f = open(f"metrics/vary_num_clients_and_rounds/pushed_model_updates.json", 'w')
  # indent=2 is not needed but makes the file human-readable
  json.dump(pushed_model_updates, f, indent=2) 
  f.close()


  f = open(f"metrics/vary_num_clients_and_rounds/pushed_model_updates_percentage.json", 'r')
  pushed_model_updates_percentage = json.load(f)
  f.close()
  pushed_model_updates_percentage[f"{mode}"] = (sum(num_clients[:STOP_ROUND])/(STOP_ROUND*TOTAL_NUM_CLIENTS))*100
  f = open(f"metrics/vary_num_clients_and_rounds/pushed_model_updates_percentage.json", 'w')
  # indent=2 is not needed but makes the file human-readable
  json.dump(pushed_model_updates_percentage, f, indent=2) 
  f.close()

  print("pushed_model_updates: {}".format(pushed_model_updates))
  print("pushed_model_updates_percentage: {}".format(pushed_model_updates_percentage))


  # #****************************************************************************************************#
  # #**********************store stopped rounds for diffferent mode in a .json file**********************#
  # #****************************************************************************************************#
  f = open(f"metrics/vary_num_clients_and_rounds/modes_stopped_round.json", 'r')
  stopped_rounds = json.load(f)
  f.close()

  stopped_rounds[mode] = STOP_ROUND

  f = open(f"metrics/vary_num_clients_and_rounds/modes_stopped_round.json", 'w')
  # indent=2 is not needed but makes the file human-readable
  json.dump(stopped_rounds, f, indent=2) 
  f.close()


  #*********************************************************************************#
  #**********************store traning time(s) in a .json file**********************#
  #*********************************************************************************#
  f = open(f"metrics/vary_num_clients_and_rounds/modes_training_time.json", 'r')
  training_times = json.load(f)
  f.close()

  training_times[mode] = training_time

  f = open(f"metrics/vary_num_clients_and_rounds/modes_training_time.json", 'w')
  # indent=2 is not needed but makes the file human-readable
  json.dump(training_times, f, indent=2) 
  f.close()












def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', nargs=1, type=str, help='Running mode. Must be one of the following modes: {}'.format(MODES))
    
    args = parser.parse_args()
    mode = args.mode[0]
    
    return args, mode


if __name__ == '__main__':
    args, mode = parse_args() # get argument from the command line
    
    # load the data
    print(f'selected mode: {mode}')
    if mode == 'constant':
        num_clients = vary_num_clients(mode,NUM_CLIENTS,34,NUM_ROUNDS)
        all_sampled_clients, global_accuracy, final_num_clients, training_time = fl_iterative(num_clients,mode)
        store_results(mode, num_clients, all_sampled_clients, global_accuracy, final_num_clients,training_time)
    elif mode == 'exponential':
        num_clients = vary_num_clients(mode,NUM_CLIENTS,34,NUM_ROUNDS)
        all_sampled_clients, global_accuracy, final_num_clients, training_time = fl_iterative(num_clients,mode)
        store_results(mode, num_clients, all_sampled_clients, global_accuracy, final_num_clients,training_time)
    elif mode == 'linear':
        num_clients = vary_num_clients(mode,NUM_CLIENTS,34,NUM_ROUNDS)
        all_sampled_clients, global_accuracy, final_num_clients, training_time = fl_iterative(num_clients,mode)
        store_results(mode, num_clients, all_sampled_clients, global_accuracy, final_num_clients,training_time)
    elif mode == 'sigmoid':
        num_clients = vary_num_clients(mode,NUM_CLIENTS,34,NUM_ROUNDS)
        all_sampled_clients, global_accuracy, final_num_clients, training_time = fl_iterative(num_clients,mode)
        store_results(mode, num_clients, all_sampled_clients, global_accuracy, final_num_clients,training_time)
    elif mode == 'reciprocal':
        num_clients = vary_num_clients(mode,NUM_CLIENTS,34,NUM_ROUNDS)
        all_sampled_clients, global_accuracy, final_num_clients, training_time = fl_iterative(num_clients,mode)
        store_results(mode, num_clients, all_sampled_clients, global_accuracy, final_num_clients,training_time)
    else:
        raise Exception('Unrecognised mode: {}. Possible modes are: {}'.format(mode, MODES))