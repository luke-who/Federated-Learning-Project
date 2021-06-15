# from tff import NUM_EPOCHS
from matplotlib import pyplot as plt
import json, ast
import math
import numpy as np
import pickle
import argparse
import pandas as pd
import seaborn as sns
import collections
import tensorflow_federated as tff
from tensorflow.python.framework.constant_op import constant



NUM_CLIENTS = [5,34,338]
NUM_ROUNDS = 150
NUM_EPOCHS = 5


MODES = ['reduction_functions', 'femnist_distribution', 'uniform_vs_num_clients_weighting', 'accuracy_5_34_338_comparison', 'reduction_functions_comparison','updates_comparison']
modes = ["constant","exponential","linear","sigmoid","reciprocal"]
num_rounds = np.arange(1,NUM_ROUNDS+1)
num_clients = str(NUM_CLIENTS[0])


# def movingaverage(interval, window_size):
#   # window = np.ones(int(window_size))/float(window_size)
#   # return np.convolve(interval, window, 'same')
#   cumsum_vec = np.cumsum(np.insert(interval, 0, 0)) 
#   ma_vec = (cumsum_vec[window_size:] - cumsum_vec[:-window_size]) / window_size
#   return ma_vec

##############################################################################################################################################################
################################# Plot the graph of functions for different modes of reducing sampled clients ################################################
##############################################################################################################################################################

def reduction_functions():
  #**********************plot constant function***********************#
  x = np.arange(0,150,0.1)
  y_constant = [338]*len(x)
  plt.plot(x,y_constant,label="constant")

  #**********************plot exponential function***********************#
  y_exponential = [-np.exp((x_head-1)/10.3)+338 for x_head in x if x_head < 60] + [34]*(len(x)-600)
  plt.plot(x,y_exponential,label="exponential reduction")

  #**********************plot linear function***********************#
  y_linear = [-5.065*x_head+338 for x_head in x if x_head < 60] + [34]*(len(x)-600)
  plt.plot(x,y_linear,label="linear reduction")

  #**********************plot sigmoid function***********************#
  y_sigmoid = -304/(1+np.exp(-0.26*(x-20)))+338
  plt.plot(x,y_sigmoid,label="sigmoid reduction")

  #**********************plot reciprocal function***********************#
  y_reciprocal = 50/x+34
  plt.plot(x,y_reciprocal,label="reciprocal reduction")


  plt.xlim(0,150)
  plt.ylim(0,400)
  plt.xlabel("Rounds")
  plt.ylabel("Number of clients")
  plt.legend()
  # plt.title("Reduction functions")
  plt.show()


  return




##############################################################################################################################################################
############################################## Plot the graph of functions for FEMNIST distribution ##########################################################
##############################################################################################################################################################
def femnist_distribution():
  emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()
  client_data_counted_list=[len(emnist_train.create_tf_dataset_for_client(emnist_train.client_ids[x])) for x in range(len(emnist_train.client_ids))] #a list of the number of data in each client correspond to its index 

  ## A dictionary with unique elements from the sequence as keys and their frequencies (counts) as values, in this case key=data count of a client & value=number of clients who have the same data count
  counted_client_data = collections.Counter(client_data_counted_list)

  ## Sort counted_client_data by keys in ascending order and store them in a list of tuples, where each tuple has a (key,value)
  counted_client_data_sorted = sorted(counted_client_data.items())
  # print(counted_client_data_sorted)
  ## Unzip 
  data_count_per_client,num_clients=zip(*counted_client_data_sorted) #zip(*) is the inverse of zip(), unpack the list of tuples(pairs) into two tuples, namely (keys) and (values)
  #alternatively
  # data_count_per_client,num_clients= dict(counted_client_data_sorted).keys(),dict(counted_client_data_sorted).values()

  #-----------------Plot the bar plot of the distribution of clients data---------------------##
  plt.rcParams.update({'figure.figsize':(10,6), 'figure.dpi':100})
  fig, ax = plt.subplots()
  ax.bar(data_count_per_client,num_clients)
  ax.set_xlabel('Data amount per client(digits)')
  ax.set_ylabel('Frequency(Number of clients)')
  # ax.set_title('Data_Distribution_FEMNIST')
  plt.show()

  return




##############################################################################################################################################################
#################################################### Plot for different weightings strategies ################################################################
##############################################################################################################################################################
def uniform_vs_num_clients_weighting():
  with open(f"metrics/num_examples_vs_uniform/{34}_clients_{NUM_ROUNDS}_rounds_{NUM_EPOCHS}_epochs_accuracy_global.txt","rb") as fp: #unpickling
      global_accuracy = pickle.load(fp)
  # plot global accuracy & loss for all training rounds
  plt.plot(num_rounds, [x*100 for x in global_accuracy[:NUM_ROUNDS]], label=f"{NUM_CLIENTS[1]} clients, weighted by NUM_EXAMPLES")
  # num_rounds_av = movingaverage(num_rounds, 4)
  # plt.plot(num_rounds_av, global_accuracy[:147])

  with open(f"metrics/num_examples_vs_uniform/{34}_clients_uniform_weights_{NUM_ROUNDS}_rounds_{NUM_EPOCHS}_epochs_accuracy_global.txt","rb") as fp: #unpickling
    global_accuracy = pickle.load(fp)
  plt.plot(num_rounds, [x*100 for x in global_accuracy[:NUM_ROUNDS]], label=f"{NUM_CLIENTS[1]} clients, weighted by UNIFORM")

  plt.xlabel('Rounds',size=12)
  plt.ylabel('Test accuracy (%)',size=12)
  plt.legend()
  plt.show()



##############################################################################################################################################################
########################################### Plot for the training accuracy of randomly selected 5&34&338 clients #############################################
##############################################################################################################################################################
def accuracy_5_34_338_comparison():
  for n in range(len(NUM_CLIENTS)):
    with open(f"metrics/{NUM_CLIENTS[n]}_clients_{NUM_ROUNDS}_rounds_{NUM_EPOCHS}_epochs_accuracy_global.txt","rb") as fp: #unpickling
      global_accuracy = pickle.load(fp)
    plt.plot(num_rounds, [x*100 for x in global_accuracy[:NUM_ROUNDS]], label=f"{NUM_CLIENTS[n]} random clients")
  plt.xlabel('Rounds',size=12)
  plt.ylabel('Test accuracy (%)',size=12)
  plt.legend()
  plt.show()

#####################################################################################################################
################ Plot accuracy for various modes of varying num of randomly selected/sampled clients#################
#####################################################################################################################
def reduction_functions_comparison(mode):
  for mode_index, mode in enumerate(mode):
    if mode == "constant":
      continue
    else:
      with open(f"metrics/vary_num_clients_and_rounds/{NUM_CLIENTS[-1]} -> {NUM_CLIENTS[-1]} clients_constant_accuracy_global.txt","rb") as fp: #unpickling
        global_accuracy = pickle.load(fp)
      plt.plot(np.arange(len(global_accuracy)), [x*100 for x in global_accuracy[:NUM_ROUNDS]], label=f"{NUM_CLIENTS[-1]} clients, constant")
      with open(f"metrics/vary_num_clients_and_rounds/{NUM_CLIENTS[-1]} -> {34} clients_{mode}_accuracy_global.txt","rb") as fp: #unpickling
        global_accuracy = pickle.load(fp)
      plt.plot(np.arange(len(global_accuracy)), [x*100 for x in global_accuracy[:NUM_ROUNDS]], label=f"{NUM_CLIENTS[-1]} -> {34} clients, {mode} reduction")



      plt.xlabel('Rounds',size=12)
      plt.ylabel('Test accuracy (%)',size=12)
      plt.legend()
      plt.show()



###############################################################################################################################
#### Plot the bar graph of model update percentage & training time of different modes & rounds of reducing sampled clients ####
###############################################################################################################################
def updates_comparison():

  #*****************************************************************************************************************#
  #**********************plot bar chart of pushed_model_updates in different modes**********************************#
  #*****************************************************************************************************************#
  with open(f"metrics/vary_num_clients_and_rounds/pushed_model_updates.json","r") as f:
    pushed_model_updates = json.load(f)
  modes = [mode for mode, _ in pushed_model_updates.items()]
  updates = [update for _, update in pushed_model_updates.items()]
  with open(f"metrics/vary_num_clients_and_rounds/modes_stopped_round.json","r") as f:
    modes_stopped_round = json.load(f)
    modes_stopped_round = [value for key,value in modes_stopped_round.items()]
  data = {"modes": modes,
          "updates": updates}
  df = pd.DataFrame(data, columns=['modes', 'updates'])
  # plt.figure(figsize=(5, 5),dpi=300)
  plots = sns.barplot(x="modes", y="updates", data=df)


  # Iterrating over the bars one-by-one
  for bar in plots.patches:
    # Using Matplotlib's annotate function and
    # passing the coordinates where the annotation shall be done
    plots.annotate(format(bar.get_height(), '.2f'), #two decimal for pushed_model_updates_percentage
                    (bar.get_x() + bar.get_width() / 2,
                    bar.get_height()), ha='center', va='center',
                    size=12, xytext=(0, 5),
                    textcoords='offset points')
  # Setting the title for the graph
  # plt.title("Model updates comparison")
  # plt.ylabel("Total Updates",fontdict= { 'fontsize': 11, 'fontweight':'bold'})
  plt.ylabel("Total Updates (updates/round)",fontdict= { 'fontsize': 11, 'fontweight':'bold'})
  plt.xlabel("Reduction mode",fontdict= { 'fontsize': 11, 'fontweight':'bold'})
  # Fianlly showing the plot
  plt.show()



  #*****************************************************************************************************************#
  #**********************plot bar chart of averaged pushed_model_updates in different modes*************************#
  #*****************************************************************************************************************#
  # with open(f"metrics/vary_num_clients_and_rounds/pushed_model_updates.json","r") as f:
  #   pushed_model_updates = json.load(f)
  # modes = [mode for mode, _ in pushed_model_updates.items()]
  # updates = [update for _, update in pushed_model_updates.items()]
  # with open(f"metrics/vary_num_clients_and_rounds/modes_stopped_round.json","r") as f:
  #   modes_stopped_round = json.load(f)
  #   modes_stopped_round = [value for key,value in modes_stopped_round.items()]
  # average_updates = [update/stopped_round for update,stopped_round in zip(updates,modes_stopped_round)]
  # data = {"modes": modes,
  #         "average_updates": average_updates}
  # df = pd.DataFrame(data, columns=['modes', 'average_updates'])
  # # plt.figure(figsize=(5, 5),dpi=300)
  # plots = sns.barplot(x="modes", y="average_updates", data=df)


  # # Iterrating over the bars one-by-one
  # for bar in plots.patches:
  #   # Using Matplotlib's annotate function and
  #   # passing the coordinates where the annotation shall be done
  #   # plots.annotate(format(bar.get_height(), '.2f'), #two decimal for pushed_model_updates_percentage
  #   plots.annotate(format(int(bar.get_height())), #integer for pushed_model_updates_percentage
  #                   (bar.get_x() + bar.get_width() / 2,
  #                   bar.get_height()), ha='center', va='center',
  #                   size=12, xytext=(0, 5),
  #                   textcoords='offset points')
  # # Setting the title for the graph
  # # plt.title("Model updates comparison")
  # # plt.ylabel("Total Updates",fontdict= { 'fontsize': 11, 'fontweight':'bold'})
  # plt.ylabel("Average Updates (updates/round)",fontdict= { 'fontsize': 11, 'fontweight':'bold'})
  # plt.xlabel("Reduction mode",fontdict= { 'fontsize': 11, 'fontweight':'bold'})
  # # Fianlly showing the plot
  # plt.show()



  #*****************************************************************************************************************#
  #**********************plot bar chart of pushed_model_updates_percentage in different modes***********************#
  #*****************************************************************************************************************#
  # with open(f"metrics/vary_num_clients/pushed_model_updates_percentage.txt","rb") as fp: #unpickling
  #   pushed_model_updates_percentage = pickle.load(fp)
  # modes = [mode for mode, _ in pushed_model_updates_percentage.items()]
  # update_percentages = [update_percentage for _, update_percentage in pushed_model_updates_percentage.items()]

  # data = {"modes": modes,
  #         "update_percentages": update_percentages}
  # df = pd.DataFrame(data, columns=['modes', 'update_percentages'])
  # # plt.figure(figsize=(5, 5),dpi=300)
  # plots = sns.barplot(x="modes", y="update_percentages", data=df)

  # # Iterrating over the bars one-by-one
  # for bar in plots.patches:
  #   # Using Matplotlib's annotate function and
  #   # passing the coordinates where the annotation shall be done
  #   plots.annotate(format(bar.get_height(), '.2f'),
  #                   (bar.get_x() + bar.get_width() / 2,
  #                   bar.get_height()), ha='center', va='center',
  #                   size=12, xytext=(0, 5),
  #                   textcoords='offset points')
  # # Setting the title for the graph
  # plt.title("Model updates comparison")
  # plt.ylabel("Model updates(%)",fontdict= { 'fontsize': 11, 'fontweight':'bold'})
  # plt.xlabel("Reduction mode",fontdict= { 'fontsize': 11, 'fontweight':'bold'})
  # # Fianlly showing the plot
  # plt.show()

  # #****************************************************************************************************************#
  # #**********************plot bar chart of training time in different modes - varied clients***********************#
  # #****************************************************************************************************************#
  # f = open(f"metrics/vary_num_clients/modes_training_time.json", 'r')
  # modes_training_time = json.load(f)
  # f.close()

  # modes = [mode for mode, _ in modes_training_time.items()]
  # training_times = [training_time for _, training_time in modes_training_time.items()]

  # # plt.rcParams.update({'figure.figsize':(10,6), 'figure.dpi':300})
  # fig, ax = plt.subplots()
  # ax.bar(modes,training_times,color=['green', 'red', 'purple', 'blue', 'navy'])
  # ax.set_xlabel('Reduction mode',fontweight="bold")
  # ax.set_ylabel('Training time(s)',fontweight="bold")
  # ax.set_title('Training time comparison')

  # label = ["{:.2f}".format(t) for _,t in enumerate(training_times)]
  # for rect, label in zip(ax.patches, label):
  #     height = rect.get_height()
  #     ax.text(rect.get_x() + rect.get_width() / 2, height + 5, label,
  #             ha='center', va='bottom')
  # plt.show()

  #**************************************************************************************************************************#
  #**********************plot bar chart of training time in different modes -varied clients and rounds***********************#
  #**************************************************************************************************************************#
  f = open(f"metrics/vary_num_clients_and_rounds/modes_training_time.json", 'r')
  modes_training_time = json.load(f)
  f.close()

  modes = [mode for mode, _ in modes_training_time.items()]
  training_times = [training_time for _, training_time in modes_training_time.items()]

  # plt.rcParams.update({'figure.figsize':(10,6), 'figure.dpi':300})
  fig, ax = plt.subplots()
  ax.bar(modes,training_times,color=['green', 'red', 'purple', 'blue', 'navy'])
  ax.set_xlabel('Reduction mode',fontweight="bold")
  ax.set_ylabel('Training time(s)',fontweight="bold")
  # ax.set_title('Training time comparison')

  label = ["{:.2f}".format(t) for _,t in enumerate(training_times)]
  for rect, label in zip(ax.patches, label):
      height = rect.get_height()
      ax.text(rect.get_x() + rect.get_width() / 2, height + 5, label,
              ha='center', va='bottom')
  plt.show()








#******************************************************parsing the command line arguments********************************************************************#
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', nargs=1, type=str, help='Running mode. Must be one of the following modes: {}'.format(MODES))
    
    args = parser.parse_args()
    mode = args.mode[0]
    
    return args, mode


if __name__ == '__main__':
    args, mode = parse_args() # get argument from the command line
    
    # load the data
    print(f'plot mode: {mode}')
    if mode == 'reduction_functions':
        reduction_functions()
    elif mode == 'femnist_distribution':
        femnist_distribution()
    elif mode == 'uniform_vs_num_clients_weighting':
        uniform_vs_num_clients_weighting()
    elif mode == 'accuracy_5_34_338_comparison':
        accuracy_5_34_338_comparison()
    elif mode == 'reduction_functions_comparison':
        reduction_functions_comparison(modes)
    elif mode == 'updates_comparison':
        updates_comparison()
    else:
        raise Exception('Unrecognised mode: {}. Possible modes are: {}'.format(mode, MODES))
























#############################################################################################################################################################
############################################ Other useful ones but not included in the comand line arguments ################################################
#############################################################################################################################################################
# #-----------------Plot the line graphs of the global/server evaluation set accuracy evaluated on global model vs num_rounds for all num_clients --------------------#
# for n in range(len(NUM_CLIENTS)):
#   with open(f"metrics/{NUM_CLIENTS[n]}_clients_{NUM_ROUNDS}_rounds_{NUM_EPOCHS}_epochs.json", 'r') as f:
#       sample_clients = json.load(f)
#       sample_clientnum_examples_vs_uniforms = sample_clients.replace(" ",",")
#       sample_clients = ast.literal_eval(sample_clients)

#   with open(f"metrics/{NUM_CLIENTS[n]}_clients_{NUM_ROUNDS}_rounds_{NUM_EPOCHS}_epochs_accuracy_local.txt","rb") as fp: #unpickling
#       local_clients_accuracy = pickle.load(fp)

###############################################################################################################################################################
##################Plot the line graphs of the global/server evaluation set accuracy evaluated on global model vs num_rounds for varied num_clients ############
###############################################################################################################################################################
#----------------- --------------------#
# with open(f"metrics/{NUM_CLIENTS[1]} -> {NUM_CLIENTS[0]} clients_{-2}_steps_{NUM_ROUNDS}_rounds_{NUM_EPOCHS[0]}_epochs_accuracy_global.txt","rb") as fp: #unpickling
#     global_accuracy = pickle.load(fp)
# # plot global accuracy & loss for all training rounds for varied clients
# plt.plot(num_rounds, global_accuracy[:NUM_ROUNDS], label=f"{NUM_CLIENTS[1]} -> {NUM_CLIENTS[0]} clients, steps={-2}")
# with open(f"metrics/{NUM_CLIENTS[1]}_clients_{NUM_ROUNDS}_rounds_{NUM_EPOCHS[0]}_epochs_accuracy_global.txt","rb") as fp: #unpickling
#     global_accuracy = pickle.load(fp)
# # plot global accuracy & loss for all training rounds for fixed clients
# plt.plot(num_rounds, global_accuracy[:NUM_ROUNDS], label=f"{NUM_CLIENTS[1]} clients, steps={0}")

# plt.xlabel('Rounds',size=15)
# plt.ylabel('Global validation accuracy',size=15)
# plt.legend()
# plt.title(f'Global validation accuracy - {NUM_CLIENTS[1]} -> {NUM_CLIENTS[0]} & {NUM_CLIENTS[1]} clients, {NUM_ROUNDS} rounds, {num_epochs} epochs',size=15)
# plt.show()



#-----------------Plot the line graphs of the local/clients' evaluation set accuracy evaluated on global model vs num_rounds for all clients --------------------#
# for n in range(len(NUM_CLIENTS)):
#   plt.figure(figsize=(13, 8), dpi=100)
#   for c,client in enumerate(sample_clients):
#     plt.plot(num_rounds, local_clients_accuracy[c][:NUM_ROUNDS], label=f"client_{client}")
#   plt.legend(prop={'size':10})
#   plt.title(f'Local validation accuracy - {NUM_ROUNDS} rounds, {NUM_CLIENTS[n]} clients', size=25)
#   plt.xlabel('rounds',size=20)
#   plt.ylabel('accuracy',size=20)
#   plt.show()




#-----------------Plot the histogram of the num_clients vs local/clients' evaluation set accuracy evaluated on global model for that round --------------------#
# plt.figure(figsize=(13, 8), dpi=100)
# plt.rcParams.update({'figure.figsize':(13,8), 'figure.dpi':100})

# Plot Histogram on num_client vs accuracy
# print(np.shape(np.array(local_clients_accuracy)[:,99]))
# plt.hist(np.array(local_clients_accuracy)[:,NUM_ROUNDS-1], bins=np.arange(0,1,0.01))
# # plt.gca().set(title=f'Frequency Histogram-Evaluation Accuracy @ {NUM_ROUNDS} rounds_{NUM_CLIENTS[1]} clients', ylabel='Frequency(Number of clients)', xlabel='Accuracy')
# plt.title(f'Clients distribution over Local validation accuracy @ {NUM_ROUNDS} rounds, {NUM_EPOCHS} epochs, {NUM_CLIENTS[0]} clients', size=10)
# plt.xlabel('Local validation accuracy', size=20)
# plt.ylabel('Frequency(Number of clients)', size=20)
# plt.show()