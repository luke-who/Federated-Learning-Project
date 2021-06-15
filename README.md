# Optimising Federated Machine Learning Model Aggregation Teqniques for IOT 
-----------------------------------------------------------------------------------
[![python](https://img.shields.io/badge/python-3.8.8-blue?style=plastic&logo=python)](https://www.python.org/downloads/release/python-388/)
[![pip](https://img.shields.io/badge/pypi-v21.0.1-informational?&logo=pypi)](https://pypi.org/project/pip/21.0.1/)

[![tensorflow](https://img.shields.io/badge/tensorflow-2.4.1-orange?logo=TensorFlow)](https://pypi.org/project/tensorflow/2.4.1/)
[![tensorflow-federated](https://img.shields.io/badge/tensorflow--federated-0.18.0-yellowgreen?logo=TensorFlow)](https://pypi.org/project/tensorflow-federated/0.18.0/)

[![Ubuntu 20.04 LTS](https://img.shields.io/badge/ubuntu-20.04_LTS-orange?&color=E95420&logo=ubuntu)](https://releases.ubuntu.com/20.04/)
[![GCP](https://img.shields.io/badge/Google%20Cloud%20Platform(GCP)-_-4285F4?labelColor=white&logo=Google-Cloud)](https://cloud.google.com/)
<!-- [![GCP](https://img.shields.io/badge/Google%20Cloud%20Platform(GCP)-_-4285F4?logo=Google-Cloud)](https://cloud.google.com/) -->
<!-- [![GCP](https://img.shields.io/badge/Google%20Cloud%20Platform(GCP)-white?logo=Google-Cloud)](https://cloud.google.com/) -->

##Motivation 
This project attempts to explore the communication efficiency and scalibility of the existing federated machine learning techniques and possible improvements and futher optimisation 
##Aim
The aim of this project is to investigate, design and evaluate different methods to reduce overall data communication during federated learning scenarios, therefore further improve the existing federated learning system without sacrificing convergence rate

##Objectives
* Research the necessary library and development environment to conduct various FL simulations
* Create suitable unbalanced dataset, to simulate real world FL system and evaluate corresponding methods
* Build FL model with existing machine learning framework using basic model aggregation such as averaged weights update ($Fed\Avg$)
* Investigate the effect of parameters: number of clients, rounds, epochs, learning rate, optimisation functions on the global model 
* Benchmark the FL algorithm, use metrics such as learning accuracy and loss to evaluate model performance and convergence rate by deploying different communication reduction strategies
* Choose the best method out of all proposed reduction strategies for optimising communication, calculate the amount of reduction achieved

# Getting Started
-----------------------------------------------------------------------------------
<!-- TODO: Guide users through getting your code up and running on their own system. In this section you can talk about: -->
##1.    Installation process on Ubuntu

For setting up GCP, since the algorithm is not memory optimised, the memory usage was very intensive while running different reduction functions in `tff_vary_num_clients_and_rounds.py`.  This is likely due to the fact that TFF is not currently optimised for selecting a varying number of clients as it seems to mess up 
with the state during the iterative process and taking up huge accumulative memory.  As a result, the RAM in VM required on GCP for running tff_vary_num_clients_and_rounds.py was 128GB, at its peak it's using around 50% of the total memory so it's sth to keep in mind. 
###Install TensorFlow with pip
1. Install the Python development environment on your system

    `sudo apt update ; sudo apt upgrade`

    `sudo apt install python3-dev python3-pip python3-venv`
2. Check python3 and pip3 version

    `python3 --version`

    `pip3 --version`
3. Create a virtual environment (recommended)

    `python3 -m venv --system-site-packages ./venv`

    activate it `source ~/venv/bin/activate`
4. Go inside created virtual environment

    `cd venv`

    upgrade pip

    `(venv) $ pip install --upgrade pip`

    list packages installed within the virtual environment

    `(venv) $ pip list`
5. Install the TensorFlow pip package 

    `(venv) $ pip install testresources`

    `(venv) $ pip install tensorflow==2.4.1`

6. Verify the install:

    `(venv) $ python -c "import tensorflow as tf;print(tf.reduce_sum(tf.random.normal([1000, 1000])))"`

###Install the released TensorFlow Federated Python package
1. Install Tensorflow Federated 

    `(venv) $ pip install tensorflow-federated==0.18.0`

2. Test Tensorflow Federated 

    `(venv) $ python -c "import tensorflow_federated as tff; print(tff.federated_computation(lambda: 'Hello World')())"`

3. Exit virtualenv until you're done using tensorflow/tensorflow-federated

    `(venv) $ deactivate`

##2.    Software dependencies
###Python version
The python version in this project throughout was [3.8.8](https://www.python.org/downloads/release/python-388/), [pyenv](https://github.com/pyenv/pyenv) was used to manage different python versions
###pypi packages
All the dependacies, versions and necessary packages are exported & listed in [requirements.txt](requirements.txt)(albeit not all of them are useful to run on local machines). 
###To install the requirements, do `pip3 install -r requirements.txt`
###Dealing with OSError: [Errno 24] Too many open files
First `sudo nano /etc/pam.d/common-session`

Then add `session required pam_limits.so` to `/etc/pam.d/common-session`

Then `sudo nano /etc/security/limits.conf`

add

`*          hard    nofile      500000`

`*          soft    nofile      500000`

Then set 

`ulimit -n 500000`

##3.	Latest releases
##4.	API references

# Build and Test
-----------------------------------------------------------------------------------
<!-- TODO: Describe and show how to build your code and run the tests.  -->
## Running tff_vary_num_clients_and_rounds.py:

`python3 tff_vary_num_clients_and_rounds.py MODE` to run the script with different mode arguments.
<!-- `python3 tff_vary_num_clients_and_rounds.py mode &` to run it in the background -->


The mode you can select are: MODE = `[constant,exponential,linear,sigmoid,reciprocal]`


## Running tff_UNIFORM_vs_NUM_EXAMPLES.py & tff_train_test_split.py (in 'other' folder):

`python3 tff_UNIFORM_vs_NUM_EXAMPLES.py` and `python3 tff_train_test_split.py` respectively to run these two scripts, no arguments/mode needed.
<!-- `python3 tff_vary_num_clients_and_rounds.py mode &` to run it in the background -->


## Running plot.py:

`python3 plot.py mode` to run the script with different mode arguments.
<!-- `python3 tff_vary_num_clients_and_rounds.py mode &` to run it in the background -->


The mode you can select are: mode = `[reduction_functions, femnist_distribution, uniform_vs_num_clients_weighting, accuracy_5_34_338_comparison, reduction_functions_comparison,updates_comparison]`
# Contribute
-----------------------------------------------------------------------------------
<!-- TODO: Explain how other users and developers can contribute to make your code better.  -->

<!-- If you want to learn more about creating good readme files then refer the following [guidelines](https://docs.microsoft.com/en-us/azure/devops/repos/git/create-a-readme?view=azure-devops). You can also seek inspiration from the below readme files:
- [ASP.NET Core](https://github.com/aspnet/Home)
- [Visual Studio Code](https://github.com/Microsoft/vscode)
- [Chakra Core](https://github.com/Microsoft/ChakraCore) -->

# LICENSE
-----------------------------------------------------------------------------------
[MIT License](LICENSE)