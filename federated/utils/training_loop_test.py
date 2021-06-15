# Copyright 2019, Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for shared training loops."""

import collections
import os

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

from utils import training_loop

_Batch = collections.namedtuple('Batch', ['x', 'y'])


def _build_federated_averaging_process():
  return tff.learning.build_federated_averaging_process(
      _uncompiled_model_fn,
      client_optimizer_fn=tf.keras.optimizers.SGD,
      server_optimizer_fn=tf.keras.optimizers.SGD)


def _uncompiled_model_fn():
  keras_model = tff.simulation.models.mnist.create_keras_model(
      compile_model=False)
  input_spec = _create_input_spec()
  return tff.learning.from_keras_model(
      keras_model=keras_model,
      input_spec=input_spec,
      loss=tf.keras.losses.SparseCategoricalCrossentropy())


def _batch_fn():
  batch = _Batch(
      x=np.ones([1, 784], dtype=np.float32), y=np.ones([1, 1], dtype=np.int64))
  return batch


def _create_input_spec():
  return _Batch(
      x=tf.TensorSpec(shape=[None, 784], dtype=tf.float32),
      y=tf.TensorSpec(dtype=tf.int64, shape=[None, 1]))


class TrainingLoopArgumentsTest(tf.test.TestCase):

  def test_raises_non_iterative_process(self):
    bad_iterative_process = _build_federated_averaging_process().next
    federated_data = [[_batch_fn()]]

    def client_datasets_fn(round_num):
      del round_num
      return federated_data

    def validation_fn(state, round_num):
      del state, round_num
      return {}

    root_output_dir = self.get_temp_dir()
    with self.assertRaises(TypeError):
      training_loop.run(
          iterative_process=[bad_iterative_process],
          client_datasets_fn=client_datasets_fn,
          validation_fn=validation_fn,
          total_rounds=10,
          experiment_name='non_iterative_process',
          root_output_dir=root_output_dir)

  def test_raises_non_callable_client_dataset(self):
    iterative_process = _build_federated_averaging_process()
    client_dataset = [[_batch_fn()]]

    def validation_fn(state, round_num):
      del state, round_num
      return {}

    root_output_dir = self.get_temp_dir()
    with self.assertRaises(TypeError):
      training_loop.run(
          iterative_process=iterative_process,
          client_datasets_fn=client_dataset,
          validation_fn=validation_fn,
          total_rounds=10,
          experiment_name='non_callable_client_dataset',
          root_output_dir=root_output_dir)

  def test_raises_non_callable_evaluate_fn(self):
    iterative_process = _build_federated_averaging_process()
    federated_data = [[_batch_fn()]]

    def client_datasets_fn(round_num):
      del round_num
      return federated_data

    metrics_dict = {}
    root_output_dir = self.get_temp_dir()
    with self.assertRaises(TypeError):
      training_loop.run(
          iterative_process=iterative_process,
          client_datasets_fn=client_datasets_fn,
          validation_fn=metrics_dict,
          total_rounds=10,
          experiment_name='non_callable_evaluate',
          root_output_dir=root_output_dir)

  def test_raises_non_str_output_dir(self):
    iterative_process = _build_federated_averaging_process()
    federated_data = [[_batch_fn()]]

    def client_datasets_fn(round_num):
      del round_num
      return federated_data

    def validation_fn(state, round_num):
      del state, round_num
      return {}

    with self.assertRaises(TypeError):
      training_loop.run(
          iterative_process=iterative_process,
          client_datasets_fn=client_datasets_fn,
          validation_fn=validation_fn,
          total_rounds=10,
          experiment_name='non_str_output_dir',
          root_output_dir=1)


class ExperimentRunnerTest(tf.test.TestCase):

  def test_fedavg_training_decreases_loss(self):
    batch = _batch_fn()
    federated_data = [[batch]]
    iterative_process = _build_federated_averaging_process()

    def client_datasets_fn(round_num):
      del round_num
      return federated_data

    def validation_fn(state, round_num):
      model = state.model
      del round_num
      keras_model = tff.simulation.models.mnist.create_keras_model(
          compile_model=True)
      model.assign_weights_to(keras_model)
      return {'loss': keras_model.evaluate(batch.x, batch.y)}

    initial_state = iterative_process.initialize()

    root_output_dir = self.get_temp_dir()
    final_state = training_loop.run(
        iterative_process=iterative_process,
        client_datasets_fn=client_datasets_fn,
        validation_fn=validation_fn,
        total_rounds=1,
        experiment_name='fedavg_decreases_loss',
        root_output_dir=root_output_dir)
    self.assertLess(
        validation_fn(final_state, 0)['loss'],
        validation_fn(initial_state, 0)['loss'])

  def test_checkpoint_manager_saves_state(self):
    experiment_name = 'checkpoint_manager_saves_state'
    iterative_process = _build_federated_averaging_process()
    federated_data = [[_batch_fn()]]

    def client_datasets_fn(round_num):
      del round_num
      return federated_data

    def validation_fn(state, round_num):
      del state, round_num
      return {}

    root_output_dir = self.get_temp_dir()
    final_state = training_loop.run(
        iterative_process=iterative_process,
        client_datasets_fn=client_datasets_fn,
        validation_fn=validation_fn,
        total_rounds=1,
        experiment_name=experiment_name,
        root_output_dir=root_output_dir)
    final_model = final_state.model

    ckpt_manager = tff.simulation.FileCheckpointManager(
        os.path.join(root_output_dir, 'checkpoints', experiment_name))
    restored_state, restored_round = ckpt_manager.load_latest_checkpoint(
        final_state)

    self.assertEqual(restored_round, 0)

    keras_model = tff.simulation.models.mnist.create_keras_model(
        compile_model=True)
    restored_model = restored_state.model

    restored_model.assign_weights_to(keras_model)
    restored_loss = keras_model.test_on_batch(federated_data[0][0].x,
                                              federated_data[0][0].y)

    final_model.assign_weights_to(keras_model)
    final_loss = keras_model.test_on_batch(federated_data[0][0].x,
                                           federated_data[0][0].y)
    self.assertEqual(final_loss, restored_loss)

  def test_fn_writes_metrics(self):
    experiment_name = 'test_metrics'
    iterative_process = _build_federated_averaging_process()
    batch = _batch_fn()
    federated_data = [[batch]]

    def client_datasets_fn(round_num):
      del round_num
      return federated_data

    def test_fn(state):
      model = state.model
      keras_model = tff.simulation.models.mnist.create_keras_model(
          compile_model=True)
      model.assign_weights_to(keras_model)
      return {'loss': keras_model.evaluate(batch.x, batch.y)}

    def validation_fn(state, round_num):
      del state, round_num
      return {}

    root_output_dir = self.get_temp_dir()
    training_loop.run(
        iterative_process=iterative_process,
        client_datasets_fn=client_datasets_fn,
        validation_fn=validation_fn,
        total_rounds=1,
        experiment_name=experiment_name,
        root_output_dir=root_output_dir,
        rounds_per_eval=10,
        test_fn=test_fn)

    csv_file = os.path.join(root_output_dir, 'results', experiment_name,
                            'experiment.metrics.csv')
    metrics_manager = tff.simulation.CSVMetricsManager(csv_file)
    fieldnames, metrics = metrics_manager.get_metrics()
    self.assertLen(metrics, 2)
    self.assertIn('test/loss', fieldnames)

  def test_training_loop_works_on_restart(self):
    experiment_name = 'test_metrics'
    iterative_process = _build_federated_averaging_process()
    batch = _batch_fn()
    federated_data = [[batch]]

    def client_datasets_fn(round_num):
      del round_num
      return federated_data

    def validation_fn(state, round_num):
      del state, round_num
      return {}

    root_output_dir = self.get_temp_dir()
    training_loop.run(
        iterative_process=iterative_process,
        client_datasets_fn=client_datasets_fn,
        validation_fn=validation_fn,
        total_rounds=2,
        experiment_name=experiment_name,
        root_output_dir=root_output_dir)

    training_loop.run(
        iterative_process=iterative_process,
        client_datasets_fn=client_datasets_fn,
        validation_fn=validation_fn,
        total_rounds=4,
        experiment_name=experiment_name,
        root_output_dir=root_output_dir)

    csv_file = os.path.join(root_output_dir, 'results', experiment_name,
                            'experiment.metrics.csv')
    metrics_manager = tff.simulation.CSVMetricsManager(csv_file)
    _, metrics = metrics_manager.get_metrics()
    # Test that 4 rounds of metrics are logged (the CSV length adds one for
    # the header).
    self.assertLen(metrics, 5)


if __name__ == '__main__':
  tf.test.main()
