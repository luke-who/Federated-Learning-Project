# Copyright 2020, Google LLC.
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
"""End-to-end tests for federated trainer tasks."""

import os.path

from absl.testing import parameterized
import tensorflow as tf
import tensorflow_federated as tff

from optimization.cifar100 import federated_cifar100
from optimization.emnist import federated_emnist
from optimization.emnist_ae import federated_emnist_ae
from optimization.shakespeare import federated_shakespeare
from optimization.shared import training_specs
from optimization.stackoverflow import federated_stackoverflow
from optimization.stackoverflow_lr import federated_stackoverflow_lr
from utils import training_loop


def iterative_process_builder(model_fn):
  return tff.learning.build_federated_averaging_process(
      model_fn=model_fn,
      client_optimizer_fn=tf.keras.optimizers.SGD,
      server_optimizer_fn=tf.keras.optimizers.SGD)


class FederatedTasksTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('cifar100', 'cifar100', federated_cifar100.configure_training),
      ('emnist_cr', 'emnist_cr', federated_emnist.configure_training),
      ('emnist_ae', 'emnist_ae', federated_emnist_ae.configure_training),
      ('shakespeare', 'shakespeare', federated_shakespeare.configure_training),
      ('stackoverflow_nwp', 'stackoverflow_nwp',
       federated_stackoverflow.configure_training),
      ('stackoverflow_lr', 'stackoverflow_lr',
       federated_stackoverflow_lr.configure_training),
  )
  def test_run_federated(self, task_name, config_fn):
    task_spec = training_specs.TaskSpec(
        iterative_process_builder=iterative_process_builder,
        client_epochs_per_round=1,
        client_batch_size=10,
        clients_per_round=1,
        client_datasets_random_seed=1)
    runner_spec = config_fn(task_spec)

    total_rounds = 1
    root_output_dir = self.get_temp_dir()
    exp_name = 'test_run_federated'

    training_loop.run(
        iterative_process=runner_spec.iterative_process,
        client_datasets_fn=runner_spec.client_datasets_fn,
        validation_fn=runner_spec.validation_fn,
        # For efficiency, we avoid using the entire test set here
        test_fn=None,
        total_rounds=total_rounds,
        root_output_dir=root_output_dir,
        experiment_name=exp_name)

    results_dir = os.path.join(root_output_dir, 'results', exp_name)
    self.assertTrue(tf.io.gfile.exists(results_dir))

    scalar_manager = tff.simulation.CSVMetricsManager(
        os.path.join(results_dir, 'experiment.metrics.csv'))
    fieldnames, metrics = scalar_manager.get_metrics()

    self.assertIn(
        'train/train/loss',
        fieldnames,
        msg='The output metrics should have a `train/loss` column if training '
        'is successful.')
    self.assertIn(
        'eval/loss',
        fieldnames,
        msg='The output metrics should have a `train/loss` column if validation'
        ' metrics computation is successful.')
    self.assertLen(
        metrics,
        total_rounds + 1,
        msg='The number of rows in the metrics CSV should be the number of '
        'training rounds + 1 (as there is an extra row for validation set'
        'metrics after training has completed.')


if __name__ == '__main__':
  tf.test.main()
