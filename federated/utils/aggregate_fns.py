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
"""Library of `tff.templates.MeasuredProcess` aggregation implementations.

Intended for usage with the `tff.learning` API for composing different
communication primitives in federated learning.
"""

from typing import Union

import attr
import tensorflow as tf
import tensorflow_federated as tff


@attr.s(auto_attribs=True, eq=False, frozen=True)
class NormClippedAggregationMetrics(object):
  """Structure metrics returned by a norm clipped averaging prcoess.

  Fields:
  -   `max_norm`: A float. The maximum client global norm observed in a round.
  -   `num_clipped_updates`: An integer. The number of updates that were clipped
        this round.
  """
  max_global_norm: float
  num_clipped: int


def build_fixed_clip_norm_mean_process(
    *,
    clip_norm: float,
    model_update_type: Union[tff.StructType, tff.TensorType],
) -> tff.templates.MeasuredProcess:
  """Returns process that clips the client deltas before averaging.

  The returned `MeasuredProcess` has a next function with the TFF type
  signature:

  ```
  (<()@SERVER, {model_update_type}@CLIENTS> ->
   <state=()@SERVER,
    result=model_update_type@SERVER,
    measurements=NormClippedAggregationMetrics@SERVER>)
  ```

  Args:
    clip_norm: the clip norm to apply to the global norm of the model update.
      See https://www.tensorflow.org/api_docs/python/tf/clip_by_global_norm for
        details.
    model_update_type: a `tff.Type` describing the shape and type of the value
      that will be clipped and averaged.

  Returns:
    A `tff.templates.MeasuredProcess` with the type signature detailed above.
  """

  @tff.federated_computation
  def initialize_fn():
    return tff.federated_value((), tff.SERVER)

  @tff.federated_computation(
      tff.type_at_server(()), tff.type_at_clients(model_update_type),
      tff.type_at_clients(tf.float32))
  def next_fn(state, deltas, weights):

    @tff.tf_computation(model_update_type)
    def clip_by_global_norm(update):
      clipped_update, global_norm = tf.clip_by_global_norm(
          tf.nest.flatten(update), tf.constant(clip_norm))
      was_clipped = tf.cond(
          tf.greater(global_norm, tf.constant(clip_norm)),
          lambda: tf.constant(1),
          lambda: tf.constant(0),
      )
      clipped_update = tf.nest.pack_sequence_as(update, clipped_update)
      return clipped_update, global_norm, was_clipped

    clipped_deltas, client_norms, client_was_clipped = tff.federated_map(
        clip_by_global_norm, deltas)

    return tff.templates.MeasuredProcessOutput(
        state=state,
        result=tff.federated_mean(clipped_deltas, weight=weights),
        measurements=tff.federated_zip(
            NormClippedAggregationMetrics(
                max_global_norm=tff.aggregators.federated_max(client_norms),
                num_clipped=tff.federated_sum(client_was_clipped),
            )))

  return tff.templates.MeasuredProcess(
      initialize_fn=initialize_fn, next_fn=next_fn)
