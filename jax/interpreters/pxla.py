# Copyright 2018 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Note: import <name> as <name> is required for names to be exported.
# See PEP 484 & https://github.com/jax-ml/jax/issues/7570

from jax._src.interpreters import pxla as _src_pxla
from jax._src import mesh as _src_mesh
from jax._src import op_shardings as _src_op_shardings
from jax._src import sharding_impls as _src_sharding_impls
from jax._src import sharding_specs as _src_sharding_specs

_deprecations = {
    # Added Dec 2025
    "Index": (
        "jax.interpreters.pxla.Index is deprecated.",
        _src_pxla.Index,
    ),
    "MapTracer": (
        "jax.interpreters.pxla.MapTracer is deprecated.",
        _src_pxla.MapTracer,
    ),
    "MeshAxisName": (
        "jax.interpreters.pxla.MeshAxisName is deprecated. Use jax.sharding.Mesh axis names directly.",
        _src_pxla.MeshAxisName,
    ),
    "MeshComputation": (
        "jax.interpreters.pxla.MeshComputation is deprecated.",
        _src_pxla.MeshComputation,
    ),
    "MeshExecutable": (
        "jax.interpreters.pxla.MeshExecutable is deprecated.",
        _src_pxla.MeshExecutable,
    ),
    "PmapExecutable": (
        "jax.interpreters.pxla.PmapExecutable is deprecated.",
        _src_pxla.PmapExecutable,
    ),
    "global_aval_to_result_handler": (
        "jax.interpreters.pxla.global_aval_to_result_handler is deprecated.",
        _src_pxla.global_aval_to_result_handler,
    ),
    "global_avals_to_results_handler": (
        "jax.interpreters.pxla.global_avals_to_results_handler is deprecated.",
        _src_pxla.global_avals_to_results_handler,
    ),
    "global_result_handlers": (
        "jax.interpreters.pxla.global_result_handlers is deprecated.",
        _src_pxla.global_result_handlers,
    ),
    "parallel_callable": (
        "jax.interpreters.pxla.parallel_callable is deprecated.",
        _src_pxla.parallel_callable,
    ),
    "shard_args": (
        "jax.interpreters.pxla.shard_args is deprecated.",
        _src_pxla.shard_args,
    ),
    "xla_pmap_p": (
        "jax.interpreters.pxla.xla_pmap_p is deprecated.",
        _src_pxla.xla_pmap_p,
    ),
    "thread_resources": (
        "jax.interpreters.pxla.thread_resources is deprecated.",
        _src_mesh.thread_resources,
    ),
    "are_hlo_shardings_equal": (
        "jax.interpreters.pxla.are_hlo_shardings_equal is deprecated.",
        _src_op_shardings.are_hlo_shardings_equal,
    ),
    "is_hlo_sharding_replicated": (
        "jax.interpreters.pxla.is_hlo_sharding_replicated is deprecated.",
        _src_op_shardings.is_hlo_sharding_replicated,
    ),
    "op_sharding_to_indices": (
        "jax.interpreters.pxla.op_sharding_to_indices is deprecated.",
        _src_op_shardings.op_sharding_to_indices,
    ),
    "ArrayMapping": (
        "jax.interpreters.pxla.ArrayMapping is deprecated.",
        _src_sharding_impls.ArrayMapping,
    ),
    "UNSPECIFIED": (
        "jax.interpreters.pxla.UNSPECIFIED is deprecated.",
        _src_sharding_impls.UNSPECIFIED,
    ),
    "array_mapping_to_axis_resources": (
        "jax.interpreters.pxla.array_mapping_to_axis_resources is deprecated.",
        _src_sharding_impls.array_mapping_to_axis_resources,
    ),
    "Chunked": (
        "jax.interpreters.pxla.Chunked is deprecated.",
        _src_sharding_specs.Chunked,
    ),
    "NoSharding": (
        "jax.interpreters.pxla.NoSharding is deprecated.",
        _src_sharding_specs.NoSharding,
    ),
    "Replicated": (
        "jax.interpreters.pxla.Replicated is deprecated.",
        _src_sharding_specs.Replicated,
    ),
    "ShardedAxis": (
        "jax.interpreters.pxla.ShardedAxis is deprecated.",
        _src_sharding_specs.ShardedAxis,
    ),
    "ShardingSpec": (
        "jax.interpreters.pxla.ShardingSpec is deprecated.",
        _src_sharding_specs.ShardingSpec,
    ),
    "Unstacked": (
        "jax.interpreters.pxla.Unstacked is deprecated.",
        _src_sharding_specs.Unstacked,
    ),
    "spec_to_indices": (
        "jax.interpreters.pxla.spec_to_indices is deprecated.",
        _src_sharding_specs.spec_to_indices,
    ),
}

import typing as _typing
if _typing.TYPE_CHECKING:
  Index = _src_pxla.Index
  MapTracer = _src_pxla.MapTracer
  MeshAxisName = _src_pxla.MeshAxisName
  MeshComputation = _src_pxla.MeshComputation
  MeshExecutable = _src_pxla.MeshExecutable
  PmapExecutable = _src_pxla.PmapExecutable
  global_aval_to_result_handler = _src_pxla.global_aval_to_result_handler
  global_avals_to_results_handler = _src_pxla.global_avals_to_results_handler
  global_result_handlers = _src_pxla.global_result_handlers
  parallel_callable = _src_pxla.parallel_callable
  shard_args = _src_pxla.shard_args
  xla_pmap_p = _src_pxla.xla_pmap_p
  thread_resources = _src_mesh.thread_resources
  are_hlo_shardings_equal = _src_op_shardings.are_hlo_shardings_equal
  is_hlo_sharding_replicated = _src_op_shardings.is_hlo_sharding_replicated
  op_sharding_to_indices = _src_op_shardings.op_sharding_to_indices
  ArrayMapping = _src_sharding_impls.ArrayMapping
  UNSPECIFIED = _src_sharding_impls.UNSPECIFIED
  array_mapping_to_axis_resources = _src_sharding_impls.array_mapping_to_axis_resources
  Chunked = _src_sharding_specs.Chunked
  NoSharding = _src_sharding_specs.NoSharding
  Replicated = _src_sharding_specs.Replicated
  ShardedAxis = _src_sharding_specs.ShardedAxis
  ShardingSpec = _src_sharding_specs.ShardingSpec
  Unstacked = _src_sharding_specs.Unstacked
  spec_to_indices = _src_sharding_specs.spec_to_indices
else:
  from jax._src.deprecations import deprecation_getattr as _deprecation_getattr
  __getattr__ = _deprecation_getattr(__name__, _deprecations)
  del _deprecation_getattr
del _typing
