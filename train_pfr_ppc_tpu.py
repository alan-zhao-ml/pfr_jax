"""Trainer for the custom JAX DNN model in TFlex."""

import itertools
import os
import time
from typing import Dict, List

from absl import app
from absl import flags
from absl import logging
import flax.linen as nn
from flax.training import train_state
import jax
import jax.experimental.multihost_utils as jax_multihost_utils
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import tensorflow as tf

from google3.learning.brain.google.data.python.ops import array_record_dataset

FLAGS = flags.FLAGS

_TRAIN_FILES = flags.DEFINE_string(
    'train_files', '', 'File pattern for training data.'
)
_VAL_FILES = flags.DEFINE_string(
    'val_files', '', 'File pattern for validation data.'
)
_WORKING_DIR = flags.DEFINE_string(
    'working_dir', '', 'Directory for checkpoints and logs.'
)
_EXPORT_DIR = flags.DEFINE_string(
    'export_dir', '', 'Directory to export the model.'
)
_PER_DEVICE_BATCH_SIZE = flags.DEFINE_integer(
    'per_device_batch_size', 8192, 'Batch size per TPU core.'
)
_NUM_TRAIN_EXAMPLES = flags.DEFINE_integer(
    'num_train_examples',
    518133001,
    'Number of training examples in the dataset.',
)
_REPLICATE_ALL_EMBEDDINGS = flags.DEFINE_bool(
    'replicate_all_embeddings',
    True,
    'If True, replicates all embedding tables.',
)
_XPROF_PORT = flags.DEFINE_integer('xprof_port', None, 'port for xprof')
_EVAL_STEPS = flags.DEFINE_integer(
    'eval_steps',
    200,
    'Number of batches to evaluate during training metrics syncs.',
)
_MODEL_TYPE = flags.DEFINE_string(
    'model_type',
    'baseline',
    'Model type to train. One of: baseline, pfr.',
)
_ACTIVATION_FN = flags.DEFINE_string(
    'activation_fn',
    'relu',
    'Activation function for standard models. One of: relu, tanh.',
)
_MAX_SEQ_LEN = flags.DEFINE_integer(
    'max_seq_len', 4, 'Max sequence length for multivalent features.'
)
_NUM_EPOCHS = flags.DEFINE_integer(
    'num_epochs', 15, 'Number of epochs to train.'
)
_L2_REG_WEIGHT = flags.DEFINE_float(
    'l2_reg_weight',
    0.001,
    'L2 regularization weight for embeddings in baseline model.',
)
_L2_REG_STYLE = flags.DEFINE_string(
    'l2_reg_style',
    'sgd_l2_wd',
    'L2 regularization style for baseline. One of: sgd_l2_wd, standard.',
)
_DENSE_LR = flags.DEFINE_float(
    'dense_lr', 0.001, 'Learning rate for dense layers.'
)
_EMBED_LR = flags.DEFINE_float(
    'embed_lr', 0.02, 'Learning rate for embeddings.'
)
_PFR_SHARE_TAU = flags.DEFINE_bool(
    'pfr_share_tau',
    False,
    'If True, shares a single log_variance parameter for all features in PFR.',
)
_PFR_INITIAL_TAU = flags.DEFINE_float(
    'pfr_initial_tau',
    1.0,
    'Initial value for tau (standard deviation of the prior) in PFR.',
)
_INIT_PHASE_PCT = flags.DEFINE_float(
    'init_phase_pct',
    0.0,
    'Percentage (0 to 100) of training steps to compute only Data loss and'
    ' Correction 1 while freezing tau.',
)
_USE_WN_ALL_LAYER = flags.DEFINE_bool(
    'use_wn_all_layer',
    True,
    'If True, uses weight normalization (without scale g) on all layers of'
    ' the DNN tower.',
)
_EMB_OPT_USING_C2 = flags.DEFINE_bool(
    'emb_opt_using_c2',
    False,
    'Whether to allow Correction 2 to optimize the embeddings.',
)
_MLP_OPT_USING_C2 = flags.DEFINE_bool(
    'mlp_opt_using_c2',
    False,
    'Whether to allow Correction 2 to optimize the MLP weights. Only effective'
    ' if emb_opt_using_c2 is True.',
)
_INIT_EMB_WITH_TAU = flags.DEFINE_bool(
    'init_emb_with_tau',
    False,
    'Whether to initialize embeddings using initial_tau as stddev.',
)


# Feature Embeddings Configuration
# (vocab_size, embedding_dim)
EMBEDDING_CONFIGS = {
    'campaign_id': (328_508, 64),
    'ad_group_id': (541_516, 64),
    'youtube_ad_external_video_id': (963_056, 64),
    'youtube_ad_video_customer_id': (139_840, 64),
    'youtube_ad_video_technics_taxonomic_entities': (56, 8),
    'ono_call_to_action_text': (1_044, 16),
    'youtube_ad_video_visible_url_domain': (91_948, 32),
    'top_engaged_yt_channel_surrogate_ids_2m_vsfeh': (1_053_976, 64),
    'viral_num_fcap_id_repeats_shorts_7d': (76, 4),
    'viral_num_fcap_id_repeats_shorts_30d': (336, 4),
    'yt_shorts_user_engaged_views_bucketized_logratio_week': (80, 4),
    'yt_shorts_user_engaged_views_bucketized_logratio_day': (73, 4),
    'youtube_software_interface': (8, 4),
    'ui_feature_type': (88, 4),
    'reel_watch_endpoint_source': (88, 4),
    'viral_device_height': (188, 4),
    'content_user_connection_rtt_percentile': (48, 4),
}

MULTIVALENT_FEATURES = [
    'youtube_ad_video_technics_taxonomic_entities',
    'top_engaged_yt_channel_surrogate_ids_2m_vsfeh',
    'ui_feature_type',
]

# Hashing strategies (True for Hash, False for Direct)
HASH_STRATEGIES = {
    'campaign_id': True,
    'ad_group_id': True,
    'youtube_ad_external_video_id': True,
    'youtube_ad_video_customer_id': True,
    'youtube_ad_video_technics_taxonomic_entities': True,
    'ono_call_to_action_text': True,
    'youtube_ad_video_visible_url_domain': True,
    'top_engaged_yt_channel_surrogate_ids_2m_vsfeh': True,
    'viral_num_fcap_id_repeats_shorts_7d': False,
    'viral_num_fcap_id_repeats_shorts_30d': False,
    'yt_shorts_user_engaged_views_bucketized_logratio_week': False,
    'yt_shorts_user_engaged_views_bucketized_logratio_day': False,
    'youtube_software_interface': False,
    'ui_feature_type': True,
    'reel_watch_endpoint_source': False,
    'viral_device_height': True,
    'content_user_connection_rtt_percentile': True,
}


@jax.custom_gradient
def stable_pfr_loss_for_feature(
    log_var: jnp.ndarray,
    level_norm_sq_gathered: jnp.ndarray,
    tilde_f_gathered: jnp.ndarray,
    k_m: int,
    d_m: int,
    alpha: jnp.ndarray | float,
    active_row_mask: jnp.ndarray,
):
  """Computes PFR correction loss with stable custom log-variance gradients."""
  clipped_log_var = jnp.clip(log_var[0], -20.0, 20.0)
  prior_var = jnp.exp(clipped_log_var)
  active_row_mask = active_row_mask.astype(jnp.float32)

  weighted_norm_sq = level_norm_sq_gathered * active_row_mask
  correction1 = (
      jnp.sum((alpha / (2.0 * prior_var)) * weighted_norm_sq)
      + (k_m * d_m / 2.0) * clipped_log_var
  )

  log_det_gathered = jnp.sum(
      jnp.log(tilde_f_gathered + 1.0 / prior_var + 1e-8), axis=-1
  )
  weighted_log_det = log_det_gathered * active_row_mask
  correction2 = (alpha / 2.0) * jnp.sum(weighted_log_det)

  loss_val = correction1 + correction2

  def grad_fn(g):
    edf_gathered = jnp.sum(
        tilde_f_gathered / (tilde_f_gathered + 1.0 / prior_var + 1e-8),
        axis=-1,
    )
    weighted_edf = edf_gathered * active_row_mask

    stable_grad = (
        0.5
        * alpha
        * jnp.sum(weighted_edf - (1.0 / prior_var) * weighted_norm_sq)
    )
    sum_alpha = alpha * jnp.sum(active_row_mask)
    stable_grad += 0.5 * d_m * (k_m - sum_alpha)
    v_value = log_var[0]
    stable_grad = jnp.where(
        ((v_value > -20.0) & (v_value < 20.0))
        | ((v_value <= -20.0) & (stable_grad < 0.0))
        | ((v_value >= 20.0) & (stable_grad > 0.0)),
        stable_grad,
        0.0,
    )

    level_norm_sq_grad = g * (alpha / (2.0 * prior_var)) * active_row_mask

    tilde_f_grad = (
        g
        * (alpha / 2.0)
        * active_row_mask[:, None]
        / (tilde_f_gathered + 1.0 / prior_var + 1e-8)
    )

    return (
        jnp.array([stable_grad]) * g,
        level_norm_sq_grad,
        tilde_f_grad,
        None,
        None,
        None,
        None,
    )

  return loss_val, grad_fn


class PCTRModel(nn.Module):
  """Poisson Regression DNN Model for pCTR."""

  embedding_configs: Dict[str, tuple[int, int]]
  multivalent_features: List[str]
  use_wn_all_layer: bool = True
  activation_fn: str = 'relu'

  def _apply_weight_norm_dense_no_g(self, x, dense_layer, bias):
    raw_proj = dense_layer(x)
    v = self.variables['params'][dense_layer.name]['kernel']
    v_norm = jnp.linalg.norm(v, axis=0, keepdims=True) + 1e-8
    return raw_proj / v_norm + bias

  def setup(self):
    # Create embeddings dictionary
    self.embeddings = {
        name: nn.Embed(num_embeddings=cfg[0], features=cfg[1])
        for name, cfg in self.embedding_configs.items()
    }
    # Dense Layers
    if self.use_wn_all_layer:
      self.dense1 = nn.Dense(512, use_bias=False)
      self.dense2 = nn.Dense(256, use_bias=False)
      self.dense3 = nn.Dense(128, use_bias=False)
      self.bias1 = self.param('bias1', nn.initializers.zeros, (512,))
      self.bias2 = self.param('bias2', nn.initializers.zeros, (256,))
      self.bias3 = self.param('bias3', nn.initializers.zeros, (128,))
    else:
      self.dense1 = nn.Dense(512)
      self.dense2 = nn.Dense(256)
      self.dense3 = nn.Dense(128)

    self.output_layer = nn.Dense(1)  # Linear output for log-rate

  def __call__(
      self, inputs: Dict[str, jnp.ndarray], masks: Dict[str, jnp.ndarray]
  ):
    embedded_features = []

    for name in self.embedding_configs:
      x = inputs[name]
      # Lookup
      embeds = self.embeddings[name](
          x
      )  # Shape: (Batch, Len, Dim) or (Batch, Dim)

      if name in self.multivalent_features:
        # Average pooling for multivalent features, ignoring padding
        mask = masks[name][..., jnp.newaxis]  # (Batch, Len, 1)
        valid_embeds = embeds * mask
        pooled = jnp.sum(valid_embeds, axis=1) / (jnp.sum(mask, axis=1) + 1e-6)
        embedded_features.append(pooled)
      else:
        embedded_features.append(embeds)

    # Concatenate all embeddings
    x = jnp.concatenate(embedded_features, axis=-1)

    # MLP
    activation = nn.tanh if self.activation_fn == 'tanh' else nn.relu
    if self.use_wn_all_layer:
      x = activation(
          self._apply_weight_norm_dense_no_g(x, self.dense1, self.bias1)
      )
      x = activation(
          self._apply_weight_norm_dense_no_g(x, self.dense2, self.bias2)
      )
      x = activation(
          self._apply_weight_norm_dense_no_g(x, self.dense3, self.bias3)
      )
    else:
      x = activation(self.dense1(x))
      x = activation(self.dense2(x))
      x = activation(self.dense3(x))

    log_lambda = self.output_layer(x)

    return jnp.squeeze(log_lambda, axis=-1)


class PFRCTRModel(nn.Module):
  """Poisson Regression DNN Model with Per-Feature Regularization (PFR)."""

  embedding_configs: Dict[str, tuple[int, int]]
  multivalent_features: List[str]
  share_tau: bool = False
  use_wn_all_layer: bool = True
  initial_tau: float = 1.0
  init_emb_with_tau: bool = False

  def _apply_weight_norm_dense_no_g(self, x, dense_layer, bias):
    raw_proj = dense_layer(x)
    v = self.variables['params'][dense_layer.name]['kernel']
    v_norm = jnp.linalg.norm(v, axis=0, keepdims=True) + 1e-8
    return raw_proj / v_norm + bias

  def setup(self):
    if self.init_emb_with_tau:
      self.embeddings = {
          name: nn.Embed(
              num_embeddings=cfg[0],
              features=cfg[1],
              embedding_init=nn.initializers.normal(stddev=self.initial_tau),
          )
          for name, cfg in self.embedding_configs.items()
      }
    else:
      self.embeddings = {
          name: nn.Embed(num_embeddings=cfg[0], features=cfg[1])
          for name, cfg in self.embedding_configs.items()
      }
    if self.use_wn_all_layer:
      self.dense1 = nn.Dense(512, use_bias=False)
      self.dense2 = nn.Dense(256, use_bias=False)
      self.dense3 = nn.Dense(128, use_bias=False)
      self.bias1 = self.param('bias1', nn.initializers.zeros, (512,))
      self.bias2 = self.param('bias2', nn.initializers.zeros, (256,))
      self.bias3 = self.param('bias3', nn.initializers.zeros, (128,))
    else:
      self.dense1 = nn.Dense(512)
      self.dense2 = nn.Dense(256)
      self.dense3 = nn.Dense(128)

    self.output_layer = nn.Dense(1)

    initial_log_var = 2.0 * np.log(self.initial_tau)

    if self.share_tau:
      shared_log_var = self.param(
          'log_variance_shared', nn.initializers.constant(initial_log_var), (1,)
      )
      self.log_variance = {
          name: shared_log_var for name in self.embedding_configs
      }
    else:
      self.log_variance = {
          name: self.param(
              f'log_variance_{name}',
              nn.initializers.constant(initial_log_var),
              (1,),
          )
          for name in self.embedding_configs
      }

  def get_embeddings(
      self, inputs: Dict[str, jnp.ndarray], masks: Dict[str, jnp.ndarray]
  ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    embedded_features = []
    transformed_inputs = {}
    for name in self.embedding_configs:
      x = inputs[name]
      embeds = self.embeddings[name](x)
      if name in self.multivalent_features:
        mask = masks[name][..., jnp.newaxis]
        valid_embeds = embeds * mask
        pooled = jnp.sum(valid_embeds, axis=1) / (jnp.sum(mask, axis=1) + 1e-6)
        embedded_features.append(pooled)
        transformed_inputs[name] = pooled
      else:
        embedded_features.append(embeds)
        transformed_inputs[name] = embeds
    return jnp.concatenate(embedded_features, axis=-1), transformed_inputs

  def forward_mlp(self, x: jnp.ndarray) -> jnp.ndarray:
    if self.use_wn_all_layer:
      x = nn.tanh(
          self._apply_weight_norm_dense_no_g(x, self.dense1, self.bias1)
      )
      x = nn.tanh(
          self._apply_weight_norm_dense_no_g(x, self.dense2, self.bias2)
      )
      x = nn.tanh(
          self._apply_weight_norm_dense_no_g(x, self.dense3, self.bias3)
      )
    else:
      x = nn.tanh(self.dense1(x))
      x = nn.tanh(self.dense2(x))
      x = nn.tanh(self.dense3(x))
    return x

  def logits_from_embeddings(self, x: jnp.ndarray) -> jnp.ndarray:
    prelogits = self.forward_mlp(x)
    logits = self.output_layer(prelogits)
    # Clip logits for numerical stability during exponentiation.
    # Note: Clipping clamps derivatives in VJP to 0 for saturated examples,
    # which slightly zeroes out their Fisher contribution.
    return jnp.clip(logits, -20.0, 20.0)

  def __call__(
      self, inputs: Dict[str, jnp.ndarray], masks: Dict[str, jnp.ndarray]
  ):
    x, _ = self.get_embeddings(inputs, masks)
    logits = self.logits_from_embeddings(x)
    return jnp.squeeze(logits, axis=-1)

  def compute_pfr_loss(
      self,
      g_i: jnp.ndarray | Dict[str, jnp.ndarray],
      lambdas: jnp.ndarray | Dict[str, jnp.ndarray],
      head_loss: jnp.ndarray,
      inputs: Dict[str, jnp.ndarray],
      masks: Dict[str, jnp.ndarray],
      feature_slices: dict[str, tuple[int, int]],
      embedding_tables: dict[str, jnp.ndarray],
      sample_weight: jnp.ndarray,
      init_phase: bool,
  ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Computes Diagonal Fisher and PFR regularization.

    Args:
      g_i: Gradients of logits w.r.t. concatenated embeddings, shaped
        (Global_batch, Total_Dim), or Dict of feature names to their gradients.
      lambdas: Predicted Poisson rates (exp(logits)), shaped (Global_batch,), or
        Dict of feature names to their rates.
      head_loss: Base task loss (e.g., Poisson head loss), scalar.
      inputs: Dict of feature names to their integer embedding indices in the
        batch.
      masks: Dict of feature names to their validity masks (for padding).
      feature_slices: Map of feature names to their (start, end) index ranges in
        concatenated embedding vector.
      embedding_tables: Dict of current embedding tables.
      sample_weight: Per-sample weights for the batch.
      init_phase: Whether to run only prior regularization (freeze period).

    Returns:
      A tuple of (total_loss, head_loss, scaled_pfr_loss, scaled_c1, scaled_c2)
    """
    # pylint: disable=invalid-name
    pfr_loss = 0.0
    total_c1 = 0.0
    total_c2 = 0.0
    if isinstance(lambdas, dict):
      B = sample_weight.shape[0]
    else:
      B = lambdas.shape[0]
    N = _NUM_TRAIN_EXAMPLES.value

    for name, cfg in self.embedding_configs.items():
      # k_m is the vocabulary size, , d_m is the embedding dimension
      k_m = cfg[0]
      d_m = cfg[1]

      indices = inputs[name]
      if jnp.ndim(indices) == 1:
        indices = jnp.expand_dims(indices, axis=-1)

      _, seq_len = indices.shape

      weights = masks.get(name, None)
      if weights is None:
        weights = jnp.ones_like(indices, dtype=jnp.float32)

      safe_indices = jnp.where(indices < 0, 0, indices)
      sum_weights = jnp.sum(weights, axis=-1, keepdims=True)
      safe_sum_weights = jnp.where(sum_weights == 0.0, 1.0, sum_weights)
      norm_weights = jnp.where(
          sum_weights == 0.0, 0.0, weights / safe_sum_weights
      )

      token_active = norm_weights > 0.0
      # same_id physical shape per core: (Local_batch, seq_len, seq_len)
      # Holds only Local information (deduplicates within each sequence).
      same_id = safe_indices[:, :, None] == safe_indices[:, None, :]
      # row_weight_by_occurrence physical shape per core: (Local_batch, seq_len)
      # Holds only Local information.
      row_weight_by_occurrence = jnp.sum(
          jnp.where(same_id, norm_weights[:, None, :], 0.0), axis=-1
      )
      token_positions = jnp.arange(seq_len)
      # earlier_token shape: (1, seq_len, seq_len)
      earlier_token = (
          token_positions[None, None, :] < token_positions[None, :, None]
      )
      # earlier_same_id physical shape per core: (Local_batch, seq_len)
      # Holds only Local information.
      earlier_same_id = jnp.any(
          same_id & token_active[:, None, :] & earlier_token, axis=-1
      )
      # first_occurrence physical shape per core: (Local_batch, seq_len)
      # Holds only Local information.
      first_occurrence = token_active & ~earlier_same_id

      # flat_indices physical shape per core: (Local_batch * seq_len,)
      # Holds only Local vocabulary IDs.
      flat_indices = safe_indices.reshape(-1)
      # flat_row_weights physical shape per core: (Local_batch * seq_len, 1)
      # Holds only Local information.
      flat_row_weights = (
          row_weight_by_occurrence * first_occurrence.astype(jnp.float32)
      ).reshape(-1, 1)

      # flat_sample_weights physical shape per core: (Local_batch * seq_len,)
      # Holds only Local weights.
      flat_sample_weights = jnp.repeat(sample_weight, seq_len, axis=0)

      num_dense_segments = flat_indices.shape[0]
      # NOTE: jnp.unique and segment_sum induce global communication under SPMD.
      # unique_indices and dense_batch_fisher represent Global aggregated
      # concepts. unique_indices shape (conceptual, also likely physical):
      # (Global_batch * seq_len,)
      unique_indices, dense_inverse = jnp.unique(
          flat_indices,
          return_inverse=True,
          size=num_dense_segments,
          fill_value=k_m,
      )
      valid_unique = unique_indices < k_m
      safe_unique_indices = jnp.where(valid_unique, unique_indices, 0)

      # flat_visits physical shape per core: (Local_batch * seq_len,)
      # Holds only Local visitation indicators.
      flat_visits = (
          first_occurrence.reshape(-1).astype(jnp.float32) * flat_sample_weights
      )
      # dense_batch_visits shape (conceptual, also likely physical):
      # (Global_batch * seq_len,)
      # Holds Globally Aggregated Visit counts for unique batch tokens.
      dense_batch_visits = jax.ops.segment_sum(
          flat_visits, dense_inverse, num_segments=num_dense_segments
      )

      touched = ((dense_batch_visits > 0.0) & valid_unique).astype(jnp.float32)

      embedding_table = embedding_tables[name]
      # embedding_table_gathered shape (conceptual, also likely physical):
      # (Global_batch * seq_len, d_m)
      embedding_table_gathered = embedding_table[safe_unique_indices]
      # level_norm_sq_gathered shape (conceptual, also likely physical):
      # (Global_batch * seq_len,)
      level_norm_sq_gathered = jnp.sum(embedding_table_gathered**2, axis=-1)

      S_m = jnp.sum(touched)
      alpha = k_m / jnp.maximum(S_m, 1.0)

      log_var = self.log_variance[name]

      if init_phase:
        clipped_log_var = jnp.clip(log_var[0], -20.0, 20.0)
        prior_var = jnp.exp(clipped_log_var)

        weighted_norm_sq = level_norm_sq_gathered * touched
        c1 = (alpha / (2.0 * prior_var)) * jnp.sum(weighted_norm_sq) + (
            k_m * d_m / 2.0
        ) * clipped_log_var

        pfr_loss += c1
        total_c1 += c1
      else:
        if isinstance(g_i, dict):
          g_i_m = g_i[name]
        else:
          start, end = feature_slices[name]
          g_i_m = g_i[:, start:end]

        if isinstance(lambdas, dict):
          local_lambdas = lambdas[name]
        else:
          local_lambdas = lambdas

        # flat_diag_fisher physical shape per core: (Local_batch * seq_len, d_m)
        # Holds instantaneous Local token-level Fisher Information.
        flat_g_i = jnp.repeat(g_i_m, seq_len, axis=0) * flat_row_weights
        flat_lambdas = jnp.repeat(local_lambdas, seq_len, axis=0)
        flat_diag_fisher = (
            jnp.expand_dims(flat_lambdas, axis=-1)
            * (flat_g_i**2)
            * jnp.expand_dims(flat_sample_weights, axis=-1)
        )
        # dense_batch_fisher shape (conceptual, also likely physical):
        # (Global_batch * seq_len, d_m)
        # Holds Globally Aggregated Fisher contribution for unique batch tokens.
        dense_batch_fisher = jax.ops.segment_sum(
            flat_diag_fisher, dense_inverse, num_segments=num_dense_segments
        )
        tilde_f_gathered = (N / B) * dense_batch_fisher

        feature_loss = stable_pfr_loss_for_feature(
            log_var,
            level_norm_sq_gathered,
            tilde_f_gathered,
            k_m,
            d_m,
            alpha,
            touched,
        )
        pfr_loss += feature_loss

        # Recompute components for logging
        clipped_log_var = jnp.clip(log_var[0], -20.0, 20.0)
        prior_var = jnp.exp(clipped_log_var)

        weighted_norm_sq = level_norm_sq_gathered * touched
        c1 = (
            jnp.sum((alpha / (2.0 * prior_var)) * weighted_norm_sq)
            + (k_m * d_m / 2.0) * clipped_log_var
        )
        total_c1 += c1

        log_det_gathered = jnp.sum(
            jnp.log(tilde_f_gathered + 1.0 / prior_var + 1e-8), axis=-1
        )
        weighted_log_det = log_det_gathered * touched
        c2 = jnp.sum((alpha / 2.0) * weighted_log_det)
        total_c2 += c2

    total_loss = head_loss + (1.0 / N) * pfr_loss
    scaled_pfr_loss = (1.0 / N) * pfr_loss
    scaled_c1 = (1.0 / N) * total_c1
    scaled_c2 = (1.0 / N) * total_c2
    return total_loss, head_loss, scaled_pfr_loss, scaled_c1, scaled_c2


def get_tf_feature_spec():
  """Returns the TensorFlow parsing spec for the features."""
  spec = {}
  for name, _ in EMBEDDING_CONFIGS.items():
    dtype = tf.int64

    if name in MULTIVALENT_FEATURES:
      spec[name] = tf.io.VarLenFeature(dtype)
    else:
      spec[name] = tf.io.FixedLenFeature([], dtype, default_value=0)

  # Labels and Weights
  spec['ppc_y_i'] = tf.io.FixedLenFeature([], tf.int64, default_value=0)
  spec['sample_weight'] = tf.io.FixedLenFeature(
      [], tf.float32, default_value=1.0
  )
  return spec


def make_dataset(file_patterns, batch_size, is_training=True):
  """Creates a tf.data.Dataset from RecordIO files."""
  feature_spec = get_tf_feature_spec()

  def batched_parse_fn(serialized):
    parsed = tf.io.parse_example(serialized, feature_spec)
    processed_inputs = {}
    masks = {}

    for name, cfg in EMBEDDING_CONFIGS.items():
      tensor = parsed[name]

      # Handle Sparse Tensors (Multivalent)
      if isinstance(tensor, tf.SparseTensor):
        tensor = tf.sparse.to_dense(tensor, default_value=-1)

        max_len = _MAX_SEQ_LEN.value
        curr_len = tf.shape(tensor)[1]
        pad_size = tf.maximum(0, max_len - curr_len)

        tensor = tf.pad(tensor, [[0, 0], [0, pad_size]], constant_values=-1)[
            :, :max_len
        ]
        pad_value = -1

        # Create mask
        mask = tf.cast(
            tf.not_equal(tensor, pad_value),
            tf.float32,
        )
        masks[name] = mask

        # Prevent padding value (-1) from interfering with hashing or
        # clipping bounds
        tensor = tf.where(
            tensor == -1, tf.constant(0, dtype=tensor.dtype), tensor
        )

      # Apply Hashing if needed
      if HASH_STRATEGIES.get(name, False):
        tensor = tf.math.floormod(tensor, cfg[0])
      else:
        # Direct indexing, ensure bounded
        tensor = tf.clip_by_value(tensor, 0, cfg[0] - 1)

      # Cast integers to int32 for TPU compatibility
      if tensor.dtype == tf.int64:
        tensor = tf.cast(tensor, tf.int32)

      processed_inputs[name] = tensor

    # Add labels
    labels = {
        'ppc_y_i': tf.cast(parsed['ppc_y_i'], tf.float32),
        'sample_weight': parsed['sample_weight'],
    }
    return processed_inputs, masks, labels

  dataset = tf.data.Dataset.list_files(file_patterns, shuffle=is_training)
  if jax.process_count() > 1:
    dataset = dataset.shard(
        num_shards=jax.process_count(), index=jax.process_index()
    )

  dataset = dataset.interleave(
      array_record_dataset.ArrayRecordDataset,
      cycle_length=64,  # Read up to 64 files concurrently
      num_parallel_calls=tf.data.AUTOTUNE,
      deterministic=False,  # Yield examples as soon as they are ready
  )
  if is_training:
    dataset = dataset.shuffle(100000)

  dataset = dataset.batch(batch_size, drop_remainder=True)
  dataset = dataset.map(
      batched_parse_fn,
      num_parallel_calls=tf.data.AUTOTUNE,
      deterministic=False,  # Allow out-of-order parsing for speed
  )
  dataset = dataset.prefetch(tf.data.AUTOTUNE)
  return dataset


def poisson_loss(log_lambda, y, sample_weight):
  """Computes weighted Poisson loss."""
  log_lambda = jnp.clip(log_lambda, -20.0, 20.0)  # Numerical guard
  loss = jnp.exp(log_lambda) - y * log_lambda
  weighted_loss = loss * sample_weight
  return jnp.mean(weighted_loss)


def train_step(state, inputs, masks, labels):
  """Performs a single training step."""

  def loss_fn(params):
    log_lambda = state.apply_fn({'params': params}, inputs, masks)
    loss = poisson_loss(log_lambda, labels['ppc_y_i'], labels['sample_weight'])

    l2_reg = _L2_REG_WEIGHT.value
    if l2_reg > 0:
      l2_penalty = 0.0
      for name, cfg in EMBEDDING_CONFIGS.items():
        vocab_size = cfg[0]
        embedding_table = params[f'embeddings_{name}']['embedding']
        indices = inputs[name]

        if jnp.ndim(indices) == 1:
          indices = jnp.expand_dims(indices, axis=-1)

        # Map negative indices (padding) to vocab_size (filler)
        safe_indices = jnp.where(indices < 0, vocab_size, indices)
        flat_indices = safe_indices.reshape(-1)
        num_segments = flat_indices.shape[0]

        # Deduplicate to find unique touched rows
        unique_indices = jnp.unique(
            flat_indices,
            size=num_segments,
            fill_value=vocab_size,
        )
        valid_unique = unique_indices < vocab_size
        safe_unique_indices = jnp.where(valid_unique, unique_indices, 0)

        touched = valid_unique.astype(jnp.float32)
        embedding_table_gathered = embedding_table[safe_unique_indices]
        level_norm_sq_gathered = jnp.sum(embedding_table_gathered**2, axis=-1)

        if _L2_REG_STYLE.value == 'standard':
          num_touched = jnp.sum(touched)
          alpha = vocab_size / jnp.maximum(num_touched, 1.0)
          l2_penalty += alpha * jnp.sum(level_norm_sq_gathered * touched)
        else:  # 'sgd_l2_wd'
          l2_penalty += jnp.sum(level_norm_sq_gathered * touched)

      if _L2_REG_STYLE.value == 'standard':
        num_examples = _NUM_TRAIN_EXAMPLES.value
        loss += (1.0 / num_examples) * 0.5 * l2_reg * l2_penalty
      else:
        loss += 0.5 * l2_reg * l2_penalty

    return loss

  grad_fn = jax.value_and_grad(loss_fn)
  loss, grads = grad_fn(state.params)
  state = state.apply_gradients(grads=grads)
  return state, loss


def train_step_pfr(state, inputs, masks, labels, model, is_freeze_period=False):
  """Performs a single training step for PFR."""

  # Compute inside pfr_loss_fn to avoid double pass
  # (inputs would otherwise be evaluated twice)
  def pfr_loss_fn(params):
    # 1. Forward to get pooled embeddings x natively inside the grad trace
    def get_embeddings_fn(p):
      return model.apply(
          {'params': p},
          inputs,
          masks,
          method=model.get_embeddings,
      )

    x, transformed_inputs = get_embeddings_fn(params)

    # 3. Calculate feature slices
    start_idx = 0
    feature_slices = {}
    active_feature_names = list(model.embedding_configs.keys())
    for name in active_feature_names:
      dim = transformed_inputs[name].shape[-1]
      feature_slices[name] = (start_idx, start_idx + dim)
      start_idx += dim

    # Compute base head loss purely from inner variables
    def logits_fn_differentiable(inner_x):
      return model.apply(
          {'params': params},
          inner_x,
          method=model.logits_from_embeddings,
      )

    log_lambda = jnp.squeeze(logits_fn_differentiable(x), axis=-1)
    head_loss = poisson_loss(
        log_lambda, labels['ppc_y_i'], labels['sample_weight']
    )

    # 2. Compute g_i w.r.t logits (eta) using jax.vjp.
    # Dense parameters are stopped to prevent dragging them into the
    # regularization loop.
    def logits_fn_stopped(inner_x):
      return model.apply(
          {
              'params': jax.lax.stop_gradient(params),
          },
          inner_x,
          method=model.logits_from_embeddings,
      )

    if is_freeze_period:
      g_i = None
    else:
      if _EMB_OPT_USING_C2.value:
        if _MLP_OPT_USING_C2.value:
          logits, vjp_fn = jax.vjp(logits_fn_differentiable, x)
        else:
          logits, vjp_fn = jax.vjp(logits_fn_stopped, x)
        g_i = vjp_fn(jnp.ones_like(logits))[0]
      else:
        logits, vjp_fn = jax.vjp(logits_fn_stopped, jax.lax.stop_gradient(x))
        g_i = jax.lax.stop_gradient(vjp_fn(jnp.ones_like(logits))[0])

    lambdas_for_fisher = jax.lax.stop_gradient(jnp.exp(log_lambda))

    candidate_embedding_tables = {
        name: params[f'embeddings_{name}']['embedding']
        for name in active_feature_names
    }

    # Apply PFR regularizations
    loss_tuple = model.apply(
        {'params': params},
        g_i,
        lambdas_for_fisher,
        head_loss,
        inputs,
        masks,
        feature_slices,
        candidate_embedding_tables,
        labels['sample_weight'],
        is_freeze_period,
        method=model.compute_pfr_loss,
    )
    total_loss, head_loss_val, pfr_loss_val, c1_val, c2_val = loss_tuple

    return total_loss, (
        log_lambda,
        head_loss_val,
        pfr_loss_val,
        c1_val,
        c2_val,
    )

  grad_fn = jax.value_and_grad(pfr_loss_fn, has_aux=True)
  (
      total_loss,
      (_, head_loss_val, pfr_loss_val, c1_val, c2_val),
  ), grads = grad_fn(state.params)

  # 7. Apply gradients to state
  state = state.apply_gradients(grads=grads)

  return (
      state,
      total_loss,
      head_loss_val,
      pfr_loss_val,
      c1_val,
      c2_val,
  )


def eval_step(state, inputs, masks, labels):
  """Performs a single evaluation step on TPU."""
  log_lambda = state.apply_fn({'params': state.params}, inputs, masks)
  loss = poisson_loss(log_lambda, labels['ppc_y_i'], labels['sample_weight'])
  return loss, log_lambda


def eval_step_pfr(state, inputs, masks, labels):
  """Performs a single evaluation step on TPU for PFR."""
  log_lambda = state.apply_fn({'params': state.params}, inputs, masks)
  loss = poisson_loss(log_lambda, labels['ppc_y_i'], labels['sample_weight'])
  return loss, log_lambda


def run_training():
  """The main training function."""
  # Mesh / Multi-device setup (Pure Data Parallelism for Dense,
  # Model Parallelism for Embeddings)
  local_num_devices = jax.local_device_count()
  global_devices = jax.devices()
  mesh = jax.sharding.Mesh(global_devices, ('batch',))

  # Define Sharding Specs
  # We use tree_map_with_path to dynamically assign sharding strategies.
  # 1. Embeddings: Partitioned along the vocabulary dimension ('batch' axis of
  #    the mesh). This prevents dense gradient all-reduces across the
  #    interconnect.
  # 2. Dense Layers: Replicated across all devices.
  def sharding_policy(path, val):
    if _REPLICATE_ALL_EMBEDDINGS.value:
      return jax.sharding.PartitionSpec()  # Replicate everything (R2 way)

    str_path = '/'.join(str(k.key) for k in path if hasattr(k, 'key'))
    if 'embeddings' in str_path:
      # Only partition large embedding tables if
      # divisible by number of devices.
      # Otherwise fallback to replication to avoid IndivisibleError.
      vocab_size = val.shape[0]
      num_devices = jax.device_count()
      if vocab_size >= 100000 and vocab_size % num_devices == 0:
        # The vocabulary axis is the first dimension of the embedding kernel.
        return jax.sharding.PartitionSpec('batch')
    return jax.sharding.PartitionSpec()  # Replicate

  # Partition batches across the 'batch' axis of the mesh
  data_sharding = jax.sharding.NamedSharding(
      mesh,
      jax.sharding.PartitionSpec(
          'batch',
      ),
  )

  logging.info('Local Devices: %s', local_num_devices)
  logging.info('Global Devices: %s', jax.device_count())

  # Get file patterns
  def _add_readahead(path: str) -> str:
    if path.startswith('/cns/'):
      return f'/readahead/256M{path}'
    return path

  train_files = _add_readahead(_TRAIN_FILES.value)
  validation_files = _add_readahead(_VAL_FILES.value)

  # Batch size: scale locally proportional to number of available local devices.
  per_device_batch_size = _PER_DEVICE_BATCH_SIZE.value
  local_batch_size = per_device_batch_size * local_num_devices
  logging.info('Per-device batch size: %d', per_device_batch_size)
  logging.info('Local batch size: %d', local_batch_size)
  logging.info(
      'Global batch size (Total): %d',
      per_device_batch_size * jax.device_count(),
  )

  train_ds = make_dataset(train_files, local_batch_size, is_training=True)
  val_ds = make_dataset(validation_files, local_batch_size, is_training=False)

  # Setup TensorBoard Summary Writer (Only on Leader)
  summary_writer = None
  if jax.process_index() == 0:
    summary_writer = tf.summary.create_file_writer(_WORKING_DIR.value)

  # Initialize validation AUC metric
  # Using large num_thresholds for precision on low-CTR data
  val_auc_metric = tf.keras.metrics.AUC(
      name='val_auc', from_logits=True, num_thresholds=10000
  )
  val_head_nll_metric = tf.keras.metrics.Mean(name='val_head_nll')
  val_torso_nll_metric = tf.keras.metrics.Mean(name='val_torso_nll')
  val_tail_nll_metric = tf.keras.metrics.Mean(name='val_tail_nll')

  if _MODEL_TYPE.value == 'pfr':
    model = PFRCTRModel(
        embedding_configs=EMBEDDING_CONFIGS,
        multivalent_features=MULTIVALENT_FEATURES,
        share_tau=_PFR_SHARE_TAU.value,
        use_wn_all_layer=_USE_WN_ALL_LAYER.value,
        initial_tau=_PFR_INITIAL_TAU.value,
        init_emb_with_tau=_INIT_EMB_WITH_TAU.value,
    )
  else:
    model_class = PCTRModel

    if model_class == PCTRModel:
      model = model_class(
          embedding_configs=EMBEDDING_CONFIGS,
          multivalent_features=MULTIVALENT_FEATURES,
          use_wn_all_layer=_USE_WN_ALL_LAYER.value,
          activation_fn=_ACTIVATION_FN.value,
      )
    else:
      model = model_class(
          embedding_configs=EMBEDDING_CONFIGS,
          multivalent_features=MULTIVALENT_FEATURES,
          activation_fn=_ACTIVATION_FN.value,
      )

  # Dummy inputs for initialization
  dummy_inputs = {}
  dummy_masks = {}
  init_batch = 2
  for name in EMBEDDING_CONFIGS:
    if name in MULTIVALENT_FEATURES:
      dummy_inputs[name] = jnp.zeros(
          (init_batch, _MAX_SEQ_LEN.value), dtype=jnp.int32
      )
      dummy_masks[name] = jnp.zeros(
          (init_batch, _MAX_SEQ_LEN.value), dtype=jnp.float32
      )
    else:
      dummy_inputs[name] = jnp.zeros((init_batch,), dtype=jnp.int32)

  key = jax.random.PRNGKey(0)
  init_variables = model.init(key, dummy_inputs, dummy_masks)
  params = init_variables['params']

  # --- Split Optimization Setup ---
  # Define labels for parameter tree
  def map_path_to_label(path, _):
    # path is a tuple of keys. We check for 'embeddings' or embedding key words.
    str_path = '/'.join(str(k.key) for k in path if hasattr(k, 'key'))
    if 'embeddings' in str_path:
      return 'embed'
    if (
        _MODEL_TYPE.value == 'pfr'
        and _INIT_PHASE_PCT.value > 0.0
        and 'log_variance' in str_path
    ):
      return 'tau'
    return 'dense'

  param_labels = jax.tree_util.tree_map_with_path(map_path_to_label, params)

  # --- Learning Rate Schedules ---
  # Compute dynamic train steps and warmup
  per_device_batch_size = _PER_DEVICE_BATCH_SIZE.value
  global_batch_size = per_device_batch_size * jax.device_count()
  steps_per_epoch = _NUM_TRAIN_EXAMPLES.value // global_batch_size
  num_epochs = _NUM_EPOCHS.value

  warmup_steps = int(steps_per_epoch * 0.1)

  embed_lr_schedule = optax.join_schedules(
      [
          optax.linear_schedule(0.0, _EMBED_LR.value, warmup_steps),
          optax.constant_schedule(_EMBED_LR.value),
      ],
      boundaries=[warmup_steps],
  )

  dense_lr_schedule = optax.join_schedules(
      [
          optax.linear_schedule(0.0, _DENSE_LR.value, warmup_steps),
          optax.constant_schedule(_DENSE_LR.value),
      ],
      boundaries=[warmup_steps],
  )

  # Custom Schedule for Tau (Freeze for first percent, then use dense schedule)
  total_steps = steps_per_epoch * num_epochs
  init_phase_steps = int(total_steps * (_INIT_PHASE_PCT.value / 100.0))
  tau_lr_schedule = dense_lr_schedule  # Fallback to satisfy linter
  if _INIT_PHASE_PCT.value > 0.0:
    tau_lr_schedule = optax.join_schedules(
        [
            optax.constant_schedule(0.0),
            dense_lr_schedule,
        ],
        boundaries=[init_phase_steps],
    )

  # Optax Optimizers Configuration
  optimizers = {
      'embed': optax.adagrad(
          learning_rate=embed_lr_schedule, initial_accumulator_value=1.0
      ),
      'dense': optax.adam(learning_rate=dense_lr_schedule),
  }
  if _MODEL_TYPE.value == 'pfr' and _INIT_PHASE_PCT.value > 0.0:
    optimizers['tau'] = optax.adam(learning_rate=tau_lr_schedule)

  # Adagrad for Embeddings, Adam for Dense Tower and Tau
  tx = optax.chain(
      optax.clip_by_global_norm(1.0),
      optax.multi_transform(
          optimizers,
          param_labels,
      ),
  )

  state = train_state.TrainState.create(
      apply_fn=model.apply, params=params, tx=tx
  )

  # Push state to devices according to our dynamic sharding policy
  replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
  param_sharding_tree = jax.tree_util.tree_map_with_path(
      lambda path, _: jax.sharding.NamedSharding(
          mesh, sharding_policy(path, _)
      ),
      params,
  )
  state_sharding = train_state.TrainState(
      step=replicated,
      apply_fn=state.apply_fn,
      params=param_sharding_tree,
      tx=state.tx,
      opt_state=jax.tree_util.tree_map(lambda _: replicated, state.opt_state),
  )

  state = jax.device_put(state, state_sharding)

  # JIT compile the training and evaluation steps with explicit shardings
  jitted_train_step_init = None
  jitted_train_step_standard = None
  jitted_train_step = None
  if _MODEL_TYPE.value == 'pfr':
    jitted_train_step_init = jax.jit(
        lambda s, i, m, l: train_step_pfr(
            s, i, m, l, model, is_freeze_period=True
        ),
        in_shardings=(
            state_sharding,
            data_sharding,
            data_sharding,
            data_sharding,
        ),
        out_shardings=(
            state_sharding,
            jax.sharding.NamedSharding(
                mesh, jax.sharding.PartitionSpec()
            ),  # total_loss
            jax.sharding.NamedSharding(
                mesh, jax.sharding.PartitionSpec()
            ),  # head_loss_val
            jax.sharding.NamedSharding(
                mesh, jax.sharding.PartitionSpec()
            ),  # pfr_loss_val
            jax.sharding.NamedSharding(
                mesh, jax.sharding.PartitionSpec()
            ),  # c1_val
            jax.sharding.NamedSharding(
                mesh, jax.sharding.PartitionSpec()
            ),  # c2_val
        ),
    )
    jitted_train_step_standard = jax.jit(
        lambda s, i, m, l: train_step_pfr(
            s, i, m, l, model, is_freeze_period=False
        ),
        in_shardings=(
            state_sharding,
            data_sharding,
            data_sharding,
            data_sharding,
        ),
        out_shardings=(
            state_sharding,
            jax.sharding.NamedSharding(
                mesh, jax.sharding.PartitionSpec()
            ),  # total_loss
            jax.sharding.NamedSharding(
                mesh, jax.sharding.PartitionSpec()
            ),  # head_loss_val
            jax.sharding.NamedSharding(
                mesh, jax.sharding.PartitionSpec()
            ),  # pfr_loss_val
            jax.sharding.NamedSharding(
                mesh, jax.sharding.PartitionSpec()
            ),  # c1_val
            jax.sharding.NamedSharding(
                mesh, jax.sharding.PartitionSpec()
            ),  # c2_val
        ),
    )
  else:
    jitted_train_step = jax.jit(
        train_step,
        in_shardings=(
            state_sharding,
            data_sharding,
            data_sharding,
            data_sharding,
        ),
        out_shardings=(
            state_sharding,
            jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec()),
        ),
    )

  if _MODEL_TYPE.value == 'pfr':
    jitted_eval_step = jax.jit(
        eval_step_pfr,
        in_shardings=(
            state_sharding,
            data_sharding,
            data_sharding,
            data_sharding,
        ),
        out_shardings=(
            jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec()),
            data_sharding,
        ),
    )
  else:
    jitted_eval_step = jax.jit(
        eval_step,
        in_shardings=(
            state_sharding,
            data_sharding,
            data_sharding,
            data_sharding,
        ),
        out_shardings=(
            jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec()),
            data_sharding,
        ),
    )

  # Checkpointing Setup
  options = ocp.CheckpointManagerOptions(max_to_keep=3, create=True)
  ckpt_dir = os.path.join(_WORKING_DIR.value, 'checkpoints')
  ckpt_mngr = ocp.CheckpointManager(ckpt_dir, options=options)

  total_steps = 0
  latest_step = ckpt_mngr.latest_step()
  if latest_step is not None:
    logging.info('Found checkpoint at %d. Restoring state...', latest_step)

    target_for_restore = {'state': state}

    try:
      restored = ckpt_mngr.restore(
          latest_step, args=ocp.args.StandardRestore(target_for_restore)
      )
      # Handle composite vs Direct State restores for backwards compatibility
      if isinstance(restored, dict) and 'state' in restored:
        state = restored['state']
      else:
        state = restored
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.warning(
          'Failed to restore with composite structure (%s). Attempting legacy'
          ' direct state restore...',
          e,
      )
      state = ckpt_mngr.restore(
          latest_step, args=ocp.args.StandardRestore(state)
      )

    total_steps = int(state.step)
    logging.info('Restored state at step %d', total_steps)

  # Training Loop
  t_start = time.time()
  steps_at_start = total_steps

  logging.info('Steps per epoch: %d', steps_per_epoch)

  start_epoch = total_steps // steps_per_epoch
  start_step_in_epoch = total_steps % steps_per_epoch
  logging.info(
      'Starting from epoch %d, step %d in epoch',
      start_epoch,
      start_step_in_epoch,
  )

  for epoch in range(start_epoch, num_epochs):
    logging.info('Epoch %d started', epoch)
    epoch_ds = train_ds
    if epoch == start_epoch and start_step_in_epoch > 0:
      logging.info(
          'Skipping %d batches in epoch %d', start_step_in_epoch, epoch
      )
      epoch_ds = train_ds.skip(start_step_in_epoch)
    for batch in epoch_ds.as_numpy_iterator():
      inputs, masks, labels = batch

      # Push batch data to devices (Sharded across Batch Axis)
      inputs, masks, labels = jax.tree_util.tree_map(
          lambda x: jax.make_array_from_process_local_data(data_sharding, x),
          (inputs, masks, labels),
      )

      head_loss_val = None
      pfr_loss_val = None
      c1_val = None
      c2_val = None
      if _MODEL_TYPE.value == 'pfr':
        is_freeze = total_steps < init_phase_steps
        if is_freeze:
          (
              state,
              loss,
              head_loss_val,
              pfr_loss_val,
              c1_val,
              c2_val,
          ) = jitted_train_step_init(state, inputs, masks, labels)
        else:
          (
              state,
              loss,
              head_loss_val,
              pfr_loss_val,
              c1_val,
              c2_val,
          ) = jitted_train_step_standard(state, inputs, masks, labels)
      else:
        state, loss = jitted_train_step(state, inputs, masks, labels)

      if total_steps % 500 == 0:
        # Note: float() forces a device sync here every 500 steps, which is
        # acceptable for logging
        loss_val = float(loss)
        logging.info('Step %d, Loss: %f', total_steps, loss_val)

        t_now = time.time()
        steps_per_sec = 0.0
        if total_steps > 0:
          steps_per_sec = (total_steps - steps_at_start) / (t_now - t_start)
          logging.info('Step %d, Steps/sec: %f', total_steps, steps_per_sec)

        if summary_writer is not None:
          with summary_writer.as_default():
            tf.summary.scalar('train_loss', loss_val, step=total_steps)
            if head_loss_val is not None:
              tf.summary.scalar(
                  'head_loss', float(head_loss_val), step=total_steps
              )
            if pfr_loss_val is not None:
              tf.summary.scalar(
                  'pfr_loss_scaled', float(pfr_loss_val), step=total_steps
              )
            if c1_val is not None:
              tf.summary.scalar(
                  'pfr_c1_scaled', float(c1_val), step=total_steps
              )
            if c2_val is not None:
              tf.summary.scalar(
                  'pfr_c2_scaled', float(c2_val), step=total_steps
              )
            if _MODEL_TYPE.value == 'pfr':
              current_params = state.params
              if _PFR_SHARE_TAU.value:
                log_var = current_params.get('log_variance_shared', None)
                if log_var is not None:
                  log_var_val = float(log_var[0])
                  prior_var = np.exp(log_var_val)
                  if total_steps % 500 == 0:
                    logging.info('Shared Prior Variance (tau^2): %f', prior_var)
                  tf.summary.scalar(
                      'prior_var/shared', prior_var, step=total_steps
                  )
              else:
                for name in EMBEDDING_CONFIGS:
                  log_var = current_params.get(f'log_variance_{name}', None)
                  if log_var is not None:
                    log_var_val = float(log_var[0])
                    prior_var = np.exp(log_var_val)
                    # Clip log output to avoid log pollution, but send all to TB
                    if total_steps % 500 == 0:
                      logging.info(
                          'Feature: %s | Prior Variance (tau^2): %f',
                          name,
                          prior_var,
                      )
                    tf.summary.scalar(
                        f'prior_var/{name}', prior_var, step=total_steps
                    )

            if total_steps > 0:
              tf.summary.scalar(
                  'steps_per_second', steps_per_sec, step=total_steps
              )
            if total_steps % 1000 == 0:
              summary_writer.flush()

        t_start = t_now
        steps_at_start = total_steps

      # Save checkpoint periodically
      if total_steps > 0 and total_steps % 5000 == 0:
        logging.info('Saving checkpoint at step %d', total_steps)
        save_dict = {'state': state}
        ckpt_mngr.save(total_steps, args=ocp.args.StandardSave(save_dict))

        # Validation at Checkpoint boundaries (Incremental CPU Metric Handling)
        logging.info(
            'Executing validation checks over %d batches...', _EVAL_STEPS.value
        )
        val_losses = []
        all_val_preds = []
        all_val_labels = []

        for val_batch in itertools.islice(
            val_ds.as_numpy_iterator(), _EVAL_STEPS.value
        ):
          inputs_cpu, masks_cpu, labels_cpu = val_batch

          # Push batch data to devices for TPU Evaluation
          inputs_tpu, masks_tpu, labels_tpu = jax.tree_util.tree_map(
              lambda x: jax.make_array_from_process_local_data(
                  data_sharding, x
              ),
              (inputs_cpu, masks_cpu, labels_cpu),
          )

          if _MODEL_TYPE.value == 'pfr':
            val_loss_tpu, log_lambda_tpu = jitted_eval_step(
                state, inputs_tpu, masks_tpu, labels_tpu
            )
          else:
            val_loss_tpu, log_lambda_tpu = jitted_eval_step(
                state, inputs_tpu, masks_tpu, labels_tpu
            )

          val_losses.append(val_loss_tpu)

          # Fetch predictions for this batch only (bounded memory)
          preds_local = np.concatenate(
              [np.asarray(s.data) for s in log_lambda_tpu.addressable_shards]
          )

          # Gather predictions, labels, and weights from all JAX processes
          global_preds = jax_multihost_utils.process_allgather(
              preds_local, tiled=True
          )
          global_labels = jax_multihost_utils.process_allgather(
              labels_cpu['ppc_y_i'], tiled=True
          )
          global_weights = jax_multihost_utils.process_allgather(
              labels_cpu['sample_weight'], tiled=True
          )
          all_val_preds.append(global_preds)
          all_val_labels.append(global_labels)

          val_auc_metric.update_state(
              y_true=global_labels,
              y_pred=global_preds,
              sample_weight=global_weights,
          )

          campaign_ids = inputs_cpu.get('campaign_id')
          if campaign_ids is not None:
            global_campaign_ids = jax_multihost_utils.process_allgather(
                campaign_ids, tiled=True
            )
            # Total distinct campaigns in dataset: 326,679
            # Head (Top 1%): 0-3266
            # Torso (Top 1%-10%): 3267 - 32668
            # Tail (Bottom 90%): >= 32668
            head_mask = global_campaign_ids < 3267
            torso_mask = (global_campaign_ids >= 3267) & (
                global_campaign_ids < 32668
            )
            tail_mask = global_campaign_ids >= 32668

            lambdas = np.exp(global_preds)
            nll_unreduced = lambdas - global_labels * global_preds

            if np.any(head_mask):
              val_head_nll_metric.update_state(
                  nll_unreduced[head_mask],
                  sample_weight=global_weights[head_mask],
              )
            if np.any(torso_mask):
              val_torso_nll_metric.update_state(
                  nll_unreduced[torso_mask],
                  sample_weight=global_weights[torso_mask],
              )
            if np.any(tail_mask):
              val_tail_nll_metric.update_state(
                  nll_unreduced[tail_mask],
                  sample_weight=global_weights[tail_mask],
              )

        if val_losses:
          # Block and compute average validation loss
          avg_val_loss = float(jnp.mean(jnp.stack(val_losses)))
          val_auc = float(val_auc_metric.result().numpy())
          val_head_nll = float(val_head_nll_metric.result().numpy())
          val_torso_nll = float(val_torso_nll_metric.result().numpy())
          val_tail_nll = float(val_tail_nll_metric.result().numpy())

          # Compute Reward Metric
          all_preds_arr = np.concatenate(all_val_preds)
          all_labels_arr = np.concatenate(all_val_labels)
          total_samples = len(all_preds_arr)
          num_sets = total_samples // 50
          if num_sets > 0:
            preds_sets = all_preds_arr[: num_sets * 50].reshape(num_sets, 50)
            labels_sets = all_labels_arr[: num_sets * 50].reshape(num_sets, 50)
            top_pred_indices = np.argmax(preds_sets, axis=1)
            row_indices = np.arange(num_sets)
            set_rewards = labels_sets[row_indices, top_pred_indices]
            val_reward = float(np.mean(set_rewards))
          else:
            val_reward = 0.0
        else:
          avg_val_loss = 0.0
          val_auc = 0.0
          val_head_nll = 0.0
          val_torso_nll = 0.0
          val_tail_nll = 0.0
          val_reward = 0.0

        logging.info(
            'Step %d, Val Loss: %f, Val AUC: %f | Head NLL: %f, Torso NLL: %f,'
            ' Tail NLL: %f | Val Reward: %f',
            total_steps,
            avg_val_loss,
            val_auc,
            val_head_nll,
            val_torso_nll,
            val_tail_nll,
            val_reward,
        )
        if summary_writer is not None:
          with summary_writer.as_default():
            tf.summary.scalar('val_loss', avg_val_loss, step=total_steps)
            tf.summary.scalar('val_auc', val_auc, step=total_steps)
            tf.summary.scalar('val_reward', val_reward, step=total_steps)
            if val_head_nll > 0:
              tf.summary.scalar('val_head_nll', val_head_nll, step=total_steps)
              tf.summary.scalar(
                  'val_torso_nll', val_torso_nll, step=total_steps
              )
              tf.summary.scalar('val_tail_nll', val_tail_nll, step=total_steps)
            summary_writer.flush()

        # Reset metric for next eval
        val_auc_metric.reset_states()
        val_head_nll_metric.reset_states()
        val_torso_nll_metric.reset_states()
        val_tail_nll_metric.reset_states()

      total_steps += 1
    start_step_in_epoch = 0

  # Export Model (Simplified)
  export_dir = _EXPORT_DIR.value
  logging.info('Export directory: %s', export_dir)

  logging.info('Training finished')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  if _XPROF_PORT.value is not None:
    jax.profiler.start_server(_XPROF_PORT.value)

  run_training()


if __name__ == '__main__':
  app.run(main)
