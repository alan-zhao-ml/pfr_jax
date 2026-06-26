r"""Local CPU Training wrapper for Sim3 Synthetic Dataset (PFR Shared Tau).

Example usage:
  blaze run //experimental/users/alzhao/pfr_jax/test_sim3:train_pfr_sim3_cpu -- \
    --alsologtostderr \
    --batch_size=4096 \
    --pfr_initial_tau=1.0

  # Run with different Weight Normalization styles (use_wn, use_no_wn):
  blaze run //experimental/users/alzhao/pfr_jax/test_sim3:train_pfr_sim3_cpu -- \
    --alsologtostderr --wn_style=use_no_wn
"""

import math
import os
import time

# Force CPU execution locally
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"

# pylint: disable=g-import-not-at-top
from typing import Dict, List
from absl import app
from absl import flags
from absl import logging
import flax.linen as nn
from flax.training import train_state
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "train_csv",
    "/google/src/cloud/alzhao/YTAds_pCTR/google3/experimental/users/alzhao/pfr_jax/testdata/training_sim3.csv",
    "Path to training CSV.",
)
flags.DEFINE_string(
    "test_csv",
    "/google/src/cloud/alzhao/YTAds_pCTR/google3/experimental/users/alzhao/pfr_jax/testdata/test_sim3.csv",
    "Path to test CSV.",
)
flags.DEFINE_integer("batch_size", 4096, "Batch size")
flags.DEFINE_integer("epochs", 120, "Number of epochs")
flags.DEFINE_float("learning_rate", 0.01, "Base learning rate")
flags.DEFINE_float(
    "pfr_initial_tau",
    1.0,
    "Initial value for tau (standard deviation of the prior) in PFR.",
)
flags.DEFINE_enum(
    "wn_style",
    "use_wn",
    ["use_wn", "use_no_wn"],
    "Whether use row normalization in dense layers.",
)
flags.DEFINE_bool(
    "emb_opt_using_c2",
    True,
    "Whether to allow Correction 2 to optimize the embeddings.",
)
flags.DEFINE_bool(
    "mlp_opt_using_c2",
    True,
    "Whether to allow Correction 2 to optimize the MLP weights. Only effective"
    " if emb_opt_using_c2 is True.",
)
flags.DEFINE_bool(
    "sharetau",
    True,
    "Whether to share tau across all features.",
)
flags.DEFINE_bool(
    "init_emb_with_tau",
    False,
    "Whether to initialize embeddings using initial_tau as stddev.",
)
flags.DEFINE_float(
    "init_phase_pct",
    0.0,
    "Percentage (0 to 100) of training steps to compute only Data loss and"
    " Correction 1 while freezing tau.",
)
flags.DEFINE_integer(
    "num_train_examples",
    400000,
    "Number of training examples in the dataset",
)

SIM3_EMBEDDING_CONFIGS = {
    "f1": (100, 10),
    "f2": (500, 10),
    "f3": (1000, 10),
    "f4": (200, 10),
    "f5": (100, 10),
    "f6": (500, 10),
    "f7": (1000, 10),
    "f8": (200, 10),
}


def load_and_batch_csv(filepath, batch_size, shuffle=False):
    """Loads the CSV and yields JAX-compatible dictionary batches."""
    df = pd.read_csv(filepath)

    if shuffle:
        df = df.sample(frac=1).reset_index(drop=True)

    y_true = df["observed"].values.astype(np.float32)
    truth_rate = df["true_rate"].values.astype(np.float32)

    feature_data = {
        k: (df[k].values - 1).astype(np.int32)
        for k in SIM3_EMBEDDING_CONFIGS.keys()
    }

    num_samples = len(df)
    num_batches = math.ceil(num_samples / batch_size)

    for i in range(num_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, num_samples)

        batch_inputs = {
            k: jnp.asarray(v[start:end]) for k, v in feature_data.items()
        }
        batch_masks = {}
        batch_labels = {
            "y_i": jnp.asarray(y_true[start:end]),
            "sample_weight": jnp.ones((end - start,), dtype=jnp.float32),
            "true_rate": jnp.asarray(truth_rate[start:end]),
        }
        yield batch_inputs, batch_masks, batch_labels


@jax.custom_gradient
def stable_pfr_loss_for_feature(
    v_m: jnp.ndarray,
    u_k_norm_sq: jnp.ndarray,
    hat_lambda_gathered: jnp.ndarray,
    k_m: int,
    d_m: int,
    alpha: float,
    touched: jnp.ndarray,
):
    """Computes PFR correction loss with stable custom log-variance gradients."""
    clipped_v_m = jnp.clip(v_m[0], -20.0, 20.0)
    tau_sq = jnp.exp(clipped_v_m)
    touched = touched.astype(jnp.float32)

    weighted_u_k_norm_sq = u_k_norm_sq * touched
    correction1 = (alpha / (2.0 * tau_sq)) * jnp.sum(weighted_u_k_norm_sq) + (
        k_m * d_m / 2.0
    ) * clipped_v_m

    log_det_gathered = jnp.sum(
        jnp.log(hat_lambda_gathered + 1.0 / tau_sq + 1e-8), axis=-1
    )
    weighted_log_det = log_det_gathered * touched
    correction2 = (alpha / 2.0) * jnp.sum(weighted_log_det)

    loss_val = correction1 + correction2

    def grad_fn(g):
        edf_gathered = jnp.sum(
            hat_lambda_gathered / (hat_lambda_gathered + 1.0 / tau_sq + 1e-8),
            axis=-1,
        )
        weighted_edf = edf_gathered * touched

        stable_grad = (
            0.5
            * alpha
            * jnp.sum(weighted_edf - (1.0 / tau_sq) * weighted_u_k_norm_sq)
        )
        sum_alpha = alpha * jnp.sum(touched)
        stable_grad += 0.5 * d_m * (k_m - sum_alpha)
        v_value = v_m[0]
        stable_grad = jnp.where(
            ((v_value > -20.0) & (v_value < 20.0))
            | ((v_value <= -20.0) & (stable_grad < 0.0))
            | ((v_value >= 20.0) & (stable_grad > 0.0)),
            stable_grad,
            0.0,
        )

        u_k_norm_sq_grad = g * (alpha / (2.0 * tau_sq)) * touched

        hat_lambda_gathered_grad = (
            g
            * (alpha / 2.0)
            * touched[:, jnp.newaxis]
            / (hat_lambda_gathered + 1.0 / tau_sq + 1e-8)
        )

        return (
            jnp.array([stable_grad]) * g,
            u_k_norm_sq_grad,
            hat_lambda_gathered_grad,
            None,
            None,
            None,
            None,
        )

    return loss_val, grad_fn


class PFRCTRModel(nn.Module):
    """Poisson Regression DNN Model with Per-Feature Regularization (PFR)."""

    embedding_configs: Dict[str, tuple[int, int]]
    multivalent_features: List[str]
    initial_tau: float = 1.0
    wn_style: str = "use_wn"
    sharetau: bool = False
    init_emb_with_tau: bool = False

    def setup(self):
        if self.init_emb_with_tau:
            self.embeddings = {
                name: nn.Embed(
                    num_embeddings=cfg[0],
                    features=cfg[1],
                    embedding_init=nn.initializers.normal(
                        stddev=self.initial_tau
                    ),
                )
                for name, cfg in self.embedding_configs.items()
            }
        else:
            self.embeddings = {
                name: nn.Embed(num_embeddings=cfg[0], features=cfg[1])
                for name, cfg in self.embedding_configs.items()
            }
        if self.wn_style == "use_wn":
            self.dense1 = nn.Dense(32, use_bias=False)
            self.dense2 = nn.Dense(16, use_bias=False)
            self.bias1 = self.param("bias1", nn.initializers.zeros, (32,))
            self.bias2 = self.param("bias2", nn.initializers.zeros, (16,))
        elif self.wn_style == "use_no_wn":
            self.dense1 = nn.Dense(32)
            self.dense2 = nn.Dense(16)
        else:
            raise ValueError(f"Unknown wn_style: {self.wn_style}")

        self.output_layer = nn.Dense(1)

        initial_log_var = 2.0 * np.log(self.initial_tau)
        if self.sharetau:
            shared_log_var = self.param(
                "shared_log_variance",
                nn.initializers.constant(initial_log_var),
                (1,),
            )
            self.log_variance = {
                name: shared_log_var for name in self.embedding_configs
            }
        else:
            self.log_variance = {
                name: self.param(
                    f"log_variance_{name}",
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
                pooled = jnp.sum(valid_embeds, axis=1) / (
                    jnp.sum(mask, axis=1) + 1e-6
                )
                embedded_features.append(pooled)
                transformed_inputs[name] = pooled
            else:
                embedded_features.append(embeds)
                transformed_inputs[name] = embeds
        return jnp.concatenate(embedded_features, axis=-1), transformed_inputs

    def _apply_weight_norm_dense_no_g(self, x, dense_layer, bias):
        raw_proj = dense_layer(x)
        v = self.variables["params"][dense_layer.name]["kernel"]
        v_norm = jnp.linalg.norm(v, axis=0, keepdims=True) + 1e-8
        return raw_proj / v_norm + bias

    def forward_mlp(self, x: jnp.ndarray) -> jnp.ndarray:
        if self.wn_style == "use_wn":
            x = nn.tanh(
                self._apply_weight_norm_dense_no_g(x, self.dense1, self.bias1)
            )
            x = nn.tanh(
                self._apply_weight_norm_dense_no_g(x, self.dense2, self.bias2)
            )
        elif self.wn_style == "use_no_wn":
            x = nn.tanh(self.dense1(x))
            x = nn.tanh(self.dense2(x))
        else:
            raise ValueError(f"Unknown wn_style: {self.wn_style}")
        return x

    def logits_from_embeddings(self, x: jnp.ndarray) -> jnp.ndarray:
        prelogits = self.forward_mlp(x)
        logits = self.output_layer(prelogits)
        return jnp.clip(logits, -20.0, 20.0)

    def __call__(
        self, inputs: Dict[str, jnp.ndarray], masks: Dict[str, jnp.ndarray]
    ):
        x, _ = self.get_embeddings(inputs, masks)
        logits = self.logits_from_embeddings(x)
        return jnp.squeeze(logits, axis=-1)

    def update_fisher_and_compute_loss(
        self,
        g_i: jnp.ndarray,
        lambdas: jnp.ndarray,
        head_loss: jnp.ndarray,
        inputs: Dict[str, jnp.ndarray],
        masks: Dict[str, jnp.ndarray],
        feature_slices: dict[str, tuple[int, int]],
        embedding_tables: dict[str, jnp.ndarray],
        sample_weight: jnp.ndarray,
        init_phase: bool = False,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        # pylint: disable=invalid-name
        pfr_loss = 0.0
        total_correction1 = 0.0
        total_correction2 = 0.0
        B = lambdas.shape[0]
        N = FLAGS.num_train_examples

        for name, cfg in self.embedding_configs.items():
            K_m = cfg[0]
            d_m = cfg[1]
            start, end = feature_slices[name]

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
            if seq_len == 1:
                first_occurrence = token_active
                row_weight_by_occurrence = norm_weights
            else:
                same_id = safe_indices[:, :, None] == safe_indices[:, None, :]
                row_weight_by_occurrence = jnp.sum(
                    jnp.where(same_id, norm_weights[:, None, :], 0.0), axis=-1
                )
                token_positions = jnp.arange(seq_len)
                earlier_token = (
                    token_positions[None, None, :]
                    < token_positions[None, :, None]
                )
                earlier_same_id = jnp.any(
                    same_id & token_active[:, None, :] & earlier_token, axis=-1
                )
                first_occurrence = token_active & ~earlier_same_id

            flat_indices = safe_indices.reshape(-1)
            flat_row_weights = (
                row_weight_by_occurrence * first_occurrence.astype(jnp.float32)
            ).reshape(-1, 1)

            flat_sample_weights = jnp.repeat(sample_weight, seq_len, axis=0)

            num_dense_segments = flat_indices.shape[0]
            unique_size = min(K_m, num_dense_segments)

            unique_indices, dense_inverse = jnp.unique(
                flat_indices,
                return_inverse=True,
                size=unique_size,
                fill_value=K_m,
            )
            valid_unique = unique_indices < K_m
            safe_unique_indices = jnp.where(valid_unique, unique_indices, 0)
            flat_visits = (
                first_occurrence.reshape(-1).astype(jnp.float32)
                * flat_sample_weights
            )
            dense_batch_visits = jax.ops.segment_sum(
                flat_visits, dense_inverse, num_segments=unique_size
            )

            touched = ((dense_batch_visits > 0.0) & valid_unique).astype(
                jnp.float32
            )

            v_m = self.log_variance[name]
            if init_phase:
                v_m = jax.lax.stop_gradient(v_m)
                hat_lambda_gathered = None
            else:
                g_i_m = g_i[:, start:end]
                flat_g_i = jnp.repeat(g_i_m, seq_len, axis=0) * flat_row_weights
                flat_lambdas = jnp.repeat(lambdas, seq_len, axis=0)
                flat_diag_fisher = (
                    jnp.expand_dims(flat_lambdas, axis=-1)
                    * (flat_g_i**2)
                    * jnp.expand_dims(flat_sample_weights, axis=-1)
                )
                dense_batch_fisher = jax.ops.segment_sum(
                    flat_diag_fisher,
                    dense_inverse,
                    num_segments=unique_size,
                )
                hat_lambda_gathered = (N / B) * dense_batch_fisher

            embedding_table = embedding_tables[name]
            embedding_table_gathered = embedding_table[safe_unique_indices]
            u_k_norm_sq_gathered = jnp.sum(embedding_table_gathered**2, axis=-1)

            S_m = jnp.sum(touched)
            alpha = K_m / jnp.maximum(S_m, 1.0)

            if init_phase:
                clipped_v_m = jnp.clip(v_m[0], -20.0, 20.0)
                tau_sq = jnp.exp(clipped_v_m)

                weighted_u_k_norm_sq = u_k_norm_sq_gathered * touched
                correction1 = (alpha / (2.0 * tau_sq)) * jnp.sum(
                    weighted_u_k_norm_sq
                ) + (K_m * d_m / 2.0) * clipped_v_m
                pfr_loss += correction1
                total_correction1 += correction1
                # Correction 2 is 0 in init phase
            else:
                feature_loss = stable_pfr_loss_for_feature(
                    v_m,
                    u_k_norm_sq_gathered,
                    hat_lambda_gathered,
                    K_m,
                    d_m,
                    alpha,
                    touched,
                )
                pfr_loss += feature_loss

                clipped_v_m = jnp.clip(v_m[0], -20.0, 20.0)
                tau_sq = jnp.exp(clipped_v_m)

                weighted_u_k_norm_sq = u_k_norm_sq_gathered * touched
                correction1 = (alpha / (2.0 * tau_sq)) * jnp.sum(
                    weighted_u_k_norm_sq
                ) + (K_m * d_m / 2.0) * clipped_v_m
                total_correction1 += correction1

                log_det_gathered = jnp.sum(
                    jnp.log(hat_lambda_gathered + 1.0 / tau_sq + 1e-8), axis=-1
                )
                weighted_log_det = log_det_gathered * touched
                correction2 = (alpha / 2.0) * jnp.sum(weighted_log_det)
                total_correction2 += correction2

        total_loss = head_loss + (1.0 / N) * pfr_loss
        scaled_pfr_loss = (1.0 / N) * pfr_loss
        scaled_correction1 = (1.0 / N) * total_correction1
        scaled_correction2 = (1.0 / N) * total_correction2
        return (
            total_loss,
            head_loss,
            scaled_pfr_loss,
            scaled_correction1,
            scaled_correction2,
        )


def poisson_loss(log_lambda, y, sample_weight):
    """Computes weighted Poisson loss manually guarded."""
    log_lambda = jnp.clip(log_lambda, -20.0, 20.0)
    loss = jnp.exp(log_lambda) - y * log_lambda
    weighted_loss = loss * sample_weight
    return jnp.mean(weighted_loss)


def train_step_pfr(state, inputs, masks, labels, model, init_phase=False):
    """Performs a single training step evaluating PFR computations."""

    # Avoids double pass on large Embedding tables, though MLP is repeated
    # for VJP.

    def pfr_loss_fn(params):
        def get_embeddings_fn(p):
            return model.apply(
                {"params": p},
                inputs,
                masks,
                method=model.get_embeddings,
            )

        x, transformed_inputs = get_embeddings_fn(params)

        def logits_fn_differentiable(inner_x):
            return model.apply(
                {"params": params},
                inner_x,
                method=model.logits_from_embeddings,
            )

        log_lambda = jnp.squeeze(logits_fn_differentiable(x), axis=-1)
        head_loss = poisson_loss(
            log_lambda, labels["y_i"], labels["sample_weight"]
        )

        def logits_fn_stopped(inner_x):
            return model.apply(
                {"params": jax.lax.stop_gradient(params)},
                inner_x,
                method=model.logits_from_embeddings,
            )

        if init_phase:
            g_i = None
        else:
            if FLAGS.emb_opt_using_c2:
                if FLAGS.mlp_opt_using_c2:
                    # Keep both x (embeddings) and params (MLP weights) live so gradients
                    # flow back to both. Allowing Correction 2 to impact MLP will incur
                    # higher-order derivatives for MLP parameters.
                    logits, vjp_fn = jax.vjp(logits_fn_differentiable, x)
                else:
                    # Keep x live so gradients flow back to the embeddings.
                    # MLP weights are kept fixed (stopped) in logits_fn_stopped for
                    # efficiency, avoiding expensive higher-order derivatives for the MLP
                    # weights.
                    logits, vjp_fn = jax.vjp(logits_fn_stopped, x)

                g_i = vjp_fn(jnp.ones_like(logits))[0]
            else:
                logits, vjp_fn = jax.vjp(
                    logits_fn_stopped, jax.lax.stop_gradient(x)
                )
                g_i = jax.lax.stop_gradient(vjp_fn(jnp.ones_like(logits))[0])

        start_idx = 0
        feature_slices = {}
        active_feature_names = list(model.embedding_configs.keys())
        for name in active_feature_names:
            dim = transformed_inputs[name].shape[-1]
            feature_slices[name] = (start_idx, start_idx + dim)
            start_idx += dim

        lambdas_stopped = jax.lax.stop_gradient(jnp.exp(log_lambda))

        candidate_embedding_tables = {
            name: params[f"embeddings_{name}"]["embedding"]
            for name in active_feature_names
        }

        loss_tuple = model.apply(
            {"params": params},
            g_i,
            lambdas_stopped,
            head_loss,
            inputs,
            masks,
            feature_slices,
            candidate_embedding_tables,
            labels["sample_weight"],
            init_phase,
            method=model.update_fisher_and_compute_loss,
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
        (
            total_loss,
            (_, _, _, _, _),
        ),
        grads,
    ) = grad_fn(state.params)

    state = state.apply_gradients(grads=grads)

    return state, total_loss


def main(_):
    logging.set_verbosity(logging.INFO)
    logging.info("Initializing PFR model on CPU...")

    model = PFRCTRModel(
        embedding_configs=SIM3_EMBEDDING_CONFIGS,
        multivalent_features=[],
        initial_tau=FLAGS.pfr_initial_tau,
        wn_style=FLAGS.wn_style,
        sharetau=FLAGS.sharetau,
        init_emb_with_tau=FLAGS.init_emb_with_tau,
    )

    dummy_inputs, dummy_masks, _ = next(load_and_batch_csv(FLAGS.train_csv, 2))
    key = jax.random.PRNGKey(42)
    init_variables = model.init(key, dummy_inputs, dummy_masks)
    params = init_variables["params"]

    tx = optax.adam(learning_rate=FLAGS.learning_rate)
    state = train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx
    )

    @jax.jit
    def jitted_train_step_init(s, i, m, labels):
        state, loss = train_step_pfr(s, i, m, labels, model, init_phase=True)
        return state, loss

    @jax.jit
    def jitted_train_step_standard(s, i, m, labels):
        state, loss = train_step_pfr(s, i, m, labels, model, init_phase=False)
        return state, loss

    @jax.jit
    def jitted_eval_step(s, i, m):
        variables = {"params": s.params}
        return s.apply_fn(variables, i, m)

    best_nll = float("inf")
    best_rmse = float("inf")
    best_nll_epoch = 0
    best_rmse_epoch = 0

    total_train_start = time.perf_counter()

    batches_per_epoch = math.ceil(FLAGS.num_train_examples / FLAGS.batch_size)
    total_steps = batches_per_epoch * FLAGS.epochs
    init_phase_limit_steps = math.floor(
        total_steps * (FLAGS.init_phase_pct / 100.0)
    )
    global_step = 0

    for epoch in range(FLAGS.epochs):
        logging.info("--- Epoch %d ---", epoch + 1)
        train_batches = load_and_batch_csv(
            FLAGS.train_csv, FLAGS.batch_size, shuffle=True
        )

        for _, (inputs, masks, labels) in enumerate(train_batches):
            if global_step < init_phase_limit_steps:
                state, loss = jitted_train_step_init(
                    state, inputs, masks, labels
                )
                if global_step % 10 == 0:
                    logging.info(
                        "Init Phase Step %d: Total Loss = %f", global_step, loss
                    )
            else:
                state, loss = jitted_train_step_standard(
                    state, inputs, masks, labels
                )
                if global_step % 10 == 0:
                    logging.info(
                        "Standard Phase Step %d: Total Loss = %f",
                        global_step,
                        loss,
                    )
            global_step += 1

        test_batches = load_and_batch_csv(
            FLAGS.test_csv, FLAGS.batch_size, shuffle=False
        )
        log_lambdas_test = []
        y_test_true = []
        y_test_expected = []

        for inputs, masks, labels in test_batches:
            preds = jitted_eval_step(state, inputs, masks)
            log_lambdas_test.append(np.asarray(preds))
            y_test_true.append(np.asarray(labels["y_i"]))
            y_test_expected.append(np.asarray(labels["true_rate"]))

        all_log_lambdas = np.concatenate(log_lambdas_test)
        all_lambdas = np.exp(all_log_lambdas)
        all_true_expected = np.concatenate(y_test_expected)
        all_y = np.concatenate(y_test_true)

        poisson_nll = np.mean(all_lambdas - all_y * all_log_lambdas)
        true_log_rates = np.log(all_true_expected + 1e-8)
        rmse = np.sqrt(np.mean((all_log_lambdas - true_log_rates) ** 2))

        logging.info(
            "Validation [Epoch %d] | Poisson NLL: %.4f | True-Rate RMSE: %.4f |"
            " Epoch: %d | LR: %f",
            epoch + 1,
            poisson_nll,
            rmse,
            epoch + 1,
            FLAGS.learning_rate,
        )

        # Extract taus in epoch
        taus = {}
        if FLAGS.sharetau:
            if "shared_log_variance" in state.params:
                shared_log_var = float(state.params["shared_log_variance"][0])
                taus["shared"] = np.exp(0.5 * shared_log_var)
        else:
            for name in SIM3_EMBEDDING_CONFIGS:
                param_name = f"log_variance_{name}"
                if param_name in state.params:
                    log_var = float(state.params[param_name][0])
                    taus[name] = np.exp(0.5 * log_var)

        taus_str = ", ".join([f"{k}: {v:.4f}" for k, v in taus.items()])
        logging.info("Taus [Epoch %d]: %s", epoch + 1, taus_str)
        if poisson_nll < best_nll:
            best_nll = poisson_nll
            best_nll_epoch = epoch + 1

        if rmse < best_rmse:
            best_rmse = rmse
            best_rmse_epoch = epoch + 1

    total_train_end = time.perf_counter()
    total_time = total_train_end - total_train_start
    avg_epoch_time = total_time / FLAGS.epochs if FLAGS.epochs > 0 else 0
    logging.info("Total training time: %.2f seconds", total_time)
    logging.info("Average Epoch Run Time: %.2f seconds", avg_epoch_time)

    if FLAGS.init_phase_pct > 0:
        logging.info(
            "Tau training and Correction 2 optimizations started at global step %d"
            " (around Epoch %d, Epoch Step %d)",
            init_phase_limit_steps,
            math.floor(init_phase_limit_steps / batches_per_epoch) + 1,
            init_phase_limit_steps % batches_per_epoch,
        )

    logging.info(
        (
            "Best Validation Poisson NLL: %.4f | Best Validation RMSE: %.4f |"
            " Epoch of Best NLL: %d | Epoch of Best RMSE: %d | LR: %f |"
            " Initial Tau: %f"
        ),
        best_nll,
        best_rmse,
        best_nll_epoch,
        best_rmse_epoch,
        FLAGS.learning_rate,
        FLAGS.pfr_initial_tau,
    )


if __name__ == "__main__":
    app.run(main)
