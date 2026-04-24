import numpy as np
try:
    from .AffCouplingLayer import AffCouplingLayer
except ImportError:
    from AffCouplingLayer import AffCouplingLayer
import time
import random
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import os
import sys
from pathlib import Path

from Misc.SmileyGenerator import SmileyGenerator
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from Backends import get_backend
import gc

class InvertibleNeuralNetwork:
    def __init__(self, layers, in_out_length, internalLayers=2, internalLayerLength=8, backend="cpu", batch_size=128, vector_size=64):
        # Stack of affine coupling layers. Composition remains invertible.
        self.layers = []
        for i in range(layers):
            if i % 3 == 2:  # every 3rd layer: random permutation
                perm = np.random.permutation(in_out_length).astype(np.int32)
            else:
                perm = None  # use checkerboard (flipped alternately)
            use_flipped = (i % 2 == 1) and (i % 3 != 2)
            self.layers.append(AffCouplingLayer(in_out_length, internalLayers, internalLayerLength, backend=backend, batch_size=batch_size, coupling_layers=layers, flip_checkerboard=use_flipped, perm=perm))
        # Top-level step profiler (minibatch granularity).
        self._profiler_enabled = False
        self._profile_totals = {
            "train_forward": 0.0,
            "backpropagate": 0.0,
            "train_backward": 0.0,
            "frontpropagate": 0.0,
            "loss_compute": 0.0,
            "train_step_total": 0.0,
        }
        self._profile_steps = 0
        self.backend = get_backend(backend, batch_size=batch_size, vector_size=vector_size, internal_width=internalLayerLength*2,
                           coupling_layers=layers, 
                           internal_network_layers=internalLayers)
        self._last_loss_metrics = {
            "direction": None,
            "loss": 0.0,
            "forward_loss": None,
            "reverse_loss": None,
        }
        self._staging_inputs = np.empty((batch_size, vector_size), dtype=np.float32)
        self._staging_targets = np.empty((batch_size, vector_size), dtype=np.float32)
        self.pix_mean = None   # per-pixel mean for input normalization, shape (D,) float32
        self.pix_std  = None   # per-pixel std  for input normalization, shape (D,) float32
        self.alternate = 0  # Phase index for mixed reverse/forward training updates.
        self._direction_loss_ema = {"forward": None, "reverse": None}
        self._direction_ema_decay = 0.95
        self._initial_reverse_boost_epochs = 12
        self._direction_balance_tolerance = 1.03
        self._boosted_direction = None

    ##################
    # Profiler stuff #
    ##################
    def enable_step_profiler(self, enabled=True):
        # Turns on timing for train_forward/backprop/loss/total in train_minibatch.
        self._profiler_enabled = bool(enabled)

    def enable_internal_network_profiler(self, enabled=True):
        # Propagates profiler state into every sub-network (st1/st2 in each layer).
        state = bool(enabled)
        for layer in self.layers:
            for net in (layer.st1, layer.st2):
                if hasattr(net, "enable_internal_profiler"):
                    net.enable_internal_profiler(state)

    def reset_step_profile(self):
        for key in self._profile_totals:
            self._profile_totals[key] = 0.0
        self._profile_steps = 0

    def reset_internal_network_profile(self):
        for layer in self.layers:
            for net in (layer.st1, layer.st2):
                if hasattr(net, "reset_internal_profile"):
                    net.reset_internal_profile()

    def get_step_profile(self, reset=False):
        steps = max(1, self._profile_steps)
        report = {
            "steps": self._profile_steps,
            "avg_train_forward_s": self._profile_totals["train_forward"] / steps,
            "avg_backpropagate_s": self._profile_totals["backpropagate"] / steps,
            "avg_loss_compute_s": self._profile_totals["loss_compute"] / steps,
            "avg_step_total_s": self._profile_totals["train_step_total"] / steps,
        }
        if reset:
            self.reset_step_profile()
        return report

    def get_internal_network_profile(self, reset=False):
        # Aggregates low-level timers from all subnetworks into one report.
        aggregate_totals = {
            "backprop_delta_total": 0.0,
            "backprop_delta_device": 0.0,
            "backprop_delta_cpu": 0.0,
            "update_total": 0.0,
            "update_delta_propagation": 0.0,
            "update_gradient_build": 0.0,
            "update_weight_apply": 0.0,
        }
        aggregate_counts = {
            "backprop_calls": 0,
            "update_calls": 0,
        }

        for layer in self.layers:
            for net in (layer.st1, layer.st2):
                if hasattr(net, "get_internal_profile"):
                    prof = net.get_internal_profile(reset=False)
                    for key in aggregate_totals:
                        aggregate_totals[key] += prof["totals"].get(key, 0.0)
                    for key in aggregate_counts:
                        aggregate_counts[key] += prof["counts"].get(key, 0)

        report = {
            "counts": aggregate_counts,
            "totals": aggregate_totals,
        }
        if reset:
            self.reset_internal_network_profile()
        return report

    def get_last_loss_metrics(self):
        metrics = dict(self._last_loss_metrics)
        return metrics

    def _extract_loss_scalar(self, loss_value):
        if np.isscalar(loss_value):
            return float(loss_value)
        host_value = self.backend.to_host(loss_value)
        return float(np.asarray(host_value, dtype=np.float32).reshape(-1)[0])

    def _update_direction_loss_ema(self, direction, loss):
        loss = float(loss)
        previous = self._direction_loss_ema.get(direction)
        if previous is None or not np.isfinite(previous):
            self._direction_loss_ema[direction] = loss
            return
        decay = self._direction_ema_decay
        self._direction_loss_ema[direction] = decay * previous + (1.0 - decay) * loss

    def _get_boosted_direction(self, epoch):
        if epoch < self._initial_reverse_boost_epochs:
            return "reverse"
        forward_ema = self._direction_loss_ema["forward"]
        reverse_ema = self._direction_loss_ema["reverse"]
        if forward_ema is None or reverse_ema is None:
            return None
        if reverse_ema > forward_ema * self._direction_balance_tolerance:
            return "reverse"
        if forward_ema > reverse_ema * self._direction_balance_tolerance:
            return "forward"
        return None

    def _choose_training_direction(self, epoch):
        boosted_direction = self._get_boosted_direction(epoch)
        if boosted_direction != self._boosted_direction:
            self.alternate = 0
            self._boosted_direction = boosted_direction
        cycle_len = 3 if boosted_direction else 2
        if boosted_direction is None:
            direction = "reverse" if self.alternate == 0 else "forward"
        elif self.alternate < 2:
            direction = boosted_direction
        else:
            direction = "forward" if boosted_direction == "reverse" else "reverse"
        self.alternate = (self.alternate + 1) % cycle_len
        return direction == "reverse"

    def clamp_all_weights(self, max_val=15.0):
        """Clamp all subnetwork weights to prevent long-term explosion. Call once per epoch."""
        for layer in self.layers:
            for net in (layer.st1, layer.st2):
                if hasattr(net, "clamp_weights"):
                    net.clamp_weights(max_val)

    def reset_direction_scheduler(self):
        self.alternate = 0
        self._boosted_direction = None
        self._direction_loss_ema = {"forward": None, "reverse": None}

    def capture_parameter_state(self):
        state = []
        for layer in self.layers:
            layer_state = {}
            for net_name in ("st1", "st2"):
                net = getattr(layer, net_name)
                net_state = {
                    "weights": [net._get_weight_host(i).copy() for i in range(net.layers)],
                }
                if net.bias:
                    net_state["biases"] = [net._get_bias_host(i).copy() for i in range(net.layers)]
                layer_state[net_name] = net_state
            state.append(layer_state)
        return state

    def restore_parameter_state(self, state):
        for layer, layer_state in zip(self.layers, state):
            for net_name in ("st1", "st2"):
                net = getattr(layer, net_name)
                net_state = layer_state[net_name]
                for i, weights in enumerate(net_state["weights"]):
                    net.weightsByLayer[i] = np.array(weights, dtype=np.float32, copy=True)
                    net.weight_velocities[i] = np.zeros_like(net.weightsByLayer[i], dtype=np.float32)
                    if net.bias:
                        net.biasByLayer[i] = np.array(net_state["biases"][i], dtype=np.float32, copy=True)
                        net.bias_velocities[i] = np.zeros_like(net.biasByLayer[i], dtype=np.float32)
                if net._device_resident_weights:
                    net._upload_all_params_to_device()
                else:
                    net._invalidate_device_cache()
        self.reset_direction_scheduler()
    
    def save_weights(self, filepath):
        """Save all network weights, biases, and permutations to a .npz file."""
        data = {}
        for li, layer in enumerate(self.layers):
            # Save permutation (pull from device if needed)
            perm = layer.backend.to_host(layer.perm) if layer._use_device else np.asarray(layer.perm)
            data[f"layer{li}_perm"] = perm

            for net_name, net in [("st1", layer.st1), ("st2", layer.st2)]:
                for wi in range(net.layers):
                    data[f"layer{li}_{net_name}_w{wi}"] = net._get_weight_host(wi)
                    if net.bias:
                        data[f"layer{li}_{net_name}_b{wi}"] = net._get_bias_host(wi)
        np.savez(filepath, **data)

    def load_weights(self, filepath):
        """Load weights, biases, and permutations from a .npz file created by save_weights."""
        data = np.load(filepath)
        for li, layer in enumerate(self.layers):
            # Restore permutation
            perm = data[f"layer{li}_perm"]
            inv_perm = np.argsort(perm)
            if layer._use_device:
                layer.perm = layer.backend.to_device(perm, dtype=np.int32)
                layer._inv_perm = layer.backend.to_device(inv_perm, dtype=np.int32)
            else:
                layer.perm = perm
                layer._inv_perm = inv_perm

            for net_name, net in [("st1", layer.st1), ("st2", layer.st2)]:
                for wi in range(net.layers):
                    w = data[f"layer{li}_{net_name}_w{wi}"].astype(np.float32)
                    b = data[f"layer{li}_{net_name}_b{wi}"].astype(np.float32) if net.bias else None

                    net.weightsByLayer[wi] = w
                    if b is not None:
                        net.biasByLayer[wi] = np.atleast_2d(b)

                    # Reset velocities
                    net.weight_velocities[wi] = np.zeros_like(w)
                    if net.bias:
                        net.bias_velocities[wi] = np.zeros_like(net.biasByLayer[wi])

                # Re-upload to device if using GPU
                if net._device_resident_weights:
                    net._upload_all_params_to_device()
                net._invalidate_device_cache()
            
    ###########
    # Compute #
    ###########
    def forward(self, x):
        # Forward composition through all coupling layers.
        for layer in self.layers:
            x = layer.forward(x)
        return self.backend.to_host(x)

    def forward_batch(self, inputs, batch_size=None):
        """Run inference on a batch of inputs.

        Args:
            inputs: Array-like input of shape (B, D) or (D,)
            batch_size: Optional chunk size for large batches

        Returns:
            Output array of shape (B, D)
        """
        inputs_arr = np.asarray(inputs)
        if inputs_arr.ndim == 1:
            inputs_arr = np.atleast_2d(inputs_arr)

        total = inputs_arr.shape[0]
        if batch_size is None or batch_size <= 0 or batch_size >= total:
            return self.forward(inputs_arr)

        outputs = []
        for start in range(0, total, batch_size):
            batch = inputs_arr[start:start + batch_size]
            outputs.append(self.forward(batch))

        return np.vstack(outputs)

    def backward(self, y):
        # Inverse composition through coupling layers in reverse order.
        for layer in reversed(self.layers):
            y = layer.backward(y)
        return self.backend.to_host(y)

    def backward_batch(self, outputs, batch_size=None):
        """Run inverse inference on a batch of outputs.

        Args:
            outputs: Array-like output of shape (B, D) or (D,)
            batch_size: Optional chunk size for large batches

        Returns:
            Reconstructed input array of shape (B, D)
        """
        outputs_arr = np.asarray(outputs)
        if outputs_arr.ndim == 1:
            outputs_arr = np.atleast_2d(outputs_arr)

        total = outputs_arr.shape[0]
        if batch_size is None or batch_size <= 0 or batch_size >= total:
            return self.backward(outputs_arr)

        reconstructions = []
        for start in range(0, total, batch_size):
            batch = outputs_arr[start:start + batch_size]
            reconstructions.append(self.backward(batch))

        return np.vstack(reconstructions)
    
    def train_forward(self, x):
        # Training forward path (stores per-layer caches).
        for layer in self.layers:
            x = layer.train_forward(x)
        return x

    def train_backward(self, y):
        for layer in reversed(self.layers):
            y = layer.train_backward(y)
        return y
    
    def backpropagate(self, diffs, learningRate, momentum=0.9):
        # Propagate dL/dOutput backward across coupling stack.
        
        current_diffs = self.backend.to_device(diffs)
        for layer in reversed(self.layers):
            input_diffs = layer.backpropagate(layer.forward_input, learningRate, current_diffs, momentum=momentum, weightDecay=0.01)
            current_diffs = input_diffs

    def frontpropagate(self, diffs, learningRate, momentum=0.9):
        # Propagate dL/dReconstruction forward across coupling stack.
        # Layers are iterated in forward order (0..N) since train_backward
        # processed them in reverse order (N..0).
        current_diffs = self.backend.to_device(diffs)
        for layer in self.layers:
            output_diffs = layer.frontpropagate(layer.backward_input, learningRate, current_diffs, momentum=momentum, weightDecay=0.01)
            current_diffs = output_diffs
    
    def train_sample(self, inputs, targets, learningRate):
        """Train on a single sample and return loss"""
        calc_outputs = self.train_forward(inputs)
        diffs = targets - calc_outputs
        self.backpropagate(diffs, learningRate)
        return 0.5 * np.sum(diffs ** 2)

    def train_minibatch(self, batch_samples, learningRate, momentum=0.9):
        """Train over one minibatch with a single matrix forward/backward pass."""
        if not batch_samples:
            return 0.0
        n = len(batch_samples)
        for i, (inp, tgt) in enumerate(batch_samples):
            self._staging_inputs[i] = inp
            self._staging_targets[i] = tgt
        return self._run_staged_minibatch(n, learningRate, epoch=0, momentum=momentum)

    def train_minibatch_raw(self, raw_data, batch_indices, learningRate, epoch, momentum=0.9):
        """Train over one minibatch reading directly from a raw uint8 dataset array.
        Uses vectorized fancy indexing to fill staging buffers — no per-sample Python loop."""
        n = len(batch_indices)
        if n == 0:
            return 0.0
        idx = np.asarray(batch_indices)
        if self.pix_mean is not None:
            self._staging_inputs[:n]  = (raw_data[idx, 0] * np.float32(1.0 / 255.0) - self.pix_mean) / self.pix_std
            self._staging_targets[:n] = (raw_data[idx, 1] * np.float32(1.0 / 255.0) - self.pix_mean) / self.pix_std
        else:
            self._staging_inputs[:n]  = raw_data[idx, 0] * np.float32(1.0 / 255.0)
            self._staging_targets[:n] = raw_data[idx, 1] * np.float32(1.0 / 255.0)
        return self._run_staged_minibatch(n, learningRate, epoch=epoch, momentum=momentum)

    def _run_staged_minibatch(self, n, learningRate, epoch=0, momentum=0.9):
        """Core minibatch training; assumes _staging_inputs/targets[:n] are pre-filled."""
        batch_inputs  = self.backend.to_device(self._staging_inputs[:n])
        batch_targets = self.backend.to_device(self._staging_targets[:n])
        n_samples = n

        # Reverse pass: f⁻¹(frown) should ≈ smile.
        # Reverse-heavy updates help early while the inverse map lags, but
        # keeping that 2:1 bias forever makes the reverse objective dominate
        # once the losses are close and can destabilize the forward map.
        if self._choose_training_direction(epoch):
            rev_outputs = self.train_backward(batch_targets)   # caches go to layer.backward_input etc.
            rev_diffs_dev, rev_partial_sums_dev = self.backend.subtract_divide_loss(
                batch_inputs, rev_outputs, float(n_samples)
            )
            self.frontpropagate(rev_diffs_dev, learningRate, momentum=momentum)
            rev_loss = self._extract_loss_scalar(rev_partial_sums_dev)
            self._update_direction_loss_ema("reverse", rev_loss)
            self._last_loss_metrics = {
                "direction": "reverse",
                "loss": rev_loss,
                "forward_loss": None,
                "reverse_loss": rev_loss,
            }
            return rev_loss

        calc_outputs = self.train_forward(batch_inputs)  # stays on device
        # weighted_loss_diffs uses clamp(inp-tgt, 0, 1) thresholds that only work in [0,1]
        # pixel space. With per-pixel z-score normalization the thresholds are meaningless and
        # the weighting amplifies gradients ~8x unintentionally. Since normalization already
        # removes background-pixel dominance, plain MSE is correct here.
        train_diffs_dev, partial_sums_dev = self.backend.subtract_divide_loss(
            batch_targets, calc_outputs, float(n_samples)
        )
        self.backpropagate(train_diffs_dev, learningRate, momentum=momentum)
        loss = self._extract_loss_scalar(partial_sums_dev)
        self._update_direction_loss_ema("forward", loss)
        self._last_loss_metrics = {
            "direction": "forward",
            "loss": loss,
            "forward_loss": loss,
            "reverse_loss": None,
        }
        return loss
    
    def train_batch_parallel(self, batch_indices, testSet, learningRate, num_workers=None):
        """Train on a batch of samples in parallel using thread pool.
        
        Args:
            batch_indices: List of indices to train on
            testSet: The dataset
            learningRate: Learning rate
            num_workers: Number of parallel workers (defaults to CPU count)
        
        Returns:
            Average loss for the batch
        """
        if num_workers is None:
            num_workers = os.cpu_count()
        
        # For small batches, sequential is faster due to overhead
        if len(batch_indices) < num_workers * 2:
            total_loss = 0
            for idx in batch_indices:
                total_loss += self.train_sample(testSet[idx][0], testSet[idx][1], learningRate)
            return total_loss / len(batch_indices)
        
        # Parallel processing for larger batches
        def train_one(idx):
            inputs = testSet[idx][0]
            outputs = testSet[idx][1]
            return self.train_sample(inputs, outputs, learningRate)
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            losses = list(executor.map(train_one, batch_indices))
        
        return sum(losses) / len(losses)
    
    def get_param_stats(self):
        max_w = 0.0
        max_b = 0.0
        for layer in self.layers:
            for net in (layer.st1, layer.st2):
                for i in range(net.layers):
                    w = net._get_weight_host(i)
                    max_w = max(max_w, float(np.max(np.abs(w))))
                    if net.bias:
                        b = net._get_bias_host(i)
                        max_b = max(max_b, float(np.max(np.abs(b))))
        return max_w, max_b


if __name__ == "__main__":
    DATASET_PATH = Path(__file__).resolve().parent.parent / "smiley_dataset.npy"
    DATASET_SIZE = 100_000

    if DATASET_PATH.exists():
        print(f"Loading dataset from {DATASET_PATH} ...")
        raw_data = np.load(str(DATASET_PATH))          # (N, 2, 784) uint8
        print(f"Loaded {len(raw_data):,} pairs.")
    else:
        print(f"Generating {DATASET_SIZE:,} smiley pairs (this may take a minute)...")
        gen = SmileyGenerator(size=28, seed=42)
        pairs = gen.generate_dataset_pairs(DATASET_SIZE, eyes_ratio=0.0)
        raw_data = pairs.reshape(DATASET_SIZE, 2, -1)  # (N, 2, 784) uint8
        np.save(str(DATASET_PATH), raw_data)
        print(f"Saved dataset to {DATASET_PATH}")

    dataset_size = len(raw_data)

    # Compute per-pixel normalization statistics over the full dataset (both channels).
    # Normalizing to zero-mean unit-variance per pixel conditions the coupling subnetworks
    # on inputs that are no longer heavily zero-skewed, improving gradient flow.
    print("Computing per-pixel normalization statistics...")
    all_pixels = raw_data.reshape(-1, raw_data.shape[-1]).astype(np.float32) * (1.0 / 255.0)
    pix_mean = all_pixels.mean(axis=0).astype(np.float32)           # (D,)
    pix_std  = all_pixels.std(axis=0).clip(.05).astype(np.float32) # (D,) clipped to avoid /0
    del all_pixels
    print(f"  pixel mean range: [{pix_mean.min():.4f}, {pix_mean.max():.4f}]")
    print(f"  pixel std  range: [{pix_std.min():.4f},  {pix_std.max():.4f}]")

    backend = "opencl"  # "cpu" or "opencl"
    vector_size = 28*28
    internal_network_layers = 2
    coupling_layers = 12
    hidden_width = 512
    assert vector_size % 2 == 0, "vector_size must be even for coupling split"

    sanity_size = 32
    sanity_indices = list(range(sanity_size))

    initial_lr = 0.00002
    min_lr = 0.000002
    output_error_track = []
    startTime = time.perf_counter()
    learnRate = 0

    num_epochs = 1280
    samples_per_epoch = DATASET_SIZE  # random chunk drawn from the full dataset each epoch
    batch_size = min(256, samples_per_epoch)
    updates_per_epoch = (samples_per_epoch + batch_size - 1) // batch_size
    total_iterations = num_epochs * updates_per_epoch
    global_iteration = 0

    cosine_epochs = min(max(1, num_epochs // 2), 160)  # reach the floor before the late-drift regime
    cosine_iterations = cosine_epochs * updates_per_epoch

    # smiley_weights has the following parameters:
    # 28*28 vectorsize
    # 2 internal layers
    # 512 hidden width
    # 12 coupling layers
    # batch size 256

    inn = InvertibleNeuralNetwork(coupling_layers, vector_size, internal_network_layers, hidden_width, backend=backend, batch_size=batch_size, vector_size=vector_size)
    inn.pix_mean = pix_mean
    inn.pix_std  = pix_std

    def run_training_loop(inn):
        """Run the training loop on inn, using the dataset and hyperparams from the enclosing scope."""
        import math
        import msvcrt

        inn.enable_step_profiler(False)
        inn.enable_internal_network_profiler(False)

        def reset_velocities(inn):
            for layer in inn.layers:
                for net in [layer.st1, layer.st2]:
                    for i in range(net.layers):
                        v = net.weight_velocities[i]
                        if net.backend.is_device_array(v):
                            net.weight_velocities[i] = net.backend.to_device(
                                np.zeros_like(net.backend.to_host(v)), dtype=np.float32
                            )
                        elif isinstance(v, np.ndarray):
                            net.weight_velocities[i] = np.zeros_like(v)
                        if net.bias:
                            bv = net.bias_velocities[i]
                            if net.backend.is_device_array(bv):
                                net.bias_velocities[i] = net.backend.to_device(
                                    np.zeros_like(net.backend.to_host(bv)), dtype=np.float32
                                )
                            else:
                                net.bias_velocities[i] = np.zeros_like(bv)

        reset_velocities(inn)

        loop_error_track = []
        loop_forward_track = []
        loop_reverse_track = []
        loop_start_time = time.perf_counter()
        global_iteration = 0
        learnRate = 0
        best_loss = float('inf')
        best_state = inn.capture_parameter_state()
        best_epoch = -1
        lr_scale = 1.0
        late_drift_ratio = 1.03
        late_drift_patience = 4
        late_drift_streak = 0
        late_drift_lr_gate = 4.0 * min_lr
        epoch_error = float('nan')

        for epoch in range(num_epochs):
            inn.reset_step_profile()
            inn.reset_internal_network_profile()
            indices = random.sample(range(dataset_size), samples_per_epoch)
            random.shuffle(indices)

            max_momentum = 0.65
            min_momentum = 0.45
            momentum_decay_steps = 30 * updates_per_epoch

            epoch_error = 0
            epoch_forward_total = 0.0
            epoch_reverse_total = 0.0
            epoch_forward_samples = 0
            epoch_reverse_samples = 0
            for start in range(0, len(indices), batch_size):
                batch_indices = indices[start:start + batch_size]
                iteration = global_iteration

                warmup_steps = 1 * (dataset_size // batch_size)
                if iteration < warmup_steps:
                    base_lr = initial_lr * (iteration / warmup_steps)
                elif iteration < cosine_iterations:
                    progress = (iteration - warmup_steps) / (cosine_iterations - warmup_steps)
                    base_lr = min_lr + 0.5 * (initial_lr - min_lr) * (1 + math.cos(math.pi * progress))
                else:
                    base_lr = min_lr
                learnRate = base_lr * lr_scale

                if iteration < momentum_decay_steps:
                    mom_progress = iteration / momentum_decay_steps
                    current_momentum = min_momentum + 0.5 * (max_momentum - min_momentum) * (1.0 + math.cos(math.pi * mom_progress))
                else:
                    current_momentum = min_momentum

                batch_loss = inn.train_minibatch_raw(raw_data, batch_indices, learnRate, epoch, momentum=current_momentum)
                batch_metrics = inn.get_last_loss_metrics()
                if not np.isfinite(batch_loss):
                    direction = batch_metrics.get("direction") or "unknown"
                    print(f"Non-finite batch loss at epoch {epoch+1}, step {global_iteration+1}. "
                        f"direction={direction}, LR={learnRate:.6f}, momentum={current_momentum:.4f}")
                    epoch_error = float("nan")
                    break
                batch_weight = len(batch_indices)
                epoch_error += batch_loss * batch_weight
                if batch_metrics["direction"] == "forward":
                    epoch_forward_total += batch_loss * batch_weight
                    epoch_forward_samples += batch_weight
                elif batch_metrics["direction"] == "reverse":
                    epoch_reverse_total += batch_loss * batch_weight
                    epoch_reverse_samples += batch_weight
                global_iteration += 1

            if not np.isfinite(epoch_error):
                break

            try:
                if msvcrt.kbhit() and msvcrt.getch().decode().lower() == 'c':
                    print("Training interrupted by user")
                    break
            except (ImportError, Exception):
                pass

            epoch_error /= samples_per_epoch
            loop_error_track.append(epoch_error)
            epoch_forward_error = epoch_forward_total / max(1, epoch_forward_samples)
            epoch_reverse_error = epoch_reverse_total / max(1, epoch_reverse_samples)
            loop_forward_track.append(epoch_forward_error)
            loop_reverse_track.append(epoch_reverse_error)

            if epoch_error < best_loss:
                best_loss = epoch_error
                best_state = inn.capture_parameter_state()
                best_epoch = epoch
                late_drift_streak = 0
            elif epoch_error > best_loss * 1.10 and epoch >= 5:
                inn.restore_parameter_state(best_state)
                reset_velocities(inn)
                lr_scale = max(0.35, lr_scale * 0.7)
                late_drift_streak = 0
                print(
                    f"  >> Restored best params and reduced LR scale to {lr_scale:.3f} "
                    f"(loss {epoch_error:.6f} > 1.10x best {best_loss:.6f})"
                )
            else:
                late_drift_active = (
                    best_epoch >= 0
                    and epoch >= max(40, best_epoch + 8)
                    and learnRate <= late_drift_lr_gate
                    and epoch_error > best_loss * late_drift_ratio
                )
                late_drift_streak = late_drift_streak + 1 if late_drift_active else 0
                if late_drift_streak >= late_drift_patience:
                    inn.restore_parameter_state(best_state)
                    reset_velocities(inn)
                    lr_scale = max(0.20, lr_scale * 0.6)
                    late_drift_streak = 0
                    print(
                        f"  >> Late-stage drift restore, LR scale now {lr_scale:.3f} "
                        f"(loss {epoch_error:.6f} > {late_drift_ratio:.2f}x best {best_loss:.6f})"
                    )

            inn.clamp_all_weights(max_val=15.0)

            max_w, max_b = inn.get_param_stats()

            if (epoch + 1) % 1 == 0:
                step_profile = inn.get_step_profile(reset=False)
                internal_profile = inn.get_internal_network_profile(reset=False)
                steps = max(1, step_profile["steps"])

                internal_per_step_backprop_delta = internal_profile["totals"]["backprop_delta_total"] / steps
                internal_per_step_update_total = internal_profile["totals"]["update_total"] / steps
                internal_per_step_update_delta = internal_profile["totals"]["update_delta_propagation"] / steps
                internal_per_step_update_grad = internal_profile["totals"]["update_gradient_build"] / steps
                internal_per_step_update_apply = internal_profile["totals"]["update_weight_apply"] / steps

                print(
                    f"Epoch {epoch+1}, Combined: {epoch_error:.6f}, Forward: {epoch_forward_error:.6f}, "
                    f"Reverse: {epoch_reverse_error:.6f}, LR: {learnRate:.6f}, "
                    f"max_w={max_w:.2f}, max_b={max_b:.2f}"
                )

            if (epoch + 1) % 1000 == 0:
                reset_velocities(inn)

        print(f"Time: {time.perf_counter() - loop_start_time:.2f}s")
        plt.plot(loop_error_track, label='combined')
        plt.plot(loop_forward_track, label='forward')
        plt.plot(loop_reverse_track, label='reverse')
        plt.yscale('log')
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.title('Training Error')
        plt.legend()
        plt.show()

        return loop_error_track, epoch_error

    if (True):  # Set to True to load pretrained weights if available
        try:
            inn.load_weights("inn_smiley_frowny_weights.npz")
            print("Loaded pretrained weights from inn_smiley_frowny_weights.npz")

        except FileNotFoundError:
            print("Pretrained weights not found, starting with random initialization.")

        def unnorm(v, shape=(28, 28)):
            """Invert per-pixel normalization and clip to [0, 1]."""
            return (v.reshape(shape) * pix_std.reshape(shape) + pix_mean.reshape(shape)).clip(0.0, 1.0)

        # Pick a random set of samples to visualize
        n_display = 8
        display_indices = random.sample(range(dataset_size), n_display)

        # --- Forward pass: input -> INN output ---
        fig, axes = plt.subplots(n_display, 3, figsize=(9, 3 * n_display))
        fig.suptitle('Forward: Input → INN Output → Reconstruction', fontsize=14)
        for col, label in enumerate(['Input (smile)', 'INN Forward Output', 'INN Reconstruction']):
            axes[0, col].set_title(label)
        for row, idx in enumerate(display_indices):
            inp_raw = raw_data[idx, 0].astype(np.float32) * (1.0 / 255.0)
            inp_vec = np.atleast_2d((inp_raw - pix_mean) / pix_std)
            fwd_out = inn.forward(inp_vec)
            recon   = inn.backward(fwd_out)
            axes[row, 0].imshow(unnorm(inp_vec),  cmap='gray'); axes[row, 0].axis('off')
            axes[row, 1].imshow(unnorm(fwd_out),  cmap='gray'); axes[row, 1].axis('off')
            axes[row, 2].imshow(unnorm(recon),    cmap='gray'); axes[row, 2].axis('off')
        plt.tight_layout()
        plt.show()

        # --- Backward pass: target -> INN inverse output ---
        fig2, axes2 = plt.subplots(n_display, 3, figsize=(9, 3 * n_display))
        fig2.suptitle('Backward: Target → INN Inverse → Forward Re-check', fontsize=14)
        for col, label in enumerate(['Target (no-smile)', 'INN Backward Output', 'INN Re-forward']):
            axes2[0, col].set_title(label)
        for row, idx in enumerate(display_indices):
            tgt_raw = raw_data[idx, 1].astype(np.float32) * (1.0 / 255.0)
            tgt_vec = np.atleast_2d((tgt_raw - pix_mean) / pix_std)
            bwd_out = inn.backward(tgt_vec)
            refwd   = inn.forward(bwd_out)
            axes2[row, 0].imshow(unnorm(tgt_vec), cmap='gray'); axes2[row, 0].axis('off')
            axes2[row, 1].imshow(unnorm(bwd_out), cmap='gray'); axes2[row, 1].axis('off')
            axes2[row, 2].imshow(unnorm(refwd),   cmap='gray'); axes2[row, 2].axis('off')
        plt.tight_layout()
        plt.show()

        # Continue training the loaded model
    else:
        output_error_track, epoch_error = run_training_loop(inn)

    

    # Normalize input with the same per-pixel stats used during training.
    image_raw = raw_data[0, 0].astype(np.float32) * (1.0 / 255.0)
    image_vector = np.atleast_2d((image_raw - pix_mean) / pix_std)

    target_raw = raw_data[0, 1].astype(np.float32) * (1.0 / 255.0)
    target_vector = np.atleast_2d((target_raw - pix_mean) / pix_std)

    # Pass through INN forward
    inn_output = inn.forward(image_vector)

    # Pass through INN backward (reconstruction)
    reconstructed_vector = inn.backward(inn_output)

    inn_input = inn.backward(target_vector)

    reconstructed_output = inn.forward(inn_input)

    def unnorm(v):
        """Invert per-pixel normalization and clip to [0, 1]."""
        return (v.reshape(28, 28) * pix_std.reshape(28, 28) + pix_mean.reshape(28, 28)).clip(0.0, 1.0)

    # Reshape vectors back to 28x28 for display (unnormalized pixel space)
    original_img = unnorm(image_vector)
    output_img   = unnorm(inn_output)
    reconstructed_img = unnorm(reconstructed_vector)

    original_target = unnorm(target_vector)
    input_img   = unnorm(inn_input)
    reconstructed_output_img = unnorm(reconstructed_output)
    
    # Plot original, INN output, and reconstruction side by side
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    axes[0].imshow(original_img, cmap='gray')
    axes[0].set_title('Original (smile)')
    axes[0].axis('off')
    
    axes[1].imshow(output_img, cmap='gray')
    axes[1].set_title('INN Forward Output')
    axes[1].axis('off')
    
    axes[2].imshow(reconstructed_img, cmap='gray')
    axes[2].set_title('INN Reconstruction (backward)')
    axes[2].axis('off')
    
    plt.suptitle('MNIST Image through Invertible Neural Network')
    plt.tight_layout()
    plt.show()

    error_map = np.abs(output_img - target_raw.reshape(28, 28))
    plt.imshow(error_map, cmap='hot')
    plt.colorbar()
    plt.show()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(original_target, cmap='gray')
    axes[0].set_title('Original (frown)')
    axes[0].axis('off')
    
    axes[1].imshow(input_img, cmap='gray')
    axes[1].set_title('INN Backward Output')
    axes[1].axis('off')
    
    axes[2].imshow(reconstructed_output_img, cmap='gray')
    axes[2].set_title('INN Re-forward Check')
    axes[2].axis('off')
    
    plt.suptitle('MNIST Image through Invertible Neural Network')
    plt.tight_layout()
    plt.show()

    error_map = np.abs(input_img - image_raw.reshape(28, 28))
    plt.imshow(error_map, cmap='hot')
    plt.colorbar()
    plt.show()
    
    # Print reconstruction error
    recon_error = np.mean((image_vector - reconstructed_vector) ** 2)
    print(f"Reconstruction MSE: {recon_error:.8f}")

    # inn.save_weights("inn_smiley_frowny_weights.npz")
