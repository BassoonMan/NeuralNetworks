import numpy as np
from AffCouplingLayer import AffCouplingLayer
import time
import random
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import os

class InvertibleNeuralNetwork:
    def __init__(self, layers, in_out_length, internalLayers=2, internalLayerLength=8, backend="cpu"):
        # Stack of affine coupling layers. Composition remains invertible.
        self.layers = []
        for i in range(layers):
            self.layers.append(AffCouplingLayer(in_out_length, internalLayers, internalLayerLength, backend=backend))
        # Top-level step profiler (minibatch granularity). This I think just keeps track of total timings.
        self._profiler_enabled = False
        self._profile_totals = {
            "train_forward": 0.0,
            "backpropagate": 0.0,
            "loss_compute": 0.0,
            "train_step_total": 0.0,
        }
        self._profile_steps = 0

    ##################
    # Profiler stuff #
    ##################
    def enable_step_profiler(self, enabled=True):
        # Turns on timing for train_forward/backprop/loss/total in train_minibatch.
        self._profiler_enabled = bool(enabled)

    def enable_internal_network_profiler(self, enabled=True):
        # Propagates profiler state into every sub-network (s1/s2/t1/t2 in each layer).
        state = bool(enabled)
        for layer in self.layers:
            for net in (layer.s1, layer.s2, layer.t1, layer.t2):
                if hasattr(net, "enable_internal_profiler"):
                    net.enable_internal_profiler(state)

    def reset_step_profile(self):
        for key in self._profile_totals:
            self._profile_totals[key] = 0.0
        self._profile_steps = 0

    def reset_internal_network_profile(self):
        for layer in self.layers:
            for net in (layer.s1, layer.s2, layer.t1, layer.t2):
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
            for net in (layer.s1, layer.s2, layer.t1, layer.t2):
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
            
    ###########
    # Compute #
    ###########
    def forward(self, x):
        # Forward composition through all coupling layers.
        for layer in self.layers:
            x = layer.forward(x)
        return x

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
        return y

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
        current_diffs = diffs
        for layer in reversed(self.layers):
            input_diffs = layer.backpropagate(layer.forward_input, learningRate, current_diffs, momentum=momentum)
            current_diffs = input_diffs

    def frontpropagate(self, output, diffs, learningRate):
        for layer in self.layers:
            layer.frontpropagate(output, learningRate, diffs)
    
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

        batch_inputs = np.vstack([sample[0] for sample in batch_samples])
        batch_targets = np.vstack([sample[1] for sample in batch_samples])

        # Timing checkpoints for top-level step profiler.
        step_t0 = time.perf_counter() if self._profiler_enabled else 0.0

        t0 = time.perf_counter() if self._profiler_enabled else 0.0
        calc_outputs = self.train_forward(batch_inputs)
        t1 = time.perf_counter() if self._profiler_enabled else 0.0

        diffs = (batch_targets - calc_outputs) / len(batch_samples)

        t2 = time.perf_counter() if self._profiler_enabled else 0.0
        self.backpropagate(diffs, learningRate, momentum=momentum)
        t3 = time.perf_counter() if self._profiler_enabled else 0.0

        raw_diffs = batch_targets - calc_outputs
        loss = 0.5 * np.sum(raw_diffs ** 2) / len(batch_samples)

        if self._profiler_enabled:
            t4 = time.perf_counter()
            self._profile_totals["train_forward"] += (t1 - t0)
            self._profile_totals["backpropagate"] += (t3 - t2)
            self._profile_totals["loss_compute"] += (t4 - t3)
            self._profile_totals["train_step_total"] += (t4 - step_t0)
            self._profile_steps += 1

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


if __name__ == "__main__":
    testSet3 = []
    backend = "opencl"
    vector_size = 64
    dataset_size = 2048
    internal_layers = 2
    hidden_width = 64
    assert vector_size % 2 == 0, "vector_size must be even for coupling split"

    rng = np.random.default_rng(42)
    for _ in range(dataset_size):
        inp = (rng.random(vector_size) > 0.5).astype(np.float32)
        # Deterministic bijection-like target pattern for stress/perf testing.
        out = 0.1 + 0.8 * np.roll(inp, 1)
        testSet3.append([np.atleast_2d(inp), np.atleast_2d(out)])

    inn = InvertibleNeuralNetwork(5, vector_size, internal_layers, hidden_width, backend=backend)
    inn.enable_step_profiler(True)
    inn.enable_internal_network_profiler(True)

    test_input = testSet3[15][0]
    test_target = testSet3[15][1]
    # from gradient_check import numerical_gradient_check
    # numerical_gradient_check(inn, test_input, test_target)

    # num = 100000
    initial_lr = 0.002
    min_lr = 0.00005
    output_error_track = []
    startTime = time.perf_counter()
    learnRate = 0

    num_epochs = 20
    samples_per_epoch = len(testSet3)
    batch_size = min(128, samples_per_epoch)
    updates_per_epoch = (samples_per_epoch + batch_size - 1) // batch_size
    total_iterations = num_epochs * updates_per_epoch
    global_iteration = 0

    def reset_velocities(inn):
        for layer in inn.layers:
            for net in [layer.s1, layer.s2, layer.t1, layer.t2]:
                for i in range(net.layers):
                    if isinstance(net.weight_velocities[i], np.ndarray):
                        net.weight_velocities[i] = np.zeros_like(net.weight_velocities[i])
                    if net.bias:
                        net.bias_velocities[i] = np.zeros_like(net.bias_velocities[i])

    reset_velocities(inn)

    import math
    
    # Main training loop:
    #   - shuffle data
    #   - iterate minibatches with cosine LR + momentum schedule
    #   - accumulate epoch loss
    #   - print top-level and internal profiler summaries
    for epoch in range(num_epochs):
        inn.reset_step_profile()
        inn.reset_internal_network_profile()
        # Shuffle order each epoch
        indices = list(range(len(testSet3)))
        random.shuffle(indices)
        
        epoch_error = 0
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start:start + batch_size]
            batch_samples = [testSet3[idx] for idx in batch_indices]

            iteration = global_iteration
            learnRate = min_lr + 0.5 * (initial_lr - min_lr) * (
                1 + math.cos(math.pi * iteration / total_iterations))
            
            current_momentum = min(0.9, 0.5 + 0.4 * (iteration / total_iterations))

            batch_loss = inn.train_minibatch(batch_samples, learnRate, momentum=current_momentum)
            if not np.isfinite(batch_loss):
                print(f"Non-finite batch loss at epoch {epoch+1}, step {global_iteration+1}. "
                      f"LR={learnRate:.6f}, momentum={current_momentum:.4f}")
                epoch_error = float("nan")
                break
            epoch_error += batch_loss * len(batch_indices)
            global_iteration += 1

        if not np.isfinite(epoch_error):
            break
        
        epoch_error /= samples_per_epoch
        output_error_track.append(epoch_error)
        
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
                f"Epoch {epoch+1}, Error: {epoch_error:.6f}, LR: {learnRate:.6f}, "
                f"step_avg={step_profile['avg_step_total_s']:.6f}s "
                f"(fwd={step_profile['avg_train_forward_s']:.6f}s, "
                f"bwd={step_profile['avg_backpropagate_s']:.6f}s, "
                f"loss={step_profile['avg_loss_compute_s']:.6f}s), "
                f"internal/step: delta={internal_per_step_backprop_delta:.6f}s, "
                f"update={internal_per_step_update_total:.6f}s "
                f"(prop={internal_per_step_update_delta:.6f}s, "
                f"grad={internal_per_step_update_grad:.6f}s, "
                f"apply={internal_per_step_update_apply:.6f}s)"
            )

        if (epoch + 1) % 1000 == 0:
            reset_velocities(inn)
        
        if epoch_error < 0.000001:
            print(f"Converged at epoch {epoch+1}")
            break

    print(f"Time: {time.perf_counter() - startTime:.2f}s")
    plt.plot(output_error_track)
    plt.yscale('log')  # log scale shows convergence progress better
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title('Training Error')
    plt.show()
    x1 = testSet3[0][0]
    y1 = inn.forward(x1)
    x1_reconstructed = inn.backward(y1)
    print("Input:", x1)
    print("Output:", y1)
    print("Reconstructed Input:", x1_reconstructed)

    y2 = testSet3[0][1]
    x2 = inn.backward(y2)
    y2_reconstructed = inn.forward(x2)
    print("Input(backwards):", y2)
    print("Output(backwards):", x2)
    print("Reconstructed Input(backwards):", y2_reconstructed)

