import numpy as np
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
from Misc.MNIST import MnistDataloader as Mnist
from Backends import get_backend
import gc
gc.disable()  # disable automatic collection

class InvertibleNeuralNetwork:
    def __init__(self, layers, in_out_length, internalLayers=2, internalLayerLength=8, backend="cpu", batch_size=128, vector_size=64):
        # Stack of affine coupling layers. Composition remains invertible.
        self.layers = []
        for i in range(layers):
            self.layers.append(AffCouplingLayer(in_out_length, internalLayers, internalLayerLength, backend=backend, batch_size=batch_size))
        # Top-level step profiler (minibatch granularity). This I think just keeps track of total timings.
        self._profiler_enabled = False
        self._profile_totals = {
            "train_forward": 0.0,
            "backpropagate": 0.0,
            "loss_compute": 0.0,
            "train_step_total": 0.0,
        }
        self._profile_steps = 0
        # print(batch_size, vector_size, internalLayerLength)
        self.backend = get_backend(backend, batch_size=batch_size, vector_size=vector_size, internal_width=internalLayerLength*2)
        self._prev_loss_value = 0.0
        self._pending_loss = None
        self._staging_inputs = np.empty((batch_size, vector_size), dtype=np.float32)
        self._staging_targets = np.empty((batch_size, vector_size), dtype=np.float32)

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
        
        n = len(batch_samples)
        for i, (inp, tgt) in enumerate(batch_samples):
            self._staging_inputs[i] = inp
            self._staging_targets[i] = tgt
        batch_inputs = self.backend.to_device(self._staging_inputs[:n])
        batch_targets = self.backend.to_device(self._staging_targets[:n])
        # batch_inputs = self.backend.to_device(np.vstack([sample[0] for sample in batch_samples]))
        # batch_targets = self.backend.to_device(np.vstack([sample[1] for sample in batch_samples]))

        n_samples = len(batch_samples)

        # Timing checkpoints for top-level step profiler.
        step_t0 = time.perf_counter() if self._profiler_enabled else 0.0

        t0 = time.perf_counter() if self._profiler_enabled else 0.0
        calc_outputs = self.train_forward(batch_inputs)  # stays on device
        t1 = time.perf_counter() if self._profiler_enabled else 0.0
        

        diffs, partial_sums_dev = self.backend.subtract_divide_loss(batch_targets, calc_outputs, float(n_samples))
        self._pending_loss = partial_sums_dev
        t2 = time.perf_counter() if self._profiler_enabled else 0.0
        self.backpropagate(diffs, learningRate, momentum=momentum)
        t3 = time.perf_counter() if self._profiler_enabled else 0.0

        loss = 0.5 * float(n_samples) * self._prev_loss_value  
        self._prev_loss_value = float(np.sum(self.backend.to_host(self._pending_loss)))


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
    dataset_size = 2048

    gen = SmileyGenerator(size=28, seed=42)
    pairs = gen.generate_dataset_pairs(dataset_size, eyes_ratio=0.0)

    # Build training pairs (same format as your testSet3)
    testSet = []
    [testSet.append([
        np.atleast_2d(pairs[i][0].flatten().astype(np.float32) / 255.0),
        np.atleast_2d(pairs[i][1].flatten().astype(np.float32) / 255.0)
    ]) for i in range(dataset_size)]

    backend = "opencl"  # "cpu" or "opencl"
    vector_size = 28*28
    internal_network_layers = 3
    coupling_layers = 6
    hidden_width = 256
    assert vector_size % 2 == 0, "vector_size must be even for coupling split"

    initial_lr = 0.001
    min_lr = 0.0001
    output_error_track = []
    startTime = time.perf_counter()
    learnRate = 0

    num_epochs = 500
    samples_per_epoch = len(testSet)
    batch_size = min(128, samples_per_epoch)
    updates_per_epoch = (samples_per_epoch + batch_size - 1) // batch_size
    total_iterations = num_epochs * updates_per_epoch
    global_iteration = 0

    cosine_epochs = 200  # LR reaches min_lr here regardless of num_epochs
    cosine_iterations = cosine_epochs * updates_per_epoch

    inn = InvertibleNeuralNetwork(coupling_layers, vector_size, internal_network_layers, hidden_width, backend=backend, batch_size=batch_size, vector_size=vector_size)

    if (False):  # Set to True to load pretrained weights if available
        try:
            inn.load_weights("inn_smiley_weights.npz")
            print("Loaded pretrained weights from inn_smiley_weights.npz")
        except FileNotFoundError:
            print("Pretrained weights not found, starting with random initialization.")
    else:

        inn.enable_step_profiler(True)
        inn.enable_internal_network_profiler(True)

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

        import math
        import msvcrt
        
        # Main training loop:
        #   - shuffle data
        #   - iterate minibatches with cosine LR + momentum schedule
        #   - accumulate epoch loss
        #   - print top-level and internal profiler summaries
        for epoch in range(num_epochs):
            inn.reset_step_profile()
            inn.reset_internal_network_profile()
            # Shuffle order each epoch
            indices = list(range(len(testSet)))
            random.shuffle(indices)
            
            epoch_error = 0
            for start in range(0, len(indices), batch_size):
                batch_indices = indices[start:start + batch_size]
                batch_samples = [testSet[idx] for idx in batch_indices]
                iteration = global_iteration

                warmup_steps = cosine_iterations // 40
                if iteration < warmup_steps:
                    learnRate = initial_lr * (iteration / warmup_steps)
                elif iteration < cosine_iterations:
                    progress = (iteration - warmup_steps) / (cosine_iterations - warmup_steps)
                    learnRate = min_lr + 0.5 * (initial_lr - min_lr) * (1 + math.cos(math.pi * progress))
                else:
                    learnRate = min_lr

                # learnRate = min_lr + 0.5 * (initial_lr - min_lr) * (
                #     1 + math.cos(math.pi * iteration / total_iterations))
                
                current_momentum = min(0.75, 0.5 + 0.25 * min(1.0, iteration / cosine_iterations))

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

            # Check for user interrupt (press 'c' to break)
            try:
                if msvcrt.kbhit() and msvcrt.getch().decode().lower() == 'c':
                    print("Training interrupted by user")
                    break
            except (ImportError, Exception):
                pass
            
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
            gc.collect()

        gc.collect()
        print(f"Time: {time.perf_counter() - startTime:.2f}s")
        plt.plot(output_error_track)
        plt.yscale('log')  # log scale shows convergence progress better
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.title('Training Error')
        plt.show()

    

    image_vector = np.atleast_2d(np.array(testSet[0][0]).flatten().astype(np.float32))
    
    # Pass through INN forward
    inn_output = inn.forward(image_vector)
    #print(inn_output.shape)
    
    # Pass through INN backward (reconstruction)
    reconstructed_vector = inn.backward(inn_output)
    
    # Reshape vectors back to 28x28 for display
    original_img = image_vector.reshape(28, 28)
    output_img = inn_output.reshape(28, 28)
    reconstructed_img = reconstructed_vector.reshape(28, 28)
    
    # Plot original, INN output, and reconstruction side by side
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    axes[0].imshow(original_img, cmap='gray')
    axes[0].set_title(f'Original (label: {testSet[0][1]})')
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

    error_map = np.abs(inn_output.reshape(28,28) - testSet[0][1].reshape(28,28))
    plt.imshow(error_map, cmap='hot')
    plt.colorbar()
    plt.show()
    
    # Print reconstruction error
    recon_error = np.mean((image_vector - reconstructed_vector) ** 2)
    print(f"Reconstruction MSE: {recon_error:.8f}")

    # inn.save_weights("inn_smiley_weights.npz")
