import numpy as np
import random as rand
import time
from typing import List
from Backends import get_backend

def soft_clip(x, limit=2.0):
    return limit * np.tanh(x / limit)

class ArrayNetworkFeedforward:
    """ 
    Class for a neural network implemented using arrays
    ...

         
    Attributes:
    -----------
    weightsByLayer : 2D double array
        Array of weights for the neural network, each column being weights for a layer
    layerLengths : int array
        Array of how many neurons are in each layer
    layers : int
        How many layers there are
    biasByLayer : 2D double array
        Array for the bias for each neuron in the network, each column a layer, each row in a column a neurons bias
    bias : bool
        Keeps track of whether the network has a bias
    layerAct : func Array
        Each layers activation function

    Methods:
    --------
    evaluateNetwork(inputs)
        Evaluates the network for a given input
    printNetwork()
        Prints the weights and biases of the network out
    updateNetworkGeneralizedDelta(self, inputs, diffs, learnRate)
        Updates the network weights using gradient descent
    """
    #
    def __init__(self, inputCount, layers, neuronPerLayer, layerAct, random = False, isBias = True, backend="cpu"):
        """ Sets up the network, randomizing the initial weights
         
        Parameters:
        -----------
            inputCount : int
                Length of Input vector
            layers : int
                Number of layers in network
            neuronPerLayer : int array
                Array with each index specifying how many neurons are in that indicies layer
            random : bool
                Switches between randomized initial weights (true) and 0 initialization (false)
            isBias : bool
                Switches between the network having a bias on each neuron or not (true or false)
        """
        self.bias = isBias
        self.weightsByLayer = [] # Stored as (nextlayerDim X PreviousLayerDim)
        self.layerLengths = []
        self.layers = 0
        self.biasByLayer = [] # Should be (1 X LayerDim)
        self.outputsByLayerNonActivated = [] # Should always be stored as an array of (1 X layerDim)
        self.outputsByLayerActivated = [] # Should always be stored as an array of (1 X layerDim)
        self.layerAct = [] # List of activation functions and their derivatives
        # Backend object (NumPy for CPU, OpenCL backend for GPU path).
        self.backend = get_backend(backend)
        self.backend_name = backend
        self._use_opencl_backend = backend.lower() in ("opencl", "gpu", "amd")
        # Minimum cached batch size where GPU cached path is even considered.
        self._opencl_cached_min_batch = 8
        # Cached passes feed backprop buffers that currently live on host.
        # Using OpenCL here causes frequent device->host sync and copy overhead.
        # Keep cached/training path on CPU by default; inference (cached=False)
        # still uses OpenCL+CLBlast.
        self._use_opencl_for_cached = False
        # When True, cached forward activations are kept as device buffers.
        self._cache_opencl_buffers_on_device = False
        # Controls whether backprop math tries to stay on device.
        self._use_device_backprop_math = self._use_opencl_backend
        # Numeric guardrails used to avoid exploding updates / NaN cascades.
        self._grad_clip_guard = 5.0
        self._weight_clip_guard = 10.0
        # This eliminates host<->device round-trips during updates.
        self._device_resident_weights = self._use_opencl_backend
        # Device-side parameter caches (transposed weights for backprop).
        # When _device_resident_weights is True, _device_weight_cache is not needed
        # since weightsByLayer already holds device arrays. Only the transpose cache
        # is maintained.
        self._device_weight_cache = {}
        self._device_weight_t_cache = {}
        self._device_bias_cache = {}
        # Internal profiler captures hot sub-phases inside this network.
        self._internal_profiler_enabled = False
        self._internal_profile_totals = {
            "backprop_delta_total": 0.0,
            "backprop_delta_device": 0.0,
            "backprop_delta_cpu": 0.0,
            "update_total": 0.0,
            "update_delta_propagation": 0.0,
            "update_gradient_build": 0.0,
            "update_weight_apply": 0.0,
        }
        self._internal_profile_counts = {
            "backprop_calls": 0,
            "update_calls": 0,
        }
        self._paranoid_stabilize = False  # Set True only when debugging NaN issues
        for i in range(layers):
            if (i == 0):
                n = neuronPerLayer[0]
                m = inputCount
                # First layer depends on input count and point to the first layer length
            else:
                n = neuronPerLayer[i]
                m = neuronPerLayer[i-1]
                # Subsequent layers take previous layer length and point to next layer length
            if (isBias):
                bias = []
                for j in range(neuronPerLayer[i]):
                    if (random):
                        # Keep biases near zero for stable affine coupling startup
                        bias.append(rand.uniform(-0.01, 0.01))
                    else:
                        bias.append(0)
                self.biasByLayer.append(np.atleast_2d(bias))
                # If bias is turned on add a random bias for each neuron for each layer
            weights = np.zeros((m, n))
            if (random):
                # Xavier-style initialization to avoid exploding activations/gradients
                limit = np.sqrt(6.0 / (m + n))
                for j in range(n):
                    for k in range(m):
                        weights[k][j] = rand.uniform(-limit, limit)
            # If random is turned on give each layer's weights a random value
            self.weightsByLayer.append(weights)
            self.layerLengths.append(n)
        # Sentinel at the end keeps old indexing patterns compatible.
        self.weightsByLayer.append(0)
        self.layers = layers
        self.layerAct = layerAct
        self.weight_velocities = []
        self.bias_velocities = []
        for i in range(layers):
            self.weight_velocities.append(np.zeros_like(self.weightsByLayer[i]) 
                                        if isinstance(self.weightsByLayer[i], np.ndarray) 
                                        else 0)
            if isBias:
                self.bias_velocities.append(np.zeros_like(self.biasByLayer[i]))

        # Cache activation compatibility (array-native vs scalar-only) once.
        # This avoids costly try/except checks inside training loops.
        self._act_fast = []
        self._act_deriv_fast = []
        probe = np.array([[0.0, -1.0, 1.0]], dtype=np.float32)
        for i in range(self.layers):
            self._act_fast.append(self._is_array_compatible(self.layerAct[i][0], probe))
            self._act_deriv_fast.append(self._is_array_compatible(self.layerAct[i][1], probe))

        # Upload weights, biases, and velocities to device if using GPU backend.
        if self._device_resident_weights:
            self._upload_all_params_to_device()


    def _upload_all_params_to_device(self):
        """Move all weight/bias/velocity arrays to device. Called once at init
        and can be called again if params are replaced from host."""
        for i in range(self.layers):
            w = self.weightsByLayer[i]
            if isinstance(w, np.ndarray):
                self.weightsByLayer[i] = self.backend.to_device(w, dtype=np.float32)
            v = self.weight_velocities[i]
            if isinstance(v, np.ndarray):
                self.weight_velocities[i] = self.backend.to_device(v, dtype=np.float32)
            if self.bias:
                b = self.biasByLayer[i]
                if isinstance(b, np.ndarray):
                    self.biasByLayer[i] = self.backend.to_device(b, dtype=np.float32)
                bv = self.bias_velocities[i]
                if isinstance(bv, np.ndarray):
                    self.bias_velocities[i] = self.backend.to_device(bv, dtype=np.float32)
        # Rebuild transpose cache
        self._device_weight_t_cache.clear()
        self._device_weight_cache.clear()
        self._device_bias_cache.clear()

    def _get_weight_host(self, layer_index: int) -> np.ndarray:
        """Return weight matrix on host. Pulls from device if needed."""
        w = self.weightsByLayer[layer_index]
        if self.backend.is_device_array(w):
            return self.backend.to_host(w)
        return np.asarray(w, dtype=np.float32)

    def _get_bias_host(self, layer_index: int) -> np.ndarray:
        """Return bias vector on host. Pulls from device if needed."""
        b = self.biasByLayer[layer_index]
        if self.backend.is_device_array(b):
            return self.backend.to_host(b)
        return np.asarray(b, dtype=np.float32)
    
    @staticmethod
    def _is_array_compatible(func, probe: np.ndarray) -> bool:
        """Return True when callable can consume/return ndarray with matching shape."""
        try:
            out = np.asarray(func(probe))
            return out.shape == probe.shape
        except Exception:
            return False

    @staticmethod
    def _apply_elementwise(func, x: np.ndarray, fast_path: bool) -> np.ndarray:
        """Apply callable over ndarray using fast path when possible."""
        if fast_path:
            return func(x)
        mapped = np.frompyfunc(func, 1, 1)(x)
        return np.asarray(mapped, dtype=x.dtype)
    
    def _maybe_stabilize(self, x, clip_value=None):
        if self._paranoid_stabilize:
            return self._stabilize_host_array(x, clip_value=clip_value or self._grad_clip_guard)
        return x
    
    def _maybe_stabilize_dev(self, x, clip_value=None):
        """Device-aware stabilization: soft-clip on device if possible."""
        if self._paranoid_stabilize:
            cv = clip_value or self._grad_clip_guard
            if self.backend.is_device_array(x):
                return self._soft_clip_backend(x, cv)
            return self._stabilize_host_array(x, clip_value=cv)
        return x

    def _apply_activation_backend(self, x, layer_index: int):
        # Try backend-native activation first (GPU-friendly path).
        # If unsupported, fallback to host callable execution.
        func = self.layerAct[layer_index][0]
        func_name = getattr(func, "__name__", "")
        backend_out = self.backend.apply_activation(x, func_name)
        if backend_out is not None:
            return backend_out

        host_x = self.backend.to_host(x)
        host_out = self._apply_elementwise(func, host_x, self._act_fast[layer_index])
        return self.backend.to_device(host_out, dtype=host_out.dtype)

    def _get_device_weight(self, layer_index: int):
        # Lazy-upload each layer matrix once, then reuse until invalidated.
        if self._device_resident_weights:
            # Already on device
            return self.weightsByLayer[layer_index]
        cached = self._device_weight_cache.get(layer_index)
        if cached is None:
            cached = self.backend.to_device(self.weightsByLayer[layer_index], dtype=np.float32)
            self._device_weight_cache[layer_index] = cached
        return cached

    def _get_device_weight_t(self, layer_index: int):
        """Return transposed weight matrix on device (cached)."""
        cached = self._device_weight_t_cache.get(layer_index)
        if cached is None:
            if self._device_resident_weights:
                w_host = self.backend.to_host(self.weightsByLayer[layer_index])
            else:
                w_host = self.weightsByLayer[layer_index]
            cached = self.backend.to_device(np.ascontiguousarray(w_host.T), dtype=np.float32)
            self._device_weight_t_cache[layer_index] = cached
        return cached

    def _get_device_bias(self, layer_index: int):
        if not self.bias:
            return None
        if self._device_resident_weights:
            return self.biasByLayer[layer_index]
        cached = self._device_bias_cache.get(layer_index)
        if cached is None:
            cached = self.backend.to_device(self.biasByLayer[layer_index], dtype=np.float32)
            self._device_bias_cache[layer_index] = cached
        return cached

    def _invalidate_device_cache(self, layer_index=None):
        """Invalidate transpose cache after weight updates.
        When weights are device-resident, the weight/bias caches are not used
        (weightsByLayer IS the device array). Only the transpose cache needs
        invalidation since it's derived from the weight values."""
        if layer_index is None:
            self._device_weight_t_cache.clear()
            if not self._device_resident_weights:
                self._device_weight_cache.clear()
                self._device_bias_cache.clear()
            return
        self._device_weight_t_cache.pop(layer_index, None)
        if not self._device_resident_weights:
            self._device_weight_cache.pop(layer_index, None)
            self._device_bias_cache.pop(layer_index, None)

    def set_cached_opencl(self, enabled: bool, min_batch: int = 8, cache_on_device: bool = False):
        # Control if cached forward path uses OpenCL and where cached tensors live.
        self._use_opencl_for_cached = bool(enabled)
        self._opencl_cached_min_batch = max(1, int(min_batch))
        self._cache_opencl_buffers_on_device = bool(cache_on_device)

    def set_device_backprop_math(self, enabled: bool):
        # Toggle for delta/update math backend route.
        self._use_device_backprop_math = bool(enabled)

    def enable_internal_profiler(self, enabled=True):
        self._internal_profiler_enabled = bool(enabled)

    def reset_internal_profile(self):
        for key in self._internal_profile_totals:
            self._internal_profile_totals[key] = 0.0
        for key in self._internal_profile_counts:
            self._internal_profile_counts[key] = 0

    def get_internal_profile(self, reset=False):
        backprop_calls = max(1, self._internal_profile_counts["backprop_calls"])
        update_calls = max(1, self._internal_profile_counts["update_calls"])
        report = {
            "counts": dict(self._internal_profile_counts),
            "totals": dict(self._internal_profile_totals),
            "avg_backprop_delta_total_s": self._internal_profile_totals["backprop_delta_total"] / backprop_calls,
            "avg_backprop_delta_device_s": self._internal_profile_totals["backprop_delta_device"] / backprop_calls,
            "avg_backprop_delta_cpu_s": self._internal_profile_totals["backprop_delta_cpu"] / backprop_calls,
            "avg_update_total_s": self._internal_profile_totals["update_total"] / update_calls,
            "avg_update_delta_propagation_s": self._internal_profile_totals["update_delta_propagation"] / update_calls,
            "avg_update_gradient_build_s": self._internal_profile_totals["update_gradient_build"] / update_calls,
            "avg_update_weight_apply_s": self._internal_profile_totals["update_weight_apply"] / update_calls,
        }
        if reset:
            self.reset_internal_profile()
        return report

    def _soft_clip_backend(self, x, limit=2.0):
        if limit <= 0:
            return x
        # Use fused kernel if backend supports it
        if hasattr(self.backend, 'soft_clip') and self.backend.is_device_array(x):
            return self.backend.soft_clip(x, limit)
        if not self.backend.is_device_array(x):
            x = self.backend.to_device(x, dtype=np.float32)
        scaled = self.backend.divide(x, float(limit))
        tanh_scaled = self.backend.apply_activation(scaled, "tanh")
        return self.backend.multiply(float(limit), tanh_scaled)

    def _activation_derivative_backend(self, x, layer_index: int):
        # Uses backend-native derivative when available; otherwise host fallback.
        if not self.backend.is_device_array(x):
            x_dev = self.backend.to_device(x, dtype=np.float32)
        else:
            x_dev = x
        activation_name = getattr(self.layerAct[layer_index][0], "__name__", "")
        deriv = self.backend.apply_activation_derivative(x_dev, activation_name)
        if deriv is not None:
            return deriv
        x_host = self.backend.to_host(x_dev)
        deriv_host = self._apply_elementwise(
            self.layerAct[layer_index][1],
            x_host,
            self._act_deriv_fast[layer_index]
        )
        return self.backend.to_device(deriv_host, dtype=np.float32)

    @staticmethod
    def _stabilize_host_array(x, clip_value=1e4):
        # Final host-side guard: sanitize non-finite values and bound magnitude.
        arr = np.asarray(x, dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=clip_value, neginf=-clip_value)
        if clip_value is not None and clip_value > 0:
            arr = np.clip(arr, -clip_value, clip_value)
        return arr

    def evaluateNetwork(self, inputs:List[float], cached:bool=True) -> np.ndarray:
        """ Evaluates the network for a given input
         
        Parameters:
        -----------
            inputs : array(float)
                Vector of input values
        """
        if (cached):
            self.outputsByLayerActivated=[]
            self.outputsByLayerNonActivated=[]
        if not self.backend.is_device_array(inputs):
            input_cols = inputs.shape[1]
        else:
            input_cols = self.backend.to_host(inputs).shape[1]
        w0 = self._get_weight_host(0) if self._device_resident_weights else self.weightsByLayer[0]
        if (input_cols != w0.shape[0]):
            print("evaluateNetwork: Wrong number of inputs!")
            print(str(input_cols) + " Vs " + str(w0.shape[0]))
            return None
        # Inputs are expected as (B x D) where B can be 1.
        temp = inputs
        batch_rows = temp.shape[0] if hasattr(temp, "shape") and len(temp.shape) > 1 else 1
        # Cached passes are often used by backprop and can be copy-heavy.
        # So OpenCL cached mode is gated by explicit flags + minimum batch size.
        using_opencl = self._use_opencl_backend and (
            (not cached) or (
                self._use_opencl_for_cached and
                (batch_rows >= self._opencl_cached_min_batch)
            )
        )
        
        # When weights are device-resident, always use OpenCL path to avoid
        # pulling weights to host for every forward pass.
        if self._device_resident_weights:
            using_opencl = True
        
        # Forward pass through all layers.
        for i in range(self.layers):
        # For each layer:
            if self.bias:
                if using_opencl:
                    w = self._get_device_weight(i)
                    b = self._get_device_bias(i)
                    temp = self.backend.add(self.backend.matmul(temp, w), b)
                else:
                    temp = np.add(np.matmul(temp, self.weightsByLayer[i]), self.biasByLayer[i])
            else:
                if using_opencl:
                    w = self._get_device_weight(i)
                    temp = self.backend.matmul(temp, w)
                else:
                    temp = np.matmul(temp, self.weightsByLayer[i])
            # Save pre-activation / activation caches for backprop when requested.
            if (cached):
                if using_opencl and self._cache_opencl_buffers_on_device:
                    self.outputsByLayerNonActivated.append(temp)
                else:
                    self.outputsByLayerNonActivated.append(self.backend.to_host(temp) if using_opencl else temp)
            if using_opencl:
                temp = self._apply_activation_backend(temp, i)
            else:
                temp = self._apply_elementwise(self.layerAct[i][0], temp, self._act_fast[i])
            if (cached):
                if using_opencl and self._cache_opencl_buffers_on_device:
                    self.outputsByLayerActivated.append(temp)
                else:
                    self.outputsByLayerActivated.append(self.backend.to_host(temp) if using_opencl else temp)
                # These caches are consumed by backpropDelta/updateNetworkGeneralizedDelta.
        # Return device array when OpenCL was used, host array otherwise.
        # Callers must handle both cases via backend.is_device_array() or _to_dev().
        return temp
    
    def printNetwork(self):
        """ Prints the weights and biases of the network out
        """
        for i in range(self.layers):
            print("The", i, "th layer has weights: ")
            w = self._get_weight_host(i) if self._device_resident_weights else self.weightsByLayer[i]
            print(w)
            if self.bias:
                b = self._get_bias_host(i) if self._device_resident_weights else self.biasByLayer[i]
                print("The biases on layer", i, "are", b)

    def outputEmbedding(self):
        """ Prints the weights and biases of the network out
        """
        w0 = self._get_weight_host(0) if self._device_resident_weights else self.weightsByLayer[0]
        with open("wordEmbedding.txt", "w") as txt_file:
            for line in w0:
                txt_file.write(" ".join(map(str, line)) + "\n")
            txt_file.close()
        return w0
    
    def backpropDelta(self, outerGrad, clip_limit=1.0):
        """Propagate outer gradient to get dL/d(network_input).
        Parameters:
        -----------
            outerGrad : (1, output_dim)
                This is the gradient coming into this layer from the layer after it.
            clip_limit : float
                If > 0, soft clip deltas after each layer to prevent amplification cascade.
        Returns:
        -----------
            result : (1, input_dim)
                This is the gradient with respect to the input of this network, which is then used to propagate to the previous layer.
        """
        backprop_t0 = time.perf_counter() if self._internal_profiler_enabled else 0.0

        # Phase A: propagate outer gradient through this subnet to produce dL/dInput.
        # GPU route keeps delta math on device for this pass.
        if self._use_device_backprop_math and self._use_opencl_backend:
            device_t0 = time.perf_counter() if self._internal_profiler_enabled else 0.0
            delta = outerGrad
            for i in range(self.layers):
                layer_idx = self.layers - 1 - i
                preact = self.outputsByLayerNonActivated[layer_idx]
                act_derivs = self._activation_derivative_backend(preact, layer_idx)
                delta = self.backend.multiply(delta, act_derivs)
                delta = self.backend.matmul(delta, self._get_device_weight_t(layer_idx))

                if clip_limit > 0:
                    delta = self._soft_clip_backend(delta, clip_limit)
            result = self._soft_clip_backend(delta, self._grad_clip_guard)
            if self._internal_profiler_enabled:
                t_end = time.perf_counter()
                self._internal_profile_totals["backprop_delta_device"] += (t_end - device_t0)
                self._internal_profile_totals["backprop_delta_total"] += (t_end - backprop_t0)
                self._internal_profile_counts["backprop_calls"] += 1
            return result
        
        cpu_t0 = time.perf_counter() if self._internal_profiler_enabled else 0.0
        # CPU fallback route (or when device math is disabled).
        delta = outerGrad
        for i in range(self.layers):
            layer_idx = self.layers - 1 - i
            
            # Apply activation derivative at this layer
            preact = self.backend.to_host(self.outputsByLayerNonActivated[layer_idx])
            act_derivs = self._apply_elementwise(
                self.layerAct[layer_idx][1],
                preact,
                self._act_deriv_fast[layer_idx]
            )
            delta = np.multiply(delta, act_derivs)
            
            # Propagate through this layer's weights
            w_host = self._get_weight_host(layer_idx) if self._device_resident_weights else self.weightsByLayer[layer_idx]
            delta = np.matmul(delta, w_host.T)

            # Clip after each layer to prevent amplification cascade
            if clip_limit > 0:
                delta = soft_clip(delta, clip_limit)

        result = self._stabilize_host_array(delta, clip_value=self._grad_clip_guard)
        if self._internal_profiler_enabled:
            t_end = time.perf_counter()
            self._internal_profile_totals["backprop_delta_cpu"] += (t_end - cpu_t0)
            self._internal_profile_totals["backprop_delta_total"] += (t_end - backprop_t0)
            self._internal_profile_counts["backprop_calls"] += 1
        return result

    def updateNetworkGeneralizedDelta(self, inputs, diffs, learnRate, weightDecay = .001, momentum=0.9):
        """ Updates the network weights using gradient descent
         
        Parameters:
        -----------
            inputs : double array
                Vector of input values, (1XD)
            diffs : double array
                Vector corresponding to the difference between target and outputs, (1XD)
            learnRate : double
                Proportionality constant for how much the change to the weights should be
            weightDecay : double
                Proportionality constant for how much the weights should be decayed by each update step
            momentum : double
                Momentum constant for how much of the previous update step's change should be applied in this step
        """
        update_t0 = time.perf_counter() if self._internal_profiler_enabled else 0.0


        device_update_path = self._use_device_backprop_math and self._use_opencl_backend

        if device_update_path:
            self._update_device_path(inputs, diffs, learnRate, weightDecay, momentum, update_t0)
        else:
            self._update_host_path(inputs, diffs, learnRate, weightDecay, momentum, update_t0)


    def _update_device_path(self, inputs, diffs, learnRate, weightDecay, momentum, update_t0):
        """Fully device-resident update path. Weights, velocities, gradients all stay on GPU."""
        inputs_dev = self.backend.to_device(inputs, dtype=np.float32)
        delta_dev = self.backend.to_device(diffs, dtype=np.float32)

        changeManifold = []
        biasManifold = []
        delta_stage_total = 0.0
        grad_stage_total = 0.0

        lr_scalar = float(learnRate)
        wd_scalar = float(weightDecay)
        mom_scalar = float(momentum)
        decay_factor = 1.0 - lr_scalar * wd_scalar

        for i in range(self.layers):
            delta_stage_t0 = time.perf_counter() if self._internal_profiler_enabled else 0.0
            layer_idx = self.layers - 1 - i
            outputs_cached = self.outputsByLayerNonActivated[layer_idx]

            if i != self.layers - 1:
                currentInputs_dev = self.outputsByLayerActivated[self.layers - i - 2]
            else:
                currentInputs_dev = inputs_dev

            # Ensure cached activations are on device
            if not self.backend.is_device_array(currentInputs_dev):
                currentInputs_dev = self.backend.to_device(currentInputs_dev, dtype=np.float32)

            # Delta propagation
            if i != 0:
                # Sentinel weight at self.layers index for outgoing layer
                protoNewDeltas_dev = self.backend.matmul(delta_dev, self._get_device_weight_t(layer_idx + 1))
                act_deriv_dev = self._activation_derivative_backend(outputs_cached, layer_idx)
                delta_dev = self.backend.multiply(protoNewDeltas_dev, act_deriv_dev)
            else:
                act_deriv_dev = self._activation_derivative_backend(outputs_cached, layer_idx)
                delta_dev = self.backend.multiply(delta_dev, act_deriv_dev)

            if self._grad_clip_guard > 0:
                delta_dev = self._soft_clip_backend(delta_dev, self._grad_clip_guard)

            if self._internal_profiler_enabled:
                delta_stage_total += (time.perf_counter() - delta_stage_t0)

            # Gradient build — on device
            grad_stage_t0 = time.perf_counter() if self._internal_profiler_enabled else 0.0

            # change = currentInputs.T @ delta  (prevD x batch) @ (batch x curD) = (prevD x curD)
            change_dev = self.backend.matmul(
                self.backend.transpose(currentInputs_dev),
                delta_dev
            )
            if self._grad_clip_guard > 0:
                change_dev = self._soft_clip_backend(change_dev, self._grad_clip_guard)

            # Bias gradient: sum over batch dimension
            bias_change_dev = self.backend.sum(delta_dev, axis=0, keepdims=True)
            if self._grad_clip_guard > 0:
                bias_change_dev = self._soft_clip_backend(bias_change_dev, self._grad_clip_guard)

            if self._internal_profiler_enabled:
                grad_stage_total += (time.perf_counter() - grad_stage_t0)

            changeManifold.append((layer_idx, change_dev, bias_change_dev))

        # Weight apply — all on device
        weight_stage_t0 = time.perf_counter() if self._internal_profiler_enabled else 0.0

        for layer_idx, change_dev, bias_change_dev in changeManifold:
            w = self.weightsByLayer[layer_idx]

            if mom_scalar > 0:
                # velocity = momentum * velocity + change
                self.weight_velocities[layer_idx] = self.backend.add(
                    self.backend.multiply(mom_scalar, self.weight_velocities[layer_idx]),
                    change_dev
                )
                effective_change = self.backend.multiply(lr_scalar, self.weight_velocities[layer_idx])
            else:
                effective_change = self.backend.multiply(lr_scalar, change_dev)

            # w = w * (1 - lr * wd) + effective_change
            decayed_w = self.backend.multiply(decay_factor, w)
            self.weightsByLayer[layer_idx] = self.backend.add(decayed_w, effective_change)

            if self._paranoid_stabilize:
                self.weightsByLayer[layer_idx] = self._soft_clip_backend(
                    self.weightsByLayer[layer_idx], self._weight_clip_guard
                )

            # Invalidate transpose cache since weights changed
            self._device_weight_t_cache.pop(layer_idx, None)

            if self.bias:
                if mom_scalar > 0:
                    self.bias_velocities[layer_idx] = self.backend.add(
                        self.backend.multiply(mom_scalar, self.bias_velocities[layer_idx]),
                        bias_change_dev
                    )
                    effective_bias = self.backend.multiply(lr_scalar, self.bias_velocities[layer_idx])
                else:
                    effective_bias = self.backend.multiply(lr_scalar, bias_change_dev)

                self.biasByLayer[layer_idx] = self.backend.add(
                    self.biasByLayer[layer_idx], effective_bias
                )

                if self._paranoid_stabilize:
                    self.biasByLayer[layer_idx] = self._soft_clip_backend(
                        self.biasByLayer[layer_idx], self._weight_clip_guard
                    )

        if self._internal_profiler_enabled:
            update_t_end = time.perf_counter()
            self._internal_profile_totals["update_delta_propagation"] += delta_stage_total
            self._internal_profile_totals["update_gradient_build"] += grad_stage_total
            self._internal_profile_totals["update_weight_apply"] += (update_t_end - weight_stage_t0)
            self._internal_profile_totals["update_total"] += (update_t_end - update_t0)
            self._internal_profile_counts["update_calls"] += 1

    def _update_host_path(self, inputs, diffs, learnRate, weightDecay, momentum, update_t0):
        """Original host-based update path for CPU backend."""
        inputs_host = self.backend.to_host(inputs)

        changeManifold = []
        biasManifold = []
        delta = None
        delta_stage_total = 0.0
        grad_stage_total = 0.0

        for i in range(self.layers):
            delta_stage_t0 = time.perf_counter() if self._internal_profiler_enabled else 0.0
            layer_idx = self.layers - 1 - i
            outputs = self.backend.to_host(self.outputsByLayerNonActivated[layer_idx])

            if i != self.layers - 1:
                currentInputs = self.backend.to_host(self.outputsByLayerActivated[self.layers - i - 2])
            else:
                currentInputs = inputs_host

            if i != 0:
                w_host = self._get_weight_host(layer_idx + 1) if self._device_resident_weights else self.weightsByLayer[layer_idx + 1]
                protoNewDeltas = np.matmul(delta, w_host.T)
                act_deriv = self._apply_elementwise(
                    self.layerAct[layer_idx][1], outputs, self._act_deriv_fast[layer_idx]
                )
                newDeltas = self._stabilize_host_array(
                    np.multiply(protoNewDeltas, act_deriv), clip_value=self._grad_clip_guard
                )
            else:
                act_deriv = self._apply_elementwise(
                    self.layerAct[layer_idx][1], outputs, self._act_deriv_fast[layer_idx]
                )
                newDeltas = self._stabilize_host_array(
                    np.multiply(diffs, act_deriv), clip_value=self._grad_clip_guard
                )

            if self._internal_profiler_enabled:
                delta_stage_total += (time.perf_counter() - delta_stage_t0)

            grad_stage_t0 = time.perf_counter() if self._internal_profiler_enabled else 0.0
            change = self._stabilize_host_array(
                np.matmul(currentInputs.T, newDeltas), clip_value=self._grad_clip_guard
            )
            changeBias = self._stabilize_host_array(
                np.sum(newDeltas, axis=0, keepdims=True), clip_value=self._grad_clip_guard
            )
            if self._internal_profiler_enabled:
                grad_stage_total += (time.perf_counter() - grad_stage_t0)

            changeManifold.append((layer_idx, change, changeBias))
            delta = newDeltas

        # Apply gradients
        weight_stage_t0 = time.perf_counter() if self._internal_profiler_enabled else 0.0

        for layer_idx, change_h, bias_h in changeManifold:
            w_host = self._get_weight_host(layer_idx) if self._device_resident_weights else self.weightsByLayer[layer_idx]

            if momentum > 0:
                vel = self.backend.to_host(self.weight_velocities[layer_idx]) if self.backend.is_device_array(self.weight_velocities[layer_idx]) else self.weight_velocities[layer_idx]
                vel = momentum * vel + change_h
                effective_change = learnRate * vel
                if self._device_resident_weights:
                    self.weight_velocities[layer_idx] = self.backend.to_device(vel, dtype=np.float32)
                else:
                    self.weight_velocities[layer_idx] = vel
            else:
                effective_change = learnRate * change_h

            new_w = self._maybe_stabilize(
                np.add(w_host * (1 - learnRate * weightDecay), effective_change),
                clip_value=self._weight_clip_guard
            )
            if self._device_resident_weights:
                self.weightsByLayer[layer_idx] = self.backend.to_device(new_w, dtype=np.float32)
            else:
                self.weightsByLayer[layer_idx] = new_w
            self._invalidate_device_cache(layer_idx)

            if self.bias:
                b_host = self._get_bias_host(layer_idx) if self._device_resident_weights else self.biasByLayer[layer_idx]
                if momentum > 0:
                    bv = self.backend.to_host(self.bias_velocities[layer_idx]) if self.backend.is_device_array(self.bias_velocities[layer_idx]) else self.bias_velocities[layer_idx]
                    bv = momentum * bv + bias_h
                    effective_bias = learnRate * bv
                    if self._device_resident_weights:
                        self.bias_velocities[layer_idx] = self.backend.to_device(bv, dtype=np.float32)
                    else:
                        self.bias_velocities[layer_idx] = bv
                else:
                    effective_bias = learnRate * bias_h

                new_b = self._maybe_stabilize(
                    np.add(b_host, effective_bias), clip_value=self._weight_clip_guard
                )
                if self._device_resident_weights:
                    self.biasByLayer[layer_idx] = self.backend.to_device(new_b, dtype=np.float32)
                else:
                    self.biasByLayer[layer_idx] = new_b

        if self._internal_profiler_enabled:
            update_t_end = time.perf_counter()
            self._internal_profile_totals["update_delta_propagation"] += delta_stage_total
            self._internal_profile_totals["update_gradient_build"] += grad_stage_total
            self._internal_profile_totals["update_weight_apply"] += (update_t_end - weight_stage_t0)
            self._internal_profile_totals["update_total"] += (update_t_end - update_t0)
            self._internal_profile_counts["update_calls"] += 1

    def computeJacobian(self, inputs):
        """Compute d(output)/d(input) Jacobian matrix"""
        self.evaluateNetwork(inputs, True)
        # Start with identity
        jacobian = np.eye(self.weightsByLayer[0].shape[0])
        for i in range(self.layers):
            # Derivative of activation
            preact = self.outputsByLayerNonActivated[i]
            act_deriv = self._apply_elementwise(self.layerAct[i][1], preact, self._act_deriv_fast[i])
            act_deriv = np.asarray(act_deriv).flatten()
            # Chain: J = diag(f') @ W @ J_prev
            jacobian = np.diag(act_deriv) @ self.weightsByLayer[i].T @ jacobian
        return jacobian
