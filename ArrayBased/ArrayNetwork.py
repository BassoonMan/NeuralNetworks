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
        # Device-side parameter caches to avoid repeated host->device copies.
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
        cached = self._device_weight_cache.get(layer_index)
        if cached is None:
            cached = self.backend.to_device(self.weightsByLayer[layer_index], dtype=np.float32)
            self._device_weight_cache[layer_index] = cached
        return cached

    def _get_device_weight_t(self, layer_index: int):
        # Transposed cache is used in backprop delta propagation matmuls.
        cached = self._device_weight_t_cache.get(layer_index)
        if cached is None:
            cached = self.backend.to_device(self.weightsByLayer[layer_index].T, dtype=np.float32)
            self._device_weight_t_cache[layer_index] = cached
        return cached

    def _get_device_bias(self, layer_index: int):
        if not self.bias:
            return None
        cached = self._device_bias_cache.get(layer_index)
        if cached is None:
            cached = self.backend.to_device(self.biasByLayer[layer_index], dtype=np.float32)
            self._device_bias_cache[layer_index] = cached
        return cached

    def _invalidate_device_cache(self, layer_index=None):
        # Called after parameter updates so stale device weights are never reused.
        if layer_index is None:
            self._device_weight_cache.clear()
            self._device_weight_t_cache.clear()
            self._device_bias_cache.clear()
            return
        self._device_weight_cache.pop(layer_index, None)
        self._device_weight_t_cache.pop(layer_index, None)
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
        # Backend-safe soft clipping: limit * tanh(x/limit).
        if limit <= 0:
            return x
        xd = self.backend.to_device(x, dtype=np.float32)
        scaled = self.backend.divide(xd, float(limit))
        tanh_scaled = self.backend.apply_activation(scaled, "tanh")
        return self.backend.multiply(float(limit), tanh_scaled)

    def _activation_derivative_backend(self, x, layer_index: int):
        # Uses backend-native derivative when available; otherwise host fallback.
        x_dev = self.backend.to_device(x, dtype=np.float32)
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
        if (inputs.shape[1] != self.weightsByLayer[0].shape[0]):
            print("evaluateNetwork: Wrong number of inputs!")
            print(str(inputs.shape[1]) + " Vs " + str(self.weightsByLayer[0].shape[0]))
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
        if using_opencl:
            temp = self.backend.to_device(temp, dtype=np.float32)
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
        return self.backend.to_host(temp) if using_opencl else temp
    
    def printNetwork(self):
        """ Prints the weights and biases of the network out
        """
        for i in range(self.layers):
            print("The", i, "th layer has weights: ")
            print(self.weightsByLayer[i])
            if self.bias:
                print("The biases on layer", i, "are", self.biasByLayer[i])

    def outputEmbedding(self):
        """ Prints the weights and biases of the network out
        """
        with open("wordEmbedding.txt", "w") as txt_file:
            for line in self.weightsByLayer[0]:
                # txt_file.write(str(line.tolist()) + ",\n") # works with any number of elements in a line
                txt_file.write(" ".join(map(str, line)) + "\n")
            txt_file.close()
        return self.weightsByLayer[0]
    
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
            delta = self.backend.to_device(outerGrad, dtype=np.float32)
            for i in range(self.layers):
                layer_idx = self.layers - 1 - i
                preact = self.outputsByLayerNonActivated[layer_idx]
                act_derivs = self._activation_derivative_backend(preact, layer_idx)
                delta = self.backend.multiply(delta, act_derivs)
                delta = self.backend.matmul(delta, self._get_device_weight_t(layer_idx))

                if clip_limit > 0:
                    delta = self._soft_clip_backend(delta, clip_limit)
            result = self._stabilize_host_array(self.backend.to_host(delta), clip_value=self._grad_clip_guard)
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
            delta = np.matmul(delta, self.weightsByLayer[layer_idx].T)

            # Clip after each layer to prevent amplification cascade
            if clip_limit > 0:
                delta = soft_clip(delta, clip_limit)  # soft_clip instead of np.clip

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
        # Phase B: parameter update using cached activations and output diffs.
        # This method is profiled in three sub-stages:
        #   1) delta propagation
        #   2) gradient build
        #   3) weight/bias apply
        #inputs_host = self.backend.to_host(inputs)
        device_update_path = self._use_device_backprop_math and self._use_opencl_backend
        if device_update_path:
            inputs_dev = self.backend.to_device(inputs)      # no-op if already on device
            inputs_host = None
        else:
            inputs_host = self.backend.to_host(inputs)
            inputs_dev = None
        #inputs_dev = self.backend.to_device(inputs_host, dtype=np.float32) if device_update_path else None
        changeManifold = [] # Array of all the weight changes
        biasManifold = [] # Array of all the bias changes
        delta = None
        delta_dev = self.backend.to_device(diffs, dtype=np.float32) if device_update_path else None
        delta_stage_total = 0.0
        grad_stage_total = 0.0
        # Build per-layer delta and gradients in reverse order.
        for i in range(self.layers):
            delta_stage_t0 = time.perf_counter() if self._internal_profiler_enabled else 0.0
            outputs_cached = self.outputsByLayerNonActivated[self.layers - i - 1]
            if (i != self.layers - 1):
                currentInputs_cached = self.outputsByLayerActivated[self.layers - i - 2]
            else:
                currentInputs_cached = inputs_dev if device_update_path else inputs_host  #(1 X inD)
            # First layer in reverse uses raw network inputs; others use cached activations.
            if (i != 0):
            # The last layer has no outgoing weights and no previous deltas
                if device_update_path:
                    protoNewDeltas_dev = self.backend.matmul(delta_dev, self._get_device_weight_t(self.layers - i))
                    act_deriv_dev = self._activation_derivative_backend(outputs_cached, self.layers - 1 - i)
                    newDeltas_dev = self.backend.multiply(protoNewDeltas_dev, act_deriv_dev)
                    # Keep propagation on device; host materialization happens in grad stage.
                    if self._grad_clip_guard > 0:
                        newDeltas_dev = self._soft_clip_backend(newDeltas_dev, self._grad_clip_guard)
                    delta_dev = newDeltas_dev
                else:
                    outputs = self.backend.to_host(outputs_cached)
                    outgoingWeights = self.weightsByLayer[self.layers - i] #(nextLayerD X curLayerD)
                    protoNewDeltas = np.matmul(delta, outgoingWeights.T) # Delta is outgoing weights times the previous deltas (1 X curLayerD)
                    act_deriv = self._apply_elementwise(self.layerAct[self.layers - 1 - i][1], outputs, self._act_deriv_fast[self.layers - 1 - i])
                    newDeltas = self._stabilize_host_array(np.multiply(protoNewDeltas, act_deriv), clip_value=self._grad_clip_guard) #(1 X curLayerD)
            else:
                if device_update_path:
                    act_deriv_dev = self._activation_derivative_backend(outputs_cached, self.layers - 1 - i)
                    newDeltas_dev = self.backend.multiply(delta_dev, act_deriv_dev)
                    if self._grad_clip_guard > 0:
                        newDeltas_dev = self._soft_clip_backend(newDeltas_dev, self._grad_clip_guard)
                    delta_dev = newDeltas_dev
                else:
                    outputs = self.backend.to_host(outputs_cached)
                    act_deriv = self._apply_elementwise(self.layerAct[self.layers - 1 - i][1], outputs, self._act_deriv_fast[self.layers - 1 - i])
                    newDeltas = self._stabilize_host_array(np.multiply(diffs, act_deriv), clip_value=self._grad_clip_guard) #(1 X curLayerD)
            if self._internal_profiler_enabled:
                delta_stage_total += (time.perf_counter() - delta_stage_t0)

            grad_stage_t0 = time.perf_counter() if self._internal_profiler_enabled else 0.0
            if device_update_path:
                newDeltas = self._stabilize_host_array(self.backend.to_host(delta_dev), clip_value=self._grad_clip_guard)
                currentInputs = self.backend.to_host(currentInputs_cached)
                change = self._stabilize_host_array(np.matmul(currentInputs.T, newDeltas), clip_value=self._grad_clip_guard)
            else:
                currentInputs = self.backend.to_host(currentInputs_cached)
                change = self._stabilize_host_array(np.matmul(currentInputs.T, newDeltas), clip_value=self._grad_clip_guard) #(curLayerD X prevLayerD)
            # Gradient tensors for this layer.
            changeBias = self._stabilize_host_array(newDeltas, clip_value=self._grad_clip_guard) # Bias is just product of learn rate and deltas # (1 X curLayerD)
            if self._internal_profiler_enabled:
                grad_stage_total += (time.perf_counter() - grad_stage_t0)
            changeManifold.append(change)
            biasManifold.append(changeBias)
            # add both to the manifold
            delta = newDeltas
            # set up for next layer
        # Apply gradients with momentum + weight decay, then invalidate device caches.
        weight_stage_t0 = time.perf_counter() if self._internal_profiler_enabled else 0.0
        for i in range(self.layers):
            idx = self.layers - i - 1
            if isinstance(self.weightsByLayer[idx], np.ndarray):
                if momentum > 0:
                    self.weight_velocities[idx] = (momentum * self.weight_velocities[idx] + 
                                                changeManifold[i])
                    self.weight_velocities[idx] = self._stabilize_host_array(self.weight_velocities[idx], clip_value=self._grad_clip_guard)
                    # print(learnRate)
                    # print(self.weight_velocities[idx])
                    effective_change = learnRate * self.weight_velocities[idx]
                else:
                    effective_change = learnRate * changeManifold[i]
                effective_change = self._stabilize_host_array(effective_change, clip_value=self._grad_clip_guard)
                
                self.weightsByLayer[idx] = np.add(
                    self.weightsByLayer[idx] * (1 - learnRate * weightDecay),
                    effective_change
                )
                self.weightsByLayer[idx] = self._stabilize_host_array(self.weightsByLayer[idx], clip_value=self._weight_clip_guard)
                self._invalidate_device_cache(idx)
                if self.bias:
                    if momentum > 0:
                        self.bias_velocities[idx] = (momentum * self.bias_velocities[idx] + 
                                                    biasManifold[i])
                        self.bias_velocities[idx] = self._stabilize_host_array(self.bias_velocities[idx], clip_value=self._grad_clip_guard)
                        effective_bias_change = learnRate * self.bias_velocities[idx]
                    else:
                        effective_bias_change = learnRate * biasManifold[i]
                    effective_bias_change = self._stabilize_host_array(effective_bias_change, clip_value=self._grad_clip_guard)
                    
                    self.biasByLayer[idx] = np.add(
                        self.biasByLayer[idx],
                        effective_bias_change
                    )
                    self.biasByLayer[idx] = self._stabilize_host_array(self.biasByLayer[idx], clip_value=self._weight_clip_guard)

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
