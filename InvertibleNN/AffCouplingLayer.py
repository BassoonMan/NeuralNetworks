import numpy as np
import sys
from pathlib import Path

# Add the parent directory to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ArrayBased.ArrayNetwork import ArrayNetworkFeedforward as ArrayNetwork
from Backends import get_backend

def leakyReLU(x):
    if isinstance(x, np.ndarray):
        return np.where(x < 0, 0.01 * x, x)
    return 0.01 * x if x < 0 else x
def leakyReLUDerivative(x):
    if isinstance(x, np.ndarray):
        return np.where(x < 0, 0.01, 1.0)
    return 0.01 if x < 0 else 1.0
def soft_clip(x, limit=2.0):
    return limit * np.tanh(x / limit)
def tanh(x):
    return np.tanh(x)
def tanhDerivative(x):
    if isinstance(x, np.ndarray):
        return 1.0 - np.tanh(x) ** 2
    return 1.0 - np.tanh(x) ** 2
def identity(x):
    return x
def identityDerivative(x):
    if isinstance(x, np.ndarray):
        return np.ones_like(x, dtype=float)
    return 1.0

class AffCouplingLayer:
    def __init__(self, inputLength, internalLayers, internalLayerLength, backend="cpu"):
        """ Initializes the Affine Layer
        Parameters:
        -----------
            inputLength : int
                This is the length of the input vector to the coupling layer. Must be even.
            internalLayers : int
                This is the number of layers in the s* and t* subnetworks.
            internalLayerLength : int
                This is the number of neurons in each internal layer of the s* and t* subnetworks.
            backend : int
                Determines if the code runs on GPU or CPU.
        """
        assert inputLength % 2 == 0, "Input length must be even for coupling layer."
        self.inputLength = inputLength
        self.perm = np.random.permutation(inputLength)
        self._inv_perm = np.argsort(self.perm)
        networkInputLength = inputLength // 2
        self.backend = get_backend(backend)
        self._use_device = str(backend).lower() in ("opencl", "gpu", "amd")
        # s* networks produce scale-like terms; tanh bounded outputs improve stability.
        self.s1 = ArrayNetwork(
            networkInputLength,
            internalLayers, 
            [internalLayerLength] * (internalLayers - 1) + [networkInputLength], 
            [[leakyReLU, leakyReLUDerivative]] * (internalLayers - 1) + [[tanh, tanhDerivative]],
            isBias = True, 
            random=True,
            backend=backend
        )
        self.s2 = ArrayNetwork(
            networkInputLength,
            internalLayers, 
            [internalLayerLength] * (internalLayers - 1) + [networkInputLength], 
            [[leakyReLU, leakyReLUDerivative]] * (internalLayers - 1) + [[tanh, tanhDerivative]],
            isBias = True, 
            random=True,
            backend=backend
        )
        # t* networks produce translation terms; identity output allows full shift range.
        self.t1 = ArrayNetwork(
            networkInputLength,
            internalLayers, 
            [internalLayerLength] * (internalLayers - 1) + [networkInputLength], 
            [[leakyReLU, leakyReLUDerivative]] * (internalLayers - 1) + [[identity, identityDerivative]], 
            isBias = True, 
            random=True,
            backend=backend
        )
        self.t2 = ArrayNetwork(
            networkInputLength,
            internalLayers, 
            [internalLayerLength] * (internalLayers - 1) + [networkInputLength], 
            [[leakyReLU, leakyReLUDerivative]] * (internalLayers - 1) + [[identity, identityDerivative]], 
            isBias = True, 
            random=True,
            backend=backend
        )

        # For OpenCL training, keep cached activations on device to reduce
        # device->host copies during forward cache population.
        if self._use_device:
            for net in (self.s1, self.s2, self.t1, self.t2):
                if hasattr(net, "set_cached_opencl"):
                    net.set_cached_opencl(True, min_batch=8, cache_on_device=True)
    
    # ------------------------------------------------------------------
    # Device-aware helpers
    # ------------------------------------------------------------------
    def _to_dev(self, x):
        """Move to device if using GPU backend, otherwise return as-is."""
        if self._use_device:
            return self.backend.to_device(x, dtype=np.float32)
        return np.asarray(x, dtype=np.float32)

    def _to_host(self, x):
        """Ensure numpy array on host."""
        if self._use_device:
            return self.backend.to_host(x)
        return np.asarray(x)

    def _soft_clip_dev(self, x, limit=2.0):
        """Device-aware soft clip: limit * tanh(x / limit)."""
        if limit <= 0:
            return x
        if self._use_device and self.backend.is_device_array(x):
            scaled = self.backend.divide(x, float(limit))
            tanh_val = self.backend.apply_activation(scaled, "tanh")
            return self.backend.multiply(float(limit), tanh_val)
        return limit * np.tanh(np.asarray(x) / limit)

    def _exp_dev(self, x):
        """Device-aware exp."""
        if self._use_device and self.backend.is_device_array(x):
            try:
                import pyopencl.clmath as clmath
                return clmath.exp(x)
            except Exception:
                return self.backend.to_device(np.exp(self.backend.to_host(x)))
        return np.exp(np.asarray(x))

    def _tanh_dev(self, x):
        """Device-aware tanh."""
        if self._use_device and self.backend.is_device_array(x):
            return self.backend.apply_activation(x, "tanh")
        return np.tanh(np.asarray(x))

    def _multiply(self, a, b):
        """Device-aware elementwise multiply."""
        if self._use_device and (self.backend.is_device_array(a) or self.backend.is_device_array(b)):
            return self.backend.multiply(a, b)
        return np.multiply(a, b)

    def _add(self, a, b):
        """Device-aware elementwise add."""
        if self._use_device and (self.backend.is_device_array(a) or self.backend.is_device_array(b)):
            return self.backend.add(a, b)
        return np.add(a, b)

    def _subtract(self, a, b):
        """Device-aware elementwise subtract."""
        if self._use_device and (self.backend.is_device_array(a) or self.backend.is_device_array(b)):
            return self.backend.subtract(a, b)
        return np.subtract(a, b)

    def _concat_dev(self, a, b):
        """Device-aware horizontal concatenation (axis=1)."""
        # OpenCL doesn't have a native concat; pull to host, concat, re-upload.
        if self._use_device and (self.backend.is_device_array(a) or self.backend.is_device_array(b)):
            a_h = self.backend.to_host(a)
            b_h = self.backend.to_host(b)
            return self.backend.to_device(np.concatenate((a_h, b_h), axis=1), dtype=np.float32)
        return np.concatenate((np.asarray(a), np.asarray(b)), axis=1)

    def _split_dev(self, x):
        """Device-aware split into two halves along axis=1."""
        if self._use_device and self.backend.is_device_array(x):
            x_h = self.backend.to_host(x)
            a, b = np.split(x_h, 2, axis=1)
            return self.backend.to_device(a, dtype=np.float32), self.backend.to_device(b, dtype=np.float32)
        arr = np.asarray(x)
        a, b = np.split(arr, 2, axis=1)
        return a, b

    def _permute_cols(self, x, perm):
        """Device-aware column permutation."""
        if self._use_device and self.backend.is_device_array(x):
            x_h = self.backend.to_host(x)
            return self.backend.to_device(x_h[:, perm], dtype=np.float32)
        return np.asarray(x)[:, perm]

    def _evaluate_network_device(self, net, inputs, cached=True):
        """Run subnet forward and return result on device (if using GPU).
        
        evaluateNetwork always returns host arrays, so re-upload the result
        to avoid a device->host->device round-trip on subsequent math.
        """
        # Ensure inputs are host for evaluateNetwork (it handles its own device path internally)
        inputs_host = self._to_host(inputs)
        result_host = net.evaluateNetwork(inputs_host, cached)
        if self._use_device:
            return self._to_dev(result_host)
        return result_host

    def forward(self, x):
        """ Passes an input through the network in the forward direction. Does not store any intermediate tensors, so this is suitable for inference but not training.
        Parameters:
        -----------
            x : numpy.ndarray
                Input into the coupling layer. Should have shape (batch_size, inputLength).
        """
        assert x.shape[1] == self.inputLength, f"Expected input with {self.inputLength} features, got {x.shape[1]}"
        assert len(x.shape) == 2, f"Expected 2D input (batch_size, inputLength), got shape {x.shape}"
        # Inference forward (no training caches):
        # 1) permute and split
        # 2) v1 = u1 * exp(s2(u2)) + t2(u2)
        # 3) v2 = u2 * exp(s1(v1)) + t1(v1)
        # 4) concat and undo permutation index order
        x_perm = x[:, self.perm] # Randomly permute x along the column axis
        #print("Perm:", x_perm.shape)
        u1, u2 = np.split(x_perm, 2, axis=1) # split into two parts along the column axis
        #print("u1, u2:", u1.shape, u2.shape)
        temp1 = soft_clip(self.s2.evaluateNetwork(u2, cached=False), 2)
        #print("s2:", temp1.shape)
        temp2 = self.t2.evaluateNetwork(u2, cached=False) # Evaluate the first set of networks
        #print("t2:", temp2.shape)
        v1 = u1 * np.exp(temp1) + temp2 # calculate the first output
        #print("v1:", v1.shape)
        temp1 = soft_clip(self.s1.evaluateNetwork(v1, cached=False), 2)
        #print("s1:", temp1.shape)
        temp2 = self.t1.evaluateNetwork(v1, cached=False) # Evaluate the second set of networks
        #print("t1:", temp2.shape)
        v2 = u2 * np.exp(temp1) + temp2 # Calculate the second output
        #print("v2:", v2.shape)
        output = np.concatenate((v1, v2), axis=1) # Fuse back together
        #print("Output before inverse perm:", output.shape)
        #print("Inv Perm:", output[:, self._inv_perm].shape)
        return output[:, self._inv_perm]
    
    def train_forward(self, x):
        # Same transform as forward(), but stores intermediate tensors used by backprop.
        self.forward_input = x
        x_perm = x[:, self.perm] # Randomly permute x
        u1, u2 = np.split(x_perm, 2, axis=1) # split into two parts
        self.forward_x_perm = x_perm  # cache for gradient computation
        temp1 = soft_clip(self.s2.evaluateNetwork(u2), 2)
        temp2 = self.t2.evaluateNetwork(u2) # Evaluate the first set of networks
        v1 = u1 * np.exp(temp1) + temp2 # calculate the first output
        temp1 = soft_clip(self.s1.evaluateNetwork(v1), 2)
        temp2 = self.t1.evaluateNetwork(v1) # Evaluate the second set of networks
        v2 = u2 * np.exp(temp1) + temp2 # Calculate the second output
        self.forward_v1 = v1
        self.forward_v2 = v2
        output = np.concatenate((v1, v2), axis=1) # Fuse back together
        return output[:, self._inv_perm]

    def backward(self, y):
        # Exact inverse of forward().
        # Because affine coupling equations are triangular, inversion is closed-form.
        y = y[:, self.perm]
        v1, v2 = np.split(y, 2, axis=1) # Split
        temp1 = soft_clip(self.s1.evaluateNetwork(v1, cached=False), 2)
        temp2 = self.t1.evaluateNetwork(v1, cached=False) # Evaluate the first set of networks
        u2 = (v2-temp2) * np.exp(-temp1) # Calculate the second input
        temp1 = soft_clip(self.s2.evaluateNetwork(u2, cached=False), 2)
        temp2 = self.t2.evaluateNetwork(u2, cached=False) # Evaluate the second set of networks
        u1 = (v1-temp2) * np.exp(-temp1) # Calculate the first input
        x = np.concatenate((u1, u2), axis=1) # Fuse back together
        return x[:, self._inv_perm] # Invert the permutation from forward

    def train_backward(self, y):
        # Cached inverse path used by frontpropagate variant.
        y = y[:, self.perm]
        v1, v2 = np.split(y, 2, axis=1) # Split
        temp1 = soft_clip(self.s1.evaluateNetwork(v1), 2)
        temp2 = self.t1.evaluateNetwork(v1) # Evaluate the first set of networks
        u2 = (v2-temp2) * np.exp(-temp1) # Calculate the second input
        temp1 = soft_clip(self.s2.evaluateNetwork(u2), 2)
        temp2 = self.t2.evaluateNetwork(u2) # Evaluate the second set of networks
        u1 = (v1-temp2) * np.exp(-temp1) # Calculate the first input
        self.back_u1 = u1
        self.back_u2 = u2
        x = np.concatenate((u1, u2), axis=1) # Fuse back together
        return x[:, self._inv_perm] # Invert the permutation from forward

    def backpropagate(self, x, learningRate, diffs, weightDecay=.001, clip=True, momentum=0.9):
        # Backprop is organized in 3 phases:
        #   1) Build all needed deltas/outer grads from cached forward tensors.
        #   2) Apply updates to s1/t1/s2/t2 subnetworks.
        #   3) Return dL/dInput to previous coupling layer.
        #
        # I have an assumption that x and diffs are on the host, at least for now.

        # Permute and split — avoid bouncing between host/device by just keeping on host until end.
        # x_dev = self._to_dev(x)
        x_perm = x[:, self.perm]
        # x_perm = self._permute_cols(x_dev, self.perm)
        u1_h, u2_h = np.split(x_perm, 2, axis=1)
        # u1, u2 = self._split_dev(x_perm)
        u1 = self._to_dev(u1_h)
        u2 = self._to_dev(u2_h)

        # diffs_dev = self._to_dev(diffs)
        diffs_perm = diffs[:, self.perm]
        # diffs_perm = self._permute_cols(diffs_dev, self.perm)
        diff1_h, diff2_h = np.split(diffs_perm, 2, axis=1)
        # diff1, diff2 = self._split_dev(diffs_perm)
        diff1 = self._to_dev(diff1_h)
        diff2 = self._to_dev(diff2_h)

        if clip:
            diff1 = self._soft_clip_dev(diff1, 1)
            diff2 = self._soft_clip_dev(diff2, 1)

        # ===== PHASE 1: Compute ALL gradients (no weight updates) =====

        # s1 values — evaluate and keep on device
        forward_v1_dev = self._to_dev(self.forward_v1)
        s1_raw = self._evaluate_network_device(self.s1, forward_v1_dev, True)
        s1_clipped = self._soft_clip_dev(s1_raw, 2)
        s1_tanh_input = self.backend.divide(s1_raw, 2.0) if self._use_device and self.backend.is_device_array(s1_raw) else s1_raw / 2.0
        s1_clip_deriv = self._subtract(1.0, self._multiply(self._tanh_dev(s1_tanh_input), self._tanh_dev(s1_tanh_input)))

        # t1 cache — just need the cached activations for backpropDelta
        self._evaluate_network_device(self.t1, forward_v1_dev, True)

        # Indirect path: dL/dv1 via s1 and t1
        exp_s1 = self._exp_dev(s1_clipped)
        s1_outer = self._multiply(self._multiply(self._multiply(diff2, u2), exp_s1), s1_clip_deriv)
        if clip:
            s1_outer = self._soft_clip_dev(s1_outer, 1)

        # backpropDelta expects and returns host arrays, so convert at boundary
        s1_outer_host = self._to_host(s1_outer)
        diff2_host = self._to_host(diff2)
        clip_limit = 1.0 if clip else 0
        dL_dv1_via_s1 = self.s1.backpropDelta(s1_outer_host, clip_limit)
        dL_dv1_via_t1 = self.t1.backpropDelta(diff2_host, clip_limit)

        # Bring back to device
        dL_dv1_via_s1 = self._to_dev(dL_dv1_via_s1)
        dL_dv1_via_t1 = self._to_dev(dL_dv1_via_t1)
        if clip:
            dL_dv1_via_s1 = self._soft_clip_dev(dL_dv1_via_s1, 1)
            dL_dv1_via_t1 = self._soft_clip_dev(dL_dv1_via_t1, 1)

        diff1_total = self._add(self._add(diff1, dL_dv1_via_s1), dL_dv1_via_t1)
        if clip:
            diff1_total = self._soft_clip_dev(diff1_total, 1)

        # s2 values
        s2_raw = self._evaluate_network_device(self.s2, u2, True)
        s2_clipped = self._soft_clip_dev(s2_raw, 2)
        s2_tanh_input = self.backend.divide(s2_raw, 2.0) if self._use_device and self.backend.is_device_array(s2_raw) else s2_raw / 2.0
        s2_clip_deriv = self._subtract(1.0, self._multiply(self._tanh_dev(s2_tanh_input), self._tanh_dev(s2_tanh_input)))

        # Precompute all diffs — all on device
        exp_s1_for_diffs = exp_s1  # reuse from above
        exp_s2 = self._exp_dev(s2_clipped)

        diffs_for_s1 = self._multiply(self._multiply(self._multiply(diff2, u2), exp_s1_for_diffs), s1_clip_deriv)
        diffs_for_t1 = diff2
        diffs_for_s2 = self._multiply(self._multiply(self._multiply(diff1_total, u1), exp_s2), s2_clip_deriv)
        diffs_for_t2 = diff1_total

        if clip:
            diffs_for_s1 = self._soft_clip_dev(diffs_for_s1, 1)
            diffs_for_t1 = self._soft_clip_dev(diffs_for_t1, 0.5)
            diffs_for_s2 = self._soft_clip_dev(diffs_for_s2, 1)
            diffs_for_t2 = self._soft_clip_dev(diffs_for_t2, 0.5)

        # ===== PHASE 2: Update ALL weights =====
        # updateNetworkGeneralizedDelta handles its own device path internally.
        # Pass device arrays directly — it will call to_device (no-op) or to_host as needed.
        forward_v1_host = self._to_host(forward_v1_dev)
        u2_host = self._to_host(u2)
        diffs_for_s1_host = self._to_host(diffs_for_s1)
        diffs_for_t1_host = self._to_host(diffs_for_t1)
        diffs_for_s2_host = self._to_host(diffs_for_s2)
        diffs_for_t2_host = self._to_host(diffs_for_t2)

        self.s1.updateNetworkGeneralizedDelta(forward_v1_host, diffs_for_s1_host, learningRate, weightDecay, momentum)
        self.t1.updateNetworkGeneralizedDelta(forward_v1_host, diffs_for_t1_host, learningRate, weightDecay, momentum)
        self.s2.updateNetworkGeneralizedDelta(u2_host, diffs_for_s2_host, learningRate, weightDecay, momentum)
        self.t2.evaluateNetwork(u2_host, True)  # set cache before update
        self.t2.updateNetworkGeneralizedDelta(u2_host, diffs_for_t2_host, learningRate, weightDecay, momentum)

        # ===== PHASE 3: Input grads for multi-layer =====
        du1 = self._multiply(diff1_total, exp_s2)
        du2 = self._multiply(diff2, exp_s1)
        input_diffs = self._concat_dev(du1, du2)
        if clip:
            input_diffs = self._soft_clip_dev(input_diffs, 1)

        # Return host array with inverse permutation applied
        input_diffs_host = self._to_host(input_diffs)
        return input_diffs_host[:, self._inv_perm]
        
    def frontpropagate(self, y, learningRate, diffs, weightDecay = .001):
        # Alternative direction update path used in some experiments.
        # diffs: [(u1-target), (u2-target)]
        y = y[:, self.perm]
        v1, v2 = np.split(y, 2, axis=1) # Split
        diffs_perm = diffs[:, self.perm]
        diff1, diff2 = np.split(diffs_perm, 2, axis=1)
        diff1 = soft_clip(diff1, 1)
        diff2 = soft_clip(diff2, 1)

        s1_raw = self.s1.evaluateNetwork(v1, True)
        s1_clipped = soft_clip(s1_raw, 2)
        s1_clip_deriv = 1 - np.tanh(s1_raw / 2.0) ** 2

        diffs_for_s1 = soft_clip(-diff2 * self.back_u2 * s1_clip_deriv, 1)
        self.s1.updateNetworkGeneralizedDelta(v1, diffs_for_s1, learningRate, weightDecay)

        self.t1.evaluateNetwork(v1, True)
        diffs_for_t1 = soft_clip(-diff2 * np.exp(-s1_clipped), .5)
        self.t1.updateNetworkGeneralizedDelta(v1, diffs_for_t1, learningRate, weightDecay)

        s2_raw = self.s2.evaluateNetwork(self.back_u2, True)
        s2_clipped = soft_clip(s2_raw, 2)
        s2_clip_deriv = 1 - np.tanh(s2_raw / 2.0) ** 2

        diffs_for_s2 = soft_clip(-diff1 * self.back_u1 * s2_clip_deriv, 1)
        self.s2.updateNetworkGeneralizedDelta(self.back_u2, diffs_for_s2, learningRate, weightDecay)

        self.t2.evaluateNetwork(self.back_u2, True)
        diffs_for_t2 = soft_clip(-diff1 * np.exp(-s2_clipped), 0.5)
        self.t2.updateNetworkGeneralizedDelta(self.back_u2, diffs_for_t2, learningRate, weightDecay)
