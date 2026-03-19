import numpy as np
import sys
from pathlib import Path

# Add the parent directory to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ArrayBased.ArrayNetwork import ArrayNetworkFeedforward as ArrayNetwork
from Backends import get_backend

def leakyReLU(x):
    return np.where(x < 0, 0.01 * x, x)

def leakyReLUDerivative(x):
    return np.where(x < 0, 0.01, 1.0)

def soft_clip(x, limit=2.0):
    return limit * np.tanh(x / limit)

def tanh(x):
    return np.tanh(x)

def tanhDerivative(x):
    return 1.0 - np.tanh(x) ** 2

def identity(x):
    return x

def identityDerivative(x):
    return np.ones_like(x, dtype=float)

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
        if self._use_device and hasattr(self.backend, 'soft_clip') and self.backend.is_device_array(x):
            return self.backend.soft_clip(x, limit)
        if self._use_device and self.backend.is_device_array(x):
            scaled = self.backend.divide(x, float(limit))
            tanh_val = self.backend.apply_activation(scaled, "tanh")
            return self.backend.multiply(float(limit), tanh_val)
        return limit * np.tanh(np.asarray(x) / limit)

    def _add(self, a, b):
        """Device-aware elementwise add."""
        if self._use_device and (self.backend.is_device_array(a) or self.backend.is_device_array(b)):
            return self.backend.add(a, b)
        return np.add(a, b)

    def forward(self, x):
        """ Passes an input through the network in the forward direction. Does not store any intermediate tensors, so this is suitable for inference but not training.
        Parameters:
        -----------
            x : numpy.ndarray
                Input into the coupling layer. Should have shape (batch_size, inputLength).
        """
        assert x.shape[1] == self.inputLength, f"Expected input with {self.inputLength} features, got {x.shape[1]}"
        assert len(x.shape) == 2, f"Expected 2D input (batch_size, inputLength), got shape {x.shape}"
        # Same transform as forward(), but stores intermediate tensors used by backprop.
        self.forward_input = x
        x_perm = x[:, self.perm] # Randomly permute x
        u1, u2 = np.split(x_perm, 2, axis=1) # split into two parts
        u1 = self._to_dev(u1)
        u2 = self._to_dev(u2)
        s2_raw = self.s2.evaluateNetwork(u2, False)
        t2_raw = self.t2.evaluateNetwork(u2, False)
        v1 = self.backend.coupling_forward(u1, s2_raw, t2_raw, 2.0)

        s1_raw = self.s1.evaluateNetwork(v1, False)
        t1_raw = self.t1.evaluateNetwork(v1, False)
        v2 = self.backend.coupling_forward(u2, s1_raw, t1_raw, 2.0)
         # Calculate the second output
        v1 = self._to_host(v1)
        v2 = self._to_host(v2)
        output = np.concatenate((v1, v2), axis=1) # Fuse back together
        return output[:, self._inv_perm]
    
    def train_forward(self, x):
        # Same transform as forward(), but stores intermediate tensors used by backprop.
        self.forward_input = x
        x_perm = x[:, self.perm] # Randomly permute x
        u1, u2 = np.split(x_perm, 2, axis=1) # split into two parts
        u1 = self._to_dev(u1)
        u2 = self._to_dev(u2)
        s2_raw = self.s2.evaluateNetwork(u2)
        t2_raw = self.t2.evaluateNetwork(u2)
        v1 = self.backend.coupling_forward(u1, s2_raw, t2_raw, 2.0)

        s1_raw = self.s1.evaluateNetwork(v1)
        t1_raw = self.t1.evaluateNetwork(v1)
        v2 = self.backend.coupling_forward(u2, s1_raw, t1_raw, 2.0)
         # Calculate the second output
        v1 = self._to_host(v1)
        v2 = self._to_host(v2)
        self.forward_v1 = v1
        self.forward_v2 = v2
        output = np.concatenate((v1, v2), axis=1) # Fuse back together
        return output[:, self._inv_perm]

    def backward(self, y):
        # Exact inverse of forward().
        # Because affine coupling equations are triangular, inversion is closed-form.
        y = y[:, self.perm]
        v1, v2 = np.split(y, 2, axis=1) # Split
        v1 = self._to_dev(v1)
        v2 = self._to_dev(v2)
        s1_raw = self.s1.evaluateNetwork(v1, False)
        t1_raw = self.t1.evaluateNetwork(v1, False)
        u2 = self.backend.coupling_backward(v2, s1_raw, t1_raw, 2.0)

        s2_raw = self.s2.evaluateNetwork(u2, False)
        t2_raw = self.t2.evaluateNetwork(u2, False)
        u1 = self.backend.coupling_backward(v1, s2_raw, t2_raw, 2.0)
        u1 = self._to_host(u1)
        u2 = self._to_host(u2)
        x = np.concatenate((u1, u2), axis=1) # Fuse back together
        return x[:, self._inv_perm] # Invert the permutation from forward

    def train_backward(self, y):
        # Cached inverse path used by frontpropagate variant.
        y = y[:, self.perm]
        v1, v2 = np.split(y, 2, axis=1) # Split
        s1_raw = self.s1.evaluateNetwork(v1)
        t1_raw = self.t1.evaluateNetwork(v1)
        u2 = self.backend.coupling_backward(v2, s1_raw, t1_raw, 2.0)

        s2_raw = self.s2.evaluateNetwork(u2)
        t2_raw = self.t2.evaluateNetwork(u2)
        u1 = self.backend.coupling_backward(v1, s2_raw, t2_raw, 2.0)
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
        s1_raw = self.s1.evaluateNetwork(forward_v1_dev, True)
        clip_limit = 1.0 if clip else 0.0
        s1_outer = self.backend.coupling_s_outer_grad(diff2, u2, s1_raw, s_limit=2.0, clip_limit=clip_limit)
        dL_dv1_via_s1 = self.s1.backpropDelta(s1_outer, clip_limit)
        dL_dv1_via_t1 = self.t1.backpropDelta(diff2, clip_limit)

        diff1_total = self._add(self._add(diff1, dL_dv1_via_s1), dL_dv1_via_t1)
        if clip:
            diff1_total = self._soft_clip_dev(diff1_total, 1)

        # s2 values
        s2_raw = self.s2.evaluateNetwork(u2, True)

        diffs_for_s1 = s1_outer
        diffs_for_t1 = self._soft_clip_dev(diff2, 0.5) if clip else diff2
        diffs_for_s2 = self.backend.coupling_s_outer_grad(diff1_total, u1, s2_raw, s_limit=2.0, clip_limit=clip_limit)
        diffs_for_t2 = self._soft_clip_dev(diff1_total, 0.5) if clip else diff1_total

        # ===== PHASE 2: Update ALL weights =====

        self.s1.updateNetworkGeneralizedDelta(forward_v1_dev, diffs_for_s1, learningRate, weightDecay, momentum)
        self.t1.updateNetworkGeneralizedDelta(forward_v1_dev, diffs_for_t1, learningRate, weightDecay, momentum)
        self.s2.updateNetworkGeneralizedDelta(u2, diffs_for_s2, learningRate, weightDecay, momentum)
        self.t2.updateNetworkGeneralizedDelta(u2, diffs_for_t2, learningRate, weightDecay, momentum)

        # ===== PHASE 3: Input grads for multi-layer =====
        du1 = self.backend.coupling_input_grad(diff1_total, s2_raw, 2.0)
        du2 = self.backend.coupling_input_grad(diff2, s1_raw, 2.0)
        u1 = self._to_host(du1)
        u2 = self._to_host(du2)
        input_diffs = np.concatenate((np.asarray(u1), np.asarray(u2)), axis=1)
        if clip:
            input_diffs = soft_clip(input_diffs, 1)

        # Return host array with inverse permutation applied
        return input_diffs[:, self._inv_perm]
        
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
