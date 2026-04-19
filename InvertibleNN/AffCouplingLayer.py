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

def split_tanh_identity(x):
    """First half of columns: tanh, second half: identity."""
    out = x.copy() if isinstance(x, np.ndarray) else np.array(x, dtype=np.float32)
    half = out.shape[1] // 2
    out[:, :half] = np.tanh(out[:, :half])
    return out

def split_tanh_identity_deriv(x):
    out = np.ones_like(x, dtype=np.float32)
    half = out.shape[1] // 2
    t = np.tanh(x[:, :half])
    out[:, :half] = 1.0 - t * t
    return out

class AffCouplingLayer:
    def __init__(self, inputLength, internalLayers, internalLayerLength, backend="cpu", batch_size=128):
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
        self.backend = get_backend(backend, batch_size=batch_size, vector_size=inputLength, internal_width=internalLayerLength)
        self._use_device = str(backend).lower() in ("opencl", "gpu", "amd")
        if self._use_device:
            self.perm = self.backend.to_device(self.perm, dtype=np.int32)
            self._inv_perm = self.backend.to_device(self._inv_perm, dtype=np.int32)
        self.st1 = ArrayNetwork(
            networkInputLength,
            internalLayers,
            [internalLayerLength * 2] * (internalLayers - 1) + [networkInputLength * 2],
            [[leakyReLU, leakyReLUDerivative]] * (internalLayers - 1) 
                + [[split_tanh_identity, split_tanh_identity_deriv]],
            isBias=True,
            random=True,
            backend=backend,
            batch_size=batch_size,
        )
        self.st2 = ArrayNetwork(
            networkInputLength,
            internalLayers,
            [internalLayerLength * 2] * (internalLayers - 1) + [networkInputLength * 2],
            [[leakyReLU, leakyReLUDerivative]] * (internalLayers - 1)
                + [[split_tanh_identity, split_tanh_identity_deriv]],
            isBias=True,
            random=True,
            backend=backend,
            batch_size=batch_size,
        )
        self.limit = 1.5

        # For OpenCL training, keep cached activations on device to reduce
        # device->host copies during forward cache population.
        if self._use_device:
            for net in (self.st1, self.st2):#(self.s1, self.s2, self.t1, self.t2):
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
        if self._use_device:
            return self.backend.soft_clip(x, limit)
        return limit * np.tanh(np.asarray(x) / limit)

    def _add(self, a, b):
        """Device-aware elementwise add."""
        if self._use_device:
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
        u1, u2 = self.backend.permute_split(x, self.perm)

            # Stage 1: st2 on u2
        st2_raw = self.st2.evaluateNetwork(u2, False)  # (B, 2*half)
        v1 = self.backend.coupling_forward_merged(u1, st2_raw, self.limit)

        # Stage 2: st1 on v1
        st1_raw = self.st1.evaluateNetwork(v1, False)  # (B, 2*half)
        return self.backend.fused_forward_invperm_merged(v1, u2, self._inv_perm, st1_raw, self.limit)

        # s2_raw = self.s2.evaluateNetwork(u2, False)
        # t2_raw = self.t2.evaluateNetwork(u2, False)
        # v1 = self.backend.coupling_forward(u1, s2_raw, t2_raw, 2.0)

        # s1_raw = self.s1.evaluateNetwork(v1, False)
        # t1_raw = self.t1.evaluateNetwork(v1, False)
        # return self.backend.fused_forward_invperm(v1, u2, self._inv_perm, s1_raw, t1_raw, 2.0)

    
    def train_forward(self, x):
        # Same transform as forward(), but stores intermediate tensors used by backprop.
        self.forward_input = x

        u1, u2 = self.backend.permute_split(x, self.perm)

        st2_raw = self.st2.evaluateNetwork(u2, True)   # cached
        v1 = self.backend.coupling_forward_merged(u1, st2_raw, self.limit)
        self.forward_v1 = v1

        st1_raw = self.st1.evaluateNetwork(v1, True)   # cached
        self.cached_st1_raw = st1_raw
        self.cached_st2_raw = st2_raw
        return self.backend.fused_forward_invperm_merged(v1, u2, self._inv_perm, st1_raw, self.limit)

        # s2_raw = self.s2.evaluateNetwork(u2)
        # t2_raw = self.t2.evaluateNetwork(u2)
        # v1 = self.backend.coupling_forward(u1, s2_raw, t2_raw, 2.0)
        # self.forward_v1 = v1

        # s1_raw = self.s1.evaluateNetwork(v1)
        # t1_raw = self.t1.evaluateNetwork(v1)
        # return self.backend.fused_forward_invperm(v1, u2, self._inv_perm, s1_raw, t1_raw, 2.0)


    def backward(self, y):
        # Exact inverse of forward().
        # Because affine coupling equations are triangular, inversion is closed-form.
        v1, v2 = self.backend.permute_split(y, self.perm)

        st1_raw = self.st1.evaluateNetwork(v1, False)
        u2 = self.backend.coupling_backward_merged(v2, st1_raw, self.limit)

        st2_raw = self.st2.evaluateNetwork(u2, False)
        u1 = self.backend.coupling_backward_merged(v1, st2_raw, self.limit)
        return self.backend.concat_invperm(u1, u2, self._inv_perm)

    def backpropagate(self, x, learningRate, diffs, weightDecay=0.0, clip=True, momentum=0.9):
        # Backprop is organized in 3 phases:
        #   1) Build all needed deltas/outer grads from cached forward tensors.
        #   2) Apply updates to s1/t1/s2/t2 subnetworks.
        #   3) Return dL/dInput to previous coupling layer.
        #
        # I have an assumption that x and diffs are on the host, at least for now.

        # Permute and split — avoid bouncing between host/device by just keeping on host until end.
        backend = self.backend
        permute_split = backend.permute_split
        coupling_s_outer_concat = backend.coupling_s_outer_concat
        add2_clip = backend.add2_clip
        fused_coupling_grads_concat = backend.fused_coupling_grads_concat
        coupling_input_grad_merged = backend.coupling_input_grad_merged
        concat_invperm = backend.concat_invperm


        u1, u2 = permute_split(x, self.perm)

        diff1, diff2 = permute_split(diffs, self.perm)#np.split(diffs_perm, 2, axis=1)

        if clip:
            diff1 = self._soft_clip_dev(diff1, 1)
            diff2 = self._soft_clip_dev(diff2, 1)

        # ===== PHASE 1: Compute ALL gradients (no weight updates) =====

        # st1_raw = self.st1.evaluateNetwork(self.forward_v1, True)  # (B, 2*half), cached
        st1_raw = self.cached_st1_raw
        st2_raw = self.cached_st2_raw

        # half = st1_raw.shape[1] // 2
        clip_limit = 1.0 if clip else 0.0

        # s-portion outer gradient (uses first half of st1_raw as s1_raw)
        combined_outer_st1, s1_outer = coupling_s_outer_concat(diff2, u2, st1_raw, s_limit=self.limit, clip_limit=clip_limit)

        # Single backpropDelta call replaces two
        dL_dv1 = self.st1.backpropDelta(combined_outer_st1, clip_limit)

        diff1_total = add2_clip(diff1, dL_dv1, 1.0 if clip else 0.0)

        t_clip = 0.5 if clip else 0.0
        diffs_st1, diffs_st2 = fused_coupling_grads_concat(
            s1_outer, diff2, diff1_total, u1, st2_raw,
            s_limit=self.limit, s_clip_limit=clip_limit, t_clip_limit=t_clip)

        # ===== PHASE 2: Update ALL weights =====
        self.st1.updateNetworkGeneralizedDelta(self.forward_v1, diffs_st1, learningRate, weightDecay, momentum)
        self.st2.updateNetworkGeneralizedDelta(u2, diffs_st2, learningRate, weightDecay, momentum)

        # ===== PHASE 3: Input grads for multi-layer =====
        du1 = coupling_input_grad_merged(diff1_total, st2_raw, self.limit)
        du2 = coupling_input_grad_merged(diff2, st1_raw, self.limit)
        return concat_invperm(du1, du2, self._inv_perm)
