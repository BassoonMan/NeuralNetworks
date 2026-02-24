import numpy as np
import sys
from pathlib import Path

# Add the parent directory to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ArrayBased.ArrayNetwork import ArrayNetworkFeedforward as ArrayNetwork

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
        # Each coupling layer applies a learned permutation then splits channels
        # into two halves (u1, u2). One half conditions transforms applied to the other.
        # This structure is invertible by construction.
        self.inputLength = inputLength
        self.perm = np.random.permutation(inputLength)
        networkInputLength = inputLength // 2
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
        if str(backend).lower() in ("opencl", "gpu", "amd"):
            for net in (self.s1, self.s2, self.t1, self.t2):
                if hasattr(net, "set_cached_opencl"):
                    net.set_cached_opencl(True, min_batch=8, cache_on_device=True)
    
    def forward(self, x):
        # Inference forward (no training caches):
        # 1) permute and split
        # 2) v1 = u1 * exp(s2(u2)) + t2(u2)
        # 3) v2 = u2 * exp(s1(v1)) + t1(v1)
        # 4) concat and undo permutation index order
        x_perm = x[:, self.perm] # Randomly permute x along the column axis
        u1, u2 = np.split(x_perm, 2, axis=1) # split into two parts along the column axis
        temp1 = soft_clip(self.s2.evaluateNetwork(u2, cached=False), 2)
        temp2 = self.t2.evaluateNetwork(u2, cached=False) # Evaluate the first set of networks
        v1 = u1 * np.exp(temp1) + temp2 # calculate the first output
        temp1 = soft_clip(self.s1.evaluateNetwork(v1, cached=False), 2)
        temp2 = self.t1.evaluateNetwork(v1, cached=False) # Evaluate the second set of networks
        v2 = u2 * np.exp(temp1) + temp2 # Calculate the second output
        output = np.concatenate((v1, v2), axis=1) # Fuse back together
        return output[:, np.argsort(self.perm)]
    
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
        return output[:, np.argsort(self.perm)]

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
        return x[:, np.argsort(self.perm)] # Invert the permutation from forward
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
        return x[:, np.argsort(self.perm)] # Invert the permutation from forward

    def backpropagate(self, x, learningRate, diffs, weightDecay = .001, clip = True, momentum=0.9):
        # Backprop is organized in 3 phases:
        #   1) Build all needed deltas/outer grads from cached forward tensors.
        #   2) Apply updates to s1/t1/s2/t2 subnetworks.
        #   3) Return dL/dInput to previous coupling layer.
        x_perm = x[:, self.perm] # Randomly permute x
        u1, u2 = np.split(x_perm, 2, axis=1) # split into two parts
        diffs_perm = diffs[:, self.perm]
        diff1, diff2 = np.split(diffs_perm, 2, axis=1)
        if clip:
            diff1 = soft_clip(diff1, 1)
            diff2 = soft_clip(diff2, 1)

        # ===== PHASE 1: Compute ALL gradients (no weight updates) =====

        # s1 values
        s1_raw = self.s1.evaluateNetwork(self.forward_v1, True)
        s1_clipped = soft_clip(s1_raw, 2)
        s1_clip_deriv = 1 - np.tanh(s1_raw / 2.0) ** 2

        # t1 cache
        self.t1.evaluateNetwork(self.forward_v1, True)

        # Indirect path
        s1_outer = diff2 * u2 * np.exp(s1_clipped) * s1_clip_deriv
        if clip:
            s1_outer = soft_clip(s1_outer, 1)

        clip_limit = 1.0 if clip else 0
        dL_dv1_via_s1 = self.s1.backpropDelta(s1_outer, clip_limit)
        dL_dv1_via_t1 = self.t1.backpropDelta(diff2, clip_limit)
        if clip:
            dL_dv1_via_s1 = soft_clip(dL_dv1_via_s1, 1)
            dL_dv1_via_t1 = soft_clip(dL_dv1_via_t1, 1)

        diff1_total = diff1 + dL_dv1_via_s1 + dL_dv1_via_t1
        if clip:
            diff1_total = soft_clip(diff1_total, 1)

        # s2 values
        s2_raw = self.s2.evaluateNetwork(u2, True)
        s2_clipped = soft_clip(s2_raw, 2)
        s2_clip_deriv = 1 - np.tanh(s2_raw / 2.0) ** 2

        # Precompute all diffs
        diffs_for_s1 = diff2 * u2 * np.exp(s1_clipped) * s1_clip_deriv
        diffs_for_t1 = diff2
        diffs_for_s2 = diff1_total * u1 * np.exp(s2_clipped) * s2_clip_deriv
        diffs_for_t2 = diff1_total

        if clip:
            diffs_for_s1 = soft_clip(diffs_for_s1, 1)
            diffs_for_t1 = soft_clip(diffs_for_t1, 0.5)
            diffs_for_s2 = soft_clip(diffs_for_s2, 1)
            diffs_for_t2 = soft_clip(diffs_for_t2, 0.5)
        # ===== PHASE 2: Update ALL weights =====
        self.s1.updateNetworkGeneralizedDelta(self.forward_v1, diffs_for_s1, learningRate, weightDecay, momentum)
        self.t1.updateNetworkGeneralizedDelta(self.forward_v1, diffs_for_t1, learningRate, weightDecay, momentum)
        self.s2.updateNetworkGeneralizedDelta(u2, diffs_for_s2, learningRate, weightDecay, momentum)
        self.t2.evaluateNetwork(u2, True)  # set cache before update
        self.t2.updateNetworkGeneralizedDelta(u2, diffs_for_t2, learningRate, weightDecay, momentum)

        # ===== PHASE 3: Input grads for multi-layer =====
        du1 = diff1_total * np.exp(s2_clipped)
        du2 = diff2 * np.exp(s1_clipped)
        input_diffs = np.concatenate((du1, du2), axis=1)
        if clip:
            input_diffs = soft_clip(input_diffs, 1)
        return input_diffs[:, np.argsort(self.perm)]
        
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
        self.s2.updateNetworkGeneralizedDelta(self.back_u2, diffs_for_s2, learningRate, weightDecay) # was forward_v1, but Ai corrected with back_u2

        self.t2.evaluateNetwork(self.back_u2, True)
        diffs_for_t2 = soft_clip(-diff1 * np.exp(-s2_clipped), 0.5)
        self.t2.updateNetworkGeneralizedDelta(self.back_u2, diffs_for_t2, learningRate, weightDecay) # AI corrected with moving negative outside network into exp

        #y = y[:, np.argsort(self.perm)] # Invert the permutation from forward
