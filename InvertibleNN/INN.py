import numpy as np
from AffCouplingLayer import AffCouplingLayer
import time
import random
import matplotlib.pyplot as plt

class InvertibleNeuralNetwork:
    def __init__(self, layers, in_out_length, internalLayers=2, internalLayerLength=8):
        self.layers = []
        for i in range(layers):
            self.layers.append(AffCouplingLayer(in_out_length, internalLayers, internalLayerLength))
            
    
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, y):
        for layer in reversed(self.layers):
            y = layer.backward(y)
        return y
    
    def train_forward(self, x):
        for layer in self.layers:
            x = layer.train_forward(x)
        return x

    def train_backward(self, y):
        for layer in reversed(self.layers):
            y = layer.train_backward(y)
        return y
    
    def backpropagate(self, diffs, learningRate, momentum=0.9):
        current_diffs = diffs
        for layer in reversed(self.layers):
            input_diffs = layer.backpropagate(layer.forward_input, learningRate, current_diffs, momentum=momentum)
            current_diffs = input_diffs

    def frontpropagate(self, output, diffs, learningRate):
        for layer in self.layers:
            layer.frontpropagate(output, learningRate, diffs)


if __name__ == "__main__":
    testSet3 = []
    pairs = [
        ([0, 0, 0, 0], [0.1, 0.1, 0.1, 0.1]),
        ([1, 0, 0, 0], [0.9, 0.1, 0.1, 0.1]),
        ([0, 1, 0, 0], [0.1, 0.1, 0.9, 0.1]),
        ([0, 0, 1, 0], [0.1, 0.1, 0.1, 0.9]),
        ([0, 0, 0, 1], [0.1, 0.9, 0.1, 0.1]),
        ([1, 1, 0, 0], [0.9, 0.1, 0.9, 0.1]),
        ([1, 0, 1, 0], [0.9, 0.1, 0.1, 0.9]),
        ([1, 0, 0, 1], [0.9, 0.9, 0.1, 0.1]),
        ([0, 1, 1, 0], [0.1, 0.1, 0.9, 0.9]),
        ([0, 1, 0, 1], [0.1, 0.9, 0.9, 0.1]),
        ([0, 0, 1, 1], [0.1, 0.9, 0.1, 0.9]),
        ([1, 1, 1, 0], [0.9, 0.1, 0.9, 0.9]),
        ([1, 1, 0, 1], [0.9, 0.9, 0.9, 0.1]),
        ([1, 0, 1, 1], [0.9, 0.9, 0.1, 0.9]),
        ([0, 1, 1, 1], [0.1, 0.9, 0.9, 0.9]),
        ([1, 1, 1, 1], [0.9, 0.9, 0.9, 0.9]),
    ]

    for inp, out in pairs:
        testSet3.append([np.atleast_2d(inp), np.atleast_2d(out)])
    inn = InvertibleNeuralNetwork(5, 4, 2, 16)
    # Numerical gradient check - add this before training loop
    def numerical_gradient_check(inn, inputs, targets):
        eps = 1e-5
        
        # Save ALL weights across ALL layers
        all_saved = {}
        for layer_idx, layer in enumerate(inn.layers):
            networks = [('s1', layer.s1), ('s2', layer.s2), 
                        ('t1', layer.t1), ('t2', layer.t2)]
            all_saved[layer_idx] = {}
            for name, net in networks:
                all_saved[layer_idx][name] = {
                    'net': net,
                    'weights': [w.copy() if isinstance(w, np.ndarray) else w for w in net.weightsByLayer],
                    'biases': [b.copy() for b in net.biasByLayer] if net.bias else []
                }
        
        # Compute analytical gradients
        calc_out = inn.train_forward(inputs)
        diffs = targets - calc_out
        
        # Save weights again (after train_forward, before backprop)
        for layer_idx, layer in enumerate(inn.layers):
            networks = [('s1', layer.s1), ('s2', layer.s2), 
                        ('t1', layer.t1), ('t2', layer.t2)]
            for name, net in networks:
                all_saved[layer_idx][name]['weights'] = [
                    w.copy() if isinstance(w, np.ndarray) else w for w in net.weightsByLayer
                ]
                if net.bias:
                    all_saved[layer_idx][name]['biases'] = [b.copy() for b in net.biasByLayer]
        
        # Run full backprop with lr=1, no decay, no clip
        # Need to temporarily modify all layers
        current_diffs = diffs
        for layer in reversed(inn.layers):
            input_diffs = layer.backpropagate(layer.forward_input, 1.0, current_diffs, weightDecay=0, clip=False, momentum=0)
            current_diffs = input_diffs
        
        # Capture analytical gradients for ALL layers
        all_analytical = {}
        for layer_idx, layer in enumerate(inn.layers):
            networks = [('s1', layer.s1), ('s2', layer.s2), 
                        ('t1', layer.t1), ('t2', layer.t2)]
            all_analytical[layer_idx] = {}
            for name, net in networks:
                all_analytical[layer_idx][name] = []
                for li in range(net.layers):
                    if isinstance(net.weightsByLayer[li], np.ndarray):
                        grad = net.weightsByLayer[li] - all_saved[layer_idx][name]['weights'][li]
                        all_analytical[layer_idx][name].append(grad)
                    else:
                        all_analytical[layer_idx][name].append(None)
        
        # Restore ALL weights
        for layer_idx, layer in enumerate(inn.layers):
            networks = [('s1', layer.s1), ('s2', layer.s2), 
                        ('t1', layer.t1), ('t2', layer.t2)]
            for name, net in networks:
                for li in range(len(all_saved[layer_idx][name]['weights'])):
                    if isinstance(all_saved[layer_idx][name]['weights'][li], np.ndarray):
                        net.weightsByLayer[li] = all_saved[layer_idx][name]['weights'][li].copy()
                    else:
                        net.weightsByLayer[li] = all_saved[layer_idx][name]['weights'][li]
                if net.bias:
                    for li in range(len(all_saved[layer_idx][name]['biases'])):
                        net.biasByLayer[li] = all_saved[layer_idx][name]['biases'][li].copy()
        
        # Numerical gradients for ALL layers
        total = 0
        wrong = 0
        for layer_idx, layer in enumerate(inn.layers):
            networks = [('s1', layer.s1), ('s2', layer.s2), 
                        ('t1', layer.t1), ('t2', layer.t2)]
            for name, net in networks:
                for li in range(net.layers):
                    if not isinstance(net.weightsByLayer[li], np.ndarray):
                        continue
                    for r in range(net.weightsByLayer[li].shape[0]):
                        for c in range(net.weightsByLayer[li].shape[1]):
                            original = net.weightsByLayer[li][r, c].copy()
                            
                            net.weightsByLayer[li][r, c] = original + eps
                            loss_plus = 0.5 * np.sum((targets - inn.forward(inputs)) ** 2)
                            
                            net.weightsByLayer[li][r, c] = original - eps
                            loss_minus = 0.5 * np.sum((targets - inn.forward(inputs)) ** 2)
                            
                            net.weightsByLayer[li][r, c] = original
                            
                            num_grad = (loss_plus - loss_minus) / (2 * eps)
                            ana_grad = all_analytical[layer_idx][name][li][r, c]
                            
                            total += 1
                            denom = max(abs(num_grad), abs(ana_grad), 1e-8)
                            rel_err = abs(num_grad + ana_grad) / denom
                            
                            if rel_err > 0.05:
                                wrong += 1
                                print(f"❌ Layer{layer_idx} {name} L{li}[{r},{c}]: "
                                    f"num={num_grad:+.6f} ana={ana_grad:+.6f} rel_err={rel_err:.4f}")
                            else:
                                print(f"✅ Layer{layer_idx} {name} L{li}[{r},{c}]: "
                                      f"num={num_grad:+.6f} ana={ana_grad:+.6f}")
        
        print(f"\n{total - wrong}/{total} gradients correct")

    test_input = testSet3[15][0]
    test_target = testSet3[15][1]
    #numerical_gradient_check(inn, test_input, test_target)

    # num = 100000
    initial_lr = 0.005
    min_lr = 0.0002
    output_error_track = []
    startTime = time.perf_counter()
    learnRate = 0

    num_epochs = 10000
    total_iterations = num_epochs * len(testSet3)
    samples_per_epoch = len(testSet3)

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

    for epoch in range(num_epochs):
        # Shuffle order each epoch
        indices = list(range(len(testSet3)))
        random.shuffle(indices)
        
        epoch_error = 0
        for idx in indices:
            iteration = epoch * len(testSet3) + idx
            learnRate = min_lr + 0.5 * (initial_lr - min_lr) * (
                1 + math.cos(math.pi * iteration / total_iterations))
            
            current_momentum = min(0.9, 0.5 + 0.4 * (iteration / total_iterations))
            
            inputs = testSet3[idx][0]
            outputs = testSet3[idx][1]
            
            calc_outputs = inn.train_forward(inputs)
            diffs = outputs - calc_outputs
            inn.backpropagate(diffs, learnRate)
            epoch_error += 0.5 * np.sum(diffs ** 2)
        
        epoch_error /= samples_per_epoch
        output_error_track.append(epoch_error)
        
        if (epoch + 1) % 200 == 0:
            print(f"Epoch {epoch+1}, Error: {epoch_error:.6f}, LR: {learnRate:.6f}")

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

