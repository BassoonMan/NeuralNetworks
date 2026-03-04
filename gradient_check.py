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
