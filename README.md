# Neural Network Visualizer — From Scratch in JavaScript

A fully interactive neural network trained **live in the browser** with **zero libraries** (except Chart.js for the loss curve). Forward propagation, backpropagation, and gradient descent are implemented from raw math — no TensorFlow, no PyTorch.

![HTML](https://img.shields.io/badge/HTML-5-orange) ![JS](https://img.shields.io/badge/JavaScript-ES6-yellow) ![Chart.js](https://img.shields.io/badge/Chart.js-4.4.1-red) ![ML](https://img.shields.io/badge/ML-From_Scratch-purple)

---

## What makes this IIT-level

- Implements **backpropagation from scratch** using the chain rule — no autograd, no library
- Solves **XOR** (the classic proof that shallow networks fail non-linear problems)
- Node colours show **live activation values** — you can watch the network learn in real time
- Swappable **activation functions** (ReLU / Sigmoid / Tanh) so you can observe the effect
- **Math explainer panel** shows the exact forward/backward pass equations at each epoch

---

## Features

| Feature | Description |
|---|---|
| Live network graph | Nodes glow brighter as activation increases; edges coloured by weight sign |
| Loss + accuracy chart | Real-time Chart.js line chart plotting BCE loss and accuracy |
| Truth table | Shows raw and rounded predictions for all 4 XOR inputs each epoch |
| Hyperparameter controls | Learning rate, training speed, and activation function — all live |
| Math explainer | Shows the actual forward pass values and gradient equations as text |
| Xavier initialisation | Proper weight init to prevent vanishing gradients from the start |

---

## Project Structure

```
neural-network-visualizer/
└── neural-network-visualizer.html    ← Entire project in one file
```

---

## The Math — Step by Step

### Architecture: 2 → 4 → 4 → 1

```
Input layer:   2 neurons  (XOR inputs A and B)
Hidden layer 1: 4 neurons  (ReLU activation)
Hidden layer 2: 4 neurons  (ReLU activation)
Output layer:  1 neuron   (Sigmoid activation → probability 0–1)
```

---

### Forward Propagation

For each layer: compute weighted sum, add bias, apply activation function.

```js
// Layer 1 (hidden)
const z1 = W1.map((row, i) =>
  row.reduce((sum, w, j) => sum + w * a0[j], 0) + b1[i]
);
const a1 = z1.map(relu); // apply ReLU element-wise

// Layer 2 (hidden)
const z2 = W2.map((row, i) =>
  row.reduce((sum, w, j) => sum + w * a1[j], 0) + b2[i]
);
const a2 = z2.map(relu);

// Output layer (sigmoid for binary classification)
const z3 = W3.map((row, i) =>
  row.reduce((sum, w, j) => sum + w * a2[j], 0) + b3[i]
);
const a3 = z3.map(sigmoid); // prediction: probability between 0 and 1
```

---

### Loss Function — Binary Cross-Entropy

```js
// For a single sample:
const loss = -(target * Math.log(pred + 1e-9) + (1 - target) * Math.log(1 - pred + 1e-9));
// 1e-9 prevents log(0) = -Infinity (numerical stability)
```

Why BCE and not MSE? Because BCE is the correct log-likelihood loss for binary classification. Its derivative with a sigmoid output simplifies cleanly to `pred - target`, making backprop elegant.

---

### Backpropagation — Chain Rule Layer by Layer

The goal: compute `dL/dW` for every weight matrix so we know which direction to nudge each weight.

```
dL/dW3 = dL/da3 · da3/dz3 · dz3/dW3     (output layer)
dL/dW2 = dL/da3 · da3/dz3 · dz3/da2 · da2/dz2 · dz2/dW2   (layer 2)
dL/dW1 = ...same, one more step back...   (layer 1)
```

In code:

```js
// OUTPUT LAYER
// For BCE + sigmoid, dL/dz3 = pred - target (beautiful simplification)
const dL_dz3 = a3.map((p, i) => (p - target[i]) * sigmoidDeriv(z3[i]));

// Update W3 and b3
W3.forEach((row, i) => row.forEach((_, j) => {
  W3[i][j] -= lr * dL_dz3[i] * a2[j];  // W -= lr * gradient
}));
b3.forEach((_, i) => { b3[i] -= lr * dL_dz3[i]; });

// HIDDEN LAYER 2 — gradient flows backward through W3
const dL_da2 = a2.map((_, j) =>
  W3.reduce((sum, row, i) => sum + row[j] * dL_dz3[i], 0)
);
const dL_dz2 = dL_da2.map((d, j) => d * reluDeriv(z2[j]));

W2.forEach((row, i) => row.forEach((_, j) => {
  W2[i][j] -= lr * dL_dz2[i] * a1[j];
}));
// ... same pattern for layer 1
```

---

### Activation Functions

```js
// ReLU — fast, no vanishing gradient for positive values
const relu    = x => Math.max(0, x);
const reluD   = x => x > 0 ? 1 : 0;  // derivative

// Sigmoid — squashes to (0,1), used in output for binary prob
const sigmoid = x => 1 / (1 + Math.exp(-x));
const sigD    = x => { const s = sigmoid(x); return s * (1 - s); };

// Tanh — squashes to (-1,1), zero-centred (better than sigmoid in hidden)
const tanh    = x => Math.tanh(x);
const tanhD   = x => 1 - Math.tanh(x) ** 2;
```

---

### Weight Initialisation — Xavier / Glorot

```js
function xavierInit(rows, cols) {
  const scale = Math.sqrt(6 / (rows + cols));
  // Uniform distribution in [-scale, +scale]
  return Array.from({length: rows}, () =>
    Array.from({length: cols}, () => (Math.random() * 2 - 1) * scale)
  );
}
```

Why Xavier? If weights are too large, activations saturate (gradient ≈ 0 — vanishing gradient). If too small, signals shrink to 0 through the layers. Xavier scales the initial distribution based on layer size to keep variance stable.

---

### Training Loop — requestAnimationFrame

```js
function loop() {
  if (!training) return;

  // Run N epochs per frame (controlled by speed slider)
  const steps = parseInt(document.getElementById('speed').value);
  for (let i = 0; i < steps; i++) trainEpoch();

  drawNet();        // redraw network with new activations
  updateLossChart();

  animId = requestAnimationFrame(loop); // schedule next frame (~60fps)
}
```

`requestAnimationFrame` syncs to the browser's refresh rate (~60fps) and pauses when the tab is hidden — more efficient than `setInterval`.

---

### Why XOR?

XOR is the simplest function that is **not linearly separable** — you cannot draw a single straight line to separate the two classes.

```
Input → Output
(0,0) → 0
(0,1) → 1   ← class 1
(1,0) → 1   ← class 1
(1,1) → 0
```

A single-layer perceptron (no hidden layers) provably cannot learn this. Adding one hidden layer with non-linear activations is enough. This is the historical result that revived neural network research in the 1980s (Minsky & Papert's criticism, then Rumelhart et al.'s backprop solution).

---

## What to say in an interview

**"Why is XOR significant?"**
It's the classic proof that non-linear activations are necessary. A single perceptron computes a linear decision boundary. XOR's four points require two boundaries — impossible with one layer. Adding a hidden layer with ReLU or sigmoid creates a composition of linear boundaries that can carve out any non-linear region. The Universal Approximation Theorem proves that a single hidden layer of sufficient width can approximate any continuous function.

**"Walk me through backprop for the output layer."**
For binary cross-entropy with a sigmoid output, the gradient simplifies elegantly. The derivative of BCE with respect to z (pre-activation) is just `pred - target`. This means the gradient is large when the prediction is confident and wrong, and small when the prediction is correct — exactly what we want. From z, we multiply by the activations of the previous layer to get `dL/dW` and update each weight by subtracting `lr * dL/dW`.

**"What would you add next?"**
Momentum (store the previous gradient direction and blend it with current), Adam optimiser (adaptive per-weight learning rates), mini-batch gradient descent (average gradients over N samples before updating), and dropout regularisation (randomly zero out neurons during training to prevent overfitting).

---

## YouTube Resources

- **The spelled-out intro to neural networks and backpropagation — Andrej Karpathy**: https://www.youtube.com/watch?v=VMj-3S1tku0
  — 2.5 hours. The single best explanation of backprop from scratch. Highly recommended before your interview.

- **3Blue1Brown Neural Networks series**: https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi
  — Visual intuition for what the math means geometrically.

---

## License

MIT — free to use, modify, and include in your portfolio.
