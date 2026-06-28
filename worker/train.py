# tasks/train.py — Celery task: train a neural network
# This runs in a separate worker process.
# Progress is published to Redis so the API server can stream it via WebSocket.

import json, time, math, random, pickle, os
from celery_app import celery
from db.redis_c import redis_client_sync as redis
import numpy as np

# ─── Activation functions ────────────────────────────────────
def relu(z):      return np.maximum(0, z)
def relu_d(z):    return (z > 0).astype(float)
def sigmoid(z):   return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
def sigmoid_d(z): s = sigmoid(z); return s * (1 - s)
def tanh_d(z):    return 1 - np.tanh(z) ** 2

ACTIVATIONS = {
    "relu":    (relu,         relu_d),
    "sigmoid": (sigmoid,      sigmoid_d),
    "tanh":    (np.tanh,      tanh_d),
}

# ─── Weight init ─────────────────────────────────────────────
def xavier_init(fan_in, fan_out):
    scale = math.sqrt(6 / (fan_in + fan_out))
    return np.random.uniform(-scale, scale, (fan_out, fan_in))

# ─── Network class ───────────────────────────────────────────
class NeuralNetwork:
    """
    Fully-connected network with configurable layers, activations,
    and learning rate schedule. Backpropagation computed analytically
    — no autograd library. Uses NumPy for vectorised matrix ops.
    """

    def __init__(self, layer_sizes, activation="relu", lr=0.01, momentum=0.9):
        self.layers     = layer_sizes
        self.act_name   = activation
        self.act, self.act_d = ACTIVATIONS[activation]
        self.lr         = lr
        self.momentum   = momentum

        # Weights and biases (Xavier init)
        self.W = [xavier_init(layer_sizes[i], layer_sizes[i+1])
                  for i in range(len(layer_sizes)-1)]
        self.b = [np.zeros((1, n)) for n in layer_sizes[1:]]

        # Momentum accumulators (for SGD + momentum)
        self.vW = [np.zeros_like(w) for w in self.W]
        self.vb = [np.zeros_like(b) for b in self.b]

    def forward(self, X):
        self.cache = []   # stores (z, a) per layer for backprop
        a = X
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = a @ W.T + b
            # Last layer: sigmoid for binary, softmax for multi-class
            if i == len(self.W) - 1:
                a_new = sigmoid(z)
            else:
                a_new = self.act(z)
            self.cache.append((z, a))
            a = a_new
        self.cache.append((None, a))  # final output
        return a

    def backward(self, X, y):
        m       = X.shape[0]          # batch size
        grads_W = [None] * len(self.W)
        grads_b = [None] * len(self.b)

        # Output layer gradient: dL/dz = pred - y  (BCE + sigmoid)
        pred     = self.cache[-1][1]
        dz       = (pred - y) / m

        for i in reversed(range(len(self.W))):
            _, a_prev = self.cache[i]
            grads_W[i] = dz.T @ a_prev
            grads_b[i] = dz.sum(axis=0, keepdims=True)

            if i > 0:
                z_prev, _ = self.cache[i]
                dz = (dz @ self.W[i]) * self.act_d(z_prev)

        # SGD with momentum update
        for i in range(len(self.W)):
            self.vW[i] = self.momentum * self.vW[i] - self.lr * grads_W[i]
            self.vb[i] = self.momentum * self.vb[i] - self.lr * grads_b[i]
            self.W[i] += self.vW[i]
            self.b[i]  += self.vb[i]

    def loss(self, pred, y):
        # Binary cross-entropy with epsilon for numerical stability
        eps = 1e-9
        return -np.mean(y * np.log(pred + eps) + (1-y) * np.log(1 - pred + eps))

    def accuracy(self, pred, y):
        return np.mean(np.round(pred) == y)

    def to_dict(self):
        """Serialise weights for checkpoint storage."""
        return {
            "layers":     self.layers,
            "activation": self.act_name,
            "W":          [w.tolist() for w in self.W],
            "b":          [b.tolist() for b in self.b],
        }


@celery.task(bind=True, name="tasks.train_network")
def train_network(self, job_id: str, config: dict):
    """
    Celery task. Runs the training loop and publishes metrics to Redis
    after every epoch. The FastAPI WebSocket handler subscribes and
    streams those metrics to the browser in real time.

    config keys:
      layer_sizes  : list[int]    e.g. [2, 8, 8, 1]
      activation   : str          "relu" | "sigmoid" | "tanh"
      learning_rate: float
      momentum     : float
      epochs       : int
      batch_size   : int
      dataset_id   : str          loaded from DB / S3
    """
    from db.postgres_sync import get_db_sync
    from db.mongo_sync    import mongo

    db = get_db_sync()

    # Update job status → running
    db.execute(
        "UPDATE training_jobs SET status='running', started_at=NOW() WHERE id=%s",
        (job_id,)
    )
    db.commit()

    net = NeuralNetwork(
        layer_sizes = config["layer_sizes"],
        activation  = config.get("activation", "relu"),
        lr          = config.get("learning_rate", 0.01),
        momentum    = config.get("momentum", 0.9),
    )

    # Load dataset (XOR for demo; real impl loads from S3/DB by dataset_id)
    X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
    y = np.array([[0],[1],[1],[0]],         dtype=float)

    epochs     = config.get("epochs", 1000)
    batch_size = config.get("batch_size", 4)
    channel    = f"job:{job_id}:metrics"

    best_loss  = float("inf")

    for epoch in range(1, epochs + 1):
        # Mini-batch SGD
        idx = np.random.permutation(len(X))
        for start in range(0, len(X), batch_size):
            batch_idx = idx[start:start+batch_size]
            Xb, yb    = X[batch_idx], y[batch_idx]
            pred      = net.forward(Xb)
            net.backward(Xb, yb)

        # Full-pass metrics
        pred_full = net.forward(X)
        loss_val  = float(net.loss(pred_full, y))
        acc_val   = float(net.accuracy(pred_full, y))

        # Save checkpoint if best so far
        if loss_val < best_loss:
            best_loss = loss_val
            checkpoint = json.dumps(net.to_dict())
            redis.set(f"job:{job_id}:best_checkpoint", checkpoint)

        # Publish metrics to Redis channel
        payload = json.dumps({
            "epoch":    epoch,
            "loss":     round(loss_val, 6),
            "accuracy": round(acc_val,  4),
            "lr":       net.lr,
        })
        redis.publish(channel, payload)

        # Log to MongoDB (experiment tracking)
        mongo["metrics"].insert_one({
            "job_id":    job_id,
            "epoch":     epoch,
            "loss":      loss_val,
            "accuracy":  acc_val,
            "timestamp": time.time(),
        })

        # Allow early stop via Redis flag
        if redis.get(f"job:{job_id}:stop"):
            break

        time.sleep(0.01)  # yield CPU; remove in production

    # Final status update
    final_pred = net.forward(X)
    db.execute(
        """UPDATE training_jobs
           SET status='completed', completed_at=NOW(),
               final_loss=%s, final_accuracy=%s
           WHERE id=%s""",
        (float(net.loss(final_pred, y)), float(net.accuracy(final_pred, y)), job_id)
    )
    db.commit()

    # Persist final checkpoint
    mongo["checkpoints"].insert_one({
        "job_id":    job_id,
        "weights":   net.to_dict(),
        "loss":      float(net.loss(final_pred, y)),
        "created_at": datetime.utcnow(),
    })

    return {"job_id": job_id, "status": "completed"}
