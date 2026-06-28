# Neural Network Visualizer

An end-to-end ML platform for training, tracking, and serving neural networks. Built on FastAPI + Celery + NumPy — the backpropagation engine is still implemented from scratch, but now runs as an async distributed task with real-time metric streaming, experiment tracking, model registry, and a REST inference API.

![Python](https://img.shields.io/badge/Python-3.11-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green) ![Celery](https://img.shields.io/badge/Celery-5.3-red) ![NumPy](https://img.shields.io/badge/NumPy-1.26-orange) ![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-blue) ![MongoDB](https://img.shields.io/badge/MongoDB-7-green) ![Redis](https://img.shields.io/badge/Redis-7-red)

---

## Architecture

```
React Dashboard
  │  REST (auth, datasets, jobs, inference, experiments)
  │  WebSocket /ws/jobs/:id  (live metrics stream)
  ▼
FastAPI (Python 3.11)
  ├── POST /api/jobs  → enqueues Celery task
  ├── WS   /ws/jobs/:id → subscribes to Redis channel
  └── GET  /api/inference/:modelId → loads checkpoint, runs forward pass
         │
    Celery Worker (NumPy backprop engine)
         │  publishes epoch metrics to Redis pub/sub
         │  saves checkpoints to MongoDB + S3
         ▼
  ┌──────────────────────────────────────────────┐
  │  PostgreSQL   │  MongoDB        │  Redis       │
  │  Users, jobs, │  Metrics,       │  Task queue, │
  │  datasets     │  checkpoints,   │  pub/sub,    │
  │               │  experiment runs│  cache       │
  └──────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| API server | FastAPI 0.110 | Async Python, automatic OpenAPI docs, WebSocket support built in |
| ML engine | NumPy 1.26 (custom backprop) | No TensorFlow/PyTorch — the implementation IS the project |
| Task queue | Celery 5 + Redis broker | Training runs in a separate worker process; API stays responsive |
| Metric streaming | Redis pub/sub → WebSocket | Worker publishes per-epoch; browser receives in real time |
| Auth | JWT (python-jose) + bcrypt (passlib) | Stateless, horizontally scalable |
| Primary DB | PostgreSQL 16 (asyncpg + SQLAlchemy) | Users, training jobs, datasets — relational |
| Experiment DB | MongoDB 7 (Motor async driver) | Per-epoch metric logs, model checkpoints — document model fits naturally |
| Object storage | S3 / MinIO | Model checkpoint files (weight matrices as JSON/pickle) |
| Container | Docker + Compose | Full stack in one command |

---

## Project Structure

```
Neural-Network-Visualizer/
├── docker-compose.yml
├── server/
│   ├── main.py              ← FastAPI app + WebSocket metric stream
│   ├── celery_app.py        ← Celery instance configuration
│   ├── requirements.txt
│   ├── Dockerfile
│   ├── routers/
│   │   ├── auth.py          ← POST /register /login /refresh
│   │   ├── datasets.py      ← Upload CSV/NPZ, list, preview
│   │   ├── jobs.py          ← Start/stop/list training jobs
│   │   ├── inference.py     ← Single + batch prediction endpoints
│   │   ├── experiments.py   ← List runs, compare metrics
│   │   └── registry.py      ← Register/list/delete model versions
│   ├── auth/
│   │   └── jwt.py           ← Token creation + verification
│   └── db/
│       ├── postgres.py      ← SQLAlchemy async engine
│       ├── mongo.py         ← Motor async client + collections
│       ├── redis_c.py       ← aioredis async + sync redis clients
│       └── schema.sql       ← PostgreSQL DDL
└── worker/
    └── train.py             ← Celery task: full backprop training loop
```

---

## Quick Start

```bash
# 1. Configure environment
cp server/.env.example server/.env
# Add your JWT_SECRET, DB passwords, and S3 credentials

# 2. Start everything
docker compose up --build
# This starts: FastAPI, Celery worker, PostgreSQL, MongoDB, Redis

# 3. Access points:
#    API docs:   http://localhost:8000/docs     (Swagger UI, auto-generated)
#    API:        http://localhost:8000/api
#    WebSocket:  ws://localhost:8000/ws/jobs/:jobId?token=<jwt>
#    Health:     http://localhost:8000/health
```

---

## API Reference

### Authentication

```
POST /api/auth/register    { name, email, password }
POST /api/auth/login       { email, password }
POST /api/auth/refresh     { refreshToken }
```

### Datasets

```
POST   /api/datasets                   Upload CSV or NPZ file (multipart/form-data)
GET    /api/datasets                   List your datasets
GET    /api/datasets/:id               Dataset info + first 10 rows preview
DELETE /api/datasets/:id
```

### Training Jobs

```
POST /api/jobs
Body:
{
  "name":          "XOR experiment 3",
  "dataset_id":    "uuid",
  "layer_sizes":   [2, 8, 8, 1],
  "activation":    "relu",
  "learning_rate": 0.01,
  "momentum":      0.9,
  "epochs":        2000,
  "batch_size":    4
}
Response: { "job_id": "uuid", "status": "queued" }

GET    /api/jobs              List jobs (paginated)
GET    /api/jobs/:id          Status + final metrics
POST   /api/jobs/:id/stop     Set Redis stop flag → worker exits cleanly
```

### Real-Time Metrics (WebSocket)

```
Connect: ws://host/ws/jobs/<job_id>?token=<access_jwt>

Server streams (every epoch):
{
  "epoch":    150,
  "loss":     0.043821,
  "accuracy": 0.9750,
  "lr":       0.01
}
```

Architecture: the Celery worker calls `redis.publish(f"job:{job_id}:metrics", payload)` after each epoch. The FastAPI WebSocket handler subscribes to that channel and forwards to the browser. No polling, no database overhead on the hot path.

### Inference

```
POST /api/inference/:modelId
Body: { "inputs": [[0, 1], [1, 0]] }
Response: { "predictions": [0.97, 0.96], "model_version": "v3" }

POST /api/inference/batch
Body: { "model_id": "uuid", "rows": [[...], [...]] }
```

### Experiments

```
GET /api/experiments                   List all runs with summary metrics
GET /api/experiments/:id               Full run: params + per-epoch metrics
GET /api/experiments/compare?ids=a,b   Side-by-side metric comparison
```

### Model Registry

```
POST   /api/registry                   Register checkpoint { job_id, name, version }
GET    /api/registry                   List registered models
DELETE /api/registry/:id
```

---

## The Backpropagation Engine

The core ML code (`worker/train.py`) implements everything from scratch using NumPy — no TensorFlow, no PyTorch. Here's what each piece does at the enterprise level:

### Forward pass — vectorised

```python
# Process an entire batch at once (matrix multiply, not loops)
# X shape: (batch_size, n_features)
# W shape: (n_neurons_out, n_neurons_in)
z = X @ W.T + b          # weighted sum: (batch, out)
a = relu(z)              # activation:   (batch, out)
```

Vectorising across the batch dimension is what makes NumPy fast enough for training — one matrix multiply instead of a loop over samples.

### Backward pass — chain rule in matrix form

```python
# Output layer (BCE + sigmoid → elegant simplification)
dz = (prediction - y) / batch_size      # (batch, 1)

# Gradient of W: outer product of error and previous activation
dW = dz.T @ a_prev                      # (out, in)
db = dz.sum(axis=0, keepdims=True)      # (1, out)

# Propagate error back through this layer for next iteration
dz_prev = (dz @ W) * relu_derivative(z_prev)
```

### SGD with momentum

```python
# Momentum prevents oscillation and helps escape shallow minima
# vW accumulates gradient direction (exponential moving average)
vW = momentum * vW - lr * dW
W  += vW
```

### Why Celery instead of running in the API process?

Training 1000 epochs blocks the CPU for seconds. If this ran in the FastAPI handler, every request during training would time out. Celery moves the computation to a worker process (or machine) and the API remains responsive. The worker and API communicate through Redis, not shared memory — so workers can run on different machines entirely (GPU instances in production).

---

## Interview Q&A

**"Why not just use PyTorch?"**
PyTorch is the right choice for production ML. This project implements backprop from scratch because understanding the algorithm is what differentiates a senior engineer from someone who knows the API. In interviews, you can walk through the chain rule derivation for every layer. That's the point.

**"How does the metric streaming work end-to-end?"**
Celery worker publishes JSON to a Redis pub/sub channel named `job:<id>:metrics` after each epoch. The FastAPI WebSocket handler subscribes using `aioredis` in an async loop, forwarding each message to the browser WebSocket connection. The browser receives events within milliseconds of each epoch completing — no polling, no database writes on the hot path.

**"How would you support GPU training?"**
Replace NumPy with CuPy (same API, GPU-backed arrays). Run the Celery workers on GPU instances in Kubernetes. The rest of the stack — FastAPI, Redis, databases — runs unchanged on CPU instances.

**"What's in the model registry?"**
Each registered model version is a record pointing to a checkpoint in S3 (or MongoDB GridFS). It stores: the weight matrices, the architecture config (layer sizes, activation), training metrics, the dataset it was trained on, and a version tag. The inference endpoint loads the weights, reconstructs the network, and runs the forward pass.

**"How would you A/B test two model versions?"**
Add a `variant` field to the inference request. Route 50% of traffic to model v1 and 50% to v2. Log prediction + variant to an events table. Run a significance test after N predictions. Promote the winner to `stable` in the model registry.

---

## Production checklist

- [ ] Set `SECRET_KEY` and all DB passwords to random values in CI secrets
- [ ] Enable PostgreSQL SSL mode in `DATABASE_URL`
- [ ] Use Redis Cluster for pub/sub at scale (Redis Sentinel for HA)
- [ ] Store model checkpoints in S3 (not MongoDB) for files over 16MB
- [ ] Configure Celery `task_serializer = 'json'` and `result_expires`
- [ ] Add Prometheus metrics endpoint (`/metrics`) and Grafana dashboard
- [ ] Run Celery workers on GPU instances for large networks
- [ ] Add DVC for dataset versioning (Git-style versioning for data files)
- [ ] Kubernetes: separate deployments for API, worker, and flower (Celery monitor)

---

## YouTube Resources

- **Andrej Karpathy — The spelled-out intro to backpropagation (micrograd)**: https://www.youtube.com/watch?v=VMj-3S1tku0  
  The definitive from-scratch backprop video. Watch before any ML system design interview.

- **3Blue1Brown — Neural Networks series**: https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi  
  Visual intuition for what the matrix math means geometrically.

---

## License

MIT
