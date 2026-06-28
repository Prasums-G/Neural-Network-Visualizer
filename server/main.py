# ═══════════════════════════════════════════════════════════
#  main.py — Neural Network Platform API (FastAPI)
#  Stack: FastAPI + PostgreSQL + MongoDB + Redis + Celery
#  Endpoints: auth, datasets, training jobs, inference,
#             experiments, model registry, live metrics (WS)
# ═══════════════════════════════════════════════════════════

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from contextlib import asynccontextmanager
import asyncio, json, uuid, os
from datetime import datetime

from db.postgres  import get_db, init_db
from db.mongo     import mongo_client
from db.redis_c   import redis_client
from auth.jwt     import create_tokens, verify_token
from routers      import auth, datasets, jobs, inference, experiments, registry
from tasks.train  import train_network
from celery_app   import celery

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield

app = FastAPI(
    title="Neural Network Platform",
    description="Enterprise ML platform — train, track, deploy neural networks",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ─────────────────────────────────────────────────
# POST /api/auth/register
# POST /api/auth/login
# POST /api/auth/refresh
app.include_router(auth.router,        prefix="/api/auth",        tags=["auth"])

# POST   /api/datasets          — upload CSV/NPZ
# GET    /api/datasets          — list user datasets
# GET    /api/datasets/:id      — dataset info + preview
# DELETE /api/datasets/:id
app.include_router(datasets.router,    prefix="/api/datasets",    tags=["datasets"])

# POST   /api/jobs              — start training job
# GET    /api/jobs              — list jobs
# GET    /api/jobs/:id          — job status + metrics
# POST   /api/jobs/:id/stop     — stop a running job
app.include_router(jobs.router,        prefix="/api/jobs",        tags=["jobs"])

# POST /api/inference/:modelId  — single prediction
# POST /api/inference/batch     — batch predictions
app.include_router(inference.router,   prefix="/api/inference",   tags=["inference"])

# GET /api/experiments          — list experiments
# GET /api/experiments/:id      — run details (params, metrics, artefacts)
# GET /api/experiments/compare  — side-by-side metric comparison
app.include_router(experiments.router, prefix="/api/experiments", tags=["experiments"])

# GET    /api/registry          — list registered model versions
# POST   /api/registry          — register a checkpoint as a named model
# DELETE /api/registry/:id
app.include_router(registry.router,    prefix="/api/registry",    tags=["registry"])

app.get("/health")(lambda: {"status": "ok", "ts": datetime.utcnow().isoformat()})

# ── WebSocket: live training metrics ──────────────────────
#
#  Client connects to:  ws://host/ws/jobs/<job_id>?token=<jwt>
#  Server streams:      { epoch, loss, accuracy, weights_snapshot }
#  every N epochs (configurable, default = every epoch).
#
#  Architecture:
#    Celery worker (train_network task) publishes metrics to
#    Redis channel "job:<job_id>:metrics" after each epoch.
#    The WebSocket handler subscribes to that channel and
#    forwards messages to the connected client(s).
#    This decouples training (runs in worker) from streaming (API server).

active_connections: dict[str, list[WebSocket]] = {}

@app.websocket("/ws/jobs/{job_id}")
async def job_metrics_ws(job_id: str, websocket: WebSocket, token: str = ""):
    # Authenticate
    try:
        payload = verify_token(token)
    except Exception:
        await websocket.close(code=1008)
        return

    await websocket.accept()
    active_connections.setdefault(job_id, []).append(websocket)

    # Subscribe to Redis pub/sub channel for this job
    pubsub = redis_client.pubsub()
    await pubsub.subscribe(f"job:{job_id}:metrics")

    try:
        async for message in pubsub.listen():
            if message["type"] == "message":
                await websocket.send_text(message["data"])
    except WebSocketDisconnect:
        pass
    finally:
        active_connections[job_id].remove(websocket)
        await pubsub.unsubscribe(f"job:{job_id}:metrics")
        await pubsub.close()
