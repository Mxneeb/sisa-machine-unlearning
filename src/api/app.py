"""
Flask API backend for the SISA Unlearning Studio dashboard.
"""
import json
import os
import sys
import time
import threading
from pathlib import Path

from flask import Flask, jsonify, render_template, request

# Ensure src is on the path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sisa_engine.sharding import ShardedDataset
from sisa_engine.trainer import SISATrainer

app = Flask(
    __name__,
    template_folder="../web/templates",
    static_folder="../web/static",
)

# ── Global state (in-memory for demo; production would use a DB) ──────────────
_state = {
    "status": "idle",          # idle | training | trained | unlearning
    "training_log": [],
    "metrics": {},
    "forget_log": [],
    "trainer": None,
    "sharded_dataset": None,
}
_lock = threading.Lock()

CHECKPOINT_DIR = str(Path(__file__).resolve().parents[2] / "checkpoints")
MAPPING_PATH = os.path.join(CHECKPOINT_DIR, "shard_mapping.json")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def get_status():
    with _lock:
        return jsonify({
            "status": _state["status"],
            "training_log": _state["training_log"][-20:],
            "metrics": _state["metrics"],
            "forget_log": _state["forget_log"],
        })


@app.route("/api/train", methods=["POST"])
def start_training():
    """Start SISA training on MNIST (lightweight demo dataset)."""
    data = request.get_json(silent=True) or {}
    num_shards = int(data.get("num_shards", 3))
    num_slices = int(data.get("num_slices", 3))
    epochs = int(data.get("epochs_per_slice", 2))

    with _lock:
        if _state["status"] == "training":
            return jsonify({"error": "Training already in progress"}), 409
        _state["status"] = "training"
        _state["training_log"] = []
        _state["metrics"] = {}

    thread = threading.Thread(
        target=_run_training,
        args=(num_shards, num_slices, epochs),
        daemon=True,
    )
    thread.start()
    return jsonify({"message": "Training started", "num_shards": num_shards})


@app.route("/api/forget", methods=["POST"])
def forget_request():
    """Submit a forget request for a specific data index."""
    data = request.get_json(silent=True) or {}
    idx = data.get("index")
    if idx is None:
        return jsonify({"error": "Missing 'index' field"}), 400
    idx = int(idx)

    with _lock:
        if _state["status"] != "trained":
            return jsonify({"error": "Model not trained yet"}), 409
        trainer = _state["trainer"]
        if trainer is None:
            return jsonify({"error": "Trainer not initialized"}), 500

    try:
        result = trainer.unlearn(idx)
        entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "index": idx,
            **result,
        }
        with _lock:
            _state["forget_log"].append(entry)
        return jsonify(entry)
    except KeyError:
        return jsonify({"error": f"Index {idx} not found in training data"}), 404


@app.route("/api/forget_log")
def get_forget_log():
    with _lock:
        return jsonify(_state["forget_log"])


@app.route("/api/metrics")
def get_metrics():
    with _lock:
        return jsonify(_state["metrics"])


# ── Background training worker ────────────────────────────────────────────────

def _run_training(num_shards: int, num_slices: int, epochs_per_slice: int):
    """Runs in a background thread. Loads MNIST and trains with SISA."""
    try:
        import torchvision
        import torchvision.transforms as T

        transform = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
        data_dir = str(Path(__file__).resolve().parents[2] / "data")
        train_dataset = torchvision.datasets.MNIST(
            data_dir, train=True, download=True, transform=transform
        )

        sd = ShardedDataset(train_dataset, num_shards=num_shards, num_slices=num_slices)
        sd.save_mapping(MAPPING_PATH)

        log_entries = []

        def progress_cb(shard_id, slice_id, metrics):
            entry = f"Shard {shard_id} | Slice {slice_id} | Acc: {metrics['accuracy']:.2%}"
            log_entries.append(entry)
            with _lock:
                _state["training_log"] = log_entries[:]

        trainer = SISATrainer(
            sd,
            checkpoint_dir=CHECKPOINT_DIR,
            dataset_name="mnist",
            epochs_per_slice=epochs_per_slice,
            progress_callback=progress_cb,
        )

        t0 = time.time()
        results = trainer.train_all()
        elapsed = time.time() - t0

        with _lock:
            _state["status"] = "trained"
            _state["trainer"] = trainer
            _state["sharded_dataset"] = sd
            _state["metrics"] = {
                "total_training_time_seconds": round(elapsed, 2),
                "num_shards": num_shards,
                "num_slices": num_slices,
                "dataset": "MNIST",
                "training_samples": len(train_dataset),
                "shard_results": results,
            }
            _state["training_log"].append(
                f"Training complete in {elapsed:.1f}s"
            )

    except Exception as exc:
        with _lock:
            _state["status"] = "idle"
            _state["training_log"].append(f"ERROR: {exc}")


if __name__ == "__main__":
    app.run(debug=True, port=5000, use_reloader=False)
