"""Reproducible Lab 1 CPU measurements; run from any working directory."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import json
import platform
import random
import subprocess
import sys
import threading
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import psutil
import torch
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RESULTS = Path(__file__).resolve().parents[1] / 'results'
NAMES = ('logistic_regression', 'random_forest')

def seed():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.set_num_threads(1)

def data():
    x, y = load_breast_cancer(return_X_y=True)
    return train_test_split(x, y, test_size=0.30, stratify=y, random_state=42)

def model(name):
    if name == 'logistic_regression':
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
    return RandomForestClassifier(n_estimators=100, random_state=42)

def measure_rss(action):
    """Sample absolute process RSS at ~1 ms; includes Python and imported libraries."""
    proc = psutil.Process()
    samples = [proc.memory_info().rss]
    stopped = threading.Event()
    def monitor():
        while not stopped.wait(0.001):
            samples.append(proc.memory_info().rss)
    thread = threading.Thread(target=monitor)
    thread.start()
    try:
        action()
        samples.append(proc.memory_info().rss)
    finally:
        stopped.set()
        thread.join()
    return {'baseline_bytes': samples[0], 'peak_bytes': max(samples), 'samples': len(samples)}

def memory_worker(name, phase):
    seed()
    xtr, xte, ytr, _ = data()
    fitted = model(name)
    if phase == 'training':
        result = measure_rss(lambda: fitted.fit(xtr, ytr))
    else:
        fitted.fit(xtr, ytr)
        fitted.predict(xte[:1])
        def predict_many():
            for _ in range(100):
                fitted.predict(xte[:1])
        result = measure_rss(predict_many)
    print(json.dumps(result))

def main():
    seed()
    RESULTS.mkdir(exist_ok=True)
    xtr, xte, ytr, yte = data()
    rows, raw = [], {}
    for name in NAMES:
        model(name).fit(xtr, ytr)  # unrecorded warm-up
        train_times = []
        for _ in range(5):
            fitted = model(name)
            start = time.perf_counter_ns()
            fitted.fit(xtr, ytr)
            train_times.append((time.perf_counter_ns() - start) / 1e6)
        accuracy = accuracy_score(yte, fitted.predict(xte))
        sample = xte[:1]
        fitted.predict(sample)  # unrecorded warm-up
        infer_times = []
        for _ in range(100):
            start = time.perf_counter_ns()
            fitted.predict(sample)
            infer_times.append((time.perf_counter_ns() - start) / 1e6)
        dest = RESULTS / name / 'model.joblib'
        dest.parent.mkdir(exist_ok=True)
        joblib.dump(fitted, dest)
        restored = joblib.load(dest)
        assert np.array_equal(fitted.predict(xte), restored.predict(xte))
        memory = {}
        for phase in ('training', 'inference'):
            completed = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--memory', name, phase], check=True, capture_output=True, text=True)
            memory[phase] = json.loads(completed.stdout)
        size = dest.stat().st_size
        row = dict(model=name, accuracy=accuracy, training_median_ms=float(np.median(train_times)), inference_median_ms=float(np.median(infer_times)), model_bytes=size, model_KB=size/1024, training_peak_MB=memory['training']['peak_bytes']/1024**2, inference_peak_MB=memory['inference']['peak_bytes']/1024**2)
        rows.append(row)
        raw[name] = dict(training_ms=train_times, inference_ms=infer_times, memory=memory)
        print(json.dumps(row), flush=True)
    frame = pd.DataFrame(rows)
    frame[['model','accuracy']].to_csv(RESULTS/'baseline_accuracy.csv', index=False, float_format='%.4f')
    frame.to_csv(RESULTS/'measurements.csv', index=False, float_format='%.6f')
    (RESULTS/'raw_measurements.json').write_text(json.dumps(raw, indent=2)+'\n')
    (RESULTS/'machine.json').write_text(json.dumps(dict(platform=platform.platform(), processor=platform.processor(), architecture=platform.machine(), logical_cpus=psutil.cpu_count(), total_memory_bytes=psutil.virtual_memory().total, train_samples=len(ytr), test_samples=len(yte), train_class_counts=np.bincount(ytr).tolist(), test_class_counts=np.bincount(yte).tolist(), seed=42, numerical_threads=1), indent=2)+'\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--memory', nargs=2, metavar=('MODEL','PHASE'))
    args = parser.parse_args()
    if args.memory:
        memory_worker(*args.memory)
    else:
        main()
