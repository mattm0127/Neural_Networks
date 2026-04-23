import numpy as np
import torch

def generate_big_data(n=1000):
    # We want half to pass, half to fail
    half = n // 2
    
    # --- GENERATE PASSES ---
    p_pass = np.random.uniform(0.95, 1.0, half) # High purity
    m_pass = np.random.uniform(0.0, 0.1, half)  # Low moisture
    ph_pass = np.random.uniform(0.4, 0.6, half)
    labels_pass = np.ones(half)
    
    # --- GENERATE FAILS (Randomly bad) ---
    p_fail = np.random.uniform(0.0, 0.94, half) # Low purity
    m_fail = np.random.uniform(0.1, 0.6, half)   # High moisture
    ph_fail = np.random.uniform(0.2, 0.8, half)
    labels_fail = np.zeros(half)
    
    # Combine them
    X = np.vstack([
        np.stack([p_pass, m_pass, ph_pass], axis=1),
        np.stack([p_fail, m_fail, ph_fail], axis=1)
    ])
    y = np.concatenate([labels_pass, labels_fail]).reshape(-1, 1)
    
    # Shuffle so the model doesn't just see all Passes then all Fails
    indices = np.arange(n)
    np.random.shuffle(indices)
    
    return torch.tensor(X[indices], dtype=torch.float32), torch.tensor(y[indices], dtype=torch.float32)


big_data = generate_big_data()