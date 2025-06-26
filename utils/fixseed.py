import os
import random
import numpy as np
import warnings
from typing import Optional, List, Dict, Any

def fix_all_seeds(
    seed: int = 42,
    deterministic_cudnn: bool = True,
    warn_only: bool = False,
    verbose: bool = True
) -> Dict[str, str]:
    """
    Fix random seeds for all major ML/DL libraries to ensure reproducibility.
    
    Args:
        seed (int): The seed value to use across all libraries. Default: 42
        deterministic_cudnn (bool): Whether to make CuDNN deterministic (slower but reproducible). Default: True
        warn_only (bool): If True, only warn about missing packages instead of failing silently. Default: False
        verbose (bool): Whether to print status messages. Default: True
    
    Returns:
        Dict[str, str]: Status of each library's seed setting
    
    Example:
        >>> status = fix_all_seeds(seed=12345, verbose=True)
        >>> print(status)
    """
    
    status = {}
    
    def log_status(library: str, message: str, success: bool = True):
        """Helper function to log status"""
        status[library] = message
        if verbose:
            prefix = "✓" if success else "✗"
            print(f"{prefix} {library}: {message}")
    
    # Python built-in random
    try:
        random.seed(seed)
        log_status("Python random", f"Seed set to {seed}")
    except Exception as e:
        log_status("Python random", f"Failed: {str(e)}", False)
    
    # NumPy
    try:
        np.random.seed(seed)
        log_status("NumPy", f"Seed set to {seed}")
    except Exception as e:
        log_status("NumPy", f"Failed: {str(e)}", False)
    
    # PyTorch
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
        
        if deterministic_cudnn:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            log_status("PyTorch", f"Seed set to {seed} (deterministic CuDNN enabled)")
        else:
            torch.backends.cudnn.benchmark = True
            log_status("PyTorch", f"Seed set to {seed} (CuDNN benchmark enabled)")
            
        # Set number of threads for reproducibility
        torch.set_num_threads(1)
        
    except ImportError:
        msg = "Not installed"
        log_status("PyTorch", msg, not warn_only)
        if warn_only:
            warnings.warn("PyTorch not found. Install with: pip install torch")
    except Exception as e:
        log_status("PyTorch", f"Error: {str(e)}", False)
    
    # # TensorFlow
    # try:
    #     import tensorflow as tf
    #     tf.random.set_seed(seed)
        
    #     # For TensorFlow 2.x
    #     if hasattr(tf, 'config') and hasattr(tf.config, 'experimental'):
    #         try:
    #             tf.config.experimental.enable_op_determinism()
    #             log_status("TensorFlow", f"Seed set to {seed} (deterministic ops enabled)")
    #         except:
    #             log_status("TensorFlow", f"Seed set to {seed} (deterministic ops not available)")
    #     else:
    #         log_status("TensorFlow", f"Seed set to {seed}")
            
    # except ImportError:
    #     msg = "Not installed"
    #     log_status("TensorFlow", msg, not warn_only)
    #     if warn_only:
    #         warnings.warn("TensorFlow not found. Install with: pip install tensorflow")
    # except Exception as e:
    #     log_status("TensorFlow", f"Error: {str(e)}", False)
    
    # # JAX
    # try:
    #     import jax
    #     from jax import random as jax_random
    #     # JAX uses a different approach with PRNGKeys
    #     key = jax_random.PRNGKey(seed)
    #     # Store the key globally for later use
    #     globals()['_jax_rng_key'] = key
    #     log_status("JAX", f"PRNG key initialized with seed {seed}")
    # except ImportError:
    #     msg = "Not installed"
    #     log_status("JAX", msg, not warn_only)
    #     if warn_only:
    #         warnings.warn("JAX not found. Install with: pip install jax")
    # except Exception as e:
    #     log_status("JAX", f"Error: {str(e)}", False)
    
    # # Scikit-learn (affects some random operations)
    # try:
    #     from sklearn.utils import check_random_state
    #     # sklearn doesn't have a global seed, but we can verify it works
    #     check_random_state(seed)
    #     log_status("scikit-learn", f"Random state can be set to {seed}")
    # except ImportError:
    #     msg = "Not installed"
    #     log_status("scikit-learn", msg, not warn_only)
    #     if warn_only:
    #         warnings.warn("scikit-learn not found. Install with: pip install scikit-learn")
    # except Exception as e:
    #     log_status("scikit-learn", f"Error: {str(e)}", False)
    
    # Pandas (for sampling operations)
    try:
        import pandas as pd
        # Pandas doesn't have global seed, but uses numpy's random state
        log_status("Pandas", f"Will use NumPy's seed ({seed}) for random operations")
    except ImportError:
        msg = "Not installed"
        log_status("Pandas", msg, not warn_only)
        if warn_only:
            warnings.warn("Pandas not found. Install with: pip install pandas")
    except Exception as e:
        log_status("Pandas", f"Error: {str(e)}", False)
    
    # Transformers (Hugging Face)
    try:
        import transformers
        # Transformers uses the underlying framework's seed
        log_status("Transformers", f"Will use PyTorch/TensorFlow seed ({seed})")
    except ImportError:
        msg = "Not installed"
        log_status("Transformers", msg, not warn_only)
        if warn_only:
            warnings.warn("Transformers not found. Install with: pip install transformers")
    except Exception as e:
        log_status("Transformers", f"Error: {str(e)}", False)
    
    # Datasets (Hugging Face)
    try:
        import datasets
        # Datasets library uses numpy's random for shuffling
        log_status("Datasets", f"Will use NumPy's seed ({seed}) for shuffling")
    except ImportError:
        msg = "Not installed"
        log_status("Datasets", msg, not warn_only)
        if warn_only:
            warnings.warn("Datasets not found. Install with: pip install datasets")
    except Exception as e:
        log_status("Datasets", f"Error: {str(e)}", False)
    
    # Set environment variables for additional reproducibility
    env_vars = {
        'PYTHONHASHSEED': str(seed),
        'CUBLAS_WORKSPACE_CONFIG': ':4096:8',  # For PyTorch deterministic operations
        'TF_DETERMINISTIC_OPS': '1',  # For TensorFlow deterministic operations
        'TF_CUDNN_DETERMINISTIC': '1' if deterministic_cudnn else '0',
    }
    
    for var, value in env_vars.items():
        os.environ[var] = value
    
    log_status("Environment", f"Set environment variables for reproducibility")
    
    # Additional warnings and tips
    if verbose:
        print("\n" + "="*60)
        print("REPRODUCIBILITY NOTES:")
        print("="*60)
        print("1. For complete reproducibility, set the seed BEFORE importing ML libraries")
        print("2. Some operations may still be non-deterministic on GPU")
        print("3. Multi-threading can introduce randomness - consider single-threaded execution")
        print("4. Different hardware/drivers may produce different results")
        if deterministic_cudnn:
            print("5. Deterministic CuDNN is enabled - this may slow down training")
        print("6. For JAX, use the global '_jax_rng_key' variable for random operations")
        print("="*60)
    
    return status