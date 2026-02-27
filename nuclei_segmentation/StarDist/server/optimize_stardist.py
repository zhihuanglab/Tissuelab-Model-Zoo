"""
Optimize StarDist prediction performance by compiling the model with TensorFlow functions
"""
import tensorflow as tf
import numpy as np

def optimize_model_prediction(model):
    """
    Optimize StarDist model prediction by compiling with tf.function
    
    Args:
        model: StarDist2D or StarDist3D model instance
        
    Returns:
        Optimized model with compiled predict function
    """
    if not hasattr(model, 'keras_model'):
        return model
    
    # Try to optimize with tf.function
    try:
        # Create a compiled version with optimized settings
        @tf.function(jit_compile=False, reduce_retracing=True)
        def optimized_predict(x):
            return model.keras_model(x, training=False)
        
        # Replace the predict method with optimized version
        model.keras_model._optimized_predict = optimized_predict
        
        # Store original predict for reference
        original_predict = model.keras_model.predict
        
        # Wrap to handle multiple outputs and batch processing
        def fast_predict(x, **kwargs):
            # Convert to tensor if numpy array
            if isinstance(x, np.ndarray):
                x_tensor = tf.convert_to_tensor(x, dtype=tf.float32)
            else:
                x_tensor = x
            
            # Handle different input shapes
            if isinstance(x_tensor, tf.Tensor) and x_tensor.ndim == 4:
                # Batch prediction (typical case in StarDist)
                outputs = model.keras_model._optimized_predict(x_tensor)
                
                # Convert outputs back to numpy and maintain batch dimension
                if isinstance(outputs, (list, tuple)):
                    return [output.numpy() for output in outputs]
                else:
                    return outputs.numpy()
            elif isinstance(x_tensor, tf.Tensor) and x_tensor.ndim == 3:
                # Single sample without batch dimension - add it
                x_tensor = tf.expand_dims(x_tensor, 0)
                outputs = model.keras_model._optimized_predict(x_tensor)
                
                # Remove batch dimension
                if isinstance(outputs, (list, tuple)):
                    return [output[0].numpy() for output in outputs]
                else:
                    return outputs[0].numpy()
            else:
                # Fallback to original method for other cases
                return original_predict(x, **kwargs)
        
        # Monkey patch the predict method
        model.keras_model.predict = fast_predict
        
        print("[OK] Model prediction optimized with tf.function")
        
    except Exception as e:
        print(f"Warning: Could not optimize model with tf.function: {e}")
        print("Using standard prediction method")
    
    return model


def configure_tensorflow_optimizations():
    """
    Configure TensorFlow for optimal CPU performance
    """
    import os
    
    # CRITICAL: Disable oneDNN which conflicts with AVX512 instructions
    # This is likely why 256 CPUs are slower than 32 CPUs!
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    print("[OK] Disabled oneDNN to enable AVX512 optimizations")
    
    # Limit thread count to avoid over-parallelization overhead
    # Too many threads can actually slow things down due to context switching
    os.environ['OMP_NUM_THREADS'] = '16'
    os.environ['MKL_NUM_THREADS'] = '16'
    
    try:
        # Disable XLA JIT for more predictable behavior
        tf.config.optimizer.set_jit(False)
        
        # Set thread pool size (inter_op should be small, intra_op can be larger)
        tf.config.threading.set_inter_op_parallelism_threads(2)
        tf.config.threading.set_intra_op_parallelism_threads(16)
        
        print("[OK] TensorFlow thread configuration set: inter_op=2, intra_op=16")
        
    except Exception as e:
        print(f"Warning: Could not configure TensorFlow threading: {e}")

