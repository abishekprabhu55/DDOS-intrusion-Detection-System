import sys
import traceback

# Suppress matplotlib GUI
import matplotlib
matplotlib.use('Agg')

try:
    print("Starting import checks...")
    
    # Test imports
    import pandas as pd
    print("✓ pandas imported")
    
    import numpy as np
    print("✓ numpy imported")
    
    import matplotlib.pyplot as plt
    print("✓ matplotlib imported")
    
    # Test tensorflow
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models
    print("✓ tensorflow imported")
    
    # Test sklearn
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    print("✓ sklearn imported")
    
    # Now test the actual script
    print("\nImporting anomaly_detection...")
    from anomaly_detection import DDoSDataAnalysis
    
    print("\nInitializing DDoSDataAnalysis...")
    ddos_analysis = DDoSDataAnalysis()
    print("✓ DDoSDataAnalysis initialized")
    
    print("\nPerforming EDA...")
    eda_insights = ddos_analysis.perform_comprehensive_eda()
    print("✓ EDA completed")
    
except Exception as e:
    print(f"\n✗ ERROR occurred:")
    print(f"Type: {type(e).__name__}")
    print(f"Message: {str(e)}")
    print("\nFull traceback:")
    traceback.print_exc()
    sys.exit(1)

print("\n✓ All tests passed!")
