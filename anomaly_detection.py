
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, classification_report, roc_auc_score
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

# Try to import TensorFlow (optional for confusion matrix plotting)
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    print("Warning: TensorFlow not available. DNN model will be skipped.")
    TENSORFLOW_AVAILABLE = False

class DDoSDataAnalysis:
    def __init__(self):
        """
        Initialize the analysis with the dataset
        """
        import os
        
        # Hardcoded file path
        file_path = r'C:\Users\FINGERS\DDOS\DDOS\DNS-testing.parquet'
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset not found at: {file_path}")
        
        # Load CSV or Parquet file
        if file_path.endswith('.csv'):
            self.raw_data = pd.read_csv(file_path)
        elif file_path.endswith('.parquet'):
            self.raw_data = pd.read_parquet(file_path)
        else:
            raise ValueError("File must be CSV or Parquet format")
        
        self.processed_data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None 
        self.y_test = None
    
    def perform_comprehensive_eda(self):
        """
        Perform comprehensive Exploratory Data Analysis
        
        Returns:
        --------
        dict: Detailed EDA insights
        """
        # Dataset Overview
        eda_insights = {
            'dataset_shape': self.raw_data.shape,
            'columns': list(self.raw_data.columns),
            'missing_values': self.raw_data.isnull().sum(),
            'data_types': self.raw_data.dtypes
        }
        
        # Clean numeric columns for visualization
        numeric_cols = self.raw_data.select_dtypes(include=[np.number]).columns
        clean_numeric_data = self.raw_data[numeric_cols].replace([np.inf, -np.inf], np.nan).dropna()

        # Visualizations
        plt.figure(figsize=(15, 10))
        
        # 1. Missing Values Bar Plot
        plt.subplot(2, 2, 1)
        missing_values = self.raw_data.isnull().sum()
        missing_values = missing_values[missing_values > 0]
        plt.bar(missing_values.index, missing_values.values, color='purple')
        plt.title('Missing Values per Column')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        
        # 2. Target Label Distribution
        plt.subplot(2, 2, 2)
        if ' Label' in self.raw_data.columns:
            self.raw_data[' Label'].value_counts().plot(kind='pie', autopct='%1.1f%%')
            plt.title('Label Distribution')
        
        # 3. Boxplots for Numeric Features
        plt.subplot(2, 2, 3)
        if not clean_numeric_data.empty:
            sns.boxplot(data=clean_numeric_data)
            plt.title('Boxplot of Numeric Features')
            plt.xticks(rotation=45)
        
        # 4. Correlation Heatmap
        plt.subplot(2, 2, 4)
        if not clean_numeric_data.empty:
            corr_matrix = clean_numeric_data.corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5, fmt=".2f", square=True)
            plt.title('Feature Correlation Heatmap')
        
        plt.tight_layout()
        plt.savefig('eda_analysis.png', dpi=100, bbox_inches='tight')
        plt.close()
        
        return eda_insights
    
    def preprocess_data(self, target_column='Label', test_size=0.2):
        """
        Comprehensive data preprocessing
        
        Parameters:
        -----------
        target_column : str, optional
            Name of the target column
        test_size : float, optional
            Proportion of the dataset to include in the test split
        """
        # Create a copy of the data
        df = self.raw_data.copy()
        
        # Remove irrelevant columns
        df = df.drop(['Timestamp'], axis=1, errors='ignore')
        
        # Handle categorical variables (including target column)
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        categorical_cols.append('Label') if 'Label' in df.columns and df['Label'].dtype == 'category' else None
        
        label_encoders = {}
        for col in categorical_cols:
            if col in df.columns:
                encoder = LabelEncoder()
                df[col] = encoder.fit_transform(df[col].astype(str))
                label_encoders[col] = encoder
        
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Check and handle NaN/Infinite values
        if X.isnull().any().any() or np.isinf(X.values).any():
            X = X.fillna(X.mean())
            X = X.replace([np.inf, -np.inf], X.max().max())
            lower_bound, upper_bound = -1e6, 1e6
            X = X.clip(lower=lower_bound, upper=upper_bound)
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Scale the features
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)
        
        self.processed_data = df
    
    def feature_analysis(self, n_features=10):
        """
        Advanced feature importance and selection
        
        Parameters:
        -----------
        n_features : int, optional
            Number of top features to analyze
        
        Returns:
        --------
        pd.DataFrame: Feature importances
        """
        # Feature selection using ANOVA F-test
        selector = SelectKBest(score_func=f_classif, k=n_features)
        X_new = selector.fit_transform(self.X_train, self.y_train)
        
        # Get selected feature names
        feature_names = self.processed_data.drop(columns=['Label']).columns.tolist()
        selected_indices = np.where(selector.get_support())[0]
        selected_features = [feature_names[i] for i in selected_indices]
        
        # Random Forest Feature Importance
        rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_classifier.fit(self.X_train, self.y_train)
        
        feature_imp = pd.DataFrame({
            'feature': selected_features,
            'importance': rf_classifier.feature_importances_[selected_indices]
        }).sort_values('importance', ascending=False)
        
        # Visualization
        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=feature_imp)
        plt.title(f'Top {n_features} Most Important Features')
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=100, bbox_inches='tight')
        plt.close()
        
        return feature_imp
    
    def build_deep_neural_network(self, input_dim):
        """
        Build a deep neural network for DDoS detection
        
        Parameters:
        -----------
        input_dim : int
            Number of input features
        
        Returns:
        --------
        keras.Sequential: Compiled neural network model
        """
        model = models.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(32, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            layers.Dense(16, activation='relu'),
            layers.Dropout(0.2),
            
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        
        return model
    
    def train_models_with_tuning(self):
        """
        Train multiple machine learning models including deep neural network
        
        Returns:
        --------
        dict: Model performance metrics
        """
        # Define traditional ML models (simplified - no GridSearchCV for speed)
        models_ml = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42),
            'XGBoost': XGBClassifier(learning_rate=0.1, n_estimators=100, random_state=42, verbosity=0)
        }
        
        results = {}
        
        # Train traditional ML models
        for name, model in models_ml.items():
            print(f"Training {name}...")
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)
            
            # Get prediction probabilities for ROC-AUC
            try:
                if hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(self.X_test)
                    # For binary classification, use positive class probabilities
                    if y_pred_proba.shape[1] == 2:
                        micro_auc = roc_auc_score(self.y_test, y_pred_proba[:, 1])
                    else:
                        # For multiclass, use one-vs-rest with micro averaging
                        micro_auc = roc_auc_score(self.y_test, y_pred_proba, multi_class='ovr', average='micro')
                else:
                    micro_auc = None
            except Exception as e:
                print(f"  Warning: Could not calculate ROC-AUC for {name}: {str(e)}")
                micro_auc = None
            
            results[name] = {
                'best_params': {'model': name},
                'accuracy': accuracy_score(self.y_test, y_pred),
                'precision': precision_score(self.y_test, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(self.y_test, y_pred, average='weighted', zero_division=0),
                'f1_score': f1_score(self.y_test, y_pred, average='weighted', zero_division=0),
                'micro_auc': micro_auc,
                'classification_report': classification_report(self.y_test, y_pred, zero_division=0),
                'confusion_matrix': confusion_matrix(self.y_test, y_pred)
            }
        
        # Train Deep Neural Network (only if TensorFlow is available)
        if TENSORFLOW_AVAILABLE:
            print("\nTraining Deep Neural Network...")
            input_dim = self.X_train.shape[1]
            dnn_model = self.build_deep_neural_network(input_dim)
            
            early_stopping = EarlyStopping(
                monitor='val_loss', 
                patience=15, 
                restore_best_weights=True
            )
            
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
            
            # Ensure y_train is in the right format for Keras
            y_train_keras = np.array(self.y_train).reshape(-1) if isinstance(self.y_train, np.ndarray) else self.y_train.values
            
            history = dnn_model.fit(
                self.X_train, y_train_keras,
                epochs=100,
                batch_size=32,
                validation_split=0.2,
                callbacks=[early_stopping, reduce_lr],
                verbose=0
            )
            
            # DNN predictions
            y_pred_dnn = (dnn_model.predict(self.X_test, verbose=0) > 0.5).astype(int).flatten()
            y_pred_dnn_proba = dnn_model.predict(self.X_test, verbose=0).flatten()
            
            # Ensure y_test is in correct format
            y_test_eval = np.array(self.y_test).reshape(-1) if isinstance(self.y_test, np.ndarray) else self.y_test.values
            
            # Calculate micro-AUC for DNN
            try:
                dnn_micro_auc = roc_auc_score(y_test_eval, y_pred_dnn_proba)
            except Exception as e:
                print(f"  Warning: Could not calculate ROC-AUC for DNN: {str(e)}")
                dnn_micro_auc = None
            
            results['Deep Neural Network'] = {
                'best_params': {'epochs': len(history.history['loss']), 'batch_size': 32},
                'accuracy': accuracy_score(y_test_eval, y_pred_dnn),
                'precision': precision_score(y_test_eval, y_pred_dnn, average='weighted', zero_division=0),
                'recall': recall_score(y_test_eval, y_pred_dnn, average='weighted', zero_division=0),
                'f1_score': f1_score(y_test_eval, y_pred_dnn, average='weighted', zero_division=0),
                'micro_auc': dnn_micro_auc,
                'classification_report': classification_report(y_test_eval, y_pred_dnn, zero_division=0),
                'confusion_matrix': confusion_matrix(y_test_eval, y_pred_dnn),
                'model': dnn_model,
                'training_history': history
            }
            
            # Visualization of training history
            if 'training_history' in results['Deep Neural Network']:
                self._plot_training_history(results['Deep Neural Network']['training_history'])
        
        # Visualization of confusion matrices
        self._plot_confusion_matrices(results)
        
        # Visualization of model performance
        performance_df = pd.DataFrame.from_dict(results, orient='index')
        performance_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'micro_auc']
        
        plt.figure(figsize=(12, 6))
        performance_df[performance_metrics].plot(kind='bar')
        plt.title('Model Performance Comparison (Including Deep Neural Network)')
        plt.ylabel('Score')
        plt.tight_layout()
        plt.savefig('model_performance.png', dpi=100, bbox_inches='tight')
        plt.close()
        
        return results
    
    def _plot_training_history(self, history):
        """
        Plot neural network training history
        
        Parameters:
        -----------
        history : keras.callbacks.History
            Training history object
        """
        plt.figure(figsize=(14, 5))
        
        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=100, bbox_inches='tight')
        plt.close()
        
    def _plot_confusion_matrices(self, results):
        """
        Plot confusion matrices for all models
        
        Parameters:
        -----------
        results : dict
            Dictionary containing model results with confusion matrices
        """
        models_to_plot = list(results.keys())
        n_models = len(models_to_plot)
        
        # Create subplots for all models
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()
        
        for idx, model_name in enumerate(models_to_plot):
            cm = results[model_name]['confusion_matrix']
            
            # Plot confusion matrix as heatmap
            sns.heatmap(
                cm, 
                annot=True, 
                fmt='d', 
                cmap='Blues', 
                ax=axes[idx],
                cbar_kws={'label': 'Count'},
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive']
            )
            
            axes[idx].set_title(f'{model_name}\nConfusion Matrix')
            axes[idx].set_ylabel('True Label')
            axes[idx].set_xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.savefig('confusion_matrices.png', dpi=100, bbox_inches='tight')
        plt.close()
        

def main():
    # Initialize analysis
    ddos_analysis = DDoSDataAnalysis()
    
    # Perform Exploratory Data Analysis
    print("Performing Comprehensive EDA...")
    eda_insights = ddos_analysis.perform_comprehensive_eda()
    print("EDA Insights:", eda_insights)
    
    # Preprocess the data
    print("\nPreprocessing Data...")
    ddos_analysis.preprocess_data()
    
    # Feature Analysis
    print("\nPerforming Feature Analysis...")
    feature_importance = ddos_analysis.feature_analysis()
    
    # Train Multiple Models with Tuning
    print("\nTraining and Tuning Machine Learning Models...")
    ml_results = ddos_analysis.train_models_with_tuning()
    
    # Results Processing for Machine Learning Models
    results_list = []
    for model_name, model_results in ml_results.items():
        results_list.append({
            'Model': model_name,
            'Accuracy': model_results['accuracy'],
            'Precision': model_results['precision'],
            'Recall': model_results['recall'],
            'F1-Score': model_results['f1_score'],
            'Micro-AUC': model_results.get('micro_auc', 'N/A')
        })
    
    results_df = pd.DataFrame(results_list)
    print("\nModel Performance Summary:")
    print(results_df)

if __name__ == "__main__":
    main()