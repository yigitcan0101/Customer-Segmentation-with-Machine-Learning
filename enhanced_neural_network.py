# src/models/enhanced_neural_network.py

import tensorflow as tf
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import os
from datetime import datetime


class EnhancedCustomerSegmentationModel:
    """
    Enhanced Neural Network - 14 Features ile Müşteri Segmentasyonu
    ML-discovered features ile improved accuracy hedefi
    """

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.segment_mapping = None
        self.reverse_mapping = None
        self.enhanced_feature_columns = [
            # Original RFM features (1-5)
            'Recency', 'Frequency', 'Monetary', 'AvgOrderValue', 'CustomerValue',
            # ML-discovered features (6-14)
            'auto_product_category_id', 'auto_geographic_segment_id', 'behavior_pattern_id',
            'purchase_span_days', 'order_value_consistency', 'price_tier_preference',
            'bulk_buying_tendency', 'segment_pattern_alignment', 'geographic_behavior_score'
        ]
        self.history = None

    def load_enhanced_data(self):
        """
        Enhanced TensorFlow verilerini yükle (14 features)
        """
        print("📊 Enhanced ML verisi yükleniyor...")

        try:
            # Enhanced features yükle
            X = np.load("data/processed/X_ml_enhanced_features.npy")
            y = np.load("data/processed/y_ml_enhanced_labels.npy")

            # Feature info yükle
            with open("data/processed/ml_enhanced_feature_info.json", "r") as f:
                feature_info = json.load(f)

            self.enhanced_feature_columns = feature_info['feature_names']
            self.segment_mapping = feature_info['segment_mapping']

            # Reverse mapping oluştur
            self.reverse_mapping = {v: k for k, v in self.segment_mapping.items()}

            print(f"✅ Enhanced veri yüklendi: {X.shape[0]} müşteri, {X.shape[1]} enhanced feature")
            print(f"📈 Enhanced Segment sayısı: {len(self.segment_mapping)}")
            print(f"🏷️ Enhanced Segmentler: {list(self.segment_mapping.keys())}")

            # Data validation
            if X.shape[1] != 14:
                print(f"⚠️ Uyarı: Beklenen 14 feature, bulunan {X.shape[1]} feature")

            # Train/Test split (80-20)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # Enhanced feature normalization
            print("🔧 Enhanced features normalize ediliyor...")
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            print(f"📊 Enhanced Training set: {X_train_scaled.shape}")
            print(f"📋 Enhanced Test set: {X_test_scaled.shape}")

            # Enhanced feature statistics
            print(f"\n📈 Enhanced Feature İstatistikleri:")
            for i, col in enumerate(self.enhanced_feature_columns):
                if i < X_train.shape[1]:
                    print(f"  {i + 1:2d}. {col:<25}: mean={X_train[:, i].mean():.2f}, std={X_train[:, i].std():.2f}")

            return X_train_scaled, X_test_scaled, y_train, y_test

        except FileNotFoundError as e:
            print(f"❌ Enhanced veri dosyası bulunamadı: {e}")
            print("💡 Önce ml_auto_segmentation_engine.py çalıştırın!")
            return None, None, None, None

    def build_enhanced_model(self, input_dim=14, num_classes=10):
        """
        Enhanced Neural Network mimarisi (14 features için optimize)
        """
        print("🧠 Enhanced Neural Network modeli oluşturuluyor...")
        print("🎯 Amaç: 14 enhanced features ile improved accuracy")

        # Enhanced Model Architecture (daha deep ve complex)
        model = tf.keras.Sequential([
            # Input layer - Enhanced marketing features
            tf.keras.layers.InputLayer(input_shape=(input_dim,), name='enhanced_marketing_features'),

            # Hidden Layer 1 - Enhanced pattern analysis (increased capacity)
            tf.keras.layers.Dense(128, activation='relu', name='enhanced_analysis_1'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),

            # Hidden Layer 2 - Deep customer behavior modeling
            tf.keras.layers.Dense(64, activation='relu', name='deep_customer_behavior'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.4),

            # Hidden Layer 3 - Advanced segment characteristics
            tf.keras.layers.Dense(32, activation='relu', name='advanced_segment_characteristics'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),

            # Hidden Layer 4 - Fine-tuned segmentation
            tf.keras.layers.Dense(16, activation='relu', name='fine_tuned_segmentation'),
            tf.keras.layers.Dropout(0.2),

            # Output layer - Enhanced segment prediction
            tf.keras.layers.Dense(num_classes, activation='softmax', name='enhanced_segment_prediction')
        ])

        # Enhanced model compilation - improved optimization
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', 'sparse_categorical_crossentropy']
        )

        self.model = model

        print("✅ Enhanced Model oluşturuldu!")
        print(f"📊 Enhanced Model parametreleri: {model.count_params():,}")
        print(f"🎯 Target: >93.6% accuracy (original model'den daha iyi)")

        # Enhanced Model özetini göster
        print(f"\n🏗️ ENHANCED MODEL MİMARİSİ:")
        model.summary()

        return model

    def train_enhanced_model(self, X_train, X_test, y_train, y_test, epochs=100):
        """
        Enhanced model eğitimi - improved performance için optimize
        """
        print("🎯 Enhanced model eğitimi başlıyor...")
        print("🚀 Target: Original %93.6'dan daha yüksek accuracy")

        # Enhanced callbacks - better performance için
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=20,  # Daha patient (enhanced model için)
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,  # Enhanced için daha patient
                min_lr=0.00001,
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                'data/processed/enhanced_customer_segmentation_model_best.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]

        # Enhanced model eğitimi
        start_time = datetime.now()

        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )

        end_time = datetime.now()
        training_duration = end_time - start_time

        print("✅ Enhanced Model eğitimi tamamlandı!")
        print(f"⏱️ Eğitim süresi: {training_duration}")

        # Enhanced eğitim sonuçları
        final_accuracy = max(self.history.history['val_accuracy'])
        final_loss = min(self.history.history['val_loss'])

        print(f"🎯 En iyi validation accuracy: {final_accuracy:.4f}")
        print(f"📉 En düşük validation loss: {final_loss:.4f}")

        # Performance comparison with original
        original_accuracy = 0.936  # Original model accuracy
        improvement = final_accuracy - original_accuracy

        print(f"\n📊 PERFORMANCE COMPARISON:")
        print(f"   Original Model: {original_accuracy:.3f}")
        print(f"   Enhanced Model: {final_accuracy:.3f}")
        print(f"   Improvement: {improvement:+.3f} ({improvement * 100:+.1f}%)")

        return self.history

    def evaluate_enhanced_model(self, X_test, y_test):
        """
        Enhanced model performansını değerlendir
        """
        print("📊 Enhanced Model performansı değerlendiriliyor...")

        # Test skorları
        test_loss, test_accuracy, _ = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"✅ Enhanced Test Accuracy: {test_accuracy:.4f}")
        print(f"📉 Enhanced Test Loss: {test_loss:.4f}")

        # Tahminler
        y_pred_prob = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_prob, axis=1)

        # Enhanced segment bazlı performans raporu
        print(f"\n📋 ENHANCED SEGMENT BAZLI PERFORMANS RAPORU:")
        target_names = [self.reverse_mapping[i] for i in range(len(self.reverse_mapping))]
        classification_rep = classification_report(y_test, y_pred, target_names=target_names)
        print(classification_rep)

        # Enhanced confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        self.plot_enhanced_confusion_matrix(cm, target_names)

        # Feature importance analysis (enhanced features için)
        self.analyze_feature_importance(X_test, y_test)

        return test_accuracy, y_pred, y_pred_prob

    def plot_enhanced_confusion_matrix(self, cm, target_names):
        """
        Enhanced confusion matrix görselleştirme
        """
        plt.figure(figsize=(14, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=target_names, yticklabels=target_names)
        plt.title('🎯 Enhanced Müşteri Segmentasyonu - Confusion Matrix\n(14 Features ile Geliştirilmiş Performans)')
        plt.xlabel('Tahmin Edilen Segment')
        plt.ylabel('Gerçek Segment')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('data/processed/enhanced_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()

    def analyze_feature_importance(self, X_test, y_test):
        """
        Enhanced features'ların önem analizi
        """
        print(f"\n🔍 ENHANCED FEATURE IMPORTANCE ANALYSIS:")

        # Simple feature importance via permutation
        baseline_accuracy = self.model.evaluate(X_test, y_test, verbose=0)[1]

        feature_importance = {}

        for i, feature_name in enumerate(self.enhanced_feature_columns):
            # Create permuted version
            X_test_permuted = X_test.copy()
            np.random.shuffle(X_test_permuted[:, i])

            # Evaluate with permuted feature
            permuted_accuracy = self.model.evaluate(X_test_permuted, y_test, verbose=0)[1]
            importance = baseline_accuracy - permuted_accuracy
            feature_importance[feature_name] = importance

        # Sort by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

        print("📊 Feature Importance Ranking:")
        for i, (feature, importance) in enumerate(sorted_features):
            category = "🎯 Original RFM" if i < 5 else "🧠 ML-Discovered"
            print(f"  {i + 1:2d}. {category} {feature:<25}: {importance:.4f}")

    def plot_enhanced_training_history(self):
        """
        Enhanced eğitim süreci grafiği
        """
        if self.history is None:
            print("❌ Enhanced Model henüz eğitilmemiş!")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Enhanced Accuracy grafiği
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy', linewidth=2)
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        ax1.axhline(y=0.936, color='red', linestyle='--', label='Original Model (93.6%)')
        ax1.set_title('🎯 Enhanced Model Accuracy\n(14 Features ile Geliştirilmiş)')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Enhanced Loss grafiği
        ax2.plot(self.history.history['loss'], label='Training Loss', linewidth=2)
        ax2.plot(self.history.history['val_loss'], label='Validation Loss', linewidth=2)
        ax2.set_title('📉 Enhanced Model Loss\n(14 Features Optimization)')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('data/processed/enhanced_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()

    def predict_enhanced_customer_segment(self, customer_data):
        """
        Enhanced features ile yeni müşteri segment tahmini
        """
        if isinstance(customer_data, list):
            if len(customer_data) != 14:
                print(f"❌ Hata: 14 enhanced feature bekleniyor, {len(customer_data)} alındı")
                return None
            customer_data = np.array(customer_data).reshape(1, -1)

        # Enhanced normalize
        customer_scaled = self.scaler.transform(customer_data)

        # Enhanced tahmin
        prediction_prob = self.model.predict(customer_scaled)
        predicted_segment_idx = np.argmax(prediction_prob)
        predicted_segment = self.reverse_mapping[predicted_segment_idx]
        confidence = prediction_prob[0][predicted_segment_idx]

        print(f"🎯 ENHANCED MÜŞTERİ SEGMENT TAHMİNİ:")
        print(f"📊 Enhanced girdi: 14 features")
        print(f"🏷️ Tahmin edilen segment: {predicted_segment}")
        print(f"📈 Enhanced güven skoru: {confidence:.4f}")

        return predicted_segment, confidence

    def save_enhanced_model(self, model_path="data/processed/enhanced_customer_segmentation_model.h5"):
        """
        Enhanced modeli kaydet
        """
        # Enhanced model kaydet
        self.model.save(model_path)

        # Enhanced scaler kaydet
        import joblib
        joblib.dump(self.scaler, "data/processed/enhanced_scaler.pkl")

        # Enhanced segment mapping kaydet
        with open("data/processed/enhanced_segment_mapping.json", "w") as f:
            json.dump(self.segment_mapping, f)

        print(f"✅ Enhanced Model kaydedildi: {model_path}")
        print(f"✅ Enhanced Scaler kaydedildi: enhanced_scaler.pkl")
        print(f"✅ Enhanced Segment mapping kaydedildi: enhanced_segment_mapping.json")

    def load_enhanced_model(self, model_path="data/processed/enhanced_customer_segmentation_model.h5"):
        """
        Enhanced modeli yükle
        """
        try:
            self.model = tf.keras.models.load_model(model_path)

            import joblib
            self.scaler = joblib.load("data/processed/enhanced_scaler.pkl")

            with open("data/processed/enhanced_segment_mapping.json", "r") as f:
                self.segment_mapping = json.load(f)

            self.reverse_mapping = {v: k for k, v in self.segment_mapping.items()}

            print(f"✅ Enhanced Model yüklendi: {model_path}")
            return True
        except Exception as e:
            print(f"❌ Enhanced Model yükleme hatası: {e}")
            return False


def main():
    """
    Enhanced Neural Network ana çalıştırma fonksiyonu
    """
    print("=" * 80)
    print("🎯 ENHANCED CUSTOMER SEGMENTATION NEURAL NETWORK")
    print("🧠 14 Features ile Geliştirilmiş Segmentasyon")
    print("🚀 Target: Original %93.6'dan daha yüksek accuracy")
    print("=" * 80)

    # Enhanced Model instance
    enhanced_model = EnhancedCustomerSegmentationModel()

    # Enhanced veri yükleme
    X_train, X_test, y_train, y_test = enhanced_model.load_enhanced_data()

    if X_train is None:
        print("❌ Enhanced veri yüklenemedi!")
        return None

    # Enhanced model oluşturma
    enhanced_model.build_enhanced_model(input_dim=X_train.shape[1], num_classes=len(enhanced_model.segment_mapping))

    # Enhanced model eğitimi
    enhanced_model.train_enhanced_model(X_train, X_test, y_train, y_test, epochs=75)

    # Enhanced performans değerlendirmesi
    accuracy, y_pred, y_pred_prob = enhanced_model.evaluate_enhanced_model(X_test, y_test)

    # Enhanced grafikleri çiz
    enhanced_model.plot_enhanced_training_history()

    # Enhanced model kaydet
    enhanced_model.save_enhanced_model()

    print(f"\n🚀 ENHANCED NEURAL NETWORK HAZIR!")
    print(f"📊 Enhanced Final Accuracy: {accuracy:.4f}")
    print(f"🎯 Performance: {'🏆 IMPROVED' if accuracy > 0.936 else '📈 TRAINING NEEDED'}")

    # Enhanced demo prediction
    print(f"\n🎯 Enhanced Demo Prediction:")
    demo_features = [
        30, 8, 2500.0, 312.5, 200,  # Original RFM features
        3, 2, 1, 365, 0.8, 2, 1.5, 0.7, 0.6  # Enhanced ML features
    ]

    result = enhanced_model.predict_enhanced_customer_segment(demo_features)

    return enhanced_model


if __name__ == "__main__":
    enhanced_model = main()