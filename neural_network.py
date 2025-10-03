# src/models/neural_network.py

import tensorflow as tf
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd


class CustomerSegmentationModel:
    """
    Pazarlama parametreleri ile müşteri segmentasyonu yapan Neural Network
    """

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.segment_mapping = None
        self.reverse_mapping = None
        self.feature_columns = ['Recency', 'Frequency', 'Monetary', 'AvgOrderValue', 'CustomerValue']
        self.history = None

    def load_data(self):
        """
        TensorFlow hazır verilerini yükle ve pazarlama analizine hazırla
        """
        print("📊 Pazarlama verisi yükleniyor...")

        # TensorFlow dosyalarını yükle
        X = np.load("data/processed/X_features.npy")
        y = np.load("data/processed/y_labels.npy")

        # Segment mapping yükle
        with open("data/processed/segment_mapping.json", "r") as f:
            self.segment_mapping = json.load(f)

        # Reverse mapping oluştur (çıktı yorumu için)
        self.reverse_mapping = {v: k for k, v in self.segment_mapping.items()}

        print(f"✅ Veri yüklendi: {X.shape[0]} müşteri, {X.shape[1]} pazarlama parametresi")
        print(f"📈 Segment sayısı: {len(self.segment_mapping)}")
        print(f"🏷️ Segmentler: {list(self.segment_mapping.keys())}")

        # Train/Test split (80-20)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Pazarlama parametrelerini normalize et (StandardScaler)
        print("🔧 Pazarlama parametreleri normalize ediliyor...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        print(f"📊 Training set: {X_train_scaled.shape}")
        print(f"📋 Test set: {X_test_scaled.shape}")

        # Pazarlama parametresi istatistikleri
        print(f"\n📈 Pazarlama Parametresi İstatistikleri:")
        for i, col in enumerate(self.feature_columns):
            print(f"  {col:<15}: mean={X_train[:, i].mean():.2f}, std={X_train[:, i].std():.2f}")

        return X_train_scaled, X_test_scaled, y_train, y_test

    def build_model(self, input_dim=5, num_classes=9):
        """
        Pazarlama odaklı Neural Network modelini oluştur
        """
        print("🧠 Neural Network modeli oluşturuluyor...")
        print("🎯 Amaç: Pazarlama parametrelerinden müşteri segmenti tahmini")

        # Model mimarisi
        model = tf.keras.Sequential([
            # Input layer - Pazarlama parametreleri
            tf.keras.layers.InputLayer(input_shape=(input_dim,), name='marketing_features'),

            # Hidden Layer 1 - Pazarlama pattern analizi
            tf.keras.layers.Dense(64, activation='relu', name='marketing_analysis_1'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),

            # Hidden Layer 2 - Müşteri davranış modelleme
            tf.keras.layers.Dense(32, activation='relu', name='customer_behavior'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.4),

            # Hidden Layer 3 - Segment karakteristikleri
            tf.keras.layers.Dense(16, activation='relu', name='segment_characteristics'),
            tf.keras.layers.Dropout(0.3),

            # Output layer - Müşteri segmentleri
            tf.keras.layers.Dense(num_classes, activation='softmax', name='segment_prediction')
        ])

        # Model compilation - Pazarlama optimizasyonu
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', 'sparse_categorical_crossentropy']
        )

        self.model = model

        print("✅ Model oluşturuldu!")
        print(f"📊 Model parametreleri: {model.count_params():,}")

        # Model özetini göster
        print(f"\n🏗️ MODEL MİMARİSİ:")
        model.summary()

        return model

    def train_model(self, X_train, X_test, y_train, y_test, epochs=100):
        """
        Pazarlama verisi ile modeli eğit
        """
        print("🎯 Pazarlama tabanlı model eğitimi başlıyor...")

        # Callbacks - Pazarlama performansı için
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=0.0001,
                verbose=1
            )
        ]

        # Model eğitimi
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )

        print("✅ Model eğitimi tamamlandı!")

        # Eğitim sonuçları
        final_accuracy = max(self.history.history['val_accuracy'])
        print(f"🎯 En iyi validation accuracy: {final_accuracy:.4f}")

        return self.history

    def evaluate_model(self, X_test, y_test):
        """
        Pazarlama segmentasyonu performansını değerlendir
        """
        print("📊 Model performansı değerlendiriliyor...")

        # Test skorları
        test_loss, test_accuracy, _ = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"✅ Test Accuracy: {test_accuracy:.4f}")
        print(f"📉 Test Loss: {test_loss:.4f}")

        # Tahminler
        y_pred_prob = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_prob, axis=1)

        # Segment bazlı performans raporu
        print(f"\n📋 SEGMENT BAZLI PERFORMANS RAPORU:")
        target_names = [self.reverse_mapping[i] for i in range(len(self.reverse_mapping))]
        print(classification_report(y_test, y_pred, target_names=target_names))

        # Pazarlama önerileri için confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        self.plot_confusion_matrix(cm, target_names)

        return test_accuracy, y_pred, y_pred_prob

    def plot_confusion_matrix(self, cm, target_names):
        """
        Pazarlama segmentleri confusion matrix'i
        """
        plt.figure(figsize=(12, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=target_names, yticklabels=target_names)
        plt.title('🎯 Müşteri Segmentasyonu - Confusion Matrix\n(Pazarlama Performansı)')
        plt.xlabel('Tahmin Edilen Segment')
        plt.ylabel('Gerçek Segment')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('data/processed/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_training_history(self):
        """
        Eğitim süreci grafiği
        """
        if self.history is None:
            print("❌ Model henüz eğitilmemiş!")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Accuracy grafiği
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('🎯 Model Accuracy (Pazarlama Öğrenimi)')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)

        # Loss grafiği
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('📉 Model Loss (Pazarlama Optimizasyonu)')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig('data/processed/training_history.png', dpi=300, bbox_inches='tight')
        plt.show()

    def predict_customer_segment(self, customer_data):
        """
        Yeni müşteri için segment tahmini (Pazarlama kullanımı)
        """
        if isinstance(customer_data, list):
            customer_data = np.array(customer_data).reshape(1, -1)

        # Normalize et
        customer_scaled = self.scaler.transform(customer_data)

        # Tahmin
        prediction_prob = self.model.predict(customer_scaled)
        predicted_segment_idx = np.argmax(prediction_prob)
        predicted_segment = self.reverse_mapping[predicted_segment_idx]
        confidence = prediction_prob[0][predicted_segment_idx]

        print(f"🎯 MÜŞTERİ SEGMENT TAHMİNİ:")
        print(f"📊 Girdi verisi: {customer_data[0]}")
        print(f"🏷️ Tahmin edilen segment: {predicted_segment}")
        print(f"📈 Güven skoru: {confidence:.4f}")

        return predicted_segment, confidence

    def save_model(self, filepath="data/processed/customer_segmentation_model.h5"):
        """
        Eğitilmiş modeli kaydet
        """
        self.model.save(filepath)

        # Scaler'ı da kaydet
        import joblib
        joblib.dump(self.scaler, "data/processed/scaler.pkl")

        print(f"✅ Model kaydedildi: {filepath}")


def main():
    """
    Ana çalıştırma fonksiyonu - Pazarlama odaklı model
    """
    print("=" * 60)
    print("🎯 PAZARLAMA TABANLI MÜŞTERİ SEGMENTASYONU")
    print("🧠 Neural Network ile RFM Analizi")
    print("=" * 60)

    # Model instance
    model = CustomerSegmentationModel()

    # Veri yükleme
    X_train, X_test, y_train, y_test = model.load_data()

    # Model oluşturma
    model.build_model(input_dim=X_train.shape[1], num_classes=len(model.segment_mapping))

    # Model eğitimi
    model.train_model(X_train, X_test, y_train, y_test, epochs=50)

    # Performans değerlendirmesi
    accuracy, y_pred, y_pred_prob = model.evaluate_model(X_test, y_test)

    # Grafikleri çiz
    model.plot_training_history()

    # Model kaydet
    model.save_model()

    print(f"\n🚀 Pazarlama tabanlı segmentasyon modeli hazır!")
    print(f"📊 Final accuracy: {accuracy:.4f}")

    return model


if __name__ == "__main__":
    model = main()