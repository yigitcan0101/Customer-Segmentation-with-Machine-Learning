# src/data/rfm_calculator.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os


def calculate_rfm():
    """
    Online retail verisinden RFM analizi yapıp segmentasyon oluştur
    """
    print("📖 Online retail verisi okunuyor...")

    # Excel dosyasını masaüstünden oku
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    excel_path = os.path.join(desktop_path, "online_retail_II.xlsx")

    # Excel'i direkt oku (CSV yerine)
    try:
        df = pd.read_excel(excel_path)
        print(f"✅ Excel dosyası okundu: {len(df)} satır")
    except FileNotFoundError:
        print(f"❌ Excel dosyası bulunamadı: {excel_path}")
        return None

    # Veri temizleme
    print("🧹 Veri temizleme işlemi...")

    # Customer ID eksik olanları çıkar
    original_len = len(df)
    df = df.dropna(subset=['Customer ID'])
    print(f"📊 Customer ID temizleme: {original_len} → {len(df)} satır")

    # Negatif quantity ve price değerlerini çıkar (iade işlemleri)
    df = df[(df['Quantity'] > 0) & (df['Price'] > 0)]
    print(f"📊 Negatif değerler temizleme: {len(df)} satır")

    # InvoiceDate'i datetime'a çevir
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    # TotalAmount hesapla
    df['TotalAmount'] = df['Quantity'] * df['Price']

    # Referans tarih (analizin yapıldığı tarih - veri setindeki son tarihten 1 gün sonra)
    reference_date = df['InvoiceDate'].max() + timedelta(days=1)
    print(f"📅 Referans tarihi: {reference_date.strftime('%Y-%m-%d')}")

    # RFM HESAPLAMALARİ
    print("🔢 RFM hesaplamaları yapılıyor...")

    rfm = df.groupby('Customer ID').agg({
        'InvoiceDate': lambda x: (reference_date - x.max()).days,  # Recency
        'Invoice': 'nunique',  # Frequency (benzersiz invoice sayısı)
        'TotalAmount': 'sum'  # Monetary
    }).round(2)

    # Sütun isimlerini değiştir
    rfm.columns = ['Recency', 'Frequency', 'Monetary']

    # RFM skorlarını hesapla (1-5 arasında, 5 en iyi)
    print("🏷️ RFM skorları hesaplanıyor...")

    # Recency için ters sıralama (düşük recency = yüksek skor)
    rfm['R_Score'] = pd.qcut(rfm['Recency'].rank(method='first'), 5, labels=[5, 4, 3, 2, 1])
    rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
    rfm['M_Score'] = pd.qcut(rfm['Monetary'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])

    # RFM skorunu birleştir
    rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)

    # Segment ataması
    print("🎯 Segmentasyon yapılıyor...")

    def assign_segment(row):
        """RFM skorlarına göre segment ataması"""
        r, f, m = int(row['R_Score']), int(row['F_Score']), int(row['M_Score'])

        # Champions: En iyi müşteriler (yüksek RF ve M)
        if r >= 4 and f >= 4 and m >= 4:
            return 'Champions'

        # At Risk: Risk altında (düşük R, yüksek F ve M)
        elif r <= 2 and f >= 4 and m >= 4:
            return 'At Risk'

        # Loyal: Sadık müşteriler (yüksek F, orta-yüksek R ve M)
        elif f >= 3 and r >= 3 and m >= 3:
            return 'Loyal'

        # Potential Loyalists: Potansiyel sadık müşteriler (yüksek R, orta F ve M)
        elif r >= 4 and f >= 2 and m >= 2:
            return 'Potential Loyalists'

        # New Customers: Yeni müşteriler (çok yüksek R, düşük F)
        elif r >= 4 and f <= 2:
            return 'New Customers'

        # Promising: Umut verici müşteriler (orta-yüksek R, düşük F, orta M)
        elif r >= 3 and f <= 2 and m >= 2:
            return 'Promising'

        # Need Attention: Dikkat gerektirenler (orta R, F, M)
        elif r >= 2 and f >= 2 and m >= 2:
            return 'Need Attention'

        # About to Sleep: Uyumaya başlayanlar (düşük R, orta F ve M)
        elif r <= 2 and f >= 2 and m >= 2:
            return 'About to Sleep'

        # Cannot Lose Them: Kaybetmemeli (düşük R, en yüksek F ve M)
        elif r <= 2 and f >= 4 and m >= 4:
            return 'Cannot Lose Them'

        # Hibernating: Uyuyan müşteriler (düşük hepsi)
        elif r <= 2 and f <= 2 and m <= 2:
            return 'Hibernating'

        # Lost: Kaybedilen müşteriler (en düşük R, F, orta M)
        else:
            return 'Lost'

    # Segment ataması yap
    rfm['Segment'] = rfm.apply(assign_segment, axis=1)

    # Ek özellikler hesapla
    rfm['AvgOrderValue'] = (rfm['Monetary'] / rfm['Frequency']).round(2)
    rfm['CustomerValue'] = (rfm['F_Score'].astype(int) * rfm['M_Score'].astype(int) * 10).round(2)

    # Customer ID'yi index'den sütuna çevir
    rfm = rfm.reset_index()

    # Sütun sırasını düzenle
    rfm = rfm[['Customer ID', 'Recency', 'Frequency', 'Monetary', 'R_Score', 'F_Score', 'M_Score',
               'RFM_Score', 'AvgOrderValue', 'CustomerValue', 'Segment']]

    print("✅ RFM analizi tamamlandı!")
    print(f"📊 Toplam müşteri sayısı: {len(rfm)}")
    print(f"📈 Toplam segment sayısı: {rfm['Segment'].nunique()}")

    # Segment dağılımını göster
    print(f"\n🏷️ Segment dağılımı:")
    segment_dist = rfm['Segment'].value_counts()
    for segment, count in segment_dist.items():
        percentage = (count / len(rfm)) * 100
        print(f"  {segment:<20}: {count:>6} müşteri ({percentage:>5.1f}%)")

    # CSV olarak kaydet
    csv_path = "data/processed/rfm_analysis_results.csv"
    rfm.to_csv(csv_path, index=False)
    print(f"\n✅ RFM sonuçları kaydedildi: {csv_path}")

    # İlk 5 satırı göster
    print(f"\n🔍 İlk 5 müşteri:")
    print(rfm.head())

    return rfm


def prepare_for_tensorflow():
    """
    RFM verisini TensorFlow için hazırla
    """
    print("\n" + "=" * 50)
    print("🤖 TensorFlow için veri hazırlanıyor...")

    # RFM sonuçlarını oku
    csv_path = "data/processed/rfm_analysis_results.csv"
    df = pd.read_csv(csv_path)

    # Feature columns (numeric değerler)
    feature_columns = ['Recency', 'Frequency', 'Monetary', 'AvgOrderValue', 'CustomerValue']
    X = df[feature_columns].values

    # Segment encoding
    unique_segments = df['Segment'].unique()
    segment_mapping = {segment: idx for idx, segment in enumerate(unique_segments)}

    df['SegmentEncoded'] = df['Segment'].map(segment_mapping)
    y = df['SegmentEncoded'].values

    print(f"✅ TensorFlow için hazırlık tamamlandı!")
    print(f"📊 Feature matrix shape: {X.shape}")
    print(f"📋 Label vector shape: {y.shape}")
    print(f"🏷️ Segment mapping: {segment_mapping}")
    print(f"📈 Feature columns: {feature_columns}")

    # TensorFlow verisini kaydet
    np.save("data/processed/X_features.npy", X)
    np.save("data/processed/y_labels.npy", y)

    # Mapping'i kaydet
    import json
    with open("data/processed/segment_mapping.json", "w") as f:
        json.dump(segment_mapping, f)

    print(f"💾 TensorFlow dosyaları kaydedildi!")

    return X, y, feature_columns, segment_mapping


if __name__ == "__main__":
    # RFM analizi yap
    rfm_data = calculate_rfm()

    if rfm_data is not None:
        # TensorFlow için hazırla
        X, y, features, mapping = prepare_for_tensorflow()
        print("\n🚀 TensorFlow model training için her şey hazır!")