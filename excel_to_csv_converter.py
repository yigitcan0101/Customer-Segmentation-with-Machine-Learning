# src/data/excel_to_csv_converter.py

import pandas as pd
import os


def convert_excel_to_csv():
    """
    online_retail_II.xlsx dosyasını CSV formatına dönüştürür
    """
    # Dosya yolları
    excel_path = "data/raw/online_retail_II.xlsx"
    csv_path = "data/processed/online_retail_data.csv"

    try:
        # Excel dosyasını oku
        print("📖 Excel dosyası okunuyor...")
        df = pd.read_excel(excel_path)

        # Temel bilgileri göster
        print(f"✅ Excel başarıyla okundu!")
        print(f"📊 Toplam satır: {len(df)}")
        print(f"📈 Toplam sütun: {len(df.columns)}")
        print(f"📋 Sütun isimleri: {list(df.columns)}")

        # CSV olarak kaydet
        df.to_csv(csv_path, index=False)
        print(f"✅ CSV dosyası oluşturuldu: {csv_path}")

        # İlk 5 satırı göster
        print("\n🔍 İlk 5 satır:")
        print(df.head())

        # Veri tipi bilgileri
        print("\n📋 Veri Tipleri:")
        print(df.dtypes)

        # Missing value kontrolü
        print("\n❓ Eksik Değerler:")
        print(df.isnull().sum())

        return df

    except FileNotFoundError:
        print(f"❌ HATA: Excel dosyası bulunamadı: {excel_path}")
        print("💡 Lütfen online_retail_II.xlsx dosyasını data/raw/ klasörüne koyun")
        return None

    except Exception as e:
        print(f"❌ HATA: {str(e)}")
        return None


if __name__ == "__main__":
    convert_excel_to_csv()