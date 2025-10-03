# src/data/enhanced_feature_extractor.py

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import os

warnings.filterwarnings('ignore')


class EnhancedFeatureExtractor:
    """
    Raw transaction verilerinden enhanced pazarlama features çıkartan motor
    RFM+ Intelligence: Category + Geographic + Price + Behavioral + Seasonal Features
    """

    def __init__(self):
        self.category_keywords = {
            'CHRISTMAS': ['CHRISTMAS', 'ADVENT', 'REINDEER', 'SANTA', 'XMAS', 'FESTIVE', 'HOLIDAY'],
            'HEART': ['HEART', 'LOVE', 'VALENTINE', 'ROMANTIC', 'SWEETHEART'],
            'HOME_DECOR': ['BUILDING BLOCK', 'DOOR MAT', 'DECORATION', 'ORNAMENT', 'WALL', 'HOME'],
            'HOT_WATER_BOTTLE': ['HOT WATER BOTTLE', 'HOTTIE', 'WARMER'],
            'RETRO_VINTAGE': ['RETRO', 'VINTAGE', 'VICTORIAN', 'ANTIQUE'],
            'FELTCRAFT': ['FELTCRAFT', 'CRAFT', 'DOLL', 'FELT'],
            'TEA_KITCHEN': ['TEA SET', 'BAKING SET', 'MUG', 'BOWL', 'PLATE', 'CUP'],
            'GARDEN': ['GARDEN', 'FLOWER', 'PLANT', 'SEED', 'BOTANICAL']
        }

        self.geographic_clusters = {
            'UK_CLUSTER': ['United Kingdom'],
            'EU_CORE': ['Germany', 'France', 'Netherlands', 'Belgium'],
            'EU_EXTENDED': ['EIRE', 'Austria', 'Portugal'],
            'INTERNATIONAL': ['USA', 'Australia', 'Poland']
        }

        self.price_tiers = {
            'BUDGET': (0.0, 2.0),
            'MID_TIER': (2.0, 5.0),
            'PREMIUM': (5.0, 10.0),
            'LUXURY': (10.0, float('inf'))
        }

    def extract_all_enhanced_features(self, online_retail_csv_path="data/processed/online_retail_data.csv",
                                      rfm_results_csv_path="data/processed/rfm_analysis_results.csv"):
        """
        Ana fonksiyon: Tüm enhanced features'ları çıkar ve RFM ile birleştir
        """
        print("🔍 ENHANCED FEATURE EXTRACTION BAŞLANIYOR...")
        print("=" * 60)

        # Veri yükleme
        print("📖 Veriler yükleniyor...")
        raw_data = pd.read_csv(online_retail_csv_path)
        rfm_data = pd.read_csv(rfm_results_csv_path)

        print(f"✅ Raw transactions: {len(raw_data):,} satır")
        print(f"✅ RFM customers: {len(rfm_data):,} müşteri")

        # Enhanced features extraction
        enhanced_features_list = []

        print(f"\n🧠 Enhanced features çıkarılıyor...")
        for idx, customer_id in enumerate(rfm_data['Customer ID'].unique()):
            if idx % 500 == 0:
                print(f"  İşlenen müşteri: {idx:,} / {len(rfm_data):,}")

            # Müşterinin transaction'larını al
            customer_transactions = raw_data[raw_data['Customer ID'] == customer_id]

            if len(customer_transactions) == 0:
                continue

            # Enhanced features hesapla
            features = self._extract_customer_features(customer_id, customer_transactions)
            enhanced_features_list.append(features)

        # DataFrame'e çevir
        enhanced_df = pd.DataFrame(enhanced_features_list)

        # RFM data ile merge et
        print(f"\n🔗 RFM verileri ile birleştiriliyor...")
        merged_data = rfm_data.merge(enhanced_df, on='Customer ID', how='left')

        # Missing values'ları fill et
        merged_data = self._handle_missing_values(merged_data)

        print(f"✅ Enhanced dataset hazır!")
        print(f"📊 Toplam feature sayısı: {len(merged_data.columns)}")
        print(f"👥 Toplam müşteri sayısı: {len(merged_data)}")

        # Kaydet
        output_path = "data/processed/enhanced_rfm_dataset.csv"
        merged_data.to_csv(output_path, index=False)
        print(f"💾 Enhanced dataset kaydedildi: {output_path}")

        # Feature summary
        self._print_feature_summary(merged_data)

        return merged_data

    def _extract_customer_features(self, customer_id, transactions):
        """
        Tek müşteri için tüm enhanced features'ları hesapla
        """
        features = {'Customer ID': customer_id}

        # 1. CATEGORY INTELLIGENCE
        features.update(self._extract_category_features(transactions))

        # 2. GEOGRAPHIC INTELLIGENCE
        features.update(self._extract_geographic_features(transactions))

        # 3. PRICE INTELLIGENCE
        features.update(self._extract_price_features(transactions))

        # 4. PURCHASE BEHAVIOR INTELLIGENCE
        features.update(self._extract_behavior_features(transactions))

        # 5. SEASONAL INTELLIGENCE
        features.update(self._extract_seasonal_features(transactions))

        return features

    def _extract_category_features(self, transactions):
        """
        Ürün kategori affinity features
        """
        total_items = len(transactions)
        category_features = {}

        for category, keywords in self.category_keywords.items():
            category_count = 0
            category_revenue = 0

            for _, row in transactions.iterrows():
                description = str(row['Description']).upper()
                if any(keyword in description for keyword in keywords):
                    category_count += 1
                    category_revenue += row['Price'] * row['Quantity']

            # Affinity score (0-1)
            affinity_score = category_count / total_items if total_items > 0 else 0

            # Revenue ratio
            total_revenue = (transactions['Price'] * transactions['Quantity']).sum()
            revenue_ratio = category_revenue / total_revenue if total_revenue > 0 else 0

            category_features[f'{category.lower()}_affinity'] = round(affinity_score, 4)
            category_features[f'{category.lower()}_revenue_ratio'] = round(revenue_ratio, 4)

        # Dominant category
        affinity_scores = {k: v for k, v in category_features.items() if k.endswith('_affinity')}
        dominant_category = max(affinity_scores, key=affinity_scores.get).replace('_affinity',
                                                                                  '') if affinity_scores else 'none'
        category_features['dominant_category'] = dominant_category

        return category_features

    def _extract_geographic_features(self, transactions):
        """
        Coğrafi davranış features
        """
        countries = transactions['Country'].unique()
        primary_country = transactions['Country'].mode()[0] if len(transactions) > 0 else 'Unknown'

        # Geographic cluster
        geographic_cluster = self._get_geographic_cluster(primary_country)

        # Cross-border behavior
        is_cross_border = len(countries) > 1
        country_diversity = len(countries)

        return {
            'primary_country': primary_country,
            'geographic_cluster_id': geographic_cluster,
            'is_cross_border_buyer': int(is_cross_border),
            'country_diversity_score': country_diversity
        }

    def _get_geographic_cluster(self, country):
        """
        Ülkeyi geographic cluster'a ata
        """
        for cluster_id, (cluster_name, cluster_countries) in enumerate(self.geographic_clusters.items()):
            if country in cluster_countries:
                return cluster_id
        return len(self.geographic_clusters)  # Unknown cluster

    def _extract_price_features(self, transactions):
        """
        Fiyat davranışı features
        """
        prices = transactions['Price']

        # Basic price statistics
        avg_price = prices.mean()
        price_std = prices.std()
        price_range = prices.max() - prices.min()

        # Price tier preference
        price_tier = self._get_price_tier(avg_price)

        # Price sensitivity (coefficient of variation)
        price_sensitivity = price_std / avg_price if avg_price > 0 else 0

        # Premium tolerance (% of purchases above premium threshold)
        premium_threshold = self.price_tiers['PREMIUM'][0]
        premium_purchases = (prices >= premium_threshold).sum()
        premium_tolerance = premium_purchases / len(prices) if len(prices) > 0 else 0

        return {
            'avg_price_preference': round(avg_price, 2),
            'price_tier_preference': price_tier,
            'price_sensitivity_score': round(price_sensitivity, 4),
            'premium_tolerance': round(premium_tolerance, 4),
            'price_range_behavior': round(price_range, 2)
        }

    def _get_price_tier(self, avg_price):
        """
        Ortalama fiyata göre price tier belirle
        """
        for tier_id, (tier_name, (min_price, max_price)) in enumerate(self.price_tiers.items()):
            if min_price <= avg_price < max_price:
                return tier_id
        return len(self.price_tiers) - 1  # Default to highest tier

    def _extract_behavior_features(self, transactions):
        """
        Satın alma davranış features
        """
        quantities = transactions['Quantity']

        # Quantity statistics
        avg_quantity = quantities.mean()
        max_quantity = quantities.max()

        # Purchase behavior classification
        behavior_type = self._classify_purchase_behavior(avg_quantity, max_quantity)

        # Return behavior (negative quantities)
        negative_qty = (quantities < 0).sum()
        return_frequency = negative_qty / len(quantities) if len(quantities) > 0 else 0

        # Bulk buying tendency
        bulk_threshold = 20  # 20+ items = bulk
        bulk_purchases = (quantities >= bulk_threshold).sum()
        bulk_tendency = bulk_purchases / len(quantities) if len(quantities) > 0 else 0

        # Transaction diversity (unique products)
        unique_products = transactions['Description'].nunique()
        product_diversity = unique_products / len(transactions) if len(transactions) > 0 else 0

        return {
            'avg_quantity_per_order': round(avg_quantity, 2),
            'max_quantity_behavior': int(max_quantity),
            'purchase_behavior_type': behavior_type,
            'return_frequency': round(return_frequency, 4),
            'bulk_buying_tendency': round(bulk_tendency, 4),
            'product_diversity_score': round(product_diversity, 4)
        }

    def _classify_purchase_behavior(self, avg_quantity, max_quantity):
        """
        Satın alma davranışını sınıflandır
        """
        if max_quantity >= 50:
            return 3  # WHOLESALE
        elif avg_quantity >= 10:
            return 2  # BULK
        elif avg_quantity >= 3:
            return 1  # GIFT_BUYER
        else:
            return 0  # INDIVIDUAL

    def _extract_seasonal_features(self, transactions):
        """
        Mevsimsel davranış features
        """
        # InvoiceDate'i datetime'a çevir
        transactions['InvoiceDate'] = pd.to_datetime(transactions['InvoiceDate'])

        # Month extraction
        months = transactions['InvoiceDate'].dt.month

        # Christmas season focus (November-December)
        christmas_months = [11, 12]
        christmas_purchases = months.isin(christmas_months).sum()
        christmas_intensity = christmas_purchases / len(months) if len(months) > 0 else 0

        # Season distribution
        seasons = {
            'WINTER': [12, 1, 2],
            'SPRING': [3, 4, 5],
            'SUMMER': [6, 7, 8],
            'AUTUMN': [9, 10, 11]
        }

        season_scores = {}
        for season, season_months in seasons.items():
            season_purchases = months.isin(season_months).sum()
            season_scores[f'{season.lower()}_preference'] = season_purchases / len(months) if len(months) > 0 else 0

        # Dominant season
        dominant_season = max(season_scores, key=season_scores.get).replace('_preference',
                                                                            '') if season_scores else 'none'

        return {
            'christmas_season_intensity': round(christmas_intensity, 4),
            'dominant_season': dominant_season,
            **{k: round(v, 4) for k, v in season_scores.items()}
        }

    def _handle_missing_values(self, merged_data):
        """
        Missing values'ları handle et
        """
        print("🔧 Missing values işleniyor...")

        # Numeric columns için 0 ile fill et
        numeric_columns = merged_data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col != 'Customer ID':
                merged_data[col].fillna(0, inplace=True)

        # Categorical columns için 'unknown' ile fill et
        categorical_columns = merged_data.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if col not in ['Customer ID', 'Segment']:
                merged_data[col].fillna('unknown', inplace=True)

        return merged_data

    def _print_feature_summary(self, enhanced_data):
        """
        Feature summary yazdır
        """
        print(f"\n📊 ENHANCED FEATURES SUMMARY:")
        print("=" * 60)

        # Original RFM features
        rfm_features = ['Recency', 'Frequency', 'Monetary', 'AvgOrderValue', 'CustomerValue']
        print(f"📈 RFM Features ({len(rfm_features)}): {', '.join(rfm_features)}")

        # Category features
        category_features = [col for col in enhanced_data.columns if
                             any(cat.lower() in col for cat in self.category_keywords.keys())]
        print(f"🏷️ Category Features ({len(category_features)}): {len(category_features)} features")

        # Geographic features
        geo_features = [col for col in enhanced_data.columns if
                        any(geo in col for geo in ['geographic', 'country', 'cross_border'])]
        print(f"🌍 Geographic Features ({len(geo_features)}): {', '.join(geo_features)}")

        # Price features
        price_features = [col for col in enhanced_data.columns if 'price' in col.lower()]
        print(f"💰 Price Features ({len(price_features)}): {', '.join(price_features)}")

        # Behavior features
        behavior_features = [col for col in enhanced_data.columns if any(
            behavior in col for behavior in ['behavior', 'quantity', 'return', 'bulk', 'diversity'])]
        print(f"🛒 Behavior Features ({len(behavior_features)}): {', '.join(behavior_features)}")

        # Seasonal features
        seasonal_features = [col for col in enhanced_data.columns if any(
            season in col for season in ['christmas', 'season', 'winter', 'spring', 'summer', 'autumn'])]
        print(f"📅 Seasonal Features ({len(seasonal_features)}): {', '.join(seasonal_features)}")

        print(
            f"\n🎯 TOPLAM ENHANCED FEATURES: {len(enhanced_data.columns) - len(rfm_features) - 3}")  # -3 for Customer ID, Segment, other metadata
        print(f"📊 Eski feature sayısı: {len(rfm_features)} → Yeni: {len(enhanced_data.columns)}")

        # Sample category affinity distribution
        print(f"\n🎄 Christmas Affinity Distribution:")
        if 'christmas_affinity' in enhanced_data.columns:
            christmas_stats = enhanced_data['christmas_affinity'].describe()
            print(f"  Mean: {christmas_stats['mean']:.4f}, Max: {christmas_stats['max']:.4f}")

        print(f"\n💝 Heart Theme Affinity Distribution:")
        if 'heart_affinity' in enhanced_data.columns:
            heart_stats = enhanced_data['heart_affinity'].describe()
            print(f"  Mean: {heart_stats['mean']:.4f}, Max: {heart_stats['max']:.4f}")


def main():
    """
    Enhanced Feature Extraction'ı çalıştır
    """
    print("🚀 ENHANCED FEATURE EXTRACTION - MAIN PIPELINE")
    print("=" * 70)

    # Feature extractor instance
    extractor = EnhancedFeatureExtractor()

    # Tüm enhanced features'ları çıkar
    enhanced_dataset = extractor.extract_all_enhanced_features()

    if enhanced_dataset is not None:
        print(f"\n🎉 ENHANCED FEATURE EXTRACTION TAMAMLANDI!")
        print(f"📁 Output: data/processed/enhanced_rfm_dataset.csv")
        print(f"👥 Müşteri sayısı: {len(enhanced_dataset):,}")
        print(f"📊 Feature sayısı: {len(enhanced_dataset.columns)}")

        # İlk 3 müşteriyi sample olarak göster
        print(f"\n🔍 Sample Enhanced Data (İlk 3 müşteri):")
        sample_cols = ['Customer ID', 'Segment', 'christmas_affinity', 'heart_affinity',
                       'geographic_cluster_id', 'price_tier_preference', 'purchase_behavior_type']
        available_cols = [col for col in sample_cols if col in enhanced_dataset.columns]
        print(enhanced_dataset[available_cols].head(3))

        return enhanced_dataset
    else:
        print("❌ Enhanced feature extraction başarısız!")
        return None


if __name__ == "__main__":
    enhanced_data = main()