# src/segmentation/segment_engine.py

import tensorflow as tf
import numpy as np
import pandas as pd
import json
import joblib
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


class SegmentationEngine:
    """
    Pazarlama odaklı müşteri segmentasyon motoru
    Neural Network tabanlı segment tahmini ve kampanya önerileri
    """

    def __init__(self):
        self.model = None
        self.scaler = None
        self.segment_mapping = None
        self.reverse_mapping = None
        self.segment_profiles = self._define_segment_profiles()
        self.feature_columns = ['Recency', 'Frequency', 'Monetary', 'AvgOrderValue', 'CustomerValue']
        self.model_loaded = False

    def _define_segment_profiles(self):
        """
        Pazarlama segmentlerinin detaylı profil tanımları
        """
        return {
            'Champions': {
                'description': 'En değerli müşteriler - Yüksek sadakat, yüksek harcama',
                'characteristics': ['Düşük recency', 'Yüksek frequency', 'Yüksek monetary'],
                'marketing_priority': 'HIGH',
                'campaign_type': 'VIP_RETENTION',
                'recommended_actions': [
                    'VIP exclusive offers',
                    'Early access to new products',
                    'Personal account manager',
                    'Premium customer service'
                ],
                'retention_risk': 'LOW',
                'upsell_potential': 'HIGH',
                'churn_probability': 0.05,
                'expected_clv_multiplier': 3.5
            },
            'Loyal': {
                'description': 'Sadık müşteriler - Düzenli alışveriş yapan',
                'characteristics': ['Orta recency', 'Yüksek frequency', 'Orta-yüksek monetary'],
                'marketing_priority': 'HIGH',
                'campaign_type': 'LOYALTY_BOOST',
                'recommended_actions': [
                    'Loyalty points bonus',
                    'Cross-selling campaigns',
                    'Referral programs',
                    'Product recommendations'
                ],
                'retention_risk': 'LOW',
                'upsell_potential': 'MEDIUM',
                'churn_probability': 0.10,
                'expected_clv_multiplier': 2.8
            },
            'At Risk': {
                'description': 'Risk altındaki değerli müşteriler - Acil müdahale gerekli',
                'characteristics': ['Yüksek recency', 'Yüksek frequency geçmişi', 'Yüksek monetary geçmişi'],
                'marketing_priority': 'URGENT',
                'campaign_type': 'WIN_BACK',
                'recommended_actions': [
                    'Immediate discount offers',
                    'Personal outreach calls',
                    'Win-back email campaigns',
                    'Survey for feedback'
                ],
                'retention_risk': 'HIGH',
                'upsell_potential': 'HIGH',
                'churn_probability': 0.65,
                'expected_clv_multiplier': 2.0
            },
            'Potential Loyalists': {
                'description': 'Potansiyel sadık müşteriler - Gelişim aşamasında',
                'characteristics': ['Düşük recency', 'Orta frequency', 'Orta monetary'],
                'marketing_priority': 'MEDIUM',
                'campaign_type': 'NURTURING',
                'recommended_actions': [
                    'Product education campaigns',
                    'Loyalty program enrollment',
                    'Targeted promotions',
                    'Engagement campaigns'
                ],
                'retention_risk': 'MEDIUM',
                'upsell_potential': 'HIGH',
                'churn_probability': 0.25,
                'expected_clv_multiplier': 2.2
            },
            'New Customers': {
                'description': 'Yeni müşteriler - Onboarding aşamasında',
                'characteristics': ['Düşük recency', 'Düşük frequency', 'Değişken monetary'],
                'marketing_priority': 'HIGH',
                'campaign_type': 'ONBOARDING',
                'recommended_actions': [
                    'Welcome campaign series',
                    'Product tutorials',
                    'First purchase incentives',
                    'Customer service support'
                ],
                'retention_risk': 'HIGH',
                'upsell_potential': 'UNKNOWN',
                'churn_probability': 0.40,
                'expected_clv_multiplier': 1.5
            },
            'Promising': {
                'description': 'Umut verici müşteriler - Potansiyel gelişim',
                'characteristics': ['Orta-düşük recency', 'Düşük frequency', 'Orta monetary'],
                'marketing_priority': 'MEDIUM',
                'campaign_type': 'DEVELOPMENT',
                'recommended_actions': [
                    'Targeted offers',
                    'Product recommendations',
                    'Engagement campaigns',
                    'Value demonstration'
                ],
                'retention_risk': 'MEDIUM',
                'upsell_potential': 'MEDIUM',
                'churn_probability': 0.35,
                'expected_clv_multiplier': 1.8
            },
            'Need Attention': {
                'description': 'Dikkat gerektiren müşteriler - Orta risk',
                'characteristics': ['Orta recency', 'Orta frequency', 'Orta monetary'],
                'marketing_priority': 'MEDIUM',
                'campaign_type': 'ENGAGEMENT',
                'recommended_actions': [
                    'Re-engagement campaigns',
                    'Special offers',
                    'Feedback surveys',
                    'Product updates'
                ],
                'retention_risk': 'MEDIUM',
                'upsell_potential': 'MEDIUM',
                'churn_probability': 0.30,
                'expected_clv_multiplier': 1.6
            },
            'About to Sleep': {
                'description': 'Uyumaya başlayan müşteriler - Aktivasyon gerekli',
                'characteristics': ['Yüksek recency', 'Orta frequency geçmişi', 'Orta monetary'],
                'marketing_priority': 'MEDIUM',
                'campaign_type': 'REACTIVATION',
                'recommended_actions': [
                    'Reactivation campaigns',
                    'Limited time offers',
                    'Product reminders',
                    'Incentive programs'
                ],
                'retention_risk': 'MEDIUM',
                'upsell_potential': 'LOW',
                'churn_probability': 0.45,
                'expected_clv_multiplier': 1.3
            },
            'Hibernating': {
                'description': 'Uyuyan müşteriler - Düşük aktivite',
                'characteristics': ['Yüksek recency', 'Düşük frequency', 'Düşük monetary'],
                'marketing_priority': 'LOW',
                'campaign_type': 'RECOVERY',
                'recommended_actions': [
                    'Low-cost reactivation',
                    'Basic email campaigns',
                    'Special discounts',
                    'Survey campaigns'
                ],
                'retention_risk': 'HIGH',
                'upsell_potential': 'LOW',
                'churn_probability': 0.70,
                'expected_clv_multiplier': 0.8
            },
            'Lost': {
                'description': 'Kayıp müşteriler - Minimum yatırım',
                'characteristics': ['Çok yüksek recency', 'Düşük frequency', 'Düşük monetary'],
                'marketing_priority': 'LOW',
                'campaign_type': 'MINIMAL',
                'recommended_actions': [
                    'Minimal cost campaigns',
                    'Generic newsletters',
                    'Win-back attempts',
                    'Feedback collection'
                ],
                'retention_risk': 'VERY_HIGH',
                'upsell_potential': 'VERY_LOW',
                'churn_probability': 0.85,
                'expected_clv_multiplier': 0.5
            }
        }

    def load_models(self):
        """
        Eğitilmiş model ve preprocessing araçlarını yükle
        """
        try:
            print("🔄 Segmentasyon modeli yükleniyor...")

            # TensorFlow modelini yükle
            self.model = tf.keras.models.load_model("data/processed/customer_segmentation_model.h5")
            print("✅ Neural Network modeli yüklendi")

            # Scaler'ı yükle
            self.scaler = joblib.load("data/processed/scaler.pkl")
            print("✅ Scaler yüklendi")

            # Segment mapping'i yükle
            with open("data/processed/segment_mapping.json", "r") as f:
                self.segment_mapping = json.load(f)

            # Reverse mapping oluştur
            self.reverse_mapping = {v: k for k, v in self.segment_mapping.items()}
            print("✅ Segment mapping yüklendi")

            print(f"📊 Model hazır: {len(self.segment_mapping)} segment")
            self.model_loaded = True

            return True

        except Exception as e:
            print(f"❌ Model yükleme hatası: {str(e)}")
            return False

    def predict_segment(self, customer_data, customer_id=None):
        """
        Yeni müşteri için segment tahmini ve pazarlama analizi

        Args:
            customer_data: [Recency, Frequency, Monetary, AvgOrderValue, CustomerValue]
            customer_id: Opsiyonel müşteri ID'si
        """
        if not self.model_loaded:
            print("❌ Model yüklenmemiş! load_models() fonksiyonunu çalıştırın.")
            return None

        try:
            # Input validation
            if isinstance(customer_data, list):
                customer_data = np.array(customer_data).reshape(1, -1)
            elif isinstance(customer_data, dict):
                # Dict format: {'Recency': 30, 'Frequency': 5, ...}
                customer_array = []
                for col in self.feature_columns:
                    if col in customer_data:
                        customer_array.append(customer_data[col])
                    else:
                        raise ValueError(f"Missing feature: {col}")
                customer_data = np.array(customer_array).reshape(1, -1)

            # Veri normalizasyonu
            customer_scaled = self.scaler.transform(customer_data)

            # Neural Network tahmini
            prediction_probs = self.model.predict(customer_scaled, verbose=0)[0]

            # En yüksek probability'li segment
            predicted_idx = np.argmax(prediction_probs)
            predicted_segment = self.reverse_mapping[predicted_idx]
            confidence = prediction_probs[predicted_idx]

            # İkinci en yüksek segment (alternative)
            second_idx = np.argsort(prediction_probs)[-2]
            second_segment = self.reverse_mapping[second_idx]
            second_confidence = prediction_probs[second_idx]

            # Confidence level categorization
            if confidence >= 0.8:
                confidence_level = "HIGH"
            elif confidence >= 0.6:
                confidence_level = "MEDIUM"
            else:
                confidence_level = "LOW"

            # Segment profil bilgileri
            segment_profile = self.segment_profiles.get(predicted_segment, {})

            # Sonuç objesi
            result = {
                'customer_id': customer_id,
                'input_data': {
                    'Recency': customer_data[0][0],
                    'Frequency': customer_data[0][1],
                    'Monetary': customer_data[0][2],
                    'AvgOrderValue': customer_data[0][3],
                    'CustomerValue': customer_data[0][4]
                },
                'prediction': {
                    'segment': predicted_segment,
                    'confidence': float(confidence),
                    'confidence_level': confidence_level,
                    'alternative_segment': second_segment,
                    'alternative_confidence': float(second_confidence)
                },
                'segment_profile': segment_profile,
                'marketing_insights': self._generate_marketing_insights(
                    predicted_segment, confidence, customer_data[0]
                ),
                'all_probabilities': {
                    self.reverse_mapping[i]: float(prob)
                    for i, prob in enumerate(prediction_probs)
                },
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

            return result

        except Exception as e:
            print(f"❌ Prediction hatası: {str(e)}")
            return None

    def _generate_marketing_insights(self, segment, confidence, customer_data):
        """
        Pazarlama insights'ları generate et
        """
        segment_profile = self.segment_profiles.get(segment, {})

        insights = {
            'priority_level': segment_profile.get('marketing_priority', 'MEDIUM'),
            'campaign_type': segment_profile.get('campaign_type', 'STANDARD'),
            'churn_risk': segment_profile.get('churn_probability', 0.5),
            'clv_multiplier': segment_profile.get('expected_clv_multiplier', 1.0),
            'recommended_actions': segment_profile.get('recommended_actions', []),
            'budget_allocation': self._calculate_budget_allocation(segment, confidence),
            'communication_frequency': self._determine_communication_frequency(segment),
            'channel_preference': self._determine_preferred_channels(segment),
            'next_best_action': self._determine_next_best_action(segment, customer_data)
        }

        return insights

    def _calculate_budget_allocation(self, segment, confidence):
        """
        Segment ve confidence'a göre budget allocation
        """
        base_allocations = {
            'Champions': 1000,
            'Loyal': 750,
            'At Risk': 500,
            'Potential Loyalists': 400,
            'New Customers': 300,
            'Promising': 250,
            'Need Attention': 200,
            'About to Sleep': 150,
            'Hibernating': 100,
            'Lost': 50
        }

        base_amount = base_allocations.get(segment, 200)

        # Confidence multiplier
        if confidence >= 0.8:
            multiplier = 1.2
        elif confidence >= 0.6:
            multiplier = 1.0
        else:
            multiplier = 0.8

        return int(base_amount * multiplier)

    def _determine_communication_frequency(self, segment):
        """
        Segment'e göre iletişim sıklığı
        """
        frequencies = {
            'Champions': 'Weekly',
            'Loyal': 'Bi-weekly',
            'At Risk': 'Daily (for 2 weeks)',
            'Potential Loyalists': 'Bi-weekly',
            'New Customers': 'Daily (for 1 week)',
            'Promising': 'Weekly',
            'Need Attention': 'Weekly',
            'About to Sleep': 'Bi-weekly',
            'Hibernating': 'Monthly',
            'Lost': 'Quarterly'
        }

        return frequencies.get(segment, 'Monthly')

    def _determine_preferred_channels(self, segment):
        """
        Segment'e göre tercih edilen iletişim kanalları
        """
        channels = {
            'Champions': ['Email', 'SMS', 'Phone', 'In-App'],
            'Loyal': ['Email', 'SMS', 'In-App'],
            'At Risk': ['Phone', 'Email', 'SMS'],
            'Potential Loyalists': ['Email', 'In-App', 'Social Media'],
            'New Customers': ['Email', 'SMS', 'Tutorial'],
            'Promising': ['Email', 'Social Media'],
            'Need Attention': ['Email', 'SMS'],
            'About to Sleep': ['Email', 'SMS'],
            'Hibernating': ['Email'],
            'Lost': ['Email']
        }

        return channels.get(segment, ['Email'])

    def _determine_next_best_action(self, segment, customer_data):
        """
        Bir sonraki en iyi aksiyon önerisi
        """
        recency, frequency, monetary = customer_data[0], customer_data[1], customer_data[2]

        actions = {
            'Champions': 'Offer exclusive VIP program upgrade',
            'Loyal': 'Present premium product line',
            'At Risk': 'Immediate personalized offer + call',
            'Potential Loyalists': 'Enroll in loyalty program',
            'New Customers': 'Send welcome series + tutorial',
            'Promising': 'Targeted product recommendations',
            'Need Attention': 'Engagement survey + special offer',
            'About to Sleep': 'Limited time reactivation offer',
            'Hibernating': 'Win-back campaign with discount',
            'Lost': 'Minimal cost email campaign'
        }

        return actions.get(segment, 'Standard engagement campaign')

    def bulk_predict(self, customers_df):
        """
        Toplu müşteri segmentasyonu

        Args:
            customers_df: DataFrame with columns [CustomerID, Recency, Frequency, Monetary, AvgOrderValue, CustomerValue]
        """
        if not self.model_loaded:
            print("❌ Model yüklenmemiş!")
            return None

        results = []

        print(f"🔄 {len(customers_df)} müşteri için toplu segmentasyon...")

        for idx, row in customers_df.iterrows():
            customer_data = [
                row['Recency'], row['Frequency'], row['Monetary'],
                row['AvgOrderValue'], row['CustomerValue']
            ]

            customer_id = row.get('CustomerID', f'Customer_{idx}')

            result = self.predict_segment(customer_data, customer_id)

            if result:
                results.append({
                    'CustomerID': customer_id,
                    'Predicted_Segment': result['prediction']['segment'],
                    'Confidence': result['prediction']['confidence'],
                    'Confidence_Level': result['prediction']['confidence_level'],
                    'Marketing_Priority': result['marketing_insights']['priority_level'],
                    'Campaign_Type': result['marketing_insights']['campaign_type'],
                    'Budget_Allocation': result['marketing_insights']['budget_allocation'],
                    'Churn_Risk': result['marketing_insights']['churn_risk'],
                    'Next_Best_Action': result['marketing_insights']['next_best_action']
                })

        results_df = pd.DataFrame(results)
        print(f"✅ Toplu segmentasyon tamamlandı!")

        # Summary istatistikler
        print(f"\n📊 SEGMENTASYON ÖZETI:")
        segment_counts = results_df['Predicted_Segment'].value_counts()
        for segment, count in segment_counts.items():
            pct = (count / len(results_df)) * 100
            print(f"  {segment:<20}: {count:>4} müşteri ({pct:>5.1f}%)")

        return results_df

    def generate_segment_report(self, segment_name):
        """
        Belirli bir segment için detaylı rapor
        """
        if segment_name not in self.segment_profiles:
            print(f"❌ Segment bulunamadı: {segment_name}")
            return None

        profile = self.segment_profiles[segment_name]

        print("=" * 60)
        print(f"📊 SEGMENT RAPORU: {segment_name.upper()}")
        print("=" * 60)
        print(f"📋 Tanım: {profile['description']}")
        print(f"🎯 Pazarlama Önceliği: {profile['marketing_priority']}")
        print(f"📈 Kampanya Tipi: {profile['campaign_type']}")
        print(f"⚠️ Churn Olasılığı: {profile['churn_probability']:.0%}")
        print(f"💰 CLV Multiplier: {profile['expected_clv_multiplier']}x")

        print(f"\n🔍 Karakteristikler:")
        for char in profile['characteristics']:
            print(f"  • {char}")

        print(f"\n🎯 Önerilen Aksiyonlar:")
        for action in profile['recommended_actions']:
            print(f"  • {action}")

        return profile


def main():
    """
    Test ve demo fonksiyonu
    """
    print("🧮 SEGMENTASYON ENGINE DEMO")
    print("=" * 50)

    # Engine instance
    engine = SegmentationEngine()

    # Model yükle
    if not engine.load_models():
        print("❌ Model yükleme başarısız!")
        return

    # Test müşterisi
    test_customer = {
        'Recency': 30,
        'Frequency': 8,
        'Monetary': 2500.0,
        'AvgOrderValue': 312.5,
        'CustomerValue': 240
    }

    print(f"\n🎯 TEST MÜŞTERİSİ ANALİZİ:")
    result = engine.predict_segment(test_customer, "TEST_CUSTOMER_001")

    if result:
        print(f"🏷️ Segment: {result['prediction']['segment']}")
        print(f"📊 Confidence: {result['prediction']['confidence']:.3f}")
        print(f"🎯 Priority: {result['marketing_insights']['priority_level']}")
        print(f"💰 Budget: ${result['marketing_insights']['budget_allocation']}")
        print(f"📞 Next Action: {result['marketing_insights']['next_best_action']}")

    # Segment raporu
    print(f"\n" + "=" * 50)
    engine.generate_segment_report("Champions")

    print(f"\n🚀 Segmentation Engine hazır!")


if __name__ == "__main__":
    main()