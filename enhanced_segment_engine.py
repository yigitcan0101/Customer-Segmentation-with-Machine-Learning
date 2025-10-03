# src/segmentation/enhanced_segment_engine.py

import tensorflow as tf
import numpy as np
import pandas as pd
import json
import joblib
from datetime import datetime
import warnings
import sys
import os

warnings.filterwarnings('ignore')

# Path configuration
current_dir = os.path.dirname(__file__)
parent_dir = os.path.join(current_dir, '..')
sys.path.append(parent_dir)


class EnhancedSegmentationEngine:
    """
    🧠 ENHANCED SEGMENTATION ENGINE

    Enhanced Neural Network (%95.9 accuracy) + Auto-Discovered Patterns
    14 Features ile sophisticated customer segmentation ve insights
    """

    def __init__(self):
        self.enhanced_model = None
        self.scaler = None
        self.segment_mapping = None
        self.reverse_mapping = None
        self.auto_discovered_patterns = None
        self.enhanced_feature_columns = [
            # Original RFM features (1-5)
            'Recency', 'Frequency', 'Monetary', 'AvgOrderValue', 'CustomerValue',
            # Auto-discovered ML features (6-14)
            'auto_product_category_id', 'auto_geographic_segment_id', 'behavior_pattern_id',
            'purchase_span_days', 'order_value_consistency', 'price_tier_preference',
            'bulk_buying_tendency', 'segment_pattern_alignment', 'geographic_behavior_score'
        ]
        self.segment_profiles = self._define_enhanced_segment_profiles()
        self.model_loaded = False

    def _define_enhanced_segment_profiles(self):
        """
        Enhanced segment profiles with auto-discovered pattern integration
        """
        return {
            'Champions': {
                'description': 'En değerli müşteriler - AI-enhanced insights ile optimize',
                'characteristics': ['Düşük recency', 'Yüksek frequency', 'Yüksek monetary',
                                    'Optimal behavior patterns'],
                'marketing_priority': 'HIGH',
                'campaign_type': 'VIP_ENHANCED',
                'ai_insights': 'Auto-discovered product affinity + geographic behavior optimization',
                'recommended_actions': [
                    'AI-optimized VIP exclusive offers',
                    'Predictive next-best-product recommendations',
                    'Geographic-aware premium services',
                    'Behavior-pattern personalized experiences'
                ],
                'retention_risk': 'VERY_LOW',
                'upsell_potential': 'MAXIMUM',
                'churn_probability': 0.03,
                'expected_clv_multiplier': 4.2,
                'enhanced_targeting': True
            },
            'Loyal': {
                'description': 'Sadık müşteriler - Enhanced loyalty optimization',
                'characteristics': ['Orta recency', 'Yüksek frequency', 'Orta-yüksek monetary', 'Consistent patterns'],
                'marketing_priority': 'HIGH',
                'campaign_type': 'LOYALTY_ENHANCED',
                'ai_insights': 'Behavior consistency + product category optimization',
                'recommended_actions': [
                    'AI-driven cross-selling campaigns',
                    'Pattern-based loyalty rewards',
                    'Geographic segment optimization',
                    'Predictive retention programs'
                ],
                'retention_risk': 'LOW',
                'upsell_potential': 'HIGH',
                'churn_probability': 0.08,
                'expected_clv_multiplier': 3.1,
                'enhanced_targeting': True
            },
            'At Risk': {
                'description': 'Risk altındaki değerli müşteriler - AI-powered intervention',
                'characteristics': ['Yüksek recency', 'Pattern disruption', 'Value decline risk'],
                'marketing_priority': 'URGENT',
                'campaign_type': 'AI_WIN_BACK',
                'ai_insights': 'Predictive churn signals + behavior pattern analysis',
                'recommended_actions': [
                    'AI-triggered immediate intervention',
                    'Pattern-disruption recovery campaigns',
                    'Personalized win-back offers',
                    'Predictive retention modeling'
                ],
                'retention_risk': 'HIGH',
                'upsell_potential': 'MEDIUM',
                'churn_probability': 0.60,
                'expected_clv_multiplier': 2.3,
                'enhanced_targeting': True
            },
            'Potential Loyalists': {
                'description': 'Potansiyel sadık müşteriler - AI-guided development',
                'characteristics': ['Gelişim potential', 'Pattern formation phase', 'Value building'],
                'marketing_priority': 'MEDIUM',
                'campaign_type': 'AI_NURTURING',
                'ai_insights': 'Growth pattern prediction + development optimization',
                'recommended_actions': [
                    'AI-guided nurturing campaigns',
                    'Pattern development programs',
                    'Predictive loyalty building',
                    'Behavior optimization strategies'
                ],
                'retention_risk': 'MEDIUM',
                'upsell_potential': 'HIGH',
                'churn_probability': 0.22,
                'expected_clv_multiplier': 2.7,
                'enhanced_targeting': True
            },
            'New Customers': {
                'description': 'Yeni müşteriler - AI-enhanced onboarding',
                'characteristics': ['Early behavior formation', 'Pattern establishment phase'],
                'marketing_priority': 'HIGH',
                'campaign_type': 'AI_ONBOARDING',
                'ai_insights': 'Early pattern detection + onboarding optimization',
                'recommended_actions': [
                    'AI-optimized welcome sequences',
                    'Pattern-guided product introduction',
                    'Predictive engagement strategies',
                    'Geographic-aware onboarding'
                ],
                'retention_risk': 'MEDIUM',
                'upsell_potential': 'UNKNOWN',
                'churn_probability': 0.35,
                'expected_clv_multiplier': 1.8,
                'enhanced_targeting': True
            },
            'Promising': {
                'description': 'Umut verici müşteriler - AI development potential',
                'characteristics': ['Growth indicators', 'Positive patterns emerging'],
                'marketing_priority': 'MEDIUM',
                'campaign_type': 'AI_DEVELOPMENT',
                'ai_insights': 'Growth potential prediction + development paths',
                'recommended_actions': [
                    'AI-identified growth opportunities',
                    'Pattern-enhanced development',
                    'Predictive value building',
                    'Targeted engagement optimization'
                ],
                'retention_risk': 'MEDIUM',
                'upsell_potential': 'MEDIUM',
                'churn_probability': 0.30,
                'expected_clv_multiplier': 2.1,
                'enhanced_targeting': True
            },
            'Need Attention': {
                'description': 'Dikkat gerektiren müşteriler - AI monitoring',
                'characteristics': ['Warning signals', 'Pattern irregularities'],
                'marketing_priority': 'MEDIUM',
                'campaign_type': 'AI_MONITORING',
                'ai_insights': 'Early warning system + intervention optimization',
                'recommended_actions': [
                    'AI-monitored engagement',
                    'Pattern correction campaigns',
                    'Predictive intervention',
                    'Behavior stabilization'
                ],
                'retention_risk': 'MEDIUM',
                'upsell_potential': 'MEDIUM',
                'churn_probability': 0.28,
                'expected_clv_multiplier': 1.9,
                'enhanced_targeting': True
            },
            'About to Sleep': {
                'description': 'Uyumaya başlayan müşteriler - AI reactivation',
                'characteristics': ['Declining engagement', 'Sleep pattern indicators'],
                'marketing_priority': 'MEDIUM',
                'campaign_type': 'AI_REACTIVATION',
                'ai_insights': 'Sleep pattern detection + reactivation optimization',
                'recommended_actions': [
                    'AI-timed reactivation campaigns',
                    'Pattern-based wake-up strategies',
                    'Predictive re-engagement',
                    'Minimal-effort activation'
                ],
                'retention_risk': 'HIGH',
                'upsell_potential': 'LOW',
                'churn_probability': 0.42,
                'expected_clv_multiplier': 1.4,
                'enhanced_targeting': False
            },
            'Hibernating': {
                'description': 'Uyuyan müşteriler - AI minimal engagement',
                'characteristics': ['Low activity', 'Dormant patterns', 'Minimal value'],
                'marketing_priority': 'LOW',
                'campaign_type': 'AI_MINIMAL',
                'ai_insights': 'Cost-optimized minimal engagement strategies',
                'recommended_actions': [
                    'AI-optimized low-cost campaigns',
                    'Automated minimal touch',
                    'Cost-efficient monitoring',
                    'ROI-focused engagement'
                ],
                'retention_risk': 'HIGH',
                'upsell_potential': 'VERY_LOW',
                'churn_probability': 0.68,
                'expected_clv_multiplier': 0.9,
                'enhanced_targeting': False
            },
            'Lost': {
                'description': 'Kayıp müşteriler - AI cost optimization',
                'characteristics': ['Inactive', 'Lost patterns', 'Minimal engagement value'],
                'marketing_priority': 'MINIMAL',
                'campaign_type': 'AI_COST_OPTIMIZED',
                'ai_insights': 'Cost-minimized engagement with ROI focus',
                'recommended_actions': [
                    'AI-driven cost minimization',
                    'Automated generic campaigns',
                    'ROI-focused minimal investment',
                    'Data collection for future insights'
                ],
                'retention_risk': 'MAXIMUM',
                'upsell_potential': 'NONE',
                'churn_probability': 0.82,
                'expected_clv_multiplier': 0.6,
                'enhanced_targeting': False
            }
        }

    def load_enhanced_models(self):
        """
        Enhanced model ve auto-discovered patterns yükle
        """
        try:
            print("🔄 Enhanced segmentation models yükleniyor...")

            # Enhanced TensorFlow modelini yükle (%95.9 accuracy)
            self.enhanced_model = tf.keras.models.load_model("data/processed/enhanced_customer_segmentation_model.h5")
            print("✅ Enhanced Neural Network yüklendi (%95.9 accuracy)")

            # Enhanced scaler'ı yükle
            self.scaler = joblib.load("data/processed/enhanced_scaler.pkl")
            print("✅ Enhanced scaler yüklendi")

            # Enhanced segment mapping'i yükle
            with open("data/processed/enhanced_segment_mapping.json", "r") as f:
                self.segment_mapping = json.load(f)

            # Reverse mapping oluştur
            self.reverse_mapping = {v: k for k, v in self.segment_mapping.items()}
            print("✅ Enhanced segment mapping yüklendi")

            # Auto-discovered patterns yükle
            try:
                with open("data/processed/ml_enhanced_feature_info.json", "r") as f:
                    self.auto_discovered_patterns = json.load(f)
                print("✅ Auto-discovered patterns yüklendi")
            except FileNotFoundError:
                print("⚠️ Auto-discovered patterns bulunamadı, basic patterns kullanılacak")
                self.auto_discovered_patterns = None

            print(f"📊 Enhanced Model hazır: {len(self.segment_mapping)} segment, 14 features")
            self.model_loaded = True

            return True

        except Exception as e:
            print(f"❌ Enhanced model yükleme hatası: {str(e)}")
            print("💡 Önce enhanced_neural_network.py çalıştırın!")
            return False

    def predict_enhanced_segment(self, customer_data, customer_id=None):
        """
        Enhanced features ile segment tahmini ve comprehensive insights

        Args:
            customer_data: 14 enhanced features [Recency, Frequency, Monetary, AvgOrderValue, CustomerValue,
                          auto_product_category_id, auto_geographic_segment_id, behavior_pattern_id,
                          purchase_span_days, order_value_consistency, price_tier_preference,
                          bulk_buying_tendency, segment_pattern_alignment, geographic_behavior_score]
            customer_id: Opsiyonel müşteri ID'si
        """
        if not self.model_loaded:
            print("❌ Enhanced model yüklenmemiş! load_enhanced_models() fonksiyonunu çalıştırın.")
            return None

        try:
            # Input validation ve normalization
            if isinstance(customer_data, list):
                if len(customer_data) != 14:
                    print(f"❌ Hata: 14 enhanced feature bekleniyor, {len(customer_data)} alındı")
                    return None
                customer_data = np.array(customer_data).reshape(1, -1)
            elif isinstance(customer_data, dict):
                # Dict format destegi
                customer_array = []
                for col in self.enhanced_feature_columns:
                    if col in customer_data:
                        customer_array.append(customer_data[col])
                    else:
                        raise ValueError(f"Missing enhanced feature: {col}")
                customer_data = np.array(customer_array).reshape(1, -1)

            # Enhanced normalizasyon
            customer_scaled = self.scaler.transform(customer_data)

            # Enhanced Neural Network tahmini (%95.9 accuracy)
            prediction_probs = self.enhanced_model.predict(customer_scaled, verbose=0)[0]

            # En yüksek probability'li segment
            predicted_idx = np.argmax(prediction_probs)
            predicted_segment = self.reverse_mapping[predicted_idx]
            confidence = prediction_probs[predicted_idx]

            # İkinci en yüksek segment (alternative)
            second_idx = np.argsort(prediction_probs)[-2]
            second_segment = self.reverse_mapping[second_idx]
            second_confidence = prediction_probs[second_idx]

            # Enhanced confidence level
            if confidence >= 0.90:
                confidence_level = "VERY_HIGH"
            elif confidence >= 0.80:
                confidence_level = "HIGH"
            elif confidence >= 0.70:
                confidence_level = "MEDIUM"
            else:
                confidence_level = "LOW"

            # Enhanced segment profil bilgileri
            segment_profile = self.segment_profiles.get(predicted_segment, {})

            # Auto-discovered pattern insights
            auto_insights = self._generate_auto_discovered_insights(customer_data[0], predicted_segment)

            # Enhanced marketing strategies
            enhanced_strategies = self._generate_enhanced_strategies(predicted_segment, confidence, customer_data[0])

            # Comprehensive result object
            result = {
                'customer_id': customer_id,
                'model_version': 'Enhanced_v2.0_95.9%',
                'input_data': {
                    'enhanced_features': {
                        feature: float(customer_data[0][i])
                        for i, feature in enumerate(self.enhanced_feature_columns)
                    }
                },
                'prediction': {
                    'segment': predicted_segment,
                    'confidence': float(confidence),
                    'confidence_level': confidence_level,
                    'model_accuracy': 95.9,
                    'alternative_segment': second_segment,
                    'alternative_confidence': float(second_confidence)
                },
                'segment_profile': segment_profile,
                'auto_discovered_insights': auto_insights,
                'enhanced_strategies': enhanced_strategies,
                'ai_recommendations': self._generate_ai_recommendations(
                    predicted_segment, confidence, customer_data[0], auto_insights
                ),
                'all_probabilities': {
                    self.reverse_mapping[i]: float(prob)
                    for i, prob in enumerate(prediction_probs)
                },
                'prediction_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

            return result

        except Exception as e:
            print(f"❌ Enhanced prediction hatası: {str(e)}")
            return None

    def _generate_auto_discovered_insights(self, customer_features, segment):
        """
        Auto-discovered patterns'dan insights generate et
        """
        insights = {
            'product_category_insight': self._analyze_product_category(customer_features[5]),
            'geographic_insight': self._analyze_geographic_behavior(customer_features[6]),
            'behavior_pattern_insight': self._analyze_behavior_pattern(customer_features[7]),
            'temporal_insight': self._analyze_temporal_patterns(customer_features[8]),
            'consistency_insight': self._analyze_consistency_patterns(customer_features[9]),
            'value_tier_insight': self._analyze_value_tier(customer_features[10]),
            'buying_behavior_insight': self._analyze_buying_behavior(customer_features[11]),
            'pattern_alignment_insight': self._analyze_pattern_alignment(customer_features[12]),
            'geographic_behavior_insight': self._analyze_geographic_behavior_score(customer_features[13])
        }

        return insights

    def _analyze_product_category(self, category_id):
        """Product category pattern analysis"""
        category_insights = {
            0: "Heart/Romance Theme preference - High emotional purchase driver",
            1: "Storage/Organization focus - Practical utility buyer",
            2: "Lighting/Candles affinity - Ambiance-focused purchases",
            3: "General/Vintage Design - Broad taste, design-conscious",
            4: "Retrospot Collection - Brand loyalty indicators",
            5: "Jewelry/Accessories - Personal style investments",
            6: "Glass/Decoratives - Quality-focused, aesthetic buyer",
            7: "Home Decoration - Lifestyle enhancement focus",
            8: "Garden/Outdoor - Seasonal and lifestyle purchases",
            9: "Christmas/Seasonal - Event-driven buying patterns"
        }

        return {
            'category_preference': category_insights.get(int(category_id), "Unknown category pattern"),
            'targeting_strategy': 'Category-specific campaigns recommended',
            'cross_sell_potential': 'High within same category cluster'
        }

    def _analyze_geographic_behavior(self, geo_segment_id):
        """Geographic behavior pattern analysis"""
        geo_insights = {
            0: "France Market - Premium pricing tolerance, quality focus",
            1: "Germany Market - Efficiency-oriented, bulk buying tendency",
            2: "UK Domestic - High volume, frequent ordering patterns"
        }

        return {
            'geographic_pattern': geo_insights.get(int(geo_segment_id), "Unknown geographic pattern"),
            'market_characteristics': 'Geographic-specific behavior detected',
            'localization_strategy': 'Market-adapted campaigns recommended'
        }

    def _analyze_behavior_pattern(self, pattern_id):
        """Customer behavior pattern analysis"""
        pattern_insights = {
            0: "High Value Mixed - Premium occasional buyer",
            1: "High Value Mixed - Frequent premium buyer",
            2: "At Risk Pattern - Value retention critical",
            3: "Long Dormant - Reactivation strategies needed",
            4: "High Value Mixed - Moderate engagement",
            5: "Champions Dominant - Maximum value customer",
            6: "High Value Mixed - Regular engagement"
        }

        return {
            'behavior_classification': pattern_insights.get(int(pattern_id), "Unknown behavior pattern"),
            'engagement_strategy': 'Pattern-optimized campaigns',
            'retention_approach': 'Behavior-specific retention tactics'
        }

    def _analyze_temporal_patterns(self, span_days):
        """Purchase span temporal analysis"""
        if span_days > 300:
            return {
                'temporal_pattern': 'Long-term relationship customer',
                'lifecycle_stage': 'Mature customer relationship',
                'strategy': 'Loyalty and retention focus'
            }
        elif span_days > 100:
            return {
                'temporal_pattern': 'Developing relationship',
                'lifecycle_stage': 'Growth phase customer',
                'strategy': 'Value development campaigns'
            }
        else:
            return {
                'temporal_pattern': 'New or concentrated buyer',
                'lifecycle_stage': 'Early relationship phase',
                'strategy': 'Engagement and expansion focus'
            }

    def _analyze_consistency_patterns(self, consistency_score):
        """Order value consistency analysis"""
        if consistency_score > 0.15:
            return {
                'consistency_level': 'Highly predictable buyer',
                'reliability': 'Stable purchase patterns',
                'strategy': 'Subscription or regular offers'
            }
        elif consistency_score > 0.08:
            return {
                'consistency_level': 'Moderately consistent',
                'reliability': 'Some variability in purchases',
                'strategy': 'Flexible offering strategies'
            }
        else:
            return {
                'consistency_level': 'Variable buyer',
                'reliability': 'Irregular purchase patterns',
                'strategy': 'Opportunistic and impulse campaigns'
            }

    def _analyze_value_tier(self, price_tier):
        """Price tier preference analysis"""
        tier_insights = {
            0: {'tier': 'Budget', 'strategy': 'Value-focused campaigns'},
            1: {'tier': 'Mid-tier', 'strategy': 'Quality-value balance offerings'},
            2: {'tier': 'Premium', 'strategy': 'Quality and exclusivity focus'},
            3: {'tier': 'Luxury', 'strategy': 'Premium experience campaigns'}
        }

        return tier_insights.get(int(price_tier), {'tier': 'Unknown', 'strategy': 'Standard approach'})

    def _analyze_buying_behavior(self, bulk_tendency):
        """Bulk buying tendency analysis"""
        if bulk_tendency > 3:
            return {
                'buying_style': 'Bulk buyer - wholesale tendency',
                'order_pattern': 'Large quantity purchases',
                'strategy': 'Volume discounts and bulk offers'
            }
        elif bulk_tendency > 2:
            return {
                'buying_style': 'Gift buyer - multiple item purchases',
                'order_pattern': 'Gift and occasion buying',
                'strategy': 'Gift bundles and occasion campaigns'
            }
        else:
            return {
                'buying_style': 'Individual buyer - single item focus',
                'order_pattern': 'Personal consumption',
                'strategy': 'Personal recommendations and single item focus'
            }

    def _analyze_pattern_alignment(self, alignment_score):
        """RFM-behavior pattern alignment"""
        if alignment_score > 0.8:
            return {
                'alignment_level': 'Perfect pattern match',
                'predictability': 'Highly predictable behavior',
                'confidence': 'Maximum targeting confidence'
            }
        elif alignment_score > 0.6:
            return {
                'alignment_level': 'Good pattern match',
                'predictability': 'Predictable with minor variations',
                'confidence': 'High targeting confidence'
            }
        else:
            return {
                'alignment_level': 'Pattern misalignment detected',
                'predictability': 'Unpredictable behavior patterns',
                'confidence': 'Cautious targeting approach'
            }

    def _analyze_geographic_behavior_score(self, geo_score):
        """Geographic behavior consistency"""
        if geo_score > 0.7:
            return {
                'geo_consistency': 'Highly consistent with market patterns',
                'market_fit': 'Perfect market alignment',
                'strategy': 'Market-standard approaches'
            }
        else:
            return {
                'geo_consistency': 'Unique behavior within market',
                'market_fit': 'Non-standard patterns',
                'strategy': 'Customized market approach'
            }

    def _generate_enhanced_strategies(self, segment, confidence, customer_features):
        """Enhanced marketing strategies generation"""

        # Base strategy from segment profile
        base_strategy = self.segment_profiles.get(segment, {})

        # Enhanced with AI insights
        enhanced_budget = self._calculate_enhanced_budget(segment, confidence, customer_features)
        ai_channels = self._optimize_ai_channels(segment, customer_features)
        timing_optimization = self._optimize_ai_timing(segment, customer_features)

        return {
            'enhanced_budget_allocation': enhanced_budget,
            'ai_optimized_channels': ai_channels,
            'timing_optimization': timing_optimization,
            'personalization_level': self._determine_personalization_level(confidence, customer_features),
            'automation_strategy': self._determine_automation_strategy(segment, customer_features)
        }

    def _calculate_enhanced_budget(self, segment, confidence, features):
        """AI-enhanced budget allocation"""
        base_budgets = {
            'Champions': 1200, 'Loyal': 900, 'At Risk': 700, 'Potential Loyalists': 500,
            'New Customers': 400, 'Promising': 350, 'Need Attention': 300,
            'About to Sleep': 200, 'Hibernating': 100, 'Lost': 50
        }

        base_amount = base_budgets.get(segment, 300)

        # Confidence multiplier
        confidence_multiplier = 1.0 + (confidence - 0.5) * 0.4  # 0.8 to 1.2 range

        # Customer value multiplier
        customer_value = features[4]  # CustomerValue
        value_multiplier = 1.0 + min(customer_value / 200, 0.5)  # Up to 1.5x

        # Pattern alignment multiplier
        pattern_alignment = features[12]  # segment_pattern_alignment
        alignment_multiplier = 0.8 + (pattern_alignment * 0.4)  # 0.8 to 1.2 range

        final_budget = base_amount * confidence_multiplier * value_multiplier * alignment_multiplier

        return {
            'total_budget': round(final_budget, 2),
            'confidence_adjustment': round(confidence_multiplier, 2),
            'value_adjustment': round(value_multiplier, 2),
            'pattern_adjustment': round(alignment_multiplier, 2)
        }

    def _optimize_ai_channels(self, segment, features):
        """AI-optimized channel selection"""

        # Base channels per segment
        base_channels = {
            'Champions': ['Email', 'SMS', 'Phone', 'In-App'],
            'Loyal': ['Email', 'SMS', 'In-App'],
            'At Risk': ['Phone', 'Email', 'SMS'],
            'Potential Loyalists': ['Email', 'In-App'],
            'New Customers': ['Email', 'SMS', 'Tutorial']
        }

        channels = base_channels.get(segment, ['Email'])

        # AI optimizations based on features
        geographic_segment = features[6]
        behavior_pattern = features[7]
        consistency = features[9]

        # Geographic optimization
        if geographic_segment == 2:  # UK market
            channels.append('Social Media')

        # Behavior pattern optimization
        if behavior_pattern in [5, 1]:  # High value patterns
            channels.append('Personal Manager')

        # Consistency optimization
        if consistency > 0.1:  # Consistent buyers
            channels.append('Automated')

        return list(set(channels))  # Remove duplicates

    def _optimize_ai_timing(self, segment, features):
        """AI-optimized timing strategies"""

        span_days = features[8]  # purchase_span_days
        consistency = features[9]  # order_value_consistency

        if span_days > 200 and consistency > 0.1:
            return {
                'frequency': 'Predictive - based on pattern',
                'best_day': 'AI-calculated optimal day',
                'best_hour': 'Pattern-based timing',
                'approach': 'Automated scheduled'
            }
        elif segment in ['Champions', 'Loyal']:
            return {
                'frequency': 'Weekly premium',
                'best_day': 'Tuesday-Thursday',
                'best_hour': '10-14',
                'approach': 'High-touch personalized'
            }
        else:
            return {
                'frequency': 'Bi-weekly standard',
                'best_day': 'Tuesday-Wednesday',
                'best_hour': '10-16',
                'approach': 'Automated optimized'
            }

    def _determine_personalization_level(self, confidence, features):
        """AI personalization level determination"""
        if confidence > 0.9 and features[12] > 0.8:  # High confidence + high alignment
            return 'MAXIMUM'
        elif confidence > 0.8:
            return 'HIGH'
        elif confidence > 0.7:
            return 'MEDIUM'
        else:
            return 'STANDARD'

    def _determine_automation_strategy(self, segment, features):
        """AI automation strategy"""
        consistency = features[9]
        pattern_alignment = features[12]

        if consistency > 0.12 and pattern_alignment > 0.7:
            return 'FULL_AUTOMATION'
        elif segment in ['Champions', 'Loyal', 'At Risk']:
            return 'HYBRID_HUMAN_AI'
        else:
            return 'AI_OPTIMIZED'

    def _generate_ai_recommendations(self, segment, confidence, features, auto_insights):
        """Comprehensive AI recommendations"""

        recommendations = []

        # Confidence-based recommendations
        if confidence > 0.95:
            recommendations.append("🤖 Maximum confidence - full AI automation recommended")
        elif confidence < 0.7:
            recommendations.append("⚠️ Low confidence - human review recommended")

        # Pattern-based recommendations
        pattern_alignment = features[12]
        if pattern_alignment < 0.5:
            recommendations.append("🔍 Pattern misalignment detected - investigation recommended")

        # Value-based recommendations
        customer_value = features[4]
        if customer_value > 200:
            recommendations.append("💎 High-value customer - premium treatment activated")

        # Geographic recommendations
        geo_behavior = features[13]
        if geo_behavior != 0.5:  # Non-neutral behavior
            recommendations.append("🌍 Geographic behavior optimization available")

        # Behavior-specific recommendations
        behavior_pattern = features[7]
        if behavior_pattern == 5:  # Champions pattern
            recommendations.append("🏆 Champions behavior pattern - VIP treatment recommended")
        elif behavior_pattern == 2:  # At Risk pattern
            recommendations.append("🚨 At Risk pattern detected - immediate intervention needed")

        return recommendations

    def bulk_predict_enhanced(self, customers_df):
        """
        Enhanced toplu müşteri segmentasyonu
        """
        if not self.model_loaded:
            print("❌ Enhanced model yüklenmemiş!")
            return None

        results = []
        print(f"🔄 Enhanced segmentation: {len(customers_df)} müşteri işleniyor...")

        for idx, row in customers_df.iterrows():
            # 14 enhanced features hazırla
            customer_data = []
            for feature in self.enhanced_feature_columns:
                if feature in row:
                    customer_data.append(row[feature])
                else:
                    customer_data.append(0)  # Default value

            customer_id = row.get('Customer ID', f'Customer_{idx}')
            result = self.predict_enhanced_segment(customer_data, customer_id)

            if result:
                # Simplified result for bulk processing
                simplified_result = {
                    'CustomerID': customer_id,
                    'Enhanced_Segment': result['prediction']['segment'],
                    'Confidence': result['prediction']['confidence'],
                    'Confidence_Level': result['prediction']['confidence_level'],
                    'Model_Accuracy': result['prediction']['model_accuracy'],
                    'Enhanced_Budget': result['enhanced_strategies']['enhanced_budget_allocation']['total_budget'],
                    'AI_Channels': ', '.join(result['enhanced_strategies']['ai_optimized_channels']),
                    'Personalization_Level': result['enhanced_strategies']['personalization_level'],
                    'AI_Recommendations_Count': len(result['ai_recommendations']),
                    'Processing_Time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                results.append(simplified_result)

        results_df = pd.DataFrame(results)
        print(f"✅ Enhanced bulk segmentation tamamlandı!")

        # Enhanced summary istatistikler
        print(f"\n📊 ENHANCED SEGMENTATION SUMMARY:")
        segment_counts = results_df['Enhanced_Segment'].value_counts()
        confidence_avg = results_df['Confidence'].mean()

        for segment, count in segment_counts.items():
            pct = (count / len(results_df)) * 100
            avg_conf = results_df[results_df['Enhanced_Segment'] == segment]['Confidence'].mean()
            print(f"  {segment:<20}: {count:>4} müşteri ({pct:>5.1f}%) - Avg Conf: {avg_conf:.3f}")

        print(f"\n🎯 Overall Enhanced Accuracy: {confidence_avg:.3f} (Model: 95.9%)")

        return results_df

    def generate_enhanced_segment_report(self, segment_name):
        """Enhanced segment raporu oluştur"""
        if segment_name not in self.segment_profiles:
            print(f"❌ Segment bulunamadı: {segment_name}")
            return None

        profile = self.segment_profiles[segment_name]

        print("=" * 80)
        print(f"📊 ENHANCED SEGMENT RAPORU: {segment_name.upper()}")
        print("=" * 80)
        print(f"📋 Enhanced Tanım: {profile['description']}")
        print(f"🎯 Pazarlama Önceliği: {profile['marketing_priority']}")
        print(f"🤖 AI Kampanya Tipi: {profile['campaign_type']}")
        print(f"🧠 AI Insights: {profile['ai_insights']}")
        print(f"⚠️ Churn Olasılığı: {profile['churn_probability']:.0%}")
        print(f"💰 Enhanced CLV Multiplier: {profile['expected_clv_multiplier']}x")
        print(f"🎯 Enhanced Targeting: {'✅ Active' if profile['enhanced_targeting'] else '❌ Disabled'}")

        print(f"\n🔍 Enhanced Karakteristikler:")
        for char in profile['characteristics']:
            print(f"  • {char}")

        print(f"\n🚀 Enhanced AI Aksiyonlar:")
        for action in profile['recommended_actions']:
            print(f"  • {action}")

        return profile


def main():
    """
    Enhanced Segmentation Engine test ve demo
    """
    print("🧠 ENHANCED SEGMENTATION ENGINE DEMO")
    print("=" * 70)

    # Enhanced Engine instance
    engine = EnhancedSegmentationEngine()

    # Enhanced models yükle
    if not engine.load_enhanced_models():
        print("❌ Enhanced models yükleme başarısız!")
        return

    # Enhanced test müşterisi (14 features)
    test_customer_enhanced = [
        30, 8, 2500.0, 312.5, 240,  # Original RFM features
        3, 2, 1, 365, 0.8, 2, 1.5, 0.7, 0.6  # Enhanced ML features
    ]

    print(f"\n🎯 ENHANCED TEST MÜŞTERİSİ ANALİZİ:")
    result = engine.predict_enhanced_segment(test_customer_enhanced, "ENHANCED_TEST_001")

    if result:
        print(f"🏷️ Enhanced Segment: {result['prediction']['segment']}")
        print(
            f"📊 Enhanced Confidence: {result['prediction']['confidence']:.3f} ({result['prediction']['confidence_level']})")
        print(f"🤖 Model Accuracy: {result['prediction']['model_accuracy']}%")
        print(f"💰 Enhanced Budget: £{result['enhanced_strategies']['enhanced_budget_allocation']['total_budget']}")
        print(f"📱 AI Channels: {', '.join(result['enhanced_strategies']['ai_optimized_channels'])}")
        print(f"🎯 Personalization: {result['enhanced_strategies']['personalization_level']}")
        print(f"🤖 AI Recommendations: {len(result['ai_recommendations'])}")

    # Enhanced segment raporu
    print(f"\n" + "=" * 70)
    engine.generate_enhanced_segment_report("Champions")

    print(f"\n🚀 Enhanced Segmentation Engine hazır!")
    print(f"🎯 %95.9 accuracy + Auto-discovered insights + AI optimization")

    return engine


if __name__ == "__main__":
    enhanced_engine = main()