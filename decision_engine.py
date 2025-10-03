# src/campaigns/decision_engine.py

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from segmentation.segment_engine import SegmentationEngine
import random
from typing import Dict, List, Tuple


class CampaignDecisionEngine:
    """
    Pazarlama segmentasyonu sonuçlarına göre
    optimized kampanya kararları veren motor
    """

    def __init__(self):
        self.segmentation_engine = SegmentationEngine()
        self.campaign_templates = self._load_campaign_templates()
        self.channel_costs = self._load_channel_costs()
        self.channel_performance = self._load_channel_performance()
        self.business_rules = self._load_business_rules()

    def _load_campaign_templates(self):
        """Segment bazlı kampanya şablonları"""
        return {
            'Champions': {
                'VIP_EXCLUSIVE': {
                    'name': 'VIP Exclusive Access',
                    'description': 'Early access to premium products + exclusive discounts',
                    'discount_rate': 0.15,
                    'channels': ['Email', 'SMS', 'Phone'],
                    'duration_days': 7,
                    'expected_conversion': 0.35,
                    'cost_per_contact': 12.50
                },
                'LOYALTY_REWARD': {
                    'name': 'Premium Loyalty Rewards',
                    'description': 'Extra loyalty points + premium benefits',
                    'discount_rate': 0.12,
                    'channels': ['Email', 'In-App', 'SMS'],
                    'duration_days': 14,
                    'expected_conversion': 0.28,
                    'cost_per_contact': 8.75
                }
            },

            'Loyal': {
                'CROSS_SELL': {
                    'name': 'Smart Cross-Sell Campaign',
                    'description': 'Personalized product recommendations',
                    'discount_rate': 0.10,
                    'channels': ['Email', 'In-App'],
                    'duration_days': 10,
                    'expected_conversion': 0.22,
                    'cost_per_contact': 6.25
                },
                'LOYALTY_BOOST': {
                    'name': 'Loyalty Points Boost',
                    'description': 'Double points for next purchase',
                    'discount_rate': 0.08,
                    'channels': ['Email', 'SMS'],
                    'duration_days': 14,
                    'expected_conversion': 0.18,
                    'cost_per_contact': 4.50
                }
            },

            'At Risk': {
                'WIN_BACK_URGENT': {
                    'name': 'Emergency Win-Back Campaign',
                    'description': 'Personalized offer + customer service call',
                    'discount_rate': 0.25,
                    'channels': ['Phone', 'Email', 'SMS'],
                    'duration_days': 5,
                    'expected_conversion': 0.15,
                    'cost_per_contact': 45.00
                },
                'SURVEY_FEEDBACK': {
                    'name': 'Customer Feedback & Recovery',
                    'description': 'Survey + incentive offer based on feedback',
                    'discount_rate': 0.20,
                    'channels': ['Email', 'Phone'],
                    'duration_days': 7,
                    'expected_conversion': 0.12,
                    'cost_per_contact': 25.00
                }
            },

            'Potential Loyalists': {
                'NURTURING': {
                    'name': 'Customer Development Program',
                    'description': 'Educational content + progressive offers',
                    'discount_rate': 0.12,
                    'channels': ['Email', 'Social Media'],
                    'duration_days': 21,
                    'expected_conversion': 0.16,
                    'cost_per_contact': 7.80
                },
                'LOYALTY_ENROLLMENT': {
                    'name': 'Loyalty Program Invitation',
                    'description': 'Join loyalty program + welcome bonus',
                    'discount_rate': 0.10,
                    'channels': ['Email', 'In-App'],
                    'duration_days': 14,
                    'expected_conversion': 0.20,
                    'cost_per_contact': 5.50
                }
            },

            'New Customers': {
                'ONBOARDING_SERIES': {
                    'name': 'Welcome Onboarding Series',
                    'description': '7-day email series + tutorial + first purchase incentive',
                    'discount_rate': 0.15,
                    'channels': ['Email', 'Tutorial', 'SMS'],
                    'duration_days': 7,
                    'expected_conversion': 0.25,
                    'cost_per_contact': 12.00
                },
                'SECOND_PURCHASE': {
                    'name': 'Second Purchase Incentive',
                    'description': 'Targeted offer for second purchase',
                    'discount_rate': 0.18,
                    'channels': ['Email', 'SMS'],
                    'duration_days': 14,
                    'expected_conversion': 0.22,
                    'cost_per_contact': 8.25
                }
            },

            'Promising': {
                'ENGAGEMENT_BOOST': {
                    'name': 'Engagement Boost Campaign',
                    'description': 'Product demos + limited-time offers',
                    'discount_rate': 0.15,
                    'channels': ['Email', 'Social Media'],
                    'duration_days': 10,
                    'expected_conversion': 0.14,
                    'cost_per_contact': 9.50
                }
            },

            'Need Attention': {
                'RE_ENGAGEMENT': {
                    'name': 'Re-engagement Campaign',
                    'description': 'Special offers + product updates',
                    'discount_rate': 0.18,
                    'channels': ['Email', 'SMS'],
                    'duration_days': 7,
                    'expected_conversion': 0.12,
                    'cost_per_contact': 8.75
                }
            },

            'About to Sleep': {
                'WAKE_UP_CALL': {
                    'name': 'Customer Wake-Up Campaign',
                    'description': 'Limited time offers + product reminders',
                    'discount_rate': 0.20,
                    'channels': ['Email', 'SMS'],
                    'duration_days': 5,
                    'expected_conversion': 0.10,
                    'cost_per_contact': 12.50
                }
            },

            'Hibernating': {
                'MINIMAL_TOUCH': {
                    'name': 'Low-Cost Reactivation',
                    'description': 'Basic email campaigns + surveys',
                    'discount_rate': 0.25,
                    'channels': ['Email'],
                    'duration_days': 30,
                    'expected_conversion': 0.05,
                    'cost_per_contact': 2.50
                }
            },

            'Lost': {
                'FINAL_ATTEMPT': {
                    'name': 'Final Recovery Attempt',
                    'description': 'Last chance offer + feedback collection',
                    'discount_rate': 0.30,
                    'channels': ['Email'],
                    'duration_days': 7,
                    'expected_conversion': 0.03,
                    'cost_per_contact': 1.50
                }
            }
        }

    def _load_channel_costs(self):
        """Kanal maliyetleri (per contact)"""
        return {
            'Email': 0.10,
            'SMS': 0.15,
            'Phone': 15.00,
            'In-App': 0.05,
            'Social Media': 2.50,
            'Tutorial': 5.00
        }

    def _load_channel_performance(self):
        """Kanal performans geçmişi"""
        return {
            'Email': {'open_rate': 0.22, 'click_rate': 0.045, 'conversion_rate': 0.12},
            'SMS': {'open_rate': 0.85, 'click_rate': 0.12, 'conversion_rate': 0.08},
            'Phone': {'connect_rate': 0.35, 'conversion_rate': 0.25},
            'In-App': {'view_rate': 0.65, 'click_rate': 0.18, 'conversion_rate': 0.15},
            'Social Media': {'reach_rate': 0.15, 'engagement_rate': 0.08, 'conversion_rate': 0.06},
            'Tutorial': {'completion_rate': 0.45, 'conversion_rate': 0.30}
        }

    def _load_business_rules(self):
        """İş kuralları"""
        return {
            'max_campaigns_per_customer_per_month': 3,
            'min_days_between_campaigns': 7,
            'max_discount_rate': 0.40,
            'priority_segments': ['Champions', 'At Risk', 'Loyal'],
            'budget_allocation_weights': {
                'Champions': 0.25,
                'Loyal': 0.20,
                'At Risk': 0.20,
                'Potential Loyalists': 0.15,
                'New Customers': 0.10,
                'Promising': 0.05,
                'Need Attention': 0.03,
                'About to Sleep': 0.015,
                'Hibernating': 0.01,
                'Lost': 0.005
            }
        }

    def decide_campaign(self, customer_data, budget_constraint=None, preferred_channels=None):
        """
        Ana kampanya karar algoritması

        Args:
            customer_data: Müşteri RFM verileri
            budget_constraint: Budget sınırı (opsiyonel)
            preferred_channels: Tercih edilen kanallar (opsiyonel)
        """

        # 1. Segmentasyon yap
        if not self.segmentation_engine.model_loaded:
            self.segmentation_engine.load_models()

        segment_result = self.segmentation_engine.predict_segment(customer_data)

        if not segment_result:
            return None

        segment = segment_result['prediction']['segment']
        confidence = segment_result['prediction']['confidence']

        # 2. Segment'e uygun kampanya şablonlarını al
        available_campaigns = self.campaign_templates.get(segment, {})

        if not available_campaigns:
            return self._create_default_campaign(segment)

        # 3. En uygun kampanyayı seç
        best_campaign = self._select_best_campaign(
            available_campaigns,
            confidence,
            budget_constraint,
            preferred_channels
        )

        # 4. Kampanyayı kişiselleştir
        personalized_campaign = self._personalize_campaign(
            best_campaign,
            customer_data,
            segment_result
        )

        # 5. ROI hesapla
        roi_prediction = self._calculate_roi_prediction(
            personalized_campaign,
            customer_data,
            segment_result
        )

        # 6. Final kampanya paketi
        campaign_decision = {
            'customer_segment': segment,
            'confidence': confidence,
            'campaign': personalized_campaign,
            'roi_prediction': roi_prediction,
            'recommended_timing': self._optimize_timing(segment),
            'a_b_test_variant': self._assign_ab_test_group(),
            'success_metrics': self._define_success_metrics(segment),
            'decision_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        return campaign_decision

    def _select_best_campaign(self, campaigns, confidence, budget_constraint, preferred_channels):
        """En uygun kampanyayı seç"""

        scored_campaigns = []

        for campaign_key, campaign in campaigns.items():
            score = 0

            # Confidence uyumu
            if confidence >= 0.8:
                score += campaign.get('expected_conversion', 0) * 100
            elif confidence >= 0.6:
                score += campaign.get('expected_conversion', 0) * 80
            else:
                score += campaign.get('expected_conversion', 0) * 60

            # Budget uyumu
            if budget_constraint:
                if campaign.get('cost_per_contact', 0) <= budget_constraint:
                    score += 20
                else:
                    score -= 30

            # Kanal tercihi uyumu
            if preferred_channels:
                campaign_channels = set(campaign.get('channels', []))
                preferred_set = set(preferred_channels)
                overlap = len(campaign_channels.intersection(preferred_set))
                score += overlap * 10

            scored_campaigns.append((campaign_key, campaign, score))

        # En yüksek skorlu kampanyayı seç
        if scored_campaigns:
            best = max(scored_campaigns, key=lambda x: x[2])
            return best[1]
        else:
            return list(campaigns.values())[0]

    def _personalize_campaign(self, campaign, customer_data, segment_result):
        """Kampanyayı müşteriye özel hale getir"""

        personalized = campaign.copy()

        # Customer value'ye göre discount ayarla
        customer_value = customer_data[4] if len(customer_data) > 4 else 100

        if customer_value > 200:
            personalized['discount_rate'] = min(
                campaign['discount_rate'] * 1.2,
                self.business_rules['max_discount_rate']
            )
        elif customer_value < 50:
            personalized['discount_rate'] = campaign['discount_rate'] * 0.8

        # Recency'ye göre urgency ayarla
        recency = customer_data[0]
        if recency > 90:  # Long time no purchase
            personalized['urgency'] = 'HIGH'
            personalized['duration_days'] = max(3, campaign['duration_days'] // 2)
        elif recency < 30:
            personalized['urgency'] = 'LOW'
            personalized['duration_days'] = campaign['duration_days'] * 1.5
        else:
            personalized['urgency'] = 'MEDIUM'

        # Channel optimization
        optimized_channels = self._optimize_channels(
            campaign['channels'],
            segment_result['prediction']['segment']
        )
        personalized['channels'] = optimized_channels

        return personalized

    def _optimize_channels(self, campaign_channels, segment):
        """Segment'e göre kanal karmasını optimize et"""

        # Segment bazlı kanal performansları
        segment_channel_weights = {
            'Champions': {'Email': 0.9, 'SMS': 0.8, 'Phone': 0.95, 'In-App': 0.85},
            'Loyal': {'Email': 0.85, 'SMS': 0.7, 'In-App': 0.9, 'Social Media': 0.6},
            'At Risk': {'Phone': 0.95, 'Email': 0.8, 'SMS': 0.85, 'Social Media': 0.4},
            'New Customers': {'Email': 0.9, 'SMS': 0.8, 'Tutorial': 0.95, 'In-App': 0.7},
            'Promising': {'Email': 0.8, 'Social Media': 0.85, 'SMS': 0.6}
        }

        weights = segment_channel_weights.get(segment, {})

        # Kanalları ağırlıklı skora göre sırala
        scored_channels = []
        for channel in campaign_channels:
            base_performance = self.channel_performance.get(channel, {}).get('conversion_rate', 0.1)
            segment_weight = weights.get(channel, 0.5)
            final_score = base_performance * segment_weight
            scored_channels.append((channel, final_score))

        # En iyi 3 kanalı seç
        scored_channels.sort(key=lambda x: x[1], reverse=True)
        optimized = [ch[0] for ch in scored_channels[:3]]

        return optimized

    def _calculate_roi_prediction(self, campaign, customer_data, segment_result):
        """ROI tahmini hesapla"""

        # Campaign cost
        cost_per_contact = campaign.get('cost_per_contact', 10)
        channel_costs = sum([
            self.channel_costs.get(ch, 1)
            for ch in campaign.get('channels', [])
        ])
        total_cost = cost_per_contact + channel_costs

        # Expected revenue
        customer_clv = customer_data[2] * segment_result['segment_profile'].get('expected_clv_multiplier', 1.0)
        expected_conversion = campaign.get('expected_conversion', 0.1)
        expected_revenue = customer_clv * expected_conversion

        # ROI calculation
        roi = (expected_revenue - total_cost) / total_cost if total_cost > 0 else 0

        return {
            'total_cost': round(total_cost, 2),
            'expected_revenue': round(expected_revenue, 2),
            'expected_conversion_rate': expected_conversion,
            'predicted_roi': round(roi * 100, 1),  # Percentage
            'break_even_conversion': round(total_cost / customer_clv, 4) if customer_clv > 0 else 0
        }

    def _optimize_timing(self, segment):
        """Segment'e göre optimal timing"""

        timing_rules = {
            'Champions': {'best_day': 'Tuesday', 'best_hour': 10, 'frequency': 'Weekly'},
            'Loyal': {'best_day': 'Wednesday', 'best_hour': 14, 'frequency': 'Bi-weekly'},
            'At Risk': {'best_day': 'Monday', 'best_hour': 9, 'frequency': 'Immediate'},
            'New Customers': {'best_day': 'Thursday', 'best_hour': 11, 'frequency': 'Daily for 1 week'},
            'Potential Loyalists': {'best_day': 'Friday', 'best_hour': 15, 'frequency': 'Weekly'}
        }

        default_timing = {'best_day': 'Tuesday', 'best_hour': 12, 'frequency': 'Weekly'}
        return timing_rules.get(segment, default_timing)

    def _assign_ab_test_group(self):
        """A/B test grubu ataması"""
        groups = ['Control', 'Variant_A', 'Variant_B']
        return random.choice(groups)

    def _define_success_metrics(self, segment):
        """Segment bazlı başarı metrikleri"""

        metrics = {
            'Champions': ['retention_rate', 'clv_increase', 'referral_count'],
            'Loyal': ['upsell_success', 'engagement_increase', 'frequency_improvement'],
            'At Risk': ['win_back_rate', 'churn_prevention', 'satisfaction_score'],
            'New Customers': ['second_purchase_rate', 'onboarding_completion', 'engagement_rate'],
            'Potential Loyalists': ['loyalty_program_enrollment', 'purchase_frequency_increase']
        }

        return metrics.get(segment, ['conversion_rate', 'revenue_increase', 'engagement_rate'])

    def _create_default_campaign(self, segment):
        """Default kampanya oluştur"""
        return {
            'name': f'Standard {segment} Campaign',
            'description': f'Default campaign for {segment} segment',
            'discount_rate': 0.10,
            'channels': ['Email'],
            'duration_days': 7,
            'expected_conversion': 0.08,
            'cost_per_contact': 5.00
        }


def main():
    """
    Test ve demo fonksiyonu
    """
    print("⚙️ KAMPANYA KARAR MOTORU DEMO")
    print("=" * 50)

    # Engine instance
    engine = CampaignDecisionEngine()

    # Test müşteri verisi
    test_customers = [
        {
            'name': 'High-Value Champion',
            'data': [15, 12, 5000, 416.67, 300]  # Recency, Frequency, Monetary, AvgOrder, CustomerValue
        },
        {
            'name': 'At-Risk Customer',
            'data': [120, 8, 3200, 400, 240]
        },
        {
            'name': 'New Customer',
            'data': [5, 2, 150, 75, 40]
        }
    ]

    print(f"\n🎯 KAMPANYA KARAR TESTLERİ:")
    print("-" * 60)

    for customer in test_customers:
        print(f"\n👤 Müşteri: {customer['name']}")

        # Kampanya kararı al
        decision = engine.decide_campaign(customer['data'])

        if decision:
            print(f"🏷️ Segment: {decision['customer_segment']}")
            print(f"📊 Confidence: {decision['confidence']:.3f}")
            print(f"🎯 Kampanya: {decision['campaign']['name']}")
            print(f"💰 Maliyet: ${decision['roi_prediction']['total_cost']}")
            print(f"📈 Beklenen ROI: {decision['roi_prediction']['predicted_roi']}%")
            print(f"📱 Kanallar: {', '.join(decision['campaign']['channels'])}")
            print(
                f"⏰ Timing: {decision['recommended_timing']['best_day']} at {decision['recommended_timing']['best_hour']}:00")
            print(f"🧪 A/B Group: {decision['a_b_test_variant']}")

        print("-" * 40)

    print(f"\n🚀 Kampanya Karar Motoru hazır!")


if __name__ == "__main__":
    main()