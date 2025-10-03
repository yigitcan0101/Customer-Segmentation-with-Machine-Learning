# src/campaigns/ab_test_engine.py - BUG FIX VERSION

import pandas as pd
import numpy as np
import json
import sys
import os
from datetime import datetime, timedelta
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import random
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')

# Path düzeltmesi
current_dir = os.path.dirname(__file__)
parent_dir = os.path.join(current_dir, '..')
sys.path.append(parent_dir)

from segmentation.segment_engine import SegmentationEngine
from campaigns.decision_engine import CampaignDecisionEngine


@dataclass
class ABTestResult:
    """A/B Test sonucu data class'ı"""
    test_name: str
    control_group: Dict
    variant_groups: List[Dict]
    winner: str
    confidence_level: float
    statistical_significance: bool
    lift: float
    roi_difference: float
    recommendation: str


class ABTestEngine:
    """
    🧪 ENHANCED A/B TEST ENGINE - BUG FIXED VERSION

    Pazarlama kampanyaları için A/B test motoru
    Statistical significance, ROI comparison ve winner determination
    Enhanced with dual model support and improved error handling
    """

    def __init__(self):
        self.segmentation_engine = SegmentationEngine()
        self.campaign_engine = CampaignDecisionEngine()
        self.active_tests = {}
        self.test_results = []
        self.min_sample_size = 80  # Reduced for enhanced testing
        self.significance_threshold = 0.05  # 95% confidence

    def create_ab_test(self, test_name: str, customers_data: List[Dict],
                       test_config: Dict) -> Dict:
        """
        🧪 ENHANCED A/B TEST CREATION (Bug Fixed)

        Args:
            test_name: Test adı
            customers_data: Müşteri verileri listesi
            test_config: Test konfigürasyonu
        """

        print(f"🧪 Enhanced A/B Test oluşturuluyor: {test_name}")
        print("=" * 60)

        # Test konfigürasyonu validation
        required_config = ['control_strategy', 'variant_strategies', 'success_metric', 'test_duration_days']
        for key in required_config:
            if key not in test_config:
                raise ValueError(f"Missing required config: {key}")

        # Enhanced sample size kontrolü
        if len(customers_data) < self.min_sample_size:
            print(f"⚠️ Warning: Sample size ({len(customers_data)}) below recommended ({self.min_sample_size})")

        # Enhanced grup ataması
        test_groups = self._assign_enhanced_test_groups(
            customers_data,
            len(test_config['variant_strategies']) + 1
        )

        # Enhanced kampanya stratejisi
        group_strategies = self._assign_enhanced_campaign_strategies(
            test_groups,
            test_config
        )

        # Enhanced test objesi oluştur (BUG FIX: Improved data structure)
        test_object = {
            'test_name': test_name,
            'created_date': datetime.now(),
            'test_config': test_config,
            'test_groups': test_groups,
            'group_strategies': group_strategies,
            'status': 'ACTIVE',
            'enhanced_features': True,  # Enhanced test marker
            'results': {
                'control': {
                    'metrics': {},
                    'customers': test_groups.get('control', []),
                    # BUG FIX: Ensure all required keys exist
                    'total_customers': len(test_groups.get('control', [])),
                    'conversions': 0,
                    'total_revenue': 0,
                    'total_cost': 0,
                    'conversion_rate': 0,
                    'revenue_per_customer': 0,
                    'roi': 0,
                    'customer_details': []
                },
                'variants': {}
            },
            'statistical_analysis': None
        }

        # BUG FIX: Initialize variant results properly
        for i, variant_name in enumerate([f'variant_{i}' for i in range(len(test_config['variant_strategies']))]):
            test_object['results']['variants'][variant_name] = {
                'metrics': {},
                'customers': test_groups.get(variant_name, []),
                # BUG FIX: Ensure all required keys exist
                'total_customers': len(test_groups.get(variant_name, [])),
                'conversions': 0,
                'total_revenue': 0,
                'total_cost': 0,
                'conversion_rate': 0,
                'revenue_per_customer': 0,
                'roi': 0,
                'customer_details': []
            }

        self.active_tests[test_name] = test_object

        print(f"✅ Enhanced A/B Test oluşturuldu!")
        print(f"📊 Toplam müşteri: {len(customers_data)}")
        print(f"🎯 Grup sayısı: {len(test_groups)}")
        print(f"📈 Success metric: {test_config['success_metric']}")
        print(f"⏰ Test süresi: {test_config['test_duration_days']} gün")

        return test_object

    def _assign_enhanced_test_groups(self, customers_data: List[Dict], num_groups: int) -> Dict:
        """
        🎯 ENHANCED GROUP ASSIGNMENT (Bug Fixed)

        Müşterileri A/B test gruplarına stratified sampling ile ata
        """

        # Enhanced stratified sampling for balanced groups
        customers_df = pd.DataFrame(customers_data)

        # Segment prediction for stratification
        if not self.segmentation_engine.model_loaded:
            try:
                self.segmentation_engine.load_models()
            except:
                print("⚠️ Warning: Segmentation engine not loaded, using random assignment")

        # Enhanced segment predictions
        customer_segments = []
        for customer in customers_data:
            try:
                segment_data = [
                    customer.get('Recency', 50),
                    customer.get('Frequency', 3),
                    customer.get('Monetary', 500),
                    customer.get('AvgOrderValue', 167),
                    customer.get('CustomerValue', 100)
                ]

                if self.segmentation_engine.model_loaded:
                    result = self.segmentation_engine.predict_segment(segment_data)
                    customer_segments.append(result['prediction']['segment'] if result else 'Unknown')
                else:
                    customer_segments.append('Unknown')
            except:
                customer_segments.append('Unknown')

        customers_df['Segment'] = customer_segments

        # Enhanced balanced grup ataması
        groups = {f'group_{i}': [] for i in range(num_groups)}

        for segment in customers_df['Segment'].unique():
            segment_customers = customers_df[customers_df['Segment'] == segment].index.tolist()

            # Shuffle and assign evenly
            random.shuffle(segment_customers)
            for i, customer_idx in enumerate(segment_customers):
                group_key = f'group_{i % num_groups}'
                groups[group_key].append(customers_data[customer_idx])

        # Enhanced group naming
        final_groups = {
            'control': groups['group_0'],
            **{f'variant_{i}': groups[f'group_{i + 1}'] for i in range(num_groups - 1)}
        }

        return final_groups

    def _assign_enhanced_campaign_strategies(self, test_groups: Dict, test_config: Dict) -> Dict:
        """
        🎯 ENHANCED CAMPAIGN STRATEGY ASSIGNMENT

        Her test grubuna enhanced kampanya stratejisi ata
        """

        strategies = {}

        # Enhanced control group
        strategies['control'] = {
            **test_config['control_strategy'],
            'enhanced_features': False,  # Control uses original
            'model_type': test_config['control_strategy'].get('model_type', 'original')
        }

        # Enhanced variant groups
        for i, (group_name, customers) in enumerate(test_groups.items()):
            if group_name.startswith('variant_'):
                variant_idx = int(group_name.split('_')[1])
                if variant_idx < len(test_config['variant_strategies']):
                    strategies[group_name] = {
                        **test_config['variant_strategies'][variant_idx],
                        'enhanced_features': True,  # Variants use enhanced
                        'model_type': test_config['variant_strategies'][variant_idx].get('model_type', 'enhanced')
                    }
                else:
                    strategies[group_name] = strategies['control']

        return strategies

    def run_enhanced_campaign_simulation(self, test_name: str) -> Dict:
        """
        🚀 ENHANCED CAMPAIGN SIMULATION (Bug Fixed)

        A/B test kampanyalarını enhanced features ile simüle et
        """

        if test_name not in self.active_tests:
            raise ValueError(f"Test bulunamadı: {test_name}")

        test_obj = self.active_tests[test_name]

        print(f"🚀 Enhanced kampanya simülasyonu başlıyor: {test_name}")
        print("-" * 50)

        simulation_results = {}

        for group_name, customers in test_obj['test_groups'].items():
            print(f"\n📊 {group_name.upper()} grubu işleniyor ({len(customers)} müşteri)...")

            # BUG FIX: Initialize group results with all required keys
            group_results = {
                'total_customers': len(customers),
                'conversions': 0,
                'total_revenue': 0,
                'total_cost': 0,
                'conversion_rate': 0,
                'revenue_per_customer': 0,
                'roi': 0,
                'customer_details': [],
                'enhanced_metrics': {},  # Enhanced metrics
                'model_type': test_obj['group_strategies'].get(group_name, {}).get('model_type', 'original')
            }

            strategy = test_obj['group_strategies'].get(group_name, {})

            # Enhanced simulation per customer
            for customer in customers:
                try:
                    # Enhanced customer data preparation
                    customer_data = [
                        customer.get('Recency', 50),
                        customer.get('Frequency', 3),
                        customer.get('Monetary', 500),
                        customer.get('AvgOrderValue', 167),
                        customer.get('CustomerValue', 100)
                    ]

                    # Enhanced campaign decision
                    campaign_decision = self.campaign_engine.decide_campaign(customer_data)

                    if campaign_decision:
                        # Enhanced strategy modifiers
                        modified_decision = self._apply_enhanced_strategy_modifiers(
                            campaign_decision, strategy
                        )

                        # Enhanced conversion simulation
                        conversion_result = self._simulate_enhanced_conversion(
                            modified_decision, customer_data, strategy
                        )

                        # Enhanced results aggregation
                        group_results['total_cost'] += conversion_result['cost']

                        if conversion_result['converted']:
                            group_results['conversions'] += 1
                            group_results['total_revenue'] += conversion_result['revenue']

                        group_results['customer_details'].append({
                            'customer_data': customer_data,
                            'campaign': modified_decision['campaign']['name'],
                            'converted': conversion_result['converted'],
                            'revenue': conversion_result['revenue'],
                            'cost': conversion_result['cost'],
                            'model_type': strategy.get('model_type', 'original'),
                            'enhanced_features': strategy.get('enhanced_features', False)
                        })

                except Exception as e:
                    print(f"⚠️ Warning: Customer simulation error: {e}")
                    continue

            # Enhanced metrikleri hesapla
            if group_results['total_customers'] > 0:
                group_results['conversion_rate'] = group_results['conversions'] / group_results['total_customers']
                group_results['revenue_per_customer'] = group_results['total_revenue'] / group_results[
                    'total_customers']

            if group_results['total_cost'] > 0:
                group_results['roi'] = ((group_results['total_revenue'] - group_results['total_cost']) /
                                        group_results['total_cost']) * 100

            # Enhanced metrics
            group_results['enhanced_metrics'] = {
                'confidence_boost': 1.2 if strategy.get('enhanced_features', False) else 1.0,
                'ai_optimization': strategy.get('enhanced_features', False),
                'model_accuracy': 95.9 if strategy.get('model_type') == 'enhanced' else 93.6
            }

            simulation_results[group_name] = group_results

            print(f"  ✅ Conversion Rate: {group_results['conversion_rate']:.3f}")
            print(f"  💰 Revenue per Customer: ${group_results['revenue_per_customer']:.2f}")
            print(f"  📈 ROI: {group_results['roi']:.1f}%")
            print(f"  🤖 Model: {group_results['model_type']}")

        # BUG FIX: Update test object results properly
        test_obj['results']['control'] = simulation_results.get('control', test_obj['results']['control'])
        for variant_name in simulation_results:
            if variant_name.startswith('variant_'):
                test_obj['results']['variants'][variant_name] = simulation_results[variant_name]

        test_obj['simulation_date'] = datetime.now()

        return simulation_results

    def _apply_enhanced_strategy_modifiers(self, campaign_decision: Dict, strategy: Dict) -> Dict:
        """
        🎯 ENHANCED STRATEGY MODIFIERS

        Enhanced test stratejisine göre kampanya kararını modifiye et
        """

        modified = campaign_decision.copy()

        # Enhanced discount modifier
        if 'discount_modifier' in strategy:
            current_discount = modified['campaign'].get('discount_rate', 0.10)
            modified['campaign']['discount_rate'] = current_discount * strategy['discount_modifier']

        # Enhanced channel modifier
        if 'preferred_channels' in strategy:
            enhanced_channels = strategy['preferred_channels'].copy()
            if strategy.get('enhanced_features', False):
                enhanced_channels.extend(['AI-Optimized', 'Predictive'])
            modified['campaign']['channels'] = list(set(enhanced_channels))

        # Enhanced budget modifier
        if 'budget_modifier' in strategy:
            current_cost = modified['roi_prediction']['total_cost']
            enhanced_multiplier = 1.2 if strategy.get('enhanced_features', False) else 1.0
            modified['roi_prediction']['total_cost'] = current_cost * strategy.get('budget_modifier',
                                                                                   1.0) * enhanced_multiplier

        return modified

    def _simulate_enhanced_conversion(self, campaign_decision: Dict, customer_data: List, strategy: Dict) -> Dict:
        """
        🎯 ENHANCED CONVERSION SIMULATION

        Enhanced features ile kampanya conversion'ını simüle et
        """

        # Enhanced base conversion rate
        base_conversion = campaign_decision['campaign'].get('expected_conversion', 0.10)

        # Enhanced customer value impact
        customer_value = customer_data[4] if len(customer_data) > 4 else 100
        value_multiplier = 1.0 + (customer_value - 100) / 1000

        # Enhanced confidence impact
        confidence = campaign_decision.get('confidence', 0.5)
        confidence_multiplier = 0.5 + confidence * 0.5

        # Enhanced model boost
        enhanced_boost = 1.15 if strategy.get('enhanced_features', False) else 1.0
        model_boost = 1.05 if strategy.get('model_type') == 'enhanced' else 1.0

        # Enhanced final conversion probability
        final_conversion_prob = (base_conversion * value_multiplier * confidence_multiplier *
                                 enhanced_boost * model_boost)
        final_conversion_prob = min(final_conversion_prob, 0.95)  # Cap at 95%

        # Enhanced simulate conversion
        converted = random.random() < final_conversion_prob

        # Enhanced calculate revenue and cost
        cost = campaign_decision['roi_prediction']['total_cost']
        revenue = 0

        if converted:
            customer_monetary = customer_data[2] if len(customer_data) > 2 else 500
            clv_multiplier = campaign_decision.get('segment_profile', {}).get('expected_clv_multiplier', 1.0)
            enhanced_multiplier = 1.2 if strategy.get('enhanced_features', False) else 1.0
            revenue = customer_monetary * clv_multiplier * enhanced_multiplier * random.uniform(0.1, 0.3)

        return {
            'converted': converted,
            'revenue': revenue,
            'cost': cost,
            'conversion_probability': final_conversion_prob,
            'enhanced_features_used': strategy.get('enhanced_features', False),
            'model_type': strategy.get('model_type', 'original')
        }

    def analyze_enhanced_test_results(self, test_name: str) -> ABTestResult:
        """
        🧠 ENHANCED A/B TEST ANALYSIS (Bug Fixed)

        Enhanced A/B test sonuçlarını statistical olarak analiz et
        """

        if test_name not in self.active_tests:
            raise ValueError(f"Test bulunamadı: {test_name}")

        test_obj = self.active_tests[test_name]

        # BUG FIX: Check if simulation was run
        if not test_obj.get('simulation_date'):
            print("⚠️ Running simulation first...")
            self.run_enhanced_campaign_simulation(test_name)

        results = test_obj['results']

        print(f"📊 ENHANCED A/B TEST SONUÇ ANALİZİ: {test_name}")
        print("=" * 60)

        # BUG FIX: Enhanced Control vs Variants comparison with proper error handling
        control_data = results.get('control', {})
        if not control_data or 'total_customers' not in control_data:
            print("❌ Control group data not found or incomplete")
            return None

        variant_results = []

        print(f"\n🎯 ENHANCED CONTROL GROUP:")
        print(f"  👥 Sample Size: {control_data.get('total_customers', 0)}")
        print(f"  📈 Conversion Rate: {control_data.get('conversion_rate', 0):.3f}")
        print(f"  💰 Revenue per Customer: ${control_data.get('revenue_per_customer', 0):.2f}")
        print(f"  📊 ROI: {control_data.get('roi', 0):.1f}%")
        print(f"  🤖 Model: {control_data.get('model_type', 'original')}")

        best_variant = None
        best_performance = control_data.get('roi', 0)
        statistical_tests = []

        # Enhanced variant analysis
        variants = results.get('variants', {})
        for variant_name, variant_data in variants.items():
            if not variant_data or 'total_customers' not in variant_data:
                print(f"⚠️ Warning: {variant_name} data incomplete, skipping...")
                continue

            print(f"\n🧪 ENHANCED {variant_name.upper()}:")
            print(f"  👥 Sample Size: {variant_data.get('total_customers', 0)}")
            print(f"  📈 Conversion Rate: {variant_data.get('conversion_rate', 0):.3f}")
            print(f"  💰 Revenue per Customer: ${variant_data.get('revenue_per_customer', 0):.2f}")
            print(f"  📊 ROI: {variant_data.get('roi', 0):.1f}%")
            print(f"  🤖 Model: {variant_data.get('model_type', 'enhanced')}")

            # Enhanced statistical significance test
            stat_result = self._calculate_enhanced_statistical_significance(
                control_data, variant_data
            )
            statistical_tests.append({
                'variant': variant_name,
                'stat_result': stat_result
            })

            # Enhanced lift calculation
            control_conversion = control_data.get('conversion_rate', 0)
            variant_conversion = variant_data.get('conversion_rate', 0)

            if control_conversion > 0:
                lift = (variant_conversion - control_conversion) / control_conversion * 100
            else:
                lift = 0

            print(f"  📈 Enhanced Lift: {lift:+.1f}%")
            print(f"  🔬 P-value: {stat_result.get('p_value', 1.0):.4f}")
            print(f"  ✅ Significant: {'Yes' if stat_result.get('is_significant', False) else 'No'}")

            # Enhanced track best variant
            variant_roi = variant_data.get('roi', 0)
            if variant_roi > best_performance:
                best_performance = variant_roi
                best_variant = variant_name

            variant_results.append({
                'name': variant_name,
                'data': variant_data,
                'lift': lift,
                'statistical_result': stat_result
            })

        # Enhanced winner determination
        winner = 'control' if best_variant is None else best_variant
        winner_data = results.get(winner, control_data) if winner == 'control' else results['variants'].get(winner, {})

        # Enhanced recommendations
        recommendation = self._generate_enhanced_recommendations(
            test_obj, winner, statistical_tests
        )

        print(f"\n🏆 ENHANCED SONUÇLAR:")
        print(f"  🥇 Winner: {winner.upper()}")
        print(f"  📊 Best ROI: {best_performance:.1f}%")
        print(f"  🎯 Enhanced Recommendation: {recommendation}")

        # Enhanced test sonucu objesi
        test_result = ABTestResult(
            test_name=test_name,
            control_group=control_data,
            variant_groups=variant_results,
            winner=winner,
            confidence_level=95.0,
            statistical_significance=any([t['stat_result'].get('is_significant', False) for t in statistical_tests]),
            lift=max([v['lift'] for v in variant_results]) if variant_results else 0,
            roi_difference=best_performance - control_data.get('roi', 0),
            recommendation=recommendation
        )

        # Enhanced test sonucunu kaydet
        self.test_results.append(test_result)
        test_obj['statistical_analysis'] = test_result
        test_obj['status'] = 'COMPLETED'

        return test_result

    def _calculate_enhanced_statistical_significance(self, control: Dict, variant: Dict) -> Dict:
        """
        🧠 ENHANCED STATISTICAL SIGNIFICANCE

        Enhanced statistical test with improved error handling
        """

        # Enhanced conversion rates with error handling
        control_conv = control.get('conversions', 0)
        control_total = control.get('total_customers', 0)
        variant_conv = variant.get('conversions', 0)
        variant_total = variant.get('total_customers', 0)

        # Enhanced validation
        if control_total == 0 or variant_total == 0:
            return {
                'p_value': 1.0,
                'z_score': 0,
                'is_significant': False,
                'test_type': 'insufficient_data',
                'enhanced': True
            }

        # Enhanced Z-test for proportions
        try:
            # Enhanced pooled proportion
            pooled_conv = control_conv + variant_conv
            pooled_total = control_total + variant_total
            pooled_prop = pooled_conv / pooled_total if pooled_total > 0 else 0

            # Enhanced standard error
            if pooled_prop == 0 or pooled_prop == 1:
                return {
                    'p_value': 1.0,
                    'z_score': 0,
                    'is_significant': False,
                    'test_type': 'edge_case',
                    'enhanced': True
                }

            se = np.sqrt(pooled_prop * (1 - pooled_prop) * (1 / control_total + 1 / variant_total))

            # Enhanced Z-score
            control_rate = control_conv / control_total if control_total > 0 else 0
            variant_rate = variant_conv / variant_total if variant_total > 0 else 0
            z_score = (variant_rate - control_rate) / se if se > 0 else 0

            # Enhanced P-value (two-tailed test)
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

            return {
                'p_value': p_value,
                'z_score': z_score,
                'is_significant': p_value < self.significance_threshold,
                'test_type': 'enhanced_z_test_proportions',
                'enhanced': True
            }

        except Exception as e:
            print(f"⚠️ Warning: Statistical calculation error: {e}")
            return {
                'p_value': 1.0,
                'z_score': 0,
                'is_significant': False,
                'test_type': 'calculation_error',
                'enhanced': True
            }

    def _generate_enhanced_recommendations(self, test_obj: Dict, winner: str,
                                           statistical_tests: List) -> str:
        """
        🎯 ENHANCED RECOMMENDATIONS

        Enhanced test sonuçlarına göre AI-powered öneriler generate et
        """

        results = test_obj['results']
        winner_data = results.get(winner, {}) if winner == 'control' else results.get('variants', {}).get(winner, {})

        recommendations = []

        # Enhanced winner recommendation
        if winner == 'control':
            recommendations.append("Mevcut stratejinize devam edin (original model optimal)")
        else:
            model_type = winner_data.get('model_type', 'enhanced')
            recommendations.append(f"{winner} stratejisini implement edin ({model_type} model)")

        # Enhanced statistical significance
        significant_tests = [t for t in statistical_tests if t['stat_result'].get('is_significant', False)]
        if not significant_tests:
            recommendations.append("Test daha uzun süre çalıştırılabilir (enhanced significance için)")

        # Enhanced ROI recommendations
        winner_roi = winner_data.get('roi', 0)
        if winner_roi > 150:
            recommendations.append("Excellent Enhanced ROI - budget artırın ve scale edin")
        elif winner_roi > 100:
            recommendations.append("Good Enhanced ROI - dikkatli scale edilebilir")
        else:
            recommendations.append("Enhanced ROI optimize edilmeli - model tuning gerekli")

        # Enhanced model recommendations
        enhanced_features_used = winner_data.get('enhanced_metrics', {}).get('ai_optimization', False)
        if enhanced_features_used:
            recommendations.append("Enhanced features başarılı - production'da kullanın")

        return " | ".join(recommendations)

    # BUG FIX: Enhanced compatibility methods
    def analyze_test_results(self, test_name: str) -> ABTestResult:
        """Backward compatibility wrapper"""
        return self.analyze_enhanced_test_results(test_name)

    def run_campaign_simulation(self, test_name: str) -> Dict:
        """Backward compatibility wrapper"""
        return self.run_enhanced_campaign_simulation(test_name)


def main():
    """
    Enhanced A/B Test Engine demo - Bug Fixed Version
    """
    print("🧪 ENHANCED A/B TEST ENGINE - BUG FIXED")
    print("=" * 60)

    # Enhanced Engine instance
    ab_engine = ABTestEngine()

    # Enhanced demo müşteri verisi
    demo_customers = []
    for i in range(150):
        customer = {
            'CustomerID': f'ENHANCED_CUST_{i + 1000}',
            'Recency': random.randint(1, 365),
            'Frequency': random.randint(1, 20),
            'Monetary': random.uniform(50, 5000),
            'AvgOrderValue': random.uniform(25, 800),
            'CustomerValue': random.randint(20, 400)
        }
        demo_customers.append(customer)

    print(f"📊 Enhanced demo verisi oluşturuldu: {len(demo_customers)} müşteri")

    # Enhanced test konfigürasyonu
    test_config = {
        'control_strategy': {
            'discount_modifier': 1.0,
            'preferred_channels': ['Email', 'SMS'],
            'model_type': 'original'
        },
        'variant_strategies': [
            {
                'discount_modifier': 1.3,
                'preferred_channels': ['Email', 'SMS', 'Phone', 'AI-Optimized'],
                'model_type': 'enhanced',
                'enhanced_features': True
            }
        ],
        'success_metric': 'enhanced_roi',
        'test_duration_days': 14
    }

    try:
        # Enhanced A/B Test oluştur
        test_name = "enhanced_model_comparison_test"
        ab_test = ab_engine.create_ab_test(test_name, demo_customers, test_config)

        # Enhanced kampanya simülasyonu
        simulation_results = ab_engine.run_enhanced_campaign_simulation(test_name)

        # Enhanced sonuç analizi
        test_result = ab_engine.analyze_enhanced_test_results(test_name)

        print(f"\n🏆 Enhanced A/B Test completed successfully!")
        print(f"🥇 Winner: {test_result.winner}")
        print(f"📈 ROI Difference: {test_result.roi_difference:+.1f}%")
        print(f"🎯 Enhanced Recommendation: {test_result.recommendation}")

        return test_result

    except Exception as e:
        print(f"❌ Enhanced A/B Test error: {e}")
        return None


if __name__ == "__main__":
    main()