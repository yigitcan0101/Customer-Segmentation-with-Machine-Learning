# src/utils/metrics.py

import pandas as pd
import numpy as np
import json
import sys
import os
from datetime import datetime, timedelta
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
from scipy import stats
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

# Path düzeltmesi
current_dir = os.path.dirname(__file__)
parent_dir = os.path.join(current_dir, '..')
sys.path.append(parent_dir)

from utils.config import SystemConfig


class PerformanceMetrics:
    """
    Müşteri segmentasyon sistemi performans metrikleri hesaplama ve izleme
    """

    def __init__(self):
        self.config = SystemConfig()
        self.metric_history = []
        self.alert_thresholds = self.config.METRICS_CONFIG['alert_thresholds']

    def calculate_model_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                y_pred_prob: Optional[np.ndarray] = None,
                                segment_names: Optional[List[str]] = None) -> Dict:
        """
        Neural Network model performans metrikleri hesapla

        Args:
            y_true: Gerçek segment labels
            y_pred: Tahmin edilen segment labels
            y_pred_prob: Prediction probabilities (opsiyonel)
            segment_names: Segment isimleri (opsiyonel)
        """

        print("📊 Model performans metrikleri hesaplanıyor...")

        # Temel metrikler
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )

        # Weighted averages
        precision_weighted = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )[0]
        recall_weighted = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )[1]
        f1_weighted = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )[2]

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Segment bazlı performans
        segment_performance = {}
        if segment_names:
            for i, segment in enumerate(segment_names):
                if i < len(precision):
                    segment_performance[segment] = {
                        'precision': float(precision[i]),
                        'recall': float(recall[i]),
                        'f1_score': float(f1[i]),
                        'support': int(support[i])
                    }

        # Confidence intervals (eğer probability verilmişse)
        confidence_metrics = {}
        if y_pred_prob is not None:
            max_probs = np.max(y_pred_prob, axis=1)
            confidence_metrics = {
                'mean_confidence': float(np.mean(max_probs)),
                'std_confidence': float(np.std(max_probs)),
                'high_confidence_rate': float(np.mean(max_probs > 0.8)),
                'low_confidence_rate': float(np.mean(max_probs < 0.5))
            }

        metrics = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'overall_metrics': {
                'accuracy': float(accuracy),
                'precision_weighted': float(precision_weighted),
                'recall_weighted': float(recall_weighted),
                'f1_weighted': float(f1_weighted),
                'total_samples': len(y_true)
            },
            'segment_metrics': segment_performance,
            'confidence_metrics': confidence_metrics,
            'confusion_matrix': cm.tolist(),
            'classification_report': classification_report(
                y_true, y_pred, target_names=segment_names, output_dict=True
            ) if segment_names else {}
        }

        print(f"✅ Model metrics hesaplandı - Accuracy: {accuracy:.4f}")
        return metrics

    def calculate_business_metrics(self, rfm_data: pd.DataFrame,
                                   campaign_results: Optional[Dict] = None) -> Dict:
        """
        İş metrikleri hesapla (ROI, CLV, Conversion rates)

        Args:
            rfm_data: RFM analiz sonuçları DataFrame
            campaign_results: Kampanya sonuçları (opsiyonel)
        """

        print("💰 Business performans metrikleri hesaplanıyor...")

        # RFM bazlı business metrics
        total_customers = len(rfm_data)
        total_revenue = rfm_data['Monetary'].sum()
        avg_customer_value = rfm_data['CustomerValue'].mean()
        avg_order_value = rfm_data['AvgOrderValue'].mean()

        # Segment dağılımı
        segment_distribution = rfm_data['Segment'].value_counts().to_dict()
        segment_percentages = {
            segment: (count / total_customers) * 100
            for segment, count in segment_distribution.items()
        }

        # Segment bazlı CLV analizi
        segment_clv = rfm_data.groupby('Segment').agg({
            'Monetary': ['mean', 'sum', 'count'],
            'Frequency': 'mean',
            'Recency': 'mean',
            'CustomerValue': 'mean'
        }).round(2)

        # High-value customer analysis
        high_value_threshold = rfm_data['CustomerValue'].quantile(0.8)
        high_value_customers = rfm_data[rfm_data['CustomerValue'] >= high_value_threshold]
        high_value_percentage = (len(high_value_customers) / total_customers) * 100

        # Churn risk analysis
        churn_risk_segments = ['At Risk', 'About to Sleep', 'Hibernating', 'Lost']
        churn_risk_customers = rfm_data[rfm_data['Segment'].isin(churn_risk_segments)]
        churn_risk_percentage = (len(churn_risk_customers) / total_customers) * 100

        business_metrics = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'customer_analytics': {
                'total_customers': total_customers,
                'total_revenue': float(total_revenue),
                'avg_customer_value': float(avg_customer_value),
                'avg_order_value': float(avg_order_value),
                'high_value_percentage': float(high_value_percentage),
                'churn_risk_percentage': float(churn_risk_percentage)
            },
            'segment_distribution': segment_distribution,
            'segment_percentages': segment_percentages,
            'segment_clv_analysis': segment_clv.to_dict(),
            'revenue_concentration': {
                'top_20_percent_revenue_share': float(
                    rfm_data.nlargest(int(total_customers * 0.2), 'Monetary')['Monetary'].sum() / total_revenue * 100
                )
            }
        }

        # Kampanya sonuçları varsa ekle
        if campaign_results:
            business_metrics['campaign_performance'] = self._analyze_campaign_performance(campaign_results)

        print(f"✅ Business metrics hesaplandı - Total Revenue: ${total_revenue:,.2f}")
        return business_metrics

    def calculate_segment_health(self, rfm_data: pd.DataFrame) -> Dict:
        """
        Segment sağlık skorları hesapla
        """

        print("🏥 Segment sağlık skorları hesaplanıyor...")

        segment_health = {}
        segment_targets = self.config.get_segment_targets()

        for segment in rfm_data['Segment'].unique():
            segment_data = rfm_data[rfm_data['Segment'] == segment]
            target_config = segment_targets.get(segment, {})

            # Health score components
            size_score = min(len(segment_data) / 50, 1.0)  # Ideal: 50+ customers
            value_score = min(segment_data['CustomerValue'].mean() / 200, 1.0)  # Ideal: 200+ value
            recency_score = 1.0 - min(segment_data['Recency'].mean() / 180, 1.0)  # Ideal: < 180 days
            frequency_score = min(segment_data['Frequency'].mean() / 10, 1.0)  # Ideal: 10+ frequency

            # Overall health score (0-1)
            overall_health = (size_score + value_score + recency_score + frequency_score) / 4

            # Health level categorization
            if overall_health >= 0.8:
                health_level = "EXCELLENT"
            elif overall_health >= 0.6:
                health_level = "GOOD"
            elif overall_health >= 0.4:
                health_level = "MODERATE"
            else:
                health_level = "POOR"

            segment_health[segment] = {
                'overall_health_score': float(overall_health),
                'health_level': health_level,
                'customer_count': len(segment_data),
                'avg_customer_value': float(segment_data['CustomerValue'].mean()),
                'avg_recency': float(segment_data['Recency'].mean()),
                'avg_frequency': float(segment_data['Frequency'].mean()),
                'total_revenue': float(segment_data['Monetary'].sum()),
                'size_score': float(size_score),
                'value_score': float(value_score),
                'recency_score': float(recency_score),
                'frequency_score': float(frequency_score),
                'target_metrics': target_config
            }

        print(f"✅ Segment health analizi tamamlandı - {len(segment_health)} segment")
        return segment_health

    def calculate_campaign_effectiveness(self, ab_test_results: List[Dict]) -> Dict:
        """
        Kampanya etkinlik metrikleri hesapla
        """

        print("🎯 Kampanya etkinlik metrikleri hesaplanıyor...")

        if not ab_test_results:
            return {"error": "No A/B test results available"}

        # Tüm test sonuçlarını topla
        total_tests = len(ab_test_results)
        winning_strategies = []
        roi_improvements = []
        conversion_improvements = []

        for test_result in ab_test_results:
            if hasattr(test_result, 'winner'):
                winning_strategies.append(test_result.winner)
                roi_improvements.append(test_result.roi_difference)
                conversion_improvements.append(test_result.lift)

        # Campaign effectiveness summary
        effectiveness_metrics = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_tests_run': total_tests,
            'successful_optimizations': len([x for x in roi_improvements if x > 0]),
            'avg_roi_improvement': float(np.mean(roi_improvements)) if roi_improvements else 0,
            'avg_conversion_lift': float(np.mean(conversion_improvements)) if conversion_improvements else 0,
            'best_performing_strategy': max(set(winning_strategies),
                                            key=winning_strategies.count) if winning_strategies else 'N/A',
            'strategy_win_rates': {
                strategy: winning_strategies.count(strategy) / len(winning_strategies) * 100
                for strategy in set(winning_strategies)
            } if winning_strategies else {},
            'optimization_success_rate': (
                        len([x for x in roi_improvements if x > 0]) / total_tests * 100) if total_tests > 0 else 0
        }

        print(
            f"✅ Campaign effectiveness hesaplandı - Success Rate: {effectiveness_metrics['optimization_success_rate']:.1f}%")
        return effectiveness_metrics

    def generate_alert_check(self, current_metrics: Dict,
                             historical_metrics: Optional[List[Dict]] = None) -> List[Dict]:
        """
        Performans threshold'larına göre alert kontrolü
        """

        alerts = []
        thresholds = self.alert_thresholds

        # Model performance alerts
        if 'overall_metrics' in current_metrics:
            current_accuracy = current_metrics['overall_metrics'].get('accuracy', 0)

            if historical_metrics:
                # Historical comparison
                recent_accuracies = [
                    m.get('overall_metrics', {}).get('accuracy', 0)
                    for m in historical_metrics[-5:]  # Son 5 metric
                ]
                if recent_accuracies:
                    avg_historical_accuracy = np.mean(recent_accuracies)
                    accuracy_drop = avg_historical_accuracy - current_accuracy

                    if accuracy_drop > thresholds['accuracy_drop']:
                        alerts.append({
                            'type': 'MODEL_PERFORMANCE',
                            'severity': 'HIGH',
                            'message': f'Model accuracy dropped by {accuracy_drop:.3f}',
                            'current_value': current_accuracy,
                            'threshold': avg_historical_accuracy - thresholds['accuracy_drop'],
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        })

            # Absolute accuracy threshold
            if current_accuracy < 0.8:  # 80% minimum
                alerts.append({
                    'type': 'MODEL_ACCURACY',
                    'severity': 'CRITICAL',
                    'message': f'Model accuracy below 80%: {current_accuracy:.3f}',
                    'current_value': current_accuracy,
                    'threshold': 0.8,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })

        print(f"🚨 Alert check tamamlandı - {len(alerts)} alert bulundu")
        return alerts

    def calculate_segment_performance_trends(self, rfm_data: pd.DataFrame,
                                             time_column: Optional[str] = None) -> Dict:
        """
        Segment performans trendlerini hesapla
        """

        print("📈 Segment performans trendleri hesaplanıyor...")

        # Segment bazlı trend analizi
        segment_trends = {}

        for segment in rfm_data['Segment'].unique():
            segment_data = rfm_data[rfm_data['Segment'] == segment]

            # Trend metrikleri
            trends = {
                'customer_count_trend': len(segment_data),
                'avg_monetary_trend': float(segment_data['Monetary'].mean()),
                'avg_frequency_trend': float(segment_data['Frequency'].mean()),
                'avg_recency_trend': float(segment_data['Recency'].mean()),
                'customer_value_trend': float(segment_data['CustomerValue'].mean()),
                'revenue_share': float(segment_data['Monetary'].sum() / rfm_data['Monetary'].sum() * 100)
            }

            # Segment growth potential
            if segment in ['Champions', 'Loyal']:
                growth_potential = 'MAINTAIN'
            elif segment in ['Potential Loyalists', 'Promising', 'New Customers']:
                growth_potential = 'GROW'
            elif segment in ['At Risk', 'Need Attention', 'About to Sleep']:
                growth_potential = 'RECOVER'
            else:
                growth_potential = 'MINIMAL'

            trends['growth_potential'] = growth_potential
            segment_trends[segment] = trends

        print(f"✅ Segment trends hesaplandı - {len(segment_trends)} segment")
        return segment_trends

    def calculate_roi_analytics(self, campaign_data: List[Dict]) -> Dict:
        """
        ROI detaylı analitik hesaplamaları
        """

        print("💹 ROI analytics hesaplanıyor...")

        if not campaign_data:
            return {"error": "No campaign data available"}

        # ROI statistics
        rois = [campaign.get('roi', 0) for campaign in campaign_data]
        revenues = [campaign.get('total_revenue', 0) for campaign in campaign_data]
        costs = [campaign.get('total_cost', 0) for campaign in campaign_data]

        roi_analytics = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'roi_statistics': {
                'mean_roi': float(np.mean(rois)),
                'median_roi': float(np.median(rois)),
                'std_roi': float(np.std(rois)),
                'min_roi': float(np.min(rois)),
                'max_roi': float(np.max(rois)),
                'roi_25_percentile': float(np.percentile(rois, 25)),
                'roi_75_percentile': float(np.percentile(rois, 75))
            },
            'revenue_analytics': {
                'total_revenue': float(np.sum(revenues)),
                'total_cost': float(np.sum(costs)),
                'overall_roi': float((np.sum(revenues) - np.sum(costs)) / np.sum(costs) * 100) if np.sum(
                    costs) > 0 else 0,
                'revenue_per_campaign': float(np.mean(revenues)),
                'cost_per_campaign': float(np.mean(costs))
            },
            'profitability_analysis': {
                'profitable_campaigns': len([r for r in rois if r > 0]),
                'unprofitable_campaigns': len([r for r in rois if r <= 0]),
                'profitability_rate': float(len([r for r in rois if r > 0]) / len(rois) * 100) if rois else 0
            }
        }

        print(f"✅ ROI analytics hesaplandı - Overall ROI: {roi_analytics['revenue_analytics']['overall_roi']:.1f}%")
        return roi_analytics

    def calculate_customer_lifetime_value(self, rfm_data: pd.DataFrame) -> Dict:
        """
        Customer Lifetime Value hesaplamaları
        """

        print("💎 Customer Lifetime Value hesaplanıyor...")

        # CLV formülü: Avg Order Value × Purchase Frequency × Customer Lifespan
        # Simplified CLV = Monetary × Frequency multiplier

        clv_metrics = {}

        for segment in rfm_data['Segment'].unique():
            segment_data = rfm_data[rfm_data['Segment'] == segment]

            # CLV hesaplama components
            avg_monetary = segment_data['Monetary'].mean()
            avg_frequency = segment_data['Frequency'].mean()
            avg_recency = segment_data['Recency'].mean()

            # Lifespan estimate (based on recency and frequency)
            estimated_lifespan_months = max(12 - (avg_recency / 30), 1)  # Min 1 month

            # Simplified CLV calculation
            estimated_clv = avg_monetary * (avg_frequency / 12) * estimated_lifespan_months

            clv_metrics[segment] = {
                'customer_count': len(segment_data),
                'avg_monetary': float(avg_monetary),
                'avg_frequency': float(avg_frequency),
                'avg_recency': float(avg_recency),
                'estimated_lifespan_months': float(estimated_lifespan_months),
                'estimated_clv': float(estimated_clv),
                'total_segment_value': float(estimated_clv * len(segment_data))
            }

        # Overall CLV metrics
        total_customers = len(rfm_data)
        avg_clv = np.mean([metrics['estimated_clv'] for metrics in clv_metrics.values()])
        total_customer_base_value = sum([metrics['total_segment_value'] for metrics in clv_metrics.values()])

        result = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'overall_clv_metrics': {
                'total_customers': total_customers,
                'avg_clv': float(avg_clv),
                'total_customer_base_value': float(total_customer_base_value),
                'clv_per_customer': float(total_customer_base_value / total_customers) if total_customers > 0 else 0
            },
            'segment_clv_metrics': clv_metrics
        }

        print(f"✅ CLV analizi tamamlandı - Avg CLV: ${avg_clv:.2f}")
        return result

    def _analyze_campaign_performance(self, campaign_results: Dict) -> Dict:
        """
        Kampanya performans analizi
        """

        performance_data = {
            'total_campaigns': len(campaign_results),
            'avg_conversion_rate': 0,
            'avg_roi': 0,
            'best_performing_segment': '',
            'worst_performing_segment': ''
        }

        # Detaylı analiz implementation burada olacak
        # (campaign_results structure'ına göre)

        return performance_data

    def export_metrics_report(self, metrics_data: Dict, filename: Optional[str] = None) -> str:
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"performance_metrics_{timestamp}"

        # Exports klasörünü oluştur
        export_dir = "data/processed/exports/"
        os.makedirs(export_dir, exist_ok=True)

        # Sadece string export (JSON skip)
        txt_path = f"{export_dir}{filename}.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"Performance Metrics Report - {datetime.now()}\n")
            f.write("=" * 60 + "\n")
            f.write(str(metrics_data))

        print(f"✅ Metrics exported: {txt_path}")
        return txt_path

    def get_performance_summary(self) -> Dict:
        """
        Genel performans özeti
        """

        if not self.metric_history:
            return {"message": "No metrics history available"}

        latest_metrics = self.metric_history[-1]['metrics']

        summary = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'system_status': 'OPERATIONAL',
            'key_metrics': {
                'model_accuracy': latest_metrics.get('overall_metrics', {}).get('accuracy', 0),
                'total_customers': latest_metrics.get('customer_analytics', {}).get('total_customers', 0),
                'total_revenue': latest_metrics.get('customer_analytics', {}).get('total_revenue', 0),
                'avg_roi': latest_metrics.get('revenue_analytics', {}).get('overall_roi', 0)
            },
            'health_status': {
                'model_health': 'GOOD' if latest_metrics.get('overall_metrics', {}).get('accuracy',
                                                                                        0) > 0.85 else 'NEEDS_ATTENTION',
                'data_quality': 'GOOD',
                'system_performance': 'OPTIMAL'
            }
        }

        return summary


def main():
    """
    Metrics sistemi demo ve test
    """
    print("📊 PERFORMANS METRİKLERİ SİSTEMİ DEMO")
    print("=" * 60)

    # Metrics instance
    metrics = PerformanceMetrics()

    # RFM verisi yükle
    try:
        rfm_data = pd.read_csv("data/processed/rfm_analysis_results.csv")
        print(f"✅ RFM verisi yüklendi: {len(rfm_data)} müşteri")

        # Business metrics hesapla
        business_metrics = metrics.calculate_business_metrics(rfm_data)

        # Segment health analizi
        segment_health = metrics.calculate_segment_health(rfm_data)

        # CLV analizi
        clv_metrics = metrics.calculate_customer_lifetime_value(rfm_data)

        # Genel özet
        summary = metrics.get_performance_summary()

        print(f"\n📊 PERFORMANS ÖZETİ:")
        print(f"  💰 Total Revenue: ${business_metrics['customer_analytics']['total_revenue']:,.2f}")
        print(f"  👥 Total Customers: {business_metrics['customer_analytics']['total_customers']:,}")
        print(f"  📈 Avg Customer Value: ${business_metrics['customer_analytics']['avg_customer_value']:.2f}")
        print(f"  💎 Avg CLV: ${clv_metrics['overall_clv_metrics']['avg_clv']:.2f}")

        print(f"\n🏥 SEGMENT HEALTH TOP 3:")
        health_sorted = sorted(segment_health.items(),
                               key=lambda x: x[1]['overall_health_score'], reverse=True)
        for segment, health in health_sorted[:3]:
            print(f"  {segment:<20}: {health['health_level']} ({health['overall_health_score']:.3f})")

        # Export
        all_metrics = {
            'business_metrics': business_metrics,
            'segment_health': segment_health,
            'clv_metrics': clv_metrics
        }

        export_path = metrics.export_metrics_report(all_metrics)
        print(f"\n💾 Metrics exported to: {export_path}")

    except FileNotFoundError:
        print("❌ RFM data bulunamadı! Önce RFM analizi çalıştırın.")

    print(f"\n🚀 Performance Metrics sistemi hazır!")


if __name__ == "__main__":
    main()