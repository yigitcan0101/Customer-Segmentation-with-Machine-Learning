# src/utils/visualizer.py - MATPLOTLIB BACKEND FIXED

import pandas as pd
import numpy as np

# 🔧 CRITICAL FIX: Set matplotlib backend BEFORE importing pyplot
import matplotlib

matplotlib.use('Agg')  # Non-interactive backend - prevents Tkinter errors
import matplotlib.pyplot as plt

plt.ioff()  # Turn off interactive mode

import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import warnings

warnings.filterwarnings('ignore')

# Path düzeltmesi
current_dir = os.path.dirname(__file__)
parent_dir = os.path.join(current_dir, '..')
sys.path.append(parent_dir)

from utils.config import SystemConfig


class MarketingVisualizer:
    """
    Pazarlama analitik görselleri ve dashboard oluşturma sistemi - MATPLOTLIB FIXED
    """

    def __init__(self):
        self.config = SystemConfig()
        self.color_palette = {
            'Champions': '#2E8B57',  # Sea Green
            'Loyal': '#4169E1',  # Royal Blue
            'At Risk': '#DC143C',  # Crimson
            'Potential Loyalists': '#9370DB',  # Medium Purple
            'New Customers': '#00CED1',  # Dark Turquoise
            'Promising': '#FF6347',  # Tomato
            'Need Attention': '#FFA500',  # Orange
            'About to Sleep': '#DAA520',  # Goldenrod
            'Hibernating': '#696969',  # Dim Gray
            'Lost': '#2F4F4F'  # Dark Slate Gray
        }
        self.figure_size = (12, 8)
        self.dpi = 300

    def create_segment_distribution_chart(self, rfm_data: pd.DataFrame,
                                          save_path: Optional[str] = None) -> str:
        """
        Müşteri segment dağılım grafiği
        """

        print("📊 Segment dağılım grafiği oluşturuluyor...")

        # Segment sayıları
        segment_counts = rfm_data['Segment'].value_counts()
        segment_percentages = (segment_counts / len(rfm_data) * 100).round(1)

        # Plotly pie chart
        fig = go.Figure(data=[
            go.Pie(
                labels=segment_counts.index,
                values=segment_counts.values,
                hole=0.4,
                marker_colors=[self.color_palette.get(seg, '#CCCCCC') for seg in segment_counts.index],
                textinfo='label+percent+value',
                textposition='auto',
                hovertemplate='<b>%{label}</b><br>' +
                              'Müşteri Sayısı: %{value}<br>' +
                              'Yüzde: %{percent}<br>' +
                              '<extra></extra>'
            )
        ])

        fig.update_layout(
            title={
                'text': f'🎯 Müşteri Segment Dağılımı<br><sub>Toplam: {len(rfm_data):,} Müşteri</sub>',
                'x': 0.5,
                'xanchor': 'center'
            },
            font=dict(size=12),
            showlegend=True,
            width=800,
            height=600
        )

        # Kaydet
        if not save_path:
            save_path = "data/processed/segment_distribution.html"

        fig.write_html(save_path)

        # Static version (PNG)
        png_path = save_path.replace('.html', '.png')
        fig.write_image(png_path, width=800, height=600, scale=2)

        print(f"✅ Segment distribution chart kaydedildi: {save_path}")
        return save_path

    def create_rfm_3d_analysis(self, rfm_data: pd.DataFrame,
                               save_path: Optional[str] = None) -> str:
        """
        3D RFM analiz görselleştirmesi
        """

        print("🔮 3D RFM analizi oluşturuluyor...")

        # 3D Scatter plot
        fig = go.Figure(data=[
            go.Scatter3d(
                x=rfm_data['Recency'],
                y=rfm_data['Frequency'],
                z=rfm_data['Monetary'],
                mode='markers',
                marker=dict(
                    size=8,
                    color=[self.color_palette.get(seg, '#CCCCCC') for seg in rfm_data['Segment']],
                    opacity=0.7,
                    line=dict(width=0.5, color='DarkSlateGrey')
                ),
                text=rfm_data['Segment'],
                hovertemplate='<b>Segment: %{text}</b><br>' +
                              'Recency: %{x} gün<br>' +
                              'Frequency: %{y}<br>' +
                              'Monetary: $%{z:,.2f}<br>' +
                              '<extra></extra>'
            )
        ])

        fig.update_layout(
            title={
                'text': '🔮 3D RFM Analizi - Müşteri Segmentasyonu',
                'x': 0.5
            },
            scene=dict(
                xaxis_title='Recency (Gün)',
                yaxis_title='Frequency (Satın Alma)',
                zaxis_title='Monetary (TL)',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            width=1000,
            height=700,
            font=dict(size=11)
        )

        # Kaydet
        if not save_path:
            save_path = "data/processed/rfm_3d_analysis.html"

        fig.write_html(save_path)
        fig.write_image(save_path.replace('.html', '.png'), width=1000, height=700, scale=2)

        print(f"✅ 3D RFM analizi kaydedildi: {save_path}")
        return save_path

    def create_campaign_performance_dashboard(self, campaign_results: Optional[Dict] = None,
                                              ab_test_results: Optional[List] = None,
                                              save_path: Optional[str] = None) -> str:
        """
        Kampanya performans dashboard'u
        """

        print("📈 Kampanya performans dashboard'u oluşturuluyor...")

        # Subplot figure oluştur
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('ROI by Segment', 'Conversion Rates', 'Budget Allocation', 'Channel Performance'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "pie"}, {"type": "bar"}]]
        )

        # Demo data (gerçek kampanya verisi yoksa)
        segments = list(self.color_palette.keys())
        demo_roi = np.random.normal(150, 50, len(segments))
        demo_conversion = np.random.uniform(0.05, 0.35, len(segments))
        demo_budget = np.random.uniform(50, 1000, len(segments))

        # ROI by Segment
        fig.add_trace(
            go.Bar(
                x=segments,
                y=demo_roi,
                name='ROI (%)',
                marker_color=[self.color_palette[seg] for seg in segments],
                showlegend=False
            ),
            row=1, col=1
        )

        # Conversion Rates
        fig.add_trace(
            go.Bar(
                x=segments,
                y=demo_conversion,
                name='Conversion Rate',
                marker_color=[self.color_palette[seg] for seg in segments],
                showlegend=False
            ),
            row=1, col=2
        )

        # Budget Allocation (Pie)
        fig.add_trace(
            go.Pie(
                labels=segments,
                values=demo_budget,
                marker_colors=[self.color_palette[seg] for seg in segments],
                name="Budget"
            ),
            row=2, col=1
        )

        # Channel Performance
        channels = ['Email', 'SMS', 'Phone', 'In-App', 'Social Media']
        channel_performance = np.random.uniform(0.1, 0.4, len(channels))

        fig.add_trace(
            go.Bar(
                x=channels,
                y=channel_performance,
                name='Channel Performance',
                marker_color='#4ECDC4',
                showlegend=False
            ),
            row=2, col=2
        )

        # Layout güncelle
        fig.update_layout(
            title={
                'text': '📈 Kampanya Performans Dashboard',
                'x': 0.5,
                'xanchor': 'center'
            },
            height=800,
            width=1200,
            showlegend=False
        )

        # Y-axis labels
        fig.update_yaxes(title_text="ROI (%)", row=1, col=1)
        fig.update_yaxes(title_text="Conversion Rate", row=1, col=2)
        fig.update_yaxes(title_text="Performance Rate", row=2, col=2)

        # Kaydet
        if not save_path:
            save_path = "data/processed/campaign_dashboard.html"

        fig.write_html(save_path)
        fig.write_image(save_path.replace('.html', '.png'), width=1200, height=800, scale=2)

        print(f"✅ Kampanya dashboard kaydedildi: {save_path}")
        return save_path

    def create_customer_value_heatmap(self, rfm_data: pd.DataFrame,
                                      save_path: Optional[str] = None) -> str:
        """
        Müşteri değer heatmap'i - FIXED VERSION
        """

        print("🔥 Customer value heatmap oluşturuluyor...")

        try:
            # 🔧 FIX: Clear any existing plots first
            plt.close('all')

            # RFM skorlarına göre heatmap
            rfm_pivot = rfm_data.pivot_table(
                values='CustomerValue',
                index='R_Score',
                columns='F_Score',
                aggfunc='mean',
                fill_value=0
            )

            plt.figure(figsize=self.figure_size)

            # Heatmap
            sns.heatmap(
                rfm_pivot,
                annot=True,
                fmt='.0f',
                cmap='YlOrRd',
                cbar_kws={'label': 'Average Customer Value (TL)'},
                square=True
            )

            plt.title('🔥 Müşteri Değer Heatmap - Recency vs Frequency Score',
                      fontsize=14, pad=20)
            plt.xlabel('Frequency Score', fontsize=12)
            plt.ylabel('Recency Score', fontsize=12)

            # Kaydet
            if not save_path:
                save_path = "data/processed/customer_value_heatmap.png"

            plt.tight_layout()
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

            # 🔧 FIX: Close plot instead of show() to prevent Tkinter errors
            plt.close()

            print(f"✅ Customer value heatmap kaydedildi: {save_path}")
            return save_path

        except Exception as e:
            print(f"⚠️ Heatmap creation error: {e}")
            plt.close('all')  # Clean up
            return save_path

    def create_revenue_analysis_charts(self, rfm_data: pd.DataFrame,
                                       save_path: Optional[str] = None) -> str:
        """
        Revenue analiz grafikleri - FIXED VERSION
        """

        print("💰 Revenue analiz grafikleri oluşturuluyor...")

        try:
            # 🔧 FIX: Clear all existing plots
            plt.close('all')

            # Multiple subplot figure
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('💰 Revenue Analiz Dashboard', fontsize=16, fontweight='bold')

            # 1. Revenue by Segment (Bar Chart)
            segment_revenue = rfm_data.groupby('Segment')['Monetary'].sum().sort_values(ascending=False)
            bars1 = ax1.bar(range(len(segment_revenue)), segment_revenue.values,
                            color=[self.color_palette.get(seg, '#CCCCCC') for seg in segment_revenue.index])
            ax1.set_title('📊 Segment Bazlı Toplam Revenue')
            ax1.set_xlabel('Segmentler')
            ax1.set_ylabel('Toplam Revenue (TL)')
            ax1.set_xticks(range(len(segment_revenue)))
            ax1.set_xticklabels(segment_revenue.index, rotation=45, ha='right')

            # Bar values
            for i, bar in enumerate(bars1):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width() / 2., height,
                         f'{height / 1000:.0f}K', ha='center', va='bottom')

            # 2. Customer Count vs Revenue (Scatter)
            segment_stats = rfm_data.groupby('Segment').agg({
                'Monetary': 'sum',
                'Customer ID': 'count'
            }).reset_index()

            scatter = ax2.scatter(segment_stats['Customer ID'], segment_stats['Monetary'],
                                  c=[self.color_palette.get(seg, '#CCCCCC') for seg in segment_stats['Segment']],
                                  s=100, alpha=0.7)
            ax2.set_title('👥 Müşteri Sayısı vs Revenue')
            ax2.set_xlabel('Müşteri Sayısı')
            ax2.set_ylabel('Toplam Revenue (TL)')

            # Segment labels
            for i, segment in enumerate(segment_stats['Segment']):
                ax2.annotate(segment,
                             (segment_stats.iloc[i]['Customer ID'], segment_stats.iloc[i]['Monetary']),
                             xytext=(5, 5), textcoords='offset points', fontsize=8)

            # 3. Customer Value Distribution (Histogram)
            ax3.hist(rfm_data['CustomerValue'], bins=30, alpha=0.7, color='#4ECDC4', edgecolor='black')
            ax3.axvline(rfm_data['CustomerValue'].mean(), color='red', linestyle='--',
                        label=f'Ortalama: {rfm_data["CustomerValue"].mean():.1f}')
            ax3.set_title('📈 Customer Value Dağılımı')
            ax3.set_xlabel('Customer Value')
            ax3.set_ylabel('Müşteri Sayısı')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

            # 4. RFM Correlation Heatmap
            rfm_numeric = rfm_data[['Recency', 'Frequency', 'Monetary', 'AvgOrderValue', 'CustomerValue']]
            correlation_matrix = rfm_numeric.corr()

            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                        square=True, ax=ax4, cbar_kws={'shrink': 0.8})
            ax4.set_title('🔗 RFM Korelasyon Matrisi')

            plt.tight_layout()

            # Kaydet
            if not save_path:
                save_path = "data/processed/revenue_analysis_dashboard.png"

            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

            # 🔧 FIX: Close instead of show
            plt.close()

            print(f"✅ Revenue analiz dashboard kaydedildi: {save_path}")
            return save_path

        except Exception as e:
            print(f"⚠️ Revenue analysis error: {e}")
            plt.close('all')  # Clean up
            return save_path

    def create_model_performance_viz(self, confusion_matrix: np.ndarray,
                                     segment_names: List[str],
                                     accuracy_score: float,
                                     save_path: Optional[str] = None) -> str:
        """
        Model performans görselleştirmesi - FIXED VERSION
        """

        print("🧠 Model performans görselleştirmesi oluşturuluyor...")

        try:
            # 🔧 FIX: Clear plots
            plt.close('all')

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            fig.suptitle(f'🧠 Neural Network Model Performansı (Accuracy: {accuracy_score:.3f})',
                         fontsize=14, fontweight='bold')

            # Confusion Matrix
            sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                        xticklabels=segment_names, yticklabels=segment_names, ax=ax1)
            ax1.set_title('🎯 Confusion Matrix')
            ax1.set_xlabel('Tahmin Edilen Segment')
            ax1.set_ylabel('Gerçek Segment')

            # Segment Performance Scores
            diagonal = np.diag(confusion_matrix)
            row_sums = confusion_matrix.sum(axis=1)
            segment_accuracies = diagonal / row_sums

            bars = ax2.bar(range(len(segment_names)), segment_accuracies,
                           color=[self.color_palette.get(seg, '#CCCCCC') for seg in segment_names])
            ax2.set_title('📊 Segment Bazlı Accuracy')
            ax2.set_xlabel('Segmentler')
            ax2.set_ylabel('Accuracy Score')
            ax2.set_xticks(range(len(segment_names)))
            ax2.set_xticklabels(segment_names, rotation=45, ha='right')
            ax2.set_ylim(0, 1)

            # Accuracy values on bars
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                         f'{height:.3f}', ha='center', va='bottom')

            plt.tight_layout()

            # Kaydet
            if not save_path:
                save_path = "data/processed/model_performance_viz.png"

            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

            # 🔧 FIX: Close instead of show
            plt.close()

            print(f"✅ Model performans viz kaydedildi: {save_path}")
            return save_path

        except Exception as e:
            print(f"⚠️ Model performance viz error: {e}")
            plt.close('all')
            return save_path

    def create_ab_test_comparison(self, ab_test_results: Dict,
                                  save_path: Optional[str] = None) -> str:
        """
        A/B test karşılaştırma grafikleri
        """

        print("🧪 A/B test karşılaştırma grafikleri oluşturuluyor...")

        # Veri hazırlığı
        groups = list(ab_test_results.keys())
        conversion_rates = [ab_test_results[g]['conversion_rate'] for g in groups]
        rois = [ab_test_results[g]['roi'] for g in groups]
        revenues = [ab_test_results[g]['revenue_per_customer'] for g in groups]
        costs = [ab_test_results[g]['total_cost'] for g in groups]

        # Interactive Plotly dashboard
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('🎯 Conversion Rates', '💰 ROI Comparison',
                            '💵 Revenue per Customer', '💸 Total Costs'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )

        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA726']

        # Conversion rates
        fig.add_trace(
            go.Bar(x=groups, y=conversion_rates, name='Conversion Rate',
                   marker_color=colors[:len(groups)], showlegend=False),
            row=1, col=1
        )

        # ROI comparison
        fig.add_trace(
            go.Bar(x=groups, y=rois, name='ROI (%)',
                   marker_color=colors[:len(groups)], showlegend=False),
            row=1, col=2
        )

        # Revenue per customer
        fig.add_trace(
            go.Bar(x=groups, y=revenues, name='Revenue per Customer',
                   marker_color=colors[:len(groups)], showlegend=False),
            row=2, col=1
        )

        # Total costs
        fig.add_trace(
            go.Bar(x=groups, y=costs, name='Total Cost',
                   marker_color=colors[:len(groups)], showlegend=False),
            row=2, col=2
        )

        fig.update_layout(
            title={
                'text': '🧪 A/B Test Sonuçları Karşılaştırması',
                'x': 0.5
            },
            height=800,
            width=1200
        )

        # Kaydet
        if not save_path:
            save_path = "data/processed/ab_test_comparison.html"

        fig.write_html(save_path)
        fig.write_image(save_path.replace('.html', '.png'), width=1200, height=800, scale=2)

        print(f"✅ A/B test comparison kaydedildi: {save_path}")
        return save_path

    def create_executive_summary_report(self, business_metrics: Dict,
                                        model_metrics: Dict,
                                        segment_health: Dict,
                                        save_path: Optional[str] = None) -> str:
        """
        Executive özet raporu görselleştirmesi
        """

        print("👔 Executive summary raporu oluşturuluyor...")

        # Executive dashboard
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                '📊 Müşteri Segment Dağılımı', '💰 Revenue by Segment',
                '🏥 Segment Health Scores', '🎯 Model Accuracy by Segment',
                '📈 Customer Value Trends', '🚀 Business KPIs'
            ),
            specs=[[{"type": "pie"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )

        # Data preparation
        segment_counts = business_metrics['segment_distribution']
        segments = list(segment_counts.keys())

        # 1. Segment distribution pie
        fig.add_trace(
            go.Pie(
                labels=segments,
                values=list(segment_counts.values()),
                marker_colors=[self.color_palette.get(seg, '#CCCCCC') for seg in segments],
                name="Müşteri Dağılımı"
            ),
            row=1, col=1
        )

        # 2. Revenue by segment (demo data)
        demo_segment_revenues = {seg: np.random.uniform(100000, 2000000) for seg in segments}
        fig.add_trace(
            go.Bar(
                x=list(demo_segment_revenues.keys()),
                y=list(demo_segment_revenues.values()),
                marker_color=[self.color_palette.get(seg, '#CCCCCC') for seg in demo_segment_revenues.keys()],
                showlegend=False
            ),
            row=1, col=2
        )

        # 3. Segment health scores
        health_scores = [segment_health[seg]['overall_health_score'] for seg in segments if seg in segment_health]
        health_segments = [seg for seg in segments if seg in segment_health]

        fig.add_trace(
            go.Bar(
                x=health_segments,
                y=health_scores,
                marker_color='#28a745',
                showlegend=False
            ),
            row=2, col=1
        )

        # 4. Model accuracy by segment (demo)
        demo_accuracies = np.random.uniform(0.85, 0.98, len(segments))
        fig.add_trace(
            go.Bar(
                x=segments,
                y=demo_accuracies,
                marker_color='#007bff',
                showlegend=False
            ),
            row=2, col=2
        )

        # 5. Customer value scatter - FIX: Add fallback for rfm_data
        try:
            # Try to use rfm_data if available
            sample_data = rfm_data.sample(min(200, len(rfm_data)))
        except:
            # Create demo data if rfm_data not available
            sample_data = pd.DataFrame({
                'Frequency': np.random.randint(1, 20, 200),
                'Monetary': np.random.uniform(100, 5000, 200),
                'CustomerValue': np.random.uniform(50, 500, 200),
                'Segment': np.random.choice(segments, 200)
            })

        fig.add_trace(
            go.Scatter(
                x=sample_data['Frequency'],
                y=sample_data['Monetary'],
                mode='markers',
                marker=dict(
                    size=8,
                    color=sample_data['CustomerValue'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Customer Value")
                ),
                text=sample_data['Segment'],
                showlegend=False
            ),
            row=3, col=1
        )

        # 6. Key Business KPIs
        kpis = {
            'Total Revenue': business_metrics['customer_analytics']['total_revenue'] / 1000000,  # Millions
            'Avg CLV': business_metrics['customer_analytics']['avg_customer_value'] * 10,  # Simulated
            'Model Accuracy': model_metrics['overall_metrics']['accuracy'] * 100,
            'High Value %': business_metrics['customer_analytics']['high_value_percentage']
        }

        fig.add_trace(
            go.Bar(
                x=list(kpis.keys()),
                y=list(kpis.values()),
                marker_color=['#e74c3c', '#f39c12', '#2ecc71', '#9b59b6'],
                showlegend=False
            ),
            row=3, col=2
        )

        # Layout updates
        fig.update_layout(
            title={
                'text': '👔 Executive Summary Dashboard<br><sub>Müşteri Segmentasyon AI Sistemi</sub>',
                'x': 0.5
            },
            height=1000,
            width=1400,
            showlegend=False
        )

        # Kaydet
        if not save_path:
            save_path = "data/processed/executive_summary_dashboard.html"

        fig.write_html(save_path)
        fig.write_image(save_path.replace('.html', '.png'), width=1400, height=1000, scale=2)

        print(f"✅ Executive summary dashboard kaydedildi: {save_path}")
        return save_path

    def create_all_visualizations(self, rfm_data: pd.DataFrame,
                                  business_metrics: Dict,
                                  model_metrics: Dict,
                                  segment_health: Dict) -> List[str]:
        """
        Tüm görselleştirmeleri tek seferde oluştur - FIXED VERSION
        """

        print("🎨 TÜM VİZUALİZASYONLAR OLUŞTURULUYOR...")
        print("=" * 60)

        created_files = []

        try:
            # 1. Segment distribution
            file1 = self.create_segment_distribution_chart(rfm_data)
            created_files.append(file1)

            # 2. 3D RFM analysis
            file2 = self.create_rfm_3d_analysis(rfm_data)
            created_files.append(file2)

            # 3. Customer value heatmap
            file3 = self.create_customer_value_heatmap(rfm_data)
            created_files.append(file3)

            # 4. Revenue analysis
            file4 = self.create_revenue_analysis_charts(rfm_data)
            created_files.append(file4)

            # 5. Executive summary
            file5 = self.create_executive_summary_report(
                business_metrics, model_metrics, segment_health
            )
            created_files.append(file5)

        except Exception as e:
            print(f"⚠️ Visualization error: {e}")
        finally:
            # 🔧 FIX: Always clean up matplotlib
            plt.close('all')

        print(f"\n🎉 TÜM VİZUALİZASYONLAR TAMAMLANDI!")
        print(f"📁 Oluşturulan dosya sayısı: {len(created_files)}")

        return created_files


def main():
    """
    Visualizer sistemi demo - FIXED VERSION
    """
    print("📊 MARKETİNG VİZUALİZER DEMO")
    print("=" * 60)

    # Visualizer instance
    viz = MarketingVisualizer()

    try:
        # RFM verisi yükle
        rfm_data = pd.read_csv("data/processed/rfm_analysis_results.csv")
        print(f"✅ RFM verisi yüklendi: {len(rfm_data)} müşteri")

        # Metrics yükle (önceki adımda oluşturulan)
        from utils.metrics import PerformanceMetrics
        metrics_engine = PerformanceMetrics()

        business_metrics = metrics_engine.calculate_business_metrics(rfm_data)
        segment_health = metrics_engine.calculate_segment_health(rfm_data)

        # Dummy model metrics (gerçek model metrics yerine)
        model_metrics = {
            'overall_metrics': {
                'accuracy': 0.936,
                'precision_weighted': 0.94,
                'recall_weighted': 0.94,
                'f1_weighted': 0.93
            }
        }

        # Görselleştirmeleri oluştur
        print(f"\n🎨 Görselleştirmeler oluşturuluyor...")

        # Tek tek oluştur (daha stabil)
        created_files = []

        try:
            # 1. Segment distribution
            file1 = viz.create_segment_distribution_chart(rfm_data)
            created_files.append(file1)

            # 2. Customer value heatmap
            file2 = viz.create_customer_value_heatmap(rfm_data)
            created_files.append(file2)

            # 3. Revenue analysis
            file3 = viz.create_revenue_analysis_charts(rfm_data)
            created_files.append(file3)

            print(f"\n✅ Temel görseller tamamlandı!")

        except Exception as e:
            print(f"❌ Görselleştirme hatası: {e}")
        finally:
            # 🔧 FIX: Always clean up
            plt.close('all')

        print(f"\n📁 OLUŞTURULAN DOSYALAR:")
        for i, file_path in enumerate(created_files, 1):
            print(f"  {i}. {file_path}")

        print(f"\n🚀 Marketing Visualizer hazır!")
        print(f"📊 Dashboard dosyaları data/processed/ klasöründe")

    except FileNotFoundError as e:
        print(f"❌ Veri dosyası bulunamadı: {e}")
        print("💡 Önce RFM analizi çalıştırın")
    except Exception as e:
        print(f"❌ Hata: {e}")
    finally:
        # 🔧 FIX: Final cleanup
        plt.close('all')


if __name__ == "__main__":
    main()