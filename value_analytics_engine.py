# src/analytics/value_analytics_engine.py

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns


class ValueBasedAnalyticsEngine:
    """
    Value-Based Segmentation Analytics
    Detaylı değer bazlı segmentasyon ve analitik
    """

    def __init__(self):
        self.ml_engine = None
        self.enhanced_data = None

    def load_enhanced_data(self):
        """Enhanced ML data ve raw data yükle"""
        try:
            self.enhanced_data = pd.read_csv("data/processed/ml_enhanced_rfm_dataset.csv")
            print(f"✅ Enhanced data loaded: {len(self.enhanced_data):,} customers")

            # Enhanced dataset kolonlarını göster
            print(f"📊 Available columns: {list(self.enhanced_data.columns)}")

            # Raw data ile merge et (country bilgisi için)
            try:
                raw_data = pd.read_csv("data/processed/online_retail_data.csv")
                # Customer ID bazında country bilgisini al
                customer_countries = raw_data.groupby('Customer ID')['Country'].first().reset_index()
                self.enhanced_data = self.enhanced_data.merge(customer_countries, on='Customer ID', how='left')
                print(f"✅ Country data merged successfully")
            except:
                print("⚠️ Raw data merge failed, using geographic segments instead")

            return True
        except FileNotFoundError:
            print("❌ Enhanced data not found. Run ml_auto_segmentation_engine.py first!")
            return False

    def geographic_value_analysis(self):
        """🌍 DETAYLI COĞRAFİ VALUE ANALİZİ"""
        print("\n🌍 GEOGRAPHIC VALUE ANALYSIS")
        print("=" * 60)

        # Ülke kolonu kontrolü
        if 'Country' in self.enhanced_data.columns:
            group_column = 'Country'
            print("📊 Using Country data from raw dataset")
        elif 'geographic_segment_name' in self.enhanced_data.columns:
            group_column = 'geographic_segment_name'
            print("📊 Using Geographic Segments (ML-discovered)")
        else:
            print("❌ No geographic data available")
            return None

        # Geographic bazlı aggregation
        geo_analysis = self.enhanced_data.groupby(group_column).agg({
            'Customer ID': 'count',
            'Monetary': ['sum', 'mean', 'median'],
            'AvgOrderValue': 'mean',
            'Frequency': 'mean',
            'CustomerValue': 'mean'
        }).round(2)

        geo_analysis.columns = ['Customer_Count', 'Total_Revenue', 'Avg_Customer_Value',
                                'Median_Customer_Value', 'Avg_Order_Value', 'Avg_Frequency', 'Customer_Score']

        # Revenue per customer hesapla
        geo_analysis['Revenue_Per_Customer'] = geo_analysis['Total_Revenue'] / geo_analysis['Customer_Count']

        # Market size classification
        geo_analysis['Market_Size'] = geo_analysis['Customer_Count'].apply(
            lambda x: 'LARGE' if x > 1000 else 'MEDIUM' if x > 100 else 'SMALL'
        )

        # Value tier classification
        geo_analysis['Value_Tier'] = geo_analysis['Revenue_Per_Customer'].apply(
            lambda x: 'PREMIUM' if x > 3000 else 'HIGH' if x > 1500 else 'STANDARD'
        )

        # Sort by total revenue
        geo_analysis = geo_analysis.sort_values('Total_Revenue', ascending=False)

        print(f"🏆 TOP {group_column.upper()} MARKETS BY VALUE:")
        for location, data in geo_analysis.head(10).iterrows():
            print(f"\n🌍 {location}:")
            print(f"   👥 Customers: {data['Customer_Count']:,}")
            print(f"   💰 Total Revenue: £{data['Total_Revenue']:,.2f}")
            print(f"   📈 Avg Customer Value: £{data['Avg_Customer_Value']:,.2f}")
            print(f"   🛒 Avg Order Value: £{data['Avg_Order_Value']:.2f}")
            print(f"   📊 Market Size: {data['Market_Size']}")
            print(f"   💎 Value Tier: {data['Value_Tier']}")

        return geo_analysis

    def segment_value_analysis(self):
        """📊 SEGMENT VALUE ANALİZİ"""
        print("\n📊 SEGMENT VALUE ANALYSIS")
        print("=" * 60)

        segment_analysis = self.enhanced_data.groupby('Segment').agg({
            'Customer ID': 'count',
            'Monetary': ['sum', 'mean', 'std'],
            'AvgOrderValue': 'mean',
            'Frequency': 'mean',
            'CustomerValue': 'mean',
            'Recency': 'mean'
        }).round(2)

        segment_analysis.columns = ['Customer_Count', 'Total_Revenue', 'Avg_Spend', 'Spend_Std',
                                    'Avg_Order_Value', 'Avg_Frequency', 'Avg_Customer_Score', 'Avg_Recency']

        # Calculate percentages
        total_customers = segment_analysis['Customer_Count'].sum()
        total_revenue = segment_analysis['Total_Revenue'].sum()

        segment_analysis['Customer_Percentage'] = (segment_analysis['Customer_Count'] / total_customers * 100).round(1)
        segment_analysis['Revenue_Percentage'] = (segment_analysis['Total_Revenue'] / total_revenue * 100).round(1)
        segment_analysis['Revenue_Per_Customer'] = segment_analysis['Total_Revenue'] / segment_analysis[
            'Customer_Count']

        # Sort by revenue percentage
        segment_analysis = segment_analysis.sort_values('Revenue_Percentage', ascending=False)

        print("🏆 SEGMENT VALUE RANKING:")
        for segment, data in segment_analysis.iterrows():
            print(f"\n🎯 {segment}:")
            print(f"   👥 Customers: {data['Customer_Count']:,} ({data['Customer_Percentage']}%)")
            print(f"   💰 Revenue: £{data['Total_Revenue']:,.2f} ({data['Revenue_Percentage']}%)")
            print(f"   📈 Revenue/Customer: £{data['Revenue_Per_Customer']:,.2f}")
            print(f"   🛒 Avg Order: £{data['Avg_Order_Value']:.2f}")
            print(f"   📊 Avg Frequency: {data['Avg_Frequency']:.1f}")
            print(f"   ⏰ Avg Recency: {data['Avg_Recency']:.0f} days")

        return segment_analysis

    def product_category_value_analysis(self):
        """🏷️ PRODUCT CATEGORY VALUE ANALİZİ"""
        print("\n🏷️ PRODUCT CATEGORY VALUE ANALYSIS")
        print("=" * 60)

        # Product category mapping
        category_mapping = {
            0: "Heart_Romance_Theme",
            1: "Storage_Organization",
            2: "Lighting_Candles",
            3: "General_Vintage_Design",
            4: "Retrospot_Collection",
            5: "Jewelry_Accessories",
            6: "Glass_Decoratives",
            7: "Home_Decoration",
            8: "Garden_Outdoor",
            9: "Christmas_Seasonal"
        }

        # Map categories
        self.enhanced_data['Category_Name'] = self.enhanced_data['product_category_name'].map(
            lambda x: x if pd.notna(x) else 'Unknown'
        )

        category_analysis = self.enhanced_data.groupby('Category_Name').agg({
            'Customer ID': 'count',
            'Monetary': ['sum', 'mean'],
            'AvgOrderValue': 'mean',
            'Frequency': 'mean'
        }).round(2)

        category_analysis.columns = ['Customer_Count', 'Total_Revenue', 'Avg_Customer_Value',
                                     'Avg_Order_Value', 'Avg_Frequency']

        # Calculate market share
        total_revenue = category_analysis['Total_Revenue'].sum()
        category_analysis['Market_Share'] = (category_analysis['Total_Revenue'] / total_revenue * 100).round(1)

        # Sort by market share
        category_analysis = category_analysis.sort_values('Market_Share', ascending=False)

        print("🏆 PRODUCT CATEGORY VALUE RANKING:")
        for category, data in category_analysis.head(10).iterrows():
            print(f"\n🏷️ {category}:")
            print(f"   👥 Customers: {data['Customer_Count']:,}")
            print(f"   💰 Revenue: £{data['Total_Revenue']:,.2f}")
            print(f"   📊 Market Share: {data['Market_Share']}%")
            print(f"   📈 Avg Customer Value: £{data['Avg_Customer_Value']:,.2f}")
            print(f"   🛒 Avg Order Value: £{data['Avg_Order_Value']:.2f}")

        return category_analysis

    def create_value_dashboard(self):
        """📊 VALUE ANALYTICS DASHBOARD"""
        print("\n📊 CREATING VALUE ANALYTICS DASHBOARD...")

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('💰 VALUE-BASED SEGMENTATION DASHBOARD', fontsize=16, fontweight='bold')

        # 1. Revenue by Segment
        segment_revenue = self.enhanced_data.groupby('Segment')['Monetary'].sum().sort_values(ascending=False)
        segment_revenue.plot(kind='bar', ax=ax1, color='skyblue')
        ax1.set_title('💰 Revenue by Segment')
        ax1.set_xlabel('Segment')
        ax1.set_ylabel('Total Revenue (£)')
        ax1.tick_params(axis='x', rotation=45)

        # 2. Customer Count by Geographic Area
        if 'Country' in self.enhanced_data.columns:
            geo_counts = self.enhanced_data['Country'].value_counts().head(10)
            ax2.pie(geo_counts.values, labels=geo_counts.index, autopct='%1.1f%%')
            ax2.set_title('🌍 Customer Distribution by Country')
        elif 'geographic_segment_name' in self.enhanced_data.columns:
            geo_counts = self.enhanced_data['geographic_segment_name'].value_counts()
            ax2.pie(geo_counts.values, labels=geo_counts.index, autopct='%1.1f%%')
            ax2.set_title('🌍 Customer Distribution by Geographic Segment')
        else:
            ax2.text(0.5, 0.5, 'No Geographic Data', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('🌍 Geographic Distribution (No Data)')

        # 3. Average Order Value by Product Category
        if 'product_category_name' in self.enhanced_data.columns:
            category_aov = self.enhanced_data.groupby('product_category_name')['AvgOrderValue'].mean().sort_values(
                ascending=False)
            category_aov.plot(kind='bar', ax=ax3, color='lightgreen')
            ax3.set_title('🏷️ Avg Order Value by Product Category')
            ax3.set_xlabel('Product Category')
            ax3.set_ylabel('Average Order Value (£)')
            ax3.tick_params(axis='x', rotation=45)
        else:
            ax3.text(0.5, 0.5, 'No Category Data', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('🏷️ Product Category Analysis (No Data)')

        # 4. Customer Value Distribution
        ax4.hist(self.enhanced_data['CustomerValue'], bins=30, color='orange', alpha=0.7)
        ax4.set_title('📊 Customer Value Distribution')
        ax4.set_xlabel('Customer Value Score')
        ax4.set_ylabel('Number of Customers')

        plt.tight_layout()
        plt.savefig('data/processed/value_analytics_dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("✅ Value analytics dashboard saved: value_analytics_dashboard.png")

    def generate_value_insights(self):
        """🧠 VALUE INSIGHTS GENERATION"""
        print("\n🧠 GENERATING VALUE INSIGHTS...")

        insights = []

        # Segment analysis
        segment_analysis = self.enhanced_data.groupby('Segment').agg({
            'Monetary': ['sum', 'count'],
            'CustomerValue': 'mean'
        })

        segment_analysis.columns = ['Total_Revenue', 'Customer_Count', 'Avg_Score']
        total_revenue = segment_analysis['Total_Revenue'].sum()

        # Top revenue segment
        top_segment = segment_analysis.sort_values('Total_Revenue', ascending=False).index[0]
        top_revenue_pct = (segment_analysis.loc[top_segment, 'Total_Revenue'] / total_revenue * 100)

        insights.append(f"🏆 {top_segment} segment generates {top_revenue_pct:.1f}% of total revenue")

        # Geographic insights
        if 'Country' in self.enhanced_data.columns:
            country_revenue = self.enhanced_data.groupby('Country')['Monetary'].sum()
            top_country = country_revenue.idxmax()
            country_pct = (country_revenue.max() / country_revenue.sum() * 100)
            insights.append(f"🌍 {top_country} represents {country_pct:.1f}% of total revenue")
        elif 'geographic_segment_name' in self.enhanced_data.columns:
            geo_revenue = self.enhanced_data.groupby('geographic_segment_name')['Monetary'].sum()
            top_geo = geo_revenue.idxmax()
            geo_pct = (geo_revenue.max() / geo_revenue.sum() * 100)
            insights.append(f"🌍 {top_geo} represents {geo_pct:.1f}% of total revenue")

        # Value concentration
        high_value_customers = len(self.enhanced_data[self.enhanced_data['CustomerValue'] > 200])
        high_value_pct = (high_value_customers / len(self.enhanced_data) * 100)
        insights.append(f"💎 {high_value_pct:.1f}% of customers are high-value (Score > 200)")

        print("💡 KEY VALUE INSIGHTS:")
        for i, insight in enumerate(insights, 1):
            print(f"   {i}. {insight}")

        return insights

    def run_complete_value_analysis(self):
        """🚀 COMPLETE VALUE ANALYSIS PIPELINE"""
        print("🚀 COMPLETE VALUE-BASED ANALYTICS PIPELINE")
        print("=" * 80)

        if not self.load_enhanced_data():
            return None

        # Run all analyses
        geo_analysis = self.geographic_value_analysis()
        segment_analysis = self.segment_value_analysis()
        category_analysis = self.product_category_value_analysis()

        # Generate insights
        insights = self.generate_value_insights()

        # Create dashboard
        self.create_value_dashboard()

        print(f"\n🎉 VALUE ANALYTICS COMPLETE!")
        print(f"📊 Generated comprehensive value-based segmentation analysis")

        return {
            'geographic_analysis': geo_analysis,
            'segment_analysis': segment_analysis,
            'category_analysis': category_analysis,
            'insights': insights
        }


def main():
    """Main value analytics execution"""
    print("💰 VALUE-BASED ANALYTICS ENGINE")
    print("=" * 50)

    analytics = ValueBasedAnalyticsEngine()
    results = analytics.run_complete_value_analysis()

    if results:
        print(f"\n🚀 Value analytics ready!")
        return analytics
    else:
        print(f"❌ Value analytics failed!")
        return None


if __name__ == "__main__":
    analytics = main()