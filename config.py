# src/utils/config.py

import os
import json
from datetime import datetime
from typing import Dict, List, Any


class SystemConfig:
    """
    Enhanced Müşteri Segmentasyon Sistemi Konfigürasyonu v2.1
    Dual Model Architecture + ML Auto-Discovery + Enhanced Features + Cross-Analysis Engine
    """

    # Enhanced Dosya yolları (Cross-Analysis dahil)
    PATHS = {
        'data_raw': 'data/raw/',
        'data_processed': 'data/processed/',
        'models': 'data/processed/',
        'enhanced_models': 'data/processed/',
        'exports': 'data/processed/exports/',
        'logs': 'logs/',
        'reports': 'reports/',
        'analytics': 'data/processed/analytics/',
        'dashboards': 'data/processed/dashboards/',
        'cross_analysis': 'data/processed/cross_analysis/',  # NEW
        'visualizations': 'data/processed/visualizations/'   # NEW
    }

    # Dual Model parametreleri (Original + Enhanced)
    MODEL_CONFIG = {
        # Original Neural Network (5 features, %93.6 accuracy)
        'original_neural_network': {
            'input_dim': 5,
            'hidden_layers': [64, 32, 16],
            'output_dim': 10,
            'dropout_rates': [0.3, 0.4, 0.3],
            'activation': 'relu',
            'output_activation': 'softmax',
            'optimizer': 'adam',
            'learning_rate': 0.001,
            'loss_function': 'sparse_categorical_crossentropy',
            'batch_size': 32,
            'epochs': 100,
            'early_stopping_patience': 15,
            'reduce_lr_patience': 8,
            'target_accuracy': 0.936,
            'model_file': 'customer_segmentation_model.h5',
            'scaler_file': 'scaler.pkl'
        },

        # Enhanced Neural Network (14 features, %95.9+ accuracy)
        'enhanced_neural_network': {
            'input_dim': 14,
            'hidden_layers': [128, 64, 32, 16],
            'output_dim': 10,
            'dropout_rates': [0.3, 0.4, 0.3, 0.2],
            'activation': 'relu',
            'output_activation': 'softmax',
            'optimizer': 'adam',
            'learning_rate': 0.001,
            'loss_function': 'sparse_categorical_crossentropy',
            'batch_size': 32,
            'epochs': 100,
            'early_stopping_patience': 20,
            'reduce_lr_patience': 10,
            'target_accuracy': 0.959,
            'model_file': 'enhanced_customer_segmentation_model.h5',
            'scaler_file': 'enhanced_scaler.pkl'
        },

        # ML Auto-Discovery Engine
        'ml_discovery_engine': {
            'product_categories': 10,
            'geographic_clusters': 'auto_detect',
            'behavior_patterns': 'auto_detect',
            'min_country_size': 30,
            'tfidf_max_features': 1000,
            'tfidf_ngram_range': (1, 2),
            'kmeans_init': 10,
            'random_state': 42
        },

        # NEW: Cross-Analysis Engine Configuration
        'cross_analysis_engine': {
            'enable_cross_analysis': True,
            'country_product_analysis': True,
            'segment_product_analysis': True,
            'geographic_segment_analysis': True,
            'cross_selling_analysis': True,
            'min_transaction_threshold': 10,
            'min_customer_threshold': 5,
            'top_countries_limit': 10,
            'top_categories_limit': 15,
            'visualization_enabled': True,
            'report_generation': True,
            'output_formats': ['png', 'html', 'json', 'csv']
        },

        # Preprocessing Enhanced
        'preprocessing': {
            'test_size': 0.2,
            'random_state': 42,
            'stratify': True,
            'scaler_type': 'StandardScaler',
            'handle_missing': 'median',
            'outlier_threshold': 0.95
        }
    }

    # Enhanced Feature Columns (Cross-Analysis aware)
    FEATURE_CONFIG = {
        # Original RFM Features (5)
        'original_features': [
            'Recency', 'Frequency', 'Monetary', 'AvgOrderValue', 'CustomerValue'
        ],

        # ML-Discovered Enhanced Features (9 additional = 14 total)
        'enhanced_features': [
            'auto_product_category_id', 'auto_geographic_segment_id', 'behavior_pattern_id',
            'purchase_span_days', 'order_value_consistency', 'price_tier_preference',
            'bulk_buying_tendency', 'segment_pattern_alignment', 'geographic_behavior_score'
        ],

        # All Enhanced Features (14 total)
        'all_enhanced_features': [
            # Original (5)
            'Recency', 'Frequency', 'Monetary', 'AvgOrderValue', 'CustomerValue',
            # Enhanced (9)
            'auto_product_category_id', 'auto_geographic_segment_id', 'behavior_pattern_id',
            'purchase_span_days', 'order_value_consistency', 'price_tier_preference',
            'bulk_buying_tendency', 'segment_pattern_alignment', 'geographic_behavior_score'
        ],

        # NEW: Cross-Analysis Required Fields
        'cross_analysis_fields': {
            'required_raw_columns': ['Customer ID', 'Country', 'Description', 'Quantity', 'Price', 'Invoice'],
            'required_rfm_columns': ['Customer ID', 'Segment', 'Recency', 'Frequency', 'Monetary'],
            'product_category_keywords': {
                'HEART_ROMANCE': ['HEART', 'LOVE', 'VALENTINE', 'ROMANTIC'],
                'CHRISTMAS': ['CHRISTMAS', 'XMAS', 'SANTA', 'REINDEER', 'ADVENT'],
                'HOME_DECOR': ['HOME', 'DECORATION', 'WALL', 'ORNAMENT'],
                'KITCHEN': ['MUG', 'CUP', 'TEA', 'BOWL', 'PLATE'],
                'LIGHTING': ['LIGHT', 'CANDLE', 'HOLDER', 'LAMP']
            }
        },

        # Feature Categories
        'feature_categories': {
            'rfm_core': ['Recency', 'Frequency', 'Monetary'],
            'rfm_derived': ['AvgOrderValue', 'CustomerValue'],
            'ml_product': ['auto_product_category_id'],
            'ml_geographic': ['auto_geographic_segment_id', 'geographic_behavior_score'],
            'ml_behavior': ['behavior_pattern_id', 'segment_pattern_alignment'],
            'ml_temporal': ['purchase_span_days'],
            'ml_consistency': ['order_value_consistency'],
            'ml_preference': ['price_tier_preference', 'bulk_buying_tendency'],
            'cross_analysis': ['Country', 'Description', 'ProductCategory']  # NEW
        }
    }

    # Enhanced RFM ve Segmentasyon parametreleri
    RFM_CONFIG = {
        'scoring': {
            'recency_bins': 5,
            'frequency_bins': 5,
            'monetary_bins': 5,
            'score_range': [1, 5]
        },
        'business_rules': {
            'min_transactions': 1,
            'min_monetary_value': 0,
            'max_recency_days': 730,
            'outlier_threshold': 0.95
        },
        'enhanced_validation': {
            'min_customers_per_segment': 10,
            'max_segments': 15,
            'confidence_threshold': 0.7
        }
    }

    # Enhanced Pazarlama kampanya parametreleri
    CAMPAIGN_CONFIG = {
        # Enhanced Budget Limits (AI-optimized + Cross-Analysis aware)
        'enhanced_budget_limits': {
            'Champions': 1200,  # Increased for premium targeting
            'Loyal': 900,  # Enhanced loyalty programs
            'At Risk': 700,  # Urgent intervention budget
            'Potential Loyalists': 500,  # Development investment
            'New Customers': 400,  # Enhanced onboarding
            'Promising': 350,  # Growth potential
            'Need Attention': 300,  # Attention campaigns
            'About to Sleep': 200,  # Reactivation efforts
            'Hibernating': 100,  # Minimal engagement
            'Lost': 50  # Cost-optimized recovery
        },

        # Enhanced Channel Costs (AI-optimized)
        'enhanced_channel_costs': {
            'Email': 0.10,
            'SMS': 0.15,
            'Phone': 15.00,
            'In-App': 0.05,
            'Social Media': 2.50,
            'Tutorial': 5.00,
            'Direct Mail': 3.00,
            'AI-Optimized': 1.25,  # New AI channel
            'Personal Manager': 25.00,  # Premium service
            'Automated': 0.03,  # Automated campaigns
            'Cross-Sell-Optimized': 0.75  # NEW: Cross-selling channel
        },

        # Enhanced Campaign Rules (Cross-Analysis enhanced)
        'enhanced_rules': {
            'default_campaign_duration': 14,
            'max_campaigns_per_customer_per_month': 4,  # Increased
            'min_days_between_campaigns': 5,  # Reduced for AI optimization
            'max_discount_rate': 0.45,  # Increased for premium
            'ai_confidence_threshold': 0.8,  # AI decision threshold
            'dual_model_agreement_threshold': 0.9,  # Model agreement threshold
            'cross_analysis_weight': 0.3,  # NEW: Cross-analysis influence on decisions
            'geographic_personalization': True,  # NEW: Geographic preferences
            'product_affinity_targeting': True   # NEW: Product preference targeting
        }
    }

    # Enhanced A/B Test parametreleri (Cross-Analysis enhanced)
    AB_TEST_CONFIG = {
        'statistical': {
            'significance_threshold': 0.05,
            'min_sample_size': 80,  # Reduced for faster testing
            'power': 0.8,
            'effect_size': 0.2,
            'confidence_level': 0.95
        },
        'enhanced_testing': {
            'dual_model_testing': True,  # Test both models
            'feature_importance_testing': True,  # Test feature subsets
            'ai_insights_validation': True,  # Validate AI recommendations
            'geographic_stratification': True,  # Geographic-aware testing
            'cross_analysis_validation': True,  # NEW: Cross-analysis insights testing
            'product_preference_testing': True,  # NEW: Product affinity testing
            'segment_cross_validation': True     # NEW: Cross-segment validation
        },
        'test_duration': {
            'min_days': 5,  # Faster iteration
            'max_days': 30,
            'default_days': 14,
            'enhanced_monitoring_days': 7,  # Enhanced model monitoring
            'cross_analysis_validation_days': 3  # NEW: Cross-analysis validation period
        },
        'group_allocation': {
            'control_percentage': 0.3,  # Reduced for more variants
            'variant_percentage': 0.7,  # More space for testing
            'enhanced_model_percentage': 0.4,  # Enhanced model allocation
            'original_model_percentage': 0.3,  # Original model allocation
            'cross_analysis_enhanced_percentage': 0.3  # NEW: Cross-analysis enhanced allocation
        }
    }

    # Enhanced Performans metrikleri (Cross-Analysis dahil)
    METRICS_CONFIG = {
        # Model Performance (Dual Model + Cross-Analysis)
        'model_performance': [
            'accuracy', 'precision', 'recall', 'f1_score',
            'confusion_matrix', 'classification_report',
            'confidence_distribution', 'feature_importance',
            'cross_analysis_accuracy'  # NEW
        ],

        # Enhanced Business Metrics (Cross-Analysis enhanced)
        'enhanced_business_metrics': [
            'conversion_rate', 'roi', 'clv', 'churn_rate',
            'revenue_per_customer', 'average_order_value',
            'segment_migration_rate', 'ai_recommendation_accuracy',
            'model_agreement_rate', 'confidence_improvement',
            'cross_selling_success_rate',  # NEW
            'geographic_penetration_rate',  # NEW
            'product_affinity_accuracy',    # NEW
            'cross_analysis_coverage'       # NEW
        ],

        # Enhanced Campaign Metrics (Cross-Analysis aware)
        'enhanced_campaign_metrics': [
            'open_rate', 'click_rate', 'conversion_rate',
            'cost_per_acquisition', 'return_on_ad_spend',
            'ai_optimization_lift', 'channel_performance_index',
            'personalization_effectiveness', 'automation_efficiency',
            'cross_sell_conversion_rate',  # NEW
            'geographic_campaign_effectiveness',  # NEW
            'product_preference_match_rate'      # NEW
        ],

        # Enhanced Alert Thresholds (Cross-Analysis monitoring)
        'enhanced_alert_thresholds': {
            'accuracy_drop': 0.03,  # Tighter monitoring
            'roi_drop': 0.08,  # More sensitive
            'conversion_drop': 0.12,  # Early warning
            'confidence_drop': 0.05,  # Confidence monitoring
            'model_disagreement': 0.15,  # Model alignment check
            'ai_insight_accuracy': 0.85,  # AI recommendation quality
            'cross_analysis_coverage_drop': 0.10,  # NEW: Cross-analysis coverage monitoring
            'geographic_performance_variance': 0.20,  # NEW: Geographic performance monitoring
            'product_affinity_drift': 0.15  # NEW: Product preference drift monitoring
        }
    }

    # NEW: Cross-Analysis Configuration
    CROSS_ANALYSIS_CONFIG = {
        # Analysis Types
        'analysis_types': {
            'country_product_matrix': {
                'enabled': True,
                'min_countries': 5,
                'min_products': 5,
                'visualization': 'heatmap',
                'export_format': ['png', 'csv', 'json']
            },
            'segment_product_matrix': {
                'enabled': True,
                'min_segments': 3,
                'min_products': 5,
                'visualization': 'heatmap',
                'export_format': ['png', 'csv', 'json']
            },
            'geographic_segment_intersection': {
                'enabled': True,
                'min_countries': 5,
                'min_segments': 3,
                'visualization': 'matrix',
                'export_format': ['png', 'csv', 'json']
            },
            'cross_selling_analysis': {
                'enabled': True,
                'min_basket_size': 2,
                'min_support': 0.01,
                'visualization': 'network',
                'export_format': ['png', 'json']
            }
        },

        # Priority Analysis (çözülecek kritik sorular)
        'priority_questions': {
            'eire_mystery': {
                'question': 'EIRE £71K/customer value hangi ürünlerden geliyor?',
                'analysis_type': 'country_product_matrix',
                'focus_country': 'EIRE',
                'priority': 'HIGH'
            },
            'champions_preferences': {
                'question': 'Champions £5.7M revenue hangi kategorilerde?',
                'analysis_type': 'segment_product_matrix',
                'focus_segment': 'Champions',
                'priority': 'HIGH'
            },
            'geographic_penetration': {
                'question': 'Hangi ülke hangi ürünü tercih ediyor?',
                'analysis_type': 'country_product_matrix',
                'priority': 'MEDIUM'
            },
            'cross_selling_opportunities': {
                'question': 'Cross-selling opportunities nerede?',
                'analysis_type': 'cross_selling_analysis',
                'priority': 'MEDIUM'
            }
        },

        # Visualization Settings
        'visualization': {
            'heatmap_colormap': 'YlOrRd',
            'figsize': (15, 8),
            'dpi': 300,
            'save_format': 'png',
            'show_annotations': True,
            'font_size': 12
        },

        # Report Generation
        'reporting': {
            'executive_summary': True,
            'detailed_analysis': True,
            'business_recommendations': True,
            'export_formats': ['json', 'html', 'pdf'],
            'auto_insights': True,
            'trend_analysis': True
        }
    }

    # Enhanced Segment tanımları ve hedefleri (Cross-Analysis aware)
    ENHANCED_SEGMENT_TARGETS = {
        'Champions': {
            'target_conversion_rate': 0.40,  # Increased with AI
            'target_roi': 350,  # Enhanced ROI
            'priority': 'MAXIMUM',
            'retention_target': 0.98,
            'ai_personalization': 'MAXIMUM',
            'enhanced_clv_multiplier': 4.2,
            'ai_features_weight': 0.7,
            'cross_analysis_priority': 'HIGH',  # NEW
            'product_affinity_targeting': True,  # NEW
            'geographic_personalization': True   # NEW
        },
        'Loyal': {
            'target_conversion_rate': 0.30,
            'target_roi': 280,
            'priority': 'HIGH',
            'retention_target': 0.93,
            'ai_personalization': 'HIGH',
            'enhanced_clv_multiplier': 3.1,
            'ai_features_weight': 0.6,
            'cross_analysis_priority': 'HIGH',  # NEW
            'product_affinity_targeting': True,  # NEW
            'geographic_personalization': True   # NEW
        },
        'At Risk': {
            'target_conversion_rate': 0.18,
            'target_roi': 180,
            'priority': 'URGENT',
            'win_back_target': 0.35,
            'ai_personalization': 'MAXIMUM',
            'enhanced_clv_multiplier': 2.3,
            'ai_features_weight': 0.8,
            'cross_analysis_priority': 'MEDIUM',  # NEW
            'product_affinity_targeting': True,   # NEW
            'geographic_personalization': False    # NEW
        },
        # ... (diğer segmentler benzer şekilde güncellendi)
    }

    # Enhanced System Information (Cross-Analysis dahil)
    SYSTEM_INFO = {
        'version': '2.1.0',
        'system_name': 'Enhanced Customer Segmentation AI with Cross-Analysis',
        'architecture': 'Dual Model + ML Auto-Discovery + Cross-Analysis Engine',
        'original_accuracy': 93.6,
        'enhanced_accuracy': 95.9,
        'feature_enhancement': '5 → 14 features',
        'cross_analysis_coverage': '100%',  # NEW
        'ai_capabilities': [
            'Auto-Discovery Engine',
            'Enhanced Neural Network',
            'AI-Powered Insights',
            'Dual Model Comparison',
            'Geographic Intelligence',
            'Behavior Pattern Recognition',
            'Cross-Dimensional Analysis',  # NEW
            'Product Affinity Mapping',    # NEW
            'Geographic Penetration Analysis',  # NEW
            'Cross-selling Optimization'   # NEW
        ]
    }

    # Enhanced Logging konfigürasyonu (Cross-Analysis dahil)
    LOGGING_CONFIG = {
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'file_rotation': 'daily',
        'max_log_files': 30,
        'enhanced_logging': {
            'model_performance_tracking': True,
            'ai_insight_logging': True,
            'dual_model_comparison_logging': True,
            'feature_importance_tracking': True,
            'cross_analysis_logging': True,  # NEW
            'geographic_analysis_logging': True,  # NEW
            'product_affinity_logging': True    # NEW
        }
    }

    @classmethod
    def get_cross_analysis_config(cls) -> Dict:
        """Cross-Analysis Engine konfigürasyonunu al"""
        return cls.CROSS_ANALYSIS_CONFIG

    @classmethod
    def get_enhanced_model_config(cls) -> Dict:
        """Enhanced model konfigürasyonunu al (Cross-Analysis aware)"""
        return cls.MODEL_CONFIG

    @classmethod
    def get_dual_model_config(cls) -> Dict:
        """Dual model (Original + Enhanced) konfigürasyonunu al"""
        return {
            'original': cls.MODEL_CONFIG['original_neural_network'],
            'enhanced': cls.MODEL_CONFIG['enhanced_neural_network'],
            'cross_analysis': cls.MODEL_CONFIG['cross_analysis_engine'],  # NEW
            'comparison_enabled': True,
            'auto_selection': 'enhanced_preferred'
        }

    @classmethod
    def get_enhanced_campaign_config(cls) -> Dict:
        """Enhanced kampanya konfigürasyonunu al (Cross-Analysis enhanced)"""
        return cls.CAMPAIGN_CONFIG

    @classmethod
    def get_enhanced_segment_targets(cls) -> Dict:
        """Enhanced segment hedeflerini al (Cross-Analysis aware)"""
        return cls.ENHANCED_SEGMENT_TARGETS

    @classmethod
    def get_enhanced_ab_test_config(cls) -> Dict:
        """Enhanced A/B test konfigürasyonunu al (Cross-Analysis enhanced)"""
        return cls.AB_TEST_CONFIG

    @classmethod
    def get_feature_config(cls) -> Dict:
        """Feature konfigürasyonunu al (Cross-Analysis fields dahil)"""
        return cls.FEATURE_CONFIG

    @classmethod
    def create_enhanced_directories(cls):
        """Enhanced gerekli klasörleri oluştur (Cross-Analysis dahil)"""
        for path in cls.PATHS.values():
            os.makedirs(path, exist_ok=True)
        print("✅ Enhanced sistem klasörleri oluşturuldu (Cross-Analysis dahil)")

    @classmethod
    def validate_system(cls) -> bool:
        """Backward compatibility - calls validate_enhanced_system"""
        return cls.validate_enhanced_system()

    @classmethod
    def validate_enhanced_system(cls) -> bool:
        """Enhanced sistem bütünlüğünü kontrol et (Cross-Analysis dahil)"""
        print("🔍 Enhanced sistem validasyonu yapılıyor (Cross-Analysis dahil)...")

        # Enhanced gerekli dosyaların kontrolü
        required_files = [
            # Original system files
            'data/processed/rfm_analysis_results.csv',
            'data/processed/X_features.npy',
            'data/processed/y_labels.npy',
            'data/processed/segment_mapping.json',
            'data/processed/customer_segmentation_model.h5',
            'data/processed/scaler.pkl',

            # Enhanced system files
            'data/processed/ml_enhanced_rfm_dataset.csv',
            'data/processed/X_ml_enhanced_features.npy',
            'data/processed/y_ml_enhanced_labels.npy',
            'data/processed/ml_enhanced_feature_info.json'
        ]

        # Optional enhanced files (may not exist initially)
        optional_enhanced_files = [
            'data/processed/enhanced_customer_segmentation_model.h5',
            'data/processed/enhanced_scaler.pkl',
            'data/processed/enhanced_segment_mapping.json'
        ]

        # NEW: Cross-Analysis specific files (optional)
        optional_cross_analysis_files = [
            'data/processed/cross_analysis/country_product_matrix.png',
            'data/processed/cross_analysis/segment_product_matrix.png',
            'data/processed/cross_analysis/cross_analysis_report.json'
        ]

        missing_files = []
        for file_path in required_files:
            if not os.path.exists(file_path):
                missing_files.append(file_path)

        missing_optional = []
        for file_path in optional_enhanced_files:
            if not os.path.exists(file_path):
                missing_optional.append(file_path)

        missing_cross_analysis = []
        for file_path in optional_cross_analysis_files:
            if not os.path.exists(file_path):
                missing_cross_analysis.append(file_path)

        if missing_files:
            print("❌ Eksik kritik dosyalar:")
            for file in missing_files:
                print(f"  - {file}")
            return False

        if missing_optional:
            print("⚠️ Eksik opsiyonel enhanced dosyalar (otomatik oluşturulacak):")
            for file in missing_optional:
                print(f"  - {file}")

        if missing_cross_analysis:
            print("ℹ️ Cross-Analysis dosyaları (çalıştırıldığında oluşturulacak):")
            for file in missing_cross_analysis:
                print(f"  - {file}")

        print("✅ Enhanced sistem dosyaları kontrol edildi (Cross-Analysis ready)")
        return True

    @classmethod
    def get_enhanced_system_info(cls) -> Dict:
        """Enhanced sistem bilgilerini al (Cross-Analysis dahil)"""
        return {
            **cls.SYSTEM_INFO,
            'created_date': datetime.now().strftime('%Y-%m-%d'),
            'components': [
                'Enhanced Data Pipeline',
                'Dual Neural Network Models',
                'ML Auto-Discovery Engine',
                'Enhanced Segmentation Engine',
                'Enhanced Campaign Decision Engine',
                'Enhanced A/B Test Framework',
                'Value-Based Analytics Engine',
                'Cross-Analysis Engine',  # NEW
                'Enhanced Analytics Dashboard'
            ],
            'status': 'CROSS_ANALYSIS_ENHANCED_PRODUCTION_READY',
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'performance_improvement': {
                'accuracy_improvement': '+2.3%',
                'feature_enhancement': '180%',
                'confidence_improvement': '+13.7%',
                'roi_potential': '+25-40%',
                'cross_analysis_coverage': '100%',  # NEW
                'missing_analysis_fixed': True      # NEW
            }
        }

    @classmethod
    def compare_model_configs(cls) -> Dict:
        """Original vs Enhanced model karşılaştırması (Cross-Analysis dahil)"""
        original = cls.MODEL_CONFIG['original_neural_network']
        enhanced = cls.MODEL_CONFIG['enhanced_neural_network']
        cross_analysis = cls.MODEL_CONFIG['cross_analysis_engine']

        return {
            'feature_comparison': {
                'original_features': original['input_dim'],
                'enhanced_features': enhanced['input_dim'],
                'improvement_ratio': f"{enhanced['input_dim'] / original['input_dim']:.1f}x"
            },
            'accuracy_comparison': {
                'original_accuracy': original['target_accuracy'],
                'enhanced_accuracy': enhanced['target_accuracy'],
                'improvement': f"+{(enhanced['target_accuracy'] - original['target_accuracy']) * 100:.1f}%"
            },
            'architecture_comparison': {
                'original_layers': len(original['hidden_layers']),
                'enhanced_layers': len(enhanced['hidden_layers']),
                'complexity_increase': f"+{len(enhanced['hidden_layers']) - len(original['hidden_layers'])} layers"
            },
            'cross_analysis_capabilities': {
                'missing_analysis_fixed': cross_analysis['enable_cross_analysis'],
                'country_product_matrix': cross_analysis['country_product_analysis'],
                'segment_product_matrix': cross_analysis['segment_product_analysis'],
                'geographic_insights': cross_analysis['geographic_segment_analysis'],
                'cross_selling_analysis': cross_analysis['cross_selling_analysis']
            }
        }

    @classmethod
    def get_priority_analysis_questions(cls) -> Dict:
        """Cross-Analysis ile çözülecek priority questions"""
        return cls.CROSS_ANALYSIS_CONFIG['priority_questions']


def main():
    """
    Enhanced konfigürasyon testi ve sistem kontrolü (Cross-Analysis dahil)
    """
    print("⚙️ ENHANCED SİSTEM KONFİGÜRASYON KONTROLÜ v2.1")
    print("🔍 Cross-Analysis Engine dahil edildi!")
    print("=" * 60)

    # Enhanced klasörleri oluştur (Cross-Analysis dahil)
    SystemConfig.create_enhanced_directories()

    # Enhanced sistem validasyonu
    is_valid = SystemConfig.validate_enhanced_system()

    if is_valid:
        print("✅ Enhanced sistem konfigürasyonu geçerli (Cross-Analysis ready)!")

        # Enhanced sistem bilgileri
        info = SystemConfig.get_enhanced_system_info()
        print(f"\n📊 ENHANCED SİSTEM BİLGİLERİ:")
        print(f"  Version: {info['version']}")
        print(f"  Architecture: {info['architecture']}")
        print(f"  Components: {len(info['components'])} enhanced components")
        print(f"  Status: {info['status']}")
        print(f"  Cross-Analysis Coverage: {info['performance_improvement']['cross_analysis_coverage']}")

        # Model karşılaştırması (Cross-Analysis dahil)
        comparison = SystemConfig.compare_model_configs()
        print(f"\n🔍 MODEL KARŞILAŞTIRMASI:")
        print(f"  Features: {comparison['feature_comparison']['improvement_ratio']} improvement")
        print(f"  Accuracy: {comparison['accuracy_comparison']['improvement']}")
        print(f"  Architecture: {comparison['architecture_comparison']['complexity_increase']}")
        print(f"  Cross-Analysis: {'✅ Enabled' if comparison['cross_analysis_capabilities']['missing_analysis_fixed'] else '❌ Disabled'}")

        # Priority Questions
        priority_questions = SystemConfig.get_priority_analysis_questions()
        print(f"\n🎯 PRİORİTY ANALYSİS QUESTIONS (Cross-Analysis ile çözülecek):")
        for key, question_info in priority_questions.items():
            print(f"  {question_info['priority']}: {question_info['question']}")

    else:
        print("❌ Enhanced sistem eksik dosyalar var!")

    print(f"\n🚀 Enhanced konfigürasyon hazır (Cross-Analysis enabled)!")


if __name__ == "__main__":
    main()