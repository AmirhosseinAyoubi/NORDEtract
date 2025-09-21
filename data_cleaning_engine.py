import pandas as pd
import numpy as np
import json
import os
from typing import Dict, List, Tuple, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
import warnings
warnings.filterwarnings('ignore')
class DataCleaningEngine:
    def __init__(self):
        self.transformations_applied = []
        self.cleaning_stats = {}
        self.original_shape = None
        self.cleaned_shape = None
    def load_quality_report(self, dataset_id: str) -> Dict[str, Any]:
        try:
            reports_dir = "reports"
            report_files = [f for f in os.listdir(reports_dir) if f.startswith(f"{dataset_id}_") and f.endswith("summary_report.json")]
            if not report_files:
                raise FileNotFoundError(f"No quality report found for dataset {dataset_id}")
            report_path = os.path.join(reports_dir, report_files[0])
            with open(report_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Error loading quality report: {e}")
            return {}
    def analyze_suggestions(self, quality_report: Dict[str, Any]) -> Dict[str, List[Dict]]:
        suggestions = {
            'missing_values': [],
            'outliers': [],
            'duplicates': [],
            'constant_columns': [],
            'high_cardinality': [],
            'data_types': [],
            'normalization': []
        }
        try:
            column_reports = quality_report.get('body', {}).get('column_reports', {})
            general_stats = quality_report.get('body', {}).get('general_stats', {})
            print(f"🔍 Analyzing suggestions from {len(column_reports)} columns...")
            constant_cols = general_stats.get('constant_columns', [])
            for col_name in constant_cols:
                suggestions['constant_columns'].append({
                    'column': col_name,
                    'action': 'Drop this column',
                    'severity': 'major'
                })
                print(f"  🗑️ Constant column (from general_stats): {col_name}")
            for col_name, col_report in column_reports.items():
                col_suggestions = col_report.get('suggestions', [])
                description = col_report.get('description', '')
                missing_pct = 0
                if '% missing' in description:
                    try:
                        missing_pct = float(description.split('% missing')[0].split()[-1])
                    except:
                        missing_pct = 0
                issues = col_report.get('issues', [])
                if missing_pct > 0 or any('missing' in issue.lower() for issue in issues):
                    for suggestion in col_suggestions:
                        action = suggestion.get('action', '').lower()
                        if 'impute' in action or 'missing' in action:
                            suggestions['missing_values'].append({
                                'column': col_name,
                                'action': action,
                                'severity': suggestion.get('severity', 'moderate'),
                                'missing_pct': missing_pct
                            })
                            print(f"  📝 Missing values: {col_name} ({missing_pct}%)")
                if any('outlier' in issue.lower() for issue in issues):
                    for suggestion in col_suggestions:
                        action = suggestion.get('action', '').lower()
                        if 'outlier' in action or 'cap' in action or 'robust' in action:
                            suggestions['outliers'].append({
                                'column': col_name,
                                'action': action,
                                'severity': suggestion.get('severity', 'moderate')
                            })
                            print(f"  📊 Outliers: {col_name}")
                if any('constant' in issue.lower() for issue in issues):
                    for suggestion in col_suggestions:
                        action = suggestion.get('action', '').lower()
                        if 'drop' in action and 'column' in action:
                            suggestions['constant_columns'].append({
                                'column': col_name,
                                'action': action,
                                'severity': suggestion.get('severity', 'major')
                            })
                            print(f"  🗑️ Constant column: {col_name}")
                if any('high-cardinality' in issue.lower() or 'id-like' in issue.lower() for issue in issues):
                    for suggestion in col_suggestions:
                        action = suggestion.get('action', '').lower()
                        if 'drop' in action and ('high-cardinality' in action or 'id-like' in action):
                            suggestions['high_cardinality'].append({
                                'column': col_name,
                                'action': action,
                                'severity': suggestion.get('severity', 'major')
                            })
                            print(f"  🔢 High-cardinality: {col_name}")
            duplicate_pct = general_stats.get('duplicate_rows_pct', 0)
            if duplicate_pct > 0:
                suggestions['duplicates'].append({
                    'action': 'Remove duplicate rows',
                    'duplicate_pct': duplicate_pct,
                    'severity': 'moderate' if duplicate_pct < 10 else 'major'
                })
                print(f"  🔄 Duplicates: {duplicate_pct}%")
            for col_name, col_report in column_reports.items():
                issues = col_report.get('issues', [])
                if any('outlier' in issue.lower() for issue in issues):
                    suggestions['normalization'].append({
                        'column': col_name,
                        'action': 'Apply robust scaling',
                        'severity': 'moderate'
                    })
            target_cols = ['TARGET', 'target', 'label', 'y']
            constant_cols = general_stats.get('constant_columns', []) if general_stats else []
            suggestions['normalization'] = [
                s for s in suggestions['normalization'] 
                if s['column'] not in target_cols and s['column'] not in constant_cols
            ]
            print(f"✅ Found {sum(len(v) for v in suggestions.values())} total suggestions")
            return suggestions
        except Exception as e:
            print(f"❌ Error analyzing suggestions: {e}")
            return suggestions
    def apply_missing_value_imputation(self, df: pd.DataFrame, suggestions: List[Dict]) -> pd.DataFrame:
        df_cleaned = df.copy()
        for suggestion in suggestions:
            col = suggestion['column']
            if col not in df_cleaned.columns:
                continue
            missing_pct = float(suggestion.get('missing_pct', 0))
            severity = suggestion['severity']
            if missing_pct > 50 and severity == 'major':
                df_cleaned = df_cleaned.drop(columns=[col])
                self.transformations_applied.append(f"Dropped column '{col}' (missing: {missing_pct}%)")
            elif missing_pct > 0:
                if df_cleaned[col].dtype in ['int64', 'float64']:
                    imputer = SimpleImputer(strategy='median')
                    df_cleaned[col] = imputer.fit_transform(df_cleaned[[col]]).flatten()
                    self.transformations_applied.append(f"Imputed missing values in '{col}' using median")
                else:
                    mode_value = df_cleaned[col].mode()
                    if len(mode_value) > 0:
                        df_cleaned[col] = df_cleaned[col].fillna(mode_value[0])
                        self.transformations_applied.append(f"Imputed missing values in '{col}' using mode")
                    else:
                        df_cleaned[col] = df_cleaned[col].fillna('Unknown')
                        self.transformations_applied.append(f"Imputed missing values in '{col}' with 'Unknown'")
        return df_cleaned
    def apply_outlier_treatment(self, df: pd.DataFrame, suggestions: List[Dict]) -> pd.DataFrame:
        df_cleaned = df.copy()
        for suggestion in suggestions:
            col = suggestion['column']
            if col not in df_cleaned.columns or df_cleaned[col].dtype not in ['int64', 'float64']:
                continue
            Q1 = df_cleaned[col].quantile(0.25)
            Q3 = df_cleaned[col].quantile(0.75)
            IQR = Q3 - Q1
            if IQR > 0:
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers_before = ((df_cleaned[col] < lower_bound) | (df_cleaned[col] > upper_bound)).sum()
                df_cleaned[col] = df_cleaned[col].clip(lower=lower_bound, upper=upper_bound)
                outliers_after = ((df_cleaned[col] < lower_bound) | (df_cleaned[col] > upper_bound)).sum()
                if outliers_before > 0:
                    self.transformations_applied.append(f"Capped {outliers_before} outliers in '{col}' using IQR method")
        return df_cleaned
    def apply_duplicate_removal(self, df: pd.DataFrame, suggestions: List[Dict]) -> pd.DataFrame:
        df_cleaned = df.copy()
        for suggestion in suggestions:
            if 'duplicate' in suggestion['action'].lower():
                duplicates_before = df_cleaned.duplicated().sum()
                df_cleaned = df_cleaned.drop_duplicates()
                duplicates_after = df_cleaned.duplicated().sum()
                if duplicates_before > 0:
                    self.transformations_applied.append(f"Removed {duplicates_before} duplicate rows")
        return df_cleaned
    def apply_column_cleaning(self, df: pd.DataFrame, suggestions: Dict[str, List[Dict]]) -> pd.DataFrame:
        df_cleaned = df.copy()
        for suggestion in suggestions['constant_columns']:
            col = suggestion['column']
            if col in df_cleaned.columns:
                df_cleaned = df_cleaned.drop(columns=[col])
                self.transformations_applied.append(f"Removed constant column '{col}'")
        for suggestion in suggestions['high_cardinality']:
            col = suggestion['column']
            if col in df_cleaned.columns:
                df_cleaned = df_cleaned.drop(columns=[col])
                self.transformations_applied.append(f"Removed high-cardinality column '{col}'")
        return df_cleaned
    def apply_normalization(self, df: pd.DataFrame, suggestions: List[Dict] = None) -> pd.DataFrame:
        df_cleaned = df.copy()
        if suggestions and len(suggestions) > 0:
            cols_to_normalize = [s['column'] for s in suggestions if s['column'] in df_cleaned.columns]
            if cols_to_normalize:
                scaler = RobustScaler()
                df_cleaned[cols_to_normalize] = scaler.fit_transform(df_cleaned[cols_to_normalize])
                self.transformations_applied.append(f"Applied RobustScaler normalization to {len(cols_to_normalize)} columns: {cols_to_normalize[:3]}{'...' if len(cols_to_normalize) > 3 else ''}")
        else:
            print("  ⏭️ No normalization suggestions, skipping normalization")
        return df_cleaned
    def apply_data_type_optimization(self, df: pd.DataFrame) -> pd.DataFrame:
        df_cleaned = df.copy()
        for col in df_cleaned.columns:
            if df_cleaned[col].dtype == 'object':
                try:
                    df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='ignore')
                except:
                    pass
            if df_cleaned[col].dtype in ['int64', 'int32']:
                if df_cleaned[col].min() >= 0:
                    if df_cleaned[col].max() < 255:
                        df_cleaned[col] = df_cleaned[col].astype('uint8')
                    elif df_cleaned[col].max() < 65535:
                        df_cleaned[col] = df_cleaned[col].astype('uint16')
                    elif df_cleaned[col].max() < 4294967295:
                        df_cleaned[col] = df_cleaned[col].astype('uint32')
                else:
                    if df_cleaned[col].min() > -128 and df_cleaned[col].max() < 127:
                        df_cleaned[col] = df_cleaned[col].astype('int8')
                    elif df_cleaned[col].min() > -32768 and df_cleaned[col].max() < 32767:
                        df_cleaned[col] = df_cleaned[col].astype('int16')
                    elif df_cleaned[col].min() > -2147483648 and df_cleaned[col].max() < 2147483647:
                        df_cleaned[col] = df_cleaned[col].astype('int32')
        return df_cleaned
    def generate_cleaning_report(self, original_df: pd.DataFrame, cleaned_df: pd.DataFrame) -> Dict[str, Any]:
        try:
            original_missing = original_df.isnull().sum().sum()
            cleaned_missing = cleaned_df.isnull().sum().sum()
            original_duplicates = original_df.duplicated().sum()
            cleaned_duplicates = cleaned_df.duplicated().sum()
            original_memory = original_df.memory_usage(deep=True).sum() / 1024 / 1024
            cleaned_memory = cleaned_df.memory_usage(deep=True).sum() / 1024 / 1024
            original_score = self.calculate_quality_score(original_df)
            cleaned_score = self.calculate_quality_score(cleaned_df)
            report = {
                "summary": {
                    "original_rows": len(original_df),
                    "cleaned_rows": len(cleaned_df),
                    "original_columns": len(original_df.columns),
                    "cleaned_columns": len(cleaned_df.columns),
                    "rows_removed": len(original_df) - len(cleaned_df),
                    "columns_removed": len(original_df.columns) - len(cleaned_df.columns)
                },
                "data_quality": {
                    "original_missing_values": int(original_missing),
                    "cleaned_missing_values": int(cleaned_missing),
                    "missing_reduction": int(original_missing - cleaned_missing),
                    "original_duplicates": int(original_duplicates),
                    "cleaned_duplicates": int(cleaned_duplicates),
                    "duplicate_reduction": int(original_duplicates - cleaned_duplicates)
                },
                "performance": {
                    "original_memory_mb": round(original_memory, 2),
                    "cleaned_memory_mb": round(cleaned_memory, 2),
                            "memory_reduction_mb": round(original_memory - cleaned_memory, 2),
                    "memory_reduction_pct": round((original_memory - cleaned_memory) / original_memory * 100, 2) if original_memory > 0 else 0
                },
                "quality_scores": {
                    "original_score": round(original_score, 1),
                    "cleaned_score": round(cleaned_score, 1),
                    "score_improvement": round(cleaned_score - original_score, 1)
                },
                "transformations_applied": self.transformations_applied,
                "recommendations": self.generate_recommendations(original_df, cleaned_df)
            }
            return report
        except Exception as e:
            print(f"❌ Error generating cleaning report: {e}")
            return {}
    def calculate_quality_score(self, df: pd.DataFrame) -> float:
        try:
            score = 100.0
            total_rows = len(df)
            missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            score -= min(5, missing_pct * 0.05)
            duplicate_pct = (df.duplicated().sum() / len(df)) * 100
            score -= min(5, duplicate_pct * 0.2)
            numeric_cols = df.select_dtypes(include=[np.number])
            for col in numeric_cols.columns:
                if numeric_cols[col].dtype in ['int64', 'float64']:
                    Q1 = numeric_cols[col].quantile(0.25)
                    Q3 = numeric_cols[col].quantile(0.75)
                    IQR = Q3 - Q1
                    if IQR > 0:
                        outliers = ((numeric_cols[col] < (Q1 - 1.5 * IQR)) | (numeric_cols[col] > (Q3 + 1.5 * IQR))).sum()
                        if outliers > 0:
                            out_density = 100.0 * (outliers / total_rows)
                            score -= min(3, out_density * 0.1)
            return max(0, min(100, score))
        except Exception as e:
            print(f"❌ Error calculating quality score: {e}")
            return 0.0
    def generate_recommendations(self, original_df: pd.DataFrame, cleaned_df: pd.DataFrame) -> List[str]:
        recommendations = []
        try:
            remaining_missing = cleaned_df.isnull().sum().sum()
            if remaining_missing > 0:
                recommendations.append(f"Consider additional imputation strategies for {remaining_missing} remaining missing values")
            numeric_cols = cleaned_df.select_dtypes(include=[np.number])
            for col in numeric_cols.columns:
                if numeric_cols[col].std() / numeric_cols[col].mean() > 2:
                    recommendations.append(f"Column '{col}' has high variance - consider log transformation or binning")
            categorical_cols = cleaned_df.select_dtypes(include=['object'])
            for col in categorical_cols.columns:
                unique_count = cleaned_df[col].nunique()
                if unique_count > 50:
                    recommendations.append(f"Column '{col}' has {unique_count} unique values - consider grouping or encoding")
            target_indicators = ['target', 'label', 'class', 'outcome', 'result', 'y']
            has_target = any(any(indicator in col.lower() for indicator in target_indicators) for col in cleaned_df.columns)
            if not has_target:
                recommendations.append("No target variable detected - ensure you have a clear prediction target")
        except Exception as e:
            print(f"❌ Error generating recommendations: {e}")
        return recommendations
    def clean_dataset(self, df: pd.DataFrame, dataset_id: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        try:
            print(f"🧹 Starting data cleaning for dataset {dataset_id}...")
            self.original_shape = df.shape
            quality_report = self.load_quality_report(dataset_id)
            if not quality_report:
                raise ValueError("Could not load quality report")
            suggestions = self.analyze_suggestions(quality_report)
            df_cleaned = df.copy()
            df_cleaned = self.apply_column_cleaning(df_cleaned, suggestions)
            df_cleaned = self.apply_duplicate_removal(df_cleaned, suggestions['duplicates'])
            df_cleaned = self.apply_missing_value_imputation(df_cleaned, suggestions['missing_values'])
            df_cleaned = self.apply_outlier_treatment(df_cleaned, suggestions['outliers'])
            df_cleaned = self.apply_normalization(df_cleaned, suggestions['normalization'])
            df_cleaned = self.apply_data_type_optimization(df_cleaned)
            self.cleaned_shape = df_cleaned.shape
            cleaning_report = self.generate_cleaning_report(df, df_cleaned)
            print(f"✅ Data cleaning completed!")
            print(f"📊 Original: {self.original_shape[0]} rows × {self.original_shape[1]} columns")
            print(f"📊 Cleaned: {self.cleaned_shape[0]} rows × {self.cleaned_shape[1]} columns")
            print(f"📈 Quality score improved: {cleaning_report['quality_scores']['original_score']} → {cleaning_report['quality_scores']['cleaned_score']}")
            return df_cleaned, cleaning_report
        except Exception as e:
            print(f"❌ Error during data cleaning: {e}")
            return df, {"error": str(e)}
    def clean_dataset_with_report(self, df: pd.DataFrame, quality_report: Dict[str, Any], dataset_id: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        try:
            print(f"🧹 Starting data cleaning for dataset {dataset_id}...")
            self.original_shape = df.shape
            self.transformations_applied = []
            self.cleaning_stats = {}
            suggestions = self.analyze_suggestions(quality_report)
            df_cleaned = df.copy()
            df_cleaned = self.apply_column_cleaning(df_cleaned, suggestions)
            df_cleaned = self.apply_duplicate_removal(df_cleaned, suggestions['duplicates'])
            df_cleaned = self.apply_missing_value_imputation(df_cleaned, suggestions['missing_values'])
            df_cleaned = self.apply_outlier_treatment(df_cleaned, suggestions['outliers'])
            df_cleaned = self.apply_normalization(df_cleaned, suggestions['normalization'])
            df_cleaned = self.apply_data_type_optimization(df_cleaned)
            self.cleaned_shape = df_cleaned.shape
            cleaning_report = self.generate_cleaning_report(df, df_cleaned)
            print(f"✅ Data cleaning completed!")
            print(f"📊 Original: {self.original_shape[0]} rows × {self.original_shape[1]} columns")
            print(f"📊 Cleaned: {self.cleaned_shape[0]} rows × {self.cleaned_shape[1]} columns")
            print(f"📈 Quality score improved: {cleaning_report['quality_scores']['original_score']} → {cleaning_report['quality_scores']['cleaned_score']}")
            return df_cleaned, cleaning_report
        except Exception as e:
            print(f"❌ Error during data cleaning: {e}")
            return df, {"error": str(e)}
if __name__ == "__main__":
    engine = DataCleaningEngine()
    sample_data = pd.DataFrame({
        'id': range(1000),
        'value1': np.random.normal(100, 15, 1000),
        'value2': np.random.normal(50, 10, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000),
        'missing_col': [np.nan if i % 10 == 0 else i for i in range(1000)]
    })
    sample_data.loc[0:10, 'value1'] = 1000
    sample_data = pd.concat([sample_data, sample_data.head(50)], ignore_index=True)
    print("🧪 Testing Data Cleaning Engine...")
    print(f"Original data shape: {sample_data.shape}")
    print(f"Missing values: {sample_data.isnull().sum().sum()}")
    print(f"Duplicates: {sample_data.duplicated().sum()}")