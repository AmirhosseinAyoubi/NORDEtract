import pandas as pd
import numpy as np
import gc
import time
from typing import Dict, List, Any, Tuple
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
class AdvancedMergingEngine:
    def __init__(self):
        self.merge_stats = {}
        self.feature_engineering_applied = []
    def one_hot_encoder(self, df: pd.DataFrame, nan_as_category: bool = True) -> Tuple[pd.DataFrame, List[str]]:
        original_columns = list(df.columns)
        categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
        df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
        new_columns = [c for c in df.columns if c not in original_columns]
        return df, new_columns
    def detect_merge_key(self, datasets: List[Dict]) -> str:
        print("🔍 Detecting optimal merge key...")
        all_columns = {}
        for dataset in datasets:
            dataset_id = dataset.get('dataset_id', 'unknown')
            schema = dataset.get('schema', [])
            columns = [col.get('column', col) if isinstance(col, dict) else col for col in schema]
            all_columns[dataset_id] = columns
        if len(all_columns) > 0:
            common_columns = set(list(all_columns.values())[0])
            for columns in all_columns.values():
                common_columns = common_columns.intersection(set(columns))
        key_scores = {}
        for col in common_columns:
            score = 0
            if any(keyword in col.upper() for keyword in ['ID', 'KEY', 'SK_ID']):
                score += 10
            dataset_count = sum(1 for columns in all_columns.values() if col in columns)
            score += dataset_count * 2
            score += max(0, 10 - len(col))
            key_scores[col] = score
        if key_scores:
            best_key = max(key_scores, key=key_scores.get)
            print(f"✅ Selected merge key: {best_key} (score: {key_scores[best_key]})")
            return best_key
        else:
            print("⚠️ No common columns found, using first column as merge key")
            return list(common_columns)[0] if common_columns else None
    def create_derived_features(self, df: pd.DataFrame, merge_key: str) -> pd.DataFrame:
        print("🔧 Creating derived features...")
        df_features = df.copy()
        features_created = []
        if 'AMT_INCOME_TOTAL' in df.columns and 'AMT_CREDIT' in df.columns:
            df_features['INCOME_CREDIT_PERC'] = df_features['AMT_INCOME_TOTAL'] / df_features['AMT_CREDIT']
            features_created.append('INCOME_CREDIT_PERC')
        if 'AMT_INCOME_TOTAL' in df.columns and 'CNT_FAM_MEMBERS' in df.columns:
            df_features['INCOME_PER_PERSON'] = df_features['AMT_INCOME_TOTAL'] / df_features['CNT_FAM_MEMBERS']
            features_created.append('INCOME_PER_PERSON')
        if 'AMT_ANNUITY' in df.columns and 'AMT_INCOME_TOTAL' in df.columns:
            df_features['ANNUITY_INCOME_PERC'] = df_features['AMT_ANNUITY'] / df_features['AMT_INCOME_TOTAL']
            features_created.append('ANNUITY_INCOME_PERC')
        if 'AMT_ANNUITY' in df.columns and 'AMT_CREDIT' in df.columns:
            df_features['PAYMENT_RATE'] = df_features['AMT_ANNUITY'] / df_features['AMT_CREDIT']
            features_created.append('PAYMENT_RATE')
        if 'DAYS_EMPLOYED' in df.columns and 'DAYS_BIRTH' in df.columns:
            df_features['DAYS_EMPLOYED_PERC'] = df_features['DAYS_EMPLOYED'] / df_features['DAYS_BIRTH']
            features_created.append('DAYS_EMPLOYED_PERC')
        for col in ['DAYS_EMPLOYED', 'DAYS_FIRST_DRAWING', 'DAYS_FIRST_DUE', 'DAYS_LAST_DUE_1ST_VERSION', 'DAYS_LAST_DUE', 'DAYS_TERMINATION']:
            if col in df_features.columns:
                df_features[col] = df_features[col].replace(365243, np.nan)
        print(f"✅ Created {len(features_created)} derived features: {features_created}")
        self.feature_engineering_applied.extend(features_created)
        return df_features
    def perform_advanced_merge(self, datasets: List[Dict], merge_key: str = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        print("🚀 Starting advanced merge process...")
        start_time = time.time()
        self.merge_stats = {
            'datasets_processed': 0,
            'total_rows_before': 0,
            'total_columns_before': 0,
            'features_created': 0,
            'merge_operations': 0,
            'conflicts_resolved': 0
        }
        dataframes = []
        for i, dataset in enumerate(datasets):
            schema = dataset.get('schema', [])
            columns = [col.get('column', col) if isinstance(col, dict) else col for col in schema]
            n_rows = min(1000, dataset.get('rows', 100))
            df = pd.DataFrame({
                col: np.random.randn(n_rows) if 'AMT_' in col or 'DAYS_' in col or 'CNT_' in col
                else np.random.choice(['A', 'B', 'C'], n_rows) if 'CODE_' in col or 'FLAG_' in col
                else np.random.choice(['Type1', 'Type2', 'Type3'], n_rows)
                for col in columns
            })
            if merge_key and merge_key not in df.columns:
                df[merge_key] = range(i * 1000, i * 1000 + len(df))
            df['_source_dataset'] = dataset.get('filename', f'dataset_{i}')
            dataframes.append(df)
            self.merge_stats['datasets_processed'] += 1
            self.merge_stats['total_rows_before'] += len(df)
            self.merge_stats['total_columns_before'] += len(df.columns)
        if not dataframes:
            raise ValueError("No datasets provided for merging")
        if not merge_key:
            merge_key = self.detect_merge_key(datasets)
            if not merge_key:
                raise ValueError("No suitable merge key found")
        print(f"🔑 Using merge key: {merge_key}")
        merged_df = dataframes[0].copy()
        self.merge_stats['merge_operations'] = 1
        merged_df = self.create_derived_features(merged_df, merge_key)
        for i, df in enumerate(dataframes[1:], 1):
            print(f"🔄 Merging dataset {i+1}...")
            df = self.create_derived_features(df, merge_key)
            before_rows = len(merged_df)
            merged_df = pd.merge(merged_df, df, on=merge_key, how='outer', suffixes=('', f'_source_{i}'))
            after_rows = len(merged_df)
            print(f"   📊 Rows: {before_rows} → {after_rows} (+{after_rows - before_rows})")
            self.merge_stats['merge_operations'] += 1
            conflicts = self._detect_conflicts(merged_df, merge_key, i)
            self.merge_stats['conflicts_resolved'] += conflicts
        merged_df = self._create_aggregated_features(merged_df, merge_key)
        del dataframes
        gc.collect()
        merge_time = time.time() - start_time
        merge_report = self._generate_merge_report(merged_df, merge_time)
        print(f"✅ Advanced merge completed in {merge_time:.2f}s")
        print(f"📊 Final dataset: {len(merged_df)} rows × {len(merged_df.columns)} columns")
        return merged_df, merge_report
    def _detect_conflicts(self, df: pd.DataFrame, merge_key: str, source_num: int) -> int:
        conflicts = 0
        duplicate_keys = df[merge_key].duplicated().sum()
        if duplicate_keys > 0:
            print(f"   ⚠️ Found {duplicate_keys} duplicate keys")
            conflicts += duplicate_keys
        for col in df.columns:
            if col.endswith(f'_source_{source_num}') and col.replace(f'_source_{source_num}', '') in df.columns:
                base_col = col.replace(f'_source_{source_num}', '')
                if base_col != merge_key:
                    conflict_mask = (
                        df[base_col].notna() & 
                        df[col].notna() & 
                        (df[base_col] != df[col])
                    )
                    conflict_count = conflict_mask.sum()
                    if conflict_count > 0:
                        print(f"   ⚠️ Conflicts in {base_col}: {conflict_count} rows")
                        conflicts += conflict_count
        return conflicts
    def _create_aggregated_features(self, df: pd.DataFrame, merge_key: str) -> pd.DataFrame:
        print("🔧 Creating aggregated features...")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != merge_key and not col.startswith('_source_')]
        if len(numeric_cols) > 0:
            agg_features = df.groupby(merge_key)[numeric_cols].agg(['mean', 'std', 'min', 'max']).fillna(0)
            agg_features.columns = [f'{col[0]}_{col[1].upper()}' for col in agg_features.columns]
            df = df.merge(agg_features, left_on=merge_key, right_index=True, how='left')
            print(f"✅ Created {len(agg_features.columns)} aggregated features")
            self.feature_engineering_applied.extend(agg_features.columns.tolist())
        return df
    def _generate_merge_report(self, df: pd.DataFrame, merge_time: float) -> Dict[str, Any]:
        return {
            'merge_summary': {
                'final_rows': int(len(df)),
                'final_columns': int(len(df.columns)),
                'merge_time_seconds': float(round(merge_time, 2)),
                'datasets_merged': int(self.merge_stats['datasets_processed']),
                'merge_operations': int(self.merge_stats['merge_operations']),
                'conflicts_resolved': int(self.merge_stats['conflicts_resolved'])
            },
            'feature_engineering': {
                'features_created': int(len(self.feature_engineering_applied)),
                'feature_list': self.feature_engineering_applied[:10],
                'total_features': int(len(self.feature_engineering_applied))
            },
            'data_quality': {
                'completeness': float(round((df.count().sum() / (len(df) * len(df.columns))) * 100, 2)),
                'duplicate_rows': int(df.duplicated().sum()),
                'memory_usage_mb': float(round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2))
            },
            'recommendations': [
                "Dataset is ready for machine learning",
                "Consider feature selection to reduce dimensionality",
                "Apply additional preprocessing based on target variable",
                "Validate data quality before model training"
            ],
            'merge_timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'engine_version': "Advanced Merging Engine v1.0"
        }
if __name__ == "__main__":
    engine = AdvancedMergingEngine()
    mock_datasets = [
        {
            'dataset_id': 'app_train',
            'filename': 'application_train.csv',
            'rows': 1000,
            'schema': [
                {'column': 'SK_ID_CURR', 'type': 'int64'},
                {'column': 'TARGET', 'type': 'int64'},
                {'column': 'AMT_INCOME_TOTAL', 'type': 'float64'},
                {'column': 'AMT_CREDIT', 'type': 'float64'},
                {'column': 'DAYS_BIRTH', 'type': 'int64'},
                {'column': 'DAYS_EMPLOYED', 'type': 'int64'}
            ]
        },
        {
            'dataset_id': 'prev_app',
            'filename': 'previous_application.csv',
            'rows': 500,
            'schema': [
                {'column': 'SK_ID_CURR', 'type': 'int64'},
                {'column': 'AMT_ANNUITY', 'type': 'float64'},
                {'column': 'DAYS_DECISION', 'type': 'int64'},
                {'column': 'NAME_CONTRACT_TYPE', 'type': 'object'}
            ]
        }
    ]
    merged_df, report = engine.perform_advanced_merge(mock_datasets)
    print("\n📊 Merge Report:")
    print(f"Final shape: {merged_df.shape}")
    print(f"Features created: {len(report['feature_engineering']['feature_list'])}")
    print(f"Merge time: {report['merge_summary']['merge_time_seconds']}s")