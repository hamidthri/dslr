import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import sys

def extract_dataset_metadata(df):
    metadata = {}
    
    metadata['dataset_info'] = {
        'total_records': len(df),
        'total_columns': len(df.columns),
        'column_names': list(df.columns),
        'memory_usage_mb': round(df.memory_usage(deep=True).sum() / 1024**2, 2)
    }
    
    metadata['data_types'] = df.dtypes.astype(str).to_dict()
    
    metadata['missing_values'] = {
        'total_missing': int(df.isnull().sum().sum()),
        'missing_by_column': df.isnull().sum().to_dict(),
        'missing_percentage_by_column': (df.isnull().sum() / len(df) * 100).round(2).to_dict()
    }
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    
    metadata['column_types'] = {
        'numeric_columns': numeric_columns,
        'categorical_columns': categorical_columns,
        'numeric_count': len(numeric_columns),
        'categorical_count': len(categorical_columns)
    }
    
    # Add visualization-specific metadata
    metadata['viz_categories'] = {
        'identifiers': [col for col in df.columns if col.lower() in ['index', 'first name', 'last name', 'name']],
        'grouping_vars': [col for col in categorical_columns if col.lower() in ['hogwarts house', 'house', 'best hand', 'hand']],
        'measurement_vars': [col for col in numeric_columns if col.lower() not in ['index']],
        'date_vars': [col for col in df.columns if col.lower() in ['birthday', 'birth', 'date']],
        'subject_vars': [col for col in df.columns if col not in ['Index', 'Hogwarts House', 'First Name', 'Last Name', 'Birthday', 'Best Hand']]
    }
    
    metadata['numeric_statistics'] = {}
    for col in numeric_columns:
        if df[col].notna().any():
            metadata['numeric_statistics'][col] = {
                'min': float(df[col].min()) if pd.notna(df[col].min()) else None,
                'max': float(df[col].max()) if pd.notna(df[col].max()) else None,
                'mean': float(df[col].mean()) if pd.notna(df[col].mean()) else None,
                'median': float(df[col].median()) if pd.notna(df[col].median()) else None,
                'std': float(df[col].std()) if pd.notna(df[col].std()) else None,
                'unique_values': int(df[col].nunique()),
                'zero_values': int((df[col] == 0).sum()),
                'negative_values': int((df[col] < 0).sum()) if df[col].dtype in ['int64', 'float64'] else 0
            }
    
    metadata['categorical_statistics'] = {}
    for col in categorical_columns:
        value_counts = df[col].value_counts()
        metadata['categorical_statistics'][col] = {
            'unique_values': int(df[col].nunique()),
            'most_frequent': str(value_counts.index[0]) if len(value_counts) > 0 else None,
            'most_frequent_count': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
            'value_distribution': value_counts.head(10).to_dict()
        }
    
    if 'Birthday' in df.columns:
        df['Birthday'] = pd.to_datetime(df['Birthday'], errors='coerce')
        metadata['date_analysis'] = {
            'date_range': {
                'earliest': str(df['Birthday'].min().date()) if pd.notna(df['Birthday'].min()) else None,
                'latest': str(df['Birthday'].max().date()) if pd.notna(df['Birthday'].max()) else None
            },
            'birth_year_distribution': df['Birthday'].dt.year.value_counts().to_dict() if df['Birthday'].notna().any() else {}
        }
    
    subject_columns = [col for col in df.columns if col not in ['Index', 'Hogwarts House', 'First Name', 'Last Name', 'Birthday', 'Best Hand']]
    
    if subject_columns:
        metadata['subject_performance'] = {}
        for col in subject_columns:
            if df[col].dtype in ['int64', 'float64'] and df[col].notna().any():
                metadata['subject_performance'][col] = {
                    'average_score': float(df[col].mean()),
                    'performance_range': float(df[col].max() - df[col].min()),
                    'students_above_average': int((df[col] > df[col].mean()).sum()),
                    'top_score': float(df[col].max()),
                    'lowest_score': float(df[col].min())
                }
    
    if 'Hogwarts House' in df.columns:
        house_performance = {}
        for house in df['Hogwarts House'].unique():
            if pd.notna(house):
                house_data = df[df['Hogwarts House'] == house]
                house_numeric = house_data.select_dtypes(include=[np.number])
                if not house_numeric.empty:
                    house_performance[house] = {
                        'student_count': len(house_data),
                        'average_overall_performance': float(house_numeric.mean().mean()),
                        'best_subject': house_numeric.mean().idxmax(),
                        'best_subject_score': float(house_numeric.mean().max())
                    }
        metadata['house_analysis'] = house_performance
    
    metadata['data_quality'] = {
        'completeness_score': round((1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100, 2),
        'duplicate_rows': int(df.duplicated().sum()),
        'columns_with_missing_data': len([col for col in df.columns if df[col].isnull().any()]),
        'data_integrity_issues': []
    }
    
    for col in numeric_columns:
        if df[col].dtype in ['int64', 'float64']:
            if (df[col] == np.inf).any() or (df[col] == -np.inf).any():
                metadata['data_quality']['data_integrity_issues'].append(f"Infinite values found in {col}")
    
    metadata['visualization_recommendations'] = {
        'distribution_plots': numeric_columns[:5],
        'categorical_plots': categorical_columns,
        'correlation_analysis': numeric_columns if len(numeric_columns) > 1 else [],
        'time_series_potential': ['Birthday'] if 'Birthday' in df.columns else [],
        'grouping_variables': categorical_columns,
        'top_subjects_for_comparison': list(metadata.get('subject_performance', {}).keys())[:5]
    }
    
    return metadata

def print_visualization_guide(metadata):
    """Print comprehensive visualization guide using metadata."""
    print("\n" + "="*60)
    print("COMPREHENSIVE DATASET VISUALIZATION GUIDE")
    print("="*60)
    
    viz_cats = metadata['viz_categories']
    
    print(f"\nVARIABLE CATEGORIES FOR VISUALIZATION:")
    print(f"  • Identifiers (skip): {viz_cats['identifiers']}")
    print(f"  • Grouping Variables: {viz_cats['grouping_vars']}")
    print(f"  • Measurements: {viz_cats['measurement_vars'][:5]}{'...' if len(viz_cats['measurement_vars']) > 5 else ''}")
    print(f"  • Date Variables: {viz_cats['date_vars']}")
    print(f"  • Subject Scores: {len(viz_cats['subject_vars'])} subjects available")
    
    print(f"\nRECOMMENDED VISUALIZATIONS BASED ON METADATA:")
    
    print(f"\n1. CATEGORICAL ANALYSIS:")
    if viz_cats['grouping_vars']:
        for var in viz_cats['grouping_vars']:
            dist = metadata['categorical_statistics'].get(var, {})
            print(f"   • {var}: {dist.get('unique_values', 0)} categories, most common: {dist.get('most_frequent', 'N/A')}")
    
    print(f"\n2. PERFORMANCE ANALYSIS:")
    if 'subject_performance' in metadata:
        top_subjects = sorted(metadata['subject_performance'].items(), 
                            key=lambda x: x[1]['average_score'], reverse=True)[:3]
        for subject, stats in top_subjects:
            print(f"   • {subject}: avg={stats['average_score']:.1f}, range={stats['performance_range']:.1f}")
    
    print(f"\n3. HOUSE COMPARISON:")
    if 'house_analysis' in metadata:
        for house, stats in metadata['house_analysis'].items():
            print(f"   • {house}: {stats['student_count']} students, best at {stats['best_subject']}")
    
    print(f"\n4. DATA QUALITY INSIGHTS:")
    quality = metadata['data_quality']
    print(f"   • Completeness: {quality['completeness_score']}%")
    print(f"   • Columns with missing data: {quality['columns_with_missing_data']}")
    print(f"   • Duplicate rows: {quality['duplicate_rows']}")

def create_comprehensive_visualizations(df, metadata):
    """Create visualizations using both metadata insights and data."""
    viz_cats = metadata['viz_categories']
    subject_cols = [col for col in viz_cats['measurement_vars'] if col != 'Index']
    
    # Create a comprehensive dashboard
    fig = plt.figure(figsize=(20, 15))
    fig.suptitle('Dataset Analysis Dashboard Based on Metadata', fontsize=20, y=0.95)
    
    # 1. House Distribution with metadata insights
    plt.subplot(3, 4, 1)
    if 'Hogwarts House' in df.columns:
        house_data = df['Hogwarts House'].value_counts()
        colors = ['#d32f2f', '#ffb300', '#1976d2', '#388e3c']  # House colors
        house_data.plot(kind='bar', color=colors[:len(house_data)])
        plt.title('Student Distribution by House')
        plt.xticks(rotation=45)
    
    # 2. Hand preference with percentage from metadata
    plt.subplot(3, 4, 2)
    if 'Best Hand' in df.columns:
        hand_stats = metadata['categorical_statistics'].get('Best Hand', {})
        df['Best Hand'].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=90)
        plt.title(f'Hand Preference\n(Most common: {hand_stats.get("most_frequent", "N/A")})')
    
    # 3. Birth year trends using date analysis
    plt.subplot(3, 4, 3)
    if metadata.get('date_analysis'):
        birth_years = metadata['date_analysis']['birth_year_distribution']
        if birth_years:
            years = sorted(birth_years.keys())
            counts = [birth_years[year] for year in years]
            plt.plot(years, counts, marker='o', linewidth=2, markersize=6)
            plt.title('Students by Birth Year')
            plt.xlabel('Year')
            plt.ylabel('Count')
    
    # 4. Top performing subjects from metadata
    plt.subplot(3, 4, 4)
    if 'subject_performance' in metadata:
        subject_means = {k: v['average_score'] for k, v in metadata['subject_performance'].items()}
        top_subjects = dict(sorted(subject_means.items(), key=lambda x: x[1], reverse=True)[:5])
        plt.bar(range(len(top_subjects)), list(top_subjects.values()), color='lightcoral')
        plt.xticks(range(len(top_subjects)), list(top_subjects.keys()), rotation=45, ha='right')
        plt.title('Top 5 Subjects by Average Score')
    
    # 5. House performance comparison using house_analysis
    plt.subplot(3, 4, 5)
    if 'house_analysis' in metadata:
        house_perf = {k: v['average_overall_performance'] for k, v in metadata['house_analysis'].items()}
        plt.bar(house_perf.keys(), house_perf.values(), color='lightgreen')
        plt.title('Average Performance by House')
        plt.xticks(rotation=45)
    
    # 6. Data quality visualization
    plt.subplot(3, 4, 6)
    missing_data = metadata['missing_values']['missing_by_column']
    cols_with_missing = {k: v for k, v in missing_data.items() if v > 0}
    if cols_with_missing:
        plt.barh(range(len(cols_with_missing)), list(cols_with_missing.values()))
        plt.yticks(range(len(cols_with_missing)), list(cols_with_missing.keys()))
        plt.title('Missing Values by Column')
        plt.xlabel('Count')
    else:
        plt.text(0.5, 0.5, 'No Missing Values!', ha='center', va='center', fontsize=14, color='green')
        plt.title('Data Quality: Excellent')
    
    # 7. Subject correlation heatmap
    plt.subplot(3, 4, 7)
    if len(subject_cols) > 1:
        corr_matrix = df[subject_cols].corr()
        sns.heatmap(corr_matrix.iloc[:5, :5], annot=True, cmap='coolwarm', center=0, 
                   square=True, linewidths=0.5)
        plt.title('Subject Correlations')
    
    # 8. Performance distribution
    plt.subplot(3, 4, 8)
    if subject_cols:
        overall_scores = df[subject_cols].mean(axis=1)
        plt.hist(overall_scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title('Overall Performance Distribution')
        plt.xlabel('Average Score')
        plt.ylabel('Frequency')
    
    # 9. Best subject by house
    plt.subplot(3, 4, 9)
    if 'house_analysis' in metadata:
        house_best = {k: v['best_subject'] for k, v in metadata['house_analysis'].items()}
        best_subjects = list(house_best.values())
        subject_counts = {subj: best_subjects.count(subj) for subj in set(best_subjects)}
        plt.bar(subject_counts.keys(), subject_counts.values(), color='orange')
        plt.title('Most Excelled Subjects by House')
        plt.xticks(rotation=45, ha='right')
    
    # 10. Memory usage and dataset size info
    plt.subplot(3, 4, 10)
    dataset_info = metadata['dataset_info']
    info_text = f"""Dataset Overview:
Records: {dataset_info['total_records']:,}
Columns: {dataset_info['total_columns']}
Memory: {dataset_info['memory_usage_mb']} MB
Completeness: {metadata['data_quality']['completeness_score']}%"""
    plt.text(0.1, 0.5, info_text, fontsize=12, verticalalignment='center')
    plt.axis('off')
    plt.title('Dataset Statistics')
    
    # 11. Subject difficulty analysis
    plt.subplot(3, 4, 11)
    if 'subject_performance' in metadata:
        subject_difficulty = {k: v['performance_range'] for k, v in metadata['subject_performance'].items()}
        top_difficult = dict(sorted(subject_difficulty.items(), key=lambda x: x[1], reverse=True)[:5])
        plt.bar(range(len(top_difficult)), list(top_difficult.values()), color='red', alpha=0.7)
        plt.xticks(range(len(top_difficult)), list(top_difficult.keys()), rotation=45, ha='right')
        plt.title('Most Challenging Subjects\n(by score range)')
    
    # 12. Students above average per subject
    plt.subplot(3, 4, 12)
    if 'subject_performance' in metadata:
        above_avg = {k: v['students_above_average'] for k, v in metadata['subject_performance'].items()}
        top_achievers = dict(sorted(above_avg.items(), key=lambda x: x[1], reverse=True)[:5])
        plt.bar(range(len(top_achievers)), list(top_achievers.values()), color='green', alpha=0.7)
        plt.xticks(range(len(top_achievers)), list(top_achievers.keys()), rotation=45, ha='right')
        plt.title('Subjects with Most\nAbove-Average Students')
    
    plt.tight_layout()
    plt.savefig('comprehensive_analysis_dashboard.png', dpi=300, bbox_inches='tight')
    plt.show()

def load_and_analyze_dataset(file_path):
    df = pd.read_csv(file_path)
    metadata = extract_dataset_metadata(df)
    return df, metadata

def save_metadata(metadata, output_file='dataset_metadata.json'):
    with open(output_file, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = "dataset_train.csv"

    df, metadata = load_and_analyze_dataset(file_path)
    
    print("Dataset Metadata Analysis Complete")
    print(f"Total Records: {metadata['dataset_info']['total_records']}")
    print(f"Total Columns: {metadata['dataset_info']['total_columns']}")
    print(f"Missing Values: {metadata['missing_values']['total_missing']}")
    print(f"Data Completeness: {metadata['data_quality']['completeness_score']}%")
    
    if 'house_analysis' in metadata:
        print("\nHouse Performance Summary:")
        for house, stats in metadata['house_analysis'].items():
            print(f"{house}: {stats['student_count']} students, avg performance: {stats['average_overall_performance']:.2f}")
    
    # Print comprehensive visualization guide
    print_visualization_guide(metadata)
    
    # Ask user for visualization
    try:
        create_comprehensive_visualizations(df, metadata)
    except KeyboardInterrupt:
        print("\nExiting...")
    
    save_metadata(metadata)
    print(f"\nDetailed metadata saved to dataset_metadata.json")