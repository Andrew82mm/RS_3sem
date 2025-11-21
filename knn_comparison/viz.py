import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_comparison(df, save_path='results/knn_comparison.png'):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('KNN Algorithm Comparison', fontsize=16, fontweight='bold')
    x = np.arange(len(df))
    w = 0.35

    # 1. Build time
    ax = axes[0, 0]
    ax.bar(x - w/2, df['Build Time User (s)'], w, label='User', alpha=.8)
    ax.bar(x + w/2, df['Build Time Item (s)'], w, label='Item', alpha=.8)
    ax.set_title('Index Build Time'); ax.set_ylabel('sec')
    ax.set_xticks(x); ax.set_xticklabels(df['Algorithm'], rotation=45, ha='right')
    ax.legend(); ax.grid(axis='y', alpha=.3)

    # 2. Memory
    ax = axes[0, 1]
    ax.bar(x, df['Total Memory (MB)'], label='Total Memory', alpha=.8, color='skyblue')
    ax.set_title('Total Memory Usage'); ax.set_ylabel('MB')
    ax.set_xticks(x); ax.set_xticklabels(df['Algorithm'], rotation=45, ha='right')
    ax.legend(); ax.grid(axis='y', alpha=.3)

    # 3. Query time
    ax = axes[0, 2]
    ax.bar(x - w/2, df['Avg Query Time User (ms)'], w, label='User', alpha=.8)
    ax.bar(x + w/2, df['Avg Query Time Item (ms)'], w, label='Item', alpha=.8)
    ax.set_title('Query Time'); ax.set_ylabel('ms'); ax.set_yscale('log')
    ax.set_xticks(x); ax.set_xticklabels(df['Algorithm'], rotation=45, ha='right')
    ax.legend(); ax.grid(axis='y', alpha=.3)

    # 4. Recall
    ax = axes[1, 0]
    ax.bar(x - w/2, df['Recall@20 User'], w, label='User', alpha=.8)
    ax.bar(x + w/2, df['Recall@20 Item'], w, label='Item', alpha=.8)
    ax.set_title('Recall@20'); ax.set_ylabel('R@20'); ax.set_ylim(0, 1.05)
    ax.set_xticks(x); ax.set_xticklabels(df['Algorithm'], rotation=45, ha='right')
    ax.legend(); ax.grid(axis='y', alpha=.3)

    # 5. Precision
    ax = axes[1, 1]
    ax.bar(x - w/2, df['Precision@20 User'], w, label='User', alpha=.8)
    ax.bar(x + w/2, df['Precision@20 Item'], w, label='Item', alpha=.8)
    ax.set_title('Precision@20'); ax.set_ylabel('P@20'); ax.set_ylim(0, 1.05)
    ax.set_xticks(x); ax.set_xticklabels(df['Algorithm'], rotation=45, ha='right')
    ax.legend(); ax.grid(axis='y', alpha=.3)

    # 6. Speed vs Accuracy scatter
    ax = axes[1, 2]
    for _, row in df.iterrows():
        avg_q = (row['Avg Query Time User (ms)'] + row['Avg Query Time Item (ms)']) / 2
        avg_r = (row['Recall@20 User'] + row['Recall@20 Item']) / 2
        ax.scatter(avg_q, avg_r, s=200, alpha=.7)
        ax.annotate(row['Algorithm'], (avg_q, avg_r), xytext=(5, 5),
                    textcoords='offset points', fontsize=9)
    ax.set_xlabel('Avg Query Time (ms, log)'); ax.set_ylabel('Avg Recall@20')
    ax.set_xscale('log'); ax.grid(True, alpha=.3); ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()