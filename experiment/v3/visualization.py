"""
Visualization utilities for tournament results and model comparisons.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Tuple
import json


def plot_win_matrix(
    win_matrix: np.ndarray,
    model_names: List[str],
    title: str = "Win Matrix",
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = "Blues",
    annotate: bool = True,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot win matrix as a heatmap.
    
    Args:
        win_matrix: [n_models, n_models] array of win counts
        model_names: List of model names
        title: Plot title
        figsize: Figure size
        cmap: Colormap
        annotate: Show values in cells
        save_path: Optional path to save figure
    
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create mask for diagonal (self vs self)
    mask = np.eye(len(model_names), dtype=bool)
    
    sns.heatmap(
        win_matrix,
        mask=mask,
        annot=annotate,
        fmt='.1f',
        cmap=cmap,
        xticklabels=model_names,
        yticklabels=model_names,
        ax=ax,
        cbar_kws={'label': 'Wins'}
    )
    
    ax.set_xlabel('Opponent (Loser)')
    ax.set_ylabel('Winner')
    ax.set_title(title)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_probability_matrix(
    prob_matrix: np.ndarray,
    model_names: List[str],
    title: str = "Win Probability Matrix",
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = "RdYlGn",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot probability matrix as a heatmap.
    
    Args:
        prob_matrix: [n_models, n_models] array of win probabilities
        model_names: List of model names
        title: Plot title
        figsize: Figure size
        cmap: Colormap (diverging recommended)
        save_path: Optional path to save figure
    
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    mask = np.eye(len(model_names), dtype=bool)
    
    sns.heatmap(
        prob_matrix,
        mask=mask,
        annot=True,
        fmt='.2f',
        cmap=cmap,
        center=0.5,
        vmin=0,
        vmax=1,
        xticklabels=model_names,
        yticklabels=model_names,
        ax=ax,
        cbar_kws={'label': 'P(row beats col)'}
    )
    
    ax.set_xlabel('Opponent')
    ax.set_ylabel('Model')
    ax.set_title(title)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_rankings(
    rankings: List[Tuple[str, float]],
    title: str = "Model Rankings",
    figsize: Tuple[int, int] = (10, 6),
    color: str = "steelblue",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot model rankings as a horizontal bar chart.
    
    Args:
        rankings: List of (model_name, score) tuples, sorted by score
        title: Plot title
        figsize: Figure size
        color: Bar color
        save_path: Optional path to save figure
    
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    names = [r[0] for r in rankings]
    scores = [r[1] for r in rankings]
    
    y_pos = np.arange(len(names))
    
    bars = ax.barh(y_pos, scores, color=color, alpha=0.8)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()  # Top rank at top
    ax.set_xlabel('Score')
    ax.set_title(title)
    
    # Add value labels
    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{score:.3f}', va='center', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_score_distribution(
    scores: List[float],
    title: str = "Score Distribution",
    figsize: Tuple[int, int] = (10, 6),
    bins: int = 30,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot distribution of scores.
    
    Args:
        scores: List of score values
        title: Plot title
        figsize: Figure size
        bins: Number of histogram bins
        save_path: Optional path to save figure
    
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.hist(scores, bins=bins, edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(scores), color='red', linestyle='--', label=f'Mean: {np.mean(scores):.2f}')
    ax.axvline(np.median(scores), color='green', linestyle='--', label=f'Median: {np.median(scores):.2f}')
    
    ax.set_xlabel('Score')
    ax.set_ylabel('Frequency')
    ax.set_title(title)
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_match_history(
    match_history: List[Dict],
    model_names: List[str],
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot cumulative wins over time for each model.
    
    Args:
        match_history: List of match dicts with 'winner' key
        model_names: List of model names
        figsize: Figure size
        save_path: Optional path to save figure
    
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Track cumulative wins
    cumulative_wins = {name: [0] for name in model_names}
    
    for match in match_history:
        winner = match.get('winner', 'tie')
        for name in model_names:
            if winner == name:
                cumulative_wins[name].append(cumulative_wins[name][-1] + 1)
            elif winner == 'tie' and name in [match.get('model_a'), match.get('model_b')]:
                cumulative_wins[name].append(cumulative_wins[name][-1] + 0.5)
            else:
                cumulative_wins[name].append(cumulative_wins[name][-1])
    
    # Plot
    for name in model_names:
        ax.plot(cumulative_wins[name], label=name, linewidth=2)
    
    ax.set_xlabel('Match Number')
    ax.set_ylabel('Cumulative Wins')
    ax.set_title('Model Performance Over Tournament')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_pairwise_comparison(
    win_matrix: np.ndarray,
    model_names: List[str],
    model_a_idx: int,
    model_b_idx: int,
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot detailed pairwise comparison between two models.
    
    Args:
        win_matrix: Win matrix
        model_names: Model names
        model_a_idx: Index of first model
        model_b_idx: Index of second model
        figsize: Figure size
        save_path: Optional path to save
    
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    wins_a = win_matrix[model_a_idx, model_b_idx]
    wins_b = win_matrix[model_b_idx, model_a_idx]
    
    name_a = model_names[model_a_idx]
    name_b = model_names[model_b_idx]
    
    categories = [f'{name_a} wins', f'{name_b} wins']
    values = [wins_a, wins_b]
    colors = ['#2ecc71', '#e74c3c']
    
    bars = ax.bar(categories, values, color=colors, alpha=0.8)
    
    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.0f}', ha='center', va='bottom', fontsize=12)
    
    ax.set_ylabel('Number of Wins')
    ax.set_title(f'{name_a} vs {name_b}')
    
    # Add win rate annotation
    total = wins_a + wins_b
    if total > 0:
        ax.text(0.5, 0.95, f'Win Rate: {name_a}={wins_a/total:.1%}, {name_b}={wins_b/total:.1%}',
                transform=ax.transAxes, ha='center', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def create_summary_dashboard(
    tournament_stats: Dict,
    figsize: Tuple[int, int] = (16, 12),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a summary dashboard with multiple plots.
    
    Args:
        tournament_stats: Dict with tournament results
        figsize: Figure size
        save_path: Optional path to save
    
    Returns:
        matplotlib Figure
    """
    fig = plt.figure(figsize=figsize)
    
    model_names = tournament_stats['model_names']
    win_matrix = np.array(tournament_stats['win_matrix'])
    alpharank_scores = tournament_stats.get('alpharank_scores', [])
    elo_scores = tournament_stats.get('elo_scores', [])
    
    # 1. Win matrix heatmap (top left)
    ax1 = fig.add_subplot(2, 2, 1)
    mask = np.eye(len(model_names), dtype=bool)
    sns.heatmap(win_matrix, mask=mask, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=model_names, yticklabels=model_names, ax=ax1)
    ax1.set_title('Win Matrix')
    ax1.set_xlabel('Opponent')
    ax1.set_ylabel('Winner')
    
    # 2. AlphaRank scores (top right)
    ax2 = fig.add_subplot(2, 2, 2)
    if alpharank_scores:
        rankings = sorted(zip(model_names, alpharank_scores), key=lambda x: x[1], reverse=True)
        names = [r[0] for r in rankings]
        scores = [r[1] for r in rankings]
        y_pos = np.arange(len(names))
        bars = ax2.barh(y_pos, scores, color='steelblue', alpha=0.8)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(names)
        ax2.invert_yaxis()
        ax2.set_xlabel('AlphaRank Score')
        ax2.set_title('AlphaRank Rankings')
        for bar, score in zip(bars, scores):
            ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{score:.3f}', va='center', fontsize=9)
    
    # 3. Elo scores (bottom left)
    ax3 = fig.add_subplot(2, 2, 3)
    if elo_scores:
        rankings = sorted(zip(model_names, elo_scores), key=lambda x: x[1], reverse=True)
        names = [r[0] for r in rankings]
        scores = [r[1] for r in rankings]
        y_pos = np.arange(len(names))
        bars = ax3.barh(y_pos, scores, color='coral', alpha=0.8)
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(names)
        ax3.invert_yaxis()
        ax3.set_xlabel('Elo-like Score')
        ax3.set_title('Elo Rankings (Avg Win Rate)')
        for bar, score in zip(bars, scores):
            ax3.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{score:.2f}', va='center', fontsize=9)
    
    # 4. Summary stats (bottom right)
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')
    
    total_matches = tournament_stats.get('total_matches', 0)
    summary_text = f"""Tournament Summary
    
Total Matches: {total_matches}
Number of Models: {len(model_names)}
Models: {', '.join(model_names)}

AlphaRank Winner: {model_names[np.argmax(alpharank_scores)] if alpharank_scores else 'N/A'}
Elo Winner: {model_names[np.argmax(elo_scores)] if elo_scores else 'N/A'}
"""
    ax4.text(0.1, 0.5, summary_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='center', family='monospace')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig
