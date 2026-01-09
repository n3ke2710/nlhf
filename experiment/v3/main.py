#!/usr/bin/env python3
"""
NLHF Tournament - главный CLI для запуска экспериментов.

Usage:
    python main.py --config config/models.yaml --prompts 20
    python main.py --models gpt-4o-mini gpt-3.5-turbo --arbiter llm
"""

import os
import sys
import argparse
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from models import ModelRegistry, BaseModel
from models.base import ModelConfig, create_response_fn
from models.mock import create_mock_models
from models.api import create_openai_models
from arbiters import HeuristicArbiter, LLMArbiter, HybridArbiter
from arbiters.hybrid import create_arbiter
from tournament import Tournament
from visualization import (
    plot_win_matrix,
    plot_probability_matrix,
    plot_rankings,
    create_summary_dashboard
)


def load_env():
    """Load .env file if exists."""
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    value = value.strip('"').strip("'")
                    os.environ.setdefault(key, value)


def load_prompts(dataset_name: str, split: str) -> List[str]:
    """Load prompts from dataset."""
    from datasets import load_dataset
    
    print(f"Loading dataset {dataset_name}...")
    dataset = load_dataset(dataset_name, split=split)
    prompts = [item["prompt"] for item in dataset]
    print(f"  ✓ Loaded {len(prompts)} prompts")
    return prompts


def create_models_from_args(args) -> Dict[str, BaseModel]:
    """Create models based on CLI arguments."""
    
    if args.config:
        # Load from YAML config
        from config.models import load_models_from_yaml
        models_config = load_models_from_yaml(args.config)
        return models_config.create_models()
    
    elif args.models:
        # Create API models from list
        return create_openai_models(
            model_names=args.models,
            api_key=args.api_key
        )
    
    else:
        # Default: mock models
        print("⚠️  Using mock models (no --models or --config specified)")
        return create_mock_models()


def run_tournament(
    models: Dict[str, BaseModel],
    prompts: List[str],
    arbiter,
    num_prompts: int = 50,
    matches_per_pair: int = 2,
    alpha: float = 1.0,
    log_dir: str = "./output/logs"
) -> Tournament:
    """Run the tournament."""
    
    model_names = list(models.keys())
    get_response = create_response_fn(models)
    
    tournament = Tournament(
        model_names=model_names,
        alpha=alpha,
        log_dir=log_dir
    )
    
    # Wrap arbiter for tournament
    def arbiter_fn(prompt, resp_a, resp_b):
        result = arbiter(prompt, resp_a, resp_b)
        # Return winner, score_a, score_b
        return result.winner, result.score_a, result.score_b
    
    tournament.set_arbiter(arbiter_fn)
    
    print(f"\nRunning tournament: {len(model_names)} models, "
          f"{num_prompts} prompts, {matches_per_pair} matches/pair")
    
    for prompt_idx in tqdm(range(min(num_prompts, len(prompts))), desc="Matches"):
        prompt = prompts[prompt_idx]
        
        for _ in range(matches_per_pair):
            model_a, model_b = random.sample(model_names, 2)
            
            response_a = get_response(model_a, prompt)
            response_b = get_response(model_b, prompt)
            
            tournament.run_match(
                model_a=model_a,
                model_b=model_b,
                prompt=prompt,
                response_a=response_a,
                response_b=response_b
            )
    
    return tournament


def save_results(tournament: Tournament, output_dir: str):
    """Save all results and graphics."""
    import matplotlib.pyplot as plt
    
    os.makedirs(output_dir, exist_ok=True)
    graphics_dir = os.path.join(output_dir, "graphics")
    os.makedirs(graphics_dir, exist_ok=True)
    
    model_names = tournament.model_names
    win_matrix = tournament.get_win_matrix()
    prob_matrix = tournament.get_probability_matrix()
    alpharank = tournament.get_alpharank_scores()
    elo = tournament.get_elo_scores()
    
    print(f"\nSaving results to {output_dir}/")
    
    # Win matrix
    fig = plot_win_matrix(win_matrix, model_names)
    fig.savefig(os.path.join(graphics_dir, "win_matrix.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # Probability matrix
    fig = plot_probability_matrix(prob_matrix, model_names)
    fig.savefig(os.path.join(graphics_dir, "probability_matrix.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # Rankings
    alpharank_rankings = list(zip(model_names, alpharank))
    alpharank_rankings.sort(key=lambda x: x[1], reverse=True)
    
    elo_rankings = list(zip(model_names, elo))
    elo_rankings.sort(key=lambda x: x[1], reverse=True)
    
    fig = plot_rankings(alpharank_rankings, title="AlphaRank")
    fig.savefig(os.path.join(graphics_dir, "alpharank.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    fig = plot_rankings(elo_rankings, title="Elo")
    fig.savefig(os.path.join(graphics_dir, "elo.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # Dashboard
    tournament_stats = {
        'model_names': model_names,
        'win_matrix': win_matrix.tolist(),
        'alpharank_scores': alpharank.tolist(),
        'elo_scores': elo.tolist()
    }
    fig = create_summary_dashboard(tournament_stats)
    fig.savefig(os.path.join(graphics_dir, "dashboard.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"  ✓ Graphics saved to {graphics_dir}/")
    
    # Print summary
    print("\n" + "=" * 60)
    print("TOURNAMENT RESULTS")
    print("=" * 60)
    
    print("\nAlphaRank Rankings:")
    for i, (name, score) in enumerate(alpharank_rankings, 1):
        bar = "█" * int(score * 25)
        print(f"  {i}. {name}: {score:.4f}  {bar}")
    
    print("\nElo Rankings:")
    for i, (name, score) in enumerate(elo_rankings, 1):
        bar = "█" * int(score * 5)
        print(f"  {i}. {name}: {score:.2f}  {bar}")
    
    print("\n" + "=" * 60)
    print(f"🏆 WINNER (AlphaRank): {alpharank_rankings[0][0]}")
    print(f"🏆 WINNER (Elo): {elo_rankings[0][0]}")
    print("=" * 60)


def parse_args():
    parser = argparse.ArgumentParser(
        description="NLHF Tournament - Compare LLM models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare OpenAI models with heuristic arbiter
  python main.py --models gpt-4o-mini gpt-3.5-turbo --arbiter heuristic

  # Use config file with LLM arbiter  
  python main.py --config config/models.yaml --arbiter llm

  # Quick test with mock models
  python main.py --prompts 10 --matches 1
"""
    )
    
    # Models
    model_group = parser.add_argument_group("Models")
    model_group.add_argument(
        "--config", "-c",
        type=str,
        help="Path to models.yaml config file"
    )
    model_group.add_argument(
        "--models", "-m",
        nargs="+",
        help="List of OpenAI model names to compare"
    )
    model_group.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("OPENAI_API_KEY"),
        help="OpenAI API key (or set OPENAI_API_KEY env var)"
    )
    
    # Tournament
    tournament_group = parser.add_argument_group("Tournament")
    tournament_group.add_argument(
        "--prompts", "-p",
        type=int,
        default=50,
        help="Number of prompts (default: 50)"
    )
    tournament_group.add_argument(
        "--matches",
        type=int,
        default=2,
        help="Matches per model pair per prompt (default: 2)"
    )
    tournament_group.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Softmax temperature for AlphaRank (default: 1.0)"
    )
    tournament_group.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    
    # Arbiter
    arbiter_group = parser.add_argument_group("Arbiter")
    arbiter_group.add_argument(
        "--arbiter", "-a",
        choices=["heuristic", "llm", "hybrid"],
        default="heuristic",
        help="Arbiter type (default: heuristic)"
    )
    arbiter_group.add_argument(
        "--arbiter-model",
        type=str,
        default="gpt-4o-mini",
        help="Model for LLM arbiter (default: gpt-4o-mini)"
    )
    arbiter_group.add_argument(
        "--arbiter-url",
        type=str,
        default="https://api.openai.com/v1",
        help="Base URL for arbiter API"
    )
    
    # Output
    output_group = parser.add_argument_group("Output")
    output_group.add_argument(
        "--output", "-o",
        type=str,
        default="./output",
        help="Output directory (default: ./output)"
    )
    
    return parser.parse_args()


def main():
    # Load environment
    load_env()
    
    # Parse arguments
    args = parse_args()
    
    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    print("=" * 60)
    print("NLHF Tournament")
    print("=" * 60)
    
    # Create models
    print("\n📦 Loading models...")
    models = create_models_from_args(args)
    print(f"  ✓ Loaded {len(models)} models: {list(models.keys())}")
    
    # Load prompts
    print("\n📝 Loading prompts...")
    prompts = load_prompts("trl-lib/tldr", f"validation[:{args.prompts * 2}]")
    
    # Create arbiter
    print(f"\n⚖️  Creating {args.arbiter} arbiter...")
    arbiter = create_arbiter(
        arbiter_type=args.arbiter,
        api_key=args.api_key,
        base_url=args.arbiter_url,
        model=args.arbiter_model
    )
    print(f"  ✓ Arbiter ready")
    
    # Run tournament
    tournament = run_tournament(
        models=models,
        prompts=prompts,
        arbiter=arbiter,
        num_prompts=args.prompts,
        matches_per_pair=args.matches,
        alpha=args.alpha,
        log_dir=os.path.join(args.output, "logs")
    )
    
    # Save results
    save_results(tournament, args.output)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
