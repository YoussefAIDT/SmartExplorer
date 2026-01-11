"""Système de comparaison d'algorithmes"""

import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import os
from datetime import datetime

class AlgorithmComparator:
    """Compare les performances de différents algorithmes"""
    
    def __init__(self):
        self.results = defaultdict(lambda: {
            'rewards': [],
            'coverage': [],
            'steps': [],
            'efficiency': [],
            'redundancy': [],
            'distance': [],
            'collect_rate': []
        })
        self.episode_counts = defaultdict(int)
    
    def record_episode(self, algorithm_name, metrics):
        """Enregistre les métriques d'un épisode"""
        episode = self.episode_counts[algorithm_name]
        self.episode_counts[algorithm_name] += 1
        
        for key, value in metrics.items():
            if key in self.results[algorithm_name]:
                self.results[algorithm_name][key].append(value)
    
    def get_average_metrics(self, algorithm_name, window=100):
        """Retourne les métriques moyennes sur une fenêtre"""
        if algorithm_name not in self.results:
            return {}
        
        metrics = {}
        for key, values in self.results[algorithm_name].items():
            if values:
                window_values = values[-window:] if len(values) >= window else values
                metrics[f'avg_{key}'] = np.mean(window_values)
                metrics[f'std_{key}'] = np.std(window_values)
                metrics[f'best_{key}'] = max(values) if values else 0
        
        return metrics
    
    def plot_comparison(self, save_dir="comparison_plots"):
        """Génère des graphiques de comparaison"""
        os.makedirs(save_dir, exist_ok=True)
        
        algorithms = list(self.results.keys())
        if not algorithms:
            print("Aucune donnée à comparer")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Comparaison des Algorithmes RL', fontsize=16, fontweight='bold')
        
        # 1. Récompenses
        ax = axes[0, 0]
        for alg in algorithms:
            rewards = self.results[alg]['rewards']
            if rewards:
                episodes = range(len(rewards))
                ax.plot(episodes, rewards, label=alg, alpha=0.6)
                if len(rewards) >= 10:
                    window = min(50, len(rewards) // 10)
                    moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
                    ax.plot(range(window-1, len(rewards)), moving_avg, label=f'{alg} (moy)', linewidth=2)
        ax.set_title('Récompenses par Épisode')
        ax.set_xlabel('Épisode')
        ax.set_ylabel('Récompense Totale')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Couverture
        ax = axes[0, 1]
        for alg in algorithms:
            coverage = self.results[alg]['coverage']
            if coverage:
                episodes = range(len(coverage))
                ax.plot(episodes, coverage, label=alg, alpha=0.6)
        ax.set_title('Couverture de la Carte')
        ax.set_xlabel('Épisode')
        ax.set_ylabel('Couverture (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Efficacité
        ax = axes[0, 2]
        for alg in algorithms:
            efficiency = self.results[alg]['efficiency']
            if efficiency:
                episodes = range(len(efficiency))
                ax.plot(episodes, efficiency, label=alg, alpha=0.6)
        ax.set_title('Efficacité d\'Exploration')
        ax.set_xlabel('Épisode')
        ax.set_ylabel('Cellules/Step')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Redondance
        ax = axes[1, 0]
        for alg in algorithms:
            redundancy = self.results[alg]['redundancy']
            if redundancy:
                episodes = range(len(redundancy))
                ax.plot(episodes, redundancy, label=alg, alpha=0.6)
        ax.set_title('Redondance (Revisites)')
        ax.set_xlabel('Épisode')
        ax.set_ylabel('Redondance (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 5. Taux de collecte
        ax = axes[1, 1]
        for alg in algorithms:
            collect_rate = self.results[alg]['collect_rate']
            if collect_rate:
                episodes = range(len(collect_rate))
                ax.plot(episodes, collect_rate, label=alg, alpha=0.6)
        ax.set_title('Taux de Collecte')
        ax.set_xlabel('Épisode')
        ax.set_ylabel('Objets Collectés (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 6. Comparaison finale
        ax = axes[1, 2]
        metrics_to_compare = ['rewards', 'coverage', 'efficiency', 'collect_rate']
        x = np.arange(len(metrics_to_compare))
        width = 0.8 / len(algorithms)
        
        for i, alg in enumerate(algorithms):
            values = []
            for metric in metrics_to_compare:
                if self.results[alg][metric]:
                    avg = np.mean(self.results[alg][metric][-100:]) if len(self.results[alg][metric]) >= 100 else np.mean(self.results[alg][metric])
                    values.append(avg)
                else:
                    values.append(0)
            
            # Normaliser pour comparaison
            values = np.array(values)
            if values.max() > 0:
                values = values / values.max() * 100
            
            ax.bar(x + i * width, values, width, label=alg, alpha=0.7)
        
        ax.set_title('Comparaison Finale (Normalisée)')
        ax.set_xlabel('Métriques')
        ax.set_ylabel('Score Normalisé (%)')
        ax.set_xticks(x + width * (len(algorithms) - 1) / 2)
        ax.set_xticklabels(metrics_to_compare)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{save_dir}/algorithm_comparison_{timestamp}.png"
        plt.savefig(filename, dpi=150)
        print(f"Graphiques de comparaison sauvegardes dans {filename}")
        plt.close()
    
    def print_summary(self):
        """Affiche un résumé textuel de la comparaison"""
        print("\n" + "=" * 80)
        print("RESUME DE LA COMPARAISON DES ALGORITHMES")
        print("=" * 80)
        
        for alg in sorted(self.results.keys()):
            metrics = self.get_average_metrics(alg, window=100)
            print(f"\n{alg}:")
            print(f"  Episodes: {self.episode_counts[alg]}")
            if metrics:
                print(f"  Recompense moyenne: {metrics.get('avg_rewards', 0):.2f} ± {metrics.get('std_rewards', 0):.2f}")
                print(f"  Couverture moyenne: {metrics.get('avg_coverage', 0):.2f}% ± {metrics.get('std_coverage', 0):.2f}%")
                print(f"  Efficacite moyenne: {metrics.get('avg_efficiency', 0):.4f}")
                print(f"  Redondance moyenne: {metrics.get('avg_redundancy', 0):.2f}%")
                print(f"  Taux de collecte: {metrics.get('avg_collect_rate', 0):.2f}%")
        
        print("=" * 80)

