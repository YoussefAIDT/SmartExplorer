"""Métriques avancées pour l'évaluation de l'exploration"""

import numpy as np
from collections import deque

class AdvancedMetrics:
    """Calcule des métriques avancées pour l'exploration"""
    
    def __init__(self, window_size=100):
        self.window_size = window_size
        
        # Historique des épisodes
        self.episode_rewards = []
        self.episode_steps = []
        self.episode_coverage = []
        self.episode_discovered = []
        self.episode_collected = []
        self.episode_paths = []
        
        # Métriques par épisode
        self.current_visited = set()
        self.current_path = []
        self.current_discovered = set()
        self.current_collected = set()
        self.total_items = 0
    
    def reset_episode(self, total_items=0):
        """Réinitialise les métriques pour un nouvel épisode"""
        self.current_visited = set()
        self.current_path = []
        self.current_discovered = set()
        self.current_collected = set()
        self.total_items = total_items
    
    def update_step(self, position, is_new_cell, discovered_cells, collected_items):
        """Met à jour les métriques à chaque step"""
        pos_tuple = tuple(position)
        
        # Chemin parcouru (on ajoute chaque pas, même les revisites, pour la redondance)
        self.current_path.append(pos_tuple)
        self.current_visited.add(pos_tuple)
        
        # Cellules découvertes
        for cell_pos in discovered_cells:
            self.current_discovered.add(cell_pos)
        
        # Objets collectés
        self.current_collected = collected_items.copy()
    
    def calculate_efficiency(self, steps, discovered_count):
        """Calcule l'efficacité d'exploration (cellules découvertes par step)"""
        if steps == 0:
            return 0.0
        return discovered_count / steps
    
    def calculate_redundancy(self, visited_count, path_length):
        """Calcule le pourcentage de revisites"""
        if path_length == 0:
            return 0.0
        unique_visits = len(set(self.current_path))
        if unique_visits == 0:
            return 100.0
        redundancy = ((path_length - unique_visits) / path_length) * 100
        return max(0.0, redundancy)
    
    def calculate_distance(self, path):
        """Calcule la distance totale parcourue (Manhattan)"""
        if len(path) < 2:
            return 0.0
        
        total_distance = 0.0
        for i in range(1, len(path)):
            y1, x1 = path[i-1]
            y2, x2 = path[i]
            distance = abs(y2 - y1) + abs(x2 - x1)
            total_distance += distance
        
        return total_distance
    
    def calculate_path_efficiency(self, path, coverage):
        """Calcule l'efficacité du chemin (couverture / distance)"""
        distance = self.calculate_distance(path)
        if distance == 0:
            return 0.0
        return coverage / distance
    
    def calculate_collect_rate(self, collected_count, total_items):
        """Calcule le taux de collecte"""
        if total_items == 0:
            return 0.0
        return (collected_count / total_items) * 100
    
    def finish_episode(self, steps, coverage, discovered_count, collected_count, total_items):
        """Finalise les métriques d'un épisode"""
        # Efficacité d'exploration
        efficiency = self.calculate_efficiency(steps, discovered_count)
        
        # Redondance
        redundancy = self.calculate_redundancy(len(self.current_visited), len(self.current_path))
        
        # Distance parcourue
        distance = self.calculate_distance(self.current_path)
        
        # Efficacité du chemin
        path_efficiency = self.calculate_path_efficiency(self.current_path, coverage)
        
        # Taux de collecte
        collect_rate = self.calculate_collect_rate(collected_count, total_items)
        
        # Stocker les métriques
        metrics = {
            'efficiency': efficiency,
            'redundancy': redundancy,
            'distance': distance,
            'path_efficiency': path_efficiency,
            'collect_rate': collect_rate,
            'discovered_count': discovered_count,
            'collected_count': collected_count
        }
        
        return metrics
    
    def get_moving_average(self, metric_name, window=None):
        """Retourne la moyenne mobile d'une métrique"""
        if window is None:
            window = self.window_size
        
        if metric_name not in self.episode_rewards:
            return 0.0
        
        values = getattr(self, f'episode_{metric_name}', [])
        if len(values) < window:
            return np.mean(values) if values else 0.0
        
        return np.mean(values[-window:])
    
    def get_summary(self):
        """Retourne un résumé des métriques"""
        if not self.episode_rewards:
            return {}
        
        return {
            'avg_reward': np.mean(self.episode_rewards[-self.window_size:]),
            'avg_steps': np.mean(self.episode_steps[-self.window_size:]),
            'avg_coverage': np.mean(self.episode_coverage[-self.window_size:]),
            'best_coverage': max(self.episode_coverage) if self.episode_coverage else 0.0,
            'total_episodes': len(self.episode_rewards)
        }

