"""Classe principale de l'environnement Grid World - VERSION CORRIGÉE"""

import numpy as np
from config import Config

class GridWorld:
    def __init__(self, grid_size=None):
        self.grid_size = grid_size or Config.GRID_SIZE
        self.grid = None
        self.agent_pos = [1, 1]
        self.initial_pos = [1, 1]
        self.visited = set()
        self.walkable_count = 1
        self.collected_items = set()  # Suivi des objets collectés (position)
        self.step_count = 0
        self.map_type = Config.MAP_TYPE
        self.reset()
    
    def reset(self):
        """Réinitialise l'environnement sans régénérer la carte"""
        if self.grid is None:
            self.grid = np.full((self.grid_size, self.grid_size), 'EMPTY', dtype=object)
        
        self.agent_pos = list(self.initial_pos)
        # S'assurer que la case de départ est marchable
        self.grid[self.agent_pos[0]][self.agent_pos[1]] = 'EMPTY'
        self.visited = {tuple(self.agent_pos)}
        self.collected_items = set()
        self.step_count = 0
        self.walkable_count = self._count_walkable_cells()
        return self.get_state()
    
    def get_state(self):
        """Retourne l'état actuel (position de l'agent)"""
        return tuple(self.agent_pos)
    
    def step(self, action):
        """Exécute une action et retourne (next_state, reward, done, info)
        
        CORRECTION PRINCIPALE : Le robot ne peut JAMAIS traverser les obstacles non-walkable
        """
        # Actions: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        dy, dx = moves[action]
        
        new_y = self.agent_pos[0] + dy
        new_x = self.agent_pos[1] + dx
        
        # Vérifier les limites
        if not self.is_valid_position(new_y, new_x):
            return self.get_state(), -5, False, {
                'cell_type': 'BOUNDARY', 
                'moved': False,
                'coverage': self.get_coverage()
            }
        
        cell_type = self.grid[new_y][new_x]
        
        # ❌ BLOQUER TOUS LES OBSTACLES NON-WALKABLE
        if not Config.CELL_TYPES[cell_type]['walkable']:
            # Le robot NE BOUGE PAS et reçoit une pénalité
            penalty = Config.CELL_TYPES[cell_type]['reward']
            return self.get_state(), penalty, False, {
                'cell_type': cell_type, 
                'moved': False, 
                'reason': f'blocked_by_{cell_type.lower()}',
                'coverage': self.get_coverage()
            }
        
        # ✅ MOUVEMENT VALIDE
        # La cellule précédente reste EMPTY ou son type d'origine
        
        # Déplacer l'agent

        self.agent_pos = [new_y, new_x]
        self.step_count += 1
        
        # ===== CALCUL DES RÉCOMPENSES COMBINÉES =====
        reward = 0.0
        
        # 1. RÉCOMPENSE D'EXPLORATION
        is_new_cell = (new_y, new_x) not in self.visited
        exploration_reward = 0.0
        if is_new_cell:
            # Bonus pour découvrir une nouvelle cellule
            exploration_reward = Config.NEW_CELL_REWARD * Config.EXPLORATION_BONUS_SCALE
            reward += exploration_reward
            self.visited.add((new_y, new_x))
        
        # Pénalité par step pour encourager l'efficacité
        step_penalty = Config.STEP_PENALTY
        reward += step_penalty
        
        # 2. RÉCOMPENSE DE COLLECTE (une seule fois par objet)
        collect_reward = 0.0
        cell_pos = (new_y, new_x)
        
        if cell_type == 'GOLD' and cell_pos not in self.collected_items:
            collect_reward = Config.GOLD_FIRST_TIME_REWARD
            self.collected_items.add(cell_pos)
        elif cell_type == 'DIAMOND' and cell_pos not in self.collected_items:
            collect_reward = Config.DIAMOND_FIRST_TIME_REWARD
            self.collected_items.add(cell_pos)
        elif cell_type == 'TRAP':
            # Piège : pénalité même si déjà visité
            collect_reward = Config.TRAP_PENALTY
        elif cell_type in ['GOLD', 'DIAMOND'] and cell_pos in self.collected_items:
            # Objet déjà collecté : pas de récompense
            collect_reward = 0.0
        else:
            # Autres cellules (EMPTY, etc.) : récompense de base
            collect_reward = Config.CELL_TYPES.get(cell_type, {}).get('reward', 0.0)

        
        reward += collect_reward * Config.COLLECT_REWARD_WEIGHT
        
        coverage = self.get_coverage()
        
        # Terminaison basée sur la couverture
        done = coverage >= Config.COVERAGE_TARGET
        
        return self.get_state(), reward, done, {
            'cell_type': cell_type, 
            'moved': True,
            'coverage': coverage,
            'exploration_reward': exploration_reward,
            'collect_reward': collect_reward,
            'step_penalty': step_penalty,
            'total_reward': reward
        }
    
    def get_cell_type(self, y, x):
        """Retourne le type de cellule à la position (y, x)"""
        if self.is_valid_position(y, x):
            return self.grid[y][x]
        return None
    
    def is_valid_position(self, y, x):
        """Vérifie si une position est dans les limites"""
        return 0 <= y < self.grid_size and 0 <= x < self.grid_size
    
    def is_walkable(self, y, x):
        """Vérifie si une position est marchable"""
        if not self.is_valid_position(y, x):
            return False
        cell_type = self.grid[y][x]
        return Config.CELL_TYPES[cell_type]['walkable']
    
    def get_walkable_neighbors(self, y, x):
        """Retourne les voisins marchables"""
        neighbors = []
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_y, new_x = y + dy, x + dx
            if self.is_walkable(new_y, new_x):
                neighbors.append((new_y, new_x))
        return neighbors

    def get_coverage(self):
        """Retourne le pourcentage de cellules marchables déjà visitées"""
        if self.walkable_count == 0:
            return 0.0
        return len(self.visited) / self.walkable_count

    def _count_walkable_cells(self):
        """Compte le nombre de cellules marchables sur la carte courante"""
        count = 0
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                cell_type = self.grid[y][x]
                if Config.CELL_TYPES[cell_type]['walkable']:
                    count += 1
        return max(count, 1)