# ==================== strategies/reinforcement.py ====================
"""Stratégie d'exploration par Reinforcement Learning (Q-Learning) adaptée au labyrinthe"""

import random
import numpy as np
from config import Config

class RLStrategy:
    def __init__(self):
        self.q_table = {}
        self.epsilon = Config.EPSILON_START
        self.alpha = Config.LEARNING_RATE
        self.gamma = Config.DISCOUNT_FACTOR
        self.actions = [0, 1, 2, 3]  # UP, DOWN, LEFT, RIGHT
        self.action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    
    def get_q_value(self, state, action):
        """Obtient la Q-value pour un état-action"""
        return self.q_table.get((state, action), 0.0)
    
    def choose_action(self, state, env=None):
        """Choisit une action avec epsilon-greedy
        
        Args:
            state: État actuel (position)
            env: Environnement (optionnel) pour filtrer les actions valides
        """
        # Exploration
        if random.random() < self.epsilon:
            if env:
                # Choisir parmi les actions qui ne mènent pas à un mur
                valid_actions = self._get_valid_actions(state, env)
                if valid_actions:
                    return random.choice(valid_actions)
            return random.choice(self.actions)
        
        # Exploitation
        else:
            if env:
                valid_actions = self._get_valid_actions(state, env)
                if valid_actions:
                    q_values = [(a, self.get_q_value(state, a)) for a in valid_actions]
                else:
                    q_values = [(a, self.get_q_value(state, a)) for a in self.actions]
            else:
                q_values = [(a, self.get_q_value(state, a)) for a in self.actions]
            
            # Trouver la meilleure action
            max_q = max(q for _, q in q_values)
            best_actions = [a for a, q in q_values if q == max_q]
            return random.choice(best_actions)
    
    def _get_valid_actions(self, state, env):
        """Retourne les actions qui ne mènent pas directement à un mur ou obstacle
        
        CORRECTION: Filtrer strictement les actions qui passent par des obstacles
        """
        y, x = state
        valid = []
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        for action, (dy, dx) in enumerate(moves):
            ny, nx = y + dy, x + dx
            # CRITICAL: Vérifier que la nouvelle position est valide ET marchable
            # Ne jamais autoriser les murs
            if env.is_valid_position(ny, nx):
                cell_type = env.grid[ny][nx]
                # Murs ne doivent JAMAIS être passables
                if cell_type != 'WALL' and env.is_walkable(ny, nx):
                    valid.append(action)
        
        # Si aucune action valide, retourner toutes les actions
        # (l'environnement gèrera la collision)
        return valid if valid else self.actions
    
    def update(self, state, action, reward, next_state, done=False, env=None):
        """Met à jour la Q-table avec Q-Learning"""
        current_q = self.get_q_value(state, action)
        
        if done:
            # Si l'épisode est terminé, pas de future reward
            target = reward
        else:
            # Calculer la meilleure future Q-value
            max_next_q = max([self.get_q_value(next_state, a) for a in self.actions])
            target = reward + self.gamma * max_next_q
        
        # Mise à jour Q-Learning
        new_q = current_q + self.alpha * (target - current_q)
        self.q_table[(state, action)] = new_q
    
    def decay_epsilon(self):
        """Diminue le taux d'exploration"""
        self.epsilon = max(Config.EPSILON_MIN, self.epsilon * Config.EPSILON_DECAY)
    
    def get_policy(self, state):
        """Retourne la meilleure action selon la politique actuelle"""
        q_values = [self.get_q_value(state, a) for a in self.actions]
        max_q = max(q_values)
        best_actions = [a for a, q in zip(self.actions, q_values) if q == max_q]
        return random.choice(best_actions)
    
    def save_policy(self, filename='q_table.npy'):
        """Sauvegarde la Q-table"""
        import pickle
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)
        print(f"✓ Q-table sauvegardée dans {filename}")
    
    def load_policy(self, filename='q_table.npy'):
        """Charge une Q-table"""
        import pickle
        try:
            with open(filename, 'rb') as f:
                self.q_table = pickle.load(f)
            print(f"✓ Q-table chargée depuis {filename}")
            return True
        except FileNotFoundError:
            print(f"✗ Fichier {filename} non trouvé")
            return False