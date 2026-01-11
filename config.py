"""Configuration globale du projet - VERSION CORRIGÉE"""

import os

class Config:
    # Environnement
    GRID_SIZE = 15
    CELL_SIZE = 40
    
    # Types de cellules avec rewards ET walkability
    CELL_TYPES = {
        'EMPTY': {
            'reward': -0.1, 
            'color': (245, 245, 245),  # Blanc cassé
            'symbol': ' ',
            'walkable': True,
            'description': 'Cellule vide'
        },
        'WALL': {
            'reward': -10, 
            'color': (60, 40, 20),  # Marron foncé style pierre
            'symbol': '#',
            'walkable': False,
            'description': 'Mur solide'
        },
        'WATER': {
            'reward': -15, 
            'color': (30, 100, 200),  # Bleu eau
            'symbol': '~',
            'walkable': False,
            'description': 'Eau profonde'
        },
        'LAVA': {
            'reward': -20, 
            'color': (255, 100, 0),  # Orange/rouge lave
            'symbol': '^',
            'walkable': False,
            'description': 'Lave mortelle'
        },
        'GOLD': {
            'reward': 10, 
            'color': (255, 215, 0),  # Or brillant
            'symbol': 'G',
            'walkable': True,
            'description': 'Trésor en or'
        },
        'DIAMOND': {
            'reward': 20, 
            'color': (0, 191, 255),  # Bleu diamant
            'symbol': 'D',
            'walkable': True,
            'description': 'Diamant précieux'
        },
        'TRAP': {
            'reward': -8, 
            'color': (139, 0, 0),  # Rouge foncé
            'symbol': 'X',
            'walkable': True,
            'description': 'Piège dangereux'
        },
    }


    
    # Agent
    AGENT_COLOR = (46, 204, 113)
    AGENT_VISION_RADIUS = 2
    COVERAGE_TARGET = 1.0  # Finir quand 100% est visité

    
    # Système de récompense combiné (Exploration + Collecte)
    # 1. Récompense d'exploration
    NEW_CELL_REWARD = 3.0  # Bonus pour découvrir une nouvelle cellule
    STEP_PENALTY = -0.05  # Pénalité par step pour encourager l'efficacité
    EXPLORATION_BONUS_SCALE = 1.0  # Multiplicateur pour les récompenses d'exploration
    
    # 2. Récompense de collecte (une seule fois par objet)
    COLLECT_REWARD_WEIGHT = 1.0  # Poids pour les récompenses de collecte
    GOLD_FIRST_TIME_REWARD = 15.0  # Bonus pour collecter de l'or (première fois)
    DIAMOND_FIRST_TIME_REWARD = 30.0  # Bonus pour collecter un diamant (première fois)
    TRAP_PENALTY = -5.0  # Pénalité pour marcher sur un piège
    
    # RL Algorithmes disponibles
    RL_ALGORITHM = 'DQN'  # Options: 'QLEARNING', 'DQN', 'PPO', 'A2C', 'SAC', 'RAINBOW_DQN', 'CURIOSITY'
    
    # Mode comparaison (entraîne plusieurs algorithmes)
    COMPARE_ALGORITHMS = False
    ALGORITHMS_TO_COMPARE = ['DQN', 'PPO', 'A2C', 'SAC', 'RAINBOW_DQN', 'CURIOSITY']
    
    # Logging
    USE_TENSORBOARD = True
    USE_WANDB = False
    WANDB_PROJECT = "SmartExplorer"
    
    # RL Hyperparamètres (Q-Learning)
    LEARNING_RATE = 0.1
    DISCOUNT_FACTOR = 0.95
    EPSILON_START = 1.0
    EPSILON_MIN = 0.1
    EPSILON_DECAY = 0.995
    
    # Hyperparamètres DQN
    DQN_HIDDEN_SIZE = 128
    DQN_BATCH_SIZE = 32
    DQN_REPLAY_BUFFER_SIZE = 10000
    DQN_TARGET_UPDATE_FREQ = 100
    DQN_LEARNING_RATE = 0.001
    
    # Hyperparamètres PPO
    PPO_HIDDEN_SIZE = 64
    PPO_LEARNING_RATE = 0.0003
    PPO_CLIP_EPSILON = 0.2
    PPO_ENTROPY_COEF = 0.01
    PPO_VALUE_COEF = 0.5
    PPO_UPDATE_EPOCHS = 4
    PPO_BATCH_SIZE = 64
    
    # Training
    MAX_STEPS_PER_EPISODE = 500
    NUM_EPISODES = 30
    RENDER_SPEED = 5  # ms (très rapide pour les tests)
    
    # Génération de map (ajusté pour variété)
    NUM_WALLS = (20, 30)  # (min, max)
    NUM_WATER = (5, 10)
    NUM_LAVA = (3, 8)
    NUM_GOLD = (8, 15)
    NUM_DIAMONDS = (3, 8)
    NUM_TRAPS = (5, 12)
    
    # Type de carte
    MAP_TYPE = 'maze'  # 'open_world' ou 'maze'
    
    # Sauvegarde de la carte
    SAVE_MAP_DIR = "saved_maps"
    SAVE_MAP_FORMAT = "png"
    
    @staticmethod
    def ensure_save_dir():
        """Crée le répertoire de sauvegarde s'il n'existe pas"""
        if not os.path.exists(Config.SAVE_MAP_DIR):
            os.makedirs(Config.SAVE_MAP_DIR)