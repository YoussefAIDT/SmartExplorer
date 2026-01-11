"""Deep Q-Network (DQN) pour l'exploration"""

import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
from config import Config

class DQNNetwork(nn.Module):
    """Réseau de neurones pour DQN"""
    def __init__(self, state_size, action_size, hidden_size):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

class DQNStrategy:
    """Stratégie DQN pour l'exploration"""
    def __init__(self, state_size=4, action_size=4):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=Config.DQN_REPLAY_BUFFER_SIZE)
        self.epsilon = Config.EPSILON_START
        self.epsilon_min = Config.EPSILON_MIN
        self.epsilon_decay = Config.EPSILON_DECAY
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = DQNNetwork(state_size, action_size, Config.DQN_HIDDEN_SIZE).to(self.device)
        self.target_network = DQNNetwork(state_size, action_size, Config.DQN_HIDDEN_SIZE).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=Config.DQN_LEARNING_RATE)
        
        self.update_target_network()
        self.step_count = 0
    
    def update_target_network(self):
        """Copie les poids du réseau principal vers le réseau cible"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def _state_to_tensor(self, state, env):
        """Convertit l'état en vecteur pour le réseau"""
        y, x = state
        grid_size = env.grid_size
        
        # Normaliser la position
        pos_y = y / grid_size
        pos_x = x / grid_size
        
        # Vérifier les voisins (obstacles)
        neighbors = []
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = y + dy, x + dx
            if env.is_valid_position(ny, nx) and env.is_walkable(ny, nx):
                neighbors.append(1.0)
            else:
                neighbors.append(0.0)
        
        # Retourner un vecteur de taille state_size
        state_vec = np.array([pos_y, pos_x] + neighbors[:self.state_size-2], dtype=np.float32)
        return torch.FloatTensor(state_vec).unsqueeze(0).to(self.device)
    
    def choose_action(self, state, env=None):
        """Choisit une action avec epsilon-greedy"""
        if random.random() < self.epsilon:
            if env:
                valid_actions = self._get_valid_actions(state, env)
                if valid_actions:
                    return random.choice(valid_actions)
            return random.randint(0, self.action_size - 1)
        
        with torch.no_grad():
            state_tensor = self._state_to_tensor(state, env)
            q_values = self.q_network(state_tensor)
            action = q_values.argmax().item()
            
            if env:
                valid_actions = self._get_valid_actions(state, env)
                if action not in valid_actions and valid_actions:
                    action = random.choice(valid_actions)
        
        return action
    
    def _get_valid_actions(self, state, env):
        """Retourne les actions valides"""
        y, x = state
        valid = []
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        for action, (dy, dx) in enumerate(moves):
            ny, nx = y + dy, x + dx
            if env.is_valid_position(ny, nx) and env.is_walkable(ny, nx):
                valid.append(action)
        
        return valid if valid else list(range(self.action_size))
    
    def remember(self, state, action, reward, next_state, done, env):
        """Stocke une transition dans le replay buffer"""
        state_tensor = self._state_to_tensor(state, env)
        next_state_tensor = self._state_to_tensor(next_state, env) if not done else None
        
        self.memory.append((state_tensor, action, reward, next_state_tensor, done))
    
    def replay(self, batch_size=None):
        """Entraîne le réseau sur un batch d'expériences"""
        if batch_size is None:
            batch_size = Config.DQN_BATCH_SIZE
        
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        
        # Préparer les tensors
        states_list = []
        next_states_list = []
        actions_list = []
        rewards_list = []
        dones_list = []
        
        for e in batch:
            states_list.append(e[0])
            actions_list.append(e[1])
            rewards_list.append(e[2])
            if e[3] is not None:
                next_states_list.append(e[3])
            else:
                next_states_list.append(torch.zeros_like(e[0]))
            dones_list.append(e[4])
        
        states = torch.cat(states_list)
        next_states = torch.cat(next_states_list)
        actions = torch.tensor(actions_list, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards_list, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones_list, dtype=torch.float32).to(self.device)
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (Config.DISCOUNT_FACTOR * next_q_values * (1 - dones))
        
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.step_count += 1
        if self.step_count % Config.DQN_TARGET_UPDATE_FREQ == 0:
            self.update_target_network()
    
    def update(self, state, action, reward, next_state, done, env):
        """Met à jour la stratégie avec une nouvelle transition"""
        self.remember(state, action, reward, next_state, done, env)
        
        if len(self.memory) > Config.DQN_BATCH_SIZE:
            self.replay()
    
    def decay_epsilon(self):
        """Diminue le taux d'exploration"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save_policy(self, filename='dqn_model.pth'):
        """Sauvegarde le modèle"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'epsilon': self.epsilon
        }, filename)
        print(f"Modele DQN sauvegarde dans {filename}")
    
    def load_policy(self, filename='dqn_model.pth'):
        """Charge un modèle"""
        try:
            checkpoint = torch.load(filename, map_location=self.device)
            self.q_network.load_state_dict(checkpoint['q_network'])
            self.target_network.load_state_dict(checkpoint['target_network'])
            self.epsilon = checkpoint.get('epsilon', self.epsilon)
            print(f"Modele DQN charge depuis {filename}")
            return True
        except FileNotFoundError:
            print(f"Fichier {filename} non trouve")
            return False

