"""Curiosity-driven exploration avec récompense intrinsèque"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from config import Config
from strategies.dqn import DQNStrategy, DQNNetwork
import random

class ForwardModel(nn.Module):
    """Modèle forward pour prédire le prochain état"""
    def __init__(self, state_size, action_size, hidden_size=64):
        super(ForwardModel, self).__init__()
        self.fc1 = nn.Linear(state_size + action_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, state_size)
    
    def forward(self, state, action_onehot):
        x = torch.cat([state, action_onehot], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class InverseModel(nn.Module):
    """Modèle inverse pour prédire l'action"""
    def __init__(self, state_size, action_size, hidden_size=64):
        super(InverseModel, self).__init__()
        self.fc1 = nn.Linear(state_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
    
    def forward(self, state, next_state):
        x = torch.cat([state, next_state], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.softmax(self.fc3(x), dim=-1)

class CuriosityDQNStrategy:
    """DQN avec récompense de curiosité intrinsèque"""
    def __init__(self, state_size=4, action_size=4):
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = Config.EPSILON_START
        self.epsilon_min = Config.EPSILON_MIN
        self.epsilon_decay = Config.EPSILON_DECAY
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = DQNNetwork(state_size, action_size, Config.DQN_HIDDEN_SIZE).to(self.device)
        self.target_network = DQNNetwork(state_size, action_size, Config.DQN_HIDDEN_SIZE).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=Config.DQN_LEARNING_RATE)
        
        # Modèles de curiosité
        self.forward_model = ForwardModel(state_size, action_size).to(self.device)
        self.inverse_model = InverseModel(state_size, action_size).to(self.device)
        self.curiosity_optimizer = optim.Adam(
            list(self.forward_model.parameters()) + list(self.inverse_model.parameters()),
            lr=0.001
        )
        
        self.memory = []
        self.step_count = 0
        self.curiosity_weight = 0.1  # Poids de la récompense intrinsèque
        
        self.update_target_network()
    
    def update_target_network(self):
        """Met à jour le réseau cible"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def _state_to_tensor(self, state, env):
        """Convertit l'état en vecteur"""
        y, x = state
        grid_size = env.grid_size
        
        pos_y = y / grid_size
        pos_x = x / grid_size
        
        neighbors = []
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = y + dy, x + dx
            if env.is_valid_position(ny, nx) and env.is_walkable(ny, nx):
                neighbors.append(1.0)
            else:
                neighbors.append(0.0)
        
        state_vec = np.array([pos_y, pos_x] + neighbors[:self.state_size-2], dtype=np.float32)
        return torch.FloatTensor(state_vec).unsqueeze(0).to(self.device)
    
    def _calculate_intrinsic_reward(self, state, action, next_state):
        """Calcule la récompense intrinsèque basée sur la surprise"""
        state_tensor = state.to(self.device)
        next_state_tensor = next_state.to(self.device)
        
        # One-hot action
        action_onehot = torch.zeros(self.action_size).to(self.device)
        action_onehot[action] = 1.0
        action_onehot = action_onehot.unsqueeze(0)
        
        # Prédire le prochain état
        predicted_next_state = self.forward_model(state_tensor, action_onehot)
        
        # Erreur de prédiction = surprise = curiosité
        prediction_error = torch.nn.functional.mse_loss(
            predicted_next_state, next_state_tensor, reduction='none'
        ).mean()
        
        intrinsic_reward = self.curiosity_weight * prediction_error.item()
        return intrinsic_reward
    
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
    
    def update(self, state, action, reward, next_state, done, env):
        """Met à jour avec récompense de curiosité"""
        state_tensor = self._state_to_tensor(state, env)
        next_state_tensor = self._state_to_tensor(next_state, env) if not done else None
        
        # Calculer récompense intrinsèque
        if next_state_tensor is not None:
            intrinsic_reward = self._calculate_intrinsic_reward(
                state_tensor, action, next_state_tensor
            )
            total_reward = reward + intrinsic_reward
        else:
            total_reward = reward
        
        # Stocker dans la mémoire
        self.memory.append((state_tensor, action, total_reward, next_state_tensor, done))
        
        # Entraîner les modèles de curiosité
        if len(self.memory) >= Config.DQN_BATCH_SIZE:
            self._train_curiosity()
            self._train_q_network()
    
    def _train_curiosity(self):
        """Entraîne les modèles forward et inverse"""
        if len(self.memory) < Config.DQN_BATCH_SIZE:
            return
        
        batch = random.sample(self.memory, Config.DQN_BATCH_SIZE)
        
        states = torch.cat([e[0] for e in batch])
        actions = torch.tensor([e[1] for e in batch], dtype=torch.long).to(self.device)
        next_states = torch.cat([e[3] if e[3] is not None else torch.zeros_like(e[0]) for e in batch])
        
        # Forward model
        action_onehot = torch.zeros(len(batch), self.action_size).to(self.device)
        action_onehot.scatter_(1, actions.unsqueeze(1), 1.0)
        predicted_next = self.forward_model(states, action_onehot)
        forward_loss = nn.MSELoss()(predicted_next, next_states)
        
        # Inverse model
        predicted_actions = self.inverse_model(states, next_states)
        inverse_loss = nn.CrossEntropyLoss()(predicted_actions, actions)
        
        curiosity_loss = forward_loss + inverse_loss
        
        self.curiosity_optimizer.zero_grad()
        curiosity_loss.backward()
        self.curiosity_optimizer.step()
    
    def _train_q_network(self):
        """Entraîne le réseau Q"""
        if len(self.memory) < Config.DQN_BATCH_SIZE:
            return
        
        batch = random.sample(self.memory, Config.DQN_BATCH_SIZE)
        
        states = torch.cat([e[0] for e in batch])
        actions = torch.tensor([e[1] for e in batch], dtype=torch.long).to(self.device)
        rewards = torch.tensor([e[2] for e in batch], dtype=torch.float32).to(self.device)
        next_states = torch.cat([e[3] if e[3] is not None else torch.zeros_like(e[0]) for e in batch])
        dones = torch.tensor([e[4] for e in batch], dtype=torch.float32).to(self.device)
        
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q = self.target_network(next_states).max(1)[0].detach()
        target_q = rewards + Config.DISCOUNT_FACTOR * next_q * (1 - dones)
        
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.step_count += 1
        if self.step_count % Config.DQN_TARGET_UPDATE_FREQ == 0:
            self.update_target_network()
    
    def decay_epsilon(self):
        """Diminue epsilon"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save_policy(self, filename='curiosity_dqn_model.pth'):
        """Sauvegarde le modèle"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'forward_model': self.forward_model.state_dict(),
            'inverse_model': self.inverse_model.state_dict(),
            'epsilon': self.epsilon
        }, filename)
        print(f"Modele Curiosity DQN sauvegarde dans {filename}")
    
    def load_policy(self, filename='curiosity_dqn_model.pth'):
        """Charge un modèle"""
        try:
            checkpoint = torch.load(filename, map_location=self.device)
            self.q_network.load_state_dict(checkpoint['q_network'])
            self.target_network.load_state_dict(checkpoint['target_network'])
            self.forward_model.load_state_dict(checkpoint['forward_model'])
            self.inverse_model.load_state_dict(checkpoint['inverse_model'])
            self.epsilon = checkpoint.get('epsilon', self.epsilon)
            print(f"Modele Curiosity DQN charge depuis {filename}")
            return True
        except FileNotFoundError:
            print(f"Fichier {filename} non trouve")
            return False

