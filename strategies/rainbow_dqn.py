"""Rainbow DQN - Combinaison d'améliorations DQN"""

import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
from config import Config
from strategies.dqn import DQNNetwork

class PrioritizedReplayBuffer:
    """Replay buffer avec priorités"""
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros(capacity)
        self.position = 0
    
    def add(self, experience):
        """Ajoute une expérience avec priorité maximale"""
        max_priority = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size, beta=0.4):
        """Échantillonne un batch avec priorités"""
        if len(self.buffer) == 0:
            return None, None, None
        
        priorities = self.priorities[:len(self.buffer)]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        experiences = [self.buffer[idx] for idx in indices]
        
        weights = (len(self.buffer) * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        
        return experiences, indices, weights
    
    def update_priorities(self, indices, td_errors):
        """Met à jour les priorités"""
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = abs(td_error) + 1e-6

class DuelingDQN(nn.Module):
    """Réseau Dueling DQN - sépare Value et Advantage streams"""
    def __init__(self, state_size, action_size, hidden_size):
        super(DuelingDQN, self).__init__()
        # Couche partagée (feature layer)
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        
        # Value stream - estime V(s)
        self.value_fc = nn.Linear(hidden_size, hidden_size // 2)
        self.value = nn.Linear(hidden_size // 2, 1)
        
        # Advantage stream - estime A(s,a)
        self.advantage_fc = nn.Linear(hidden_size, hidden_size // 2)
        self.advantage = nn.Linear(hidden_size // 2, action_size)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # Couches partagées
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        
        # Value stream
        value = self.relu(self.value_fc(x))
        value = self.value(value)
        
        # Advantage stream
        advantage = self.relu(self.advantage_fc(x))
        advantage = self.advantage(advantage)
        
        # Combiner: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values

class RainbowDQNStrategy:
    """Rainbow DQN avec plusieurs améliorations"""
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
        
        self.replay_buffer = PrioritizedReplayBuffer(Config.DQN_REPLAY_BUFFER_SIZE)
        self.update_target_network()
        self.step_count = 0
        
        # Double DQN
        self.use_double_dqn = True
        
        # Dueling DQN
        self.use_dueling = True
        if self.use_dueling:
            self.q_network = DuelingDQN(state_size, action_size, Config.DQN_HIDDEN_SIZE).to(self.device)
            self.target_network = DuelingDQN(state_size, action_size, Config.DQN_HIDDEN_SIZE).to(self.device)
            self.optimizer = optim.Adam(self.q_network.parameters(), lr=Config.DQN_LEARNING_RATE)
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
        """Stocke une transition"""
        state_tensor = self._state_to_tensor(state, env)
        next_state_tensor = self._state_to_tensor(next_state, env) if not done else None
        
        self.replay_buffer.add((state_tensor, action, reward, next_state_tensor, done))
    
    def replay(self):
        """Entraîne avec priorités"""
        if len(self.replay_buffer.buffer) < Config.DQN_BATCH_SIZE:
            return
        
        experiences, indices, weights = self.replay_buffer.sample(Config.DQN_BATCH_SIZE)
        if experiences is None:
            return
        
        states = torch.cat([e[0] for e in experiences])
        actions = torch.tensor([e[1] for e in experiences], dtype=torch.long).to(self.device)
        rewards = torch.tensor([e[2] for e in experiences], dtype=torch.float32).to(self.device)
        next_states = torch.cat([e[3] if e[3] is not None else torch.zeros_like(e[0]) for e in experiences])
        dones = torch.tensor([e[4] for e in experiences], dtype=torch.float32).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        if self.use_double_dqn:
            next_actions = self.q_network(next_states).argmax(1, keepdim=True)
            next_q_values = self.target_network(next_states).gather(1, next_actions)
        else:
            next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
        
        target_q_values = rewards.unsqueeze(1) + Config.DISCOUNT_FACTOR * next_q_values * (1 - dones.unsqueeze(1))
        
        td_errors = (target_q_values - current_q_values).squeeze().detach().cpu().numpy()
        self.replay_buffer.update_priorities(indices, td_errors)
        
        loss = (weights.unsqueeze(1) * (current_q_values - target_q_values.detach()) ** 2).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.step_count += 1
        if self.step_count % Config.DQN_TARGET_UPDATE_FREQ == 0:
            self.update_target_network()
    
    def update(self, state, action, reward, next_state, done, env):
        """Met à jour la stratégie"""
        self.remember(state, action, reward, next_state, done, env)
        self.replay()
    
    def decay_epsilon(self):
        """Diminue epsilon"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save_policy(self, filename='rainbow_dqn_model.pth'):
        """Sauvegarde le modèle"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'epsilon': self.epsilon
        }, filename)
        print(f"Modele Rainbow DQN sauvegarde dans {filename}")
    
    def load_policy(self, filename='rainbow_dqn_model.pth'):
        """Charge un modèle"""
        try:
            checkpoint = torch.load(filename, map_location=self.device)
            self.q_network.load_state_dict(checkpoint['q_network'])
            self.target_network.load_state_dict(checkpoint['target_network'])
            self.epsilon = checkpoint.get('epsilon', self.epsilon)
            print(f"Modele Rainbow DQN charge depuis {filename}")
            return True
        except FileNotFoundError:
            print(f"Fichier {filename} non trouve")
            return False

