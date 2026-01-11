"""SAC (Soft Actor-Critic) pour l'exploration continue"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from config import Config
import random

class SACActor(nn.Module):
    """Actor pour SAC"""
    def __init__(self, state_size, action_size, hidden_size):
        super(SACActor, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.mean = nn.Linear(hidden_size, action_size)
        self.log_std = nn.Linear(hidden_size, action_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = torch.clamp(self.log_std(x), -20, 2)
        return mean, log_std
    
    def sample(self, x):
        """Échantillonne une action"""
        mean, log_std = self.forward(x)
        std = torch.exp(log_std)
        normal = torch.distributions.Normal(mean, std)
        action = normal.rsample()
        action = torch.softmax(action, dim=-1)
        log_prob = normal.log_prob(action).sum(dim=-1, keepdim=True)
        return action, log_prob

class SACCritic(nn.Module):
    """Critic pour SAC"""
    def __init__(self, state_size, action_size, hidden_size):
        super(SACCritic, self).__init__()
        self.fc1 = nn.Linear(state_size + action_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class SACStrategy:
    """Stratégie SAC pour l'exploration"""
    def __init__(self, state_size=4, action_size=4):
        self.state_size = state_size
        self.action_size = action_size
        self.alpha = 0.2  # Temperature parameter
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = SACActor(state_size, action_size, Config.PPO_HIDDEN_SIZE).to(self.device)
        self.critic1 = SACCritic(state_size, action_size, Config.PPO_HIDDEN_SIZE).to(self.device)
        self.critic2 = SACCritic(state_size, action_size, Config.PPO_HIDDEN_SIZE).to(self.device)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=Config.PPO_LEARNING_RATE)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=Config.PPO_LEARNING_RATE)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=Config.PPO_LEARNING_RATE)
        
        self.memory = []
        self.batch_size = Config.PPO_BATCH_SIZE
    
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
        """Choisit une action"""
        state_tensor = self._state_to_tensor(state, env)
        
        with torch.no_grad():
            action_probs, _ = self.actor.sample(state_tensor)
            action = torch.multinomial(action_probs, 1).item()
        
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
        """Stocke et entraîne"""
        state_tensor = self._state_to_tensor(state, env)
        next_state_tensor = self._state_to_tensor(next_state, env) if not done else None
        
        # One-hot encoding de l'action
        action_onehot = torch.zeros(self.action_size).to(self.device)
        action_onehot[action] = 1.0
        
        self.memory.append((state_tensor, action_onehot, reward, next_state_tensor, done))
        
        if len(self.memory) >= self.batch_size:
            self._train()
    
    def _train(self):
        """Entraîne les réseaux"""
        batch = random.sample(self.memory, min(self.batch_size, len(self.memory)))
        
        states = torch.cat([e[0] for e in batch])
        actions = torch.stack([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.cat([e[3] if e[3] is not None else torch.zeros_like(e[0]) for e in batch])
        dones = torch.FloatTensor([e[4] for e in batch]).to(self.device)
        
        # Critic loss
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            q1_next = self.critic1(next_states, next_actions)
            q2_next = self.critic2(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_probs
            target_q = rewards + Config.DISCOUNT_FACTOR * (1 - dones) * q_next.squeeze()
        
        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)
        critic1_loss = F.mse_loss(q1.squeeze(), target_q)
        critic2_loss = F.mse_loss(q2.squeeze(), target_q)
        
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        # Actor loss
        new_actions, log_probs = self.actor.sample(states)
        q1_new = self.critic1(states, new_actions)
        q2_new = self.critic2(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * log_probs - q_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
    
    def decay_epsilon(self):
        """SAC n'utilise pas epsilon-greedy"""
        pass
    
    def save_policy(self, filename='sac_model.pth'):
        """Sauvegarde le modèle"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict()
        }, filename)
        print(f"Modele SAC sauvegarde dans {filename}")
    
    def load_policy(self, filename='sac_model.pth'):
        """Charge un modèle"""
        try:
            checkpoint = torch.load(filename, map_location=self.device)
            self.actor.load_state_dict(checkpoint['actor'])
            self.critic1.load_state_dict(checkpoint['critic1'])
            self.critic2.load_state_dict(checkpoint['critic2'])
            print(f"Modele SAC charge depuis {filename}")
            return True
        except FileNotFoundError:
            print(f"Fichier {filename} non trouve")
            return False

