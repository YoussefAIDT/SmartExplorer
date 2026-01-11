"""Proximal Policy Optimization (PPO) pour l'exploration"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from config import Config
import random

class PPONetwork(nn.Module):
    """Réseau Actor-Critic pour PPO"""
    def __init__(self, state_size, action_size, hidden_size):
        super(PPONetwork, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.actor = nn.Linear(hidden_size, action_size)
        self.critic = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        shared_out = self.shared(x)
        action_probs = F.softmax(self.actor(shared_out), dim=-1)
        value = self.critic(shared_out)
        return action_probs, value

class PPOStrategy:
    """Stratégie PPO pour l'exploration"""
    def __init__(self, state_size=4, action_size=4):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = PPONetwork(state_size, action_size, Config.PPO_HIDDEN_SIZE).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=Config.PPO_LEARNING_RATE)
        
        self.episode_rewards = []
        self.episode_states = []
        self.episode_actions = []
        self.episode_log_probs = []
        self.episode_values = []
        self.episode_dones = []
    
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
        """Choisit une action selon la politique"""
        state_tensor = self._state_to_tensor(state, env)
        
        with torch.no_grad():
            action_probs, value = self.network(state_tensor)
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        
        action = action.item()
        
        if env:
            valid_actions = self._get_valid_actions(state, env)
            if action not in valid_actions and valid_actions:
                action = random.choice(valid_actions)
                state_tensor = self._state_to_tensor(state, env)
                action_probs, value = self.network(state_tensor)
                dist = torch.distributions.Categorical(action_probs)
                log_prob = dist.log_prob(torch.tensor(action).to(self.device))
        
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
    
    def store_transition(self, state, action, reward, next_state, done, env):
        """Stocke une transition pour l'entraînement"""
        state_tensor = self._state_to_tensor(state, env)
        
        with torch.no_grad():
            action_probs, value = self.network(state_tensor)
            dist = torch.distributions.Categorical(action_probs)
            log_prob = dist.log_prob(torch.tensor(action).to(self.device))
        
        self.episode_states.append(state_tensor)
        self.episode_actions.append(action)
        self.episode_log_probs.append(log_prob)
        self.episode_values.append(value.squeeze())
        self.episode_rewards.append(reward)
        self.episode_dones.append(done)
    
    def update(self, state, action, reward, next_state, done, env):
        """Stocke la transition (l'entraînement se fait à la fin de l'épisode)"""
        self.store_transition(state, action, reward, next_state, done, env)
    
    def train(self):
        """Entraîne le réseau PPO sur l'épisode complet"""
        if len(self.episode_states) == 0:
            return
        
        states = torch.cat(self.episode_states)
        actions = torch.tensor(self.episode_actions, dtype=torch.long).to(self.device)
        old_log_probs = torch.stack(self.episode_log_probs).detach()
        rewards = np.array(self.episode_rewards)
        dones = np.array(self.episode_dones)
        
        # Calculer les returns et advantages
        returns = []
        advantages = []
        G = 0
        for i in reversed(range(len(rewards))):
            G = rewards[i] + Config.DISCOUNT_FACTOR * G * (1 - dones[i])
            returns.insert(0, G)
        
        returns = torch.FloatTensor(returns).to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        old_values = torch.stack(self.episode_values).squeeze()
        advantages = returns - old_values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Entraînement PPO
        for epoch in range(Config.PPO_UPDATE_EPOCHS):
            indices = list(range(len(states)))
            random.shuffle(indices)
            
            for i in range(0, len(indices), Config.PPO_BATCH_SIZE):
                batch_indices = indices[i:i+Config.PPO_BATCH_SIZE]
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                
                action_probs, values = self.network(batch_states)
                dist = torch.distributions.Categorical(action_probs)
                new_log_probs = dist.log_prob(batch_actions)
                
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - Config.PPO_CLIP_EPSILON, 1 + Config.PPO_CLIP_EPSILON) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                critic_loss = F.mse_loss(values.squeeze(), batch_returns)
                
                entropy = dist.entropy().mean()
                
                loss = actor_loss + Config.PPO_VALUE_COEF * critic_loss - Config.PPO_ENTROPY_COEF * entropy
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
                self.optimizer.step()
        
        # Réinitialiser pour le prochain épisode
        self.episode_states = []
        self.episode_actions = []
        self.episode_log_probs = []
        self.episode_values = []
        self.episode_rewards = []
        self.episode_dones = []
    
    def decay_epsilon(self):
        """PPO n'utilise pas epsilon-greedy"""
        pass
    
    def save_policy(self, filename='ppo_model.pth'):
        """Sauvegarde le modèle"""
        torch.save({
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, filename)
        print(f"Modele PPO sauvegarde dans {filename}")
    
    def load_policy(self, filename='ppo_model.pth'):
        """Charge un modèle"""
        try:
            checkpoint = torch.load(filename, map_location=self.device)
            self.network.load_state_dict(checkpoint['network'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"Modele PPO charge depuis {filename}")
            return True
        except FileNotFoundError:
            print(f"Fichier {filename} non trouve")
            return False

