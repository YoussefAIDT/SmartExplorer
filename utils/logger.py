"""Système de logging avancé avec TensorBoard et WandB"""

import os
from datetime import datetime
import numpy as np

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("TensorBoard non disponible. Installez avec: pip install tensorboard")

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("WandB non disponible. Installez avec: pip install wandb")

class Logger:
    """Logger unifié pour TensorBoard et WandB"""
    
    def __init__(self, use_tensorboard=True, use_wandb=False, project_name="SmartExplorer", run_name=None):
        self.use_tensorboard = use_tensorboard and TENSORBOARD_AVAILABLE
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = run_name or f"run_{timestamp}"
        
        # TensorBoard
        if self.use_tensorboard:
            log_dir = f"runs/{run_name}"
            os.makedirs(log_dir, exist_ok=True)
            self.tb_writer = SummaryWriter(log_dir=log_dir)
            print(f"TensorBoard: logs dans {log_dir}")
        
        # WandB
        if self.use_wandb:
            wandb.init(project=project_name, name=run_name, reinit=True)
            print(f"WandB: projet {project_name}, run {run_name}")
        
        self.metrics_history = {
            'rewards': [],
            'coverage': [],
            'steps': [],
            'efficiency': [],
            'redundancy': [],
            'distance': [],
            'collect_rate': []
        }
    
    def log_scalar(self, tag, value, step):
        """Enregistre une valeur scalaire"""
        if self.use_tensorboard:
            self.tb_writer.add_scalar(tag, value, step)
        
        if self.use_wandb:
            wandb.log({tag: value}, step=step)
        
        # Stocker dans l'historique
        if tag in self.metrics_history:
            if len(self.metrics_history[tag]) <= step:
                self.metrics_history[tag].extend([None] * (step - len(self.metrics_history[tag]) + 1))
            self.metrics_history[tag][step] = value
    
    def log_episode(self, episode, metrics):
        """Enregistre les métriques d'un épisode"""
        for key, value in metrics.items():
            self.log_scalar(f"Episode/{key}", value, episode)
    
    def log_algorithm_comparison(self, algorithm_name, episode, metrics):
        """Enregistre les métriques pour comparaison d'algorithmes"""
        for key, value in metrics.items():
            self.log_scalar(f"Algorithm/{algorithm_name}/{key}", value, episode)
    
    def log_histogram(self, tag, values, step):
        """Enregistre un histogramme"""
        if self.use_tensorboard:
            self.tb_writer.add_histogram(tag, values, step)
        
        if self.use_wandb:
            wandb.log({tag: wandb.Histogram(values)}, step=step)
    
    def log_image(self, tag, image, step):
        """Enregistre une image"""
        if self.use_tensorboard:
            self.tb_writer.add_image(tag, image, step, dataformats='HWC')
        
        if self.use_wandb:
            wandb.log({tag: wandb.Image(image)}, step=step)
    
    def log_hyperparameters(self, hyperparams):
        """Enregistre les hyperparamètres"""
        if self.use_tensorboard:
            self.tb_writer.add_hparams(hyperparams, {})
        
        if self.use_wandb:
            wandb.config.update(hyperparams)
    
    def close(self):
        """Ferme les loggers"""
        if self.use_tensorboard:
            self.tb_writer.close()
        
        if self.use_wandb:
            wandb.finish()

