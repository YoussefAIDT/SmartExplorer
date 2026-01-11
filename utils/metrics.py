# ==================== utils/metrics.py ====================
"""Calcul des métriques de performance pour navigation en labyrinthe"""

class Metrics:
    def __init__(self):
        self.episode_rewards = []
        self.episode_steps = []
        self.coverage_history = []
        self.success_history = []
    
    def record_episode(self, total_reward, steps, coverage, success=False):
        """Enregistre les métriques d'un épisode"""
        self.episode_rewards.append(total_reward)
        self.episode_steps.append(steps)
        self.coverage_history.append(coverage)
        self.success_history.append(1 if success else 0)
    
    def get_average_reward(self, last_n=10):
        """Retourne la récompense moyenne des N derniers épisodes"""
        if not self.episode_rewards:
            return 0
        return sum(self.episode_rewards[-last_n:]) / min(last_n, len(self.episode_rewards))
    
    def get_success_rate(self, last_n=10):
        """Retourne le taux de succès des N derniers épisodes"""
        if not self.success_history:
            return 0
        recent = self.success_history[-last_n:]
        return (sum(recent) / len(recent)) * 100
    
    def get_coverage_rate(self, discovered, total):
        """Calcule le taux de couverture"""
        return (discovered / total) * 100 if total > 0 else 0
    
    def print_summary(self, episode):
        """Affiche un résumé des métriques"""
        if len(self.episode_rewards) > 0:
            avg_reward = self.get_average_reward(10)
            avg_steps = sum(self.episode_steps[-10:]) / min(10, len(self.episode_steps))
            success_rate = self.get_success_rate(10)
            
            print("\n" + "=" * 60)
            print(f"   Episode {episode} - Performance Summary")
            print("=" * 60)
            print(f"  Avg Reward (last 10):    {avg_reward:>8.2f}")
            print(f"  Avg Steps (last 10):     {avg_steps:>8.2f}")
            print(f"  Success Rate (last 10):  {success_rate:>7.1f}%")
            print(f"  Current Coverage:        {self.coverage_history[-1]:>7.1f}%")
            print(f"  Total Successes:         {sum(self.success_history):>8d}/{len(self.success_history)}")
            print("=" * 60)