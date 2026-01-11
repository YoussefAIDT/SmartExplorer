"""Programme principal avancé - Comparaison des nouveaux algorithmes RL"""

import pygame
import sys
import io

# Fix encodage Windows pour l'affichage console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from config import Config
from environment.grid_world import GridWorld
from environment.map_generator import MapGenerator
from environment.renderer import Renderer
from agent.explorer import Explorer
from strategies.a2c import A2CStrategy
from strategies.sac import SACStrategy
from strategies.rainbow_dqn import RainbowDQNStrategy
from strategies.curiosity import CuriosityDQNStrategy
from strategies.dqn import DQNStrategy
from strategies.ppo import PPOStrategy
from utils.advanced_metrics import AdvancedMetrics
from utils.logger import Logger
from utils.algorithm_comparison import AlgorithmComparator


def create_strategy(algorithm_name):
    """Crée une stratégie selon le nom de l'algorithme (nouveaux algorithmes uniquement)"""
    algorithm_name = algorithm_name.upper()
    
    if algorithm_name == 'A2C':
        return A2CStrategy(), 'A2C'
    elif algorithm_name == 'SAC':
        return SACStrategy(), 'SAC'
    elif algorithm_name == 'RAINBOW_DQN':
        return RainbowDQNStrategy(), 'RAINBOW_DQN'
    elif algorithm_name == 'CURIOSITY':
        return CuriosityDQNStrategy(), 'CURIOSITY'
    elif algorithm_name == 'DQN':
        return DQNStrategy(), 'DQN'
    elif algorithm_name == 'PPO':
        return PPOStrategy(), 'PPO'
    else:
        print(f"Algorithme {algorithm_name} non reconnu, utilisation de A2C par defaut")
        return A2CStrategy(), 'A2C'


def train_algorithm(algorithm_name, env, renderer, logger, comparator, pregenerated_maps):
    """Entraîne un algorithme spécifique sur un set de cartes communes"""
    num_episodes = len(pregenerated_maps)
    print(f"Demarrage de l'entrainement pour {num_episodes} épisodes...")
    
    print(f"\n{'='*60}")
    print(f"ENTRAINEMENT: {algorithm_name}")
    print(f"{'='*60}")
    
    explorer = Explorer(env)
    rl_strategy, algorithm = create_strategy(algorithm_name)
    advanced_metrics = AdvancedMetrics()
    
    # Pour stocker les derniers metrics d'une algo
    last_advanced_metrics_result = {}

    for episode in range(1, num_episodes + 1):
        # Utiliser la carte commune pour cet épisode (Scientific Fairness)
        map_data = pregenerated_maps[episode-1]
        env.grid = map_data['grid'].copy()
        env.initial_pos = list(map_data['initial_pos'])
        env.map_type = map_data['map_type']
        
        # Recalculer les objets totaux
        total_items = sum(1 for y in range(env.grid_size) for x in range(env.grid_size) 
                         if env.grid[y][x] in ['GOLD', 'DIAMOND'])
        
        state = env.reset()
        explorer.memory.clear()
        advanced_metrics.reset_episode(total_items)
        
        total_reward = 0
        steps = 0
        done = False
        
        explorer.update_position(env.agent_pos)
        
        while not done and steps < Config.MAX_STEPS_PER_EPISODE:
            if not pygame.get_init() or not pygame.display.get_init():
                return None
                
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    # On continue l'entraînement même si la fenêtre est fermée (headless)
                    pass
            
            # Perception
            visible_cells = explorer.perceive()
            for pos, cell_type in visible_cells.items():
                explorer.memory.add_discovered(pos, cell_type)
            
            # Action
            action = rl_strategy.choose_action(state, env)
            next_state, reward, done, info = env.step(action)
            
            # Apprentissage
            rl_strategy.update(state, action, reward, next_state, done, env)
            
            # Métriques avancées (Essentiel pour le rapport final)
            discovered_map = explorer.get_discovered_map()
            advanced_metrics.update_step(
                env.agent_pos,
                info.get('exploration_reward', 0) > 0,
                discovered_map.keys(),
                env.collected_items
            )
            
            state = next_state
            total_reward += reward
            steps += 1
            explorer.update_position(env.agent_pos)
            
            # Rendu visuel
            discovered_cells = explorer.get_discovered_map()
            epsilon_val = getattr(rl_strategy, 'epsilon', 0.0)
            stats = {
                'episode': episode,
                'steps': steps,
                'total_reward': total_reward,
                'epsilon': epsilon_val,
                'discovered': len(discovered_cells),
                'coverage': env.get_coverage(),
                'success': done,
                'algorithm': algorithm,
                'collected_items': len(env.collected_items)
            }
            renderer.render(env, discovered_cells, stats, show_full_map=False)
        
        # Fin d'épisode
        if algorithm in ['A2C', 'PPO']:
            rl_strategy.train()
        
        rl_strategy.decay_epsilon()
        
        # Métriques avancées
        coverage = env.get_coverage() * 100
        last_advanced_metrics_result = advanced_metrics.finish_episode(
            steps, coverage, len(explorer.get_discovered_map()),
            len(env.collected_items), total_items
        )
        
        # Logging (Clés synchronisées avec AlgorithmComparator)
        episode_metrics = {
            'rewards': total_reward,
            'coverage': coverage,
            'steps': steps,
            'efficiency': last_advanced_metrics_result['efficiency'],
            'redundancy': last_advanced_metrics_result.get('redundancy', 0.0) if isinstance(last_advanced_metrics_result, dict) else 0.0,
            'distance': last_advanced_metrics_result.get('distance', 0.0) if isinstance(last_advanced_metrics_result, dict) else 0.0,
            'collect_rate': last_advanced_metrics_result.get('collect_rate', 0.0) if isinstance(last_advanced_metrics_result, dict) else 0.0
        }
        
        logger.log_algorithm_comparison(algorithm, episode, episode_metrics)
        comparator.record_episode(algorithm, episode_metrics)
        
        if episode % 10 == 0 or done:
            print(f"[{algorithm}] Ep {episode}: Reward {total_reward:.1f} | Cov {coverage:.1f}%")
            renderer.save_map_image(env, discovered_cells, episode, stats)
    
    # Sauvegarde finale
    rl_strategy.save_policy(f'{algorithm.lower()}_model.pth')
    return last_advanced_metrics_result


def main():
    print("=" * 80)
    print("SMART EXPLORER - VERSION AVANCEE (COMPARAISON)")
    print("Tous les Algorithmes: DQN, PPO, A2C, SAC, Rainbow DQN, Curiosity")
    print("=" * 80)
    
    # Configuration globale
    Config.NUM_EPISODES = 30 # Forcer 30 épisodes max
    
    # Initialisation
    Config.ensure_save_dir()
    env = GridWorld()
    renderer = Renderer(Config.GRID_SIZE)
    
    # Génération des cartes communes (Scientific Fairness)
    # 15 Maze + 15 Open World pour tester la généralité
    print(f"\nGénération de {Config.NUM_EPISODES} cartes (15 Maze + 15 Open World)...")
    common_maps = []
    
    # 15 Maze
    for i in range(15):
        MapGenerator.generate_map(env, 'maze')
        common_maps.append({
            'grid': env.grid.copy(),
            'initial_pos': list(env.initial_pos),
            'map_type': 'maze'
        })
        
    # 15 Open World
    for i in range(15):
        MapGenerator.generate_map(env, 'open_world')
        common_maps.append({
            'grid': env.grid.copy(),
            'initial_pos': list(env.initial_pos),
            'map_type': 'open_world'
        })
    
    # Logger
    logger = Logger(
        use_tensorboard=Config.USE_TENSORBOARD,
        use_wandb=Config.USE_WANDB,
        project_name=Config.WANDB_PROJECT
    )
    
    # Comparateur
    comparator = AlgorithmComparator()
    
    # Liste complète des algorithmes à comparer
    algorithms_to_compare = Config.ALGORITHMS_TO_COMPARE
    print(f"\nMODE COMPARAISON SCIENTIFIQUE")
    print(f"Algorithmes: {algorithms_to_compare}")
    print(f"Épisodes par algorithme: {Config.NUM_EPISODES} (Cartes identiques)")
    
    for alg_name in algorithms_to_compare:
        train_algorithm(alg_name, env, renderer, logger, comparator, common_maps)


    
    # Générer comparaison
    comparator.plot_comparison()
    comparator.print_summary()
    
    # Fermeture
    logger.close()
    
    print("\n" + "=" * 80)
    print("ENTRAINEMENT TERMINE")
    print("=" * 80)
    
    if Config.USE_TENSORBOARD:
        print(f"\nTensorBoard: lancez 'tensorboard --logdir=runs' pour visualiser")
    
    print("\nAppuyez sur ESPACE pour quitter.")
    waiting = True
    while waiting:
        if not pygame.get_init() or not pygame.display.get_init():
            break
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE):
                waiting = False
    
    renderer.close()

if __name__ == "__main__":
    main()

