"""Programme principal - Smart Explorer avec cartographie et sauvegarde"""

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
from strategies.reinforcement import RLStrategy
from strategies.dqn import DQNStrategy
from strategies.ppo import PPOStrategy
from strategies.a2c import A2CStrategy
from strategies.sac import SACStrategy
from strategies.rainbow_dqn import RainbowDQNStrategy
from strategies.curiosity import CuriosityDQNStrategy
from utils.metrics import Metrics
from utils.advanced_metrics import AdvancedMetrics

def main():
    print("=" * 60)
    print("SMART EXPLORER - Lancement de la simulation 2D")
    print("=" * 60)
    
    # 1. Initialisation des composants
    Config.ensure_save_dir()
    env = GridWorld()
    # Génération de la carte (open world ou labyrinthe)
    MapGenerator.generate_map(env, Config.MAP_TYPE)
    
    renderer = Renderer(Config.GRID_SIZE)
    explorer = Explorer(env)
    
    # Selection de l'algorithme RL
    algorithm = Config.RL_ALGORITHM.upper()
    print(f"Algorithme RL selectionne: {algorithm}")
    print(f"Type de carte: {Config.MAP_TYPE}")
    
    if algorithm == 'DQN':
        rl_strategy = DQNStrategy()
    elif algorithm == 'PPO':
        rl_strategy = PPOStrategy()
    elif algorithm == 'A2C':
        rl_strategy = A2CStrategy()
    elif algorithm == 'SAC':
        rl_strategy = SACStrategy()
    elif algorithm == 'RAINBOW_DQN':
        rl_strategy = RainbowDQNStrategy()
    elif algorithm == 'CURIOSITY':
        rl_strategy = CuriosityDQNStrategy()
    else:
        rl_strategy = RLStrategy()
        algorithm = 'QLEARNING'
    
    metrics = Metrics()
    advanced_metrics = AdvancedMetrics()
    
    # 2. Boucle d'entraînement (Episodes)
    for episode in range(1, Config.NUM_EPISODES + 1):
        state = env.reset()
        explorer.memory.clear() # L'agent oublie la carte précédente
        
        # Compter les objets totaux pour les métriques
        total_items = sum(1 for y in range(env.grid_size) for x in range(env.grid_size) 
                         if env.grid[y][x] in ['GOLD', 'DIAMOND'])
        advanced_metrics.reset_episode(total_items)
        
        total_reward = 0
        steps = 0
        done = False
        
        # Reset position explorateur
        explorer.update_position(env.agent_pos)
        
        # --- BOUCLE DE JEU (STEPS) ---
        while not done and steps < Config.MAX_STEPS_PER_EPISODE:
            # Vérifier si pygame est toujours initialisé
            if not pygame.get_init() or not pygame.display.get_init():
                return
                
            # A. Gestion Quitter
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    renderer.close()
                    return
            
            # B. Perception & Mémorisation
            visible_cells = explorer.perceive()
            for pos, cell_type in visible_cells.items():
                explorer.memory.add_discovered(pos, cell_type)
            
            # C. Action (IA)
            action = rl_strategy.choose_action(state, env)
            next_state, reward, done, info = env.step(action)
            
            # D. Apprentissage (selon l'algorithme)
            if algorithm in ['PPO', 'A2C']:
                rl_strategy.update(state, action, reward, next_state, done, env)
            elif algorithm in ['DQN', 'SAC', 'RAINBOW_DQN', 'CURIOSITY']:
                rl_strategy.update(state, action, reward, next_state, done, env)
            else:
                rl_strategy.update(state, action, reward, next_state, done)
            
            # E. Mise à jour état
            if info['moved']:
                explorer.update_position(env.agent_pos)
            
            # Métriques avancées
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
            
            # F. Affichage (Render)
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
                'exploration_reward': info.get('exploration_reward', 0.0),
                'collect_reward': info.get('collect_reward', 0.0),
                'collected_items': len(env.collected_items)
            }
            
            # Note: Mettre show_full_map=True ici pour voir toute la carte (triche/debug)
            renderer.render(env, discovered_cells, stats, show_full_map=False)
            
            # Vitesse de simulation (plus petit = plus vite)
            # pygame.time.delay(10) 
        
        # --- FIN DE L'EPISODE ---
        
        # 1. Entraînement PPO/A2C à la fin de l'épisode
        if algorithm in ['PPO', 'A2C']:
            rl_strategy.train()
        
        # 2. Réduire l'exploration (Epsilon decay)
        rl_strategy.decay_epsilon()
        
        # 3. Calculer métriques avancées
        coverage = env.get_coverage() * 100
        advanced_metrics_result = advanced_metrics.finish_episode(
            steps, coverage, len(explorer.get_discovered_map()),
            len(env.collected_items), total_items
        )
        
        # 4. Enregistrer métriques
        metrics.record_episode(total_reward, steps, coverage, done)
        
        # 5. Log Console avec métriques avancées
        status = "COUVERTURE OK" if done else "INCOMPLET"
        if episode % 10 == 0 or done:
            print(f"[Ep {episode}] {status} | Reward: {total_reward:.1f} | Steps: {steps} | "
                  f"Coverage: {coverage:.1f}% | Efficiency: {advanced_metrics_result['efficiency']:.3f} | "
                  f"Redundancy: {advanced_metrics_result['redundancy']:.1f}% | "
                  f"Collect: {advanced_metrics_result['collect_rate']:.1f}%")
        
        # 4. SAUVEGARDE IMAGE & GESTION CARTE
        # On sauvegarde si c'est une victoire OU tous les 10 épisodes
        if done or episode % 10 == 0:
            renderer.save_map_image(env, discovered_cells, episode, stats)
            
            # Pause visuelle pour apprécier la victoire
            if done:
                renderer.show_episode_complete(env, discovered_cells, stats, duration=1500)
            
        # On change de carte à chaque épisode pour varier l'entraînement (comme demandé)
        print(f"Generation d'une nouvelle carte ({Config.MAP_TYPE}) pour l'épisode suivant...")
        MapGenerator.generate_map(env, Config.MAP_TYPE)

    # --- FIN DU PROGRAMME ---
    print("\n" + "=" * 60)
    print("Entrainement termine")
    print(f"Cartes sauvegardees dans : {Config.SAVE_MAP_DIR}")
    print("=" * 60)
    
    # Sauvegarder selon l'algorithme
    if algorithm == 'DQN':
        rl_strategy.save_policy('dqn_model_final.pth')
    elif algorithm == 'PPO':
        rl_strategy.save_policy('ppo_model_final.pth')
    elif algorithm == 'A2C':
        rl_strategy.save_policy('a2c_model_final.pth')
    elif algorithm == 'SAC':
        rl_strategy.save_policy('sac_model_final.pth')
    elif algorithm == 'RAINBOW_DQN':
        rl_strategy.save_policy('rainbow_dqn_model_final.pth')
    elif algorithm == 'CURIOSITY':
        rl_strategy.save_policy('curiosity_model_final.pth')
    else:
        rl_strategy.save_policy('q_table_final.pkl')
    
    # Attendre avant de fermer
    print("Appuyez sur ESPACE pour quitter.")
    waiting = True
    while waiting:
        if not pygame.get_init() or not pygame.display.get_init():
            break
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE):
                waiting = False
    
    if pygame.get_init() and pygame.display.get_init():
        renderer.close()

if __name__ == "__main__":
    main()