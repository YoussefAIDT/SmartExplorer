"""Script de test pour vérifier que l'agent ne passe pas par les obstacles"""

import sys
import numpy as np
from environment.grid_world import GridWorld
from environment.map_generator import MapGenerator
from agent.explorer import Explorer
from strategies.reinforcement import RLStrategy
from config import Config

def test_no_obstacle_collision():
    """Teste que l'agent ne passe pas par les murs"""
    print("=" * 60)
    print("TEST: L'agent ne doit pas passer par les obstacles")
    print("=" * 60)
    
    # Créer l'environnement
    env = GridWorld()
    MapGenerator.generate_maze_dfs(env)
    
    # Créer l'agent et la stratégie
    agent = Explorer(env)
    strategy = RLStrategy()
    
    # Exécuter un épisode de test
    state = env.reset()
    agent.memory.clear()
    total_steps = 0
    wall_collisions = 0
    danger_collisions = 0
    success_moves = 0
    
    print("\nDémarrage du test (200 pas)...")
    
    for step in range(200):
        # Choisir une action valide
        action = strategy.choose_action(state, env)
        
        # Exécuter l'action
        next_state, reward, done, info = env.step(action)
        
        # Vérifier les collisions
        if not info['moved']:
            cell_type = info['cell_type']
            reason = info.get('reason', 'unknown')
            
            if cell_type == 'WALL':
                wall_collisions += 1
                if wall_collisions <= 5:  # Afficher les 5 premiers
                    print(f"  [OK] Pas {step+1}: Collision avec MUR detectee (reward={reward}) - pas de mouvement")
            else:
                if step <= 5:
                    print(f"  [OK] Pas {step+1}: Collision bloquee - {cell_type}")
        else:
            # Vérifier que la cellule n'était pas un obstacle
            y, x = next_state
            cell_type = env.grid[y][x]
            if cell_type in ['WALL', 'WATER', 'LAVA']:
                print(f"  [ERROR] Pas {step+1}: Agent a passe par un obstacle !")
                print(f"    Position: {next_state}, Type: {cell_type}")
                return False
            success_moves += 1
        
        state = next_state
        total_steps += 1
        
        if done:
            print(f"  [SUCCESS] Episode termine au pas {step+1} (agent a atteint le but)")
            break
    
    print(f"\n[RESULTS]:")
    print(f"  - Pas totaux: {total_steps}")
    print(f"  - Mouvements reussis: {success_moves}")
    print(f"  - Collisions mur bloquees: {wall_collisions}")
    print(f"  - Aucun obstacle n'a ete traverse")
    return True

def test_obstacle_properties():
    """Vérifie les propriétés des obstacles dans la config"""
    print("\n" + "=" * 60)
    print("TEST: Verification des proprietes des obstacles")
    print("=" * 60)
    
    print("\nTypes de cellules et proprietes:")
    for cell_type, props in Config.CELL_TYPES.items():
        walkable = props['walkable']
        reward = props['reward']
        color = props['color']
        symbol = props['symbol']
        status = "[PASSABLE]" if walkable else "[BLOCKED]"
        print(f"  {status} {cell_type:10} - reward={reward:5.1f}, couleur={color}, symbole='{symbol}'")
    
    # Vérifier que les obstacles ne sont pas passables
    errors = []
    if Config.CELL_TYPES['WALL']['walkable']:
        errors.append("WALL doit avoir walkable=False")
    
    if errors:
        print("\n[ERROR] Proprietes incorrectes:")
        for error in errors:
            print(f"  - {error}")
        return False
    else:
        print("\n[SUCCESS] Toutes les proprietes sont correctes")
        return True

def test_grid_walkability():
    """Teste les fonctions de vérification de marchabilité"""
    print("\n" + "=" * 60)
    print("TEST: Fonctions is_walkable et is_valid_position")
    print("=" * 60)
    
    env = GridWorld()
    MapGenerator.generate_maze_dfs(env)
    
    # Test 1: Position hors limites
    if not env.is_valid_position(-1, 0) and not env.is_valid_position(0, -1):
        print("  [OK] is_valid_position() rejette les positions hors limites")
    else:
        print("  [ERROR] is_valid_position() ne fonctionne pas correctement")
        return False
    
    # Test 2: Position valide
    if env.is_valid_position(1, 1):
        print("  [OK] is_valid_position() accepte les positions valides")
    else:
        print("  [ERROR] is_valid_position() ne fonctionne pas correctement")
        return False
    
    # Test 3: Vérifier que les murs ne sont jamais walkable
    walls_blocked = True
    for y in range(env.grid_size):
        for x in range(env.grid_size):
            if env.grid[y][x] == 'WALL':
                if env.is_walkable(y, x):
                    walls_blocked = False
                    print(f"  [ERROR] WALL a {y},{x} est walkable (ne doit pas etre)")
    
    if walls_blocked:
        print("  [OK] is_walkable() rejette tous les murs")
    else:
        return False
    
    print("\n[SUCCESS] Toutes les fonctions fonctionnent correctement")
    return True

if __name__ == "__main__":
    all_passed = True
    
    # Exécuter les tests
    all_passed &= test_obstacle_properties()
    all_passed &= test_grid_walkability()
    all_passed &= test_no_obstacle_collision()
    
    print("\n" + "=" * 60)
    if all_passed:
        print("[SUCCESS] TOUS LES TESTS REUSSIS!")
        print("L'agent ne passe plus par les obstacles.")
        sys.exit(0)
    else:
        print("[ERROR] CERTAINS TESTS ONT ECHOUE")
        sys.exit(1)
