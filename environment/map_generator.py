"""Génération procédurale de labyrinthes - VERSION CORRIGÉE AVEC OBSTACLES VARIÉS"""

import random
import numpy as np
from config import Config
from collections import deque

class MapGenerator:
    @staticmethod
    def generate_open_world(env):
        """Génère une carte open world avec plusieurs zones de départ et chemins multiples"""
        grid_size = env.grid_size
        max_attempts = 50
        attempt = 0
        
        while attempt < max_attempts:
            attempt += 1
            
            # 1. Créer une carte avec beaucoup d'espace ouvert
            env.grid = np.full((grid_size, grid_size), 'EMPTY', dtype=object)
            
            # 2. Ajouter des murs extérieurs
            env.grid[0, :] = 'WALL'
            env.grid[grid_size - 1, :] = 'WALL'
            env.grid[:, 0] = 'WALL'
            env.grid[:, grid_size - 1] = 'WALL'
            
            # 3. Créer des "régions" ouvertes avec quelques obstacles
            num_regions = random.randint(3, 6)
            regions = []
            
            for _ in range(num_regions):
                # Choisir une zone pour la région
                region_size = random.randint(3, 6)
                start_y = random.randint(2, grid_size - region_size - 2)
                start_x = random.randint(2, grid_size - region_size - 2)
                
                # Créer une zone ouverte (carré ou rectangle)
                for y in range(start_y, min(start_y + region_size, grid_size - 1)):
                    for x in range(start_x, min(start_x + region_size, grid_size - 1)):
                        if env.grid[y][x] == 'EMPTY':
                            env.grid[y][x] = 'EMPTY'
                            regions.append((y, x))
            
            # 4. Connecter les régions avec des chemins
            if len(regions) > 1:
                for i in range(len(regions) - 1):
                    y1, x1 = regions[i]
                    y2, x2 = regions[i + 1]
                    
                    # Créer un chemin entre les deux régions
                    current_y, current_x = y1, x1
                    while (current_y, current_x) != (y2, x2):
                        if random.random() < 0.5:
                            if current_y < y2:
                                current_y += 1
                            elif current_y > y2:
                                current_y -= 1
                        else:
                            if current_x < x2:
                                current_x += 1
                            elif current_x > x2:
                                current_x -= 1
                        
                        if 1 <= current_y < grid_size - 1 and 1 <= current_x < grid_size - 1:
                            env.grid[current_y][current_x] = 'EMPTY'
                            
                            # Élargir le chemin parfois
                            if random.random() < 0.3:
                                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                                    ny, nx = current_y + dy, current_x + dx
                                    if 1 <= ny < grid_size - 1 and 1 <= nx < grid_size - 1:
                                        if env.grid[ny][nx] != 'WALL':
                                            env.grid[ny][nx] = 'EMPTY'
            
            # 5. Ajouter des obstacles variés (pas trop pour garder l'open world)
            MapGenerator._add_open_world_obstacles(env)
            
            # 6. Trouver plusieurs zones de départ possibles
            start_positions = MapGenerator._find_start_positions(env, num_starts=random.randint(2, 4))
            
            if start_positions and MapGenerator._is_fully_connected(env, start_positions[0]):
                # Choisir une position de départ aléatoire
                chosen_start = random.choice(start_positions)
                env.initial_pos = list(chosen_start)
                env.agent_pos = list(chosen_start)
                env.grid[chosen_start[0]][chosen_start[1]] = 'EMPTY'
                
                env.walkable_count = env._count_walkable_cells()
                return env
            
            print(f"[Info] Tentative {attempt} échouée, nouvel essai...")
        
        # Repli sur générateur simple
        print("[WARN] Open world échoué, utilisation du générateur simple.")
        return MapGenerator.generate_simple_open_world(env)
    
    @staticmethod
    def _find_start_positions(env, num_starts=3):
        """Trouve plusieurs positions de départ possibles"""
        grid_size = env.grid_size
        start_positions = []
        
        # Chercher des zones ouvertes dans différentes parties de la carte
        corners = [
            (2, 2),
            (2, grid_size - 3),
            (grid_size - 3, 2),
            (grid_size - 3, grid_size - 3)
        ]
        
        center = (grid_size // 2, grid_size // 2)
        candidates = corners + [center]
        
        for y, x in candidates:
            if (1 <= y < grid_size - 1 and 1 <= x < grid_size - 1 and
                env.grid[y][x] == 'EMPTY'):
                # Vérifier qu'il y a de l'espace autour
                neighbors_empty = sum(1 for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]
                                     if 0 <= y+dy < grid_size and 0 <= x+dx < grid_size
                                     and env.grid[y+dy][x+dx] == 'EMPTY')
                if neighbors_empty >= 2:
                    start_positions.append((y, x))
        
        # Si pas assez trouvées, chercher aléatoirement
        while len(start_positions) < num_starts:
            y = random.randint(2, grid_size - 3)
            x = random.randint(2, grid_size - 3)
            if env.grid[y][x] == 'EMPTY' and (y, x) not in start_positions:
                start_positions.append((y, x))
        
        return start_positions[:num_starts]
    
    @staticmethod
    def _add_open_world_obstacles(env):
        """Ajoute des obstacles pour un open world (moins dense qu'un labyrinthe)"""
        grid_size = env.grid_size
        
        # Moins d'obstacles que dans un labyrinthe
        obstacle_density = 0.15  # 15% de la carte peut être obstacle
        
        # 1. Ajouter quelques murs/obstacles épars
        num_obstacles = int(grid_size * grid_size * obstacle_density)
        empty_cells = []
        
        for y in range(2, grid_size - 2):
            for x in range(2, grid_size - 2):
                if env.grid[y][x] == 'EMPTY':
                    empty_cells.append((y, x))
        
        random.shuffle(empty_cells)
        
        # Transformer quelques cellules en obstacles
        obstacle_types = ['WALL', 'WATER', 'LAVA']
        for i in range(min(num_obstacles // 3, len(empty_cells))):
            y, x = empty_cells[i]
            obstacle_type = random.choice(obstacle_types)
            env.grid[y][x] = obstacle_type
        
        # 2. Ajouter des objets précieux et pièges
        remaining_empty = empty_cells[num_obstacles // 3:]
        random.shuffle(remaining_empty)
        
        num_gold = random.randint(*Config.NUM_GOLD)
        num_diamonds = random.randint(*Config.NUM_DIAMONDS)
        num_traps = random.randint(*Config.NUM_TRAPS)
        
        idx = 0
        
        for _ in range(min(num_gold, len(remaining_empty) - idx)):
            if idx < len(remaining_empty):
                y, x = remaining_empty[idx]
                env.grid[y][x] = 'GOLD'
                idx += 1
        
        for _ in range(min(num_diamonds, len(remaining_empty) - idx)):
            if idx < len(remaining_empty):
                y, x = remaining_empty[idx]
                env.grid[y][x] = 'DIAMOND'
                idx += 1
        
        for _ in range(min(num_traps, len(remaining_empty) - idx)):
            if idx < len(remaining_empty):
                y, x = remaining_empty[idx]
                env.grid[y][x] = 'TRAP'
                idx += 1
    
    @staticmethod
    def generate_simple_open_world(env):
        """Générateur open world simple de secours"""
        grid_size = env.grid_size
        env.grid = np.full((grid_size, grid_size), 'EMPTY', dtype=object)
        
        # Murs extérieurs seulement
        env.grid[0, :] = 'WALL'
        env.grid[grid_size - 1, :] = 'WALL'
        env.grid[:, 0] = 'WALL'
        env.grid[:, grid_size - 1] = 'WALL'
        
        # Quelques obstacles épars
        for _ in range(int(grid_size * 0.5)):
            y = random.randint(2, grid_size - 3)
            x = random.randint(2, grid_size - 3)
            if random.random() < 0.3:
                env.grid[y][x] = 'WALL'
            elif random.random() < 0.5:
                env.grid[y][x] = 'WATER'
        
        MapGenerator._add_open_world_obstacles(env)
        
        # Position de départ aléatoire
        start_y = random.randint(2, grid_size - 3)
        start_x = random.randint(2, grid_size - 3)
        env.initial_pos = [start_y, start_x]
        env.agent_pos = [start_y, start_x]
        env.grid[start_y][start_x] = 'EMPTY'
        
        env.walkable_count = env._count_walkable_cells()
        return env
    
    @staticmethod
    def generate_maze_dfs(env):
        """Génère un labyrinthe avec DFS et obstacles variés (comme Minecraft 2D)"""
        grid_size = env.grid_size
        max_attempts = 100
        attempt = 0

        while attempt < max_attempts:
            attempt += 1
            
            # 1. Reset: Remplir tout de murs
            env.grid = np.full((grid_size, grid_size), 'WALL', dtype=object)
            
            # 2. Créer le labyrinthe avec DFS
            stack = [(1, 1)]
            env.grid[1][1] = 'EMPTY'
            visited = {(1, 1)}
            
            while stack:
                current = stack[-1]
                y, x = current
                
                # Voisins à distance 2
                neighbors = []
                directions = [(-2, 0), (2, 0), (0, -2), (0, 2)]
                random.shuffle(directions)
                
                for dy, dx in directions:
                    ny, nx = y + dy, x + dx
                    if (0 < ny < grid_size - 1 and 0 < nx < grid_size - 1 and 
                        (ny, nx) not in visited):
                        neighbors.append((ny, nx, dy, dx))
                
                if neighbors:
                    ny, nx, dy, dx = neighbors[0]
                    # Casser le mur
                    wall_y, wall_x = y + dy // 2, x + dx // 2
                    env.grid[wall_y][wall_x] = 'EMPTY'
                    env.grid[ny][nx] = 'EMPTY'
                    visited.add((ny, nx))
                    stack.append((ny, nx))
                else:
                    stack.pop()
            
            # 3. Setup Départ
            env.initial_pos = [1, 1]
            env.agent_pos = [1, 1]
            env.grid[1][1] = 'EMPTY'
            
            # 4. Vérification de connectivité (toutes les cellules marchables doivent être accessibles)
            if MapGenerator._is_fully_connected(env, (1, 1)):
                # 5. AJOUT D'OBSTACLES VARIÉS (comme Minecraft 2D)
                MapGenerator._add_varied_obstacles(env)
                
                # 6. Revérifier la connectivité après ajout d'obstacles
                if MapGenerator._is_fully_connected(env, (1, 1)):
                    env.walkable_count = env._count_walkable_cells()
                    return env
            
            print(f"[Info] Tentative {attempt} échouée, nouvel essai...")
        
        # Repli sur générateur simple
        print("[WARN] DFS échoué après 100 tentatives. Utilisation du générateur simple.")
        return MapGenerator.generate_simple_maze(env)
    
    @staticmethod
    def generate_map(env, map_type='open_world'):
        """Génère une carte selon le type demandé"""
        env.map_type = map_type
        if map_type == 'open_world':
            return MapGenerator.generate_open_world(env)
        else:
            return MapGenerator.generate_maze_dfs(env)

    @staticmethod
    def _add_varied_obstacles(env):
        """Ajoute des obstacles de différents types (WALL, WATER, LAVA)"""
        grid_size = env.grid_size
        
        # Collecter toutes les cellules EMPTY (sauf départ/arrivée)
        empty_cells = []
        for y in range(1, grid_size - 1):
            for x in range(1, grid_size - 1):
                if env.grid[y][x] == 'EMPTY':
                    # Éloigné du départ et de l'arrivée
                    if abs(y - 1) + abs(x - 1) > 3:
                        empty_cells.append((y, x))
        
        random.shuffle(empty_cells)
        
        # 1. Remplacer certains murs par WATER et LAVA
        wall_cells = []
        for y in range(1, grid_size - 1):
            for x in range(1, grid_size - 1):
                if env.grid[y][x] == 'WALL':
                    # Éviter les murs critiques (bords)
                    if 2 < y < grid_size - 3 and 2 < x < grid_size - 3:
                        wall_cells.append((y, x))
        
        random.shuffle(wall_cells)
        
        # Transformer certains murs en WATER
        num_water = random.randint(*Config.NUM_WATER)
        for i in range(min(num_water, len(wall_cells))):
            y, x = wall_cells[i]
            env.grid[y][x] = 'WATER'
        
        # Transformer certains murs en LAVA
        num_lava = random.randint(*Config.NUM_LAVA)
        for i in range(num_water, min(num_water + num_lava, len(wall_cells))):
            y, x = wall_cells[i]
            env.grid[y][x] = 'LAVA'
        
        # 2. Ajouter des récompenses et pièges sur les chemins
        num_gold = random.randint(*Config.NUM_GOLD)
        num_diamonds = random.randint(*Config.NUM_DIAMONDS)
        num_traps = random.randint(*Config.NUM_TRAPS)
        
        idx = 0
        
        # Or
        for _ in range(min(num_gold, len(empty_cells) - idx)):
            if idx < len(empty_cells):
                env.grid[empty_cells[idx]] = 'GOLD'
                idx += 1
        
        # Diamants
        for _ in range(min(num_diamonds, len(empty_cells) - idx)):
            if idx < len(empty_cells):
                env.grid[empty_cells[idx]] = 'DIAMOND'
                idx += 1
        
        # Pièges
        for _ in range(min(num_traps, len(empty_cells) - idx)):
            if idx < len(empty_cells):
                env.grid[empty_cells[idx]] = 'TRAP'
                idx += 1

    @staticmethod
    def generate_simple_maze(env):
        """Générateur de secours"""
        grid_size = env.grid_size
        env.grid = np.full((grid_size, grid_size), 'EMPTY', dtype=object)
        
        # Murs extérieurs
        env.grid[0, :] = 'WALL'
        env.grid[grid_size - 1, :] = 'WALL'
        env.grid[:, 0] = 'WALL'
        env.grid[:, grid_size - 1] = 'WALL'
        
        # Obstacles aléatoires
        for _ in range(int(grid_size * 1.5)):
            if random.random() < 0.5:
                y = random.randint(2, grid_size - 3)
                x = random.randint(2, grid_size - 5)
                for k in range(3):
                    if x+k < grid_size-1: 
                        env.grid[y][x+k] = 'WALL'
            else:
                x = random.randint(2, grid_size - 3)
                y = random.randint(2, grid_size - 5)
                for k in range(3):
                    if y+k < grid_size-1: 
                        env.grid[y+k][x] = 'WALL'
        
        env.initial_pos = [1, 1]
        env.agent_pos = [1, 1]
        env.grid[1][1] = 'EMPTY'
        
        MapGenerator._add_varied_obstacles(env)
        env.walkable_count = env._count_walkable_cells()
        return env
    
    @staticmethod
    def _is_fully_connected(env, start):
        """Vérifie que toutes les cellules marchables sont atteignables depuis le départ"""
        queue = deque([start])
        visited = {start}
        rows, cols = env.grid.shape
        walkable_seen = 0
        
        while queue:
            y, x = queue.popleft()
            if Config.CELL_TYPES[env.grid[y][x]]['walkable']:
                walkable_seen += 1
            
            for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:
                ny, nx = y+dy, x+dx
                if 0 <= ny < rows and 0 <= nx < cols:
                    cell_type = env.grid[ny][nx]
                    # CORRECTION: Utiliser is_walkable du Config
                    is_walkable = Config.CELL_TYPES.get(cell_type, {}).get('walkable', False)
                    if is_walkable and (ny, nx) not in visited:
                        visited.add((ny, nx))
                        queue.append((ny, nx))
        total_walkable = sum(
            1 for y in range(rows) for x in range(cols)
            if Config.CELL_TYPES[env.grid[y][x]]['walkable']
        )
        return walkable_seen == total_walkable