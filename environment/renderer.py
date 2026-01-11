"""Visualisation avec Pygame + Sauvegarde de la carte"""

import pygame
import os
from config import Config
from datetime import datetime

class Renderer:
    def __init__(self, grid_size):
        pygame.init()
        self.grid_size = grid_size
        self.cell_size = Config.CELL_SIZE
        
        # Calcul de la taille de la fenêtre (Grille + Espace stats en bas)
        self.width = grid_size * self.cell_size
        self.height = grid_size * self.cell_size + 200
        
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Smart Explorer - Cartographie 2D")
        
        # Polices : Arial pour le texte, police système pour les symboles
        self.font = pygame.font.SysFont("Arial", 24, bold=True)
        self.small_font = pygame.font.SysFont("Segoe UI Symbol", 20) 
        self.clock = pygame.time.Clock()
    
    def render(self, env, discovered_map, stats, show_full_map=False):
        """Affiche l'environnement et les statistiques à l'écran"""
        self.screen.fill((20, 20, 30)) # Fond général sombre
        
        # --- 1. DESSINER LA GRILLE ---
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                rect = pygame.Rect(
                    x * self.cell_size,
                    y * self.cell_size,
                    self.cell_size,
                    self.cell_size
                )
                
                # Une case est visible si : Découverte OU Position actuelle agent OU Mode Debug (show_full_map)
                is_known = (y, x) in discovered_map or (env.agent_pos[0] == y and env.agent_pos[1] == x)
                is_visited = (y, x) in env.visited
                
                if show_full_map or is_known:
                    cell_type = env.grid[y][x]
                    color = Config.CELL_TYPES[cell_type]['color']
                    
                    # Logique : Si découvert mais PAS ENCORE VISITÉ -> Rendre en Gris/Désaturé
                    # SAUF pour les obstacles (Mur, Eau, Lave) qui s'affichent en couleur dès qu'ils sont connus
                    is_obstacle = cell_type in ['WALL', 'WATER', 'LAVA']
                    
                    if not is_visited and not is_obstacle and not show_full_map:
                        if env.map_type == 'open_world':
                            # Pour Open World: Si vu mais pas visité, reste comme le brouillard
                            color = (15, 15, 25)
                        else:
                            # Pour les autres (Maze...): on grise pour voir ce que c'est sans que ce soit "officiel"
                            gray_val = int(0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2])
                            color = (
                                int(color[0] * 0.3 + gray_val * 0.7),
                                int(color[1] * 0.3 + gray_val * 0.7),
                                int(color[2] * 0.3 + gray_val * 0.7)
                            )
                            color = tuple(max(0, int(c * 0.8)) for c in color)

                    
                    # EFFET VISUEL : Assombrir les zones "vues par triche" (show_full_map) mais pas encore explorées
                    if show_full_map and not is_known:
                        color = tuple(max(0, int(c * 0.4)) for c in color)
                    
                    # Dessiner le fond de la case
                    pygame.draw.rect(self.screen, color, rect)
                    
                    # Effet 3D pour les obstacles
                    if cell_type in ['WALL', 'WATER', 'LAVA']:
                        pygame.draw.line(self.screen, (0, 0, 0), rect.topleft, rect.topright, 2)
                        pygame.draw.line(self.screen, (0, 0, 0), rect.topleft, rect.bottomleft, 2)
                        lighter = tuple(min(255, c + 40) for c in color)
                        pygame.draw.line(self.screen, lighter, rect.bottomleft, rect.bottomright, 2)
                        pygame.draw.line(self.screen, lighter, rect.topright, rect.bottomright, 2)
                    else:
                        # Bordure subtile pour les cellules normales
                        pygame.draw.rect(self.screen, (200, 200, 200), rect, 1)

                else:
                    # BROUILLARD DE GUERRE (Fog of War) - Cellules totalement inconnues
                    pygame.draw.rect(self.screen, (15, 15, 25), rect) # Noir bleuté très sombre
                    pygame.draw.rect(self.screen, (30, 30, 45), rect, 1) # Légère grille de Fog

                
                # DESSINER L'AGENT
                if env.agent_pos[0] == y and env.agent_pos[1] == x:
                    center = rect.center
                    radius = self.cell_size // 3
                    # Contour blanc pour contraste
                    pygame.draw.circle(self.screen, (255, 255, 255), center, radius + 2)
                    # Agent vert
                    pygame.draw.circle(self.screen, Config.AGENT_COLOR, center, radius)

        # --- 2. AFFICHER LES STATS ---
        self._render_stats(stats, show_full_map)
        
        pygame.display.flip()
        self.clock.tick(60) # 60 FPS max
    
    def _render_stats(self, stats, show_full_map):
        """Affiche le panneau de contrôle en bas"""
        base_y = self.grid_size * self.cell_size + 10
        
        # Fond du panneau
        pygame.draw.rect(self.screen, (30, 32, 40), (0, base_y - 10, self.width, 200))
        pygame.draw.line(self.screen, (100, 100, 100), (0, base_y - 10), (self.width, base_y - 10), 2)

        col1_x = 20
        col2_x = self.width // 2 + 20
        
        # Données à afficher
        lines_col1 = [
            (f"Episode: {stats['episode']}", (255, 255, 255)),
            (f"Pas: {stats['steps']}", (200, 200, 255)),
            (f"Score: {stats['total_reward']:.1f}", (255, 200, 100)),
        ]
        
        coverage_pct = stats.get('coverage', 0) * 100
        algorithm = stats.get('algorithm', 'QLEARNING')
        epsilon_val = stats.get('epsilon', 0.0)
        collected_items = stats.get('collected_items', 0)
        lines_col2 = [
            (f"Algorithme: {algorithm}", (100, 255, 100)),
            (f"Exploration: {epsilon_val:.1%}", (100, 255, 100)),
            (f"Carte connue: {coverage_pct:.1f}%", (100, 200, 255)),
            (f"Objets collectes: {collected_items}", (255, 215, 0)),
        ]
        
        for i, (txt, color) in enumerate(lines_col1):
            self.screen.blit(self.font.render(txt, True, color), (col1_x, base_y + i * 30))
            
        for i, (txt, color) in enumerate(lines_col2):
            self.screen.blit(self.font.render(txt, True, color), (col2_x, base_y + i * 30))
            
        # Message central
        if stats.get('success', False):
            msg = "Exploration suffisante"
            col = (50, 255, 50)
        else:
            msg = "Exploration en cours..."
            col = (150, 150, 150)
            
        surf = self.font.render(msg, True, col)
        rect = surf.get_rect(center=(self.width // 2, base_y + 100))
        self.screen.blit(surf, rect)

    def save_map_image(self, env, discovered_map, episode_num, stats):
        """Sauvegarde une image PNG propre de la carte avec métriques superposées"""
        # Créer le dossier
        if not os.path.exists(Config.SAVE_MAP_DIR):
            os.makedirs(Config.SAVE_MAP_DIR)
            
        # Surface temporaire (taille de la grille + petit bandeau info)
        s_size = self.grid_size * self.cell_size
        surf = pygame.Surface((s_size, s_size))
        surf.fill((20, 20, 30))
        
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                rect = pygame.Rect(x*self.cell_size, y*self.cell_size, self.cell_size, self.cell_size)
                
                # On dessine ce que le robot connait
                is_known = (y, x) in discovered_map
                is_visited = (y, x) in env.visited
                
                if is_known or (y == env.agent_pos[0] and x == env.agent_pos[1]):
                    ct = env.grid[y][x]
                    color = Config.CELL_TYPES[ct]['color']
                    
                    # Même logique de rendu (en brouillard ou gris si non visité)
                    is_obstacle = ct in ['WALL', 'WATER', 'LAVA']
                    if not is_visited and not is_obstacle:
                        if env.map_type == 'open_world':
                            color = (10, 10, 20) # Couleur du brouillard de sauvegarde
                        else:
                            gray_val = int(0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2])
                            color = (int(color[0]*0.3 + gray_val*0.7), int(color[1]*0.3 + gray_val*0.7), int(color[2]*0.3 + gray_val*0.7))
                    
                    pygame.draw.rect(surf, color, rect)
                    
                    # Effet 3D
                    if is_obstacle:
                        pygame.draw.line(surf, (0, 0, 0), rect.topleft, rect.topright, 2)
                        pygame.draw.line(surf, (0, 0, 0), rect.topleft, rect.bottomleft, 2)
                    else:
                        pygame.draw.rect(surf, (150, 150, 150), rect, 1)
                else:
                    pygame.draw.rect(surf, (10, 10, 20), rect)
        
        # --- AJOUT OVERLAY TEXTE ---
        # Petit bandeau semi-transparent en haut
        overlay = pygame.Surface((s_size, 40), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180)) # Noir transparent
        surf.blit(overlay, (0, 0))
        
        alg = stats.get('algorithm', 'IA')
        cov = stats.get('coverage', 0) * 100
        txt = f"[{alg}] Ep: {episode_num} | Couverture: {cov:.1f}%"
        
        text_surf = self.small_font.render(txt, True, (255, 255, 255))
        surf.blit(text_surf, (10, 8))
        
        # Sauvegarde
        ts = datetime.now().strftime("%H%M%S")
        fname = f"{Config.SAVE_MAP_DIR}/map_ep{episode_num}_{alg}_{ts}.png"
        pygame.image.save(surf, fname)
        print(f"[Renderer] Carte sauvegardee : {fname}")


    def show_episode_complete(self, env, discovered_map, stats, duration=1000):
        """Pause visuelle à la fin d'un épisode"""
        start = pygame.time.get_ticks()
        while pygame.time.get_ticks() - start < duration:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: return False
            self.render(env, discovered_map, stats, show_full_map=True)
        return True

    def close(self):
        pygame.quit()