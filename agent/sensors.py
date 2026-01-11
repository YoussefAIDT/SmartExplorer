"""Système de vision et capteurs de l'agent avec raycasting"""

from config import Config
import math

class Sensors:
    def __init__(self, env):
        self.env = env
        self.vision_radius = Config.AGENT_VISION_RADIUS
    
    def scan(self, position):
        """Scanne l'environnement avec raycasting - ne voit pas à travers les murs"""
        y, x = position
        visible_cells = {}
        
        # Toujours voir la cellule actuelle
        if self.env.is_valid_position(y, x):
            visible_cells[(y, x)] = self.env.get_cell_type(y, x)
        
        # Voir les cellules adjacentes directement (pas besoin de raycasting)
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if self.env.is_valid_position(ny, nx):
                    visible_cells[(ny, nx)] = self.env.get_cell_type(ny, nx)
        
        # Raycasting pour les cellules plus éloignées
        for angle in range(0, 360, 10):  # 10 degrés de résolution
            rad = math.radians(angle)
            dx = math.cos(rad)
            dy = math.sin(rad)
            
            # Lancer un rayon jusqu'à vision_radius
            for step in range(2, self.vision_radius + 1):
                target_y = int(round(y + dy * step))
                target_x = int(round(x + dx * step))
                
                if not self.env.is_valid_position(target_y, target_x):
                    break
                
                # Vérifier la ligne de vue
                if self._has_line_of_sight(y, x, target_y, target_x):
                    cell_type = self.env.get_cell_type(target_y, target_x)
                    visible_cells[(target_y, target_x)] = cell_type
                    
                    # Si on rencontre un mur opaque, arrêter le rayon
                    if cell_type == 'WALL':
                        break
                else:
                    break
        
        return visible_cells
    
    def _has_line_of_sight(self, y1, x1, y2, x2):
        """Vérifie si la ligne de vue est libre (algorithme de Bresenham)"""
        if y1 == y2 and x1 == x2:
            return True
        
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy
        
        x, y = x1, y1
        
        while True:
            # Vérifier si la cellule actuelle est un mur opaque (sauf la destination)
            if not (x == x2 and y == y2):
                if self.env.is_valid_position(y, x):
                    cell_type = self.env.get_cell_type(y, x)
                    if cell_type == 'WALL':
                        return False
            
            if x == x2 and y == y2:
                return True
            
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
            
            if not self.env.is_valid_position(y, x):
                return False

