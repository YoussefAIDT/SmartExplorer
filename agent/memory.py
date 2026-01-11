"""Mémoire de l'agent pour stocker les cellules visitées"""

class Memory:
    def __init__(self):
        self.visited_cells = set()
        self.discovered_cells = {}
    
    def add_visited(self, position):
        """Ajoute une cellule visitée et indique si elle est nouvelle"""
        before = len(self.visited_cells)
        self.visited_cells.add(position)
        return len(self.visited_cells) > before
    
    def add_discovered(self, position, cell_type):
        """Ajoute une cellule découverte"""
        self.discovered_cells[position] = cell_type
    
    def is_visited(self, position):
        """Vérifie si une cellule a été visitée"""
        return position in self.visited_cells

    def get_visited_cells(self):
        """Retourne toutes les cellules visitées"""
        return self.visited_cells
    
    def get_discovered_map(self):
        """Retourne la carte des cellules découvertes avec leur type"""
        return self.discovered_cells

    def clear(self):
        """Efface la mémoire"""
        self.visited_cells.clear()
        self.discovered_cells.clear()