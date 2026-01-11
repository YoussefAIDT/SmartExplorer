"""Logique principale de l'agent explorateur"""

from agent.sensors import Sensors
from agent.memory import Memory

class Explorer:
    def __init__(self, env):
        self.env = env
        self.sensors = Sensors(env)
        self.memory = Memory()
        self.position = [0, 0]
    
    def perceive(self):
        """L'agent observe son environnement via ses capteurs"""
        return self.sensors.scan(self.position)
    
    def update_position(self, new_position):
        """Met à jour la position de l'agent"""
        self.position = new_position
        self.memory.add_visited(tuple(new_position))
    
    def get_discovered_map(self):
        """Retourne la carte des cellules découvertes (y compris obstacles)"""
        return self.memory.get_discovered_map()

    def get_visited_cells(self):
        """Retourne les cellules effectivement visitées"""
        return self.memory.get_visited_cells()
