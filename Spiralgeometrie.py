import numpy as np
from pathlib import Path
from acoular import MicGeom, WNoiseGenerator, PointSource, Mixer


class SpiralGeometry:
    def __init__(self, num_mics=64, R=1.0, V=5.0):
        """
        Spiralgeometrie erzeugen & MicGeom initialisieren.
        """
        self.num_mics = num_mics
        self.R = R
        self.V = V
        self.positions = self._generate_spiral_positions()
        self.micgeom = MicGeom(pos_total=self.positions) # MicGeom-Objekt mit den Positionen erstellen, sodass es in der Methode as_MicGeom() zurückgegeben werden kann.

    def _generate_spiral_positions(self):
        '''
        Erzeugt die Positionen der Mikrofone anhand der eingabe werte des Obejktes.
        '''
        # noch nicht richtig implementierung der Spiralgeometrie aus dem Paper.
        # muss noch implementiert werden!
        m = np.arange(1, self.num_mics + 1) # array mit den Mikrofon-IDs (1 bis num_mics)
        r = self.R * np.sqrt(m / self.num_mics) # (8)
        phi = 2 * np.pi * m * ((1 + np.sqrt(self.V)) / 2) # (9)

        x = r * np.cos(phi)
        y = r * np.sin(phi)
        z = np.zeros_like(x)

        return np.vstack([x, y, z])

    def export_geometry_xml(self, filename='spiral_geometry.xml'):
        '''
        Exportiert die Mikrofonpositionen in eine XML-Datei.
        '''
        self.micgeom.export_mpos(filename)
        print(f"Spiral geometry exported to {filename}")
        return Path(filename)


    def as_MicGeom(self):
        """
        Gibt das MicGeom-Objekt zurück für andere Acoular-Objekte.
        """
        return self.micgeom