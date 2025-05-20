import numpy as np
from pathlib import Path
from acoular import MicGeom, WNoiseGenerator, PointSource, Mixer, WriteH5

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
        # Outsourced damit theoretisch weitere Geometrien einfacher hinzugefügt werden können.
        m = np.arange(1, self.num_mics + 1)
        r = self.R * np.sqrt(m / self.num_mics)
        phi = 2 * np.pi * m * ((1 + np.sqrt(self.V)) / 2)

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

    def create_sources(self, source_definitions=None, sfreq=51200, duration=1.0):
        """
        Erzeugt PointSources für das Array.
        
        - Wenn `source_definitions` None ist → Standard-PSF: 1 Quelle bei (0,0,0).
        - Sonst erwartet: Liste von Dicts mit keys: 'loc', 'rms', 'seed'.
        """
        # Nicht Teil der Aufgabendefinition!
        # hier müsste man mit der PointSpreadFunction arbeiten, dann kann man die Quellen ignorieren!
        num_samples = int(sfreq * duration)

        if source_definitions is None:
            # Standardfall: PSF-Demo mit 1 Quelle in der Mitte
            print("No sources defined. Using default PointSource at (0, 0, 0).")
            noise = WNoiseGenerator(sample_freq=sfreq, num_samples=num_samples, seed=1, rms=1.0)
            return PointSource(signal=noise, mics=self.micgeom, loc=(0.0, 0.0, 0.0))

        # Falls benutzerdefinierte Quellen definiert sind:
        sources = []
        for i, sdef in enumerate(source_definitions):
            noise = WNoiseGenerator(sample_freq=sfreq, num_samples=num_samples,
                                    seed=sdef.get('seed', i+1), rms=sdef.get('rms', 1.0))
            src = PointSource(signal=noise, mics=self.micgeom, loc=sdef['loc'])
            sources.append(src)

        # Mixer verwenden, falls mehrere Quellen
        if len(sources) == 1:
            return sources[0]
        else:
            # Alle Quellen im Mixer zusammenführen
            print(f"Creating Mixer with {len(sources)} sources.")
            return Mixer(source=sources[0], sources=sources[1:])

    def as_MicGeom(self):
        """
        Gibt das MicGeom-Objekt zurück für andere Acoular-Objekte.
        """
        return self.micgeom