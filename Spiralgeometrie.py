import numpy as np
from pathlib import Path
from acoular import MicGeom, WNoiseGenerator, PointSource, Mixer
from scipy.integrate import quad
from scipy.optimize import least_squares
from scipy.special import i0 # Besselsche Funktion erster Art
import matplotlib.pyplot as plt



class SpiralGeometry:
    def __init__(self, num_mics=64, D=1.0, V=5.0, H=4.0):
        self.num_mics = num_mics
        self.R = D/2 # Radius der Spirale 
        self.V = V
        self.H = H
        self.positions = self._generate_spiral_positions() # generiert Vogelspirale
        self.micgeom = MicGeom(pos_total=self.positions)

    def f_H(self, H, r):
        """
        Berechnet die Hansen-Gewichtungsfunktion f_H(r) gemäß Gleichung (3).
        Diese Funktion ist für die Gewichtung der Spiralverteilung der Mikrofone verantwortlich.
        --> muss noch überarbeitet werden!
        """
        rho = r / self.R
        inside = np.pi * H * np.sqrt(1 - rho**2)
        return i0(inside) # i0 --> Besselsche Funktion erster Art, modifiziert für Spiralverteilung

    def _generate_spiral_positions(self):
        """
        Generiert Mikrofonpositionen gemäß Gleichung (11) mit Hansen-Gewichtung.
        Dabei wird eine gewichtete Verteilung der Mikrofone im Radius erzeugt.
        """
        M = self.num_mics
        R = self.R
        H = self.H

        # --- Schritt 1: Integral von f_H über [0, R] (konstant über alle m) ---
        # Ansatz funktioniert nicht Gewichtungsmethode oder integral noch falsch!
        try:
            integral_total, _ = quad(lambda r: self.f_H(H, r), 0, R)
        except Exception as e:
            raise RuntimeError(f"Fehler beim Integrieren von f_H: {e}")

        # --- Schritt 2: Startwerte aus Gleichung (8) ---
        r_initial = R * np.sqrt(np.arange(1, M + 1) / M)

        # --- Schritt 3: Residuenfunktion für Gleichung (11) ---
        def residuals(r_vec):
            res = []
            for m in range(1, M + 1):
                sum_term = 0
                for i in range(m):
                    f_val = self.f_H(H, r_vec[i])  # kein R-Normalisieren nötig, da f_H(r) erwartet
                    if f_val <= 0 or not np.isfinite(f_val):
                        f_val = 1e-12  # numerische Sicherheit
                    sum_term += integral_total / (M * f_val)
                rhs = R * np.sqrt(sum_term)
                res.append(r_vec[m - 1] - rhs)
            return res

        # --- Schritt 4: Lösung über nichtlineares Least Squares ---
        sol = least_squares(residuals, r_initial, method='trf', bounds=(0, R))
        if not sol.success:
            raise RuntimeError("Nichtlineares Gleichungssystem zur Bestimmung der Radien konnte nicht gelöst werden.")

        r_m = sol.x

        # --- Schritt 5: Winkel gemäß Gleichung (9) ---
        m_array = np.arange(1, M + 1)
        phi = 2 * np.pi * m_array * ((1 + np.sqrt(self.V)) / 2)

        # --- Schritt 6: Umrechnung in kartesische Koordinaten ---
        x = r_m * np.cos(phi)
        y = r_m * np.sin(phi)
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
        Gibt das MicGeom-Objekt zurück.
        """
        return self.micgeom