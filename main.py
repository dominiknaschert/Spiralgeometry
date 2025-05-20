import numpy as np
import matplotlib.pyplot as plt
import acoular as ac

from Spiralgeometrie import SpiralGeometry # SpiralGeometry-Klasse importieren


def main():
    print("Generiere Spiralgeometrie")

    # Spiralgeometry-Objekt erstellen
    #-----------------------------------------------------------
    # Beispeil: SpiralGeometry-Objekt mit 64 Mikrofonen, Radius 1.0 und V=3.0 erstellen
    spiral = SpiralGeometry(num_mics=64, R=1.0, V=3.0)

    # Spiral-Positionen als xml exportieren
    spiral.export_geometry_xml('spiral_geometry.xml')

    # optional eigene Quellen definieren und an SpiralGeometry übergeben (OutofScope).
    source_definitions = [
        {'loc': (-0.1, -0.1, -0.3), 'rms': 1.0, 'seed': 1},
        {'loc': (0.15, 0.0, -0.3), 'rms': 0.7, 'seed': 2},
        {'loc': (0.0, 0.1, -0.3), 'rms': 0.5, 'seed': 3}
    ]

    # Quelle erzeugen (default: selbst erzeugte PSF mit Quelle bei (0,0,0))
    #-----------------------------------------------------------
    # Nach Saradj sollen wir die PointSpreadFunction benutzen und keine eigenen Quellen erzeugen. Allerdings gibt es in Acoular keine PointspreadFunction?
    # --> die Spectacular Bibliothek hat eine PointSpreadFunction, die wir hier verwenden können/sollen.

    # Nach Acoular kann man mit relativ wenig Zeilen eine PointSpreadFunction Simulieren und eine gleiche ähnliche Ausgabe erzeugen. Ist aber nicht im Scope unseres Projektes!
    source = spiral.create_sources(duration=1.0, source_definitions=None)

    # MicGeom direkt aus dem Spiral-Objekt holen, Das funktioniert weil wir in der Init direkt mit den Eingabeparametern des Objektes das Objekt generieren:
    # mit einer Methode as_MicGeom können wir das Objekt dann übergeben heißt wir brauchen kein MicGeom Objekt selber mehr erzeugen.
    # --> Wurde mir Sarradj am 20.05 besprochen (bester Ansatz)
    mg = spiral.as_MicGeom()


    # Der folgende Absatz soll dann zum teil von der PointSpreadFunction übernommen werden. Details sind hier noch zu klären!
    #-----------------------------------------------------------
    # Frequenzanalyse
    ps = ac.PowerSpectra(source=source, block_size=128, window='Hanning')
    # Beamforming Setup
    rg = ac.RectGrid(x_min=-0.4, x_max=0.4, y_min=-0.4, y_max=0.4, z=0.5, increment=0.02)
    st = ac.SteeringVector(grid=rg, mics=mg)
    bf = ac.BeamformerBase(freq_data=ps, steer=st)

    # Berechnung bei 2000 Hz
    f = 2000
    pm = bf.synthetic(f, 3)
    Lm = ac.L_p(pm)

    # Plot: Beamforming-Map 
    plt.figure(1)
    plt.title(f"Beamforming Map @ {f} Hz")
    plt.imshow(Lm.T, origin='lower', vmin=Lm.max() - 20, extent=rg.extend(), interpolation='bicubic')
    plt.colorbar(label="dB")

    # Mikrofonpositionen anzeigen
    plt.figure(2)
    plt.title("Mikrofonanordnung (Spiral)")
    plt.plot(mg.mpos[0], mg.mpos[1], 'o')
    plt.axis('equal')
    plt.grid()

    plt.show()
    print("Beamforming abgeschlossen und visualisiert.")


if __name__ == "__main__":
    main()