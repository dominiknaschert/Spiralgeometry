import numpy as np
import matplotlib.pyplot as plt
import acoular as ac

from Spiralgeometrie import SpiralGeometry # SpiralGeometry-Klasse importieren
from Hilfsfunktionen_PSFPresenter import plot_psf_2d # Hilfsfunktionen importieren


def main():
    print("Generiere Spiralgeometrie")

    # Spiralgeometry-Objekt erstellen
    #-----------------------------------------------------------
    # Beispeil: SpiralGeometry-Objekt mit 64 Mikrofonen, Radius 1.0 und V=3.0 erstellen
    spiral = SpiralGeometry(num_mics=64, D=1.0, V=5.0, H=1.0)

    # Spiral-Positionen als xml exportieren
    spiral.export_geometry_xml('spiral_geometry.xml')


    # MicGeom direkt aus dem Spiral-Objekt holen, Das funktioniert weil wir in der Init direkt mit den Eingabeparametern des Objektes das Objekt generieren:
    # mit der Methode as_MicGeom können wir das Objekt dann übergeben, heißt wir brauchen kein MicGeom Objekt selber mehr erzeugen.
    # --> Wurde mir Sarradj am 20.05 besprochen (bester Ansatz)
    # --> Außerdem: Einzahlwerte können einfach als Methode der PointSpreadFunction etabliert werden.
    mg = spiral.as_MicGeom()


    # Der folgende Absatz soll dann zum teil von der PointSpreadFunction übernommen werden. Details sind hier noch zu klären!
    #-----------------------------------------------------------
    # Frequenzanalyse
    #ps = ac.PowerSpectra(source=source, block_size=128, window='Hanning')
    # Beamforming Setup
    rg = ac.RectGrid(x_min=-0.5, x_max=0.5, y_min=-0.5, y_max=0.5, z=0.2, increment=0.02)
    st = ac.SteeringVector(grid=rg, mics=mg)
    psf = ac.PointSpreadFunction(steer=st, freq=2000.0)


    # Index für zentrale Quelle
    psf.grid_indices = np.array([psf.steer.grid.size // 2])
    psf.calcmode = 'single'

    # 2D-Plot
    plot_psf_2d(psf) # Die Hielfsmethode plot_psf_2d ist äquivalent zu der PointSpreadFunctionPresenter Klasse in Spectacular der man das psf-Objekt übergibt um es zu plotten.



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