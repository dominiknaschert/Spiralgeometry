import matplotlib.pyplot as plt
import numpy as np

def L_p(x):
    """Umrechnung linear → dB"""
    return 10.0 * np.log10(np.maximum(x, 1e-30))  # verhindert log(0)


def plot_psf_2d(psf_obj, ax=None, cmap='viridis', show=True):
    """
    Plottet eine 2D-PSF (wie in Spectacular) mit einer einzigen Quelle im Gitter.

    Voraussetzung: psf_obj.grid_indices enthält genau einen Index.

    --> Ersetzt die PointSpreadFunctionPresenter-Klasse in Spectacular.

    Parameters
    ----------
    psf_obj : PointSpreadFunction
    ax : matplotlib.axes.Axes, optional
    cmap : str
    show : bool

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    if len(psf_obj.grid_indices) != 1:
        raise ValueError("Für 2D-Plot muss genau ein grid_index gesetzt sein!")

    # PSF-Vektor für einen Gitterpunkt
    psf_data = psf_obj.psf[:, 0]  # (N,)
    data = L_p(psf_data)
    data -= data.max()  # auf 0 dB normalisieren

    # Grid-Shape
    shape = psf_obj.steer.grid.shape  # z. B. (nx, ny)
    data_2d = data.reshape(shape).T  # Transponieren für richtiges Layout

    # Plotbereich
    extent = psf_obj.steer.grid.extend()

    # Plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(6,5))

    im = ax.imshow(
        data_2d,
        origin='lower',
        extent=extent,
        vmin=-20,  # typischer Dynamikbereich
        cmap=cmap,
        interpolation='bicubic'
    )
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Relative SPL [dB]')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title(f'Point Spread Function @ {psf_obj.freq:.0f} Hz')

    if show:
        plt.tight_layout()
        plt.show()

    return ax

