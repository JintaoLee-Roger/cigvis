import numpy as np
import matplotlib.pyplot as plt
from cigvis.utils.surfaceutils import arbitray_line


def extract_arbitrary_line_by_view(data: np.ndarray,
                                   bmap: str = 'data',
                                   draw_arb: bool = True,
                                   *,
                                   line: bool = True,
                                   idx: int = 50,
                                   cline='#F3AA3C'):
    """
    extract arbitrary line from seismic data by clicking

    Parameters
    ----------
    - data: np.ndarray 
        3D seismic data
    - bmap: str
        background map, 'data' or 'blank'
    - line : bool
        whether to draw the broken line 
    - idx: int 
        the slice index of the seismic data if bmap is 'data'
    - cline: str
        color of the line

    Returns
    -------
    - out: np.ndarray
        extracted arbitrary line
    - p: np.ndarray
        extracted arbitrary line path
    - coords: np.ndarray
        the coordinates by clicking
    """
    fig, ax = plt.subplots()
    if bmap == 'data':
        img = data[:, :, idx]
        ax.imshow(img.T, cmap='gray', aspect='auto')
    else:
        img = np.zeros((data.shape[1], data.shape[0], 4), dtype=np.uint8)
        img = img + int(0.9 * 255)
        ax.imshow(img, aspect='auto')
        ax.grid(True)

    ax.set_title('Obtain a path by clicking, press "u" to undo, "enter" to finish') # yapf: disable
    ax.set_xlabel('Inline')
    ax.set_ylabel('Crossline')
    coords = []
    lines = []
    points = []
    out = None
    p = None
    indices = None

    def _draw_arbline():
        if (len(coords) > 0):
            ax.clear()
            nonlocal out, p, indices
            out, p, indices = arbitray_line(data, coords)
            ax.grid(False)
            ax.imshow(out.T, cmap='gray', aspect='auto')
            if line:
                for i in range(len(indices)):
                    ax.plot([indices[i]] * out.shape[1],
                            range(out.shape[1]),
                            'w--',
                            lw=0.5)
                    if i > 0 and i < len(indices) - 1:
                        ax.text(indices[i] - out.shape[0] / 50,
                                30,
                                str(indices[i]),
                                color='w',
                                fontsize=10)
            ax.set_title("Arbitrary Line")
            ax.set_ylabel('Time')
            fig.canvas.draw()
            # plt.close()

    def _click_event(event):
        if event.inaxes:
            coords.append((round(event.xdata, 2), round(event.ydata, 2)))
            p = ax.plot(event.xdata, event.ydata, 'ro')[0]
            points.append(p)
            if len(coords) > 1:
                x_coords, y_coords = zip(*coords[-2:])
                line, = ax.plot(x_coords, y_coords, c=cline)
                lines.append(line)
            fig.canvas.draw()

    def _undo_last(event):
        if event.key == 'u':
            if coords:
                coords.pop()
                p = points.pop()
                p.remove()
                if len(lines) > 0:
                    l = lines.pop()
                    l.remove()
                fig.canvas.draw()
        if event.key in ('enter', 'return', 'escape'):
            if draw_arb:
                fig.canvas.mpl_disconnect(cid_click)
                fig.canvas.mpl_disconnect(cid_key)
                _draw_arbline()
            else:
                plt.close(fig)

    cid_click = fig.canvas.mpl_connect('button_press_event', _click_event)
    cid_key = fig.canvas.mpl_connect('key_press_event', _undo_last)

    plt.show()

    return out, p, indices, np.array(coords)


if __name__ == '__main__':
    from cigsegy import SegyNP
    d = SegyNP('/Volumes/T7/DATA/cnooc_bj/data/FEF.segy')
    out, p, ind, c = extract_arbitrary_line_by_view(
        d,
        bmap='data',
        draw_arb=True,
    )
    print(c.shape)
