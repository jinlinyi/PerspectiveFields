from detectron2.utils.visualizer import Visualizer
from detectron2.utils.colormap import random_color
import numpy as np
import matplotlib.pyplot as plt

class VisualizerPerspective(Visualizer):
    def draw_arrow(self, x_pos, y_pos, x_direct, y_direct, color=None, linestyle="-", linewidth=None):
        """
        Args:
            x_data (list[int]): a list containing x values of all the points being drawn.
                Length of list should match the length of y_data.
            y_data (list[int]): a list containing y values of all the points being drawn.
                Length of list should match the length of x_data.
            color: color of the line. Refer to `matplotlib.colors` for a full list of
                formats that are accepted.
            linestyle: style of the line. Refer to `matplotlib.lines.Line2D`
                for a full list of formats that are accepted.
            linewidth (float or None): width of the line. When it's None,
                a default value will be computed and used.
        Returns:
            output (VisImage): image object with line drawn.
        """
        if color is None:
            color = random_color(rgb=True, maximum=1)
        if linewidth is None:
            linewidth = self._default_font_size / 3
        linewidth = max(linewidth, 1)
        self.output.ax.quiver(
            x_pos, y_pos, x_direct, y_direct, color=color,scale_units='xy', scale=1, antialiased=True, headaxislength=3.5, linewidths=0.1#, width=0.01
        )
        return self.output
    
    def draw_lati(self, latimap, alpha_contourf=0.4, alpha_contour=0.9, contour_only=False):
        """
        latimap range should be in radians
        """
        height, width = latimap.shape
        y, x = np.mgrid[0:height, 0:width]
        cmap = plt.get_cmap('seismic_r')
        bands=20
        levels = np.linspace(-np.pi/2,np.pi/2,bands-1)
        if not contour_only:
            pp = self.output.ax.contourf(x, y, latimap, levels=levels, cmap=cmap, alpha=alpha_contourf, antialiased=True)
            pp2 = self.output.ax.contour(x,
                              y,
                              latimap,
                              pp.levels,
                              cmap=cmap,
                              alpha=alpha_contour,
                              antialiased=True,
                              linewidths=5)
            for c in pp2.collections:
                c.set_linestyle('solid')
        else:
            # only plot central contour
            pp = self.output.ax.contour(x,
                             y,
                             latimap,
                             levels=[0],
                             cmap=cmap,
                             alpha=alpha_contour,
                             antialiased=True,
                             linewidths=15)
        return self.output