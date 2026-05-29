import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib as mpl

def plot_intensity(x, y, intensity_map, file_name = None, normalize = True, vmax = False):
    fig, ax = plt.subplots(figsize = (3.5, 3.5), dpi = 300)
    if normalize:
        intensity_map = intensity_map / np.max(intensity_map) 
    if not(vmax):
        vmax = np.max(intensity_map)
    plt.imshow(intensity_map,
            extent=[x[0], x[-1], y[0], y[-1]],
        origin='lower',
        aspect = "equal",
        vmin = np.min(intensity_map),
        vmax = vmax,
        cmap='Greys_r')
    ax.grid(False)
    step = (np.max(x) - np.min(x))/5
    x_ticks = np.arange(np.min(x), np.max(x) + step, step)
    y_ticks = np.arange(np.min(y), np.max(y) + step, step)
    plt.xticks(x_ticks, rotation=45)
    plt.yticks(y_ticks, rotation=45)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    if normalize:
        plt.colorbar(label='Intensity (Normalized)')
    else:
        plt.colorbar(label='Intensity (Raw)')
    plt.gca().set_aspect("equal")
    if not(file_name is None):
        plt.savefig(f"figures/{file_name}.png", bbox_inches = "tight")
    
    plt.show()

def plot_intensity_unnorm(x, y, intensity_map, file_name = None):
    fig, ax = plt.subplots(figsize = (3.5, 3.5), dpi = 300)
    plt.imshow(intensity_map,
            extent=[x[0], x[-1], y[0], y[-1]],
        origin='lower',
        aspect = "equal",
        cmap='Greys_r')
    ax.grid(False)
    step = (np.max(x) - np.min(x))/5
    x_ticks = np.arange(np.min(x), np.max(x) + step, step)
    y_ticks = np.arange(np.min(y), np.max(y) + step, step)
    plt.xticks(x_ticks, rotation=45)
    plt.yticks(y_ticks, rotation=45)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")

    plt.colorbar(label='Intensity (Unnormalized)')
    plt.gca().set_aspect("equal")
    if not(file_name is None):
        plt.savefig(f"figures/{file_name}.png", bbox_inches = "tight")
    
    plt.show()

def plot_many_intensity(axs, x, y, intensity_maps, is_edge = False, is_horizontal = True):
    for i, ax in enumerate(axs):
        if is_horizontal:
            if is_edge:
                ax.set_xlabel("x [mm]")
            if i == 0:
                ax.set_ylabel("y [mm]")
        else:
            if is_edge:
                ax.set_ylabel("y [mm]")
            if i == len(axs) - 1:
                ax.set_xlabel("x [mm]")
        ax.imshow(intensity_maps[i] / np.max(intensity_maps[i]),
            extent=[x[0], x[-1], y[0], y[-1]],
        origin='lower',
        aspect = "equal",
        vmin = 0,
        vmax = 1,
        cmap='Greys_r')
        ax.grid(False)
        ax.tick_params(axis='both', labelrotation=45)
        ax.set_xlabel("x [mm]")
        if i == 0:
            ax.set_ylabel("y [mm]")
            

def zernike_plot(z_map, alpha, file_name=None, res=500):
    fig = plt.figure(figsize=(2, 3), dpi=600)

    rho = np.linspace(0, 1, res)
    phi = np.linspace(0, 2*np.pi, res)
    rho_grid, phi_grid = np.meshgrid(rho, phi, indexing="ij")
    theta_grid = np.arcsin(rho_grid * np.sin(alpha))
    z = z_map(theta_grid, phi_grid)

    polar_ax = fig.add_axes([0.1, 0.25, 0.8, 0.7], projection="polar")

    pcm = polar_ax.pcolormesh(
        phi, rho, z,
        edgecolors="face",
        vmin=-np.pi,
        vmax=np.pi,
        cmap="plasma"
    )

    polar_ax.grid(False)
    polar_ax.set_xticklabels([])
    polar_ax.set_yticklabels([])

    cbar = fig.colorbar(
        pcm,
        ax=polar_ax,
        orientation="horizontal",
        pad=0.15,
        fraction=0.08
    )
    cbar.set_label("Phase (rad)")

    if file_name is not None:
        plt.savefig(f"{file_name}.png", bbox_inches="tight")

    plt.show()
    
def composite_plot(x, y, intensity_map, z_map, alpha, file_name = None):
    fig, ax = plt.subplots(dpi = 300)
    plt.imshow(intensity_map / np.max(intensity_map),
            extent=[x[0], x[-1], y[0], y[-1]],
        origin='lower',
        aspect = "equal",
        cmap='Greys')
    ax.grid(False)
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    rho = np.linspace(0, 1, 500)
    phi = np.linspace(0, 2*np.pi, 500)
    rho_grid, phi_grid = np.meshgrid(rho, phi)
    rho = np.linspace(0, 1, 500)
    phi = np.linspace(0, 2*np.pi, 500)
    rho_grid, phi_grid = np.meshgrid(rho, phi, indexing="ij")  
    theta_grid = np.arcsin(rho_grid * np.sin(alpha))           
    z = z_map(theta_grid, phi_grid)                          
    size_param = 0.05
    polar_ax= fig.add_axes([0.58 - size_param, 0.68 - size_param, 0.2 + size_param, 0.2 + size_param], projection = "polar")
    polar_ax.pcolormesh(phi,rho,z,edgecolors='face', vmin = -2*np.pi, vmax = 2*np.pi,
                        cmap = "plasma")
    polar_ax.grid(False)
    polar_ax.set_xticklabels([])
    polar_ax.set_yticklabels([])
    plt.colorbar(label='Intensity (Normalized)')
    if not(file_name is None):
        plt.savefig(f"figures/{file_name}.png", bbox_inches = "tight")
    plt.show()

def many_composite(fig, axs, x, y, intensity_maps, z_maps, alpha, vmin = -2*np.pi, vmax = 2*np.pi, is_edge = False, is_horizontal = True):
    axs = np.ravel(axs)
    v = max([np.max(i_map) for i_map in intensity_maps])
    for i, ax in enumerate(axs):
        ax.imshow(intensity_maps[i],
                  vmax = v,
                  extent=[x[0], x[-1], y[0], y[-1]],
                  origin='lower',
                  aspect="equal",
                  cmap='Greys_r')
        ax.grid(False)

        ax.tick_params(axis='x', labelrotation=45)
        ax.tick_params(axis='y', labelrotation=45)
        if is_horizontal:
            if is_edge:
                ax.set_xlabel("x [mm]")
            if i == 0:
                ax.set_ylabel("y [mm]")
        else:
            if is_edge:
                ax.set_ylabel("y [mm]")
            if i == len(axs) - 1:
                ax.set_xlabel("x [mm]")

        rho = np.linspace(0, 1, 500)
        phi = np.linspace(0, 2*np.pi, 500)
        rho_grid, phi_grid = np.meshgrid(rho, phi, indexing="ij")
        theta_grid = np.arcsin(rho_grid * np.sin(alpha))
        z = z_maps[i](theta_grid, phi_grid)
        inset = inset_axes(ax, width="35%", height="35%",
                           loc="upper right", borderpad=0)

        fig.canvas.draw()
        bbox = inset.get_window_extent(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
        inset.remove()
        polar_ax = fig.add_axes([bbox.x0, bbox.y0, bbox.width, bbox.height], projection="polar")
        

        polar_ax.pcolormesh(phi, rho, z,
                            shading='auto',
                            vmin=vmin, vmax=vmax)
        polar_ax.grid(False)
        polar_ax.set_xticklabels([])
        polar_ax.set_yticklabels([])