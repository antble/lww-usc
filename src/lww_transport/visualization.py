"""Visualization helpers for LWW device geometry and phase-space data."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import os
from pathlib import Path

import numpy as np

from .config import LWWConfig


WignerInput = str | os.PathLike[str] | np.ndarray


@dataclass(frozen=True, slots=True)
class GeometryRegion:
    """Named interval in the one-dimensional RTD structure."""

    name: str
    start_nm: float
    end_nm: float
    potential_ev: float
    kind: str

    @property
    def width_nm(self) -> float:
        return self.end_nm - self.start_nm


def rtd_geometry_regions(cfg: LWWConfig | None = None) -> list[GeometryRegion]:
    """Return emitter/spacer/barrier/well/collector regions for the RTD."""
    cfg = cfg or LWWConfig.standard_rtd()
    box = float(cfg.geometry.box)
    center = 0.5 * box
    half_well = 0.5 * float(cfg.geometry.well)
    barrier = float(cfg.geometry.barrier)
    spacer = float(cfg.geometry.spacer)
    pot = float(cfg.geometry.pot)

    left_barrier_start = center - half_well - barrier
    left_barrier_end = center - half_well
    well_start = left_barrier_end
    well_end = center + half_well
    right_barrier_start = well_end
    right_barrier_end = well_end + barrier
    left_spacer_start = max(0.0, left_barrier_start - spacer)
    right_spacer_end = min(box, right_barrier_end + spacer)

    regions: list[GeometryRegion] = []

    def add(name: str, start: float, end: float, potential: float, kind: str) -> None:
        start = max(0.0, min(box, start))
        end = max(0.0, min(box, end))
        if end > start:
            regions.append(GeometryRegion(name, start, end, potential, kind))

    add("Emitter", 0.0, left_spacer_start, 0.0, "contact")
    add("Spacer", left_spacer_start, left_barrier_start, 0.0, "spacer")
    add("Barrier", left_barrier_start, left_barrier_end, pot, "barrier")
    add("Well", well_start, well_end, 0.0, "well")
    add("Barrier", right_barrier_start, right_barrier_end, pot, "barrier")
    add("Spacer", right_barrier_end, right_spacer_end, 0.0, "spacer")
    add("Collector", right_spacer_end, box, 0.0, "contact")
    return regions


def geometry_potential_profile(
    cfg: LWWConfig | None = None,
    points: int = 1200,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a piecewise RTD geometry potential profile."""
    cfg = cfg or LWWConfig.standard_rtd()
    if points < 2:
        raise ValueError("points must be at least 2")

    x = np.linspace(0.0, float(cfg.geometry.box), points)
    potential = np.zeros_like(x)
    for region in rtd_geometry_regions(cfg):
        if region.kind == "barrier":
            mask = (x >= region.start_nm) & (x <= region.end_nm)
            potential[mask] = region.potential_ev
    return x, potential


def plot_rtd_geometry(
    cfg: LWWConfig | None = None,
    ax=None,
    show_labels: bool = True,
    title: str | None = "Resonant tunneling diode geometry",
):
    """Draw the one-dimensional RTD geometry and return the Matplotlib axes."""
    cfg = cfg or LWWConfig.standard_rtd()
    if ax is None:
        import matplotlib.pyplot as plt

        _, ax = plt.subplots(figsize=(9.0, 3.4), constrained_layout=True)

    x, potential = geometry_potential_profile(cfg)
    regions = rtd_geometry_regions(cfg)
    ymax = max(float(cfg.geometry.pot) * 1.35, 0.1)

    colors = {
        "contact": "#e5e7eb",
        "spacer": "#bfdbfe",
        "barrier": "#fecaca",
        "well": "#bbf7d0",
    }
    box = float(cfg.geometry.box)
    label_y = ymax * 0.92
    width_label_y = ymax * 0.08
    x_span = max(box, 1.0)
    for region in regions:
        ax.axvspan(
            region.start_nm,
            region.end_nm,
            color=colors.get(region.kind, "#e5e7eb"),
            alpha=0.85,
            linewidth=0,
        )
        ax.axvline(region.start_nm, color="#6b7280", linewidth=0.7, alpha=0.45)
        if show_labels:
            center = 0.5 * (region.start_nm + region.end_nm)
            width_fraction = region.width_nm / x_span
            name_fontsize = 8 if width_fraction >= 0.07 else 7
            width_fontsize = 7 if width_fraction >= 0.06 else 6
            name_rotation = 90 if width_fraction < 0.045 else 0
            width_rotation = 90 if width_fraction < 0.04 else 0
            ax.text(
                center,
                label_y,
                region.name,
                ha="center",
                va="top",
                fontsize=name_fontsize,
                rotation=name_rotation,
                rotation_mode="anchor",
                color="#111827",
            )
            ax.text(
                center,
                width_label_y,
                f"{region.width_nm:.0f} nm",
                ha="center",
                va="bottom",
                fontsize=width_fontsize,
                rotation=width_rotation,
                rotation_mode="anchor",
                color="#374151",
            )

    ax.axvline(box, color="#6b7280", linewidth=0.7, alpha=0.45)
    ax.plot(x, potential, color="#111827", linewidth=2.2, drawstyle="steps-post")
    ax.set_xlim(0.0, box)
    ax.set_ylim(-0.03 * ymax, ymax)
    ax.set_xlabel("position x (nm)")
    ax.set_ylabel("potential energy (eV)")
    if title:
        ax.set_title(title)
    ax.grid(axis="y", color="#d1d5db", linewidth=0.7, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax


def save_rtd_geometry_image(
    cfg: LWWConfig | None = None,
    path: str | Path = "rtd_geometry.png",
    dpi: int = 160,
    show_labels: bool = True,
    title: str | None = "Resonant tunneling diode geometry",
) -> Path:
    """Save a PNG image of the RTD geometry and return its path."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig = Figure(figsize=(9.0, 3.4), constrained_layout=True)
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    plot_rtd_geometry(cfg, ax=ax, show_labels=show_labels, title=title)
    fig.savefig(output, dpi=dpi)
    return output


def _load_numeric_array(data: WignerInput) -> np.ndarray:
    if isinstance(data, (str, os.PathLike)):
        return np.loadtxt(Path(data), delimiter=",")
    return np.asarray(data, dtype=float)


def _wigner_matrix(wigner: WignerInput, cfg: LWWConfig | None = None) -> np.ndarray:
    values = _load_numeric_array(wigner)
    if values.ndim == 1:
        if cfg is None:
            raise ValueError("cfg is required when wigner is a flattened vector")
        if values.size != cfg.size:
            raise ValueError(f"wigner vector has {values.size} entries; expected {cfg.size}")
        return values.reshape(cfg.nx, cfg.n)

    if values.ndim != 2:
        raise ValueError("wigner must be a 1D vector, a 2D matrix, or a CSV file")

    if cfg is None:
        return values

    if values.shape == (cfg.nx, cfg.n):
        return values
    if values.shape == (cfg.n, cfg.nx):
        return values.T
    raise ValueError(f"wigner matrix has shape {values.shape}; expected {(cfg.nx, cfg.n)}")


def wigner_phase_space_grids(
    cfg: LWWConfig | None = None,
    shape: tuple[int, int] | None = None,
    centered_x: bool = True,
    x_unit: str = "um",
) -> tuple[np.ndarray, np.ndarray]:
    """Return spatial and wave-vector axes for Wigner phase-space plots."""
    if cfg is None:
        if shape is None:
            raise ValueError("shape is required when cfg is not supplied")
        nx, n = shape
        return np.arange(nx, dtype=float), np.arange(n, dtype=float)

    x = np.linspace(0.0, float(cfg.geometry.box), cfg.nx)
    if centered_x:
        x = x - 0.5 * float(cfg.geometry.box)

    if x_unit == "nm":
        x_scale = 1.0
    elif x_unit == "um":
        x_scale = 1.0e-3
    else:
        raise ValueError("x_unit must be 'nm' or 'um'")

    j = np.arange(1, cfg.n + 1, dtype=float)
    k = cfg.delk * (2.0 * j - cfg.n - 1.0)
    return x * x_scale, k


def _buffered_limits(
    values: np.ndarray,
    explicit: tuple[float, float] | None,
    buffer: float | tuple[float, float],
    name: str,
) -> tuple[float, float]:
    if explicit is not None:
        low, high = float(explicit[0]), float(explicit[1])
    else:
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            raise ValueError(f"{name} contains no finite values")
        low = float(finite.min())
        high = float(finite.max())
        span = high - low
        if span == 0.0:
            span = max(abs(high), 1.0)
        if isinstance(buffer, tuple):
            lower_buffer, upper_buffer = float(buffer[0]), float(buffer[1])
        else:
            lower_buffer = upper_buffer = float(buffer)
        low -= lower_buffer * span
        high += upper_buffer * span

    if not low < high:
        raise ValueError(f"{name} limits must be increasing")
    return low, high


def plot_wigner_phase_space(
    wigner: WignerInput,
    cfg: LWWConfig | None = None,
    ax=None,
    figsize: tuple[float, float] = (9.0, 6.4),
    x: Sequence[float] | None = None,
    k: Sequence[float] | None = None,
    title: str | None = "Wigner phase-space distribution",
    style: str = "standard",
    x_unit: str = "um",
    centered_x: bool = True,
    scale: float = 1.0,
    normalize: bool = False,
    x_lim: tuple[float, float] | None = None,
    k_lim: tuple[float, float] | None = None,
    z_lim: tuple[float, float] | None = None,
    x_buffer: float | tuple[float, float] | None = None,
    k_buffer: float | tuple[float, float] | None = None,
    z_buffer: float | tuple[float, float] | None = None,
    surface_cmap: str | None = None,
    contour_cmap: str | None = None,
    x_projection_cmap: str = "Reds",
    y_projection_cmap: str = "Blues",
    contour_levels: int = 40,
    contour_offset: float | None = None,
    surface_alpha: float | None = None,
    z_projection_alpha: float | None = None,
    x_projection_alpha: float | None = None,
    y_projection_alpha: float | None = None,
    z_projection: bool = True,
    x_projection: bool | None = None,
    y_projection: bool | None = None,
    colorbar: bool | None = None,
    colorbar_label: str | None = None,
    colorbar_shrink: float = 0.72,
    colorbar_pad: float = 0.08,
    colorbar_aspect: int = 18,
    x_projection_offset: float | None = None,
    y_projection_offset: float | None = None,
    surface_edgecolor: str = "none",
    surface_linewidth: float = 0.0,
    transparent_panes: bool | None = None,
    show_grid: bool | None = None,
    x_label: str | None = None,
    k_label: str | None = None,
    z_label: str | None = None,
    box_aspect: tuple[float, float, float] | None = (1.35, 1.0, 0.72),
    stride: int = 1,
    elev: float | None = None,
    azim: float | None = None,
):
    """Draw a 3D Wigner phase-space surface with a contour projection.

    ``wigner`` may be a flattened simulator vector, a ``(nx, n)`` matrix, or a
    CSV file containing either form. Flattened vectors require ``cfg`` so the
    phase-space dimensions can be restored.
    """
    style_key = style.lower()
    if style_key not in {"standard", "floating", "reference"}:
        raise ValueError("style must be 'standard', 'floating', or 'reference'")
    floating_style = style_key in {"floating", "reference"}

    if surface_cmap is None:
        surface_cmap = "RdBu_r" if floating_style else "Blues"
    if contour_cmap is None:
        contour_cmap = "coolwarm"
    if surface_alpha is None:
        surface_alpha = 0.45 if floating_style else 0.72
    if z_projection_alpha is None:
        z_projection_alpha = 0.8 if floating_style else 0.95
    if x_projection_alpha is None:
        x_projection_alpha = 0.3
    if y_projection_alpha is None:
        y_projection_alpha = 0.3
    if x_projection is None:
        x_projection = floating_style
    if y_projection is None:
        y_projection = floating_style
    if colorbar is None:
        colorbar = floating_style
    if transparent_panes is None:
        transparent_panes = floating_style
    if x_buffer is None:
        x_buffer = 1.0 / 3.0 if floating_style else 0.0
    if k_buffer is None:
        k_buffer = 0.25 if floating_style else 0.0
    if z_buffer is None:
        z_buffer = (0.35, 1.25) if floating_style else (0.25, 0.12)
    if elev is None:
        elev = 22.0 if floating_style else 24.0
    if azim is None:
        azim = -56.0 if floating_style else -62.0

    matrix = _wigner_matrix(wigner, cfg)
    nx, n = matrix.shape

    if x is None or k is None:
        default_x, default_k = wigner_phase_space_grids(
            cfg,
            shape=matrix.shape,
            centered_x=centered_x,
            x_unit=x_unit,
        )
        if x is None:
            x = default_x
        if k is None:
            k = default_k

    x_values = np.asarray(x, dtype=float)
    k_values = np.asarray(k, dtype=float)
    if x_values.shape != (nx,):
        raise ValueError(f"x has shape {x_values.shape}; expected {(nx,)}")
    if k_values.shape != (n,):
        raise ValueError(f"k has shape {k_values.shape}; expected {(n,)}")

    z = np.asarray(matrix, dtype=float) * float(scale)
    if normalize:
        max_abs = float(np.nanmax(np.abs(z)))
        if max_abs > 0.0:
            z = z / max_abs

    finite = z[np.isfinite(z)]
    if finite.size == 0:
        raise ValueError("wigner contains no finite values")

    x_limits = _buffered_limits(x_values, x_lim, x_buffer, "x")
    k_limits = _buffered_limits(k_values, k_lim, k_buffer, "k")
    z_limits = _buffered_limits(finite, z_lim, z_buffer, "z")
    z_span = z_limits[1] - z_limits[0]
    if contour_offset is None:
        contour_offset = z_limits[0]
    if x_projection_offset is None:
        x_projection_offset = x_limits[0]
    if y_projection_offset is None:
        y_projection_offset = k_limits[1]

    if ax is None:
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=figsize, constrained_layout=True)
        ax = fig.add_subplot(111, projection="3d")

    x_grid, k_grid = np.meshgrid(x_values, k_values, indexing="ij")
    stride = max(1, int(stride))
    surface = ax.plot_surface(
        x_grid,
        k_grid,
        z,
        rstride=stride,
        cstride=stride,
        cmap=surface_cmap,
        edgecolor=surface_edgecolor,
        linewidth=surface_linewidth,
        antialiased=True,
        alpha=surface_alpha,
    )
    if z_projection:
        ax.contourf(
            x_grid,
            k_grid,
            z,
            zdir="z",
            offset=contour_offset,
            levels=contour_levels,
            cmap=contour_cmap,
            alpha=z_projection_alpha,
        )
    if x_projection:
        ax.contourf(
            x_grid,
            k_grid,
            z,
            zdir="x",
            offset=x_projection_offset,
            levels=contour_levels,
            cmap=x_projection_cmap,
            alpha=x_projection_alpha,
        )
    if y_projection:
        ax.contourf(
            x_grid,
            k_grid,
            z,
            zdir="y",
            offset=y_projection_offset,
            levels=contour_levels,
            cmap=y_projection_cmap,
            alpha=y_projection_alpha,
        )

    ax.set(xlim=x_limits, ylim=k_limits, zlim=z_limits)
    ax.set_xlabel(x_label or (f"X ({x_unit})" if cfg is not None else "X"))
    ax.set_ylabel(k_label or ("K (1/nm)" if cfg is not None else "K"))
    ax.set_zlabel(z_label or ("normalized Wigner" if normalize else "Wigner"))
    if title:
        ax.set_title(title)
    if colorbar:
        cbar = ax.get_figure().colorbar(
            surface,
            ax=ax,
            shrink=colorbar_shrink,
            pad=colorbar_pad,
            aspect=colorbar_aspect,
        )
        cbar.set_label(colorbar_label or ("normalized Wigner" if normalize else "Wigner"))
    ax.view_init(elev=elev, azim=azim)
    if transparent_panes:
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    if show_grid is not None:
        ax.grid(show_grid)
    if box_aspect is not None:
        try:
            ax.set_box_aspect(box_aspect)
        except AttributeError:
            pass
    return ax


def save_wigner_phase_space_image(
    wigner: WignerInput,
    path: str | Path = "wigner_phase_space.png",
    cfg: LWWConfig | None = None,
    dpi: int = 170,
    **plot_kwargs,
) -> Path:
    """Save a 3D Wigner phase-space image and return its path."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig = Figure(figsize=plot_kwargs.pop("figsize", (9.0, 6.4)), constrained_layout=True)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111, projection="3d")
    plot_wigner_phase_space(wigner, cfg=cfg, ax=ax, **plot_kwargs)
    fig.savefig(output, dpi=dpi)
    return output


def save_wigner_phase_space_images(
    wigners: Mapping[object, WignerInput] | Sequence[WignerInput],
    output_dir: str | Path,
    cfg: LWWConfig | None = None,
    prefix: str = "wigner_phase_space",
    extension: str = "png",
    dpi: int = 170,
    **plot_kwargs,
) -> list[Path]:
    """Save phase-space images for multiple Wigner distributions."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    suffix = extension.lstrip(".")

    if isinstance(wigners, Mapping):
        items = list(wigners.items())
    else:
        items = list(enumerate(wigners))

    paths: list[Path] = []
    for key, wigner in items:
        key_text = str(key).replace("/", "_").replace(" ", "_")
        path = output / f"{prefix}_{key_text}.{suffix}"
        paths.append(save_wigner_phase_space_image(wigner, path, cfg=cfg, dpi=dpi, **plot_kwargs))
    return paths


__all__ = [
    "GeometryRegion",
    "geometry_potential_profile",
    "plot_rtd_geometry",
    "plot_wigner_phase_space",
    "rtd_geometry_regions",
    "save_rtd_geometry_image",
    "save_wigner_phase_space_image",
    "save_wigner_phase_space_images",
    "wigner_phase_space_grids",
]
