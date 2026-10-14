"""Raw CSV, pixel images and publication figures; no frontend dependencies."""
import io
import numpy as np
import matplotlib
from matplotlib.figure import Figure
from PIL import Image

def export_csv(raw_slice):
    buffer = io.StringIO()
    np.savetxt(buffer, raw_slice, delimiter=",", fmt="%.6g")
    return buffer.getvalue()

def scale_bar(width, pixel_size):
    """Choose a 1/2/5 physical length within one quarter of image width."""
    target = width * pixel_size / 4
    exponent = np.floor(np.log10(target))
    base = 10 ** exponent
    length = max(value * base for value in (1, 2, 5) if value * base <= target)
    label = f"{length / 1000:g} mm" if length >= 1000 else f"{length:g} μm"
    return length / pixel_size, label

def get_export_colormap(name):
    cmap_name = {'Cool': 'cool', 'ice': 'cool'}.get(name, name.lower())
    return matplotlib.colormaps[cmap_name if cmap_name in matplotlib.colormaps else 'viridis']

def publication_png(adjusted, colormap, pixel_size=0, color_label='Intensity',
                    x_label='X (pixels)', y_label='Y (pixels)'):
    figure = Figure(figsize=(6, 5))
    axis = figure.subplots()
    image = axis.imshow(adjusted, cmap=get_export_colormap(colormap), vmin=0, vmax=1)
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    figure.colorbar(image, ax=axis, label=color_label)
    if pixel_size > 0:
        length, label = scale_bar(adjusted.shape[1], pixel_size)
        x, y = adjusted.shape[1] * .06, adjusted.shape[0] * .9
        axis.plot([x, x + length], [y, y], color='white', linewidth=3)
        axis.text(x + length / 2, y - adjusted.shape[0] * .03, label,
                  ha='center', color='black', bbox=dict(facecolor='white', edgecolor='none'))
    buffer = io.BytesIO()
    figure.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
    return buffer.getvalue()


def pixel_image(adjusted, colormap, image_format='PNG'):
    """Export full-resolution colored pixels; PNG/TIFF carry 300 DPI."""
    rgba = get_export_colormap(colormap)(adjusted)
    image = Image.fromarray((rgba[:, :, :3] * 255).astype(np.uint8))
    buffer = io.BytesIO()
    options = {'dpi': (300, 300)} if image_format in ('PNG', 'TIFF') else {}
    image.save(buffer, format=image_format, **options)
    return buffer.getvalue()
