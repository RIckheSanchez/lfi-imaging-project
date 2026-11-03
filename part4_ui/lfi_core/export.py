"""Raw CSV, pixel images and publication figures; no frontend dependencies."""
import io
from datetime import datetime
from pathlib import Path
from .. import APP_NAME, __version__
import numpy as np
import matplotlib
from matplotlib.figure import Figure
from PIL import Image

def export_csv(raw_slice, metadata=None):
    from .processing import finite_values
    finite_values(raw_slice)
    buffer = io.StringIO()
    np.savetxt(buffer, raw_slice, delimiter=",", fmt="%.6g")
    return (format_export_metadata(metadata) if metadata is not None else '') + buffer.getvalue()

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
    from .processing import finite_values
    finite_values(adjusted)
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
    from .processing import finite_values
    finite_values(adjusted)
    rgba = get_export_colormap(colormap)(adjusted)
    image = Image.fromarray((rgba * 255).astype(np.uint8))
    if image_format == 'JPEG':
        image = image.convert('RGB')
    buffer = io.BytesIO()
    options = {'dpi': (300, 300)} if image_format in ('PNG', 'TIFF') else {}
    image.save(buffer, format=image_format, **options)
    return buffer.getvalue()



def format_export_metadata(metadata):
    """One comma-free comment per field; values cannot inject additional lines."""
    def clean(value):
        return ' '.join(str(value).splitlines()).replace(',', ';')
    return ''.join(f'# {clean(key)}: {clean(value)}\n' for key, value in metadata.items())


def scan_export_metadata(data, slice_index, channel, timestamp=None):
    """Capture channel provenance, independent of Qt or translated UI labels."""
    state = data.metadata.get('ui_preprocessing', {})
    channel_state = state.get('channels', {}).get(channel, {})
    processed = channel_state.get('processed', bool(state) and channel == 'amplitude')
    if channel == 'amplitude':
        normalisation = state.get('normalisation', 'unknown') if state else 'none'
        units = state.get('amplitude_units', 'input units')
    else:
        normalisation = 'none (phase channel is not normalised)'
        units = 'radians' if data.metadata.get('phase_converted') is True else 'unconfirmed (raw input units)'
    metadata = {
        'App name': APP_NAME,
        'App version': __version__,
        'Export timestamp': timestamp or datetime.now().astimezone().isoformat(),
        'Source file': Path(data.file_path).name,
        'Target type': data.target_type,
        'Channel': channel,
        'Values': 'pre-processed' if processed else 'raw',
        'Slice index (zero-based)': slice_index,
        'Z value': data.z_indices[slice_index],
        'Shape': f'{data.height} rows (Y) x {data.width} columns (X)',
        'Pre-processing status': 'applied' if processed else ('unchanged (pipeline enabled)' if state else 'not applied'),
        'Normalisation': normalisation,
        'Units': units,
        'Numeric format': '%.6g',
    }
    if channel == 'phase':
        phase = data.metadata.get('part2_preprocessing', {})
        metadata['Phase encoding'] = data.metadata.get('phase_encoding_used') or 'unknown'
        metadata['Phase converted'] = data.metadata.get('phase_converted') is True
        metadata['Unwrap status'] = channel_state.get('unwrapping_status',
            phase.get('phase_unwrapping_status', {}).get('status', 'unknown (no processing metadata)'))
    return metadata


def view_png_filename(source_path, mode):
    return f'lfi_3d_{mode}_{Path(source_path).stem or "scan"}.png'


def view_export_metadata(data, slice_index, result, timestamp=None):
    """The 3D view is amplitude-based, regardless of the selected 2D channel."""
    metadata = scan_export_metadata(data, slice_index, 'amplitude', timestamp)
    metadata.update({
        'Export type': '3D viewport framebuffer PNG',
        'Slice scope': 'entire rendered volume; slice index and Z above are 2D selection context only',
        'Numeric format': 'not applicable to PNG (CSV uses %.6g)',
        '3D mode': result['mode'],
        'Percentile thresholds': result['percentile_range'],
        'Threshold values': result.get('thresholds', result.get('normalization_range', 'unknown')),
        'Downsampling step': result['step'],
        'Rendered volume shape (Z Y X)': ' x '.join(map(str, result['original_shape'])),
        'Displayed points': result['sampled_points'],
        'Axes': 'sample indices; physical XYZ proportions are not represented',
        'Part 3 reconstruction': result.get('ui_part3', 'disabled'),
    })
    return metadata
