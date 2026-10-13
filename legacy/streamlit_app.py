"""
LFI Imaging Viewer - Final Version with Real Modules
Integrated with Zeyu Wang's loader.py and data_types.py
Author: He Zhenyu (49174058)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "files"))

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import matplotlib
from matplotlib.figure import Figure
import io
import tempfile
import os
import h5py
import hashlib


# ── Translations ──────────────────────────────────────────────────

STRINGS = {
    'render_mode': {"EN": '3D mode', "CN": '三维模式'},
    'volume_mode': {"EN": 'Volume', "CN": '体渲染'},
    'isosurface_mode': {"EN": 'Isosurface', "CN": '等值面'},
    'iso_percentile': {"EN": 'Isomin percentile', "CN": '等值面下限百分位'},

    'export_kind': {"EN": 'Image type', "CN": '图像类型'},
    'pixel_image': {"EN": 'Pixel image', "CN": '像素图像'},
    'publication': {"EN": 'Publication figure', "CN": '论文插图'},

    'pixel_size': {"EN": 'Pixel size (μm)', "CN": '像素尺寸（μm）'},
    'scale_hint': {"EN": 'Set a positive pixel size to display a scale bar.', "CN": '请输入大于 0 的像素尺寸以显示比例尺。'},

    'raw_value': {"EN": 'Raw value', "CN": '原始值'},
    'pixel_x': {"EN": 'Pixel X', "CN": '像素 X'},
    'pixel_y': {"EN": 'Pixel Y', "CN": '像素 Y'},
    'selected_value': {"EN": '**Selected pixel value:**', "CN": '**所选像素值：**'},

    # Sidebar — data import
    'data_import':          {'EN': '📁 Data Import',                         'CN': '📁 数据导入'},
    'load_real_title':      {'EN': '🎯 Load Data',                           'CN': '🎯 加载数据'},
    'modules_detected':     {'EN': '✅ loader.py and data_types.py detected', 'CN': '✅ 检测到 loader.py 和 data_types.py'},
    'no_modules_warning':   {'EN': '⚠️ loader.py or data_types.py not found. Place them in the same folder and restart.',
                              'CN': '⚠️ 未找到 loader.py 或 data_types.py，请将文件放在同目录后重启。'},
    'upload_label':         {'EN': 'Upload HDF5 file',                       'CN': '上传 HDF5 文件'},
    'upload_help':          {'EN': 'Upload a real LFI scan file',             'CN': '上传真实的 LFI 扫描数据'},
    'or':                   {'EN': '**or**',                                  'CN': '**或者**'},
    'use_local_path':       {'EN': 'Use local file path (supports large files)', 'CN': '使用本地文件路径（支持大文件）'},
    'path_input_label':     {'EN': 'Full path to HDF5 file',                 'CN': '输入 HDF5 文件完整路径'},
    'path_placeholder':     {'EN': '/Users/vincent/Desktop/LFI Viewer/Realdata/chip_borad1.hdf5',
                              'CN': '/Users/vincent/Desktop/LFI Viewer/Realdata/chip_borad1.hdf5'},
    'path_help':            {'EN': 'Supports files of any size',              'CN': '支持任意大小的文件'},
    'btn_load_path':        {'EN': '📥 Load from Path',                      'CN': '📥 从路径加载'},
    'file_not_found':       {'EN': '❌ File not found: ',                    'CN': '❌ 文件不存在：'},
    'loading_path':         {'EN': 'Loading from local path...',              'CN': '正在从本地路径加载...'},
    'load_success':         {'EN': '✅ Loaded successfully!',                 'CN': '✅ 加载成功！'},
    'data_summary':         {'EN': '📊 Data Summary',                        'CN': '📊 数据摘要'},
    'load_failed':          {'EN': '❌ Load failed: ',                        'CN': '❌ 加载失败：'},
    'btn_load_upload':      {'EN': '📥 Load Uploaded File',                  'CN': '📥 加载上传文件'},
    'loading_upload':       {'EN': 'Loading via loader.py...',                'CN': '正在使用 loader.py 加载...'},
    'load_real_success':    {'EN': '✅ Data loaded successfully!',            'CN': '✅ 数据加载成功！'},

    # Sidebar — display settings
    'display_settings':     {'EN': '### Display Settings',                   'CN': '### 显示参数'},
    'channel_label':        {'EN': 'Channel',                                'CN': '通道'},
    'channel_help':         {'EN': 'Amplitude = reflectivity / structure. Phase = depth / sub-surface detail.',
                              'CN': '振幅 = 反射率 / 结构轮廓。相位 = 深度 / 内部细节。'},
    'channel_amp':          {'EN': 'Amplitude',                              'CN': '振幅'},
    'channel_phase':        {'EN': 'Phase',                                  'CN': '相位'},
    'brightness':           {'EN': 'Brightness',                             'CN': '亮度'},
    'contrast':             {'EN': 'Contrast',                               'CN': '对比度'},
    'colormap':             {'EN': 'Colormap',                               'CN': '色彩映射'},

    # Welcome screen
    'welcome_info':         {'EN': '👈 Load a dataset from the sidebar to get started.',
                              'CN': '👈 请从左侧边栏加载数据。'},
    'welcome_how':          {'EN': '### How to load data',                   'CN': '### 如何加载数据'},
    'welcome_steps':        {'EN': '1. **Upload** an HDF5 file directly, or\n2. **Check** "Use local file path" and paste the full path to your file',
                              'CN': '1. 直接**上传** HDF5 文件，或\n2. **勾选**"使用本地文件路径"并粘贴文件完整路径'},
    'welcome_formats':      {'EN': '### Supported formats',                  'CN': '### 支持的格式'},
    'welcome_formats_desc': {'EN': '- Standard LFI format: `/amplitude/<z>`, `/phase/<z>`\n- Frame format: `/frames/<id>/Image Data/QCL0/Left/IMAGE_AMPLITUDE`',
                              'CN': '- 标准 LFI 格式：`/amplitude/<z>`、`/phase/<z>`\n- 帧格式：`/frames/<id>/Image Data/QCL0/Left/IMAGE_AMPLITUDE`'},

    # Data source banner
    'using_real_data':      {'EN': '✅ Viewing <b>real HDF5 data</b> loaded via loader.py',
                              'CN': '✅ 当前使用<b>真实 HDF5 数据</b>（通过 loader.py 加载）'},

    # Tab names
    'tab_2d':               {'EN': '📊 2D Slice',          'CN': '📊 二维切片'},
    'tab_3d':               {'EN': '🎲 3D Reconstruction',  'CN': '🎲 三维重建'},
    'tab_waveform':         {'EN': '📈 Waveform',           'CN': '📈 波形分析'},
    'tab_export':           {'EN': '💾 Export',             'CN': '💾 导出'},

    # Tab 1 — 2D Slice
    'slice_viewer_title':   {'EN': '2D Slice Viewer',       'CN': '二维切片查看器'},
    'z_depth':              {'EN': 'Z Depth',               'CN': 'Z 深度'},
    'z_index':              {'EN': 'Z Index: ',             'CN': 'Z 层编号：'},
    'single_layer':         {'EN': 'Single-layer 2D data.', 'CN': '单层 2D 数据。'},
    'scan_info':            {'EN': '### Scan Info',         'CN': '### 扫描信息'},
    'target':               {'EN': '**Target:**',           'CN': '**目标类型：**'},
    'size':                 {'EN': '**Size:**',             'CN': '**尺寸：**'},
    'slices':               {'EN': '**Slices:**',           'CN': '**层数：**'},
    'z_range':              {'EN': '**Z Range:**',          'CN': '**Z 范围：**'},
    'file':                 {'EN': '**File:**',             'CN': '**文件：**'},
    'pixel_info':           {'EN': '### Pixel Info',        'CN': '### 像素信息'},
    'channel_info':         {'EN': '**Channel:**',          'CN': '**通道：**'},
    'current_z':            {'EN': '**Current Z:**',        'CN': '**当前 Z：**'},
    'center':               {'EN': '**Center:**',           'CN': '**中心：**'},
    'value_center':         {'EN': '**Value at center:**',  'CN': '**中心点数值：**'},
    'range':                {'EN': '**Range:**',            'CN': '**数值范围：**'},
    'histogram':            {'EN': '### Histogram',         'CN': '### 强度直方图'},
    'intensity':            {'EN': 'Intensity',             'CN': '强度'},
    'phase_rad':            {'EN': 'Phase (rad)',           'CN': '相位（rad）'},
    'count':                {'EN': 'Count',                 'CN': '计数'},
    'x_pixels':             {'EN': 'X (pixels)',            'CN': 'X（像素）'},
    'y_pixels':             {'EN': 'Y (pixels)',            'CN': 'Y（像素）'},

    # Tab 2 — 3D
    '3d_title':             {'EN': '### 3D Volume Reconstruction', 'CN': '### 三维体积重建'},
    'rendering_info':       {'EN': '### Rendering Info',           'CN': '### 渲染信息'},
    'rotate_hint':          {'EN': '🔄 **Rotate:** Drag\n\n🔍 **Zoom:** Scroll',
                              'CN': '🔄 **旋转：** 拖拽\n\n🔍 **缩放：** 滚轮'},
    'btn_reset_camera':     {'EN': '↩️ Reset Camera',             'CN': '↩️ 重置视角'},
    'btn_render_3d':        {'EN': '▶️ Render / Refresh 3D',      'CN': '▶️ 渲染 / 刷新三维'},
    'volume_info':          {'EN': '### Volume Info',              'CN': '### 体积信息'},
    'dimensions':           {'EN': '**Dimensions:**',              'CN': '**数据维度：**'},
    'total_points':         {'EN': '**Total Points:**',            'CN': '**总点数：**'},
    'displayed_points':     {'EN': '**Displayed:**',               'CN': '**显示点数：**'},
    'downsample':           {'EN': '**Downsample:**',              'CN': '**降采样：**'},
    'render_hint':          {'EN': '👉 Click **▶️ Render / Refresh 3D** on the right to start rendering.',
                              'CN': '👉 点击右侧 **▶️ 渲染 / 刷新三维** 按钮开始渲染。'},
    '3d_single_layer':      {'EN': 'Single-layer 2D data — 3D reconstruction is not available.',
                              'CN': '单层 2D 数据，无法进行三维重建。'},
    'x_axis':               {'EN': 'X',        'CN': 'X'},
    'y_axis':               {'EN': 'Y',        'CN': 'Y'},
    'z_axis':               {'EN': 'Z Depth',  'CN': 'Z 深度'},

    # Tab 3 — Waveform
    'waveform_title':           {'EN': '### Signal Waveform Analysis', 'CN': '### 信号波形分析'},
    'x_position':               {'EN': 'X Position',                   'CN': 'X 坐标'},
    'y_position':               {'EN': 'Y Position',                   'CN': 'Y 坐标'},
    'amp_profile_title':        {'EN': 'Amplitude Profile at',         'CN': '振幅剖面（坐标'},
    'phase_profile_title':      {'EN': 'Phase Profile at',             'CN': '相位剖面（坐标'},
    'z_index_axis':             {'EN': 'Z Index',                      'CN': 'Z 层编号'},
    'amplitude_axis':           {'EN': 'Amplitude',                    'CN': '振幅'},
    'waveform_single_layer':    {'EN': 'Single-layer data — Z-depth waveform is not available.',
                                  'CN': '单层数据，无法显示 Z 深度波形。'},

    # Tab 4 — Export
    'export_title':         {'EN': '### Export Data & Images',    'CN': '### 数据与图像导出'},
    'export_slice_title':   {'EN': '#### 📄 Export Current Slice','CN': '#### 📄 导出当前切片'},
    'export_slice_caption': {'EN': 'Exports with current channel, colormap, brightness & contrast applied.',
                              'CN': '导出时将应用当前通道、色彩映射及亮度/对比度设置。'},
    'format_label':         {'EN': 'Format',                      'CN': '格式'},
    'btn_export_image':     {'EN': 'Export Image',                'CN': '导出图像'},
    'btn_download':         {'EN': 'Download',                    'CN': '下载'},
    'export_csv_title':     {'EN': '#### 📊 Export Raw Data',    'CN': '#### 📊 导出原始数据'},
    'export_csv_caption':   {'EN': 'Exports raw values of the selected channel for the current slice as CSV.',
                              'CN': '将当前切片的所选通道的原始数值导出为 CSV 文件。'},
    'btn_export_csv':       {'EN': 'Export CSV',                  'CN': '导出 CSV'},
    'btn_download_csv':     {'EN': 'Download CSV',                'CN': '下载 CSV'},
}

def t(key):
    """Return the UI string for the current language."""
    lang = st.session_state.get('lang', 'EN')
    return STRINGS[key][lang]


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


def create_3d_trace(X, Y, Z, values, colorscale, mode='volume', percentile=60):
    coordinates = dict(x=X.ravel(), y=Y.ravel(), z=Z.ravel(), value=values.ravel(),
                       colorscale=colorscale)
    if mode == 'isosurface':
        return go.Isosurface(**coordinates, isomin=float(np.percentile(values, percentile)),
                             isomax=float(np.percentile(values, 95)), surface_count=3,
                             caps=dict(x_show=False, y_show=False, z_show=False))
    return go.Volume(**coordinates, isomin=float(values.min()), isomax=float(values.max()),
                     opacity=.1, surface_count=17)


# ── Cached helpers ────────────────────────────────────────────────

@st.cache_data
def get_normalized_slice(_data, data_key, z_index, channel='amplitude'):
    """Return normalized [0,1] slice for the given channel ('amplitude' or 'phase')."""
    slice_data = _data.get_slice(z_index)
    arr = slice_data[channel]
    s_min, s_max = arr.min(), arr.max()
    normalized = (arr - s_min) / (s_max - s_min + 1e-10)
    return normalized, s_min, s_max

@st.cache_data
def get_downsampled_volume(_data, data_key, max_points=200_000):
    """Cache and downsample 3D amplitude volume for fast rendering."""
    amp   = _data.amplitude
    total = amp.size
    step  = max(1, int((total / max_points) ** (1 / 3)))
    sampled = amp[::step, ::step, ::step]
    Z, Y, X = np.mgrid[
        0:amp.shape[0]:step,
        0:amp.shape[1]:step,
        0:amp.shape[2]:step
    ]
    return X, Y, Z, sampled, step


# ── Module imports ────────────────────────────────────────────────

try:
    from loader import load_single_scan
    from data_types import LFIRawData
    REAL_MODULES_AVAILABLE = True
except ImportError:
    REAL_MODULES_AVAILABLE = False


# ── Loaders ───────────────────────────────────────────────────────

def _load_frames_format(filepath: str):
    """Fallback loader for HDF5 files using /frames/.../IMAGE_AMPLITUDE structure."""
    with h5py.File(filepath, 'r') as f:
        frames_group = f['frames']
        frame_keys   = sorted(frames_group.keys(), key=lambda x: int(x))

        amp_slices, phase_slices, z_indices = [], [], []
        for k in frame_keys:
            base = frames_group[k]['Image Data']['QCL0']['Left']
            amp_slices.append(base['IMAGE_AMPLITUDE'][:].astype(np.float64))
            phase_slices.append(base['IMAGE_PHASE'][:].astype(np.float64))
            z_indices.append(int(k))

        amplitude = np.stack(amp_slices,   axis=0)
        phase     = np.stack(phase_slices, axis=0)
        metadata  = {k: v for k, v in f.attrs.items()}

    stem   = os.path.splitext(os.path.basename(filepath))[0]
    known  = ['Chip', 'Leaf', 'Skin', 'Marble', 'Mushroom', 'Sound', 'Tomography']
    target = next((tp for tp in known if tp.lower() in stem.lower()), 'Unknown')

    return LFIRawData(
        amplitude=amplitude, phase=phase, z_indices=z_indices,
        target_type=target, file_path=filepath, metadata=metadata
    )


def smart_load(filepath: str):
    """Try standard loader first; fall back to frames-format loader."""
    try:
        return load_single_scan(filepath)
    except ValueError as e:
        if 'frames' in str(e) or 'amplitude' in str(e):
            return _load_frames_format(filepath)
        raise


def load_uploaded_bytes(name, data):
    """Load upload bytes and return a stable key; always clean up the temporary file."""
    data_key = ("upload", name, len(data), hashlib.sha1(data).hexdigest())
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.hdf5') as tmp_file:
            tmp_path = tmp_file.name
            tmp_file.write(data)
        loaded_data = smart_load(tmp_path)
        if loaded_data.target_type == 'Unknown':
            known_types = ['Chip', 'Leaf', 'Skin', 'Marble', 'Mushroom',
                           'Sound', 'Tomography', 'Beam Profile', 'Spectrum']
            loaded_data.target_type = next(
                (target for target in known_types if target.lower() in name.lower()), 'Unknown')
        return loaded_data, data_key
    finally:
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


# ── Constants ─────────────────────────────────────────────────────

COLORMAP_MAPPING = {
    'Viridis': 'Viridis',
    'Plasma':  'Plasma',
    'Inferno': 'Inferno',
    'Magma':   'Magma',
    'Turbo':   'Turbo',
    'Hot':     'Hot',
    'Cool':    'ice',
}

# ── Page config ───────────────────────────────────────────────────

st.set_page_config(
    page_title="LFI Imaging Viewer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #065A82;
        margin-bottom: 1rem;
    }
    .hero-title {
        font-size: 4.5rem;
        font-weight: 900;
        color: #065A82;
        letter-spacing: -1px;
        line-height: 1.1;
        margin-bottom: 0.5rem;
    }
    .hero-subtitle {
        font-size: 1.2rem;
        color: #888;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #02C39A;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.25rem;
        padding: 0.75rem;
        color: #155724;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ── Session state defaults ────────────────────────────────────────

for key, default in {
    'data':          None,
    'data_key':      None,
    'current_z_idx': 0,
    'brightness':    60,
    'contrast':      75,
    'colormap':      'Viridis',
    'channel':       'Amplitude',
    'lang':          'EN',
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ── Sidebar ───────────────────────────────────────────────────────

with st.sidebar:
    # Language toggle
    lang_col1, lang_col2 = st.columns(2)
    with lang_col1:
        if st.button("🇬🇧 English", use_container_width=True,
                     type="primary" if st.session_state.lang == 'EN' else "secondary"):
            st.session_state.lang = 'EN'
            st.rerun()
    with lang_col2:
        if st.button("🇨🇳 中文", use_container_width=True,
                     type="primary" if st.session_state.lang == 'CN' else "secondary"):
            st.session_state.lang = 'CN'
            st.rerun()

    st.divider()
    st.markdown(f'<p class="main-header">{t("data_import")}</p>', unsafe_allow_html=True)

    if not REAL_MODULES_AVAILABLE:
        st.error(t('no_modules_warning'))
    else:
        st.markdown(f"### {t('load_real_title')}")
        st.success(t('modules_detected'))

        uploaded_file = st.file_uploader(
            t('upload_label'),
            type=['hdf5', 'h5'],
            help=t('upload_help')
        )

        st.markdown(t('or'))
        use_local_path = st.checkbox(t('use_local_path'))

        if use_local_path:
            file_path = st.text_input(
                t('path_input_label'),
                placeholder=t('path_placeholder'),
                help=t('path_help')
            )
            if file_path and st.button(t('btn_load_path'), type="primary"):
                if os.path.exists(file_path):
                    try:
                        with st.spinner(t('loading_path')):
                            resolved_path = os.path.realpath(os.path.expanduser(file_path))
                            loaded_data = smart_load(resolved_path)
                            loaded_key = (resolved_path, os.path.getmtime(resolved_path),
                                          os.path.getsize(resolved_path))
                            st.session_state.data = loaded_data
                            st.session_state.data_key = loaded_key
                            st.session_state.current_z_idx = 0
                        st.success(t('load_success'))
                        st.balloons()
                        with st.expander(t('data_summary')):
                            st.code(st.session_state.data.summary())
                    except Exception as e:
                        st.error(t('load_failed') + str(e))
                else:
                    st.error(t('file_not_found') + file_path)

        if uploaded_file is not None:
            if st.button(t('btn_load_upload'), type="primary", use_container_width=True):
                try:
                    with st.spinner(t('loading_upload')):
                        loaded_data, loaded_key = load_uploaded_bytes(
                            uploaded_file.name, uploaded_file.getvalue())
                        st.session_state.data = loaded_data
                        st.session_state.data_key = loaded_key
                        st.session_state.current_z_idx = 0
                    st.success(t('load_real_success'))
                    st.balloons()
                    with st.expander(t('data_summary')):
                        st.code(st.session_state.data.summary())
                except Exception as e:
                    st.error(t('load_failed') + str(e))
                    st.exception(e)

    st.divider()

    if st.session_state.data:
        st.markdown(t('display_settings'))

        ch_options = [t('channel_amp'), t('channel_phase')]
        ch_index   = 1 if st.session_state.channel == 'Phase' else 0
        ch_choice  = st.radio(
            t('channel_label'),
            ch_options,
            index=ch_index,
            horizontal=True,
            help=t('channel_help')
        )
        st.session_state.channel = 'Phase' if ch_choice == t('channel_phase') else 'Amplitude'

        st.session_state.brightness = st.slider(t('brightness'), 0, 100, st.session_state.brightness)
        st.session_state.contrast   = st.slider(t('contrast'),   0, 100, st.session_state.contrast)
        st.session_state.colormap   = st.selectbox(
            t('colormap'),
            ["Viridis", "Plasma", "Inferno", "Magma", "Turbo", "Hot", "Cool"]
        )


    pixel_size = st.number_input(t('pixel_size'), min_value=0.0, value=0.0, key='pixel_size_um')
    if pixel_size == 0:
        st.caption(t('scale_hint'))


# ── Main content ──────────────────────────────────────────────────

st.markdown('<p style="font-size:4.5rem; font-weight:900; color:#065A82; letter-spacing:-1px; line-height:1.1; margin-bottom:0.2rem;">🔬 LFI Imaging Viewer</p>', unsafe_allow_html=True)
st.markdown('<p style="font-size:1.1rem; color:#999; margin-top:0; margin-bottom:1.5rem;">Laser Feedback Interferometry</p>', unsafe_allow_html=True)

if st.session_state.data is None:
    st.info(t('welcome_info'))
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(t('welcome_how'))
        st.markdown(t('welcome_steps'))
    with col2:
        st.markdown(t('welcome_formats'))
        st.markdown(t('welcome_formats_desc'))

else:
    st.markdown(
        f'<div class="success-box">{t("using_real_data")}</div>',
        unsafe_allow_html=True
    )

    data_key = st.session_state.data_key

    tab1, tab2, tab3, tab4 = st.tabs([
        t('tab_2d'), t('tab_3d'), t('tab_waveform'), t('tab_export')
    ])

    # ── Tab 1: 2D Slice Viewer ────────────────────────────────────
    with tab1:
        col_main, col_info = st.columns([2, 1])
        channel_key = st.session_state.channel.lower()

        with col_main:
            ch_display = t('channel_phase') if channel_key == 'phase' else t('channel_amp')
            st.markdown(f"### {t('slice_viewer_title')} — **{ch_display}**")

            n_slices  = st.session_state.data.n_slices
            z_indices = st.session_state.data.z_indices

            if n_slices > 1:
                st.session_state.current_z_idx = st.slider(
                    t('z_depth'),
                    min_value=0,
                    max_value=n_slices - 1,
                    value=st.session_state.current_z_idx,
                    format=f"%d / {n_slices - 1}"
                )
                current_z = z_indices[st.session_state.current_z_idx]
                st.caption(f"{t('z_index')}{current_z}")
            else:
                st.session_state.current_z_idx = 0
                current_z = z_indices[0]
                st.info(t('single_layer'))

            normalized, s_min, s_max = get_normalized_slice(
                st.session_state.data, data_key, current_z, channel_key
            )
            raw_slice = st.session_state.data.get_slice(current_z)[channel_key]

            brightness_offset = (st.session_state.brightness - 50) / 100
            contrast_scale    = st.session_state.contrast / 50
            adjusted = (normalized - 0.5) * contrast_scale + 0.5 + brightness_offset
            adjusted = np.clip(adjusted, 0, 1)

            color_label = t('phase_rad') if channel_key == 'phase' else t('intensity')

            fig = px.imshow(
                adjusted,
                color_continuous_scale=COLORMAP_MAPPING.get(st.session_state.colormap, 'Viridis'),
                aspect='equal',
                labels={'color': color_label},
                zmin=0, zmax=1
            )
            fig.update_traces(customdata=raw_slice,
                              hovertemplate="x=%{x}<br>y=%{y}<br>" + t('raw_value') + "=%{customdata:.6g}<extra></extra>")
            fig.update_layout(
                width=600, height=600,
                xaxis_title=t('x_pixels'),
                yaxis_title=t('y_pixels'),
                coloraxis_colorbar=dict(title=color_label),
                margin=dict(l=0, r=0, t=30, b=0)
            )
            if pixel_size > 0:
                bar_pixels, bar_label = scale_bar(raw_slice.shape[1], pixel_size)
                bar_x, bar_y = raw_slice.shape[1] * 0.06, raw_slice.shape[0] * 0.9
                fig.add_shape(type="line", x0=bar_x, x1=bar_x + bar_pixels,
                              y0=bar_y, y1=bar_y, line=dict(color="white", width=5))
                fig.add_annotation(x=bar_x + bar_pixels / 2, y=bar_y,
                                   text=bar_label, showarrow=False, yshift=15,
                                   bgcolor="white", font=dict(color="black"))
            st.plotly_chart(fig, use_container_width=True)

        with col_info:
            st.markdown(t('scan_info'))
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.write(f"{t('target')} {st.session_state.data.target_type}")
            st.write(f"{t('size')} {st.session_state.data.width}×{st.session_state.data.height}")
            st.write(f"{t('slices')} {st.session_state.data.n_slices}")
            st.write(f"{t('z_range')} {z_indices[0]} → {z_indices[-1]}")
            st.write(f"{t('file')} {st.session_state.data.file_path}")
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown(t('pixel_info'))
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.write(f"{t('channel_info')} {ch_display}")
            st.write(f"{t('current_z')} {current_z}")
            cx, cy = st.session_state.data.width // 2, st.session_state.data.height // 2
            pixel_x = st.number_input(t('pixel_x'), 0, raw_slice.shape[1] - 1, cx)
            pixel_y = st.number_input(t('pixel_y'), 0, raw_slice.shape[0] - 1, cy)
            st.write(f"{t('selected_value')} {raw_slice[pixel_y, pixel_x]:.6g}")
            st.write(f"{t('range')} {raw_slice.min():.3f} – {raw_slice.max():.3f}")
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown(t('histogram'))
            hist_fig = go.Figure(data=[go.Histogram(x=raw_slice.flatten(), nbinsx=50)])
            hist_fig.update_layout(
                height=250,
                margin=dict(l=0, r=0, t=20, b=20),
                xaxis_title=color_label,
                yaxis_title=t('count'),
                showlegend=False
            )
            st.plotly_chart(hist_fig, use_container_width=True)

    # ── Tab 2: 3D Reconstruction ──────────────────────────────────
    with tab2:
        st.markdown(t('3d_title'))

        if st.session_state.data.n_slices > 1:
            col_3d, col_3d_info = st.columns([2, 1])

            with col_3d_info:
                st.markdown(t('rendering_info'))
                render_mode = st.radio(t('render_mode'), ['volume_mode', 'isosurface_mode'], format_func=t)
                iso_percentile = st.slider(t('iso_percentile'), 0, 94, 60) if render_mode == 'isosurface_mode' else 60
                st.info(t('rotate_hint'))
                if st.button(t('btn_reset_camera'), use_container_width=True):
                    st.session_state['3d_camera'] = None
                render_3d = st.button(t('btn_render_3d'), type="primary", use_container_width=True)

                amp_shape = st.session_state.data.amplitude.shape
                st.markdown(t('volume_info'))
                st.write(f"{t('dimensions')} {amp_shape}")
                st.write(f"{t('total_points')} {int(np.prod(amp_shape)):,}")

            with col_3d:
                if render_3d or st.session_state.get('3d_rendered'):
                    st.session_state['3d_rendered'] = True
                    X, Y, Z, values, step = get_downsampled_volume(st.session_state.data, data_key)
                    camera = st.session_state.get('3d_camera') or dict(eye=dict(x=1.5, y=1.5, z=1.5))

                    fig_3d = go.Figure(data=create_3d_trace(
                        X, Y, Z, values,
                        COLORMAP_MAPPING.get(st.session_state.colormap, 'Viridis'),
                        'isosurface' if render_mode == 'isosurface_mode' else 'volume', iso_percentile))
                    fig_3d.update_layout(
                        scene=dict(
                            xaxis_title=t('x_axis'),
                            yaxis_title=t('y_axis'),
                            zaxis_title=t('z_axis'),
                            aspectmode='data',
                            camera=camera,
                        ),
                        height=600,
                        margin=dict(l=0, r=0, t=0, b=0),
                        uirevision='3d_view',
                    )
                    st.plotly_chart(fig_3d, use_container_width=True,
                                   config={'doubleClick': 'reset+autosize'})
                    col_3d_info.write(f"{t('displayed_points')} {values.size:,}")
                    col_3d_info.write(f"{t('downsample')} 1/{step}")
                else:
                    st.info(t('render_hint'))
        else:
            st.info(t('3d_single_layer'))

    # ── Tab 3: Waveform ───────────────────────────────────────────
    with tab3:
        st.markdown(t('waveform_title'))

        if st.session_state.data.n_slices > 1:
            col_w1, col_w2 = st.columns(2)
            with col_w1:
                sample_x = st.slider(t('x_position'), 0, st.session_state.data.width - 1,
                                     st.session_state.data.width // 2)
            with col_w2:
                sample_y = st.slider(t('y_position'), 0, st.session_state.data.height - 1,
                                     st.session_state.data.height // 2)

            amp_profile   = st.session_state.data.amplitude[:, sample_y, sample_x]
            phase_profile = st.session_state.data.phase[:, sample_y, sample_x]

            suffix = "）" if st.session_state.lang == 'CN' else ""
            fig_amp = go.Figure()
            fig_amp.add_trace(go.Scatter(
                x=z_indices, y=amp_profile,
                mode='lines+markers', name=t('channel_amp'),
                line=dict(color='#065A82', width=2)
            ))
            fig_amp.update_layout(
                title=f"{t('amp_profile_title')} ({sample_x}, {sample_y}){suffix}",
                xaxis_title=t('z_index_axis'),
                yaxis_title=t('amplitude_axis'),
                height=300
            )
            st.plotly_chart(fig_amp, use_container_width=True)

            fig_phase = go.Figure()
            fig_phase.add_trace(go.Scatter(
                x=z_indices, y=phase_profile,
                mode='lines+markers', name=t('channel_phase'),
                line=dict(color='#1C7293', width=2)
            ))
            fig_phase.update_layout(
                title=f"{t('phase_profile_title')} ({sample_x}, {sample_y}){suffix}",
                xaxis_title=t('z_index_axis'),
                yaxis_title=t('phase_rad'),
                height=300
            )
            st.plotly_chart(fig_phase, use_container_width=True)
        else:
            st.info(t('waveform_single_layer'))

    # ── Tab 4: Export ─────────────────────────────────────────────
    with tab4:
        st.markdown(t('export_title'))

        col_e1, col_e2 = st.columns(2)

        with col_e1:
            st.markdown(t('export_slice_title'))
            st.caption(t('export_slice_caption'))
            export_kind = st.radio(t('export_kind'), ['pixel_image', 'publication'], format_func=t)
            export_format = 'PNG' if export_kind == 'publication' else st.selectbox(t('format_label'), ["PNG", "JPEG", "TIFF"])

            if st.button(t('btn_export_image'), use_container_width=True):
                current_z  = z_indices[st.session_state.current_z_idx]
                ch_key     = st.session_state.channel.lower()
                norm, _, _ = get_normalized_slice(st.session_state.data, data_key, current_z, ch_key)

                brightness_offset = (st.session_state.brightness - 50) / 100
                contrast_scale    = st.session_state.contrast / 50
                adjusted = (norm - 0.5) * contrast_scale + 0.5 + brightness_offset
                adjusted = np.clip(adjusted, 0, 1)

                if export_kind == 'publication':
                    image_bytes = publication_png(adjusted, st.session_state.colormap, pixel_size,
                                                  color_label, t('x_pixels'), t('y_pixels'))
                else:
                    rgba = get_export_colormap(st.session_state.colormap)(adjusted)
                    rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
                    img = Image.fromarray(rgb)
                    buf = io.BytesIO()
                    save_options = {'dpi': (300, 300)} if export_format in ('PNG', 'TIFF') else {}
                    img.save(buf, format=export_format, **save_options)
                    image_bytes = buf.getvalue()

                st.download_button(
                    label=f"{t('btn_download')} {export_format}",
                    data=image_bytes,
                    file_name=f"lfi_{export_kind}_{ch_key}_z{current_z}_{st.session_state.colormap}.{export_format.lower()}",
                    mime=f"image/{export_format.lower()}",
                    use_container_width=True
                )

        with col_e2:
            st.markdown(t('export_csv_title'))
            st.caption(t('export_csv_caption'))

            if st.button(t('btn_export_csv'), use_container_width=True):
                current_z  = z_indices[st.session_state.current_z_idx]
                slice_data = st.session_state.data.get_slice(current_z)
                ch_key = st.session_state.channel.lower()
                csv_data = export_csv(slice_data[ch_key])

                st.download_button(
                    label=t('btn_download_csv'),
                    data=csv_data,
                    file_name=f"lfi_{ch_key}_z{current_z}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

# ── Footer ────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style='text-align: center; color: #888; padding: 1rem; font-size: 0.85rem;'>
    <p>LFI Imaging Viewer v1.3 &nbsp;|&nbsp; REIT4841 &nbsp;|&nbsp; EECS</p>
    <p>Zhenyu He &nbsp;·&nbsp; Zeyu Wang &nbsp;·&nbsp; Mingxuan Tang &nbsp;·&nbsp; Haobo Yu</p>
</div>
""", unsafe_allow_html=True)
