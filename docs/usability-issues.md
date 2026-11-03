# Usability issues / 可用性问题

## UI-001 · Phase crosshair has insufficient contrast / Phase 十字线对比度不足

- Recorded / 记录日期：2026-10-18 (Australia/Brisbane)
- Reported against / 反馈版本：v1.3-sus
- Status / 状态：Resolved / 已修复
- Area / 范围：T3 pixel selection, 2D Phase view / T3 像素选择、二维相位视图
- Reproduction / 复现：Select a pixel, then switch the channel to Phase. / 选择像素后，将通道切换为 Phase。
- Observed / 实际表现：The crosshair blends into the phase image's dense, similarly coloured texture and is difficult to locate. This is a visibility issue; the screenshot alone does not establish a lost selection. / 十字线与相位图密集且颜色相近的纹理混在一起，难以找到。当前记录为可见性问题，不能仅凭截图判定像素选择已丢失。
- Expected / 期望：The selected-pixel marker remains clearly visible in both channels and across colormaps. / 所选像素的标记在两个通道及不同色图下均清晰可辨。
- Candidate fix / 修复方向：Evaluate a contrasting outline or two-tone crosshair and a distinct centre marker. / 评估带对比描边或双色十字线，以及明显的中心标记。
- Acceptance / 验收：After switching Amplitude ↔ Phase, the marker remains easy to locate and matches Pixel Info X/Y, including over bright, dark and densely textured regions. / 切换 Amplitude ↔ Phase 后，标记在亮区、暗区和密集纹理区域均易于定位，且与 Pixel Info 的 X/Y 一致。
- Evidence / 证据：[User screenshot / 用户截图](screenshots/issues/phase-crosshair-low-contrast.png)

Resolution: white selection lines now have wider dark outlines, with both layers kept at the same selected-pixel coordinates above the image. Regression coverage checks both channels and every colormap. No release tag changed.

## UI-002 · Preprocessing status layout is crowded / 预处理状态排版拥挤

- Recorded / 记录日期：2026-10-18 (Australia/Brisbane)
- Reported against / 反馈版本：v1.3-sus
- Status / 状态：Resolved / 已修复
- Area / 范围：T4 Pre-processing group / T4 预处理分组
- Observed / 实际表现：Amplitude and Phase status lines are densely packed with little visual hierarchy. The Phase description wraps awkwardly between “not” and “unwrapped”. / Amplitude 与 Phase 状态文字紧密堆叠，缺少视觉层级；Phase 说明在 “not” 与 “unwrapped” 之间断行，阅读不顺。
- Expected / 期望：Separate channel names, processing state and metadata details clearly; use comfortable spacing and natural wrapping at normal sidebar widths. / 通道名称、处理状态及元数据详情应层次清晰，间距舒适，在正常侧栏宽度下自然换行。
- Evidence / 证据：[User screenshot / 用户截图](screenshots/issues/preprocessing-status-layout.png)

Resolution: each channel now has a bold heading, separate state/detail lines and spacing between channel sections. The “not unwrapped” phrase uses a nonbreaking space. English/Chinese status regression coverage was updated. No release tag changed.


## UI-003 · macOS trackpad cannot zoom the 3D view

- Found: during a live demo (reported by the user; demo date not supplied).
- Status: Resolved.
- Observed: native trackpad pinch was ignored; two-finger scrolling barely changed camera distance.
- Cause: pyqtgraph 0.14 wheel handling preferred the horizontal angle delta and did not handle native zoom gestures.
- Resolution: a GLViewWidget subclass handles native pinch, prefers nonzero pixel deltas for trackpads and uses the dominant scroll axis. Camera distance is clamped to 0.05–50 times the largest volume dimension. Ctrl+wheel still changes FOV. Bilingual Zoom in/out controls support Cmd/Ctrl+= and Cmd/Ctrl+-.
- Validation: pytest-qt sends synthetic native pinch and mouse/trackpad wheel events and checks zoom direction, dominant-axis selection, distance limits and Ctrl+wheel behavior; panel controls and shortcuts have regression coverage.
- Scope: source fix only; no app rebuild, tag or merge. Physical trackpad behavior still needs confirmation in the live macOS app.


## UI-004 · Garbled Windows spin-box digits

- Status: fixed, pending verification on Windows.
- Reported environment: Windows 11, PySide6 6.12, default `windows11` style.
- Observed: QSpinBox/QDoubleSpinBox digits are painted as garbled glyphs while labels and combo boxes remain correct. Not reproduced on macOS; the style-painting cause is suspected, not established here.
- Fix: select Fusion on win32 before creating MainWindow; macOS keeps its native default. LFI_QT_STYLE can override the selection for diagnosis or force Fusion on macOS.
- Validation: pure platform/override cases, Fusion spin-box values, dark plot colors and palette-aware histogram checks. No additional UI restyling; Windows-native verification is still required.

## UI-005 · CSV exports lack identifying metadata

- Status: resolved.
- Fix: both desktop CSV entry points prepend comma-free `# ` metadata comments covering identity/version, timestamp, source, target, channel, processing state, slice/Z, shape, normalisation/units and phase provenance. Header line breaks are flattened.
- Compatibility: numeric matrix bytes and `%.6g` format are unchanged; pure export_csv callers can omit metadata for the original output. NumPy reads directly; pandas uses comment="#" and header=None; Excel shows comments as initial rows.
- Validation: numeric byte comparison, NumPy roundtrip, optional pandas reader, channel-specific provenance, header fields and both export entry points. Test files use tmp_path.

## Feature · Export the current 3D viewport

- Added bilingual Export 3D view… control next to the 3D camera controls.
- Saves grabFramebuffer() as PNG at the current camera angle, plus a same-stem .txt provenance sidecar including mode, percentile thresholds/values and downsampling step. Reuses export directory fallback and file reveal.
- The button requires a rendered result and a viewport; it is disabled again on clear and always disabled offscreen.
- Validation: button lifecycle, naming, sidecar fields and mocked framebuffer PNG save/cancel/failure paths. This automated offscreen coverage does not claim native OpenGL capture or Windows verification.
- No commit, tag, merge or application rebuild performed.
