"""Adapt the public Part 2 pipeline without changing teammate code."""
from pathlib import Path
from ..data_types import LFIRawData


class Part2Preprocessor:
    """Run Part 2 using repository TOML; params are nested config overrides.

    The result remains a UI scan, with all Part 2 products/provenance retained.
    Radian labels require both upstream conversion and Part 2 radian units.
    """
    def preprocess(self, data: LFIRawData, params: dict) -> LFIRawData:
        from part2_preprocess.pipeline import preprocess_part1
        config_path = Path(__file__).resolve().parents[3] / 'part2_preprocess/config/default.toml'
        if not config_path.is_file():
            raise FileNotFoundError(f'Part 2 configuration not found: {config_path}')
        output = preprocess_part1(data, config_path=str(config_path),
                                  config_overrides=params or None)
        products = output.metadata.get('spatial_products', {})
        product = 'amplitude_corrected' if 'amplitude_corrected' in products else 'output.amplitude'
        amplitude = products['amplitude_corrected'] if product == 'amplitude_corrected' else output.amplitude
        normalization = output.metadata.get('config_resolved', {}).get('normalisation', {}).get('method', 'none')
        metadata = dict(data.metadata)
        metadata['part2_preprocessing'] = output.metadata
        metadata['ui_preprocessing'] = {'amplitude_product': product,
                                        'normalisation': normalization,
                                        'amplitude_units': 'dimensionless' if normalization == 'zscore' else 'input units'}
        metadata['phase_converted'] = (output.metadata.get('phase_units') == 'radians'
                                       and data.metadata.get('phase_converted') is True)
        return LFIRawData(amplitude=amplitude, phase=output.phase,
                          z_indices=list(data.z_indices), target_type=data.target_type,
                          file_path=data.file_path, metadata=metadata)
