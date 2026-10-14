"""Phase labels reflect confirmed conversion metadata, never observed ranges."""


def phase_label_key(metadata):
    return 'phase_rad' if metadata.get('phase_converted') is True else 'phase_raw'


def phase_encoding_status(metadata):
    if 'phase_encoding_used' not in metadata or 'phase_converted' not in metadata:
        return 'unknown'
    return f"{metadata.get('phase_encoding_used') or 'unknown'}; phase_converted={metadata['phase_converted']}"
