"""Fetch only official TRAIN, using the mirror linked by the official Keras example."""
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen


def main():
    root = Path(__file__).resolve().parents[1]
    cfg = json.loads((root / 'configs/classification_protocol.json').read_text())
    folder = root / 'data/forda'
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / 'FordA_TRAIN.tsv'
    if not path.exists():
        with urlopen(cfg['train_url'], timeout=45) as response:
            blob = response.read()
        if len(blob) < 1000000:
            raise ValueError('Unexpectedly short data response')
        # Do not leave a partial TSV if downloading fails.
        path.write_bytes(blob)
    manifest = {'source': cfg['train_url'], 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'bytes': path.stat().st_size, 'official_test_downloaded': False}
    (folder / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
