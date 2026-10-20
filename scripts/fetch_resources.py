"""Download public inputs without executing third-party source code."""
from pathlib import Path
import argparse
import hashlib
import io
import json
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
RESOURCES = {
    'tetouan.zip': 'https://archive.ics.uci.edu/static/public/849/power+consumption+of+tetouan+city.zip',
    'SRD.java': 'https://raw.githubusercontent.com/czjthss/SeasonalClean/main/code/src/main/java/Algorithm/SRD.java',
    'Decomposition.java': 'https://raw.githubusercontent.com/czjthss/SeasonalClean/main/code/src/main/java/Algorithm/util/Decomposition.java',
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--only', choices=['data', 'source', 'all'], default='all')
    args = parser.parse_args()
    out = ROOT / 'data' / 'downloads'
    out.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, url in RESOURCES.items():
        if args.only == 'data' and name.endswith('.java'):
            continue
        if args.only == 'source' and not name.endswith('.java'):
            continue
        target = out / name
        if not target.exists():
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=30) as response:
                content = response.read()
            target.write_bytes(content)
        content = target.read_bytes()
        manifest[name] = {'url': url, 'sha256': hashlib.sha256(content).hexdigest(), 'bytes': len(content)}
        print(f'{name}: {len(content)} bytes', flush=True)
        if name.endswith('.zip'):
            with zipfile.ZipFile(io.BytesIO(content)) as archive:
                members = [m for m in archive.namelist() if m.lower().endswith('.csv')]
                if len(members) != 1:
                    raise ValueError(f'Expected one CSV, found {members}')
                # Write one fixed filename; never extract paths from the archive.
                destination = ROOT / 'data' / 'tetouan.csv'
                destination.write_bytes(archive.read(members[0]))
    path = out / 'manifest.json'
    prior = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    prior.update(manifest)
    path.write_text(json.dumps(prior, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
