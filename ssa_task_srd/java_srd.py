"""Batch I/O to a pinned, unmodified author SRD JAR; never falls back to Python."""
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import time
import numpy as np

MAGIC=0x53524431
AUTHOR_COMMIT='84486b6c15e02f17ac32c80b827afb17e09802b9'


def sha256(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class JavaSRD:
    def __init__(self,root=None):
        self.root=Path(root) if root else Path(__file__).resolve().parents[1]
        config=self.root/'data/java/runtime.json'
        if not config.is_file():
            raise RuntimeError('Java SRD is not configured. Run scripts/setup_java.py --fetch --out runs/java_setup_NEW')
        self.runtime=json.loads(config.read_text(encoding='utf-8'))
        r=self.runtime
        if r['author_commit']!=AUTHOR_COMMIT or not r['author_tracked_files_unchanged']:
            raise RuntimeError('Author source pin/cleanliness verification failed')
        if not Path(r['java_executable']).is_file():raise RuntimeError('Configured Java executable is absent')
        for file,expected in [(r['author_jar'],r['author_jar_sha256']),
            (self.root/'java/SRDBridge.java',r['bridge_source_sha256']),
            (Path(r['bridge_classes'])/'SRDBridge.class',r['bridge_class_sha256'])]:
            if not Path(file).is_file() or sha256(file)!=expected:
                raise RuntimeError(f'Java artifact changed or absent; rebuild before use: {file}')

    def command(self,*args):
        r=self.runtime
        return [r['java_executable'],'-cp',r['bridge_classes']+os.pathsep+r['author_jar'],'SRDBridge',*[str(a) for a in args]]

    def provenance(self):
        keys=['author_url','author_commit','author_jar_sha256','bridge_source_sha256','bridge_class_sha256',
            'java_version','maven_version','author_tracked_files_unchanged']
        result={key:self.runtime[key] for key in keys}
        certificate=self.root/'data/java/bridge_verification.json'
        result['adapter_verified_against_author_smoke']=False
        if certificate.is_file():
            check=json.loads(certificate.read_text(encoding='utf-8'))
            result['adapter_verified_against_author_smoke']=(
                all(check.get(k)==self.runtime[k] for k in ['author_commit','author_jar_sha256','bridge_source_sha256','bridge_class_sha256'])
                and check.get('max_abs_difference')==0.0)
        return result

    def repair_batch(self,x,period,k=6.0,max_iter=5,artifact_dir=None):
        array=np.asarray(x,dtype=np.float64)
        if array.ndim!=2 or not len(array) or not np.isfinite(array).all():raise ValueError('Expected a nonempty finite window matrix')
        if not isinstance(period,int) or period<2 or period%2 or array.shape[1]<2*period:
            raise ValueError('An even period and >=2 cycles are required')
        if not np.isfinite(k) or k<=0 or not isinstance(max_iter,int) or max_iter<1:raise ValueError('Invalid SRD parameters')
        temp_root=self.root/'data/java/tmp';temp_root.mkdir(parents=True,exist_ok=True)
        if artifact_dir is not None:
            directory=Path(artifact_dir);directory.mkdir(parents=True,exist_ok=False)
            context=nullcontext(str(directory))
        else:context=tempfile.TemporaryDirectory(prefix='srd_',dir=temp_root)
        start=time.perf_counter()
        with context as folder:
            folder=Path(folder);source=folder/'input.bin';target=folder/'output.bin'
            source.write_bytes(struct.pack('>iii',MAGIC,*array.shape)+array.astype('>f8').tobytes())
            result=subprocess.run(self.command('repair',source,target,period,k,max_iter),cwd=self.root,
                capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=180)
            if result.returncode:raise RuntimeError('Author SRD process failed: '+result.stderr[-3000:])
            payload=target.read_bytes()
            if len(payload)!=12+array.size*8 or struct.unpack('>iii',payload[:12])!=(MAGIC,*array.shape):
                raise RuntimeError('Invalid Java SRD result header or length')
            repaired=np.frombuffer(payload[12:],dtype='>f8').astype(np.float64).reshape(array.shape)
            if not np.isfinite(repaired).all():raise RuntimeError('Nonfinite Java SRD output')
            audit={'windows':len(array),'length':array.shape[1],'period':period,'k':k,'iterations':max_iter,
                'input_sha256':sha256(source),'output_sha256':sha256(target),
                'process_seconds':time.perf_counter()-start,'author_srd_called_per_window':True,
                'window_concatenation':False,'additional_scaling_in_bridge':False}
        return repaired,audit
