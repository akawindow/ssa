"""Configure existing Java/Maven per project and build the pinned, unmodified SRD artifact."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[1]
AUTHOR_COMMIT='84486b6c15e02f17ac32c80b827afb17e09802b9'
AUTHOR_URL='https://github.com/czjthss/SeasonalClean.git'


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def capture(command,cwd,log=None):
    result=subprocess.run([str(x) for x in command],cwd=cwd,stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
    if log:Path(log).write_text(result.stdout,encoding='utf-8')
    if result.returncode:raise RuntimeError(f'Exit {result.returncode}; see {log or result.stdout[-3000:]}')
    return result.stdout.strip()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--java-home');parser.add_argument('--maven-home')
    parser.add_argument('--out',required=True)
    parser.add_argument('--fetch',action='store_true',help='Fetch pinned source if not already present')
    args=parser.parse_args()
    out=ROOT/args.out;out.mkdir(parents=True,exist_ok=False)
    candidates=[args.java_home,os.environ.get('JAVA_HOME')]
    on_path=shutil.which('javac')
    if on_path:candidates.append(str(Path(on_path).resolve().parents[1]))
    candidates+=['D:/program/IntelliJ IDEA 2025.2.5/jbr','D:/program/PyCharm 2025.2.3/jbr']
    java_home=next((Path(x) for x in candidates if x and (Path(x)/'bin/javac.exe').is_file()),None)
    if java_home is None:raise RuntimeError('No JDK found; supply --java-home pointing to JDK >=17')
    java=java_home/'bin/java.exe';javac=java_home/'bin/javac.exe'
    candidates=[args.maven_home,os.environ.get('MAVEN_HOME'),'D:/program/IntelliJ IDEA 2025.2.5/plugins/maven/lib/maven3']
    maven_home=next((Path(x) for x in candidates if x and (Path(x)/'bin/m2.conf').is_file()),None)
    if maven_home is None:raise RuntimeError('No Maven found; supply --maven-home')
    boot=next((maven_home/'boot').glob('plexus-classworlds-*.jar'))
    java_version=capture([java,'-version'],ROOT,out/'java_version.log')
    javac_version=capture([javac,'-version'],ROOT,out/'javac_version.log')
    if int(re.search(r'javac (\d+)',javac_version).group(1))<17:raise RuntimeError('JDK >=17 required')
    repo=ROOT/'data/author/SeasonalClean'
    git=shutil.which('git') or 'D:/program/Git/cmd/git.exe'
    if not repo.exists():
        if not args.fetch:raise RuntimeError('Author checkout absent; rerun with --fetch')
        repo.parent.mkdir(parents=True,exist_ok=True)
        capture([git,'clone','--depth','1','--filter=blob:none','--sparse',AUTHOR_URL,repo],ROOT,out/'clone.log')
    if capture([git,'rev-parse','HEAD'],repo)!=AUTHOR_COMMIT:
        if not args.fetch:raise RuntimeError('Author commit differs from pin; inspect before fetching')
        capture([git,'fetch','--depth','1','origin',AUTHOR_COMMIT],repo,out/'fetch.log')
        capture([git,'checkout','--detach',AUTHOR_COMMIT],repo,out/'checkout.log')
    if args.fetch:
        capture([git,'sparse-checkout','set','code','data/real_clean','data/real_dirty'],repo,out/'sparse_checkout.log')
    if capture([git,'status','--porcelain','--untracked-files=no'],repo):
        raise RuntimeError('Author tracked files have local changes; refusing to build as unmodified')
    local=ROOT/'data/java';local.mkdir(parents=True,exist_ok=True)
    launcher=[java,f'-Dmaven.multiModuleProjectDirectory={repo/"code"}',f'-Dmaven.home={maven_home}',
        f'-Dclassworlds.conf={maven_home/"bin/m2.conf"}','-classpath',boot,
        'org.codehaus.plexus.classworlds.launcher.Launcher']
    maven_version=capture(launcher+['-version'],repo/'code',out/'maven_version.log')
    print('Building pinned author artifact; full output is recorded in maven_build.log',flush=True)
    capture(launcher+['-B',f'-Dmaven.repo.local={local/"m2"}','package'],repo/'code',out/'maven_build.log')
    jar=repo/'code/target/seasonal-1.0-SNAPSHOT.jar'
    bridge=ROOT/'java/SRDBridge.java';classes=local/'bridge_classes';classes.mkdir(exist_ok=True)
    capture([javac,'--release','17','-encoding','UTF-8','-cp',jar,'-d',classes,bridge],ROOT,out/'bridge_compile.log')
    source_hashes={str(p.relative_to(repo)).replace('\\','/'):digest(p) for p in sorted((repo/'code').rglob('*.java'))}
    source_hashes['code/pom.xml']=digest(repo/'code/pom.xml')
    runtime={'created_utc':datetime.now(timezone.utc).isoformat(),'java_executable':str(java.resolve()),
        'javac_executable':str(javac.resolve()),'java_home':str(java_home.resolve()),'maven_home':str(maven_home.resolve()),
        'java_version':java_version,'maven_version':maven_version,'author_url':AUTHOR_URL,'author_commit':AUTHOR_COMMIT,
        'author_repository':str(repo),'author_jar':str(jar),'author_jar_sha256':digest(jar),
        'bridge_classes':str(classes),'bridge_source_sha256':digest(bridge),
        'bridge_class_sha256':digest(classes/'SRDBridge.class'),'author_source_hashes':source_hashes,
        'author_tracked_files_unchanged':not bool(capture([git,'status','--porcelain','--untracked-files=no'],repo)),
        'global_environment_modified':False,'maven_cache':str(local/'m2')}
    (local/'runtime.json').write_text(json.dumps(runtime,indent=2),encoding='utf-8')
    (out/'runtime.json').write_text(json.dumps(runtime,indent=2),encoding='utf-8')
    print(f'Configured JDK and Maven; runtime: {local/"runtime.json"}',flush=True)


if __name__=='__main__':main()
