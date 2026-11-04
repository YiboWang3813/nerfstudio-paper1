"""Locally rebuild frozen pre-refactor operators for isolated parity probes.

Nothing is installed into the shared runtime or project site-packages. Original
Python packages are visible only inside the probe process. Old binaries and
environments are never copied. The reference uses the same target runtime and
toolchain so comparisons isolate implementation changes.
"""
from __future__ import annotations

import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path('/home/yibo/shared/cache/paper1-parity/reference-source')
PACKAGES = {
    'ours':'ours_renderer',
    'ultragauss':'ultragauss_renderer',
    'ultragray':'ultragray_renderer',
    'gaussianpile':'gaussianplie_renderer',
    'slice_adapted_3dgs':'slice_adapted_3dgs_renderer',
    'ultrags_baseline':'ultrags_baseline_renderer',
}


def build(method):
    import torch
    from torch.utils.cpp_extension import load
    package=PACKAGES[method]
    root=ROOT/'gsplat/ultrasound_ops'/package
    common=root.parent/'common'
    sources=sorted((root/'src').rglob('*.cu'))+sorted((root/'src').glob('*.cpp'))
    if method=='ultrags_baseline':
        sources=[root/path for path in ('cuda_rasterizer/rasterizer_impl.cu','cuda_rasterizer/forward.cu','cuda_rasterizer/backward.cu','rasterize_points.cu','ext.cpp')]
    hasher=hashlib.sha256()
    for base in (root,common):
        for path in sorted(base.rglob('*')):
            if path.suffix in ('.cu','.cpp','.cuh','.h','.hpp','.inl'):
                hasher.update(str(path.relative_to(base)).encode()+b'\0'+path.read_bytes()+b'\0')
    identity={
        'method':method,'role':'independently rebuilt frozen legacy source',
        'source_sha256':hasher.hexdigest(),'torch':torch.__version__,
        'torch_cuda':torch.version.cuda,'cxx11_abi':torch._C._GLIBCXX_USE_CXX11_ABI,
        'cuda_home':str(Path(os.environ['CUDA_HOME']).resolve()),
        'nvcc':subprocess.check_output([os.environ['CUDA_HOME']+'/bin/nvcc','--version'],text=True),
        'cc':subprocess.check_output([os.environ['CC'],'--version'],text=True),
        'cxx':subprocess.check_output([os.environ['CXX'],'--version'],text=True),
        'arch':os.environ['TORCH_CUDA_ARCH_LIST'],
        'cflags':['-O3','-std=c++17'],
        'cuda_flags':['-O3','--use_fast_math','-lineinfo','-std=c++17','-Xcompiler','-fPIC','-DTHRUST_IGNORE_CUB_VERSION_CHECK'],
    }
    key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:24]
    directory=Path('/home/yibo/shared/cache/torch_extensions/paper1-reference')/key
    directory.mkdir(parents=True,exist_ok=True)
    with (directory/'.build.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        module=load(name='paper1_reference_'+key,sources=[str(path) for path in sources],
                    extra_cflags=identity['cflags'],extra_cuda_cflags=identity['cuda_flags'],
                    extra_include_paths=[str(root/'include'),str(common),str(root),str(root/'third_party/glm')],
                    build_directory=str(directory),verbose=True)
        (directory/'identity.json').write_text(json.dumps(identity,indent=2)+'\n')
    return module,directory


def load_reference(method):
    """Only call from the isolated probe entry point, never a production model."""
    for name in PACKAGES.values():
        path=str(ROOT/'gsplat/ultrasound_ops'/name)
        if path not in sys.path:sys.path.insert(0,path)
    legacy=str(ROOT/'nerfstudio/paper1_legacy')
    if legacy not in sys.path:sys.path.insert(0,legacy)
    if method in PACKAGES:
        module,directory=build(method)
        package=PACKAGES[method]
        sys.modules[package+'._C']=module
        importlib.import_module(package)
        return directory
    if method not in ('nerfus','ultranerf'):raise ValueError(method)
    return None


def main():
    from nerfstudio.scripts.ultrasound_resources import resources
    with resources(gpu=True):
        for method in PACKAGES:
            directory=load_reference(method)
            print('REFERENCE_KERNEL_READY',method,str(directory),flush=True)


if __name__=='__main__':main()
