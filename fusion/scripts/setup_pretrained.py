#!/usr/bin/env python3
"""Download official pretrained assets and pinned upstream checkouts (no model execution)."""
import argparse
import json
from pathlib import Path
import subprocess

DIT_COMMIT = 'ed81ce2229091fd4ecc9a223645f95cf379d582b'
ADM_COMMIT = '22e0df8183507e13a7813f8d38d51b072ca1e67c'
VAE_COMMIT = 'f04b2c4b98319346dad8c65879f680b1997b204a'


def checkout(url, path, revision):
    if not path.exists():
        subprocess.run(['git','clone',url,str(path)],check=True)
    origin = subprocess.check_output(['git','-C',str(path),'remote','get-url','origin'],text=True).strip()
    if origin.removesuffix('.git') != url.removesuffix('.git'):
        raise ValueError(f'Unexpected origin for {path}: {origin}')
    # Reset any dirty state left by a previous interrupted run before checking.
    subprocess.run(['git','-C',str(path),'checkout','--force','--detach',revision],check=True)
    subprocess.run(['git','-C',str(path),'clean','-fd'],check=True)


def download(url, path):
    if path.exists() and path.stat().st_size > 0:
        print(f'Using existing {path}',flush=True)
        return
    path.parent.mkdir(parents=True,exist_ok=True)
    partial = path.with_suffix(path.suffix+'.part')
    subprocess.run(['curl','-fL','--retry','5','--retry-delay','3','-C','-',url,'-o',str(partial)],check=True)
    partial.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets',required=True)
    p.add_argument('--image-size',type=int,choices=[256,512],default=256)
    p.add_argument('--skip-reference',action='store_true')
    args=p.parse_args()
    root=Path(args.assets).resolve();root.mkdir(parents=True,exist_ok=True)
    checkout('https://github.com/facebookresearch/DiT.git',root/'DiT',DIT_COMMIT)
    checkout('https://github.com/openai/guided-diffusion.git',root/'guided-diffusion',ADM_COMMIT)
    filename=f'DiT-XL-2-{args.image_size}x{args.image_size}.pt'
    checkpoint=root/'pretrained_models'/filename
    download('https://dl.fbaipublicfiles.com/DiT/models/'+filename,checkpoint)
    ref_name='VIRTUAL_imagenet256_labeled.npz' if args.image_size==256 else 'VIRTUAL_imagenet512.npz'
    reference=root/'references'/ref_name
    if not args.skip_reference:
        download(f'https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/{args.image_size}/{ref_name}',reference)
    record=dict(dit_repo=str(root/'DiT'),adm_repo=str(root/'guided-diffusion'),checkpoint=str(checkpoint),
                reference=str(reference),dit_commit=DIT_COMMIT,adm_commit=ADM_COMMIT,
                vae='stabilityai/sd-vae-ft-ema',vae_revision=VAE_COMMIT)
    (root/f'assets_{args.image_size}.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))


if __name__=='__main__': main()
