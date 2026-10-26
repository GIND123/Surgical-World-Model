"""Publish the SurgVU 2026 Cat-1 model + inference code to the Hugging Face Hub."""
import os, argparse, shutil, tempfile
from huggingface_hub import HfApi

ROOT = '/mnt/5385e3d3-fb4b-4846-8f72-55998fc781fb/surgical_Image/work'

SUBMISSION_FILES = ('infer_core.py', 'track_core.py', 'inference.py', 'Dockerfile',
                    'requirements.txt', 'do_build.sh', 'do_save.sh')
TRAIN_FILES = ('make_pseudo.py', 'pseudo.py', 'build_balanced.py', 'select_frames.py',
               'final_train.py', 'build_timeline.py', 'frame_index.py', 'extract_train.py',
               'common.py', 'evalcoco.py', 'eval_pipeline.py', 'official_check.py',
               'train_cv.py', 'selftrain.py', 'ds.py', 'tune_post.py', 'calib.py',
               # round 3 / final build
               'train7.py', 'presence_eval.py', 'eval_ckpts.py', 'oracle.py',
               'package_final.sh', 'watch_ckpts.sh', 'chain_r3.sh')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', required=True)
    ap.add_argument('--weights', nargs='+', required=True)
    ap.add_argument('--names', nargs='*', default=None,
                    help='file names for the weights inside model/ (default det0.pt, det1.pt, ...)')
    ap.add_argument('--private', type=int, default=0)
    ap.add_argument('--labels_tar', default='')
    ap.add_argument('--card', default=f'{ROOT}/MODEL_CARD.md')
    ap.add_argument('--subdir', default='submission5',
                    help='which container dir to publish (the FINAL one, not submission/)')
    a = ap.parse_args()

    api = HfApi(token=os.environ['HF_TOKEN'])
    api.create_repo(a.repo, repo_type='model', private=bool(a.private), exist_ok=True)

    stage = tempfile.mkdtemp()
    os.makedirs(f'{stage}/model', exist_ok=True)
    names = a.names or [f'det{i}.pt' for i in range(len(a.weights))]
    for w, nm in zip(a.weights, names):
        shutil.copyfile(w, f'{stage}/model/{nm}')
    shutil.copyfile(f'{ROOT}/{a.subdir}/model/config.json', f'{stage}/model/config.json')

    for f in SUBMISSION_FILES:
        src = f'{ROOT}/{a.subdir}/{f}'
        if os.path.exists(src):
            shutil.copyfile(src, f'{stage}/{f}')

    os.makedirs(f'{stage}/train_code', exist_ok=True)
    for f in TRAIN_FILES:
        src = f'{ROOT}/scripts/{f}'
        if os.path.exists(src):
            shutil.copyfile(src, f'{stage}/train_code/{f}')

    if a.labels_tar and os.path.exists(a.labels_tar):
        shutil.copyfile(a.labels_tar, f'{stage}/pseudo_labels.tar.gz')

    shutil.copyfile(a.card, f'{stage}/README.md')
    print('staged:', sorted(os.listdir(stage)))
    api.upload_folder(folder_path=stage, repo_id=a.repo, repo_type='model',
                      commit_message='SurgVU 2026 Cat-1: weights, inference container, training code')
    print('DONE -> https://huggingface.co/' + a.repo)


if __name__ == '__main__':
    main()
