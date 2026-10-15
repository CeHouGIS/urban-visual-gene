"""Move version-specific files without duplicating data or invalidating old paths.

Run from the repository root with --apply. Existing report/config/output paths
become relative symlinks; old Python module names are provided by package paths.
Only this repository's active dimension-VLM processes are briefly suspended
during directory renames, then resumed even if a move fails.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal

ROOT = Path(__file__).resolve().parents[1]
MAIN = "vitl30_panorama_w4096_k32"
BASE = "global307_baseline_w1024_k8"
SCALE = "global307_dictionary_scaling_k8"
SWEEP = "global307_w4096_k_sweep"
W2048 = "global307_w2048_k64"
W4096 = "global307_w4096_k64"
VERSIONS = [MAIN, BASE, SCALE, SWEEP, W2048, W4096,
            "legacy30_paper_w1024_k8", "legacy12city", "city37_plan",
            "semantic_pipeline_smoke"]


def plan():
    moves = []

    def add(old, new, alias=False):
        moves.append({"old": old, "new": new, "compatibility_symlink": alias})

    def code(name, version):
        add(f"formal/analysis/{name}", f"experiments/{version}/code/{name}", name.endswith('.sh'))

    for p in sorted((ROOT / 'sae_experiments').glob('panorama_vitl30*')):
        if p.is_file():
            add(str(p.relative_to(ROOT)), f'experiments/{MAIN}/code/{p.name}')
    add('formal/gpu_run.py', f'experiments/{BASE}/code/gpu_run.py')
    add('formal/select_city_density_candidates.py', f'experiments/{BASE}/code/select_city_density_candidates.py')
    add('sae_experiments/evaluation/encode_batchtopk_w1024_k8.py',
        f'experiments/{BASE}/code/encode_batchtopk_w1024_k8.py')
    for name in ['analyze_dictionary_scaling.py', 'analyze_heldout_scaling.py',
                 'build_scaling_atlas.py', 'discover_scene_templates.py',
                 'plot_heldout_scaling.py', 'plot_scaling_examples.py', 'plot_scaling_sankey.py']:
        code(name, SCALE)
    for name in ['train_expanded_sae.py', 'build_expanded_sae_gallery.py', 'run_expanded_sae_ksweep.sh']:
        code(name, SWEEP)
    for name in ['test_feature_combinations.py', 'test_feature_pair_spatial.py',
                 'plot_feature_combination_examples.py', 'run_semantic_w2048_k64.sh',
                 'publish_semantic_w2048_k64.sh']:
        code(name, W2048)
    code('run_latest_semantic_umap.sh', W4096)
    for name in ['semantic_feature_umap.py', 'semantic_feature_umap.html',
                 'plot_feature_activation_maps.py', 'plot_semantic_umap_examples.py',
                 'prepare_semantic_umap_manifest.py', 'build_semantic_site.py', 'semantic_site']:
        add(f'formal/analysis/{name}', f'sae_experiments/semantic_tools/{name}', name == 'semantic_site')
    for p in sorted((ROOT / 'configs').iterdir()):
        if p.suffix not in {'.json', '.csv', '.yaml'}:
            continue
        version = 'city37_plan' if p.name == 'formal_city_manifest.json' else BASE
        add(str(p.relative_to(ROOT)), f'experiments/{version}/configs/{p.name}', True)
    for p in sorted((ROOT / 'formal').glob('watchdog*.sh')):
        version = 'city37_plan' if p.name == 'watchdog_batchtopk.sh' else BASE
        add(str(p.relative_to(ROOT)), f'experiments/{version}/code/{p.name}', True)
    for p in sorted((ROOT / 'formal/slurm').glob('*.sbatch')):
        if 'qc_' not in p.name:
            add(str(p.relative_to(ROOT)), f'experiments/city37_plan/code/slurm/{p.name}', True)
    add('formal/logs', f'experiments/{BASE}/runtime/logs', True)
    add('outputs/experiments/nas_30city_vitl_panorama_w4096_k32', f'experiments/{MAIN}/outputs', True)
    main_reports = {'semantic_umap': 'semantic_umap', 'instance_umap': 'instance_umap',
                    'vlm_interpretation': 'vlm_groups', 'dimension_vlm': 'vlm_dimensions',
                    'landscape_prototypes': 'landscape_prototypes', 'unit_motifs': 'unit_motifs'}
    for old, new in main_reports.items():
        add(f'reports/vitl30_{old}', f'experiments/{MAIN}/reports/{new}', True)
    for old, new in [
        ('reports/scaling', f'{SCALE}/reports/website'),
        ('reports/data/scaling_heldout', f'{SCALE}/outputs/heldout'),
        ('reports/figures/scaling_heldout', f'{SCALE}/reports/figures'),
        ('reports/sae_4096', f'{SWEEP}/reports/gallery'),
        ('reports/data/semantic_umap_w2048_k64_top100', f'{W2048}/outputs/semantic_top100'),
        ('reports/data/feature_combinations_w2048_k64', f'{W2048}/outputs/feature_combinations'),
        ('reports/data/semantic_umap_w4096_k64_top100', f'{W4096}/outputs/semantic_top100'),
        ('reports/data/semantic_umap_pipeline_smoke', 'semantic_pipeline_smoke/outputs'),
        ('reports/figures/legacy_12city', 'legacy12city/reports/figures'),
        ('reports/figures/chapter5', 'legacy30_paper_w1024_k8/reports/figures'),
        ('reports/chapter5_results_brief.md', 'legacy30_paper_w1024_k8/reports/chapter5_results_brief.md'),
        ('reports/urban_scene_primitives_30city_writing_plan.md', 'legacy30_paper_w1024_k8/reports/writing_plan.md'),
        ('reports/urban_scene_primitives_dictionary_scaling.md', f'{SCALE}/reports/REPORT.md'),
        ('reports/semantic_umap_latest_results.md', f'{W4096}/reports/REPORT.md'),
        ('reports/feature_combination_significance_w2048_k64.md', f'{W2048}/reports/feature_combinations.md'),
    ]:
        add(old, f'experiments/{new}', True)
    add('reports/semantic_feature_umap_plan.md', 'docs/methods/semantic_feature_umap_history.md', True)
    add('reports/project_inventory', 'docs/project_inventory', True)
    for name in ['pipeline.jpg', 'panorama_aware.jpg', 'urban_visual_genome_comic.png', 'urban_visual_genome_comic.svg']:
        add(f'experiments/legacy30_paper_w1024_k8/reports/figures/{name}', f'docs/figures/{name}', True)
    for p in sorted((ROOT / 'tests').glob('test_panorama_vitl30*.py')):
        add(str(p.relative_to(ROOT)), f'experiments/{MAIN}/tests/{p.name}')
    for name, version in [('test_formal_feature_cache.py', BASE), ('test_heldout_scaling.py', SCALE),
                          ('test_scaling_atlas.py', SCALE), ('test_scene_templates.py', SCALE),
                          ('test_expanded_sae_gallery.py', SWEEP)]:
        add(f'tests/{name}', f'experiments/{version}/tests/{name}')
    return moves


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    audit = ROOT / 'docs/project_inventory/layout_migration.json'
    if audit.exists():
        if args.apply:
            print('Layout already migrated; see docs/project_inventory/layout_migration.json.')
        else:
            print(audit.read_text())
        return
    moves = plan()
    if not args.apply:
        print(json.dumps(moves, indent=2)); return
    suspended = []
    completed = []
    try:
        for proc in Path('/proc').iterdir():
            if not proc.name.isdigit():
                continue
            try:
                cmd = (proc / 'cmdline').read_bytes().split(b'\0')
                if (b'sae_experiments.panorama_vitl30_dimension_vlm' in cmd or
                        b'sae_experiments.panorama_vitl30_dimension_vlm_runner' in cmd):
                    os.kill(int(proc.name), signal.SIGSTOP)
                    suspended.append(int(proc.name))
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                continue
        for version in VERSIONS:
            target = ROOT / 'experiments' / version
            (target / 'code').mkdir(parents=True, exist_ok=True)
            for package in [target, target / 'code']:
                (package / '__init__.py').touch(exist_ok=True)
        for row in moves:
            old, new = ROOT / row['old'], ROOT / row['new']
            if old.is_symlink() and old.resolve() == new.resolve():
                continue
            if new.exists() or new.is_symlink():
                raise RuntimeError(f'Refusing to overwrite {new}')
            if not old.exists():
                raise FileNotFoundError(old)
            new.parent.mkdir(parents=True, exist_ok=True)
            old.rename(new)
            if row['compatibility_symlink']:
                old.symlink_to(os.path.relpath(new.resolve(), old.parent.resolve()), target_is_directory=new.is_dir())
            completed.append(row)
        audit = ROOT / 'docs/project_inventory/layout_migration.json'
        audit.write_text(json.dumps({'migrated_utc': datetime.now(timezone.utc).isoformat(),
            'moves': completed, 'nas_directories_moved': False, 'data_files_reencoded': False},
            ensure_ascii=False, indent=2) + '\n')
        print(f'Moved {len(completed)} paths; preserved report/config/output aliases.')
    finally:
        for pid in suspended:
            try:
                os.kill(pid, signal.SIGCONT)
            except ProcessLookupError:
                pass


if __name__ == '__main__':
    main()
