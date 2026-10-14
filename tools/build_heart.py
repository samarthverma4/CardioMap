"""Build frontend/public/heart.glb from BodyParts3D (DBCLS, CC BY 4.0).

BodyParts3D 4.0 models the heart's cavities, the atrial walls and the coronary tree, but not the ventricular
wall. The epicardial surface is therefore reconstructed: an offset shell around the cavities and atrial walls,
at the median distance of the main coronary arteries from them, so the real arteries lie on its surface.

Inputs (download from https://dbarchive.biosciencedbc.jp/en/bodyparts3d/download.html):
  partof_element_parts.txt and the unzipped partof_BP3D_4.0_obj_99/ folder, in one directory.
Run:  python tools/build_heart.py /path/to/bp3d_dir
Out:  frontend/public/heart.glb, frontend/public/heart_meta.json

Output nodes (names are the contract with the frontend):
  LAD, LCX, RCA         coronary arteries (all branches of each), coloured from the API
  LM                    left main trunk (neutral)
  myo-LAD/LCX/RCA       epicardial surface split into each artery's territory (nearest artery)
  myo-base              atrial and other surface
  aorta, pulmonary      great vessels
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import fast_simplification
import numpy as np
import trimesh
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage import measure

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "frontend" / "public"
ATTRIBUTION = "BodyParts3D, © The Database Center for Life Science licensed under CC Attribution 4.0 International"

CAVITIES = {"LV": "FJ2422", "RV": "FJ2423", "RA": "FJ2424", "LA": "FJ2425"}
ATRIAL_WALLS = ["FJ2438", "FJ2439"]
ARTERY_CONCEPTS = {
    "LAD": ["FMA3862"],                                   # anterior interventricular branch + diagonals, conus, septal
    "LCX": ["FMA3895"],                                   # circumflex branch
    "RCA": ["FMA3802", "FMA3807", "FMA3815", "FMA3818",   # trunk, conus, anterior ventricular, marginal,
            "FMA3837", "FMA3840"],                        # posterior ventricular, posterior interventricular (right-dominant)
    "LM": ["FMA3855"],
}
VESSELS = {"aorta": ["FJ3413", "FJ3411"], "pulmonary": ["FJ2966"]}
VOXEL = 0.75       # mm
WALL_OFFSET = 1.0  # mm outside the atrial wall surfaces
OFFSET_PERCENTILE = 25
SHELL_FACES = 36000
SCALE = 0.028      # mm -> scene units


def to_scene(v: np.ndarray, centre: np.ndarray) -> np.ndarray:
    """BodyParts3D (x patient-left, y posterior, z superior) -> three.js (x viewer-right, y up, z toward viewer)."""
    c = v - centre
    return np.column_stack([c[:, 0], c[:, 2], -c[:, 1]]) * SCALE


def main(src: Path) -> None:
    objdir = src / "partof_BP3D_4.0_obj_99"
    elements: dict[str, set[str]] = defaultdict(set)
    with open(src / "partof_element_parts.txt") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            elements[r["concept id"]].add(r["element file id"])

    def load(ids) -> trimesh.Trimesh:
        return trimesh.util.concatenate([trimesh.load(objdir / f"{i}.obj", force="mesh") for i in sorted(ids)])

    arteries = {k: load(set().union(*(elements[c] for c in cs))) for k, cs in ARTERY_CONCEPTS.items()}
    cav = {k: load([v]) for k, v in CAVITIES.items()}
    walls = load(ATRIAL_WALLS)
    vessels = {k: load(v) for k, v in VESSELS.items()}

    # --- epicardial shell: offset surface around cavities (and, more tightly, atrial walls) --------------
    # The offset from the cavities is a low percentile of the coronary vertices' distance from them, so the
    # epicardial arteries sit on (not under) the surface; the atrial walls are already outer surfaces and get a thin offset.
    lo = np.minimum(walls.bounds[0], arteries["LAD"].bounds[0]) - 20
    hi = np.maximum(walls.bounds[1], arteries["LAD"].bounds[1]) + 20
    for m in cav.values():
        lo, hi = np.minimum(lo, m.bounds[0] - 20), np.maximum(hi, m.bounds[1] + 20)
    shape = np.ceil((hi - lo) / VOXEL).astype(int) + 1

    def distance_to(mesh: trimesh.Trimesh) -> np.ndarray:
        pts = np.vstack([mesh.vertices, trimesh.sample.sample_surface_even(mesh, 300000, radius=VOXEL / 2)[0]])
        grid = np.ones(shape, bool)
        grid[tuple(np.clip(np.round((pts - lo) / VOXEL).astype(int), 0, shape - 1).T)] = False
        return ndimage.distance_transform_edt(grid) * VOXEL

    cav_all = trimesh.util.concatenate(list(cav.values()))
    d_cav, d_wall = distance_to(cav_all), distance_to(walls)
    art_all = trimesh.util.concatenate([arteries[n] for n in ("LAD", "LCX", "RCA")])
    near = cKDTree(trimesh.sample.sample_surface_even(cav_all, 200000)[0])
    offset = float(np.percentile(near.query(art_all.vertices)[0], OFFSET_PERCENTILE))
    field = np.minimum(d_cav - offset, d_wall - WALL_OFFSET)
    verts, faces, _, _ = measure.marching_cubes(field, level=0.0, spacing=(VOXEL,) * 3)
    shell = trimesh.Trimesh(verts + lo, faces, process=True)
    parts = shell.split(only_watertight=False)
    shell = max(parts, key=lambda m: m.bounding_box.volume)   # outer surface; inner shells sit in cavities
    trimesh.smoothing.filter_taubin(shell, lamb=0.5, nu=-0.53, iterations=25)
    v, f = fast_simplification.simplify(shell.vertices, shell.faces, target_count=SHELL_FACES)
    shell = trimesh.Trimesh(v, f, process=True)
    trimesh.repair.fix_normals(shell)

    # --- territories: each shell face -> nearest artery (ventricles) or base (atria) -------------------
    cent = shell.triangles_center
    names = ["LAD", "LCX", "RCA"]
    apts = np.vstack([arteries[n].vertices for n in names])
    alab = np.concatenate([np.full(len(arteries[n].vertices), i) for i, n in enumerate(names)])
    terr = alab[cKDTree(apts).query(cent)[1]]
    cav_names = list(cav)
    cpts = np.vstack([cav[n].vertices for n in cav_names])
    clab = np.concatenate([np.full(len(cav[n].vertices), i) for i, n in enumerate(cav_names)])
    nearest_chamber = clab[cKDTree(cpts).query(cent)[1]]
    ventricular = np.isin(nearest_chamber, [cav_names.index("LV"), cav_names.index("RV")])

    centre = shell.bounds.mean(0)
    scene = trimesh.Scene()

    def add(name: str, m: trimesh.Trimesh, target: int | None = None):
        if target and len(m.faces) > target:
            v, f = fast_simplification.simplify(m.vertices, m.faces, target_count=target)
            m = trimesh.Trimesh(v, f, process=True)
        m = trimesh.Trimesh(to_scene(m.vertices, centre), m.faces, process=False)
        m.fix_normals()
        scene.add_geometry(m, node_name=name, geom_name=name)
        return len(m.faces)

    counts = {}
    for i, n in enumerate(names):
        counts[f"myo-{n}"] = add(f"myo-{n}", shell.submesh([np.where(ventricular & (terr == i))[0]], append=True))
    counts["myo-base"] = add("myo-base", shell.submesh([np.where(~ventricular)[0]], append=True))
    for n in names + ["LM"]:
        counts[n] = add(n, arteries[n], target=9000)
    for n, m in vessels.items():
        counts[n] = add(n, m)

    OUT.mkdir(parents=True, exist_ok=True)
    scene.export(OUT / "heart.glb")
    meta = {"source": ATTRIBUTION, "url": "https://dbarchive.biosciencedbc.jp/en/bodyparts3d/",
            "shell_offset_mm": round(offset, 2), "faces": counts,
            "note": "Epicardial surface reconstructed as an offset shell; coronary arteries are original BodyParts3D meshes."}
    (OUT / "heart_meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    print(json.dumps(meta, indent=1, ensure_ascii=False))
    print("size", round((OUT / "heart.glb").stat().st_size / 1e6, 2), "MB")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
