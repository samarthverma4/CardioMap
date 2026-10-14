"""Put the BodyParts3D coronary arteries onto a textured heart surface and write frontend/public/heart.glb.

Step 2 of the heart pipeline. Step 1 (tools/build_heart.py) builds an anatomical heart from BodyParts3D;
this step keeps its coronary arteries and replaces its reconstructed surface with a textured surface model.

Surface used: "Realistic Human Heart" by neshallads,
https://sketchfab.com/3d-models/realistic-human-heart-3f8072336ce94d18b3d0d055a1ece089, CC BY 4.0.

Run:  python tools/fit_surface.py <bp3d_heart.glb> <surface scene.gltf>
      (bp3d_heart.glb = output of build_heart.py; the default output overwrites frontend/public/heart.glb)

Method:
1. Rigid + uniform-scale ICP of the BodyParts3D surface onto the target surface (best of several starts).
2. Each artery vertex is moved by the smoothed displacement that takes the nearby BodyParts3D surface onto the
   target surface, plus a small outward offset, so the arteries ride on the new epicardium.
3. Target faces are split into territories by nearest artery (ventricular faces only) for the territory tint.
4. Textures are downsized and the base colour is desaturated so the API's artery colours stay readable.
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageEnhance
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "frontend" / "public"
CREDIT = ('"Realistic Human Heart" (https://sketchfab.com/3d-models/realistic-human-heart-3f8072336ce94d18b3d0d055a1ece089) '
          "by neshallads (https://sketchfab.com/neshallads), CC BY 4.0")
VESSELS = ["LAD", "LCX", "RCA"]
TEX_SIZE = 1024
SATURATION = 0.55
OUTWARD = 0.006   # target units the arteries are lifted off the surface
HEIGHT = 3.4      # output height in scene units


def load_named(path: Path) -> dict[str, trimesh.Trimesh]:
    scene = trimesh.load(path, force="scene")
    out = {}
    for node in scene.graph.nodes_geometry:
        tf, gname = scene.graph[node]
        out[node] = scene.geometry[gname].copy().apply_transform(tf)
    return out


def load_surface(path: Path) -> trimesh.Trimesh:
    scene = trimesh.load(path, force="scene")
    parts = []
    for node in scene.graph.nodes_geometry:
        tf, gname = scene.graph[node]
        parts.append(scene.geometry[gname].copy().apply_transform(tf))
    assert len(parts) == 1, "expected a single textured mesh"
    return parts[0]


def fit(src: np.ndarray, target: trimesh.Trimesh) -> np.ndarray:
    """Best rigid + scale transform of src points onto target, trying several yaw starts."""
    s0 = target.extents.max() / np.ptp(src, axis=0).max()
    best = (np.inf, None)
    for yaw in np.radians([-30, -15, 0, 15, 30]):
        init = trimesh.transformations.rotation_matrix(yaw, [0, 1, 0])
        init[:3, :3] *= s0
        init[:3, 3] = target.bounds.mean(0) - (init[:3, :3] @ src.mean(0))
        m, _, cost = trimesh.registration.icp(src, target, initial=init, max_iterations=60, reflection=False, scale=True)
        if cost < best[0]:
            best = (cost, m)
    print("ICP cost", round(float(best[0]), 5))
    return best[1]


def texture(img: Image.Image, desaturate: bool = False) -> Image.Image:
    img = img.convert("RGB").resize((TEX_SIZE, TEX_SIZE), Image.LANCZOS)
    if desaturate:
        img = ImageEnhance.Color(img).enhance(SATURATION)
        img = ImageEnhance.Brightness(img).enhance(1.08)
    return img


def main(bp3d_glb: Path, surface_gltf: Path, out: Path = OUT / "heart.glb") -> None:
    bp = load_named(bp3d_glb)
    target = load_surface(surface_gltf)
    shell = trimesh.util.concatenate([bp[k] for k in ("myo-LAD", "myo-LCX", "myo-RCA", "myo-base")])
    vessels_src = trimesh.util.concatenate([bp[k] for k in ("aorta", "pulmonary") if k in bp])
    src_pts = np.vstack([shell.vertices, vessels_src.vertices])

    rng = np.random.default_rng(0)
    M = fit(src_pts[rng.choice(len(src_pts), min(5000, len(src_pts)), replace=False)], target)
    shell.apply_transform(M)
    arteries = {k: bp[k].copy().apply_transform(M) for k in VESSELS + ["LM"]}

    # displacement field: shell vertex -> closest point on target, smoothed over neighbours
    closest, _, tri = trimesh.proximity.closest_point(target, shell.vertices)
    disp = closest - shell.vertices
    tree = cKDTree(shell.vertices)
    _, nb = tree.query(shell.vertices, k=24)
    disp = disp[nb].mean(1)
    normals = target.face_normals[tri]
    for k, a in arteries.items():
        d, i = tree.query(a.vertices, k=8)
        w = 1.0 / (d + 1e-6)
        w /= w.sum(1, keepdims=True)
        shift = (disp[i] * w[..., None]).sum(1) + (normals[i] * w[..., None]).sum(1) * OUTWARD
        a.vertices = a.vertices + shift

    # territories on the target surface
    cent = target.triangles_center
    apts = np.vstack([arteries[v].vertices for v in VESSELS])
    alab = np.concatenate([np.full(len(arteries[v].vertices), i) for i, v in enumerate(VESSELS)])
    terr = alab[cKDTree(apts).query(cent)[1]]
    # ventricular = nearest BodyParts3D shell face belonged to a myo-<vessel> part (not myo-base)
    lab_src = np.concatenate([np.full(len(bp[k].vertices), j) for j, k in enumerate(("myo-LAD", "myo-LCX", "myo-RCA", "myo-base"))])
    ventricular = lab_src[cKDTree(shell.vertices).query(cent)[1]] != 3

    # material: downsized textures, desaturated base colour
    mat = target.visual.material
    base = getattr(mat, "baseColorTexture", None)
    pbr = trimesh.visual.material.PBRMaterial(
        name="heart",
        baseColorTexture=texture(base, desaturate=True) if base is not None else None,
        normalTexture=texture(mat.normalTexture) if getattr(mat, "normalTexture", None) is not None else None,
        metallicFactor=0.0,
        roughnessFactor=0.55,
    )
    target.visual.material = pbr

    # normalise: centre at the origin, height HEIGHT scene units (same frame the app's camera expects)
    N = np.eye(4)
    N[:3, :3] *= HEIGHT / target.extents[1]
    N[:3, 3] = -(N[:3, :3] @ target.bounds.mean(0))
    target.apply_transform(N)
    for a in arteries.values():
        a.apply_transform(N)

    scene = trimesh.Scene()
    counts = {}
    groups = {f"myo-{v}": np.where(ventricular & (terr == i))[0] for i, v in enumerate(VESSELS)}
    groups["myo-base"] = np.where(~ventricular)[0]
    for name, faces in groups.items():
        sub = target.submesh([faces], append=True)
        sub.visual.material = pbr
        scene.add_geometry(sub, node_name=name, geom_name=name)
        counts[name] = len(faces)
    for k, a in arteries.items():
        scene.add_geometry(a, node_name=k, geom_name=k)
        counts[k] = len(a.faces)
    out.parent.mkdir(parents=True, exist_ok=True)
    scene.export(out)
    meta = json.loads((OUT / "heart_meta.json").read_text()) if (OUT / "heart_meta.json").exists() else {}
    meta.update({"surface": CREDIT, "arteries": meta.get("source", ""), "faces": counts,
                 "note": "Coronary arteries from BodyParts3D fitted onto the textured surface by ICP + local projection."})
    (OUT / "heart_meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    print(json.dumps(counts), "size", round(out.stat().st_size / 1e6, 2), "MB")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
