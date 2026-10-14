"""Put the BodyParts3D coronary arteries onto a textured heart surface and write frontend/public/heart.glb.

Step 2 of the heart pipeline. Step 1 (tools/build_heart.py) builds an anatomical heart from BodyParts3D;
this step keeps its coronary arteries and replaces its reconstructed surface with a textured surface model.

Surface used: "Realistic Human Heart" by neshallads,
https://sketchfab.com/3d-models/realistic-human-heart-3f8072336ce94d18b3d0d055a1ece089, CC BY 4.0.

Run:  python tools/fit_surface.py <bp3d_heart.glb> <surface scene.gltf>
      (bp3d_heart.glb = output of build_heart.py; the default output overwrites frontend/public/heart.glb)

Method:
1. Rigid + uniform-scale ICP of the BodyParts3D surface onto the target surface (best of several starts).
2. Non-rigid registration (NRICP, Amberg et al.) deforms the BodyParts3D surface smoothly onto the target;
   the arteries are carried by the same displacement field.
3. Seating: each artery is moved along the local surface normal so its centreline rests just above the
   surface (tube half-embedded), removing floating and buried segments.
4. Target faces are split into territories by nearest artery (ventricular faces only) for the territory tint.
5. Textures are downsized and the base colour is desaturated so the API's artery colours stay readable.
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import fast_simplification
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
NRICP_FACES = 6000
NRICP_STEPS = None  # trimesh default schedule (stiffness 0.01 -> 0.002, landmark-free)
SEAT_K = 30         # neighbours used to estimate a tube's centreline height
SEAT_ITERS = 3
SEAT_FRACTION = 0.35  # centreline height above the surface, as a fraction of the tube radius
TERRITORY_SMOOTHING = 6
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

    # 1) smooth non-rigid registration of the (decimated) BodyParts3D surface onto the target surface;
    #    arteries are carried by the resulting displacement field
    v, f = fast_simplification.simplify(shell.vertices, shell.faces, target_count=NRICP_FACES)
    src = trimesh.Trimesh(v, f, process=True)
    deformed = trimesh.registration.nricp_amberg(src, target, steps=NRICP_STEPS, distance_threshold=0.15,
                                                return_records=False)
    disp = deformed - src.vertices
    tree = cKDTree(src.vertices)
    for a in arteries.values():
        d, i = tree.query(a.vertices, k=8)
        w = 1.0 / (d + 1e-6)
        w /= w.sum(1, keepdims=True)
        a.vertices = a.vertices + (disp[i] * w[..., None]).sum(1)
    shell_deformed = shell.copy()
    d, i = tree.query(shell.vertices, k=8)
    w = 1.0 / (d + 1e-6)
    w /= w.sum(1, keepdims=True)
    shell_deformed.vertices = shell.vertices + (disp[i] * w[..., None]).sum(1)

    # 2) seating: move each artery along the local surface normal so its centreline rests just above the
    #    surface (tube half-embedded), never floating or buried. Heights are averaged over each vertex's
    #    neighbourhood (~ the tube cross-section), and the correction is smoothed along the vessel.
    prox = trimesh.proximity.ProximityQuery(target)
    for name, a in arteries.items():
        if name == "LM":   # short stem at the aortic root: too few vertices for a stable estimate
            continue
        knn = cKDTree(a.vertices).query(a.vertices, k=SEAT_K)[1]
        for _ in range(SEAT_ITERS):
            cp, _, tri = prox.on_surface(a.vertices)
            n = target.face_normals[tri]
            h = ((a.vertices - cp) * n).sum(1)            # signed height of each vertex
            centre = h[knn].mean(1)                       # ~ centreline height
            radius = 0.5 * (h[knn].max(1) - h[knn].min(1))
            shift = (SEAT_FRACTION * radius - centre)[knn].mean(1)
            a.vertices = a.vertices + n[knn].mean(1) * shift[:, None]

    # territories on the target surface
    cent = target.triangles_center
    apts = np.vstack([arteries[v].vertices for v in VESSELS])
    alab = np.concatenate([np.full(len(arteries[v].vertices), i) for i, v in enumerate(VESSELS)])
    terr = alab[cKDTree(apts).query(cent)[1]]
    # ventricular = nearest BodyParts3D shell face belonged to a myo-<vessel> part (not myo-base)
    lab_src = np.concatenate([np.full(len(bp[k].vertices), j) for j, k in enumerate(("myo-LAD", "myo-LCX", "myo-RCA", "myo-base"))])
    ventricular = lab_src[cKDTree(shell_deformed.vertices).query(cent)[1]] != 3

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
    # face labels 0..2 = LAD/LCX/RCA territory, 3 = base; smooth with a neighbour-majority filter so the
    # territory tint has clean edges instead of isolated triangles
    label = np.where(ventricular, terr, 3)
    adj = target.face_adjacency
    for _ in range(TERRITORY_SMOOTHING):
        votes = np.zeros((len(label), 4))
        np.add.at(votes, (np.arange(len(label)), label), 1.5)
        np.add.at(votes, (adj[:, 0], label[adj[:, 1]]), 1)
        np.add.at(votes, (adj[:, 1], label[adj[:, 0]]), 1)
        label = votes.argmax(1)
    groups = {f"myo-{v}": np.where(label == i)[0] for i, v in enumerate(VESSELS)}
    groups["myo-base"] = np.where(label == 3)[0]
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
                 "note": "Coronary arteries from BodyParts3D fitted onto the textured surface: ICP, non-rigid registration (NRICP), seating on the surface."})
    (OUT / "heart_meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    print(json.dumps(counts), "size", round(out.stat().st_size / 1e6, 2), "MB")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
