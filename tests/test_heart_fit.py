"""The coronary arteries in heart.glb must rest on the heart surface: not floating above it, not buried in it."""
from pathlib import Path

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")
pytest.importorskip("rtree")

GLB = Path(__file__).resolve().parents[1] / "frontend" / "public" / "heart.glb"
SURFACE = ["myo"]


@pytest.fixture(scope="module")
def scene():
    sc = trimesh.load(GLB, force="scene")
    nodes = {n: sc.geometry[sc.graph[n][1]].copy().apply_transform(sc.graph[n][0]) for n in sc.graph.nodes_geometry}
    surf = trimesh.util.concatenate([nodes[k] for k in SURFACE])
    return nodes, surf, trimesh.proximity.ProximityQuery(surf)


def test_required_nodes(scene):
    nodes, _, _ = scene
    assert {"LAD", "LCX", "RCA", "LM", *SURFACE} <= set(nodes)


@pytest.mark.parametrize("vessel", ["LAD", "LCX", "RCA"])
def test_artery_rests_on_surface(scene, vessel):
    nodes, surf, pq = scene
    p = nodes[vessel].vertices
    cp, _, tri = pq.on_surface(p)
    height = ((p - cp) * surf.face_normals[tri]).sum(1)   # signed height above the surface (scene units, heart 3.4 tall)
    assert (height > 0.08).mean() < 0.01, "artery floats above the surface"
    assert (height < -0.03).mean() < 0.01, "artery is buried in the surface"


def test_every_mesh_has_normals():
    """Normals are required for lighting, the selection glow and the weak-evidence outline.
    Without them some GPUs (Apple Metal) discard those pixels entirely."""
    import json
    import struct
    data = GLB.read_bytes()
    n = struct.unpack("<I", data[12:16])[0]
    gltf = json.loads(data[20:20 + n])
    missing = [m["name"] for m in gltf["meshes"] if any("NORMAL" not in p["attributes"] for p in m["primitives"])]
    assert not missing, f"meshes without normals: {missing}"


def test_territory_weights_follow_aha_proportions(scene):
    """Smooth territory weights exist, and the ventricular surface splits roughly 7:5:5 (AHA, right-dominant)."""
    nodes, _, _ = scene
    data = GLB.read_bytes()
    import json
    import struct
    n = struct.unpack("<I", data[12:16])[0]
    gltf = json.loads(data[20:20 + n])
    myo = next(m for m in gltf["meshes"] if m["name"] == "myo")
    assert "_TERRITORY" in myo["primitives"][0]["attributes"]
    w = nodes["myo"].vertex_attributes["_TERRITORY"][:, :3]
    strong = w.max(1) > 0.5
    share = np.bincount(w[strong].argmax(1), minlength=3) / strong.sum()
    assert np.allclose(share, [7 / 17, 5 / 17, 5 / 17], atol=0.06), share
