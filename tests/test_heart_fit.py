"""The coronary arteries in heart.glb must rest on the heart surface: not floating above it, not buried in it."""
from pathlib import Path

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")
pytest.importorskip("rtree")

GLB = Path(__file__).resolve().parents[1] / "frontend" / "public" / "heart.glb"
SURFACE = ["myo-LAD", "myo-LCX", "myo-RCA", "myo-base"]


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
