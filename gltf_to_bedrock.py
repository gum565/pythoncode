#!/usr/bin/env python3
"""Convert Blockbench-style cuboid GLTF meshes into Bedrock geometry JSON.

The converter is deliberately loss-averse: it reconstructs each source cuboid
from its existing faces and UVs, and refuses unsupported/non-cuboid geometry.
It does not voxelize meshes and never emits deprecated poly_mesh data.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import re
import struct
import sys
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


COMPONENTS = {
    5120: ("b", 1), 5121: ("B", 1), 5122: ("h", 2),
    5123: ("H", 2), 5125: ("I", 4), 5126: ("f", 4),
}
TYPE_SIZE = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16}
EPS = 1e-5
FACE_ORDER = {
    "east":  (0, +1), "west": (0, -1),
    "up":    (1, +1), "down": (1, -1),
    "south": (2, +1), "north": (2, -1),
}


def fail(message: str) -> "NoReturn":
    raise ValueError(message)


def next_pow2(n: int) -> int:
    p = 1
    while p < max(1, n):
        p <<= 1
    return p


def parse_gltf(path: Path) -> tuple[dict, list[bytes]]:
    """Load .gltf or .glb and resolve buffers."""
    if path.suffix.lower() == ".glb":
        raw = path.read_bytes()
        if len(raw) < 20 or raw[:4] != b"glTF":
            fail("Файл не похож на корректный GLB.")
        version, total = struct.unpack_from("<II", raw, 4)
        if version != 2 or total > len(raw):
            fail("Поддерживается только GLB 2.0.")
        off, doc, bin_chunk = 12, None, None
        while off + 8 <= total:
            length, kind = struct.unpack_from("<II", raw, off)
            off += 8
            chunk = raw[off:off + length]
            off += length
            if kind == 0x4E4F534A:
                doc = json.loads(chunk.decode("utf-8").rstrip("\x00 \t\r\n"))
            elif kind == 0x004E4942:
                bin_chunk = chunk
        if doc is None:
            fail("В GLB отсутствует JSON chunk.")
        buffers = []
        for i, b in enumerate(doc.get("buffers", [])):
            uri = b.get("uri")
            if uri:
                buffers.append(resolve_uri(uri, path.parent))
            elif i == 0 and bin_chunk is not None:
                buffers.append(bin_chunk)
            else:
                fail(f"Не удалось найти buffer {i} в GLB.")
        return doc, buffers

    doc = json.loads(path.read_text(encoding="utf-8"))
    buffers = []
    for i, b in enumerate(doc.get("buffers", [])):
        uri = b.get("uri")
        if not uri:
            fail(f"У buffer {i} нет URI; для встроенного бинарного буфера используйте GLB.")
        buffers.append(resolve_uri(uri, path.parent))
    return doc, buffers


def resolve_uri(uri: str, base: Path) -> bytes:
    if uri.startswith("data:"):
        header, payload = uri.split(",", 1)
        if ";base64" in header:
            return base64.b64decode(payload)
        return urllib.parse.unquote_to_bytes(payload)
    parsed = urllib.parse.urlparse(uri)
    if parsed.scheme:
        fail(f"Внешний URL в glTF не поддерживается: {uri}")
    return (base / urllib.parse.unquote(parsed.path)).read_bytes()


def read_accessor(doc: dict, buffers: list[bytes], index: int) -> np.ndarray:
    a = doc["accessors"][index]
    if "sparse" in a:
        fail("Sparse accessors не поддерживаются в этой версии конвертера.")
    if "bufferView" not in a:
        fail(f"Accessor {index} без bufferView не поддерживается.")
    bv = doc["bufferViews"][a["bufferView"]]
    fmt, scalar_size = COMPONENTS[a["componentType"]]
    n = TYPE_SIZE[a["type"]]
    count = a["count"]
    elem_size = scalar_size * n
    stride = bv.get("byteStride", elem_size)
    offset = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
    raw = buffers[bv["buffer"]]
    vals = [struct.unpack_from("<" + fmt * n, raw, offset + j * stride) for j in range(count)]
    result = np.asarray(vals)
    if a.get("normalized") and a["componentType"] != 5126:
        if a["componentType"] in (5120, 5122):
            maxv = 127 if a["componentType"] == 5120 else 32767
            result = np.maximum(result / maxv, -1.0)
        else:
            maxv = {5121: 255, 5123: 65535, 5125: 4294967295}[a["componentType"]]
            result = result / maxv
    return result.astype(float)


def quat_to_mat(q: list[float]) -> np.ndarray:
    x, y, z, w = q
    norm = math.sqrt(x*x + y*y + z*z + w*w)
    if norm < EPS:
        fail("У узла задан некорректный quaternion.")
    x, y, z, w = x/norm, y/norm, z/norm, w/norm
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ], dtype=float)


def local_matrix(node: dict) -> np.ndarray:
    if "matrix" in node:
        return np.asarray(node["matrix"], dtype=float).reshape((4, 4), order="F")
    t = np.array(node.get("translation", [0, 0, 0]), dtype=float)
    s = np.array(node.get("scale", [1, 1, 1]), dtype=float)
    r = quat_to_mat(node.get("rotation", [0, 0, 0, 1]))
    m = np.eye(4)
    m[:3, :3] = r @ np.diag(s)
    m[:3, 3] = t
    return m


def world_matrices(doc: dict) -> dict[int, np.ndarray]:
    nodes = doc.get("nodes", [])
    parents = {}
    for i, n in enumerate(nodes):
        for child in n.get("children", []):
            if child in parents:
                fail("Узел GLTF имеет более одного родителя; такой граф не поддерживается.")
            parents[child] = i
    roots = doc.get("scenes", [{}])[doc.get("scene", 0)].get("nodes", []) if doc.get("scenes") else [i for i in range(len(nodes)) if i not in parents]
    result = {}
    def visit(i: int, parent: np.ndarray):
        if i in result:
            return
        m = parent @ local_matrix(nodes[i])
        result[i] = m
        for c in nodes[i].get("children", []):
            visit(c, m)
    for root in roots:
        visit(root, np.eye(4))
    for i in range(len(nodes)):
        if i not in result:
            visit(i, np.eye(4))
    return result


def matrix_to_euler_xyz(r: np.ndarray) -> np.ndarray:
    # Equivalent to the glTF/Three.js XYZ Euler convention for ordinary model rotations.
    y = math.asin(max(-1.0, min(1.0, -r[2, 0])))
    if abs(math.cos(y)) > 1e-7:
        x = math.atan2(r[2, 1], r[2, 2])
        z = math.atan2(r[1, 0], r[0, 0])
    else:
        x = math.atan2(-r[1, 2], r[1, 1])
        z = 0.0
    return np.degrees([x, y, z])


def load_images(doc: dict, buffers: list[bytes], base: Path) -> tuple[list[Image.Image], list[str]]:
    images, labels = [], []
    for i, im in enumerate(doc.get("images", [])):
        if "uri" in im:
            blob = resolve_uri(im["uri"], base)
        elif "bufferView" in im:
            bv = doc["bufferViews"][im["bufferView"]]
            start = bv.get("byteOffset", 0)
            blob = buffers[bv["buffer"]][start:start + bv["byteLength"]]
        else:
            fail(f"Изображение {i} не содержит ни URI, ни bufferView.")
        try:
            img = Image.open(__import__("io").BytesIO(blob)).convert("RGBA")
        except Exception as e:
            fail(f"Не удалось прочитать изображение {i}: {e}")
        images.append(img)
        labels.append(im.get("name") or f"image_{i}")
    return images, labels


def material_image_map(doc: dict) -> dict[int, int | None]:
    out = {}
    for i, material in enumerate(doc.get("materials", [])):
        texinfo = material.get("pbrMetallicRoughness", {}).get("baseColorTexture")
        if texinfo is None:
            out[i] = None
            continue
        if texinfo.get("texCoord", 0) != 0:
            fail(f"Материал {i} использует TEXCOORD_{texinfo['texCoord']}; поддерживается только TEXCOORD_0.")
        if "KHR_texture_transform" in texinfo.get("extensions", {}):
            fail(f"Материал {i} использует KHR_texture_transform, который нельзя безопасно выразить текущим конвертером.")
        ti = texinfo["index"]
        out[i] = doc["textures"][ti].get("source")
    return out


def pack_atlas(images: list[Image.Image], used_image_ids: list[int]) -> tuple[Image.Image, dict[int, tuple[int, int]]]:
    """Pack images 1:1 into a power-of-two atlas with one-pixel edge padding."""
    # Deduplicate identical decoded pixels, so repeated GLTF images do not bloat the atlas.
    unique: list[int] = []
    hash_to_id = {}
    remap = {}
    for iid in used_image_ids:
        im = images[iid]
        key = hashlib.sha256(struct.pack("<II", *im.size) + im.tobytes()).hexdigest()
        if key not in hash_to_id:
            hash_to_id[key] = iid
            unique.append(iid)
        remap[iid] = hash_to_id[key]

    items = sorted(unique, key=lambda i: max(images[i].size), reverse=True)
    if not items:
        fail("В используемых материалах нет базовых текстур.")
    min_w = max(images[i].width + 2 for i in items)
    widths = []
    w = next_pow2(min_w)
    while w <= 8192:
        widths.append(w)
        w *= 2
    best = None
    for atlas_w in widths:
        x = y = row_h = 0
        placements = {}
        for iid in items:
            iw, ih = images[iid].size
            pw, ph = iw + 2, ih + 2
            if x + pw > atlas_w:
                y += row_h
                x, row_h = 0, 0
            placements[iid] = (x + 1, y + 1)
            x += pw
            row_h = max(row_h, ph)
        atlas_h = next_pow2(y + row_h)
        score = atlas_w * atlas_h
        if best is None or score < best[0]:
            best = (score, atlas_w, atlas_h, placements)
    if best is None or best[2] > 8192:
        fail("Текстуры слишком большие для одного Bedrock-атласа (лимит 8192 px).")
    _, atlas_w, atlas_h, placements = best
    atlas = Image.new("RGBA", (atlas_w, atlas_h), (0, 0, 0, 0))
    for iid in items:
        x, y = placements[iid]
        im = images[iid]
        atlas.paste(im, (x, y))
        # Duplicate edges/corners into the one-pixel gutter to prevent linear-filter bleed.
        atlas.paste(im.crop((0, 0, im.width, 1)), (x, y - 1))
        atlas.paste(im.crop((0, im.height - 1, im.width, im.height)), (x, y + im.height))
        atlas.paste(im.crop((0, 0, 1, im.height)), (x - 1, y))
        atlas.paste(im.crop((im.width - 1, 0, im.width, im.height)), (x + im.width, y))
        for dx in (-1, im.width):
            for dy in (-1, im.height):
                atlas.putpixel((x + dx, y + dy), im.getpixel((0 if dx < 0 else im.width - 1, 0 if dy < 0 else im.height - 1)))
    return atlas, {iid: placements[remap[iid]] for iid in used_image_ids}


UV_TRANSFORMS = {
    "identity": (lambda uv: np.array(uv, dtype=float), None),
    "flip_x": (lambda uv: np.array([1.0 - uv[0], uv[1]]), Image.Transpose.FLIP_LEFT_RIGHT),
    "flip_y": (lambda uv: np.array([uv[0], 1.0 - uv[1]]), Image.Transpose.FLIP_TOP_BOTTOM),
    "rotate_180": (lambda uv: np.array([1.0 - uv[0], 1.0 - uv[1]]), Image.Transpose.ROTATE_180),
    "transpose": (lambda uv: np.array([uv[1], uv[0]]), Image.Transpose.TRANSPOSE),
    "rotate_90_cw": (lambda uv: np.array([1.0 - uv[1], uv[0]]), Image.Transpose.ROTATE_270),
    "rotate_90_ccw": (lambda uv: np.array([uv[1], 1.0 - uv[0]]), Image.Transpose.ROTATE_90),
    "transverse": (lambda uv: np.array([1.0 - uv[1], 1.0 - uv[0]]), Image.Transpose.TRANSVERSE),
}


def transform_face_uv(uv_by_pos: dict[tuple, list[np.ndarray]], name: str) -> dict[tuple, list[np.ndarray]]:
    fn = UV_TRANSFORMS[name][0]
    return {p: [fn(uv) for uv in values] for p, values in uv_by_pos.items()}


def canonical_corners(face: str, lo: np.ndarray, hi: np.ndarray) -> list[np.ndarray]:
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    table = {
        "east":  [[x1,y1,z1],[x1,y1,z0],[x1,y0,z1],[x1,y0,z0]],
        "west":  [[x0,y1,z0],[x0,y1,z1],[x0,y0,z0],[x0,y0,z1]],
        "up":    [[x0,y1,z0],[x1,y1,z0],[x0,y1,z1],[x1,y1,z1]],
        "down":  [[x0,y0,z1],[x1,y0,z1],[x0,y0,z0],[x1,y0,z0]],
        "south": [[x0,y1,z1],[x1,y1,z1],[x0,y0,z1],[x1,y0,z1]],
        "north": [[x1,y1,z0],[x0,y1,z0],[x1,y0,z0],[x0,y0,z0]],
    }
    return [np.asarray(p, dtype=float) for p in table[face]]


def solve_uv(face: str, lo: np.ndarray, hi: np.ndarray, uv_by_pos: dict[tuple, list[np.ndarray]], atlas_offset: tuple[int, int], image_size: tuple[int, int], atlas_size: tuple[int, int]) -> dict:
    corners = canonical_corners(face, lo, hi)
    W, H = image_size
    ox, oy = atlas_offset
    AW, AH = atlas_size
    def key(p): return tuple(round(float(v), 6) for v in p)
    corner_uv = []
    for p in corners:
        vals = uv_by_pos.get(key(p), [])
        if not vals:
            fail(f"Грань {face}: не удалось найти UV одной из четырёх вершин куба.")
        # UV normalized in glTF, with image origin at top-left.
        arr = np.asarray(vals, dtype=float)
        if np.max(np.ptp(arr, axis=0)) > 2e-5:
            fail(f"Грань {face}: одна геометрическая вершина имеет разные UV (шов/разрыв на грани).")
        uv = arr.mean(axis=0)
        corner_uv.append(np.array([uv[0] * W + ox, uv[1] * H + oy]))
    corners_uv = np.asarray(corner_uv)
    min_uv, max_uv = corners_uv.min(axis=0), corners_uv.max(axis=0)
    # Enumerate signed UV extents and 90-degree rotations exactly as Blockbench maps cube faces.
    candidates = []
    for sx in (1, -1):
        for sy in (1, -1):
            u0 = min_uv[0] if sx > 0 else max_uv[0]
            v0 = min_uv[1] if sy > 0 else max_uv[1]
            du = (max_uv[0] - min_uv[0]) * sx
            dv = (max_uv[1] - min_uv[1]) * sy
            base = [np.array([u0, v0]), np.array([u0 + du, v0]), np.array([u0, v0 + dv]), np.array([u0 + du, v0 + dv])]
            for rotation in (0, 90, 180, 270):
                rotated = [a.copy() for a in base]
                for _ in range(rotation // 90):
                    rotated = [rotated[2], rotated[3], rotated[1], rotated[0]]
                err = float(np.max(np.abs(np.asarray(rotated) - corners_uv)))
                candidates.append((err, rotation, u0, v0, du, dv))
    err, rotation, u0, v0, du, dv = min(candidates, key=lambda v: v[0])
    if err > 0.03:
        fail(f"Грань {face}: UV нельзя точно представить прямоугольником Bedrock per-face (ошибка {err:.4f}px).")

    # Validate every additional mesh vertex on this plane against the same affine map.
    for pkey, vals in uv_by_pos.items():
        p = np.asarray(pkey, dtype=float)
        axis, sign = FACE_ORDER[face]
        if abs(p[axis] - (hi[axis] if sign > 0 else lo[axis])) > 1e-4:
            continue
        pos_corners = np.asarray([key(c) for c in corners], dtype=float)
        # Locate normalized position in the source face rectangle, then infer UV via bilinear affine interpolation.
        other = [a for a in range(3) if a != axis]
        a0, a1 = lo[other[0]], hi[other[0]]
        b0, b1 = lo[other[1]], hi[other[1]]
        tx = 0.0 if abs(a1-a0) < EPS else (p[other[0]]-a0)/(a1-a0)
        ty = 0.0 if abs(b1-b0) < EPS else (p[other[1]]-b0)/(b1-b0)
        # UV field is affine for the supported rectangular mappings; fit from the four corner positions.
        xy = np.array([[c[other[0]], c[other[1]], 1.0] for c in corners])
        coeff = np.linalg.lstsq(xy, corners_uv, rcond=None)[0]
        expected = np.array([p[other[0]], p[other[1]], 1.0]) @ coeff
        for value in vals:
            actual = np.array([value[0] * W + ox, value[1] * H + oy])
            if np.max(np.abs(actual - expected)) > 0.03:
                fail(f"Грань {face}: внутри UV развёртки обнаружен разрыв или нелинейное искажение.")

    # JSON's up/down face UV convention is reversed by Blockbench's Bedrock parser/compiler.
    if face in ("up", "down"):
        return {"uv": [round(u0 + du, 6), round(v0 + dv, 6)], "uv_size": [round(-du, 6), round(-dv, 6)], **({"uv_rotation": rotation} if rotation else {})}
    return {"uv": [round(u0, 6), round(v0, 6)], "uv_size": [round(du, 6), round(dv, 6)], **({"uv_rotation": rotation} if rotation else {})}


@dataclass
class MeshData:
    node_index: int
    lo: np.ndarray
    hi: np.ndarray
    faces: dict[str, dict]
    translation: np.ndarray
    rotation: np.ndarray
    scale: np.ndarray
    name: str


def get_triangles(doc: dict, buffers: list[bytes], node: dict) -> list[tuple[np.ndarray, np.ndarray, int | None]]:
    mesh = doc["meshes"][node["mesh"]]
    triangles = []
    for pi, prim in enumerate(mesh.get("primitives", [])):
        if prim.get("mode", 4) != 4:
            fail(f"Mesh primitive mode {prim.get('mode')} не поддерживается: ожидаются TRIANGLES.")
        attrs = prim.get("attributes", {})
        if "POSITION" not in attrs or "TEXCOORD_0" not in attrs:
            fail("У примитива нет POSITION или TEXCOORD_0.")
        pos = read_accessor(doc, buffers, attrs["POSITION"])
        uv = read_accessor(doc, buffers, attrs["TEXCOORD_0"])
        idx = read_accessor(doc, buffers, prim["indices"]).reshape(-1).astype(int) if "indices" in prim else np.arange(len(pos), dtype=int)
        if len(idx) % 3:
            fail("Количество вершин индексов примитива не делится на три.")
        material = prim.get("material")
        for j in range(0, len(idx), 3):
            ix = idx[j:j+3]
            triangles.append((pos[ix], uv[ix], material))
    return triangles


def convert(path: Path, outdir: Path, identifier: str, requested_scale: str) -> tuple[Path, Path, dict]:
    doc, buffers = parse_gltf(path)
    if doc.get("asset", {}).get("version") != "2.0":
        fail("Поддерживается только glTF 2.0.")
    if doc.get("animations") or any("skin" in n for n in doc.get("nodes", [])):
        fail("Скины и анимации GLTF пока не преобразуются в статические Bedrock-кубы.")
    images, image_names = load_images(doc, buffers, path.parent)
    if not images:
        fail("В GLTF не найдено изображений.")
    mat_to_img = material_image_map(doc)
    matrices = world_matrices(doc)

    node_specs = []
    overall = []
    for ni, node in enumerate(doc.get("nodes", [])):
        if "mesh" not in node:
            continue
        m = matrices[ni]
        linear, translation = m[:3, :3], m[:3, 3]
        scales = np.linalg.norm(linear, axis=0)
        if np.any(scales < EPS):
            fail(f"Узел {ni} имеет нулевой scale.")
        rotation = linear / scales[None, :]
        if np.max(np.abs(rotation.T @ rotation - np.eye(3))) > 1e-4 or np.linalg.det(rotation) < 0.999:
            fail(f"Узел {ni}: shear, отражение или неортогональная матрица не поддержаны.")
        triangles = get_triangles(doc, buffers, node)
        if not triangles:
            fail(f"Узел {ni} содержит пустой mesh.")
        # Mesh-local vertices remain axis-aligned in these exports. Node transforms are retained as bone transforms.
        pts = np.vstack([tri[0] * scales for tri in triangles])
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        if np.any(hi - lo < 1e-6):
            fail(f"Узел {ni}: нулевой размер по одной из осей; это не кубоид.")
        grouped: dict[str, dict] = {}
        for positions, uvs, material in triangles:
            nrm = np.cross(positions[1] - positions[0], positions[2] - positions[0])
            norm = np.linalg.norm(nrm)
            if norm < 1e-10:
                continue
            nrm = nrm / norm
            axis = int(np.argmax(np.abs(nrm)))
            if abs(abs(nrm[axis]) - 1.0) > 1e-4:
                fail(f"Узел {ni}: найдена неортогональная грань; конвертер сохраняет только исходные кубоиды.")
            plane = float(np.mean(positions[:, axis] * scales[axis]))
            if abs(plane - lo[axis]) <= 1e-4:
                sign = -1
            elif abs(plane - hi[axis]) <= 1e-4:
                sign = +1
            else:
                fail(f"Узел {ni}: поверхность находится внутри габаритного кубоида, а не на его стороне.")
            face = {(0,1): "east", (0,-1): "west", (1,1): "up", (1,-1): "down", (2,1): "south", (2,-1): "north"}[(axis, sign)]
            item = grouped.setdefault(face, {"uv_by_pos": {}, "images": set()})
            image_id = mat_to_img.get(material) if material is not None else None
            if image_id is None:
                fail(f"Узел {ni}, грань {face}: материал не содержит baseColorTexture; невозможно сохранить рисунок.")
            item["images"].add(image_id)
            for p, uv in zip(positions, uvs):
                ps = p * scales
                key = tuple(round(float(v), 6) for v in ps)
                item["uv_by_pos"].setdefault(key, []).append(np.asarray(uv, dtype=float))
        face_specs = {}
        for face, data in grouped.items():
            if len(data["images"]) != 1:
                fail(f"Узел {ni}, грань {face}: на одной стороне смешаны разные материалы/текстуры.")
            face_specs[face] = {"uv_by_pos": data["uv_by_pos"], "image_id": next(iter(data["images"]))}
        # Source face omission is kept: missing faces are omitted from the per-face UV object.
        node_specs.append({"index": ni, "lo": lo, "hi": hi, "faces": face_specs, "translation": translation.copy(), "rotation": rotation, "scale": scales, "name": node.get("name", "")})
        for corner in np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]):
            overall.append(translation + rotation @ corner)

    if not node_specs:
        fail("GLTF не содержит узлов с геометрией.")
    if requested_scale == "auto":
        bounds = np.asarray(overall)
        extent = float(np.max(bounds.max(axis=0) - bounds.min(axis=0)))
        unit_scale = 16.0 if extent <= 4.0 else 1.0
    else:
        unit_scale = float(requested_scale)

    # Keep the texture pixels intact while adapting mirrored/diagonal UV layouts.
    # Some GLTF cube faces use a reflected diagonal map that Bedrock's rectangle
    # plus uv_rotation cannot express directly. Transforming a copy of the full
    # source image and its UVs preserves the sampled texels exactly.
    variant_cache: dict[tuple[int, str], int] = {}
    used_images = set()
    for spec in node_specs:
        for face, fdata in spec["faces"].items():
            source_id = fdata["image_id"]
            selected = None
            last_error = None
            for transform_name, (uv_fn, pillow_op) in UV_TRANSFORMS.items():
                transformed_uv = transform_face_uv(fdata["uv_by_pos"], transform_name)
                try:
                    solve_uv(face, spec["lo"], spec["hi"], transformed_uv, (0, 0), images[source_id].size, images[source_id].size)
                    selected = (transform_name, transformed_uv)
                    break
                except ValueError as e:
                    last_error = e
            if selected is None:
                fail(f"Узел {spec['index']}, грань {face}: UV не удаётся выразить даже после точного отражения/поворота текстуры. {last_error}")
            transform_name, fdata["uv_by_pos"] = selected
            cache_key = (source_id, transform_name)
            if cache_key not in variant_cache:
                new_id = len(images)
                op = UV_TRANSFORMS[transform_name][1]
                images.append(images[source_id].copy() if op is None else images[source_id].transpose(op))
                image_names.append(f"{image_names[source_id]}_{transform_name}")
                variant_cache[cache_key] = new_id
            fdata["image_id"] = variant_cache[cache_key]
            used_images.add(fdata["image_id"])

    atlas, offsets = pack_atlas(images, sorted(used_images))
    atlas_wh = atlas.size

    bones = []
    for spec in node_specs:
        ni, lo, hi = spec["index"], spec["lo"], spec["hi"]
        t = np.asarray(spec["translation"]) * unit_scale
        R = spec["rotation"]
        angles = matrix_to_euler_xyz(R)
        # Blockbench's Bedrock parser mirrors X and reverses X/Y Euler signs; invert that on export.
        json_pivot = [-float(t[0]), float(t[1]), float(t[2])]
        json_rotation = [-float(angles[0]), -float(angles[1]), float(angles[2])]
        origin = [
            -float((spec["translation"][0] + hi[0]) * unit_scale),
            float((spec["translation"][1] + lo[1]) * unit_scale),
            float((spec["translation"][2] + lo[2]) * unit_scale),
        ]
        size = [float((hi - lo)[i] * unit_scale) for i in range(3)]
        uv_faces = {}
        for face, fdata in spec["faces"].items():
            iid = fdata["image_id"]
            uv_faces[face] = solve_uv(face, lo, hi, fdata["uv_by_pos"], offsets[iid], images[iid].size, atlas_wh)
        bone = {
            "name": f"part_{ni:03d}",
            "pivot": [round(v, 6) for v in json_pivot],
            "cubes": [{
                "origin": [round(v, 6) for v in origin],
                "size": [round(v, 6) for v in size],
                "uv": uv_faces,
            }],
        }
        if np.max(np.abs(json_rotation)) > 1e-4:
            bone["rotation"] = [round(float(v), 6) for v in json_rotation]
        bones.append(bone)

    safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", identifier).strip("._") or "converted_model"
    geometry_id = safe_id if safe_id.startswith("geometry.") else f"geometry.{safe_id}"
    geometry = {
        "format_version": "1.12.0",
        "minecraft:geometry": [{
            "description": {
                "identifier": geometry_id,
                "texture_width": atlas.width,
                "texture_height": atlas.height,
            },
            "bones": bones,
        }],
    }
    outdir.mkdir(parents=True, exist_ok=True)
    geometry_path = outdir / f"{safe_id}.geo.json"
    texture_path = outdir / f"{safe_id}.png"
    geometry_path.write_text(json.dumps(geometry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    atlas.save(texture_path, format="PNG", optimize=False)
    report = {
        "source": str(path), "geometry": str(geometry_path), "texture": str(texture_path),
        "cubes": len(bones), "faces_mapped": sum(len(b["cubes"][0]["uv"]) for b in bones),
        "atlas_size": list(atlas.size), "scale": unit_scale,
        "source_images": [image_names[i] for i in sorted(used_images)],
        "generator": doc.get("asset", {}).get("generator", "unknown"),
    }
    return geometry_path, texture_path, report


def main() -> int:
    parser = argparse.ArgumentParser(description="Сохранить исходные кубы GLTF в Bedrock geometry JSON без вокселизации и poly_mesh.")
    parser.add_argument("input", type=Path, help="Исходный .gltf или .glb файл")
    parser.add_argument("-o", "--out", type=Path, default=Path("bedrock_out"), help="Папка результата (по умолчанию bedrock_out)")
    parser.add_argument("--name", help="Идентификатор geometry.<name>; по умолчанию имя файла")
    parser.add_argument("--scale", default="auto", help="Множитель модели, например 1 или 16; auto для экспорта Blockbench (по умолчанию)")
    args = parser.parse_args()
    if not args.input.exists():
        parser.error(f"Файл не найден: {args.input}")
    name = args.name or args.input.stem.lower().replace(" ", "_").replace("-", "_")
    try:
        geo, tex, report = convert(args.input.resolve(), args.out.resolve(), name, args.scale)
    except Exception as e:
        print(f"Ошибка конвертации: {e}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
