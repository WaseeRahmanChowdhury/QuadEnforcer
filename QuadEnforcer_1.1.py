bl_info = {
    "name": "QuadEnforcer",
    "author": "Wasee Rahman Chowdhury",
    "version": (1, 0, 0),
    "blender": (5, 1, 0),
    "location": "View3D > Sidebar > QuadEnforcer",
    "description": "Converts mesh to clean quads with subdivision and smooth shading",
    "category": "Mesh",
}

import math
import time
import bpy
import bmesh
import bmesh.utils
from mathutils import Matrix, Vector, Quaternion, Euler
import mathutils
import numpy as np
from collections import deque
from bpy.props import FloatProperty, IntProperty, BoolProperty, EnumProperty


def triangulation(bm, face):
    res = bmesh.ops.triangulate(bm, faces=face)
    face_map = res['face_map']
    fmap = {}
    for k, v in face_map.items():
        if v not in fmap:
            fmap[v] = []
        fmap[v].append(k)
    return fmap


def is_inner(tris, p1):
    fs = p1.edge.link_faces
    return len(fs) == 2 and all(f in tris for f in fs)


def get_near_sum(p1):
    d1 = p1.calc_angle()
    p2 = p1.link_loop_radial_next.link_loop_next
    d2 = p2.calc_angle()
    return d1 + d2


def get_convex(tris, f1):
    for p1 in f1.loops:
        if is_inner(tris, p1):
            bod = is_border(p1)
            if bod == False:
                d1 = get_near_sum(p1)
                d2 = get_near_sum(p1.link_loop_radial_next)
                if d1 <= math.pi + 0.001 and d2 <= math.pi + 0.001:
                    return p1
    return None


def delete_face(bm, f1):
    for e1 in f1.edges:
        if len(e1.link_faces) == 1:
            bm.edges.remove(e1)
    if f1.is_valid:
        bm.faces.remove(f1)


def get_loop(p1, p2):
    ps = []
    p = p1
    while True:
        ps.append(p)
        p = p.link_loop_next
        if p == p2:
            ps.append(p)
            break
    return ps


def merge_faces_lv2(bm, tris, p1):
    p2 = p1.link_loop_next
    ps1 = get_loop(p2, p1)
    vs1 = [p.vert for p in ps1]
    p3 = p1.link_loop_radial_next
    p4 = p3.link_loop_next
    ps2 = get_loop(p4, p3)
    vs2 = [p.vert for p in ps2]
    vs2 = vs2[1:-1]
    vs3 = vs1 + vs2
    f3 = bm.faces.new(vs3)
    tris.remove(p1.face)
    tris.remove(p3.face)
    delete_face(bm, p1.face)
    delete_face(bm, p3.face)
    tris.append(f3)
    f3.normal_update()


def merging_lv1(bm, tris):
    ds = []
    for tri in tris:
        dc = 0
        for p in tri.loops:
            for f1 in p.edge.link_faces:
                if f1 in tris:
                    dc += 1
        ds.append((dc, tri, tri.calc_area()))
    ds = sorted(ds, key=lambda x: (x[0], x[2]))
    merged = False
    for _, f1, _ in ds:
        if f1.is_valid == False:
            continue
        convex = get_convex(tris, f1)
        if convex != None:
            merge_faces_lv2(bm, tris, convex)
            merged = True
    return merged


def merging(bm, tris):
    while True:
        res = merging_lv1(bm, tris)
        if not res:
            break


def deselect(tris):
    for tri in tris:
        tri.select = False


def min_loop_distance(i, k, total):
    if i == k:
        return 0
    if i < k:
        return min(k - i, i + total - k)
    else:
        return min(i - k, k + total - i)


def get_avg_length(f1):
    elen = []
    for p in f1.loops:
        elen.append(p.edge.calc_length())
    return sum(elen) / len(elen)


def connect_cut(bm, k1, k2, ct):
    if ct <= 1:
        e1 = bm.edges.new([k1, k2])
        fs = list(e1.link_faces)
        return fs
    m1 = k2.co - k1.co
    m1 = m1 / ct
    v1 = k1
    vs = []
    for i in range(1, ct):
        v2 = bm.verts.new(v1.co + m1)
        bm.edges.new([v1, v2])
        v1 = v2
        vs.append(v2)
    bm.edges.new([v1, k2])
    return vs


def find_sharp(f1):
    ms = []
    for p1 in f1.loops:
        d1 = p1.calc_angle()
        ms.append((d1, p1))
    _, pq = min(ms, key=lambda x: x[0])
    return pq


def is_outside(p1, p2, sn):
    m1 = p2.vert.co - p1.vert.co
    m2 = p1.link_loop_next.vert.co - p1.vert.co
    if m2.cross(m1).dot(sn) < 0:
        return True
    return False


def get_ans(f1):
    ans = []
    ppps = list(f1.loops)
    if len(ppps) <= 4:
        return []
    elif len(ppps) == 5:
        return []
    for i, p1 in enumerate(f1.loops):
        d = p1.calc_angle()
        res = d < math.pi / 1.5
        if res:
            ans.append(i)
    if len(ans) < 3:
        return None
    return ans


def get_balance(p1, p2):
    m1 = p2.vert.co - p1.vert.co
    a1 = p1.link_loop_next
    a2 = p1.link_loop_prev
    k1 = a1.vert.co - p1.vert.co
    k2 = a2.vert.co - p1.vert.co
    if m1.length == 0 or k1.length == 0 or k2.length == 0:
        return 999
    d1 = m1.angle(k1)
    d2 = m1.angle(k2)
    d90 = math.pi / 2
    d3 = abs(d90 - d1) + abs(d90 - d2)
    return d3


def get_balance2(p1, p2):
    d1 = get_balance(p1, p2)
    d2 = get_balance(p2, p1)
    return d1 + d2


def saw_cut(bm, f1, p1, p2, ct):
    vs = connect_cut(bm, p1.vert, p2.vert, ct)
    vs1 = get_loop(p2, p1)
    vs1 = [p.vert for p in vs1]
    f2 = bm.faces.new(vs1 + vs)
    f2.normal_update()
    vs2 = get_loop(p1, p2)
    vs2 = [p.vert for p in vs2]
    f3 = bm.faces.new(vs2 + list(reversed(vs)))
    f3.normal_update()
    bm.faces.remove(f1)
    return f2, f3


def reg_slice(bm, f1):
    avg = get_avg_length(f1)
    if len(f1.loops) == 4:
        return []
    ds = []
    for i, p1 in enumerate(f1.loops):
        for k, pk in enumerate(f1.loops):
            dis = min_loop_distance(i, k, len(f1.loops))
            if dis <= 2:
                continue
            d1 = get_balance2(p1, pk)
            ds.append((d1, p1, pk, i, k))
    if len(ds) == 0:
        return []
    _, p1, p2, i, k = min(ds, key=lambda x: x[0])
    dis = min_loop_distance(i, k, len(f1.loops))
    if dis - 1 == 2:
        e1 = cut_face(bm, p1.vert, p2.vert)
        fs = list(e1.link_faces)
        return fs
    m1 = p2.vert.co - p1.vert.co
    cts = math.floor(m1.length / avg) + 1
    if dis % 2 != cts % 2:
        cts += 1
    fs = saw_cut(bm, f1, p1, p2, cts)
    return fs


def cut_by_connect(bm, f0, p1, p2, seg_num):
    vs = connect_cut(bm, p1.vert, p2.vert, seg_num)
    fp1 = get_loop(p1, p2)
    fp1 = [p.vert for p in fp1]
    fp2 = get_loop(p2, p1)
    fp2 = [p.vert for p in fp2]
    f1 = bm.faces.new(fp1 + list(reversed(vs)))
    f2 = bm.faces.new(fp2 + vs)
    bm.faces.remove(f0)
    f1.normal_update()
    f2.normal_update()
    return [f1, f2]


def cut_by_vert(bm, f0, v1, v2, seg_num):
    pk1 = None
    pk2 = None
    for p1 in f0.loops:
        if p1.vert == v1:
            pk1 = p1
        if p1.vert == v2:
            pk2 = p1
    if pk1 is None or pk2 is None:
        return []
    return cut_by_connect(bm, f0, pk1, pk2, seg_num)


def wipe_side_reg(bm, f1, a1, a2, ppps):
    dis = min_loop_distance(a1, a2, len(ppps))
    p1 = ppps[a1]
    p2 = ppps[a2]
    if p2.link_loop_next == p1:
        p1, p2 = p2, p1
    p3 = p1.link_loop_prev
    p4 = p2.link_loop_next
    fs = cut_by_connect(bm, f1, p3, p4, dis)
    return fs


def check_six(bm, f1, ppps):
    flen = len(f1.loops)
    flats = []
    mm6 = math.pi / 6
    for i, p1 in enumerate(ppps):
        d = p1.calc_angle()
        p2 = ppps[(i + 3) % flen]
        d2 = p2.calc_angle()
        if abs(d - math.pi) < mm6:
            if abs(d2 - math.pi) < mm6:
                flats.append((p1, p2))
    if len(flats) > 0:
        ft = flats[0]
        fs = cut_by_connect(bm, f1, ft[0], ft[1], 0)
        return fs
    for i, p1 in enumerate(ppps):
        p2 = p1.link_loop_prev
        if is_border(p1) and is_border(p2):
            pk = ppps[(i + 3) % flen]
            if pk == p1 or pk == p2:
                continue
            if bm.edges.get((p1.vert, pk.vert)) is None:
                fs = cut_by_connect(bm, f1, p1, pk, 0)
                return fs
    ans = []
    for i, p1 in enumerate(ppps):
        d = p1.calc_angle()
        p2 = ppps[(i + 2) % flen]
        d2 = p2.calc_angle()
        p3 = ppps[(i + 4) % flen]
        d3 = p3.calc_angle()
        ans.append((d + d2 + d3, p1))
    _, p1 = max(ans, key=lambda x: x[0])
    p2 = p1.link_loop_next.link_loop_next
    p3 = p2.link_loop_next.link_loop_next
    cen = (p1.vert.co + p2.vert.co + p3.vert.co) / 3
    vcen = bm.verts.new(cen)
    f2 = bm.faces.new([p1.vert, p1.link_loop_next.vert, p2.vert, vcen])
    f3 = bm.faces.new([p2.vert, p2.link_loop_next.vert, p3.vert, vcen])
    f4 = bm.faces.new([p3.vert, p3.link_loop_next.vert, p1.vert, vcen])
    bm.faces.remove(f1)
    return [f2, f3, f4]


def line_segment_intersection_3d(p1, p2, p3, p4):
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)
    p4 = np.array(p4)
    epsilon = 1e-6
    v1 = p2 - p1
    v2 = p4 - p3
    v3 = p3 - p1
    cross_product = np.cross(v1, v2)
    mixed_product = np.dot(cross_product, v3)
    if abs(mixed_product) > epsilon:
        return None
    cross_product_sq_mag = np.dot(cross_product, cross_product)
    if cross_product_sq_mag < epsilon:
        return None
    t = np.dot(np.cross(v3, v2), cross_product) / cross_product_sq_mag
    u = np.dot(np.cross(v3, v1), cross_product) / cross_product_sq_mag
    if 0 < t < 1 and 0 < u < 1:
        return True
    else:
        return False


def plane_cross(p1, p2, ppps):
    for i in range(len(ppps)):
        i2 = (i + 1) % len(ppps)
        pk1 = ppps[i]
        pk2 = ppps[i2]
        if line_segment_intersection_3d(p1.vert.co, p2.vert.co, pk1.vert.co, pk2.vert.co):
            return True
    return False


def wipe_side_even(bm, f1, a1, a2, ppps, plen, pv):
    dis = min_loop_distance(a1, a2, len(ppps))
    p1 = ppps[a1]
    p2 = ppps[a2]
    if p2.link_loop_next == p1:
        p1, p2 = p2, p1
    p3 = p1.link_loop_prev
    p4 = p2.link_loop_next
    m1 = p1.vert.co - p2.vert.co
    m2 = p3.vert.co - p4.vert.co
    mm90 = math.pi / 2
    if m1.length > plen:
        if m2.length > m1.length * 1.2:
            d1 = p1.calc_angle()
            d2 = p2.calc_angle()
            if d1 < d2:
                pk = p3.link_loop_prev
                if pk != p4 and pk.calc_angle() > mm90:
                    fs = cut_by_connect(bm, f1, pk, p4, dis + 1)
                    return fs
            else:
                pk = p4.link_loop_next
                if p3 != pk and pk.calc_angle() > mm90:
                    fs = cut_by_connect(bm, f1, p3, pk, dis + 1)
                    return fs
        if m2.length < m1.length * 0.9 and dis > 1:
            d1 = p1.calc_angle()
            d2 = p2.calc_angle()
            if d1 < d2:
                pk = p3.link_loop_prev
                if pk != p4 and pk.calc_angle() > mm90:
                    fs = cut_by_connect(bm, f1, pk, p4, dis - 1)
                    return fs
            else:
                pk = p4.link_loop_next
                if p3 != pk and pk.calc_angle() > mm90:
                    fs = cut_by_connect(bm, f1, p3, pk, dis - 1)
                    return fs
    fs = cut_by_connect(bm, f1, p3, p4, dis)
    return fs


def cut_by_connect_vs(bm, f0, p1, p2, seg_num):
    vs = connect_cut(bm, p1.vert, p2.vert, seg_num)
    fp1 = get_loop(p1, p2)
    fp1 = [p.vert for p in fp1]
    fp2 = get_loop(p2, p1)
    fp2 = [p.vert for p in fp2]
    f1 = bm.faces.new(fp1 + list(reversed(vs)))
    f2 = bm.faces.new(fp2 + vs)
    bm.faces.remove(f0)
    f1.normal_update()
    f2.normal_update()
    return [f1, f2], vs


def wipe_side_center(bm, f1, a1, a2, ppps, plen, pv):
    dis = min_loop_distance(a1, a2, len(ppps))
    p1 = ppps[a1]
    p2 = ppps[a2]
    if p2.link_loop_next == p1:
        p1, p2 = p2, p1
    p3 = p1.link_loop_prev
    p4 = p2.link_loop_next
    m2 = p3.vert.co - p4.vert.co
    fct2 = m2.length / dis
    if fct2 <= plen * 1.3:
        fs = cut_by_connect(bm, f1, p3, p4, dis)
        return fs
    else:
        dis2 = dis + 2
        if dis - 1 > 0:
            fps = get_loop(p1, p2)
            fplen = len(fps)
            vk = fps[fplen // 2]
            vk = vk.vert
            fs, vs = cut_by_connect_vs(bm, f1, p3, p4, dis2)
            if len(vs) > 2:
                half = len(vs) // 2
                v1 = vs[half + 1]
                v2 = vs[half - 1]
                f2, f3 = fs
                fs2 = cut_by_vert(bm, f2, vk, v1, 0)
                if len(fs2) == 2:
                    f4, f5 = fs2
                    fs3 = cut_by_vert(bm, f5, vk, v2, 0)
                    if len(fs3) == 2:
                        f6, f7 = fs3
                        return [f3, f4, f6, f7]
            return fs
        else:
            if fct2 >= plen * 2.5:
                fs, vs = cut_by_connect_vs(bm, f1, p3, p4, dis2)
                return fs
            else:
                fs = cut_by_connect(bm, f1, p3, p4, dis)
                return fs


def cut_three(bm, ans, f1, ppps, even, plen, pv):
    dis = []
    for i in range(len(ans)):
        i2 = (i + 1) % len(ans)
        a1 = ans[i]
        a2 = ans[i2]
        d = min_loop_distance(a1, a2, len(ppps)) - 1
        m1 = ppps[a1].vert.co - ppps[a2].vert.co
        dis.append(((d, m1.length), i))
    if len(dis) == 0:
        return []
    _, id1 = min(dis, key=lambda x: x[0])
    a1 = ans[id1]
    a2 = ans[(id1 + 1) % len(ans)]
    if even:
        return wipe_side_even(bm, f1, a1, a2, ppps, plen, pv)
    else:
        return wipe_side_center(bm, f1, a1, a2, ppps, plen, pv)


def face_fold(bm, f1, even, plen, pv):
    if f1.is_valid == False:
        return [], []
    flen = len(f1.loops)
    ppps = list(f1.loops)
    if flen <= 4:
        return [], [f1]
    if flen == 6:
        fs = check_six(bm, f1, ppps)
        if fs != []:
            return [], fs
    ans = get_ans(f1)
    if ans is None:
        fs = reg_slice(bm, f1)
        return fs, []
    fs = cut_three(bm, ans, f1, ppps, even, plen, pv)
    return fs, []


def internal_fold(bm, tris, even, plen, pv):
    fs2 = []
    while len(tris) > 0:
        f1 = tris.pop(0)
        fs, fs4 = face_fold(bm, f1, even, plen, pv)
        fs2 += fs4
        tris += fs
        if pv[2]:
            break
    bm.normal_update()
    fs2 = list(set(fs2))
    fs2 = [f1 for f1 in fs2 if f1.is_valid]
    return fs2


def shifting_balance_pt(bm, trislist):
    while True:
        change = False
        for f1 in trislist:
            for p1 in f1.loops:
                p2 = p1.link_loop_prev
                if is_inner(trislist, p2) == False:
                    continue
                if len(p2.vert.link_edges) != 2:
                    continue
                p3 = p2.link_loop_prev
                m1 = p2.vert.co - p1.vert.co
                m2 = p3.vert.co - p2.vert.co
                k1 = m1.normalized()
                k2 = m2.normalized()
                if k1.dot(k2) > 0.999:
                    dif = abs(m1.length - m2.length)
                    mb = max(m1.length, m2.length)
                    if dif < mb * 0.03:
                        continue
                    p2.vert.co = (p1.vert.co + p3.vert.co) / 2
                    change = True
        if change == False:
            break


def basic_smooth(bm, tris, sharp):
    sm = []
    border = []
    fmap = {}
    for f1 in tris:
        if f1.normal.length == 0:
            continue
        cen = f1.calc_center_median()
        fmap[f1] = cen
        for p1 in f1.loops:
            if is_inner(tris, p1):
                fsk = [fk for fk in p1.vert.link_faces if fk in tris and fk.normal.length > 0]
                for i in range(len(fsk)):
                    f2 = fsk[i]
                    f3 = fsk[(i + 1) % len(fsk)]
                    deg = f2.normal.angle(f3.normal)
                    if deg > sharp:
                        border.append(p1.vert)
                    else:
                        sm.append(p1.vert)
            else:
                border.append(p1.vert)
    sm = list(set(sm) - set(border))
    for step in range(1):
        for v1 in sm:
            s1 = []
            for f1 in v1.link_faces:
                if f1 not in fmap:
                    continue
                s1.append(fmap[f1])
            if len(s1) == 0:
                continue
            v1.co = sum(s1, Vector()) / len(s1)


def split_edge(bm, fs, plen):
    es = []
    for f1 in fs:
        for p1 in f1.loops:
            es.append(p1.edge)
    es = list(set(es))
    for e1 in es:
        elen = e1.calc_length()
        ct = int(elen / plen)
        if ct > 0:
            bmesh.ops.bisect_edges(bm, edges=[e1], cuts=ct)


def get_loops(p1):
    ps = []
    p = p1
    for i in range(len(p1.face.loops)):
        ps.append(p)
        p = p.link_loop_next
        if p == p1:
            break
    return ps


def merge_ff(bm, p1):
    p2 = p1.link_loop_radial_next
    ps1 = get_loops(p1)
    ps2 = get_loops(p2)
    ps3 = ps1[1:] + ps2[1:]
    vs = [p.vert for p in ps3]
    bm.edges.remove(p1.edge)
    if p1.is_valid:
        delete_face(bm, p1.face)
        delete_face(bm, p2.face)
    f1 = bm.faces.new(vs)
    f1.normal_update()
    return f1


def cut_face(bm, v1, v2):
    res = bmesh.ops.connect_vert_pair(bm, verts=[v1, v2])
    es = res['edges']
    if len(es) == 0:
        return None
    e1 = es[0]
    fs = list(e1.link_faces)
    for f1 in fs:
        f1.normal_update()
    return e1


def cut_edge(bm, p1):
    res = bmesh.ops.bisect_edges(bm, edges=[p1.edge], cuts=1)
    vs = res['geom_split']
    vs = [v for v in vs if isinstance(v, bmesh.types.BMVert)]
    return vs[0]


def mid_point(p1):
    v1 = p1.vert.co
    v2 = p1.link_loop_next.vert.co
    mid = (v1 + v2) / 2
    return mid


def shift(bm, p1, p2):
    v1 = cut_edge(bm, p1)
    e1 = cut_face(bm, p2.vert, v1)
    fs = list(e1.link_faces)
    f1 = merge_ff(bm, p2)
    fs = fs + [f1]
    return list(set(fs))


def check_remote(origin, p1, p2, p3):
    m1 = p1 - origin
    m2 = p2 - origin
    m3 = p3 - origin
    d1 = m1.angle(m2)
    d2 = m2.angle(m3)
    if d1 + d2 < math.pi:
        return True
    else:
        return False


def shiftleft(bm, p1, p2):
    p3 = p1.link_loop_next
    v1 = cut_edge(bm, p2)
    e1 = cut_face(bm, p3.vert, v1)
    fs = list(e1.link_faces)
    f1 = merge_ff(bm, p1)
    fs = fs + [f1]
    return list(set(fs))


def join_faces(bm, p1):
    p2 = p1.link_loop_radial_next
    ps1 = get_loop(p1.link_loop_next, p1)
    ps2 = get_loop(p2.link_loop_next, p2)
    vs1 = [p.vert for p in ps1]
    vs2 = [p.vert for p in ps2]
    f3 = bm.faces.new(vs1 + vs2[1:-1])
    delete_face(bm, p1.face)
    delete_face(bm, p2.face)
    f3.normal_update()
    return f3


def bisect_face(bm, p1):
    e1 = p1.edge
    p2 = p1.link_loop_next
    mid = (p1.vert.co + p2.vert.co) / 2
    vm = bm.verts.new(mid)
    ps1 = get_loop(p2, p1)
    vs1 = [p.vert for p in ps1]
    f1 = bm.faces.new(vs1 + [vm])
    f1.normal_update()
    if len(e1.link_faces) > 1:
        p3 = p1.link_loop_radial_next
        p4 = p3.link_loop_next
        ps2 = get_loop(p4, p3)
        vs2 = [p.vert for p in ps2]
        f2 = bm.faces.new(vs2 + [vm])
        f2.normal_update()
        delete_face(bm, p3.face)
    delete_face(bm, p1.face)
    return vm


def is_border(p1):
    if p1.edge.is_boundary:
        return True
    if len(p1.edge.link_faces) == 1:
        return True
    if len(p1.edge.link_faces) == 2:
        f1, f2 = p1.edge.link_faces
        if f1.normal.length == 0 or f2.normal.length == 0:
            return None
        d1 = f1.normal.angle(f2.normal)
        if d1 > math.radians(1):
            return True
    return False


def convert_seq_adv_lv2(bm, tris):
    for f1 in tris:
        if f1.is_valid == False:
            continue
        for p1 in f1.loops:
            if is_border(p1):
                continue
            p2 = p1.link_loop_prev
            if is_inner(tris, p1) and is_inner(tris, p2):
                pmain = p1.link_loop_radial_next
                if is_border(pmain) == True:
                    continue
                mid = mid_point(pmain.link_loop_next)
                m1 = p1.vert.co - pmain.vert.co
                if is_border(pmain.link_loop_next) == True:
                    if p1.link_loop_next.calc_angle() + (mid - pmain.vert.co).angle(m1) < math.pi:
                        if pmain.link_loop_next.calc_angle() + p1.calc_angle() < math.pi:
                            v1 = cut_edge(bm, pmain.link_loop_next)
                            e1 = cut_face(bm, pmain.vert, v1)
                            fs = list(e1.link_faces)
                            f3 = join_faces(bm, pmain)
                            tris.append(f3)
                            tris += [f for f in fs if f.is_valid]
                            tris.remove(f1)
                            return True
    return False


def convert_seq_adv_left_lv2(bm, tris):
    for f1 in tris:
        if f1.is_valid == False:
            continue
        for p1 in f1.loops:
            if is_border(p1):
                continue
            p2 = p1.link_loop_prev
            if is_inner(tris, p1) and is_inner(tris, p2):
                pmain = p1.link_loop_prev
                if is_border(pmain) == True:
                    continue
                div_e = pmain.link_loop_radial_next.link_loop_prev
                mid = mid_point(div_e)
                m1 = pmain.link_loop_next.vert.co - pmain.vert.co
                if is_border(div_e) == True:
                    if pmain.calc_angle() + (mid - pmain.vert.co).angle(m1) < math.pi:
                        if pmain.link_loop_radial_next.calc_angle() + p1.calc_angle() < math.pi:
                            v1 = cut_edge(bm, div_e)
                            e1 = cut_face(bm, pmain.vert, v1)
                            fs = list(e1.link_faces)
                            f3 = join_faces(bm, pmain)
                            tris.append(f3)
                            tris += [f for f in fs if f.is_valid]
                            tris.remove(f1)
                            return True
    return False


def convert_seq_adv(bm, tris):
    while True:
        res2 = convert_seq_adv_left_lv2(bm, tris)
        if res2 == False:
            break


def find_odd(adj_matrix):
    degrees = np.sum(adj_matrix, axis=0)
    return [i for i, deg in enumerate(degrees) if deg % 2 == 1]


def b_find(adj_matrix, start, odd_set):
    n = adj_matrix.shape[0]
    visited = [False] * n
    parent = [-1] * n
    queue = deque([start])
    visited[start] = True
    while queue:
        u = queue.popleft()
        if (u != start or len(odd_set) == 1) and (u in odd_set):
            path = []
            cur = u
            while cur != -1:
                path.append(cur)
                cur = parent[cur]
            path.reverse()
            return path
        for v in range(n):
            if adj_matrix[u, v] > 0 and not visited[v]:
                visited[v] = True
                parent[v] = u
                queue.append(v)
    return None


def make_all_degrees_e_fast(adj_matrix):
    adj_matrix = adj_matrix.copy()
    odd_nodes = find_odd(adj_matrix)
    while len(odd_nodes) >= 2:
        A = odd_nodes[0]
        path = b_find(adj_matrix, A, set(odd_nodes))
        if path is None:
            break
        for i in range(len(path) - 1):
            u = path[i]
            v = path[i + 1]
            adj_matrix[u, v] += 1
            adj_matrix[v, u] += 1
        odd_nodes = find_odd(adj_matrix)
    return adj_matrix


def fix_balance(bm, tris, plen):
    tlen = len(tris)
    matrix = np.zeros((tlen, tlen), dtype=int)
    fmap = {}
    for i, f1 in enumerate(tris):
        fmap[f1] = i
    es = []
    for f1 in tris:
        for e1 in f1.edges:
            es.append(e1)
    es = list(set(es))
    emap = np.zeros((tlen, tlen), dtype=object)
    sfmap = []
    for e1 in es:
        sfc = False
        if len(e1.link_faces) == 1:
            sfc = True
            sfmap.append(e1)
        elif len(e1.link_faces) == 0:
            continue
        else:
            fs1 = [f for f in e1.link_faces if f in tris]
            if len(fs1) == 1:
                sfc = True
                sfmap.append(e1)
            elif len(fs1) == 2:
                pass
            else:
                continue
        if sfc == False:
            f1, f2 = e1.link_faces
            i, i2 = fmap[f1], fmap[f2]
            matrix[i, i2] += 1
            matrix[i2, i] += 1
            emap[i, i2] = e1
            emap[i2, i] = e1
    matrix2 = make_all_degrees_e_fast(matrix)
    if matrix2 is None:
        return
    diff = matrix2 - matrix
    rows, cols = diff.shape
    indices = np.triu_indices(rows, k=1)
    diff[indices] = 0
    for i in range(diff.shape[0]):
        for j in range(diff.shape[1]):
            if diff[i, j] > 0:
                e1 = emap[i, j]
                ct = diff[i, j] % 2
                bmesh.ops.bisect_edges(bm, edges=[e1], cuts=ct)
    for e1 in sfmap:
        if len(e1.link_faces) == 1:
            f1 = e1.link_faces[0]
            flen = len(f1.loops)
            if flen % 2 == 1:
                bmesh.ops.bisect_edges(bm, edges=[e1], cuts=1)


def main_process(bm, plen, sharp, even, removed):
    pv = [1, 1, 0]
    sel = [f1 for f1 in bm.faces if f1.select]
    if pv[0]:
        deselect(sel)
        if removed:
            vss = []
            for f1 in sel:
                for p1 in f1.loops:
                    vss.append(p1.vert)
            vss = list(set(vss))
            bmesh.ops.remove_doubles(bm, verts=vss, dist=0.001)
        ng = [f1 for f1 in sel if f1.is_valid and len(f1.loops) > 4]
        qd = [f1 for f1 in sel if f1.is_valid and len(f1.loops) <= 4]
        tris_map = triangulation(bm, ng)
        tris = []
        for f1 in tris_map:
            ts = tris_map[f1]
            tris += ts
        merging(bm, tris)
        tris = [f1 for f1 in tris if f1.is_valid]
        convert_seq_adv(bm, tris)
        tris = [f1 for f1 in tris if f1.is_valid]
        trislist = tris + qd
        split_edge(bm, trislist, plen)
        fix_balance(bm, trislist, plen)
        shifting_balance_pt(bm, trislist)
    else:
        trislist = sel
    if pv[1]:
        trislist = internal_fold(bm, trislist, even, plen, pv)
        basic_smooth(bm, trislist, math.radians(sharp))
        for f1 in trislist:
            f1.normal_update()


def mark_all_creases(mesh):
    bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(mesh)
    cl = bm.edges.layers.crease.verify()
    for e in bm.edges:
        e[cl] = 1.0
    bmesh.update_edit_mesh(mesh)
    bpy.ops.object.mode_set(mode='OBJECT')


def remove_all_creases(mesh):
    bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(mesh)
    cl = bm.edges.layers.crease.verify()
    for e in bm.edges:
        e[cl] = 0.0
    bmesh.update_edit_mesh(mesh)
    bpy.ops.object.mode_set(mode='OBJECT')


def apply_subd(obj):
    mod = obj.modifiers.new(name="QuadEnforcer_SubD", type='SUBSURF')
    mod.subdivision_type = 'CATMULL_CLARK'
    mod.levels = 1
    mod.render_levels = 1
    bpy.ops.object.modifier_apply(modifier=mod.name)


def apply_smooth(obj):
    bpy.ops.object.shade_smooth()
    obj.data.use_auto_smooth = True
    obj.data.auto_smooth_angle = math.radians(30)


class QUADENFORCER_OT_convert(bpy.types.Operator):
    bl_idname = 'object.quad_enforcer_convert'
    bl_label = 'Convert to Quads'
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == 'MESH' and context.mode == 'OBJECT'

    def execute(self, context):
        obj = context.active_object

        if obj is None or obj.type != 'MESH':
            self.report({'ERROR'}, "No active mesh object selected")
            return {'CANCELLED'}

        start = time.time()

        try:
            props = context.scene.quad_enforcer_props

            bpy.ops.object.mode_set(mode='EDIT')
            bm = bmesh.from_edit_mesh(obj.data)
            for face in bm.faces:
                face.select = True
            bmesh.update_edit_mesh(obj.data)

            bm = bmesh.from_edit_mesh(obj.data)
            main_process(bm, props.quad_size, props.sharp_angle, False, True)
            bmesh.update_edit_mesh(obj.data)

            bpy.ops.object.mode_set(mode='OBJECT')
            apply_smooth(obj)

            mark_all_creases(obj.data)
            apply_subd(obj)
            remove_all_creases(obj.data)

            end = time.time()
            elapsed = round(end - start, 2)
            print(f"QuadEnforcer finished in {elapsed} seconds")
            self.report({'INFO'}, f"QuadEnforcer done in {elapsed}s")
            return {'FINISHED'}

        except Exception as e:
            import traceback
            print(traceback.format_exc())
            self.report({'ERROR'}, f"QuadEnforcer failed: {str(e)}")
            try:
                if context.active_object and context.active_object.mode != 'OBJECT':
                    bpy.ops.object.mode_set(mode='OBJECT')
            except:
                pass
            return {'CANCELLED'}


class QuadEnforcerProps(bpy.types.PropertyGroup):
    quad_size: bpy.props.FloatProperty(
        name="Quad Size",
        description="Target size of the quads",
        default=0.47,
        min=0.25,
        max=1.00,
        step=0.05
    )
    sharp_angle: bpy.props.FloatProperty(
        name="Sharp Angle",
        description="Sharp angle for detecting boundaries",
        default=0,
        min=0,
        step=5
    )


class QUADENFORCER_PT_panel(bpy.types.Panel):
    bl_label = "QuadEnforcer"
    bl_idname = "QUADENFORCER_PT_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "QuadEnforcer"
    bl_context = "objectmode"

    def draw(self, context):
        layout = self.layout
        props = context.scene.quad_enforcer_props
        layout.prop(props, "quad_size")
        layout.prop(props, "sharp_angle")
        layout.operator("object.quad_enforcer_convert", icon='MESH_DATA')


classes = [
    QuadEnforcerProps,
    QUADENFORCER_OT_convert,
    QUADENFORCER_PT_panel,
]


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.quad_enforcer_props = bpy.props.PointerProperty(type=QuadEnforcerProps)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.quad_enforcer_props


if __name__ == "__main__":
    register()
