# -*- coding: utf-8 -*-
"""
Updated on Thu Jun 25 22:10:00 2026 (TUNNEL WIDEN (operator-requested, ~line 2016): after person_bbox_3d is finalized (post inventory-union override / fallback) and BEFORE the acceptance zone reads it, the tunnel is expanded by a fixed margin so the recomputed person cluster captures more of the body -- X widened +/-8cm (min-8, max+8 = +16cm span), Y +40cm deeper (far side), Z +20cm up (toward the crown). The acceptance zone (6-signal) derives its X/Z extent from person_bbox_3d, so the cluster is recomputed against the widened tunnel automatically. Single additive block, wrapped in try/except, [TUNNEL] widened margin log; no new module/function/CLI. Pairs with the get_or_create_skeleton anchor fix in temporal_consistency.py. NOTE risk: on the ~6 background-contaminated frames the +20 Z could approach the +62cm wall projection; watch the cluster top there. Jerusalem wall-clock is an estimate.)
Updated on Thu Jun 25 07:30:00 2026 (TURN-TUNNEL expand-only override (~line 1948): frames 42-50 the tunnel was too narrow (yellow segmentation contour spilled outside the green tunnel box on the XZ panel) and its top was pinned to a short Z, which clipped/collapsed the shell and dropped its height. Root: the inventory-bbox override REPLACED the current 2D-pose tunnel X/Z whenever the inventory X-span was <=70cm, but the inventory bbox lags the turning person (stale X ~[-40,-2] vs pose X ~[-66,-17] on f46; inventory Z-top -2 vs pose Z-top +18). The override now UNIONS the inventory bbox with the current 2D-pose bbox (expand-only, never shrinks below the person's current extent) for both X and Z, capped: X-span <=70cm rejects a pose-keypoint glitch / inflated cluster, and the pose Z-top is capped at floor+195cm so the +62cm wall/ceiling projection is excluded from the tunnel. Same log tag family ([TUNNEL] ...); no new module/function/CLI. NOTE: on frames where the CoP itself lacks the head (silhouette top +2..+6cm, ~half the walking frames) the shell stays short - this fix removes the ARTIFICIAL clip, not the upstream MiDaS half-shell dropout. Time is an estimate.)
Updated on Wed Jun 24 17:00:00 2026 (TEMPORAL HEIGHT ANCHOR feed (~line 3096): the person-height lock submission was gated `and state_bank.get_person_height() is None`, so it locked ONCE on frame 1 from that frame's cluster z-span (153.8cm, a short MiDaS half-shell capture). Removed the once-only gate so the frame's cluster z-span is submitted to state_bank.lock_person_height EVERY frame; the lock now accumulates the running MAX over a settling window (see temporal_consistency.py) and converges the stature to the tallest clean capture (~165-168cm) instead of freezing on a short first frame. Only the gate changed; the z-span computation, >50cm sanity, and CLUSTER_HEAD_CORRECTION_CM are untouched. No new module/function/CLI. Time is an estimate.)
Updated on Tue Jun 23 22:30:00 2026 (DISABLE FEATURE A capsule-fill: added _FEATURE_A_CAPSULE_FILL=False guard on the fill block (~line 3257). The capsule fill turned the person into a solid monolith that did not match the cluster shell -- the mannequin capsules and the cluster are different skeletons, so filling the cluster from the capsules was geometrically wrong. Person reverts to its native thin shell. Reversible: flip the flag to True. Searching for a different densification approach.)
Updated on Tue Jun 23 21:35:00 2026 (FIX FEATURE-A regression: parse_voxel_key (line ~454) and the inline parse_key inside sort_voxel_data_by_yxz (line ~4693) both split voxel keys on "," and int() each part. FEATURE A merges capsule-fill voxels keyed "cf_<ix>_<iy>_<iz>" (underscore-separated, cf_ prefix) into person voxel_data, so the YXZ-sort step raised ValueError: invalid literal for int() with base 10: 'cf_-26_127_-37' on EVERY frame -> "Processed 0 frames". Both parsers now detect the cf_ prefix and split on "_". The cf_ prefix is only a key namespace; capsule-fill voxels are still identified by their synthetic_source='capsule_fill' metadata, so this is sort-ordering only and changes no geometry.)
Updated on Tue Jun 23 16:40:00 2026 (FEATURE A capsule-guided cluster fill (2nd pass, in-frame): right after ICCS/segment capsules are established for the person (~line 3239, before POST-ICCS facing), call mmpose.keypoint_stabilizer.fill_voxels_in_capsules(voxel_size) and merge the returned capsule-volume voxels into current_clusters[person]['voxel_data'] (live ref -> render picks it up). Densifies the thin 2.5D person shell into a complete body matching the fitted mannequin. Fill voxels are centroid-only + tagged synthetic_source='capsule_fill' so they render (reconstruct STEP 4c) but never feed the next frame's shell-fit. Logs [CAPSULE-FILL] Frame N: +K capsule voxels (shell A -> B). Person-only (capsules come from the person skeleton); prereq was the SHELL-FIT centroid fallback that made the mannequin correct on f44-51.)
Updated on Tue Jun 23 15:08:59 2026 (FIX-E density-robust _body_depth (frames 38-53 incomplete person): the tunnel measured _body_depth from "significant" Y-walls only -- generate_y_wall_index (grid.py) drops every <5-pt voxel and keeps only >80-100-pt walls, so when the cloud thins to ~1/3 density on side/oblique poses the medium-density back+side returns never form a significant wall and the front-to-back span floors at the 20cm clamp. That collapsed person_y_range -> disabled the split veto and flagged the acceptance zone [WARN]NARROW. _body_depth is now ALSO measured straight from the raw CoP points in the person's XZ footprint (P10-P90 of Y, chair excluded by the X-window), taking the larger of the two so it only ever EXPANDS -- restoring an honest depth for the acceptance-zone Y-extent and the veto. Pairs with the clustering.py veto gate-by-height change.)
Updated on Fri Jun 19 17:40:00 2026 (mojibake purge: all expression/log string literals normalized to pure ASCII -- deg / -> / cm3 / [OK] / [FAIL]; docstrings & comments left as-is)
run_clustering.py

Created on Thu Apr 24 13:45:11 2025
Updated on Tue Jun 02 07:39:00 2026 (frame_XXX_results.json readable layout: indented dicts, numeric arrays one line)
Updated on Sat Jun 06 22:02:23 2026 (MP33 wrist/arm REPAIR at the source: full-frame MP33 can put a wrist on a nearby object and that defected landmark drags the fitted arm there)
Updated on Sun Jun 07 05:12:35 2026 (MP33 wrist REPAIR now SILHOUETTE-based: the upstream MP33 detector runs with enable_segmentation=True, so flaw-detect uses the person mask (arm landmark on background = flaw) and the rescan crops to the mask's bbox - the upstream equivalent of the yellow contour, robust even when the body box is fooled. Body-box fallback kept; repaired landmarks replace defected in _mp33_wrists -> fitted skeleton / 3D-pose / IoU panels. Look for [MP33-WRIST] ... (src=crop))
Updated on Sun Jun 07 19:03:41 2026 (YOLO label channel wired: Phase 1 runs the shared yolov8n-seg via get_yolo_seg_result(frame_img, frame_num) right after detect_2d_pose; after inventory reconcile, cached detections feed state_bank.apply_yolo_labels(frame_num, ...) for IoU class binding/vote/lock; no CLI flag; both calls no-op if ultralytics absent)
Updated on Sat Jun 13 17:48:25 2026 (ISSUE-1 BONE LOCK follow-up: extract_3d_from_cop now receives person_uuid=(person_actual_uuid or person_cluster_uuid) - the canonical, relabel-stable person id - so the per-UUID stature lock in mmpose_integration does not re-lock on a label shuffle.)
Updated on Sat Jun 13 10:18:21 2026 (ISSUE-1 DEFECT-2 FIX — body_yaw is no longer corrupted by chord/POSE-DB on the bad frames. The single value that reaches the skeleton's ICCS frame is facing_info['body_yaw_deg'] -> _sf_body_yaw -> fit_to_cluster_shell(body_yaw_deg=). A per-UUID TEMPORAL YAW LOCK is applied at that chokepoint (_lock_body_yaw + state_bank.last_good_body_yaw): the candidate facing yaw is accepted only when it is RELIABLE (facing_info source not in {'default','pose_db_override'}, confidence >= _YAW_CONF_MIN, value not None) and then VELOCITY-CLAMPED to +/-_YAW_MAX_RATE deg/frame so a single bad frame can't snap the frame and a real turn still tracks over a few frames; an UNRELIABLE candidate (the POSE-DB hardcoded 90/270 lateral override of frames 36-48, or the chord-empty 'default' 0.0 of frames 65-66/73-109) HOLDS the last good yaw instead of moving the ICCS frame. chord/trio/POSE-DB are thereby demoted to BOOTSTRAP-ONLY: they seed the lock on the first sighting of a UUID and after a forced reseat (uuid in state_bank.yaw_force_bootstrap, e.g. set by occlusion recovery); every other frame is geometry+temporal. The three POSE-DB-LATERAL override sites and the chord/2d-pixel yaw computations are LEFT UNTOUCHED (they still drive facing label / fitting-path routing); the lock just refuses to let their unreliable YAW move the ICCS frame. No existing names changed; self-contained in run_clustering.py. NOTE: Defect 1 (world-space smoothing in mmpose Steps 10/10b -> angle-space canonical state) is the separate staged follow-up, not in this change.)
Updated on Sat Jun 13 18:10:34 2026 (ISSUE-3 anti-merge, rigid-object primitives (chair): the inlined RigidPrimitive/fit_rigid_primitive/claim_rigid_primitives (added to run_clustering.py itself, NO new module, NO CLI change) fit an oriented bounding box to a YOLO-class-locked rigid object (object_inventory class_label in {chair,table,...}) ONCE from its voxels -> state_bank.rigid_primitives[uuid] ([RIGID-FIT] log). The fit hook (after apply_yolo_labels) is READ-ONLY and changes no clustering behaviour. A gated claim step after the per-frame point load removes those voxels from the cloud before clustering so a touching chair can't fuse into the person blob; it is OFF by default (state_bank.rigid_claim_enabled=False) and logs [RIGID-CLAIM] when on. Enabling needs a smoke-test (claiming voxels can interact with the historical-lock / tunnel-recovery path for the claimed object). Module is no-op-safe without trimesh (pure-numpy PCA OBB fallback). Person-on-chair CONTACT is the known residual (touching surfaces aren't geometrically separable; needs the YOLO-instance/per-voxel temporal layer). No existing names changed.)
Updated on Sat Jun 13 20:28:00 2026 (YOLO inventory persistence: cluster_info gains yolo_class_label/yolo_class_conf/yolo_class_locked from object_inventory per-frame; results_data gains top-level rigid_primitives dict (uuid->OBB dict) so visualization.py can render locked object OBBs on the middle panel. Read-only from state_bank; no clustering behaviour changed.)
Updated on Sun Jun 14 19:16:08 2026 (CONTROL FRAMES: added _CONTROL_KP_WORLD dict with frame 1 ground-truth world-21 joints (from frame_001_results_s.json); frames 45/80 pending. Override block inserted after fit_frame() replaces fitted_keypoints_world_21 and recalibrates bone_lengths on _sf_skeleton for all downstream frames. No new modules; no existing names changed.)
Updated on Tue Jun 16 20:24:00 2026 (FILL-TO-COP 3D proximity guard (frame-11/13 stray-voxel fix): FILL-TO-COP walked every raw CoP point in the 112cm-deep acceptance zone and injected any that plugged a 2D XZ/YZ silhouette gap, with NO 3D check — so a floor/shadow voxel ~60-100cm in front of the body (sharing a projection column) got pulled into the person cluster. Fill is now a body-anchored flood: a candidate is injected only if its voxel is within _FILL_MAX_GAP=4 voxels (~8cm) of already-claimed material (_cl_xyz, seeded from the body and grown each iteration). Real silhouette holes are adjacent and still fill; a disconnected far speck never gets a foothold and is counted in the new far_rejected=N log field. No existing names changed.)
Updated on Wed Jun 17 21:46:00 2026 (VOXEL_DATA PRESERVE in STEP 4 (frames 68-75 empty chair primitive — ACTUAL root cause): inject_historical_lock_clusters recovers the occluded chair correctly and ATTACHES a full voxel_data dict (e.g. 3497 voxels, 1561 with y_plane metadata) to the cluster. But STEP 4 here REBUILT voxel_data from scratch off the voxel_grid and kept a voxel ONLY when it had y_plane_1/y_plane_2 pattern data — a forced-rescan chair's grid cells have no y_plane pattern, so every voxel was centroid-only and dropped → 0 voxels → renderer had no shell → empty OBB primitive (while the count still showed thousands of points). All the upstream recovery/footprint/STEP-7 fixes were generating voxels that this step silently discarded. Fix: when the from-scratch rebuild keeps nothing, fall back to the voxel_data clustering.py already attached to the cluster. Normal clusters (y_plane voxels present) are unaffected.)
Updated on Thu Jun 18 07:58:00 2026 (STEP 4 voxel_data fallback generalised from "rebuild kept ZERO" to "clustering.py's voxel_data is LARGER than the rebuild" (frames 61/66 sparse person): the y_plane-only rebuild also drops centroid-only voxels added by LIMB-RECLAIM / footprint regen, so a person whose clustering.py count was 994 voxels was saved with only 155 (the y_plane subset) — the empty-cluster bug in PARTIAL form. The "==0" guard missed it (155!=0). Now whenever clustering.py's carried voxel_data has MORE voxels than the lossy rebuild, use it. Restores the full shell for the person on the merge frames; normal clusters (rebuild == clustering.py count) unchanged.)
Updated on Wed Jun 17 20:22:00 2026 (GRAVITY-ALIGNED rigid primitive (chair OBB standing on its edge): fit_rigid_primitive used a FREE oriented box (trimesh bounding_box_oriented, else full 3-D PCA). On a partial / elongated / merge-contaminated chair shell the principal axes don't align with the world, so the OBB tilted ~73° about X (R=[[1,0,0],[0,-0.3,1],[0,-1,-0.3]]) and rendered on its corner. World Z is vertical here (floor at low Z), so the box is now built gravity-aligned: vertical axis pinned to world +Z, PCA only on the horizontal X-Y footprint for the other two axes. Floor objects always stand upright. trimesh free-OBB path removed.)                                     

@author: Chaim

Main script for running point cloud clustering with integrated contour alignment,
reduced output footprint, and streamlined visualization.

Usage:
    python run_clustering.py --path "outputs/flesh/dummy_flesh_cop_frame_*.txt" --frames_dir "data/clip_frames"

    # Use contour alignment:
    python run_clustering.py --path "outputs/flesh/dummy_flesh_cop_frame_*.txt" --frames_dir "data/clip_frames" --use_contour_alignment

    # Align specific clusters:
    python run_clustering.py --path "outputs/flesh/dummy_flesh_cop_frame_*.txt" --align_specific_clusters --align_cluster1 "cluster_id_1" --align_cluster2 "cluster_id_2"

PATCH 2026-03-18 (ro_nous repair plan):
- RC-1: Early anomaly reprocessing — under-clustering is now detected
         immediately after the clustering call, before any surface accumulation
         or mesh work.  The original check fired at the start of the NEXT frame,
         after Poisson + PyMeshFix + PyMeshLab + Trimesh had already run on the
         bad clusters (~20 s wasted per reprocess in the reference run).
- RC-2: FacingHistory required_consistent_frames raised 3 → 5 at all three
         instantiation sites (lines ~1381, ~2776, ~4682), consistent with the
         patch in temporal_consistency.py.
- RC-3: Ghost cluster skip in overlay reconstruction loop — after the utils.py
         patch, reconstruct_from_voxel_metadata() returns a tagged dict for
         centroid-fallback clusters.  The overlay loop now checks the tag and
         skips 125-pt ghost blobs before they are drawn in the final video.
"""

import os
import sys
import glob
import argparse
import logging
import time
import numpy as np
import joblib
import json
# import uuid
import uuid as uuid_module
import traceback
import image_alignment
import mmpose_integration
import cluster_coordinates
from cluster_coordinates import ClusterCoordinateSystem
# After your existing imports, add:
from pose_completion_integration import PoseCompletionPipeline, add_enhanced_arguments
from enhanced_grid import EnhancedOccupancyGrid
from intelligent_completion import IntelligentCompletion
from mmpose_integration import MMPoseIntegration
from mmpose_integration import KeypointStabilizer, get_cluster_info_for_person
import torch
ENHANCED_MODULES_AVAILABLE = True
from temporal_consistency import FrameBuffer, ClusterStateBank
from temporal_consistency import extract_frame_clusters
from temporal_consistency import Pose2DHistory, FacingHistory  # <-- ADD THIS LINE
from utils import PatternEnum, VoxelPattern, YPlaneData, PatternLookupTables, compute_voxel_normal, reconstruct_from_voxel_metadata
import utils as _utils_pose
from clustering import inject_historical_lock_clusters
from typing import Dict, Optional, Tuple, List, Any, Set
try:
    import mediapipe as _mp_pose_mod
    _MP33_AVAILABLE = True
except ImportError:
    _MP33_AVAILABLE = False

# Ensure the package is in Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our modules
from ro_nous_clustering import point_cloud, clustering, opencv_integration
from ro_nous_clustering import grid, visualization, utils
import pattern_classifier
from datetime import datetime


# =============================================================================
# CONTROL FRAMES: manually verified world-21 joint positions.
#
# For frames listed here the skeleton-fitting result is REPLACED with these
# hand-specified coordinates so that downstream analysis has a reliable
# ground-truth anchor.  bone_lengths are recomputed from the control joints
# and applied to _sf_skeleton before results_s.json is written, so every
# subsequent frame inherits calibrated proportions.
#
# Joint order — world CM, axes: X=left/right, Y=depth, Z=up(−=floor):
#   0=nose          1=left_eye      2=right_eye
#   3=left_ear      4=right_ear
#   5=left_shoulder 6=right_shoulder
#   7=left_elbow    8=right_elbow
#   9=left_wrist    10=right_wrist
#   11=left_hip     12=right_hip
#   13=left_knee    14=right_knee
#   15=left_ankle   16=right_ankle
#   17=head_center  (mid ears)      — synthetic
#   18=shoulder_center (mid shoulders) — synthetic
#   19=pelvis_center (iccs origin)  — synthetic
#   20=spine_mid    (mid 18+19)     — synthetic
#
# None = data not yet provided (frame skipped, normal fitting runs).
# Updated on Sun Jun 14 19:16:08 2026
# =============================================================================
_CONTROL_KP_WORLD = {
    1: [                          # frame_001_results_s.json — confirmed good fit
        # [-17.24, 284.15, -17.66],   # 0  nose
        # [-21.81, 267.88, -16.12],   # 1  left_eye
        # [-15.68, 267.33, -16.12],   # 2  right_eye
        # [-25.04, 266.32, -19.19],   # 3  left_ear
        # [-12.78, 265.21, -19.19],   # 4  right_ear
        # [-30.76, 267.30, -40.04],   # 5  left_shoulder
        # [-7.06,  264.23, -40.04],   # 6  right_shoulder
        # [-44.92, 252.51, -54.57],   # 7  left_elbow
        # [-2.06,  264.23, -65.72],   # 8  right_elbow
        # [-52.04, 261.01, -76.32],   # 9  left_wrist
        # [-2.06,  264.23, -82.25],   # 10 right_wrist
        # [-31.93, 266.95, -77.64],   # 11 left_hip
        # [-5.89,  264.58, -77.64],   # 12 right_hip
        # [-30.33, 284.54, -110.92],  # 13 left_knee
        # [-5.89,  264.58, -115.32],  # 14 right_knee
        # [-28.73, 299.14, -149.20],  # 15 left_ankle
        # [-5.89,  264.58, -153.00],  # 16 right_ankle
        # [-18.91, 265.77, -20.73],   # 17 head_center
        # [-18.91, 265.77, -33.04],   # 18 shoulder_center
        # [-18.91, 265.77, -77.64],   # 19 pelvis_center
        # [-18.91, 265.77, -55.34]   # 20 spine_mid
        [-17.6, 285.7, -20.3],
        [-22.1, 269.4, -18.7],
        [-16.0, 268.8, -18.7],
        [-25.4, 267.8, -21.8],
        [-13.1, 266.7, -21.8],
        [-33.7, 260.0, -40.5],
        [-3.1, 266.1, -40.5],
        [-46.7, 251.1, -63.3],
        [1.9, 266.1, -66.2],
        [-45.2, 257.9, -83.6],
        [1.9, 266.1, -82.7],
        [-31.4, 268.3, -80.6],
        [-5.9, 264.6, -77.6],
        [-30.2, 283.7, -115.0],
        [-5.9, 264.6, -115.3],
        [-28.7, 277.9, -152.2],
        [-5.9, 264.6, -153.0],
        [-19.2, 267.3, -23.4],
        [-19.2, 267.3, -35.7],
        [-18.9, 265.8, -77.6],
        [-19.1, 266.5, -56.7]
    ],
    45: [
    [-8.0, 629.9, -14.9],
    [-5.0, 649.5, -13.2],
    [-9.7, 627.1, -10.8],
    [-5.0, 649.5, -13.2],
    [-19.6, 622.2, -13.4],
    [-11.0, 638.5, -30.4],
    [-22.6, 631.0, -33.0],
    [-15.4, 624.0, -53.1],
    [-30.2, 632.6, -56.2],
    [-11.7, 623.0, -68.8],
    [-30.4, 630.4, -74.4],
    [-21.1, 637.1, -77.0],
    [-23.8, 623.6, -74.9],
    [-23.4, 643.0, -110.5],
    [-11.1, 646.8, -108.8],
    [-29.5, 643.0, -140.8],
    [-15.0, 639.0, -144.5],
    [-20.5, 640.9, -26.9],
    [-20.5, 640.4, -36.9],
    [-18.9, 640.8, -77.6],
    [-17.3, 640.8, -56.7]
    ],
    80: [
    [62.5, 381.4, -60.7],
    [62.5, 391.4, -57.7],
    [50.5, 386.0, -56.5],
    [70.7, 389.1, -57.7],
    [46.7, 387.0, -56.6],
    [60.9, 388.7, -67.0],
    [38.4, 389.4, -66.5],
    [71.1, 389.6, -90.8],
    [42.6, 384.1, -90.6],
    [65.2, 385.4, -113.4],
    [57.5, 385.7, -105.0],
    [41.1, 405.3, -94.2],
    [23.0, 408.6, -89.5],
    [65.0, 370.4, -109.8],
    [42.7, 372.7, -114.6],
    [63.6, 365.4, -135.1],
    [31.3, 363.5, -137.0],
    [45.4, 360.4, -63.5],
    [45.4, 360.9, -58.3],
    [35.5, 350.4, -83.9],
    [35.6, 360.9, -64.7]
    ],
}

# =============================================================================
# ISSUE-3 anti-merge: rigid-object (chair/table/...) occupancy primitives.
# Inlined here (no new module / no CLI change).  A YOLO-class-locked rigid object
# is fit ONCE to an oriented bounding box from its voxels and stored on
# state_bank.rigid_primitives[uuid]; its occupied volume can then be CLAIMED out
# of the per-frame cloud BEFORE clustering so a touching chair can't fuse into the
# person blob.  Pure-numpy PCA OBB (no hard dependency); trimesh used only if
# already present.  No-op until a primitive is locked AND state_bank.rigid_claim_enabled.
# =============================================================================
try:
    import trimesh as _trimesh_rigid
    _HAS_TRIMESH_RIGID = True
except Exception:
    _HAS_TRIMESH_RIGID = False


class RigidPrimitive:
    """Oriented bounding box occupancy volume for one rigid object.

    center  : (3,) world centre; extents : (3,) FULL side lengths along local axes;
    R       : (3,3) columns are local axes in world  ->  local = (world-center) @ R.
    """
    __slots__ = ('center', 'extents', 'R', 'kind', 'class_label', 'frame_locked', 'n_fit')

    def __init__(self, center, extents, R, kind='obb', class_label=None,
                 frame_locked=None, n_fit=0):
        self.center = np.asarray(center, dtype=float).reshape(3)
        self.extents = np.abs(np.asarray(extents, dtype=float).reshape(3))
        self.R = np.asarray(R, dtype=float).reshape(3, 3)
        self.kind = kind
        self.class_label = class_label
        self.frame_locked = frame_locked
        self.n_fit = int(n_fit)

    def contains(self, points, margin=0.0):
        """Boolean mask of points inside the OBB (margin cm grows/shrinks faces)."""
        p = np.asarray(points, dtype=float)
        if p.ndim == 1:
            p = p[None, :]
        local = (p - self.center) @ self.R
        half = np.maximum(self.extents / 2.0 + margin, 0.0)
        return np.all(np.abs(local) <= half, axis=1)

    def to_dict(self):
        return {'center': self.center.tolist(), 'extents': self.extents.tolist(),
                'R': self.R.tolist(), 'kind': self.kind,
                'class_label': self.class_label, 'frame_locked': self.frame_locked,
                'n_fit': self.n_fit}

    @classmethod
    def from_dict(cls, d):
        return cls(d['center'], d['extents'], d['R'], d.get('kind', 'obb'),
                   d.get('class_label'), d.get('frame_locked'), d.get('n_fit', 0))

    def __repr__(self):
        c, e = self.center, self.extents
        return (f"RigidPrimitive({self.class_label}, center=[{c[0]:.0f},{c[1]:.0f},"
                f"{c[2]:.0f}], extents=[{e[0]:.0f},{e[1]:.0f},{e[2]:.0f}]cm)")


def fit_rigid_primitive(points, class_label=None, min_points=12, frame_locked=None):
    """Fit a GRAVITY-ALIGNED RigidPrimitive (OBB) to an object's voxels.

    World Z is the vertical (height) axis in this pipeline — the floor sits at
    low Z — so a static FLOOR object (chair/table/...) stands UPRIGHT. We fix one
    OBB axis to world +Z and run PCA only on the horizontal X-Y footprint to
    orient the other two. A FREE OBB (trimesh bounding_box_oriented / full 3-D
    PCA, the previous behaviour) on a partial, elongated or merge-contaminated
    shell tilts the box ~73° onto its edge — physically wrong for a floor object
    and visible as a chair primitive standing on a corner. None if too few /
    degenerate."""
    P = np.asarray(points, dtype=float)
    if P.ndim != 2 or P.shape[1] != 3:
        return None
    P = P[~np.all(np.isclose(P, 0.0), axis=1)]
    P = P[np.isfinite(P).all(axis=1)]
    if len(P) < max(4, min_points):
        return None
    if np.count_nonzero(P.std(axis=0) > 1e-3) < 2:
        return None
    c = P.mean(axis=0)
    # Horizontal (X-Y) PCA for the footprint orientation; vertical axis pinned to
    # world +Z so the box is always upright.
    XY = P[:, :2] - c[:2]
    try:
        _w2, V2 = np.linalg.eigh(np.cov(XY.T))
    except Exception:
        V2 = np.eye(2)
    if not np.isfinite(V2).all() or V2.shape != (2, 2):
        V2 = np.eye(2)
    R = np.eye(3)
    R[0, 0] = V2[0, 0]; R[1, 0] = V2[1, 0]; R[2, 0] = 0.0
    R[0, 1] = V2[0, 1]; R[1, 1] = V2[1, 1]; R[2, 1] = 0.0
    R[0, 2] = 0.0;      R[1, 2] = 0.0;      R[2, 2] = 1.0
    if np.linalg.det(R) < 0:          # keep right-handed
        R[:, 1] = -R[:, 1]
    local = (P - c) @ R
    mn, mx = local.min(axis=0), local.max(axis=0)
    center = c + R @ ((mn + mx) / 2.0)
    return RigidPrimitive(center, (mx - mn), R, 'obb', class_label, frame_locked,
                          n_fit=len(P))


def claim_rigid_primitives(points, primitives, margin=-2.0,
                           protect_centroid=None, protect_radius=None):
    """Claim every locked primitive's voxels out of `points`.  Returns
    (kept_points, claimed_index_lists).  Default margin is a small SHRINK to stay
    off contact surfaces; protect_centroid/radius spares a person in contact."""
    bp = np.asarray(points, dtype=float)
    if bp.ndim == 1:
        bp = bp[None, :]
    keep = np.ones(len(bp), dtype=bool)
    claimed = []
    for prim in primitives:
        if prim is None:
            claimed.append(np.array([], dtype=int))
            continue
        inside = prim.contains(bp, margin=margin) & keep
        if protect_centroid is not None and protect_radius is not None and inside.any():
            d = np.linalg.norm(bp - np.asarray(protect_centroid, float).reshape(3), axis=1)
            inside = inside & (d > float(protect_radius))
        claimed.append(np.where(inside)[0])
        keep &= ~inside
    return bp[keep], claimed


# =============================================================================
# STEP 1: Add this helper function near the top of run_clustering.py
# (After imports, before process_single_frame)
# =============================================================================
# ADD these two helper functions near the top of run_clustering.py (after imports):


def _format_compact_json(data):
    """Module-level compact-JSON formatter.

    Hybrid layout:
      * dicts/objects: indented with newlines (readable structure)
      * lists of numbers: ONE line  ->  [33, 18, 61]
                                       [0.0, 1.0, 0.0]
      * short lists of small items: one line when total len < 100
      * otherwise: one item per indented line

    Mirrors `write_compact_json` defined inside process_single_frame
    but available at module level so save_temporal_data and
    save_temporal_data_with_voxel_metadata can use the same format.
    """
    def _fmt(val, indent=0):
        ind = "  " * indent
        if isinstance(val, dict):
            if not val:
                return "{}"
            lines = ["{"]
            cur = ""
            items = list(val.items())
            for i, (k, v) in enumerate(items):
                if isinstance(k, tuple):
                    k = str(k)
                key_str = f'"{k}": '
                val_str = _fmt(v, indent + 1)
                item = key_str + val_str
                if i < len(items) - 1:
                    item += ","
                if cur and len(cur) + len(item) + 1 > 100:
                    lines.append(ind + "  " + cur)
                    cur = item
                else:
                    cur = (cur + " " + item) if cur else item
            if cur:
                lines.append(ind + "  " + cur)
            lines.append(ind + "}")
            return "\n".join(lines)
        if isinstance(val, list):
            if not val:
                return "[]"
            # Pure numeric list -> one line
            if all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in val):
                parts = [
                    str(round(x, 1) if isinstance(x, float) else x)
                    for x in val
                ]
                return "[" + ", ".join(parts) + "]"
            # List of lists of numbers -> one line per inner list
            if all(isinstance(x, list) for x in val):
                inner = []
                for it in val:
                    if isinstance(it, list) and all(
                            isinstance(v, (int, float)) and not isinstance(v, bool)
                            for v in it):
                        parts = [
                            str(round(v, 1) if isinstance(v, float) else v)
                            for v in it
                        ]
                        inner.append("[" + ", ".join(parts) + "]")
                    else:
                        inner.append(_fmt(it, indent + 1))
                return "[" + ", ".join(inner) + "]"
            # Mixed list
            inner = [_fmt(it, indent + 1) for it in val]
            one = "[" + ", ".join(inner) + "]"
            if len(one) < 100:
                return one
            return ("[\n" + ind + "  "
                    + (",\n" + ind + "  ").join(inner)
                    + "\n" + ind + "]")
        if isinstance(val, bool):
            return "true" if val else "false"
        if isinstance(val, float):
            return str(round(val, 1))
        if isinstance(val, int):
            return str(val)
        if val is None:
            return "null"
        if isinstance(val, str):
            return json.dumps(val)
        # Fallback: try JSON serialization, else string
        try:
            return json.dumps(val)
        except (TypeError, ValueError):
            return json.dumps(str(val))

    return _fmt(data, 0)


def sort_voxel_data_by_yxz(voxel_data_dict):
    """
    Sort voxel_data dictionary keys by Y (depth), then X (horizontal), then Z (vertical).
    
    Args:
        voxel_data_dict: Dict with keys like "(41, 16, 9)" -> voxel info
        
    Returns:
        OrderedDict sorted by Y, X, Z
    """
    from collections import OrderedDict
    
    def parse_voxel_key(key_str):
        """Parse "(X, Y, Z)" string to tuple of ints.

        Also handles capsule-fill keys "cf_<ix>_<iy>_<iz>" (FEATURE A) which use
        an underscore separator and a cf_ namespace prefix.
        """
        clean = key_str.strip("() ").strip()
        if clean.startswith("cf_"):
            parts = clean[3:].split("_")
        else:
            parts = clean.split(",")
        x, y, z = int(parts[0].strip()), int(parts[1].strip()), int(parts[2].strip())
        return (x, y, z)
    
    # Sort by Y first, then X, then Z
    sorted_keys = sorted(
        voxel_data_dict.keys(),
        key=lambda k: (parse_voxel_key(k)[1],   # Y (depth) first
                       parse_voxel_key(k)[0],   # X (horizontal) second  
                       parse_voxel_key(k)[2])   # Z (vertical) third
    )
    
    # Create OrderedDict with sorted keys
    sorted_dict = OrderedDict()
    for key in sorted_keys:
        sorted_dict[key] = voxel_data_dict[key]
    
    return sorted_dict

def _calculate_body_orientation_safe(pose_3d: np.ndarray) -> Optional[Dict]:
    """
    Calculate body orientation from 3D pose keypoints.
    
    Uses shoulder line to determine rotation_z.
    NEVER fails - returns valid=False if cannot compute.
    
    Args:
        pose_3d: 17x3 array of 3D keypoints (COCO format)
        
    Returns:
        Dict with rotation_z (degrees), body_center, valid flag
    """
    try:
        if pose_3d is None or len(pose_3d) < 17:
            return {'valid': False, 'rotation_z': None, 'body_center': None}
        
        # COCO indices
        LEFT_SHOULDER = 5
        RIGHT_SHOULDER = 6
        LEFT_HIP = 11
        RIGHT_HIP = 12
        
        left_shoulder = pose_3d[LEFT_SHOULDER]
        right_shoulder = pose_3d[RIGHT_SHOULDER]
        left_hip = pose_3d[LEFT_HIP]
        right_hip = pose_3d[RIGHT_HIP]
        
        # Check for valid keypoints (non-zero)
        if (np.allclose(left_shoulder, 0) or np.allclose(right_shoulder, 0) or
            np.allclose(left_hip, 0) or np.allclose(right_hip, 0)):
            return {'valid': False, 'rotation_z': None, 'body_center': None}
        
        # Calculate centers
        shoulder_center = (left_shoulder + right_shoulder) / 2.0
        hip_center = (left_hip + right_hip) / 2.0
        body_center = (shoulder_center + hip_center) / 2.0
        
        # Shoulder vector (left to right)
        shoulder_vec = right_shoulder - left_shoulder
        
        # Rotation around Z-axis (vertical)
        # atan2(X_component, Y_component) gives angle from camera forward direction
        rotation_z = np.degrees(np.arctan2(shoulder_vec[0], shoulder_vec[1]))
        
        return {
            'valid': True,
            'rotation_z': float(rotation_z),
            'body_center': body_center.tolist(),
            'shoulder_width': float(np.linalg.norm(shoulder_vec))
        }
        
    except Exception as e:
        logger.debug(f"Body orientation calculation failed: {e}")
        return {'valid': False, 'rotation_z': None, 'body_center': None}


def _pixel_to_XZ_at_Y(u, v, y_wall, camera_position, camera_target,
                       field_of_view, image_size):
    """
    Inverse of opencv_integration.project_3d_to_2d.
    Given a pixel (u, v) and a Y-depth, return the (X, Z) world coordinates.

    Uses the EXACT same camera model as project_3d_to_2d:
        forward, right, up = normalized camera basis
        focal = (image_height / 2) / tan(fov / 2)
        u = focal * dot(vec, right) / dot(vec, forward) + cx
        v = cy - focal * dot(vec, up) / dot(vec, forward)

    Inverse:
        ray_dir = forward + right * (u - cx)/focal + up * (cy - v)/focal
        t = (y_wall - cam_pos[1]) / ray_dir[1]
        X = cam_pos[0] + t * ray_dir[0]
        Z = cam_pos[2] + t * ray_dir[2]
    """
    cam_pos = np.array(camera_position, dtype=float)
    cam_tgt = np.array(camera_target, dtype=float)

    forward = cam_tgt - cam_pos
    forward = forward / (np.linalg.norm(forward) + 1e-10)

    world_up = np.array([0., 0., 1.])
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-10:
        world_up = np.array([0., 1., 0.])
        right = np.cross(forward, world_up)
    right = right / (np.linalg.norm(right) + 1e-10)

    up = np.cross(right, forward)
    up = up / (np.linalg.norm(up) + 1e-10)

    w, h = float(image_size[0]), float(image_size[1])
    fov_rad = float(field_of_view) * np.pi / 180.0
    focal = (h / 2.0) / np.tan(fov_rad / 2.0)

    cx, cy = w / 2.0, h / 2.0
    dx = (float(u) - cx) / focal
    dy = (cy - float(v)) / focal

    ray_dir = forward + right * dx + up * dy

    denom = ray_dir[1]
    if abs(denom) < 1e-8:
        return None
    t = (float(y_wall) - cam_pos[1]) / denom
    if t <= 0:
        return None

    x = cam_pos[0] + t * ray_dir[0]
    z = cam_pos[2] + t * ray_dir[2]
    return (float(x), float(z))


def _bbox_tunnel_to_3d(bbox_2d, person_y_wall, body_depth,
                        camera_position, camera_target,
                        field_of_view, image_size):
    """
    Build the 3D tunnel from the green 2D bbox (MMPose) placed at the
    closest Y-wall.  Uses _pixel_to_XZ_at_Y — the exact inverse of
    opencv_integration.project_3d_to_2d.

    Front face = 4 bbox corners unprojected to XZ at (Y_wall - 1 step).
    Back face  = same XZ extents at Y_wall + body_depth.
    """
    if bbox_2d is None or person_y_wall is None:
        return None
    if body_depth is None or body_depth < 5.0:
        body_depth = 50.0

    try:
        x1, y1 = float(bbox_2d[0]), float(bbox_2d[1])
        x2, y2 = float(bbox_2d[2]), float(bbox_2d[3])

        y_front = float(person_y_wall)
        y_back = y_front + float(body_depth)

        # 4 corners of green bbox → XZ at Y_wall
        corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        hits = []
        for u, v in corners:
            pt = _pixel_to_XZ_at_Y(u, v, y_front,
                                    camera_position, camera_target,
                                    field_of_view, image_size)
            if pt is not None:
                hits.append(pt)

        if len(hits) < 3:
            logger.warning(f"[TUNNEL] Only {len(hits)} valid projections at Y={y_front:.0f}")
            return None

        xs = [h[0] for h in hits]
        zs = [h[1] for h in hits]

        result = {
            'min': [min(xs), y_front, min(zs)],
            'max': [max(xs), y_back, max(zs)],
            'y_near': y_front,
            'y_far': y_back,
        }

        logger.info(
            f"[OK] TUNNEL: green bbox [{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]px "
            f"@ Y-wall={y_front:.0f} -> "
            f"X=[{min(xs):.0f},{max(xs):.0f}] "
            f"Y=[{y_front:.0f},{y_back:.0f}] "
            f"Z=[{min(zs):.0f},{max(zs):.0f}]cm "
            f"(body_depth={body_depth:.0f}cm)")
        return result

    except Exception as e:
        logger.warning(f"[TUNNEL] Failed: {e}")
        return None


# =============================================================================

# CoP space bounds (in mm) - based on your system's physical constraints
COP_BOUNDS = {
    'min': [-100.0, 240.0, -170.0],  # Slightly larger than actual to ensure coverage
    'max': [100.0, 1000.0, 160.0]
}

logger = logging.getLogger(__name__)
keypoint_stabilizer = KeypointStabilizer()

try:
    from create_meshes import (
        SurfaceAccumulator, 
        VolumeTracker, 
        KeypointValidator,
        save_surface_states,
        OPEN3D_AVAILABLE
    )
    SURFACE_ACCUMULATION_AVAILABLE = True
    logger.info("Surface accumulation modules loaded")
except ImportError:
    SURFACE_ACCUMULATION_AVAILABLE = False
    logger.warning("Surface accumulation not available")

class UniversalJSONEncoder(json.JSONEncoder):
    """Universal encoder for all numpy types and enums."""
    def default(self, obj):
        # Handle numpy types
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        # Handle enums
        elif hasattr(obj, 'name'):
            return obj.name
        elif hasattr(obj, 'value'):
            return obj.value
        # Fallback
        return super().default(obj)

def safe_get_centroid(cluster_info):
    """Ensure centroid is always a 3D list of floats"""
    centroid = cluster_info.get('centroid', [0, 0, 0])
    
    # Handle different centroid types
    if isinstance(centroid, (int, float)):
        # Single value - this is the bug causing "float object is not iterable"
        logger.warning(f"Centroid is a single float ({centroid}), using default")
        return [0.0, 0.0, 0.0]
    
    if isinstance(centroid, np.ndarray):
        centroid = centroid.tolist()
    
    if not isinstance(centroid, (list, tuple)):
        logger.warning(f"Unknown centroid type {type(centroid)}, using default")
        return [0.0, 0.0, 0.0]
    
    # Ensure exactly 3 dimensions (remove 4th dimension if present)
    if len(centroid) > 3:
        centroid = centroid[:3]
    elif len(centroid) < 3:
        # Pad with zeros if less than 3
        centroid = list(centroid) + [0.0] * (3 - len(centroid))
    
    # Convert to float
    return [float(x) for x in centroid]

def contour_alignment_init():
    """Initialize contour alignment module dynamically if not already loaded."""
    try:
        from ro_nous_clustering import contour_alignment
        logger.info("Contour alignment module imported successfully")
        return True
    except ImportError:
        # Create the module dynamically if not present
        logger.warning("Contour alignment module not found - creating dynamically")
        try:
            # Create module code
            contour_alignment_code = """ """
            
            # Create the module
            module_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                      "ro_nous_clustering", "contour_alignment.py")
            
            with open(module_path, 'w') as f:
                f.write(contour_alignment_code)
            
            # Attempt to import the newly created module
            from ro_nous_clustering import contour_alignment
            logger.info("Created and imported contour alignment module")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create contour alignment module: {str(e)}")
            return False

def enhance_clustering_with_contour_alignment(points_a, labels_a, points_b, labels_b, output_dir):
    """
    Enhance clustering by aligning contours from two different point clouds.
    
    Args:
        points_a: First point cloud
        labels_a: Labels for first point cloud
        points_b: Second point cloud to align
        labels_b: Labels for second point cloud
        output_dir: Directory to save results
        
    Returns:
        Tuple of (transformed_points, transformed_labels, alignment_params)
    """
    try:
        from ro_nous_clustering import contour_alignment
        
        logger.info("Enhancing clustering with contour alignment...")
        os.makedirs(output_dir, exist_ok=True)
        
        # Perform alignment
        alignment_params = contour_alignment.align_contours(
            points_a, labels_a, points_b, labels_b, output_dir
        )
        
        # Apply alignment to adjust cluster positions
        if alignment_params and alignment_params['overlap'] > 30:  # Only apply if reasonable overlap
            # Apply the transformation to points_b
            scale = alignment_params['scale']
            dx = alignment_params['x']
            dy = alignment_params['y']
            
            # Scale and shift points
            points_b_transformed = points_b.copy()
            points_b_transformed[:, 0] = points_b[:, 0] * scale + dx
            points_b_transformed[:, 1] = points_b[:, 1] * scale + dy
            
            # Return transformed points and original labels
            return points_b_transformed, labels_b, alignment_params
        
        return points_b, labels_b, alignment_params
        
    except Exception as e:
        logger.error(f"Error enhancing clustering with contour alignment: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return points_b, labels_b, None

def align_cluster_pair(cluster1_id, cluster1_info, cluster2_id, cluster2_info, output_dir):
    """
    Align a specific pair of clusters.
    
    Args:
        cluster1_id: ID of first cluster
        cluster1_info: Dictionary with first cluster info
        cluster2_id: ID of second cluster
        cluster2_info: Dictionary with second cluster info
        output_dir: Output directory
        
    Returns:
        Updated cluster2_info with aligned points
    """
    logger.info(f"Aligning cluster pair {cluster1_id} and {cluster2_id}")
    
    # Extract points and labels
    points1 = cluster1_info['points']
    labels1 = np.full(len(points1), cluster1_info['original_label'])
    
    points2 = cluster2_info['points']
    labels2 = np.full(len(points2), cluster2_info['original_label'])
    
    # Create directory for this pair
    pair_dir = os.path.join(output_dir, f"pair_{cluster1_id}_{cluster2_id}")
    
    # Perform alignment
    transformed_points, transformed_labels, alignment_params = enhance_clustering_with_contour_alignment(
        points1, labels1, points2, labels2, pair_dir
    )
    
    # Update cluster info with transformed points
    updated_info = cluster2_info.copy()
    
    if alignment_params and alignment_params['overlap'] > 30:
        updated_info['points'] = transformed_points
        updated_info['centroid'] = np.mean(transformed_points, axis=0)
        updated_info['alignment'] = alignment_params
    
    return updated_info

def load_cluster_data(args, frame_num):
    """
    Load existing cluster data for refinement.
    
    Args:
        args: Command-line arguments
        frame_num: Current frame number
        
    Returns:
        Tuple of (clusters_by_id, clusters_by_label)
    """
    clusters_by_id = {}
    clusters_by_label = {}
    
    # Load clusters by ID
    if args.clusters_dir:
        pattern = os.path.join(args.clusters_dir, f"cluster_*_frame_{frame_num:03d}.txt")
        id_files = glob.glob(pattern)
        
        for file_path in id_files:
            # Extract cluster ID from filename
            filename = os.path.basename(file_path)
            parts = filename.split('_')
            if len(parts) >= 4:
                cluster_id = parts[1]
                
                # Load cluster points
                cluster_points = point_cloud.load_point_cloud(file_path)
                if cluster_points is not None:
                    clusters_by_id[cluster_id] = cluster_points
        
        logger.info(f"Loaded {len(clusters_by_id)} clusters by ID for frame {frame_num}")
    
    # Load clusters by label
    if args.clusters_by_label_dir:
        pattern = os.path.join(args.clusters_by_label_dir, f"cluster_*_frame_{frame_num:03d}.txt")
        label_files = glob.glob(pattern)
        
        for file_path in label_files:
            # Extract cluster label from filename
            filename = os.path.basename(file_path)
            parts = filename.split('_')
            if len(parts) >= 4:
                try:
                    cluster_label = int(parts[1])
                    
                    # Load cluster points
                    cluster_points = point_cloud.load_point_cloud(file_path)
                    if cluster_points is not None:
                        clusters_by_label[cluster_label] = cluster_points
                except ValueError:
                    pass
        
        logger.info(f"Loaded {len(clusters_by_label)} clusters by label for frame {frame_num}")
    
    return clusters_by_id, clusters_by_label

def filter_points_with_tracking(points):
    """
    Filter points but keep track of filtered points for potential reclamation.
    
    Args:
        points: Input point cloud
        
    Returns:
        Tuple of (filtered_points, discarded_points, filtered_indices)
    """
    # Create a simple initial filter that removes outliers
    try:
        # Calculate distance from mean
        centroid = np.mean(points, axis=0)
        distances = np.sqrt(np.sum((points - centroid)**2, axis=1))
        
        # Set threshold at 3 standard deviations
        threshold = np.mean(distances) + 3 * np.std(distances)
        
        # Create filter mask (True = keep, False = discard)
        filter_mask = distances <= threshold
        
        # Get filtered and discarded points
        filtered_points = points[filter_mask]
        discarded_points = points[~filter_mask]
        
        # Keep track of filtered indices
        filtered_indices = np.where(~filter_mask)[0]
        
        logger.info(f"Filtered {len(filtered_indices)} outlier points from {len(points)} total points")
        
        return filtered_points, discarded_points, filtered_indices
    except Exception as e:
        logger.error(f"Error in filtering: {str(e)}")
        # Return original points and empty arrays as fallback
        return points, np.array([]), np.array([], dtype=int)

def apply_clustering_method(points, args):
    """
    Apply appropriate clustering method based on arguments.
    FIXED: Now includes artifact filtering for all clustering methods.
    
    Args:
        points: Point cloud to cluster
        args: Command-line arguments
        
    Returns:
        Tuple of (labels, n_clusters)
    """
    if args.use_grid:
        # Create grid with requested resolution
        if hasattr(args, 'use_mmpose') and (args.use_mmpose or getattr(args, 'use_intelligent_completion', False)) and ENHANCED_MODULES_AVAILABLE:
            occupancy_grid = EnhancedOccupancyGrid(
                resolution=args.grid_resolution,
                fixed_bounds=(COP_BOUNDS['min'], COP_BOUNDS['max'])
            )
            occupancy_grid.add_points(points, confidence=1.0, frame_id=f"frame_{args.current_frame_num if hasattr(args, 'current_frame_num') else 0}")
        else:
            occupancy_grid = grid.OccupancyGrid(
                resolution=args.grid_resolution,
                fixed_bounds=(COP_BOUNDS['min'], COP_BOUNDS['max'])
            )
            occupancy_grid.add_points(points)
        
        # Filter noise if requested (this stays the same)
        if args.filter_noise:
            occupancy_grid.filter_noise(min_points=args.min_points_per_cell)
        
        # Perform clustering on grid cells
        clusters, n_clusters = occupancy_grid.cluster_cells(
            min_points=args.min_points_per_cell,
            connectivity=args.connectivity
        )
        # ========== FIX 6: CHECK FOR TOO MANY CLUSTERS ==========
        if n_clusters > 100:  # Way too many clusters - something's wrong!
            logger.error(f"Grid clustering produced {n_clusters} clusters - likely each voxel is its own cluster!")
            logger.warning("Attempting to merge nearby clusters...")
            
            # Option 1: Force merge very close clusters
            # Get all cluster centroids
            cluster_centroids = {}
            for cluster_id, cells in clusters.items():
                if cells:
                    # Get centroid of this cluster's cells
                    cell_positions = [occupancy_grid._cell_to_point(cell) for cell in cells]
                    cluster_centroids[cluster_id] = np.mean(cell_positions, axis=0)
            
            # Merge clusters within 10cm of each other
            merged_clusters = {}
            processed = set()
            new_id = 1
            
            for cid1, cells1 in clusters.items():
                if cid1 in processed:
                    continue
                
                # Start new merged cluster
                merged_cells = list(cells1)
                processed.add(cid1)
                
                # Find nearby clusters to merge
                for cid2, cells2 in clusters.items():
                    if cid2 in processed:
                        continue
                    
                    # Check distance between centroids
                    dist = np.linalg.norm(cluster_centroids[cid1] - cluster_centroids[cid2])
                    if dist < 10.0:  # Within 10cm
                        merged_cells.extend(cells2)
                        processed.add(cid2)
                
                # Store merged cluster
                merged_clusters[new_id] = merged_cells
                new_id += 1
            
            # Replace clusters with merged version
            clusters = merged_clusters
            n_clusters = len(merged_clusters)
            logger.info(f"After merging nearby clusters: {n_clusters} clusters")
            
            # If STILL too many, something is fundamentally wrong
            if n_clusters > 50:
                logger.error("Still too many clusters after merging. Grid clustering may be broken.")
                # Could fall back to DBSCAN here if needed
        # ========== END FIX 6 ==========
        # Create point-based labels from cell clusters
        labels = np.full(len(points), -1, dtype=np.int32)  # Force integer type!
        
        # Create mapping from point to index with rounding
        point_indices = {}
        for i, point in enumerate(points):
            point_tuple = tuple(np.round(point, 3))  # Round to avoid precision issues
            point_indices[point_tuple] = i
        
        # Assign cluster labels
        for cluster_label, cluster_cells in clusters.items():
            cluster_label = int(cluster_label)  # Ensure integer
            cluster_points = occupancy_grid.get_cluster_points(cluster_cells)
            for point in cluster_points:
                point_tuple = tuple(np.round(point, 3))  # Round to avoid float precision issues
                if point_tuple in point_indices:
                    labels[point_indices[point_tuple]] = cluster_label - 1  # 0-based
        
        # ========== CRITICAL FIX: Apply artifact filtering ==========
        labels, artifact_count = clustering.filter_artifacts(labels, points)
        
        # ========== Apply significant cluster filtering ==========
        labels, removed_count = clustering.filter_significant_clusters(
            labels, points,
            min_size_percentage=1.0,  # Use 1% instead of default 3%
            remove_noise=True
        )
        
        # Recount clusters after all filtering
        unique_labels = set(labels) - {-1}
        n_clusters = len(unique_labels)
        
        if artifact_count > 0 or removed_count > 0:
            logger.info(f"Grid clustering after filtering: {n_clusters} clusters (removed {artifact_count} artifacts, {removed_count} small clusters)")
        
        return labels, n_clusters
    
    elif args.use_dbscan:
        labels, n_clusters = clustering.apply_dbscan(
            points, eps=args.eps, min_samples=args.min_samples
        )
        
        # Apply artifact filtering for DBSCAN
        labels, artifact_count = clustering.filter_artifacts(labels, points)
        
        # Apply significant cluster filtering
        labels, removed_count = clustering.filter_significant_clusters(
            labels, points,
            min_size_percentage=1.0,
            remove_noise=True
        )
        
        # Recount clusters after filtering
        unique_labels = set(labels) - {-1}
        n_clusters = len(unique_labels)
        
        if artifact_count > 0 or removed_count > 0:
            logger.info(f"DBSCAN after filtering: {n_clusters} clusters (removed {artifact_count} artifacts, {removed_count} small clusters)")
        
        return labels, n_clusters
    else:
        labels, n_clusters = clustering.apply_hdbscan(
            points, 
            min_cluster_size=args.min_cluster_size,
            min_samples=args.min_samples,
            cluster_selection_epsilon=args.cluster_selection_epsilon,
            allow_single_cluster=args.allow_single_cluster
        )
        
        # Apply artifact filtering for HDBSCAN
        labels, artifact_count = clustering.filter_artifacts(labels, points)
        
        # Apply significant cluster filtering
        labels, removed_count = clustering.filter_significant_clusters(
            labels, points,
            min_size_percentage=1.0,
            remove_noise=True
        )
        
        # Recount clusters after filtering
        unique_labels = set(labels) - {-1}
        n_clusters = len(unique_labels)
        
        if artifact_count > 0 or removed_count > 0:
            logger.info(f"HDBSCAN after filtering: {n_clusters} clusters (removed {artifact_count} artifacts, {removed_count} small clusters)")
        
        return labels, n_clusters

def process_single_frame(file_path, args, frame_num=None, existing_clusters=None,
                         frame_buffer=None, state_bank=None, skip_temporal=False):
    """
    Process a single point cloud frame with clustering.
    COMPLETELY FIXED VERSION - Proper UUID management, segment ICCS, and dynamic anomaly detection.
    
    FIXES APPLIED:
    - Removed hardcoded frame numbers (was: if frame_num == 46)
    - Added dynamic anomaly detection based on size ratios and volume changes
    - Added segment ICCS integration for limb capsule validation
    - Added pose prediction detection and tracking
    - Fixed 2D-guided 3D keypoint extraction
    - Fixed undefined cluster_coords
    """
    # Import uuid at function level to avoid scope issues
    import uuid as uuid_module
    from datetime import datetime
    import numpy as np
    
    # ============== ANOMALY DETECTION THRESHOLDS ==============
    ANOMALY_THRESHOLDS = {
        'size_ratio_min': 0.3,      # Flag if size drops below 30% of previous
        'size_ratio_max': 3.0,      # Flag if size grows above 300% of previous
        'centroid_jump_cm': 50.0,   # Flag if centroid moves more than 50cm
        'volume_change_ratio': 0.5, # Flag if volume changes by more than 50%
    }
    
    # ============== COMPACT JSON HELPERS ==============
    def compact_reconstruction_data(obj):
        """Round floats to 1 decimal and keep ONLY reconstruction-essential fields"""
        if isinstance(obj, dict):
            result = {}
            skip_fields = {'normal', 'method', 'pattern_summary', 'y1_voxels', 'y2_voxels', 
                        'both_y_voxels', 'x_span', 'y_span', 'z_span', 'density', 'voxel_indices'}
            for key, value in obj.items():
                if isinstance(key, tuple):
                    key = str(key)
                if key in skip_fields:
                    continue
                result[key] = compact_reconstruction_data(value)
            return result
        elif isinstance(obj, list):
            return [round(x, 1) if isinstance(x, float) else compact_reconstruction_data(x) for x in obj]
        elif isinstance(obj, float):
            return round(obj, 1)
        elif isinstance(obj, np.ndarray):
            return [round(float(x), 1) for x in obj]
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return round(float(obj), 1)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, tuple):
            return str(obj)
        elif hasattr(obj, 'name'):
            return obj.name
        elif hasattr(obj, 'value'):
            return obj.value
        else:
            return obj

    def write_compact_json(data, filepath):
        """Write JSON with compact formatting"""
        def format_value(val, indent=0):
            indent_str = "  " * indent
            
            if isinstance(val, dict):
                if not val:
                    return "{}"
                
                items = []
                current_line = ""
                result_lines = ["{"]
                
                for i, (k, v) in enumerate(val.items()):
                    if isinstance(k, tuple):
                        k = str(k)
                    key_str = f'"{k}": '
                    val_str = format_value(v, indent + 1)
                    
                    item = key_str + val_str
                    if i < len(val) - 1:
                        item += ","
                    
                    if current_line and len(current_line) + len(item) + 1 > 100:
                        result_lines.append(indent_str + "  " + current_line)
                        current_line = item
                    else:
                        if current_line:
                            current_line += " " + item
                        else:
                            current_line = item
                
                if current_line:
                    result_lines.append(indent_str + "  " + current_line)
                
                result_lines.append(indent_str + "}")
                return "\n".join(result_lines)
                
            elif isinstance(val, list):
                if not val:
                    return "[]"
                
                if all(isinstance(x, (int, float)) for x in val):
                    return "[" + ", ".join(str(round(x, 1) if isinstance(x, float) else x) for x in val) + "]"
                
                if all(isinstance(x, list) for x in val):
                    inner_strs = []
                    for item in val:
                        if isinstance(item, list) and all(isinstance(v, (int, float)) for v in item):
                            inner_strs.append("[" + ", ".join(str(round(v, 1) if isinstance(v, float) else v) for v in item) + "]")
                        else:
                            inner_strs.append(format_value(item, indent + 1))
                    return "[" + ", ".join(inner_strs) + "]"
                
                items = [format_value(item, indent + 1) for item in val]
                one_line = "[" + ", ".join(items) + "]"
                if len(one_line) < 100:
                    return one_line
                
                return "[\n" + indent_str + "  " + (",\n" + indent_str + "  ").join(items) + "\n" + indent_str + "]"
                
            elif isinstance(val, str):
                return f'"{val}"'
            elif isinstance(val, bool):
                return str(val).lower()
            elif isinstance(val, float):
                return str(round(val, 1))
            elif isinstance(val, int):
                return str(val)
            elif val is None:
                return "null"
            elif hasattr(val, 'name'):
                return f'"{val.name}"'
            else:
                return f'"{str(val)}"'
        
        formatted = format_value(data, 0)
        
        with open(filepath, 'w') as f:
            f.write(formatted)

    # ============== DYNAMIC ANOMALY DETECTION HELPER ==============
    def detect_cluster_anomalies(current_clusters, state_bank, frame_num):
        """
        Detect anomalies by comparing current frame to previous frame.
        Returns dict of detected anomalies for logging/handling.
        """
        anomalies = {
            'size_swaps': [],
            'size_jumps': [],
            'centroid_jumps': [],
            'volume_anomalies': [],
            'uuid_inconsistencies': []
        }
        
        if state_bank is None or frame_num is None or frame_num < 2:
            return anomalies
        
        prev_frame_num = frame_num - 1
        if prev_frame_num not in state_bank.frame_clusters:
            return anomalies
        
        prev_frame_data = state_bank.frame_clusters[prev_frame_num]
        
        # Build lookup by UUID
        current_by_uuid = {}
        for cluster_key, cluster_data in current_clusters.items():
            uuid_val = cluster_data.get('uuid')
            if uuid_val:
                current_by_uuid[uuid_val] = {
                    'key': cluster_key,
                    'data': cluster_data
                }
        
        # Check each UUID that existed in previous frame
        for prev_uuid, prev_data in prev_frame_data.items():
            prev_points = prev_data.get('total_points', prev_data.get('point_count', 0))
            prev_centroid = np.array(prev_data.get('centroid', [0, 0, 0]))
            
            if prev_uuid in current_by_uuid:
                curr_info = current_by_uuid[prev_uuid]
                curr_data = curr_info['data']
                curr_key = curr_info['key']
                curr_points = curr_data.get('total_points', curr_data.get('point_count', 0))
                curr_centroid = np.array(curr_data.get('centroid', [0, 0, 0]))
                
                # Check size ratio
                if prev_points > 0:
                    size_ratio = curr_points / prev_points
                    
                    if size_ratio < ANOMALY_THRESHOLDS['size_ratio_min']:
                        anomalies['size_jumps'].append({
                            'uuid': prev_uuid,
                            'cluster_key': curr_key,
                            'prev_points': prev_points,
                            'curr_points': curr_points,
                            'ratio': size_ratio,
                            'type': 'shrink'
                        })
                        logger.warning(f"ANOMALY Frame {frame_num}: {prev_uuid[:8]} SHRUNK "
                                      f"{prev_points} [OK] {curr_points} (ratio={size_ratio:.2f})")
                    
                    elif size_ratio > ANOMALY_THRESHOLDS['size_ratio_max']:
                        anomalies['size_jumps'].append({
                            'uuid': prev_uuid,
                            'cluster_key': curr_key,
                            'prev_points': prev_points,
                            'curr_points': curr_points,
                            'ratio': size_ratio,
                            'type': 'grow'
                        })
                        logger.warning(f"ANOMALY Frame {frame_num}: {prev_uuid[:8]} GREW "
                                      f"{prev_points} [OK] {curr_points} (ratio={size_ratio:.2f})")
                
                # Check centroid jump
                centroid_distance = np.linalg.norm(curr_centroid - prev_centroid)
                if centroid_distance > ANOMALY_THRESHOLDS['centroid_jump_cm']:
                    anomalies['centroid_jumps'].append({
                        'uuid': prev_uuid,
                        'cluster_key': curr_key,
                        'prev_centroid': prev_centroid.tolist(),
                        'curr_centroid': curr_centroid.tolist(),
                        'distance': centroid_distance
                    })
                    logger.warning(f"ANOMALY Frame {frame_num}: {prev_uuid[:8]} JUMPED "
                                  f"{centroid_distance:.1f}cm")
        
        # Check for potential UUID swaps (cross-check sizes)
        if len(current_by_uuid) >= 2 and len(prev_frame_data) >= 2:
            current_sizes = [(uuid, d['data'].get('total_points', 0)) for uuid, d in current_by_uuid.items()]
            prev_sizes = [(uuid, d.get('total_points', d.get('point_count', 0))) for uuid, d in prev_frame_data.items()]
            
            # Sort by size
            current_sizes.sort(key=lambda x: x[1], reverse=True)
            prev_sizes.sort(key=lambda x: x[1], reverse=True)
            
            # Check if largest/smallest swapped
            if len(current_sizes) >= 2 and len(prev_sizes) >= 2:
                curr_largest_uuid, curr_largest_size = current_sizes[0]
                curr_smallest_uuid, curr_smallest_size = current_sizes[-1]
                prev_largest_uuid, prev_largest_size = prev_sizes[0]
                prev_smallest_uuid, prev_smallest_size = prev_sizes[-1]
                
                # Swap detection: current largest was previous smallest (or vice versa)
                if (curr_largest_uuid == prev_smallest_uuid and 
                    curr_smallest_uuid == prev_largest_uuid and
                    prev_largest_size > prev_smallest_size * 2):  # Significant size difference
                    
                    anomalies['size_swaps'].append({
                        'frame': frame_num,
                        'uuid1': curr_largest_uuid,
                        'uuid2': curr_smallest_uuid,
                        'description': 'Largest/smallest clusters swapped UUIDs'
                    })
                    logger.error(f"[OK] ANOMALY Frame {frame_num}: POTENTIAL UUID SWAP DETECTED!")
                    logger.error(f"   {curr_largest_uuid[:8]} was smallest, now largest")
                    logger.error(f"   {curr_smallest_uuid[:8]} was largest, now smallest")
        
        return anomalies

    try:
        # ============================================================
        # STEP 1: LOAD POINT CLOUD
        # ============================================================
        points = point_cloud.load_point_cloud(file_path)
        
        if points is None or len(points) == 0:
            logger.error(f"Failed to load points from {file_path}")
            return None
        
        if frame_num is None:
            frame_num = point_cloud.extract_frame_number(file_path)
            if frame_num is None:
                frame_num = 1
        
        logger.info(f"Processing {len(points)} points from frame {frame_num}")

        # ISSUE-3 anti-merge: claim voxels occupied by locked rigid primitives
        # (chair/table/...) OUT of the cloud BEFORE clustering, so a touching
        # rigid object cannot fuse into the person blob (the 76-109 balagan).
        # GATED OFF by default (state_bank.rigid_claim_enabled): enabling needs a
        # smoke-test because removing voxels can interact with the historical-lock
        # / tunnel-recovery path for the claimed object.  Complete no-op unless
        # the module imported, a primitive is locked, AND the flag is on.
        if (state_bank is not None
                and getattr(state_bank, 'rigid_claim_enabled', False)
                and getattr(state_bank, 'rigid_primitives', None)):
            try:
                _prims = list(state_bank.rigid_primitives.values())
                _before = len(points)
                points, _claimed = claim_rigid_primitives(points, _prims, margin=-2.0)
                _n_claimed = int(sum(len(c) for c in _claimed))
                if _n_claimed > 0:
                    logger.info(f"[RIGID-CLAIM] frame {frame_num}: claimed "
                                f"{_n_claimed} voxels for {len(_prims)} rigid "
                                f"primitive(s) ({_before}->{len(points)})")
            except Exception as _rc_exc:
                logger.warning(f"[RIGID-CLAIM] skipped: {_rc_exc}")

        # ============================================================
        # STEP 2: CREATE VOXEL GRID
        # ============================================================
        voxel_grid = None
        grid = None  # Alias for inject_historical_lock_clusters
        voxel_patterns = {}
        voxel_metadata = {}
        
        if args.use_grid or args.refine_boundaries:
            from enhanced_grid import EnhancedOccupancyGrid
            voxel_grid = EnhancedOccupancyGrid(
                resolution=args.grid_resolution,
                fixed_bounds=(COP_BOUNDS['min'], COP_BOUNDS['max'])
            )
            voxel_grid.add_points(points)
            voxel_grid.analyze_voxel_patterns()
            grid = voxel_grid  # Alias
            
            logger.info(f"Created enhanced grid with {len(voxel_grid.occupied_cells)} occupied cells")
            logger.info(f"Analyzed patterns for {len(voxel_grid.voxel_patterns)} voxels")
            
            if hasattr(voxel_grid, 'voxel_patterns'):
                voxel_patterns = voxel_grid.voxel_patterns
        
        # ============================================================
        # STEP 2.5: EARLY 2D POSE DETECTION (for tunnel construction)
        # Run MMPose on the CURRENT frame BEFORE building the tunnel,
        # so the tunnel uses this frame's bbox — not the previous frame's.
        # Full 3D extraction + ICCS happens later at STEP 6.
        # ============================================================
        _early_pose_2d_bbox = None
        _early_pose_2d_keypoints = None  # Full 17 keypoints for head placement in synth
        _early_pose_strategy = None      # ISSUE #4: pose DB lookup result
        _early_frame_img = None
        _early_frame_path = None
        if args.frames_dir and state_bank is not None:
            try:
                import cv2 as _cv2_early
                _early_frame_path = point_cloud.find_matching_frame(file_path, args.frames_dir)
                if _early_frame_path and os.path.exists(_early_frame_path):
                    _early_frame_img = _cv2_early.imread(_early_frame_path)
                    if _early_frame_img is not None:
                        # Initialize MMPose if not yet done
                        if not hasattr(process_single_frame, '_mmpose_instance'):
                            from mmpose_integration import MMPoseIntegration
                            _cam_p = getattr(args, 'camera_position', [-47.0, 28.0, -20.0])
                            _cam_t = getattr(args, 'camera_target', [-25.1, 123.8, -28.3])
                            _fl = getattr(args, 'focal_length', 27.5)
                            _fv = getattr(args, 'field_of_view', 66.0)
                            if isinstance(_cam_p, str):
                                _cam_p = [float(x) for x in _cam_p.split(',')]
                            if isinstance(_cam_t, str):
                                _cam_t = [float(x) for x in _cam_t.split(',')]
                            device = 'cuda' if getattr(args, 'use_gpu', False) else 'cpu'
                            process_single_frame._mmpose_instance = MMPoseIntegration(
                                models_dir=getattr(args, 'mmpose_models_dir', 'C:/MMPose'),
                                device=device,
                                camera_position=_cam_p, camera_target=_cam_t,
                                focal_length=_fl, field_of_view=_fv
                            )
                            logger.info(f"[OK] MMPose initialized early with camera: pos={_cam_p}, target={_cam_t}, fov={_fv}")

                        _mmpose_early = process_single_frame._mmpose_instance
                        _early_results = _mmpose_early.detect_2d_pose(_early_frame_img, frame_num=frame_num)
                        if _early_results:
                            # detect_2d_pose returns a pose DICT {'keypoints': array, ...}
                            # — not a raw array.  Extract keypoints first.
                            _raw_kps_early = (_early_results.get('keypoints')
                                              if isinstance(_early_results, dict)
                                              else _early_results)
                            if _raw_kps_early is not None:
                                _ekps = np.array(_raw_kps_early)
                            else:
                                _ekps = np.array([])
                            if _ekps.ndim == 2 and _ekps.shape[0] >= 17:
                                # Stash full keypoints for head-placement synth
                                _early_pose_2d_keypoints = _ekps.tolist()
                                _evalid = _ekps[:, 2] > 0.3 if _ekps.shape[1] > 2 else np.ones(len(_ekps), dtype=bool)
                                if np.any(_evalid):
                                    _early_pose_2d_bbox = [
                                        float(np.min(_ekps[_evalid, 0])),
                                        float(np.min(_ekps[_evalid, 1])),
                                        float(np.max(_ekps[_evalid, 0])),
                                        float(np.max(_ekps[_evalid, 1])),
                                        1.0
                                    ]
                                    logger.info(
                                        f"[OK] FIX-E: CURRENT frame {frame_num} pose_2d_bbox: "
                                        f"[{_early_pose_2d_bbox[0]:.0f},{_early_pose_2d_bbox[1]:.0f},"
                                        f"{_early_pose_2d_bbox[2]:.0f},{_early_pose_2d_bbox[3]:.0f}]")
                                # ── ISSUE #4: classify pose -> scan strategy ──
                                try:
                                    _early_pose_strategy = _utils_pose.classify_pose(
                                        _early_pose_2d_keypoints)
                                    if _early_pose_strategy and _early_pose_strategy.get('ok'):
                                        logger.info(
                                            f"[POSE-DB] Frame {frame_num}: "
                                            f"{_utils_pose.axis_to_log_label(_early_pose_strategy)} "
                                            f"row#{_early_pose_strategy['db_row_id']} "
                                            f"\"{_early_pose_strategy['db_row_name']}\" - "
                                            f"{_early_pose_strategy['reason']}")
                                        _pose_match_dist = _early_pose_strategy.get('match_dist', 0.0)
                                        _pose_clamped = _early_pose_strategy.get('clamped_kps', [])
                                        _pose_prev = getattr(process_single_frame, '_prev_pose_strategy', None)
                                        _pose_rejected = False
                                        if _pose_match_dist > 50.0 and _pose_prev is not None:
                                            logger.warning(
                                                f"[POSE-DB] Frame {frame_num}: REJECTED "
                                                f"match_dist={_pose_match_dist:.1f}>50")
                                            _early_pose_strategy = _pose_prev
                                            _pose_rejected = True
                                        elif _pose_clamped and _pose_prev is not None and _pose_match_dist > 35.0:
                                            logger.warning(
                                                f"[POSE-DB] Frame {frame_num}: REJECTED "
                                                f"clamped + match_dist={_pose_match_dist:.1f}>35")
                                            _early_pose_strategy = _pose_prev
                                            _pose_rejected = True
                                        if not _pose_rejected:
                                            process_single_frame._prev_pose_strategy = _early_pose_strategy
                                    else:
                                        logger.info(
                                            f"[POSE-DB] Frame {frame_num}: fallback "
                                            f"({_early_pose_strategy.get('reason', 'unknown') if _early_pose_strategy else 'none'})")
                                except Exception as _pe:
                                    logger.warning(f"[POSE-DB] classify_pose failed: {_pe}")
                                    _early_pose_strategy = None
            except Exception as _e_early:
                logger.warning(f"[FIX-E] Early 2D detection failed: {_e_early}")

        # ============================================================
        # DIAGNOSTIC: FLAT CONTROL PROJECTION + RED CONTOUR (integer sets)
        # ------------------------------------------------------------
        # Pure geometry, no images.  Collapses the raw CoP into the XZ
        # plane at voxel resolution.  The "control filled" set is every
        # (xi, zi) cell that has at least one raw CoP point.  The "red
        # contour" is the 4-neighbour boundary of the filled set: a cell
        # is a boundary cell when it is filled AND at least one of its
        # 4-neighbours is not filled.
        #
        # These sets are the ground-truth silhouette of the raw CoP for
        # this frame.  They are also injected into person_context so
        # clustering.py can drive the Y-WALL-WALK head-recovery repair
        # off a real silhouette-vs-cluster gap instead of a heuristic.
        # ============================================================
        _ctrl_filled = None
        _ctrl_origin = None
        _ctrl_res    = None
        if args.use_grid and hasattr(voxel_grid, 'bounds') and voxel_grid.bounds is not None:
            try:
                _ctrl_origin = np.asarray(voxel_grid.bounds[0])
                _ctrl_res = float(getattr(voxel_grid, 'resolution', 2.0))
                _ctrl_filled = set()
                for _cp in points:
                    _cxi = int((_cp[0] - _ctrl_origin[0]) / _ctrl_res)
                    _czi = int((_cp[2] - _ctrl_origin[2]) / _ctrl_res)
                    _ctrl_filled.add((_cxi, _czi))

                # Red contour = 4-neighbour boundary of _ctrl_filled
                _ctrl_red = set()
                for (_xi, _zi) in _ctrl_filled:
                    if ((_xi - 1, _zi) not in _ctrl_filled
                            or (_xi + 1, _zi) not in _ctrl_filled
                            or (_xi, _zi - 1) not in _ctrl_filled
                            or (_xi, _zi + 1) not in _ctrl_filled):
                        _ctrl_red.add((_xi, _zi))

                if _ctrl_filled:
                    _xs = [c[0] for c in _ctrl_filled]
                    _zs = [c[1] for c in _ctrl_filled]
                    _xw_lo = _ctrl_origin[0] + min(_xs) * _ctrl_res
                    _xw_hi = _ctrl_origin[0] + max(_xs) * _ctrl_res
                    _zw_lo = _ctrl_origin[2] + min(_zs) * _ctrl_res
                    _zw_hi = _ctrl_origin[2] + max(_zs) * _ctrl_res
                    logger.info(
                        f"[CTRL-XZ] Frame {frame_num}: "
                        f"filled={len(_ctrl_filled)} cells, "
                        f"red_contour={len(_ctrl_red)} cells, "
                        f"X=[{_xw_lo:.0f},{_xw_hi:.0f}]cm "
                        f"Z=[{_zw_lo:.0f},{_zw_hi:.0f}]cm "
                        f"from {len(points)} CoP points")
            except Exception as _ctrl_e:
                logger.warning(f"[CTRL-XZ] projection failed: {_ctrl_e}")
                _ctrl_filled = None
                _ctrl_origin = None
                _ctrl_res    = None

        # ============================================================
        # STEP 1 / FLAT-ORACLE — historical flat-projection occupancy.
        #
        # Two persistent per-axis 2D occupancy sets:
        #   _flat_xz_history : every CoP point ever observed across all
        #                      processed frames, projected to the XZ plane
        #                      (Y collapsed — entire scene's Y range
        #                      merged into one plane).
        #   _flat_yz_history : same idea, projected to YZ (X collapsed).
        #
        # Built sibling to _ctrl_filled and using the SAME voxel-grid
        # origin and resolution, so cells in the historical oracles map
        # one-to-one with cells in _ctrl_filled.  This is essential — the
        # cross-checks downstream compare per-frame _ctrl_filled to
        # the historical sets cell-by-cell.
        #
        # Persistence policy (per the design conversation):
        #   - Cells stay filled across frames; this delivery uses the
        #     simplest "ever-filled stays filled" rule, gated only on
        #     a fixed grid-origin/resolution match between frames.  When
        #     the voxel grid origin/resolution changes (e.g. between
        #     completely separate clips), the history is RESET so we
        #     never compare cells from incompatible grids.
        #   - No decay applied yet.  The conversation's eventual rule
        #     ("keep forever IF it was ever inside a cluster footprint,
        #     decay otherwise") is a separate later delivery — this
        #     step only builds the data structures.
        #
        # No clustering decisions consume this yet.  This delivery only:
        #   1. builds the two sets,
        #   2. logs their per-frame state,
        #   3. injects them into person_context so future steps can wire
        #      them into clustering.py without another scaffolding pass.
        # ============================================================
        _flat_xz_history = None
        _flat_yz_history = None
        _flat_xz_person_history = None
        _flat_yz_person_history = None
        _flat_xz_added_this_frame = 0
        _flat_yz_added_this_frame = 0
        if _ctrl_filled is not None and _ctrl_origin is not None and _ctrl_res is not None:
            try:
                # Reset history when the grid geometry changes between
                # frames.  We compare origin to ~1e-3 cm and resolution
                # exactly; mismatch means the cells are no longer
                # commensurable with what's stored.
                _prev_origin = getattr(process_single_frame, '_flat_oracle_origin', None)
                _prev_res    = getattr(process_single_frame, '_flat_oracle_res',    None)
                _grid_compatible = (
                    _prev_origin is not None
                    and _prev_res is not None
                    and abs(float(_prev_res) - float(_ctrl_res)) < 1e-9
                    and np.allclose(np.asarray(_prev_origin),
                                    np.asarray(_ctrl_origin),
                                    atol=1e-3)
                )

                if _grid_compatible:
                    _flat_xz_history = getattr(
                        process_single_frame, '_flat_xz_history_set', None)
                    _flat_yz_history = getattr(
                        process_single_frame, '_flat_yz_history_set', None)
                    _flat_xz_person_history = getattr(
                        process_single_frame, '_flat_xz_person_set', None)
                    _flat_yz_person_history = getattr(
                        process_single_frame, '_flat_yz_person_set', None)

                if _flat_xz_history is None:
                    _flat_xz_history = set()
                    if _prev_origin is not None and not _grid_compatible:
                        logger.warning(
                            f"[FLAT-ORACLE] Grid origin/resolution changed "
                            f"between frames - resetting flat XZ/YZ history")
                if _flat_yz_history is None:
                    _flat_yz_history = set()
                if _flat_xz_person_history is None:
                    _flat_xz_person_history = set()
                if _flat_yz_person_history is None:
                    _flat_yz_person_history = set()

                # Project every CoP point onto XZ and YZ using the same
                # origin & resolution as _ctrl_filled.
                # NOTE: _cp[0]=X, _cp[1]=Y, _cp[2]=Z (matches the
                # convention used in the CTRL-XZ block above).
                _xz_before = len(_flat_xz_history)
                _yz_before = len(_flat_yz_history)
                _ox = float(_ctrl_origin[0])
                _oy = float(_ctrl_origin[1])
                _oz = float(_ctrl_origin[2])
                _r  = float(_ctrl_res)
                for _cp in points:
                    _xi = int((float(_cp[0]) - _ox) / _r)
                    _yi = int((float(_cp[1]) - _oy) / _r)
                    _zi = int((float(_cp[2]) - _oz) / _r)
                    _flat_xz_history.add((_xi, _zi))
                    _flat_yz_history.add((_yi, _zi))
                _flat_xz_added_this_frame = len(_flat_xz_history) - _xz_before
                _flat_yz_added_this_frame = len(_flat_yz_history) - _yz_before

                # Persist for the next frame using the established
                # process_single_frame._prev_* attribute pattern.
                process_single_frame._flat_xz_history_set = _flat_xz_history
                process_single_frame._flat_yz_history_set = _flat_yz_history
                process_single_frame._flat_xz_person_set  = _flat_xz_person_history
                process_single_frame._flat_yz_person_set  = _flat_yz_person_history
                process_single_frame._flat_oracle_origin  = np.asarray(_ctrl_origin).copy()
                process_single_frame._flat_oracle_res     = float(_ctrl_res)

                # Compute per-frame world-coordinate extents for the log.
                if _flat_xz_history:
                    _xs = [c[0] for c in _flat_xz_history]
                    _zs_xz = [c[1] for c in _flat_xz_history]
                    _xz_xw_lo = _ox + min(_xs) * _r
                    _xz_xw_hi = _ox + max(_xs) * _r
                    _xz_zw_lo = _oz + min(_zs_xz) * _r
                    _xz_zw_hi = _oz + max(_zs_xz) * _r
                else:
                    _xz_xw_lo = _xz_xw_hi = _xz_zw_lo = _xz_zw_hi = 0.0
                if _flat_yz_history:
                    _ys = [c[0] for c in _flat_yz_history]
                    _zs_yz = [c[1] for c in _flat_yz_history]
                    _yz_yw_lo = _oy + min(_ys) * _r
                    _yz_yw_hi = _oy + max(_ys) * _r
                    _yz_zw_lo = _oz + min(_zs_yz) * _r
                    _yz_zw_hi = _oz + max(_zs_yz) * _r
                else:
                    _yz_yw_lo = _yz_yw_hi = _yz_zw_lo = _yz_zw_hi = 0.0

                logger.info(
                    f"[FLAT-ORACLE] Frame {frame_num}: "
                    f"XZ-history={len(_flat_xz_history)} cells "
                    f"(+{_flat_xz_added_this_frame} new), "
                    f"YZ-history={len(_flat_yz_history)} cells "
                    f"(+{_flat_yz_added_this_frame} new)")
                logger.info(
                    f"[FLAT-ORACLE] XZ extent: "
                    f"X=[{_xz_xw_lo:.0f},{_xz_xw_hi:.0f}]cm "
                    f"Z=[{_xz_zw_lo:.0f},{_xz_zw_hi:.0f}]cm; "
                    f"YZ extent: "
                    f"Y=[{_yz_yw_lo:.0f},{_yz_yw_hi:.0f}]cm "
                    f"Z=[{_yz_zw_lo:.0f},{_yz_zw_hi:.0f}]cm")
            except Exception as _flat_e:
                logger.warning(f"[FLAT-ORACLE] projection failed: {_flat_e}")
                _flat_xz_history = None
                _flat_yz_history = None
        else:
            logger.info(
                f"[FLAT-ORACLE] Frame {frame_num}: skipped "
                f"(CTRL-XZ unavailable - need voxel_grid bounds)")

        # ============================================================
        # STEP 3: APPLY CLUSTERING
        # ============================================================
        labels = None
        n_clusters = 0
        clusters = {}
        method_name = ''
        
        # ============================================================
        # FIX E PLUMBING: Build person_context for split veto
        # Uses previous-frame person data from state_bank to prevent
        # splits that cut through the tracked person's body.
        # ============================================================
        person_context = None
        if state_bank is not None:
            # Find person UUID
            person_lock_uuid = state_bank.primary_subject_uuid

            # Try INVENTORY first (Issue 3 authority)
            _person_y_span = None
            _person_centroid = None
            _person_bbox_fallback = None

            if (hasattr(state_bank, 'object_inventory') and state_bank.object_inventory
                    and person_lock_uuid and person_lock_uuid in state_bank.object_inventory):
                _inv_entry = state_bank.object_inventory[person_lock_uuid]
                _box = _inv_entry.get('last_3D_box', {})
                if 'min' in _box and 'max' in _box:
                    _person_y_span = [_box['min'][1], _box['max'][1]]
                    _person_bbox_fallback = {'min': _box['min'], 'max': _box['max']}
                _person_centroid = _inv_entry.get('last_centroid', [0, 0, 0])
                logger.info(f"[OK] FIX-E: person data from INVENTORY {person_lock_uuid[:8]}")

            # Fallback: historical_locks
            elif hasattr(state_bank, 'historical_locks'):
                if not person_lock_uuid:
                    best_span = 0
                    for lock_uuid, lock_info in state_bank.historical_locks.items():
                        y_span = lock_info.get('y_span')
                        if y_span:
                            span = abs(y_span[1] - y_span[0])
                            if span > best_span:
                                best_span = span
                                person_lock_uuid = lock_uuid

                if person_lock_uuid and person_lock_uuid in state_bank.historical_locks:
                    lock_info = state_bank.historical_locks[person_lock_uuid]
                    _person_y_span = lock_info.get('y_span')
                    _person_centroid = lock_info.get('centroid', [0, 0, 0])
                    _person_bbox_fallback = lock_info.get('bbox')

            if person_lock_uuid and _person_y_span is not None:
                
                # Get 2D pose bbox — prefer CURRENT frame (early detection),
                # fallback to previous frame's history only if early failed.
                pose_2d_bbox = None
                if _early_pose_2d_bbox is not None:
                    pose_2d_bbox = _early_pose_2d_bbox
                    logger.info(
                        f"[OK] FIX-E: using CURRENT frame {frame_num} pose_2d_bbox for tunnel")
                else:
                    # Fallback: previous frame's bbox from pose history
                    _pose_hist = getattr(process_single_frame, '_pose_2d_history', None)
                    if _pose_hist is not None and hasattr(_pose_hist, 'history') and len(_pose_hist.history) > 0:
                        for _ph_frame, _ph_kps in reversed(_pose_hist.history):
                            if _ph_frame < frame_num:
                                _kps = np.array(_ph_kps)
                                if _kps.ndim == 2 and _kps.shape[0] >= 17:
                                    _valid = _kps[:, 2] > 0.3 if _kps.shape[1] > 2 else np.ones(len(_kps), dtype=bool)
                                    if np.any(_valid):
                                        pose_2d_bbox = [
                                            float(np.min(_kps[_valid, 0])),
                                            float(np.min(_kps[_valid, 1])),
                                            float(np.max(_kps[_valid, 0])),
                                            float(np.max(_kps[_valid, 1])),
                                            1.0
                                        ]
                                        logger.info(
                                            f"[OK] FIX-E: pose_2d_bbox FALLBACK from frame {_ph_frame}: "
                                            f"[{pose_2d_bbox[0]:.0f},{pose_2d_bbox[1]:.0f},"
                                            f"{pose_2d_bbox[2]:.0f},{pose_2d_bbox[3]:.0f}]")
                                break
                
                if _person_y_span is not None:
                    # ── Find the person's first Y-wall from the CURRENT frame's voxel grid ──
                    # The first Y-wall is the physical front surface as seen by MiDaS.
                    # This does NOT depend on the inventory's last_3D_box (which can
                    # be corrupted).  It comes from the actual voxel data of this frame.
                    _person_y_wall = None
                    _body_depth = 50.0  # default human body depth in cm
                    _person_cent_y = _person_centroid[1] if _person_centroid else None

                    if voxel_grid is not None:
                        # Get Y-walls from current frame
                        _yw_idx = voxel_grid.generate_y_wall_index() if hasattr(voxel_grid, 'generate_y_wall_index') else {}
                        if _yw_idx:
                            # Find the significant Y-wall closest to the person's last known Y
                            _sig_walls = sorted([
                                y_pos for y_pos, stats in _yw_idx.items()
                                if stats.get('point_count', 0) > 100
                            ])
                            if _sig_walls and _person_cent_y is not None and _person_cent_y > 100:
                                # Nearest significant Y-wall to person centroid
                                _best_yw = min(_sig_walls, key=lambda y: abs(y - _person_cent_y))
                                # The FIRST Y-wall = the smallest Y in the person's region
                                # (closest to camera).  Find walls within ±60cm of person centroid,
                                # then take the minimum (front surface).
                                _near_walls = [y for y in _sig_walls if abs(y - _person_cent_y) < 60]
                                if _near_walls:
                                    _person_y_wall = min(_near_walls)
                                    # Body depth = distance from first wall to last wall
                                    # in the person's region, clamped to 20–80cm
                                    _last_wall = max(_near_walls)
                                    _body_depth = max(20.0, min(80.0, _last_wall - _person_y_wall + 10.0))
                                    logger.info(
                                        f"[OK] FIX-E: person first Y-wall={_person_y_wall:.0f}cm "
                                        f"(from {len(_near_walls)} walls near centroid Y={_person_cent_y:.0f}), "
                                        f"body_depth={_body_depth:.0f}cm")
                                else:
                                    _person_y_wall = _best_yw
                                    logger.info(
                                        f"[OK] FIX-E: person nearest Y-wall={_person_y_wall:.0f}cm "
                                        f"(no walls within 60cm of centroid Y={_person_cent_y:.0f})")

                    # Fallback: use inventory Y-span if no Y-wall found
                    if _person_y_wall is None and _person_y_span[0] > 100:
                        _person_y_wall = float(_person_y_span[0])
                        _body_depth = max(20.0, min(80.0, float(_person_y_span[1] - _person_y_span[0]) + 10.0))
                        logger.info(f"[OK] FIX-E: person Y-wall fallback from inventory Y-span: "
                                    f"{_person_y_wall:.0f}cm, depth={_body_depth:.0f}cm")

                    # ── ROBUST DEPTH (density-independent) ───────────────────────
                    # The wall-based _body_depth above is derived only from
                    # "significant" Y-walls (generate_y_wall_index drops <5-pt voxels
                    # and keeps >80-100-pt walls), so in sparse / side-on frames the
                    # medium-density back and side returns never form a significant
                    # wall and the span floors at 20cm -- killing the split veto and
                    # narrowing the acceptance zone (frames 38-53).  Re-measure depth
                    # straight from the raw CoP points in the person's XZ footprint
                    # (P10..P90 of Y) so it reflects geometry, not voxel thresholds.
                    # Only ever EXPANDS _body_depth, never shrinks it.
                    if (_person_y_wall is not None and _person_cent_y is not None
                            and points is not None and len(points) > 0):
                        try:
                            _pts = np.asarray(points, dtype=float)
                            if (_person_bbox_fallback
                                    and isinstance(_person_bbox_fallback, dict)
                                    and 'min' in _person_bbox_fallback):
                                _bx_lo = _person_bbox_fallback['min'][0] - 10.0
                                _bx_hi = _person_bbox_fallback['max'][0] + 10.0
                            else:
                                _bcx = (float(_person_centroid[0])
                                        if _person_centroid is not None else 0.0)
                                _bx_lo, _bx_hi = _bcx - 45.0, _bcx + 45.0
                            _slab = _pts[(_pts[:, 0] >= _bx_lo) & (_pts[:, 0] <= _bx_hi)
                                         & (np.abs(_pts[:, 1] - _person_cent_y) < 60.0)]
                            if len(_slab) >= 30:
                                _rd = float(np.percentile(_slab[:, 1], 90)
                                            - np.percentile(_slab[:, 1], 10))
                                if _rd > _body_depth:
                                    logger.info(
                                        f"[OK] FIX-E: robust _body_depth "
                                        f"{_body_depth:.0f}->{min(80.0, _rd):.0f}cm from "
                                        f"{len(_slab)} raw pts (P10-P90 Y) - "
                                        f"wall-based was sparse")
                                    _body_depth = max(20.0, min(80.0, _rd))
                        except Exception as _rde:
                            logger.debug(
                                f"[FIX-E] robust depth calc skipped: {_rde}")

                    # Build tunnel from 2D bbox + first Y-wall
                    person_bbox_3d = None
                    if pose_2d_bbox is not None and _person_y_wall is not None:
                        _cam_pos  = getattr(args, 'camera_position', [-47.0, 28.0, -20.0])
                        _cam_tgt  = getattr(args, 'camera_target',   [-25.1, 123.8, -28.3])
                        _cam_fov  = getattr(args, 'field_of_view',   66.0)
                        _img_size = getattr(args, 'image_size',      (480, 864))
                        _y_wall_front = _person_y_wall - args.grid_resolution
                        person_bbox_3d = _bbox_tunnel_to_3d(
                            pose_2d_bbox,
                            _y_wall_front, _body_depth + args.grid_resolution,
                            _cam_pos, _cam_tgt, _cam_fov, _img_size
                        )

                        # Override Z with actual cluster Z bounds from inventory
                        # Override X with inventory bounds or add ±10cm padding
                        if person_bbox_3d is not None:
                            if (hasattr(state_bank, 'object_inventory') and person_lock_uuid
                                    and person_lock_uuid in state_bank.object_inventory):
                                _inv = state_bank.object_inventory[person_lock_uuid]
                                _inv_box = _inv.get('last_3D_box', {})
                                if 'min' in _inv_box and 'max' in _inv_box:
                                    # TURN-TUNNEL FIX: the inventory bbox can be stale/narrow (it
                                    # lags the turning person in frames 42-50) and was REPLACING
                                    # the current 2D-pose tunnel, so the tunnel ended up narrower
                                    # than the segmentation contour and its top pinned to a short
                                    # Z -> the shell got clipped/collapsed and lost height. The
                                    # inventory now only EXPANDS the current pose tunnel (union),
                                    # never shrinks it, capped so a pose glitch / the +62cm wall
                                    # projection can't blow it open.
                                    _pose_x_min = person_bbox_3d['min'][0]
                                    _pose_x_max = person_bbox_3d['max'][0]
                                    _pose_z_min = person_bbox_3d['min'][2]
                                    _pose_z_max = person_bbox_3d['max'][2]
                                    _FLOOR_Z_GUESS = -154.0
                                    _MAX_TUNNEL_TOP_CM = _FLOOR_Z_GUESS + 195.0  # reject bg (~216)
                                    # Z from inventory (+ locked span), unioned with the pose top
                                    # (pose top capped -> excludes the wall/ceiling from the tunnel).
                                    _inv_z_min = _inv_box['min'][2]
                                    _inv_z_max = _inv_box['max'][2]
                                    _locked_z = _inv.get('locked_z_span')
                                    if _locked_z and _inv_z_min is not None:
                                        _inv_z_max = _inv_z_min + _locked_z
                                    _z_min_u = min(_inv_z_min, _pose_z_min)
                                    _z_max_u = max(_inv_z_max, min(_pose_z_max, _MAX_TUNNEL_TOP_CM))
                                    person_bbox_3d['min'][2] = _z_min_u
                                    person_bbox_3d['max'][2] = _z_max_u
                                    # FRAME-18/20 + TURN-TUNNEL X FIX: union inventory X with the
                                    # current 2D-pose X (expand-only), capped at the anatomical span
                                    # so a pose-keypoint glitch / inflated cluster can't blow it open.
                                    _inv_x_span = _inv_box['max'][0] - _inv_box['min'][0]
                                    _MAX_PERSON_X_SPAN_CM = 70.0
                                    if _inv_x_span <= _MAX_PERSON_X_SPAN_CM:
                                        _x_min_u = min(_inv_box['min'][0], _pose_x_min)
                                        _x_max_u = max(_inv_box['max'][0], _pose_x_max)
                                        if (_x_max_u - _x_min_u) <= _MAX_PERSON_X_SPAN_CM:
                                            person_bbox_3d['min'][0] = _x_min_u
                                            person_bbox_3d['max'][0] = _x_max_u
                                            logger.info(
                                                f"[TUNNEL] XZ inventory_union_pose: "
                                                f"X=[{_x_min_u:.0f},{_x_max_u:.0f}] "
                                                f"Z=[{_z_min_u:.0f},{_z_max_u:.0f}]cm")
                                        else:
                                            person_bbox_3d['min'][0] = _inv_box['min'][0]
                                            person_bbox_3d['max'][0] = _inv_box['max'][0]
                                            logger.info(
                                                f"[TUNNEL] XZ from inventory (pose union "
                                                f"{_x_max_u - _x_min_u:.0f}cm > "
                                                f"{_MAX_PERSON_X_SPAN_CM:.0f}cm, pose X rejected): "
                                                f"X=[{_inv_box['min'][0]:.0f},{_inv_box['max'][0]:.0f}] "
                                                f"Z=[{_z_min_u:.0f},{_z_max_u:.0f}]cm")
                                    else:
                                        logger.warning(
                                            f"[TUNNEL] X override SKIPPED - inventory X-span "
                                            f"{_inv_x_span:.0f}cm > {_MAX_PERSON_X_SPAN_CM:.0f}cm "
                                            f"(inflated cluster bbox). Keeping 2D-pose X="
                                            f"[{person_bbox_3d['min'][0]:.0f},"
                                            f"{person_bbox_3d['max'][0]:.0f}]cm, "
                                            f"Z=[{_z_min_u:.0f},{_z_max_u:.0f}]cm updated.")
                                    # END FRAME-18/20 + TURN-TUNNEL FIX
                            else:
                                # No inventory — add ±10cm X padding to projected bbox
                                person_bbox_3d['min'][0] -= 10.0
                                person_bbox_3d['max'][0] += 10.0
                                logger.info(
                                    f"[TUNNEL] X padded +/-10cm (no inventory): "
                                    f"X=[{person_bbox_3d['min'][0]:.0f},{person_bbox_3d['max'][0]:.0f}]")

                    # Fallback: use stored bbox only when tunnel cannot be built
                    if person_bbox_3d is None:
                        person_bbox_3d = _person_bbox_fallback
                        if person_bbox_3d:
                            logger.info(f"[OK] FIX-E: person_bbox_3d from stored bbox (fallback)")

                    # ── TUNNEL WIDEN (operator-requested margin) ─────────────
                    # Expand the finalized tunnel so the recomputed cluster
                    # (acceptance zone below reads person_bbox_3d) captures more
                    # of the person: +/-8cm in width (X), +40cm deeper (Y far
                    # side), +20cm up (Z top, toward the crown).
                    if person_bbox_3d is not None:
                        try:
                            person_bbox_3d['min'][0] = float(person_bbox_3d['min'][0]) - 8.0
                            person_bbox_3d['max'][0] = float(person_bbox_3d['max'][0]) + 8.0
                            person_bbox_3d['max'][1] = float(person_bbox_3d['max'][1]) + 40.0
                            person_bbox_3d['max'][2] = float(person_bbox_3d['max'][2]) + 20.0
                            logger.info(
                                f"[TUNNEL] widened margin X+/-8 Y+40 Z+20 -> "
                                f"X=[{person_bbox_3d['min'][0]:.0f},{person_bbox_3d['max'][0]:.0f}] "
                                f"Y=[{person_bbox_3d['min'][1]:.0f},{person_bbox_3d['max'][1]:.0f}] "
                                f"Z=[{person_bbox_3d['min'][2]:.0f},{person_bbox_3d['max'][2]:.0f}]cm")
                        except Exception as _twe:
                            logger.warning(f"[TUNNEL] widen margin failed: {_twe}")
                    # ── END TUNNEL WIDEN ─────────────────────────────────────

                    _pc = _person_centroid

                    if _person_y_wall is not None:
                        _veto_y_range = [float(_person_y_wall),
                                         float(_person_y_wall + _body_depth)]
                    else:
                        _veto_y_range = _person_y_span

                    # ── ACCEPTANCE ZONE (6 signals) ──────────────────────────
                    _az = None
                    try:
                        if (_flat_xz_history and _flat_yz_history
                                and _ctrl_origin is not None and _ctrl_res is not None
                                and _person_centroid is not None):
                            _az_ox=float(_ctrl_origin[0]); _az_oy=float(_ctrl_origin[1])
                            _az_oz=float(_ctrl_origin[2]); _az_r=float(_ctrl_res)
                            _az_cp=getattr(args,'camera_position',[-47.,28.,-20.])
                            _az_ct=getattr(args,'camera_target',[-25.1,123.8,-28.3])
                            _az_fv=getattr(args,'field_of_view',66.)
                            _az_is=getattr(args,'image_size',(480,864))
                            # Signal 3: Z from 2D bbox projection
                            _az_z_min=-220.; _az_z_max=50.
                            if (pose_2d_bbox is not None and _person_y_wall is not None
                                    and _person_y_wall>10.):
                                try:
                                    _uc=0.5*(float(pose_2d_bbox[0])+float(pose_2d_bbox[2]))
                                    _pt=_pixel_to_XZ_at_Y(_uc,float(pose_2d_bbox[1]),
                                        _person_y_wall,_az_cp,_az_ct,_az_fv,_az_is)
                                    _pb=_pixel_to_XZ_at_Y(_uc,float(pose_2d_bbox[3]),
                                        _person_y_wall,_az_cp,_az_ct,_az_fv,_az_is)
                                    _az_z_max=max(_pt[1],_pb[1])+10.
                                    _az_z_min=min(_pt[1],_pb[1])-10.
                                except Exception: pass
                            _az_zi_lo=int((_az_z_min-_az_oz)/_az_r)
                            _az_zi_hi=int((_az_z_max-_az_oz)/_az_r)
                            # Signal 1: Y directly from tunnel (y_wall + body_depth)
                            # + velocity extension.  NO flat history — flat YZ
                            # collapses X and must not be used to derive Y bounds.
                            _az_vel_y=0.
                            try:
                                if (hasattr(state_bank,'object_inventory') and person_lock_uuid
                                        and person_lock_uuid in state_bank.object_inventory):
                                    _vel=state_bank.object_inventory[person_lock_uuid].get(
                                        'velocity',[0,0,0])
                                    _az_vel_y=float(_vel[1]) if len(_vel)>1 else 0.
                            except Exception: pass
                            _az_vel_ext=min(30.,abs(_az_vel_y)*3.)
                            # y_wall = front surface (closest to camera)
                            # y_wall + body_depth = back surface (deepest)
                            _az_y_lo_h=(float(_person_y_wall)-_az_vel_ext
                                        if _person_y_wall else float(_person_centroid[1])-60.)
                            _az_y_hi_h=(float(_person_y_wall+_body_depth)+_az_vel_ext
                                        if _person_y_wall else float(_person_centroid[1])+60.)
                            _az_y_lo=_az_y_lo_h; _az_y_hi=_az_y_hi_h
                            # Enforce minimum body depth
                            _az_y_ctr=0.5*(_az_y_lo+_az_y_hi)
                            _az_min_bd=60.
                            try:
                                if (hasattr(state_bank,'object_inventory') and person_lock_uuid
                                        and person_lock_uuid in state_bank.object_inventory):
                                    _ib=state_bank.object_inventory[person_lock_uuid].get(
                                        'last_3D_box',{})
                                    if 'min' in _ib and 'max' in _ib:
                                        _az_min_bd=max(60.,float(_ib['max'][1]-_ib['min'][1]))
                            except Exception: pass
                            if (_az_y_hi-_az_y_lo)<_az_min_bd:
                                _az_y_lo=_az_y_ctr-_az_min_bd/2.
                                _az_y_hi=_az_y_ctr+_az_min_bd/2.
                            # ── ISSUE #4 — pose-aware Y floor ──
                            # For LATERAL / MIXED / UNRELIABLE bodies the
                            # Y direction IS the depth direction of the
                            # body itself (side-on stride, oblique span).
                            # The DB-driven world-cm estimate sets the
                            # minimum AZ Y span so Y-INJECT has room to
                            # admit the missing stride material.
                            _pose_strat_for_az = _early_pose_strategy or {}
                            _pose_axis_for_az = str(_pose_strat_for_az.get(
                                'depth_axis', 'FRONTAL'))
                            if _pose_axis_for_az in ('LATERAL', 'MIXED', 'UNRELIABLE'):
                                try:
                                    _pose_y_floor = _utils_pose.compute_span_world_cm(
                                        _pose_strat_for_az, person_bbox_3d)
                                except Exception:
                                    _pose_y_floor = _az_min_bd
                                # ── ISSUE #4 fix v2 — LIVE ank_pct_bh
                                #    driven Y floor.  For LATERAL the
                                #    stride defines the body's Y depth.
                                #    Compute real-world stride from the
                                #    live keypoints:
                                #        stride_cm = (ank_pct_bh / 100)
                                #                  * BH_world_cm
                                #    BH_world_cm: use the cluster Z-span
                                #    (head-to-foot in world) as a proxy
                                #    if no explicit value is available
                                #    (typical adult ~165 cm).  The
                                #    floor is then max(stride*2.0, 80)
                                #    so even a small stride yields room
                                #    for the front+back foot positions.
                                try:
                                    _ank_pct = float(_pose_strat_for_az.get(
                                        'ank_pct_bh') or 0.0)
                                    _wri_pct = float(_pose_strat_for_az.get(
                                        'wri_pct_bh') or 0.0)
                                    _span_pct = max(_ank_pct, _wri_pct)
                                    if person_bbox_3d:
                                        _bh_world = float(
                                            person_bbox_3d['max'][2]
                                            - person_bbox_3d['min'][2])
                                    else:
                                        _bh_world = 0.0
                                    if _bh_world < 80.0:
                                        _bh_world = 165.0
                                    _live_stride_cm = (_span_pct / 100.0) * _bh_world
                                    _live_floor = max(80.0,
                                                      _live_stride_cm * 2.2)
                                    if _live_floor > _pose_y_floor:
                                        _pose_y_floor = _live_floor
                                except Exception:
                                    pass
                                if _pose_y_floor > (_az_y_hi - _az_y_lo):
                                    _az_y_lo = _az_y_ctr - _pose_y_floor / 2.
                                    _az_y_hi = _az_y_ctr + _pose_y_floor / 2.
                            # Signal 2: X directly from person_bbox_3d.
                            # NO flat XZ history — flat XZ collapses Y and must
                            # not be used to derive X bounds.
                            if person_bbox_3d:
                                _az_x_lo=float(person_bbox_3d['min'][0])-10.
                                _az_x_hi=float(person_bbox_3d['max'][0])+10.
                            else:
                                _az_x_lo=float(_person_centroid[0])-40.
                                _az_x_hi=float(_person_centroid[0])+40.
                            _az_narrow=(_body_depth < 0.5*_az_min_bd and _az_min_bd>20.)
                            _az={'x_min':float(_az_x_lo),'x_max':float(_az_x_hi),
                                 'y_min':float(_az_y_lo),'y_max':float(_az_y_hi),
                                 'z_min':float(_az_z_min),'z_max':float(_az_z_max),
                                 'active':True,'narrow':bool(_az_narrow)}
                            # ── TYPE A: Expand AZ Z from inventory-corrected bbox ──
                            # person_bbox_3d Z was corrected from inventory at L1601.
                            # The AZ Z was computed independently from 2D projection.
                            # When inventory Z is wider (legs extend lower), expand
                            # the AZ to include the full body so clustering doesn't
                            # cut legs.
                            if person_bbox_3d is not None:
                                _inv_zmin = float(person_bbox_3d['min'][2])
                                _inv_zmax = float(person_bbox_3d['max'][2])
                                if _inv_zmin < _az['z_min']:
                                    logger.info(
                                        f"[ACCEPTANCE-ZONE] Frame {frame_num}: "
                                        f"Z_min expanded {_az['z_min']:.0f}->{_inv_zmin:.0f} "
                                        f"from inventory")
                                    _az['z_min'] = _inv_zmin
                                    _az_z_min = _inv_zmin
                                if _inv_zmax > _az['z_max']:
                                    _az['z_max'] = _inv_zmax
                                    _az_z_max = _inv_zmax
                            # ── TYPE B: Expand AZ Y for LATERAL + NARROW ──
                            # In LATERAL view the hand extends toward/away
                            # from the camera (Y direction).  The AZ Y is
                            # computed from body_depth which is narrow when
                            # side-on.  Arm reach ≈ 35% of body height
                            # (Z-span).  Expand Y-min toward camera so the
                            # extended hand enters clustering instead of
                            # being excluded from the acceptance zone.
                            if (_az_narrow
                                    and _pose_axis_for_az == 'LATERAL'):
                                _z_span = _az_z_max - _az_z_min
                                if _z_span > 50:
                                    _arm_reach = _z_span * 0.35
                                    _new_y_lo = _az_y_lo - _arm_reach
                                    _new_y_hi = _az_y_hi + _arm_reach * 0.4
                                    if _new_y_lo < _az_y_lo or \
                                            _new_y_hi > _az_y_hi:
                                        logger.info(
                                            f"[ACCEPTANCE-ZONE] "
                                            f"Frame {frame_num}: "
                                            f"Y expanded "
                                            f"[{_az_y_lo:.0f},"
                                            f"{_az_y_hi:.0f}]->"
                                            f"[{_new_y_lo:.0f},"
                                            f"{_new_y_hi:.0f}] "
                                            f"for LATERAL arm reach "
                                            f"(z_span={_z_span:.0f}cm, "
                                            f"reach={_arm_reach:.0f}cm)")
                                        _az_y_lo = _new_y_lo
                                        _az_y_hi = _new_y_hi
                                        _az['y_min'] = float(_new_y_lo)
                                        _az['y_max'] = float(_new_y_hi)
                            logger.info(
                                f"[ACCEPTANCE-ZONE] Frame {frame_num}: "
                                f"X=[{_az_x_lo:.0f},{_az_x_hi:.0f}] "
                                f"Y=[{_az_y_lo:.0f},{_az_y_hi:.0f}] "
                                f"Z=[{_az_z_min:.0f},{_az_z_max:.0f}]cm "
                                f"axis={_pose_axis_for_az}"
                                f"{'[WARN]NARROW' if _az_narrow else ''}")
                    except Exception as _aze:
                        logger.debug(f"[ACCEPTANCE-ZONE] failed: {_aze}")

                    # ── HAND ZONE: detect extended arm via MP33 ──
                    # In LATERAL view, arm extends in DEPTH (Y), not
                    # in pixel space.  Detection: check if MP33 wrist
                    # landmark falls outside the GREEN bbox (pose_2d_bbox).
                    # If wrist is outside the torso projection → arm
                    # extends beyond the cluster → need hand_zone.
                    # Cross-validates with MP33 segmentation mask.
                    _hand_zone = None
                    if (_az_narrow and _MP33_AVAILABLE
                            and _early_frame_img is not None
                            and _pose_axis_for_az == 'LATERAL'
                            and person_bbox_3d is not None):
                        try:
                            if not hasattr(process_single_frame,
                                           '_mp33_phase1'):
                                process_single_frame._mp33_phase1 = \
                                    _mp_pose_mod.solutions.pose.Pose(
                                        static_image_mode=False,
                                        model_complexity=1,
                                        enable_segmentation=True,
                                        min_detection_confidence=0.3,
                                        min_tracking_confidence=0.3)
                                logger.info(
                                    "[HAND-ZONE] MP33 Phase1 init")
                            _mp33p = process_single_frame._mp33_phase1
                            import cv2 as _cv2_hz
                            _hz_rgb = _cv2_hz.cvtColor(
                                _early_frame_img,
                                _cv2_hz.COLOR_BGR2RGB)
                            _hz_res = _mp33p.process(_hz_rgb)
                            if (_hz_res.pose_landmarks is not None
                                    and _hz_res.pose_landmarks.landmark):
                                _lm = _hz_res.pose_landmarks.landmark
                                _ih, _iw = _early_frame_img.shape[:2]

                                # GREEN bbox in original frame coords
                                # pose_2d_bbox from early MMPose
                                # detection is already in original
                                # frame coords — same as MP33
                                _gb = None
                                if pose_2d_bbox is not None:
                                    _gb = (
                                        pose_2d_bbox[0],
                                        pose_2d_bbox[1],
                                        pose_2d_bbox[2],
                                        pose_2d_bbox[3])

                                # Check each wrist
                                # MP: 15=L_wri, 16=R_wri
                                # Also check fingertips: 19=L_idx, 20=R_idx
                                for _wi, _side in [
                                        (15, 'L'), (16, 'R'),
                                        (19, 'L'), (20, 'R')]:
                                    _w = _lm[_wi]
                                    if _w.visibility <= 0.1:
                                        continue
                                    _wx = _w.x * _iw
                                    _wy = _w.y * _ih

                                    # Check: wrist outside GREEN bbox?
                                    _outside_bbox = False
                                    if _gb is not None:
                                        _margin = 15
                                        _outside_bbox = (
                                            _wx < _gb[0] - _margin
                                            or _wx > _gb[2] + _margin
                                            or _wy < _gb[1] - _margin
                                            or _wy > _gb[3] + _margin)

                                    # Cross-validate: seg mask at wrist
                                    _mask_ok = False
                                    if _hz_res.segmentation_mask \
                                            is not None:
                                        _seg = _hz_res \
                                            .segmentation_mask.numpy()
                                        _wy_i = min(int(_wy),
                                                    _seg.shape[0]-1)
                                        _wx_i = min(int(_wx),
                                                    _seg.shape[1]-1)
                                        _mask_ok = _seg[
                                            _wy_i, _wx_i] > 0.3

                                    # EITHER outside bbox OR mask
                                    # confirms person at wrist pos
                                    if _outside_bbox or _mask_ok:
                                        _pb = person_bbox_3d
                                        _bx_lo = float(_pb['min'][0])
                                        _bx_hi = float(_pb['max'][0])
                                        _bz_lo = float(_pb['min'][2])
                                        _bz_hi = float(_pb['max'][2])
                                        _by_lo = float(_pb['min'][1])
                                        _by_hi = float(_pb['max'][1])
                                        _bx_c = (_bx_lo+_bx_hi)/2
                                        _bz_c = (_bz_lo+_bz_hi)/2
                                        _z_span = _bz_hi - _bz_lo
                                        _reach = _z_span * 0.35
                                        _hand_zone = {
                                            'x_center': _bx_c,
                                            'z_center': _bz_c,
                                            'y_lo': _by_lo - _reach,
                                            'y_hi': _by_hi
                                                    + _reach * 0.4,
                                            'radius_xz': max(
                                                20,
                                                (_bx_hi-_bx_lo)*0.8),
                                            'side': _side,
                                        }
                                        logger.info(
                                            f"[HAND-ZONE] "
                                            f"Frame {frame_num}: "
                                            f"{_side}_arm detected "
                                            f"(wrist px="
                                            f"{_wx:.0f},{_wy:.0f} "
                                            f"outside_bbox="
                                            f"{_outside_bbox} "
                                            f"mask={_mask_ok}), "
                                            f"zone XZ=("
                                            f"{_hand_zone['x_center']:.0f},"
                                            f"{_hand_zone['z_center']:.0f})"
                                            f"+/-{_hand_zone['radius_xz']:.0f}cm"
                                            f" Y=["
                                            f"{_hand_zone['y_lo']:.0f},"
                                            f"{_hand_zone['y_hi']:.0f}]")
                                        break
                        except Exception as _hze:
                            logger.debug(
                                f"[HAND-ZONE] Frame {frame_num}: "
                                f"failed: {_hze}")

                    # ── Save MP33 wrist pixel positions for
                    # skeleton fitting (available even if
                    # hand_zone detection failed) ──
                    # ── MP33 wrist/arm landmarks for skeleton fitting ──
                    # Full-frame MP33 can place a wrist on a nearby object
                    # (chair/toy); that defected landmark then drags the fitted
                    # arm there.  REPAIR: the MP33 detector runs with
                    # enable_segmentation=True, so it returns the PERSON
                    # SILHOUETTE — the upstream equivalent of the yellow contour.
                    # Flaw = an arm landmark landing on BACKGROUND (off the
                    # silhouette).  When flawed, crop to the silhouette's bounding
                    # box (which excludes the off-body object) and RE-DETECT with
                    # a fresh static detector; the repaired landmarks replace the
                    # defected ones in _mp33_wrists, so the FITTED SKELETON (and
                    # thus the 3D-pose / IoU panels) inherits the correction.
                    # Falls back to the mmpose body box if no mask is available.
                    _mp33_wrists = None
                    if (hasattr(process_single_frame, '_mp33_phase1')
                            and _early_frame_img is not None):
                        try:
                            _mp33p = process_single_frame._mp33_phase1
                            import cv2 as _cv2_w
                            _w_rgb = _cv2_w.cvtColor(
                                _early_frame_img,
                                _cv2_w.COLOR_BGR2RGB)
                            _wh, _ww = _early_frame_img.shape[:2]
                            # L_wrist=15, R_wrist=16, L_elbow=13, R_elbow=14,
                            # L_shoulder=11, R_shoulder=12
                            _WRIST_IDS = [(15, 'L_wrist'), (16, 'R_wrist'),
                                          (13, 'L_elbow'), (14, 'R_elbow'),
                                          (11, 'L_shoulder'), (12, 'R_shoulder')]

                            def _mp33_arm_pts(_res, _iw, _ih, _ox=0, _oy=0):
                                """Extract the 6 arm landmarks from an MP33 result
                                in FULL-frame pixel coords."""
                                _out = {}
                                if (_res is not None
                                        and _res.pose_landmarks is not None
                                        and _res.pose_landmarks.landmark):
                                    _lm = _res.pose_landmarks.landmark
                                    for _wi, _wn in _WRIST_IDS:
                                        _wp = _lm[_wi]
                                        if _wp.visibility > 0.1:
                                            _out[_wn] = (_wp.x * _iw + _ox,
                                                         _wp.y * _ih + _oy,
                                                         _wp.visibility)
                                return _out

                            # ── Pass 1: full frame (the "defected" detection) ──
                            _w_res = _mp33p.process(_w_rgb)
                            _full_w = _mp33_arm_pts(_w_res, _ww, _wh)
                            _seg = getattr(_w_res, 'segmentation_mask', None)

                            # ── Flaw detect: arm landmark on BACKGROUND (off the
                            # person silhouette).  Fall back to the mmpose body
                            # box (+10% margin) if no segmentation mask. ──
                            _arm_flaw = False
                            if _seg is not None and _full_w:
                                for _wn, (_px, _py, _v) in _full_w.items():
                                    _ix = int(round(_px)); _iy = int(round(_py))
                                    if (0 <= _iy < _seg.shape[0]
                                            and 0 <= _ix < _seg.shape[1]):
                                        if float(_seg[_iy, _ix]) < 0.5:
                                            _arm_flaw = True
                                            break
                                    else:
                                        _arm_flaw = True   # landmark off-frame
                                        break
                            elif _early_pose_2d_bbox is not None and _full_w:
                                _bx1, _by1 = (float(_early_pose_2d_bbox[0]),
                                              float(_early_pose_2d_bbox[1]))
                                _bx2, _by2 = (float(_early_pose_2d_bbox[2]),
                                              float(_early_pose_2d_bbox[3]))
                                _mgx = 0.10 * max(_bx2 - _bx1, 1.0)
                                _mgy = 0.10 * max(_by2 - _by1, 1.0)
                                for _wn, (_px, _py, _v) in _full_w.items():
                                    if (_px < _bx1 - _mgx or _px > _bx2 + _mgx
                                            or _py < _by1 - _mgy
                                            or _py > _by2 + _mgy):
                                        _arm_flaw = True
                                        break

                            # ── Pass 2: repair via crop-and-rescan ──
                            # Crop box = person SILHOUETTE extent (the off-body
                            # object lies beyond it); fall back to the body box.
                            # RE-DETECT with a fresh static detector; repaired
                            # landmarks replace the defected.
                            _use_w = _full_w
                            _w_src = 'full'
                            if _arm_flaw:
                                _cb = None
                                if _seg is not None:
                                    _ys, _xs = np.where(_seg > 0.5)
                                    if _xs.size > 0:
                                        _cb = [int(_xs.min()), int(_ys.min()),
                                               int(_xs.max()), int(_ys.max())]
                                if (_cb is None
                                        and _early_pose_2d_bbox is not None):
                                    _cb = [int(_early_pose_2d_bbox[0]),
                                           int(_early_pose_2d_bbox[1]),
                                           int(_early_pose_2d_bbox[2]),
                                           int(_early_pose_2d_bbox[3])]
                                if _cb is not None:
                                    _cx1 = max(0, _cb[0]); _cy1 = max(0, _cb[1])
                                    _cx2 = min(_ww, _cb[2]); _cy2 = min(_wh, _cb[3])
                                    if _cx2 - _cx1 > 30 and _cy2 - _cy1 > 50:
                                        if not hasattr(process_single_frame,
                                                       '_mp33_rescan'):
                                            process_single_frame._mp33_rescan = \
                                                _mp_pose_mod.solutions.pose.Pose(
                                                    static_image_mode=True,
                                                    model_complexity=1,
                                                    enable_segmentation=False,
                                                    min_detection_confidence=0.3)
                                            logger.info("[MP33-WRIST] rescan "
                                                        "(static) detector init")
                                        _rdet = process_single_frame._mp33_rescan
                                        _cres = _rdet.process(
                                            _w_rgb[_cy1:_cy2, _cx1:_cx2])
                                        _crop_w = _mp33_arm_pts(
                                            _cres, _cx2 - _cx1, _cy2 - _cy1,
                                            _ox=_cx1, _oy=_cy1)
                                        if _crop_w:
                                            _use_w = _crop_w
                                            _w_src = 'crop'
                                            logger.info(
                                                f"[MP33-WRIST] Frame "
                                                f"{frame_num}: flaw "
                                                f"off-silhouette -> rescan "
                                                f"crop, repaired "
                                                f"{list(_crop_w.keys())}")

                            # ── Build _mp33_wrists (repaired-or-full) ──
                            if _use_w:
                                _mp33_wrists = {
                                    _wn: {'px': round(_px, 1),
                                          'py': round(_py, 1),
                                          'vis': round(_v, 3)}
                                    for _wn, (_px, _py, _v) in _use_w.items()}
                                logger.info(
                                    f"[MP33-WRIST] Frame {frame_num}: "
                                    f"{list(_mp33_wrists.keys())} "
                                    f"(src={_w_src})")
                        except Exception as _we:
                            logger.debug(
                                f"[MP33-WRIST] Frame {frame_num} "
                                f"repair failed: {_we}")

                    person_context = {
                        'person_y_range': _veto_y_range,
                        'person_centroid': _pc if isinstance(_pc, list) else _pc.tolist() if hasattr(_pc, 'tolist') else list(_pc),
                        'person_uuid': person_lock_uuid,
                        'pose_2d_bbox': pose_2d_bbox,
                        'pose_2d_keypoints': _early_pose_2d_keypoints,
                        'pose_strategy': _early_pose_strategy,
                        'camera_params': {
                            'camera_position': args.camera_position,
                            'camera_target':   args.camera_target,
                            'field_of_view':   args.field_of_view,
                            'image_size':      args.image_size,
                        },
                        'person_bbox_3d': person_bbox_3d,
                        'facing': getattr(process_single_frame, '_prev_facing', None) or 'unknown',
                        'fitting_path': getattr(process_single_frame, '_prev_fitting_path', None) or 'frontal',
                        'prev_person_z_max': getattr(process_single_frame, '_prev_person_z_max', None),
                        'ctrl_xz_oracle':  _ctrl_filled,
                        'ctrl_xz_origin':  _ctrl_origin,
                        'ctrl_xz_res':     _ctrl_res,
                        'flat_xz_history': _flat_xz_history,
                        'flat_yz_history': _flat_yz_history,
                        'flat_xz_person_history': _flat_xz_person_history,
                        'flat_yz_person_history': _flat_yz_person_history,
                        'min_body_depth': (lambda: (
                            lambda _b: max(60.0, float(_b['max'][1]-_b['min'][1]))
                            if _b and 'min' in _b and 'max' in _b else 60.0
                        )(state_bank.object_inventory.get(person_lock_uuid,{}).get(
                            'last_3D_box',{})
                          if state_bank is not None
                          and hasattr(state_bank,'object_inventory')
                          and person_lock_uuid else {})
                        )(),
                        'acceptance_zone': _az,
                        'hand_zone': _hand_zone,
                        'pose_axis': _pose_axis_for_az,
                    }
                    _b3d_info = (f", 3D-bbox X=[{person_bbox_3d['min'][0]:.0f},{person_bbox_3d['max'][0]:.0f}]"
                                 if person_bbox_3d else ", no 3D-bbox")
                    logger.info(f"[OK] FIX-E: Built person_context for split veto: "
                              f"Y=[{_veto_y_range[0]:.0f},{_veto_y_range[1]:.0f}]{_b3d_info}, uuid={person_lock_uuid[:8]}")
        
        if args.use_grid:
            # Get inventory cluster count for phantom prevention cap
            _inv_count = None
            if state_bank is not None and hasattr(state_bank, 'get_inventory_cluster_count'):
                _inv_count = state_bank.get_inventory_cluster_count()

            # ── PRIMARY: Y-wall progressive clustering ──
            # Densification (dummy points, chord closing) happens INSIDE
            # apply_y_wall_progressive_clustering per-region.
            labels, n_clusters, cluster_info = clustering.apply_y_wall_progressive_clustering(
                points, 
                voxel_grid,
                min_y_span=10.0,
                validation_eps=3.0,
                validation_min_samples=args.min_points_per_cell,
                use_enhanced_tracking=True,
                min_points_per_voxel=5,
                state_bank=state_bank,
                frame_num=frame_num,
                temporal_buffer=frame_buffer,
                person_context=person_context,
                inventory_cluster_count=_inv_count
            )
            method_name = 'y_wall_progressive'
            
            if 'clusters' in cluster_info and cluster_info.get('voxel_data_preserved'):
                clusters = cluster_info['clusters']
                logger.info("[OK] Using enhanced clusters dictionary with preserved voxel metadata")
            elif 'validated_clusters' in cluster_info:
                clusters = {}
                for idx, cluster_data in enumerate(cluster_info['validated_clusters']):
                    clusters[idx] = cluster_data
            else:
                clusters = {}
                unique_labels = sorted(set(labels) - {-1}) if labels is not None else []
                for label in unique_labels:
                    cluster_points = points[labels == label] if len(labels) == len(points) else np.array([])
                    clusters[label] = {
                        'points': cluster_points,
                        'point_count': len(cluster_points),
                        'centroid': np.mean(cluster_points, axis=0).tolist() if len(cluster_points) > 0 else [0, 0, 0]
                    }
            
            logger.info(f"Clustering completed: {n_clusters} clusters found using {method_name}")

            # ── RC-1: EARLY ANOMALY REPROCESSING ──────────────────────────────
            # Detect under-clustering NOW, before any surface accumulation or
            # mesh work.  The original check (lines ~2863-2895) fires at the
            # START of the NEXT frame — by then Poisson + PyMeshFix + PyMeshLab
            # + Trimesh have already run on the bad clusters, wasting significant
            # compute (frame 103 in the reference run: ~20 s of mesh pipeline
            # ran before the reprocess was triggered).
            if args.use_grid and frame_buffer is not None and n_clusters < 2:
                _early_hist = []
                for _fb in frame_buffer.frames:
                    if _fb.get('frame_num', -1) != frame_num:
                        if 'n_clusters' in _fb:
                            _early_hist.append(_fb['n_clusters'])
                if _early_hist:
                    _early_avg = float(np.mean(_early_hist))
                    if _early_avg >= 1.8 and n_clusters < _early_avg - 0.5:
                        logger.warning("=" * 70)
                        logger.warning(
                            f"[EARLY ANOMALY] Frame {frame_num}: "
                            f"{n_clusters} cluster(s) vs historical avg "
                            f"{_early_avg:.1f} - REPROCESSING BEFORE MESH WORK"
                        )
                        logger.warning("=" * 70)
                        _ea_labels, n_clusters, _ea_info =                             clustering.apply_y_wall_progressive_clustering(
                                points,
                                voxel_grid,
                                min_y_span=10.0,
                                validation_eps=3.0,
                                validation_min_samples=args.min_points_per_cell,
                                use_enhanced_tracking=True,
                                min_points_per_voxel=5,
                                force_split=True,
                                state_bank=state_bank,
                                frame_num=frame_num,
                                temporal_buffer=frame_buffer,
                                person_context=person_context,
                                inventory_cluster_count=_inv_count
                            )
                        logger.info(
                            f"[EARLY ANOMALY] Frame {frame_num} reprocessed: "
                            f"now {n_clusters} clusters"
                        )
                        if 'clusters' in _ea_info and _ea_info.get('voxel_data_preserved'):
                            clusters = _ea_info['clusters']
                            logger.info("[OK] Using enhanced clusters from early reprocess")
                        elif 'validated_clusters' in _ea_info:
                            clusters = {
                                idx: cd
                                for idx, cd in enumerate(_ea_info['validated_clusters'])
                            }
            # ── END RC-1 ──────────────────────────────────────────────────────

        elif args.use_dbscan:
            labels = clustering.apply_dbscan(points, eps=args.eps, min_samples=args.min_samples)
            unique_labels = set(labels) - {-1}
            n_clusters = len(unique_labels)
            method_name = 'dbscan'
            
            clusters = {}
            for label in unique_labels:
                cluster_points = points[labels == label]
                clusters[label] = {
                    'points': cluster_points,
                    'point_count': len(cluster_points),
                    'centroid': np.mean(cluster_points, axis=0).tolist()
                }
        else:
            labels = clustering.apply_hdbscan(
                points, 
                min_cluster_size=args.min_cluster_size,
                min_samples=args.min_samples,
                cluster_selection_epsilon=args.cluster_selection_epsilon,
                allow_single_cluster=args.allow_single_cluster
            )
            unique_labels = set(labels) - {-1}
            n_clusters = len(unique_labels)
            method_name = 'hdbscan'
            
            clusters = {}
            for label in unique_labels:
                cluster_points = points[labels == label]
                clusters[label] = {
                    'points': cluster_points,
                    'point_count': len(cluster_points),
                    'centroid': np.mean(cluster_points, axis=0).tolist()
                }
        
        # ============================================================
        # STEP 4: BUILD current_clusters WITH VOXEL METADATA
        # ============================================================
        current_clusters = {}
        cluster_counter = 0
        
        for cluster_id, cluster_info in clusters.items():
            cluster_key = f"cluster_{cluster_counter:02d}"
            cluster_counter += 1
            
            # Extract voxel indices
            cluster_voxel_indices = []
            if 'voxel_indices' in cluster_info:
                cluster_voxel_indices = cluster_info['voxel_indices']
            elif 'voxel_count' in cluster_info and cluster_info['voxel_count'] > 0:
                if voxel_grid and 'points' in cluster_info:
                    cluster_points = cluster_info['points']
                    voxel_set = set()
                    for point in cluster_points:
                        voxel_idx = voxel_grid._point_to_cell(point)
                        if voxel_idx:
                            voxel_set.add(voxel_idx)
                    cluster_voxel_indices = list(voxel_set)
            
            # Build voxel_data dictionary
            voxel_data = {}
            if voxel_grid and cluster_voxel_indices:
                for voxel_idx in cluster_voxel_indices:
                    voxel_key = str(voxel_idx)
                    voxel_info = {}
                    
                    if hasattr(voxel_grid, 'get_complete_voxel_metadata'):
                        voxel_info = voxel_grid.get_complete_voxel_metadata(voxel_idx)
                    else:
                        if hasattr(voxel_grid, 'cell_metadata') and voxel_idx in voxel_grid.cell_metadata:
                            metadata = voxel_grid.cell_metadata[voxel_idx]
                            voxel_info['centroid'] = metadata.get('centroid', [0, 0, 0])
                        
                        if hasattr(voxel_grid, 'voxel_patterns') and voxel_idx in voxel_grid.voxel_patterns:
                            pattern_obj = voxel_grid.voxel_patterns[voxel_idx]
                            
                            if pattern_obj.y_plane_1:
                                y1_data = {
                                    'pattern_id': pattern_obj.y_plane_1.pattern_id.name if hasattr(pattern_obj.y_plane_1.pattern_id, 'name') else str(pattern_obj.y_plane_1.pattern_id),
                                    'point_count': pattern_obj.y_plane_1.point_count,
                                    'centroid_offset': pattern_obj.y_plane_1.centroid_offset.tolist() if hasattr(pattern_obj.y_plane_1.centroid_offset, 'tolist') else list(pattern_obj.y_plane_1.centroid_offset)
                                }
                                voxel_info['y_plane_1'] = y1_data
                            
                            if pattern_obj.y_plane_2:
                                y2_data = {
                                    'pattern_id': pattern_obj.y_plane_2.pattern_id.name if hasattr(pattern_obj.y_plane_2.pattern_id, 'name') else str(pattern_obj.y_plane_2.pattern_id),
                                    'point_count': pattern_obj.y_plane_2.point_count,
                                    'centroid_offset': pattern_obj.y_plane_2.centroid_offset.tolist() if hasattr(pattern_obj.y_plane_2.centroid_offset, 'tolist') else list(pattern_obj.y_plane_2.centroid_offset)
                                }
                                voxel_info['y_plane_2'] = y2_data
                    
                    if voxel_info and ('y_plane_1' in voxel_info or 'y_plane_2' in voxel_info):
                        voxel_data[voxel_key] = voxel_info

            # PREFER clustering.py's own voxel_data whenever it is LARGER than
            # the from-scratch grid rebuild above. The rebuild keeps a voxel only
            # if it has a y_plane_1/2 pattern, so it SILENTLY DROPS every
            # centroid-only voxel that clustering.py legitimately added —
            # forced-rescan recovery (occluded chair, no y_plane cells -> rebuild
            # 0) AND LIMB-RECLAIM / footprint regen (e.g. person reclaim "994
            # voxels" -> rebuild 155 on frames 61/66). Reusing clustering.py's
            # voxel_data (a superset that accepts centroid-only voxels) restores
            # the full shell. Using "> len" rather than "== 0" also catches the
            # PARTIAL-loss case; normal clusters whose cells all have y_plane are
            # unaffected (rebuild == clustering.py's count).
            _existing_vd = cluster_info.get('voxel_data')
            if isinstance(_existing_vd, dict) and len(_existing_vd) > len(voxel_data):
                voxel_data = _existing_vd

            # Calculate bbox
            centroid = cluster_info.get('centroid', [0, 0, 0])
            if isinstance(centroid, np.ndarray):
                centroid = centroid.tolist()
            
            bbox = cluster_info.get('bbox', None)
            if bbox is None:
                if 'points' in cluster_info and len(cluster_info['points']) > 0:
                    cluster_points = cluster_info['points']
                    bbox = {
                        'min': [float(cluster_points[:, 0].min()), 
                                float(cluster_points[:, 1].min()), 
                                float(cluster_points[:, 2].min())],
                        'max': [float(cluster_points[:, 0].max()), 
                                float(cluster_points[:, 1].max()), 
                                float(cluster_points[:, 2].max())]
                    }
                else:
                    x_span = cluster_info.get('x_span', 20.0)
                    y_span = cluster_info.get('y_span', 20.0)
                    z_span = cluster_info.get('z_span', 100.0)
                    padding_x = x_span * 0.2 if x_span > 0 else 20.0
                    padding_y = y_span * 0.2 if y_span > 0 else 20.0
                    padding_z = z_span * 0.3 if z_span > 0 else 30.0
                    
                    bbox = {
                        'min': [float(centroid[0] - x_span/2 - padding_x), 
                                float(centroid[1] - y_span/2 - padding_y),
                                float(centroid[2] - z_span/2 - padding_z)],
                        'max': [float(centroid[0] + x_span/2 + padding_x),
                                float(centroid[1] + y_span/2 + padding_y),
                                float(centroid[2] + z_span/2 + padding_z)]
                    }
            
            # Build cluster structure
            current_clusters[cluster_key] = {
                "original_label": cluster_id,
                "total_points": cluster_info.get('point_count', len(cluster_info.get('points', []))),
                "original_points": cluster_info.get('point_count', len(cluster_info.get('points', []))) - cluster_info.get('dummy_points', 0),
                "total_voxels": len(voxel_data),
                "centroid": centroid,
                "bbox": bbox,
                "voxel_data": voxel_data,
                "point_count": cluster_info.get('point_count', len(cluster_info.get('points', []))),
                "points": cluster_info.get('points'),
                "dummy_points": cluster_info.get('dummy_points', 0),
                "chord_voxels": cluster_info.get('chord_voxels', 0),
                "xy_slice_splits": cluster_info.get('xy_slice_splits', 0),
                "xy_slice_total": cluster_info.get('xy_slice_total', 0),
                "yz_slice_splits": cluster_info.get('yz_slice_splits', 0),
                "yz_slice_total": cluster_info.get('yz_slice_total', 0),
                "chord_angles": cluster_info.get('chord_angles', []),
                "xy_ratios": cluster_info.get('xy_ratios', []),
                "is_person_region": cluster_info.get('is_person_region', False),
                "xy_slice_details": cluster_info.get('xy_slice_details', []),
                "yz_slice_details": cluster_info.get('yz_slice_details', []),
                # BUG-3 FIX: Carry injection/recovery flags so STEP 5 artifact
                # filter can exempt these clusters from the 0-voxel check.
                "historical_lock_injected": cluster_info.get('historical_lock_injected', False),
                "tunnel_recovered": cluster_info.get('tunnel_recovered', False),
            }
            
            logger.info(f"{cluster_key}: {cluster_info.get('point_count', 0)} points, {len(voxel_data)} voxels with metadata")
        
        # ============================================================
        # STEP 5: FILTER ARTIFACTS
        # ============================================================
        filtered_clusters = {}
        artifact_count = 0
        
        for cluster_key, cluster_data in current_clusters.items():
            total_voxels = cluster_data.get('total_voxels', 0)
            total_points = cluster_data.get('total_points', 0)
            centroid = cluster_data.get('centroid', [0, 0, 0])
            
            if total_voxels <= 1 and total_points > 100:
                # BUG-3 FIX: Injected and tunnel-recovered clusters have 0 voxels
                # by construction -- they were never processed through the normal
                # region loop.  They must NOT be killed by this filter.
                if cluster_data.get('historical_lock_injected') or cluster_data.get('tunnel_recovered'):
                    filtered_clusters[cluster_key] = cluster_data
                    logger.info(f"[BUG-3 FIX] Exempting {cluster_key} from artifact filter "
                                f"(injected={cluster_data.get('historical_lock_injected')}, "
                                f"tunnel={cluster_data.get('tunnel_recovered')})")
                    continue
                logger.warning(f"ARTIFACT DETECTED: {cluster_key} has {total_voxels} voxel(s) but {total_points} points!")
                artifact_count += 1
                continue
            
            if (abs(centroid[0]) < 1 and abs(centroid[1]) < 1 and abs(centroid[2] - 250) < 1):
                logger.warning(f"SUSPICIOUS CLUSTER: {cluster_key} at origin-like position {centroid}")
                if total_voxels <= 2:
                    artifact_count += 1
                    continue
            
            if total_voxels > 0:
                points_per_voxel = total_points / total_voxels
                if points_per_voxel > 300:
                    logger.warning(f"ABNORMAL DENSITY: {cluster_key} has {points_per_voxel:.1f} points/voxel")
                    if total_voxels <= 2:
                        artifact_count += 1
                        continue
            
            filtered_clusters[cluster_key] = cluster_data
        
        if artifact_count > 0:
            logger.warning(f"Filtered out {artifact_count} artifact cluster(s)")
            current_clusters = filtered_clusters
            n_clusters = len(filtered_clusters)
        
        # ============================================================
        # STEP 6: EXTRACT POSE DATA (WITH SEGMENT ICCS INTEGRATION)
        # ============================================================
        
        def _calculate_body_orientation_safe(keypoints_3d: np.ndarray) -> Optional[Dict]:
            """Calculate body orientation from 5 core keypoints. SAFE version."""
            result = {
                'rotation_z': 0.0,
                'facing_direction': np.array([0, 1, 0]),
                'body_center': np.array([0, 0, 0]),
                'valid': False
            }
            
            if keypoints_3d is None:
                return result
            
            try:
                keypoints_3d = np.array(keypoints_3d)
                
                if keypoints_3d.shape[0] < 13:
                    return result
                
                NOSE = 0
                L_SHOULDER = 5
                R_SHOULDER = 6
                L_HIP = 11
                R_HIP = 12
                
                nose = keypoints_3d[NOSE]
                l_shoulder = keypoints_3d[L_SHOULDER]
                r_shoulder = keypoints_3d[R_SHOULDER]
                l_hip = keypoints_3d[L_HIP]
                r_hip = keypoints_3d[R_HIP]
                
                def is_valid_kp(kp):
                    return not (np.allclose(kp, 0) or np.any(np.isnan(kp)))
                
                if not all(is_valid_kp(kp) for kp in [l_shoulder, r_shoulder, l_hip, r_hip]):
                    return result
                
                shoulder_center = (l_shoulder + r_shoulder) / 2
                hip_center = (l_hip + r_hip) / 2
                body_center = (shoulder_center + hip_center) / 2
                
                result['body_center'] = body_center
                
                shoulder_vec = r_shoulder - l_shoulder
                facing_vec = np.array([-shoulder_vec[1], shoulder_vec[0], 0])
                
                facing_norm = np.linalg.norm(facing_vec[:2])
                if facing_norm > 1e-6:
                    facing_vec = facing_vec / facing_norm
                else:
                    return result
                
                if is_valid_kp(nose):
                    nose_dir = nose - shoulder_center
                    nose_dir[2] = 0
                    if np.dot(facing_vec[:2], nose_dir[:2]) < 0:
                        facing_vec = -facing_vec
                
                result['facing_direction'] = facing_vec
                rotation_z = np.degrees(np.arctan2(facing_vec[0], facing_vec[1]))
                result['rotation_z'] = rotation_z
                result['valid'] = True
                
                return result
                
            except Exception as e:
                logger.debug(f"Body orientation calculation failed: {e}")
                return result
        
        # Initialize pose variables
        pose_2d = None
        pose_3d = None
        person_cluster_uuid = None
        person_actual_uuid = None   # PERSON-LABEL-REMAP FIX: canonical person UUID
        body_orientation = None
        segment_iccs = None
        iccs = None
        # facing_info must be initialized here so it is ALWAYS defined,
        # regardless of whether pose detection succeeds or which code path runs.
        facing_info = {'facing': 'toward_camera', 'angle': 0.0, 'confidence': 0.5,
                       'stable': False, 'body_yaw_deg': 0.0, 'facing_24': 'toward_camera',
                       'fitting_path': 'frontal', 'torso_twist_deg': None,
                       'head_twist_deg': None, 'total_twist_deg': None,
                       'width_ratios': None, 'source': 'default'}
        
        # Initialize prediction tracking variables
        pose_was_predicted = False
        consecutive_rejections = 0
        segment_validation_applied = False
        
        # ============================================================
        # INITIALIZE CLUSTER COORDINATE SYSTEM (FIX FOR UNDEFINED cluster_coords)
        # ============================================================
        cluster_coords = None
        try:
            from cluster_coordinates import ClusterCoordinateSystem
            cluster_coords = ClusterCoordinateSystem()
            logger.debug("[OK] ClusterCoordinateSystem initialized")
        except ImportError as e:
            logger.warning(f"Could not import ClusterCoordinateSystem: {e}")
        except Exception as e:
            logger.warning(f"Could not initialize ClusterCoordinateSystem: {e}")
        
        # Store in state_bank for future use if not already there
        if state_bank is not None and cluster_coords is not None:
            if not hasattr(state_bank, 'cluster_coords') or state_bank.cluster_coords is None:
                state_bank.cluster_coords = cluster_coords
        
        if args.frames_dir:
            try:
                frame_path = point_cloud.find_matching_frame(file_path, args.frames_dir)
                
                if frame_path and os.path.exists(frame_path):
                    import cv2
                    
                    # Parse camera parameters from args — MUST be outside the init-once
                    # block so they are available every frame when passed to extract_3d_from_cop.
                    # Previously these were inside `if not hasattr(_mmpose_instance)` which
                    # ran only on frame 1, causing "referenced before assignment" on frames 2+.
                    camera_position = getattr(args, 'camera_position', [-47.0, 28.0, -20.0])
                    camera_target   = getattr(args, 'camera_target',   [-25.1, 123.8, -28.3])
                    focal_length    = getattr(args, 'focal_length',    27.5)
                    field_of_view   = getattr(args, 'field_of_view',   66.0)
                    if isinstance(camera_position, str):
                        camera_position = [float(x) for x in camera_position.split(',')]
                    if isinstance(camera_target, str):
                        camera_target = [float(x) for x in camera_target.split(',')]

                    # Initialize MMPose (once per process)
                    if not hasattr(process_single_frame, '_mmpose_instance'):
                        from mmpose_integration import MMPoseIntegration
                        device = 'cuda' if getattr(args, 'use_gpu', False) else 'cpu'
                        process_single_frame._mmpose_instance = MMPoseIntegration(
                            models_dir=getattr(args, 'mmpose_models_dir', 'C:/MMPose'),
                            device=device,
                            camera_position=camera_position,
                            camera_target=camera_target,
                            focal_length=focal_length,
                            field_of_view=field_of_view
                        )
                        logger.info(f"[OK] MMPose initialized with camera: pos={camera_position}, target={camera_target}, fov={field_of_view}")
                    
                    mmpose = process_single_frame._mmpose_instance
                    # Reuse frame image from early detection if available
                    frame_img = _early_frame_img if _early_frame_img is not None else cv2.imread(frame_path)
                    
                    if frame_img is not None:
                        # Get image dimensions for 2D-guided extraction
                        img_height, img_width = frame_img.shape[:2]
                        image_size = (img_width, img_height)
                        
                        # ==========================================
                        # 6.1: Detect 2D pose
                        # ==========================================
                        pose_results = mmpose.detect_2d_pose(frame_img, frame_num=frame_num)
                        # YOLO seg label channel: run ONCE per frame here and
                        # cache by frame_num; binding + mask overlay reuse it.
                        try:
                            from mmpose_integration import get_yolo_seg_result
                            get_yolo_seg_result(frame_img, frame_num)
                        except Exception as _yolo_e:
                            logger.debug(f"[YOLO] phase-1 detect skipped: {_yolo_e}")
                        
                        if pose_results:
                            pose_2d = pose_results
                            logger.info(f"[OK] Detected 2D pose for frame {frame_num}")
                            
                            # =================================================
                            # STABILIZE 2D POSE (THE ONLY PLACE!)
                            # =================================================
                            if not hasattr(process_single_frame, '_pose_2d_history'):
                                from temporal_consistency import Pose2DHistory
                                process_single_frame._pose_2d_history = Pose2DHistory(buffer_size=5)
                                logger.info("Initialized Pose2DHistory (ONLY stabilizer)")
                            
                            # FACING BUG FIX: also initialize _facing_history on the
                            # function object if it hasn't been set yet.  Previously
                            # _facing_history was only created as a local variable in the
                            # outer process_multiple_frames() scope and was never attached
                            # to process_single_frame — so hasattr(..., '_facing_history')
                            # was always False and detect_facing_from_2d_pose was never
                            # called, returning the default {'facing':'front'} every frame.
                            if not hasattr(process_single_frame, '_facing_history'):
                                from temporal_consistency import FacingHistory
                                process_single_frame._facing_history = FacingHistory(required_consistent_frames=5)
                                logger.info("Initialized FacingHistory on process_single_frame")
                            
                            pose_2d_history = process_single_frame._pose_2d_history
                            
                            # Get raw keypoints
                            raw_kps = pose_2d.get('keypoints') if isinstance(pose_2d, dict) else pose_2d
                            
                            if raw_kps is not None:
                                raw_kps = np.array(raw_kps)
                                
                                # STABILIZE
                                stable_kps = pose_2d_history.get_stable_pose(raw_kps)
                                
                                # STORE STABLE IN HISTORY (not raw!)
                                pose_2d_history.add(frame_num, stable_kps)
                                
                                # UPDATE pose_2d with stabilized keypoints
                                if isinstance(pose_2d, dict):
                                    pose_2d['keypoints'] = stable_kps
                                    pose_2d['keypoints_raw'] = raw_kps  # Keep raw for debug
                                else:
                                    pose_2d = {'keypoints': stable_kps, 'keypoints_raw': raw_kps}
                                
                                logger.info(f"[OK] Stabilized 2D pose for frame {frame_num}")
                            # =================================================
                            # ==========================================
                            # 6.2: Find person cluster
                            # ==========================================
                            if current_clusters and pose_2d:
                                person_cluster_uuid = mmpose.find_person_cluster(
                                    current_clusters,
                                    pose_2d,
                                    points,
                                    state_bank=state_bank,
                                    frame_num=frame_num
                                )
                                
                                if person_cluster_uuid:
                                    logger.info(f"[OK] Person identified in cluster {person_cluster_uuid[:8] if len(str(person_cluster_uuid)) > 8 else person_cluster_uuid} (frame {frame_num})")
                                    
                                    cluster_data = current_clusters[person_cluster_uuid]
                                    cluster_bbox = cluster_data.get('bbox')
                                    cluster_points = cluster_data.get('points', points)
                                    cluster_voxel_metadata = cluster_data.get('voxel_data', {})

                                    # PERSON-LABEL-REMAP FIX: Capture the PERSON's
                                    # actual UUID alongside the label.  Downstream
                                    # code (temporal processing at ~line 2894)
                                    # relabels clusters by centroid-X sort, so the
                                    # label 'cluster_01' that MMPose returned here
                                    # can end up pointing at the CHAIR after
                                    # renumbering (frame 45: person was cluster_01
                                    # pre-rename, but became cluster_00 post-rename
                                    # because its smaller X sorts first — while the
                                    # stale 'cluster_01' in person_cluster_uuid now
                                    # indexes the chair).  We pin the UUID here and
                                    # re-derive the label after rename; see fix
                                    # block after the cluster-relabel loop.
                                    person_actual_uuid = cluster_data.get(
                                        'uuid') or cluster_data.get('cluster_uuid')

                                    # MIDAS-JUMP GUARD: persist person cluster
                                    # Z-max so the NEXT frame's person_context
                                    # can detect a sudden drop (MiDaS jump).
                                    # Note: this is the POST-synth bbox.max[2]
                                    # (head-synth already ran inside the
                                    # clustering call).  On a normal frame
                                    # that's ~+10-12cm (head crown); a
                                    # MiDaS-jump frame will have pre-synth
                                    # z_max ~-28cm — delta > 15cm from the
                                    # previous stored value, which the guard
                                    # in clustering.py uses as a trigger.
                                    if cluster_bbox is not None:
                                        _bbox_max_prev = cluster_bbox.get('max', [0, 0, 0])
                                        try:
                                            process_single_frame._prev_person_z_max = float(_bbox_max_prev[2])
                                        except (TypeError, ValueError, IndexError):
                                            pass
                                    
                                    # TEMPORAL HEIGHT ANCHOR: submit this
                                    # frame's cluster z-span every frame (no
                                    # longer once-only). state_bank.
                                    # lock_person_height accumulates the running
                                    # max over a settling window so the locked
                                    # stature converges to the tallest clean
                                    # capture instead of freezing on a short
                                    # first frame.
                                    if (state_bank is not None and cluster_bbox is not None
                                            and hasattr(state_bank, 'lock_person_height')):
                                        _bbox_min = cluster_bbox.get('min', [0,0,0])
                                        _bbox_max = cluster_bbox.get('max', [0,0,0])
                                        _z_span = float(_bbox_max[2]) - float(_bbox_min[2])
                                        if _z_span > 50.0:
                                            from anatomical_skeleton import CLUSTER_HEAD_CORRECTION_CM
                                            _person_h = _z_span + CLUSTER_HEAD_CORRECTION_CM
                                            state_bank.lock_person_height(_person_h)
                                    
                                    # ==========================================
                                    # 6.3: Extract 3D pose from person cluster
                                    # ==========================================
                                    if person_cluster_uuid and person_cluster_uuid in current_clusters:
                                        cluster_info = current_clusters[person_cluster_uuid]
                                        cluster_points = cluster_info.get('points')
                                        cluster_bbox = cluster_info.get('bbox')
                                        cluster_voxel_metadata = cluster_info.get('voxel_data', {})
                                        
                                        if cluster_points is not None and len(cluster_points) > 0:
                                            
                                            # =============================================================
                                            # STEP A: Get previous pose and constraints from frame_buffer
                                            # =============================================================
                                            previous_pose_3d = None
                                            anatomical_constraints = None
                                            
                                            if frame_buffer is not None:
                                                previous_pose_3d = frame_buffer.get_previous_pose_3d(frame_num)
                                                anatomical_constraints = frame_buffer.get_anatomical_constraints()
                                                
                                                if previous_pose_3d is not None:
                                                    logger.debug(f"Frame {frame_num}: Using previous pose for constraints")
                                                if anatomical_constraints is not None:
                                                    logger.debug(f"Frame {frame_num}: Using accumulated constraints "
                                                                f"(H={anatomical_constraints.get('H', 0):.1f}, "
                                                                f"hip={anatomical_constraints.get('hip_width', 0):.1f})")
                                            
                                            # =============================================================
                                            # STEP B: Detect and stabilize facing BEFORE extraction (NEW!)
                                            # =============================================================
                                            facing_info = {'facing': 'toward_camera', 'angle': 0.0, 'confidence': 0.5, 'stable': False,
                                                            'body_yaw_deg': 0.0, 'facing_24': 'toward_camera', 'fitting_path': 'frontal',
                                                            'torso_twist_deg': None, 'head_twist_deg': None, 'total_twist_deg': None,
                                                            'width_ratios': None, 'source': 'default'}
                                            
                                            if hasattr(process_single_frame, '_facing_history') and pose_2d is not None:
                                                facing_history = process_single_frame._facing_history
                                                kps = pose_2d.get('keypoints') if isinstance(pose_2d, dict) else pose_2d
                                                
                                                if kps is not None:
                                                    try:
                                                        from mmpose_integration import detect_facing_from_2d_pose
                                                        raw_facing = detect_facing_from_2d_pose(kps)
                                                        facing_info = facing_history.update(raw_facing)
                                                        
                                                        logger.info(f"Frame {frame_num}: Facing={facing_info['facing']}, "
                                                                  f"angle={facing_info.get('angle', 0):.1f}deg, "
                                                                  f"stable={facing_info.get('stable', False)}")
                                                    except Exception as e:
                                                        logger.warning(f"Facing detection failed: {e}")
                                            
                                            # =============================================================
                                            # STEP C: Extract 3D pose WITH stabilized facing_info
                                            # =============================================================
                                            # FIX A: camera_params is NOT a parameter of
                                            # extract_3d_from_cop() — camera values are
                                            # already stored on self.camera_position etc.
                                            # at MMPoseIntegration.__init__ time (line 1275).
                                            # Passing camera_params here caused a TypeError
                                            # every frame, making pose_3d always None.
                                            pose_3d = mmpose.extract_3d_from_cop(
                                                cluster_points,
                                                cluster_bbox=cluster_bbox,
                                                cluster_points_only=True,
                                                voxel_metadata=cluster_voxel_metadata,
                                                frame_num=frame_num,
                                                pose_2d=pose_2d,
                                                image_size=image_size,
                                                previous_pose_3d=previous_pose_3d,
                                                anatomical_constraints=anatomical_constraints,
                                                facing_info=facing_info,
                                                person_uuid=(person_actual_uuid or person_cluster_uuid),  # ISSUE-1 bone lock (canonical id, relabel-stable)
                                            )
                                            
                                            if pose_3d is not None:
                                                logger.info(f"[OK] Extracted 3D pose for frame {frame_num}")
                                        # ==========================================
                                        # 6.3.2: Establish ICCS with segment capsules
                                        # Uses cluster_coords (now properly defined!)
                                        # ==========================================
                                        # Get cluster_coords from state_bank or local
                                        cluster_coords_ref = None
                                        if state_bank is not None and hasattr(state_bank, 'cluster_coords') and state_bank.cluster_coords is not None:
                                            cluster_coords_ref = state_bank.cluster_coords
                                        elif cluster_coords is not None:
                                            cluster_coords_ref = cluster_coords
                                        
                                        if cluster_coords_ref is not None:
                                            try:
                                                iccs = cluster_coords_ref.establish_iccs_from_mmpose(
                                                    pose_3d,
                                                    facing_info=facing_info
                                                )
                                            except Exception as e:
                                                logger.warning(f"ICCS establishment failed: {e}")
                                                iccs = None
                                        
                                        if iccs is not None and 'segment_iccs' in iccs:
                                            segment_iccs = iccs.get('segment_iccs')
                                            logger.info(f"[OK] ICCS established with {len(segment_iccs)} segment capsules")
                                            
                                            # Load segment capsules into stabilizer for validation
                                            if hasattr(mmpose, 'keypoint_stabilizer') and hasattr(mmpose.keypoint_stabilizer, 'load_segment_iccs'):
                                                mmpose.keypoint_stabilizer.load_segment_iccs(segment_iccs)
                                            
                                            # Re-stabilize pose with segment validation
                                            cluster_info_for_stabilizer = {
                                                'bbox': cluster_bbox,
                                                'voxel_data': cluster_voxel_metadata,
                                                'centroid': cluster_data.get('centroid')
                                            }
                                            
                                            if hasattr(mmpose, 'keypoint_stabilizer') and hasattr(mmpose.keypoint_stabilizer, 'stabilize_3d'):
                                                pose_3d_validated = mmpose.keypoint_stabilizer.stabilize_3d(
                                                    pose_3d, 
                                                    frame_num,
                                                    cluster_info=cluster_info_for_stabilizer,
                                                    cluster_uuid=person_cluster_uuid
                                                )
                                                
                                                if pose_3d_validated is not None:
                                                    pose_3d = pose_3d_validated
                                                    segment_validation_applied = True
                                                    logger.info(f"[OK] Pose validated against segment capsules")
                                                else:
                                                    # Stabilization rejected - use prediction
                                                    if hasattr(mmpose.keypoint_stabilizer, 'predict_pose_from_history'):
                                                        predicted = mmpose.keypoint_stabilizer.predict_pose_from_history(frame_num)
                                                        if predicted is not None:
                                                            pose_3d = predicted
                                                            pose_was_predicted = True
                                                            logger.warning(f"Frame {frame_num}: Segment validation failed, using predicted pose")
                                                        else:
                                                            if hasattr(mmpose.keypoint_stabilizer, 'get_last_valid_pose'):
                                                                last_valid = mmpose.keypoint_stabilizer.get_last_valid_pose()
                                                                if last_valid is not None:
                                                                    pose_3d = last_valid
                                                                    logger.warning(f"Frame {frame_num}: Using last valid pose as fallback")
                                        
                                        # ============================================
                                        # FEATURE A: CAPSULE-GUIDED CLUSTER FILL (2nd pass, in-frame)
                                        # The person cluster is a thin 2.5D front shell; the mannequin
                                        # capsules just fitted above ARE the true body volume.  Fill the
                                        # 2cm cells inside those capsules into the person's voxel_data so
                                        # the rendered cluster becomes a complete body (inside capsule =
                                        # body).  Centroid-only -> render via reconstruct STEP 4c; tagged
                                        # synthetic_source='capsule_fill' so they never feed next frame's
                                        # shell-fit.  Runs AFTER ICCS/capsules are established.
                                        # ============================================
                                        # DISABLED 2026-06-23 21:55: the capsule fill made the person a
                                        # solid monolith that did NOT match the cluster (the mannequin
                                        # capsules and the cluster shell are different skeletons). Flip
                                        # _FEATURE_A_CAPSULE_FILL back to True to re-enable.
                                        _FEATURE_A_CAPSULE_FILL = False
                                        try:
                                            _ks_fill = getattr(mmpose, 'keypoint_stabilizer', None)
                                            if (_FEATURE_A_CAPSULE_FILL
                                                    and _ks_fill is not None
                                                    and hasattr(_ks_fill, 'fill_voxels_in_capsules')
                                                    and person_cluster_uuid in current_clusters):
                                                _cf_res = float(getattr(voxel_grid, 'resolution', 2.0)) if voxel_grid is not None else 2.0
                                                _cf_fill = _ks_fill.fill_voxels_in_capsules(voxel_size=_cf_res)
                                                if _cf_fill:
                                                    _cf_pc = current_clusters[person_cluster_uuid]
                                                    _cf_vd = _cf_pc.get('voxel_data') or {}
                                                    _cf_before = len(_cf_vd)
                                                    _cf_vd.update(_cf_fill)
                                                    _cf_pc['voxel_data'] = _cf_vd
                                                    _cf_pc['total_voxels'] = len(_cf_vd)
                                                    _cf_pc['capsule_filled'] = len(_cf_fill)
                                                    logger.info(f"[CAPSULE-FILL] Frame {frame_num}: "
                                                                f"+{len(_cf_fill)} capsule voxels into person "
                                                                f"(shell {_cf_before} -> {len(_cf_vd)} voxels)")
                                        except Exception as _cf_e:
                                            logger.warning(f"[CAPSULE-FILL] Frame {frame_num} skipped: {_cf_e}")

                                        # ===================== POST-ICCS FACING REFINEMENT =====================
                                        # Re-resolve facing using ICCS 3D shoulder-line angle when it
                                        # provides a BETTER signal than Step B's confidence-only detection.
                                        #
                                        # CRITICAL RULE: POST-ICCS must NOT override a high-confidence
                                        # Step B result with a lower-confidence ICCS result.
                                        # Evidence: frame 42 — MMPose correctly detects front (conf=0.91)
                                        # but ICCS overrode with away_from_camera (conf=0.55), killing
                                        # rotation detection for the entire toward-camera segment.
                                        #
                                        # POLICY:
                                        #   - If Step B conf >= 0.75 and ICCS conf < Step B conf → KEEP Step B
                                        #   - If ICCS and Step B AGREE on hemisphere → use ICCS (better angle)
                                        #   - If they DISAGREE → use the higher confidence one
                                        # ======================================================================
                                        if iccs and 'rotation' in iccs:
                                            _iccs_rot = iccs['rotation']
                                            _iccs_angle = _iccs_rot.get('raw_angle', _iccs_rot.get('angle'))
                                            if _iccs_angle is not None:
                                                try:
                                                    from mmpose_integration import detect_facing_from_2d_pose as _detect_facing
                                                    _kps_for_refine = None
                                                    if pose_2d is not None:
                                                        _kps_for_refine = pose_2d.get('keypoints') if isinstance(pose_2d, dict) else pose_2d
                                                    if _kps_for_refine is not None:
                                                        # Get geo_orientation from previous frame's blanket
                                                        _geo_orient = None
                                                        if state_bank is not None and person_cluster_uuid:
                                                            _prev_blanket = state_bank.get_previous_blanket(
                                                                person_cluster_uuid) if hasattr(
                                                                    state_bank, 'get_previous_blanket') else None
                                                            if _prev_blanket and isinstance(_prev_blanket, dict):
                                                                _geo_orient = _prev_blanket.get('geo_orientation')
                                                        _refined = _detect_facing(
                                                            _kps_for_refine,
                                                            iccs_rotation_angle=float(_iccs_angle),
                                                            geo_orientation=_geo_orient
                                                        )
                                                        # =========================================
                                                        # USE RAW RESULT DIRECTLY.
                                                        # No FacingHistory buffer (was destroying
                                                        # correct transitions).
                                                        # No confidence gate (was keeping wrong
                                                        # buffer-smoothed Step B over correct ICCS).
                                                        # trio+lateral IS the per-frame ground truth.
                                                        # =========================================
                                                        facing_info = _refined
                                                        logger.info(f"Frame {frame_num}: POST-ICCS facing refined -> "
                                                                   f"{facing_info.get('facing_24', facing_info.get('facing'))} "
                                                                   f"(yaw={facing_info.get('body_yaw_deg', '?')}, "
                                                                   f"source={facing_info.get('source', '?')})")
                                                except Exception as e:
                                                    logger.debug(f"Post-ICCS facing refinement failed: {e}")
                                        # ======================================================================

                                        # ── POSE-DB ↔ FACING CROSS-CHECK ──
                                        if (_early_pose_strategy is not None
                                                and _early_pose_strategy.get('ok')
                                                and facing_info is not None):
                                            _pdb_axis = _early_pose_strategy.get('depth_axis', '')
                                            _cur_facing = facing_info.get('facing_24',
                                                            facing_info.get('facing', ''))
                                            _frontal_facings = ('toward_camera', 'away_from_camera',
                                                                'toward_camera_slight_left',
                                                                'toward_camera_slight_right',
                                                                'away_from_camera_slight_left',
                                                                'away_from_camera_slight_right')
                                            if _pdb_axis == 'LATERAL' and _cur_facing in _frontal_facings:
                                                _lat_facing = _utils_pose.derive_lateral_facing(pose_2d)
                                                if _lat_facing is not None:
                                                    _old_yaw = facing_info.get('body_yaw_deg', 0)
                                                    _new_yaw = 90.0 if _lat_facing == 'side_left' else 270.0
                                                    logger.warning(
                                                        f"[FACING-OVERRIDE] Frame {frame_num}: "
                                                        f"POSE-DB=LATERAL but facing={_cur_facing} - "
                                                        f"overriding to {_lat_facing} (yaw {_old_yaw:.1f}->{_new_yaw:.1f}deg)")
                                                    facing_info['facing'] = _lat_facing
                                                    facing_info['facing_24'] = _lat_facing
                                                    facing_info['body_yaw_deg'] = _new_yaw
                                                    facing_info['fitting_path'] = _lat_facing.replace('side_', '')
                                                    facing_info['source'] = 'pose_db_override'

                                        # ==============================================================
                                        # FRAME-TO-FRAME YAW COMPARISON
                                        # Compare current body_yaw with previous frame's body_yaw.
                                        # Sudden jumps > 45° indicate a real turn OR a detection error.
                                        # ==============================================================
                                        _cur_yaw = facing_info.get('body_yaw_deg') if facing_info else None
                                        if _cur_yaw is not None:
                                            if not hasattr(process_single_frame, '_prev_body_yaw'):
                                                process_single_frame._prev_body_yaw = None
                                                process_single_frame._prev_facing = None
                                            _prev_yaw = process_single_frame._prev_body_yaw
                                            _prev_face = process_single_frame._prev_facing
                                            if _prev_yaw is not None:
                                                _yaw_diff = abs(float(_cur_yaw) - float(_prev_yaw))
                                                if _yaw_diff > 180:
                                                    _yaw_diff = 360 - _yaw_diff
                                                _cur_face = facing_info.get('facing_24', facing_info.get('facing', '?'))
                                                if _yaw_diff > 45:
                                                    logger.warning(
                                                        f"Frame {frame_num}: [YAW-JUMP] "
                                                        f"deltayaw={_yaw_diff:.1f}deg "
                                                        f"(prev={_prev_yaw:.1f}deg/{_prev_face} -> "
                                                        f"cur={float(_cur_yaw):.1f}deg/{_cur_face}) "
                                                        f"LARGE ROTATION or detection error")
                                                else:
                                                    logger.info(
                                                        f"Frame {frame_num}: [YAW-CONT] "
                                                        f"deltayaw={_yaw_diff:.1f}deg "
                                                        f"(prev={_prev_yaw:.1f}deg -> cur={float(_cur_yaw):.1f}deg) "
                                                        f"- smooth continuity")
                                            process_single_frame._prev_body_yaw = float(_cur_yaw)
                                            process_single_frame._prev_facing = facing_info.get('facing_24', '?')
                                            # SV-FIX: also persist fitting_path for next frame's person_context
                                            process_single_frame._prev_fitting_path = facing_info.get('fitting_path', 'frontal')
                                        # ==========================================
                                        # 6.4: Calculate body orientation
                                        # ==========================================
                                        body_orientation = _calculate_body_orientation_safe(pose_3d)
                                        
                                        if body_orientation and body_orientation.get('valid'):
                                            logger.info(f"[OK] Body orientation: rotation_z={body_orientation['rotation_z']:.1f}deg")
                                    else:
                                        logger.warning(f"Frame {frame_num}: No 3D pose available "
                                                      f"(extraction and prediction both failed)")
                                        body_orientation = None
            
            except Exception as e:
                logger.warning(f"Pose extraction failed (frame {frame_num}): {e}")
                import traceback
                logger.debug(traceback.format_exc())
        
        # ============================================================
        # STEP 7: ASSIGN PERMANENT UUIDs - CENTROID-FIRST APPROACH
        # ============================================================
        if state_bank is not None and frame_num is not None:
            logger.info("=" * 60)
            logger.info(f"FRAME {frame_num}: ASSIGNING PERMANENT UUIDs")
            logger.info("=" * 60)
            
            # ============================================================
            # INITIALIZE TRACKING STRUCTURES
            # ============================================================
            if not hasattr(state_bank, 'permanent_uuid_registry'):
                state_bank.permanent_uuid_registry = {}
                state_bank.last_frame_assignments = {}
                state_bank.uuid_created_frame = {}
                state_bank.merge_prevention_pairs = set()
            # ISSUE-1 Defect-2: per-UUID temporal body_yaw lock state.
            #   last_good_body_yaw[uuid]  -> last accepted (reliable) yaw, holds
            #                                the ICCS frame through bad frames.
            #   yaw_force_bootstrap       -> uuids that must re-seed from the
            #                                injected chord/POSE-DB value next
            #                                frame (e.g. set by occlusion recovery).
            if not hasattr(state_bank, 'last_good_body_yaw'):
                state_bank.last_good_body_yaw = {}
                state_bank.yaw_force_bootstrap = set()
            # ISSUE-3 anti-merge: locked rigid-object (chair/table/...) OBB
            # primitives, keyed by object_inventory uuid.  rigid_claim_enabled
            # gates the actual voxel-claim step (OFF by default -> the whole
            # feature is a no-op until smoke-tested, because pulling voxels out of
            # the cloud before clustering can interact with the historical-lock /
            # tunnel-recovery path).
            if not hasattr(state_bank, 'rigid_primitives'):
                state_bank.rigid_primitives = {}
                state_bank.rigid_claim_enabled = False
            
            # ============================================================
            # FRAME 1: ESTABLISH INITIAL IDENTITIES
            # ============================================================
            if frame_num == 1 or len(state_bank.permanent_uuid_registry) == 0:
                logger.warning("ESTABLISHING INITIAL IDENTITIES FOR FRAME 1")
                
                sorted_current = sorted(
                    current_clusters.items(),
                    key=lambda x: x[1].get('total_points', 0),
                    reverse=True
                )
                
                for idx, (cluster_key, cluster_data) in enumerate(sorted_current):
                    permanent_uuid = str(uuid_module.uuid4())
                    
                    state_bank.permanent_uuid_registry[permanent_uuid] = {
                        'position_index': idx,
                        'initial_points': cluster_data.get('total_points', 0),
                        'initial_centroid': cluster_data.get('centroid', [0, 0, 0]),
                        'typical_size': cluster_data.get('total_points', 0),
                        'created_frame': frame_num
                    }
                    
                    cluster_data['uuid'] = permanent_uuid
                    state_bank.uuid_created_frame[permanent_uuid] = frame_num
                    
                    state_bank.last_frame_assignments[permanent_uuid] = {
                        'cluster_key': cluster_key,
                        'centroid': cluster_data.get('centroid', [0, 0, 0]),
                        'total_points': cluster_data.get('total_points', 0)
                    }
                    
                    logger.warning(f"PERMANENT UUID {idx}: {permanent_uuid[:8]}... "
                                 f"({cluster_data.get('total_points', 0)} points)")
                
                all_uuids = list(state_bank.permanent_uuid_registry.keys())
                for i, uuid1 in enumerate(all_uuids):
                    for uuid2 in all_uuids[i+1:]:
                        state_bank.merge_prevention_pairs.add(tuple(sorted([uuid1, uuid2])))
                        logger.info(f"[OK] Merge prevention: {uuid1[:8]} <-> {uuid2[:8]}")

                # ============================================================
                # ESTABLISH PRIMARY SUBJECT UUID (PERSON IDENTITY ANCHOR)
                # This UUID is carried for the entire video. All subsequent
                # frames reuse it regardless of centroid matching outcome.
                # Priority: MMPose-identified cluster > tallest cluster (z_span).
                # ============================================================
                _person_perm_uuid = None
                _person_z_span = 0

                # Priority 1: use person_cluster_uuid already identified by MMPose
                if person_cluster_uuid and person_cluster_uuid in current_clusters:
                    _cd = current_clusters[person_cluster_uuid]
                    _person_perm_uuid = _cd.get('uuid') or _cd.get('cluster_uuid')
                    bbox = _cd.get('bbox', {})
                    if 'min' in bbox and 'max' in bbox:
                        _person_z_span = bbox['max'][2] - bbox['min'][2]

                # Priority 2: tallest cluster by z_span (person is taller than chairs/objects)
                if not _person_perm_uuid:
                    for _ck, _cd in current_clusters.items():
                        _bbox = _cd.get('bbox', {})
                        if 'min' in _bbox and 'max' in _bbox:
                            _zs = _bbox['max'][2] - _bbox['min'][2]
                            if _zs > _person_z_span and _zs > 100:
                                _person_z_span = _zs
                                _person_perm_uuid = _cd.get('uuid') or _cd.get('cluster_uuid') or _ck

                if _person_perm_uuid:
                    state_bank.primary_subject_uuid = _person_perm_uuid
                    # Mark person cluster as NEVER-EXPIRE
                    if hasattr(state_bank, 'mark_as_person'):
                        state_bank.mark_as_person(_person_perm_uuid)
                    logger.warning(
                        f"[PERSON] Primary subject UUID LOCKED: {_person_perm_uuid[:8]} "
                        f"(height~{_person_z_span:.1f}cm). This UUID will persist for "
                        f"the entire video regardless of tracking gaps."
                    )
                else:
                    logger.error("[PERSON] Could not identify person cluster in frame 1 - "
                                 "primary_subject_uuid not set. Skeleton persistence will degrade.")

            # ============================================================
            # SUBSEQUENT FRAMES: CENTROID-BASED MATCHING
            # ============================================================
            else:
                from scipy.optimize import linear_sum_assignment

                current_list = list(current_clusters.items())
                n_current = len(current_list)

                previous_uuids = list(state_bank.last_frame_assignments.keys())
                n_previous = len(previous_uuids)

                logger.info(f"Matching {n_current} current clusters to {n_previous} previous UUIDs")

                assigned_current = set()
                assigned_previous = set()

                # ============================================================
                # PRE-PASS: STICKY PERSON MATCH
                # FIX B: Match ONLY clusters taller than 100 cm (z_span).
                # A person is ~150-190 cm tall; a chair is ~40-60 cm.
                # The old code had a fallback accepting any cluster within
                # 150 cm regardless of height — this caused the chair to be
                # matched as the person when the person centroid drifted.
                # That fallback is intentionally REMOVED. If no tall cluster
                # is found within 200 cm, the person is occluded this frame
                # and the last known centroid is frozen for next-frame matching.
                # ============================================================
                _primary_uuid = getattr(state_bank, 'primary_subject_uuid', None)
                if _primary_uuid and _primary_uuid in state_bank.last_frame_assignments:
                    _prev_data = state_bank.last_frame_assignments[_primary_uuid]
                    _prev_centroid = np.array(_prev_data['centroid'])
                    _primary_j = previous_uuids.index(_primary_uuid) if _primary_uuid in previous_uuids else -1

                    _best_i = None
                    _best_dist = 9999.0
                    _best_zspan = 0.0

                    for _i, (_ck, _cd) in enumerate(current_list):
                        _cc = np.array(_cd.get('centroid', [0, 0, 0]))
                        _dist = float(np.linalg.norm(_cc - _prev_centroid))
                        _bbox = _cd.get('bbox', {})
                        _zs = (_bbox['max'][2] - _bbox['min'][2]
                               if 'min' in _bbox and 'max' in _bbox else 0.0)

                        # ONLY match clusters tall enough to be a person (>100 cm z_span).
                        # Do NOT fall back to short clusters — that is how chairs get matched.
                        if _zs > 100.0 and _dist < 200.0:
                            if _dist < _best_dist:
                                _best_dist = _dist
                                _best_i = _i
                                _best_zspan = _zs

                    if _best_i is not None:
                        _match_key, _match_data = current_list[_best_i]
                        _match_data['uuid'] = _primary_uuid
                        # PERSON-LABEL-REMAP FIX: Sticky match is the source of
                        # truth for the person's canonical UUID — keep
                        # person_actual_uuid in sync so the post-rename remap
                        # below can find the person by UUID.
                        person_actual_uuid = _primary_uuid
                        assigned_current.add(_best_i)
                        if _primary_j >= 0:
                            assigned_previous.add(_primary_j)
                        state_bank.last_frame_assignments[_primary_uuid] = {
                            'cluster_key': _match_key,
                            'centroid': _match_data.get('centroid', [0, 0, 0]),
                            'total_points': _match_data.get('total_points', 0)
                        }
                        logger.warning(
                            f"[PERSON] Sticky match: {_match_key} -> "
                            f"{_primary_uuid[:8]} (dist={_best_dist:.1f}cm, "
                            f"z_span~{_best_zspan:.1f}cm)"
                        )
                    else:
                        # Person cluster genuinely not visible this frame.
                        # Do NOT delete from last_frame_assignments — freeze
                        # the last known centroid so the next frame can still
                        # match. The skeleton carries over via temporal state.
                        logger.warning(
                            f"[PERSON] Primary subject {_primary_uuid[:8]} not visible "
                            f"in frame {frame_num}. Centroid FROZEN for next-frame match."
                        )
                        if _primary_j >= 0:
                            assigned_previous.add(_primary_j)  # prevent re-assignment below

                # ============================================================
                # Build cost matrix (for non-person clusters)
                # ============================================================
                cost_matrix = np.full((n_current, n_previous), 1000.0)
                
                for i, (cluster_key, cluster_data) in enumerate(current_list):
                    current_centroid = np.array(cluster_data.get('centroid', [0, 0, 0]))
                    current_points = cluster_data.get('total_points', 0)
                    
                    for j, prev_uuid in enumerate(previous_uuids):
                        prev_data = state_bank.last_frame_assignments[prev_uuid]
                        prev_centroid = np.array(prev_data['centroid'])
                        prev_points = prev_data['total_points']
                        
                        spatial_distance = np.linalg.norm(current_centroid - prev_centroid)
                        
                        if max(current_points, prev_points) > 0:
                            size_ratio = min(current_points, prev_points) / max(current_points, prev_points)
                        else:
                            size_ratio = 0
                        
                        if spatial_distance < 100.0:
                            cost = spatial_distance * 0.8 + (1.0 - size_ratio) * 20.0
                        else:
                            cost = 1000.0
                        
                        cost_matrix[i, j] = cost
                
                # Solve optimal assignment
                row_ind, col_ind = linear_sum_assignment(cost_matrix)

                for i, j in zip(row_ind, col_ind):
                    if i < n_current and j < n_previous:
                        # Skip if already claimed by the sticky person match
                        if i in assigned_current or j in assigned_previous:
                            continue
                        cluster_key, cluster_data = current_list[i]
                        assigned_uuid = previous_uuids[j]
                        cost = cost_matrix[i, j]
                        
                        if cost < 200.0:
                            cluster_data['uuid'] = assigned_uuid
                            assigned_current.add(i)
                            assigned_previous.add(j)
                            
                            state_bank.last_frame_assignments[assigned_uuid] = {
                                'cluster_key': cluster_key,
                                'centroid': cluster_data.get('centroid', [0, 0, 0]),
                                'total_points': cluster_data.get('total_points', 0)
                            }
                            
                            logger.info(f"[OK] {cluster_key} MATCHED to {assigned_uuid[:8]}... "
                                      f"(cost={cost:.1f}, {cluster_data.get('total_points', 0)} pts)")
                        else:
                            logger.warning(f" {cluster_key} rejected match to {assigned_uuid[:8]}... "
                                         f"(cost={cost:.1f} too high)")
                
                # Handle unmatched clusters
                for i, (cluster_key, cluster_data) in enumerate(current_list):
                    if i not in assigned_current and 'uuid' not in cluster_data:
                        # --------------------------------------------------------
                        # PERSON RECOVERY: If this unmatched cluster looks like a
                        # person (z_span > 100 cm) and primary_subject_uuid is
                        # already established, reuse the person UUID instead of
                        # minting a new one. This handles the case where the
                        # sticky match above found no candidate because the cluster
                        # had an unusual centroid (e.g. split by noise) but is now
                        # the only tall object in the frame.
                        # --------------------------------------------------------
                        _primary_uuid = getattr(state_bank, 'primary_subject_uuid', None)
                        _bbox = cluster_data.get('bbox', {})
                        _zs = (_bbox['max'][2] - _bbox['min'][2]
                               if 'min' in _bbox and 'max' in _bbox else 0.0)
                        _is_person_height = _zs > 100.0

                        if (_primary_uuid
                                and _is_person_height
                                and _primary_uuid not in [cd.get('uuid') for _, cd in current_list]):
                            # Person UUID not yet claimed this frame → recover it
                            cluster_data['uuid'] = _primary_uuid
                            state_bank.last_frame_assignments[_primary_uuid] = {
                                'cluster_key': cluster_key,
                                'centroid': cluster_data.get('centroid', [0, 0, 0]),
                                'total_points': cluster_data.get('total_points', 0)
                            }
                            logger.warning(
                                f"[PERSON] UUID recovered for {cluster_key} -> "
                                f"{_primary_uuid[:8]} (z_span={_zs:.1f}cm, "
                                f"sticky match missed - recovered at unmatched stage)"
                            )
                            continue

                        # Not the person (or person already placed) → new object
                        new_uuid = str(uuid_module.uuid4())
                        cluster_data['uuid'] = new_uuid
                        
                        state_bank.permanent_uuid_registry[new_uuid] = {
                            'position_index': len(state_bank.permanent_uuid_registry),
                            'initial_points': cluster_data.get('total_points', 0),
                            'initial_centroid': cluster_data.get('centroid', [0, 0, 0]),
                            'typical_size': cluster_data.get('total_points', 0),
                            'created_frame': frame_num
                        }
                        
                        state_bank.last_frame_assignments[new_uuid] = {
                            'cluster_key': cluster_key,
                            'centroid': cluster_data.get('centroid', [0, 0, 0]),
                            'total_points': cluster_data.get('total_points', 0)
                        }
                        
                        logger.warning(f"[OK] NEW OBJECT: {cluster_key} = {new_uuid[:8]}... "
                                     f"({cluster_data.get('total_points', 0)} pts)")
                        
                        for existing_uuid in state_bank.permanent_uuid_registry.keys():
                            if existing_uuid != new_uuid:
                                pair = tuple(sorted([new_uuid, existing_uuid]))
                                state_bank.merge_prevention_pairs.add(pair)
                
                # Handle disappeared objects
                for j, prev_uuid in enumerate(previous_uuids):
                    if j not in assigned_previous:
                        _is_primary = (prev_uuid == getattr(state_bank, 'primary_subject_uuid', None))
                        if _is_primary:
                            # PRIMARY SUBJECT: NEVER delete from last_frame_assignments.
                            # Keep last known centroid frozen so future frames can
                            # re-match. The skeleton temporal state carries over intact.
                            logger.warning(
                                f"[PERSON] Primary subject {prev_uuid[:8]} disappeared "
                                f"in frame {frame_num}. Last centroid FROZEN - skeleton "
                                f"persists, will re-match when visible again."
                            )
                            # Do NOT delete from last_frame_assignments
                        else:
                            logger.warning(
                                f"[WARNING] Object {prev_uuid[:8]} disappeared in frame {frame_num}"
                            )
                            if prev_uuid in state_bank.last_frame_assignments:
                                del state_bank.last_frame_assignments[prev_uuid]
            
            # ============================================================
            # DYNAMIC ANOMALY DETECTION (replaces hardcoded frame checks)
            # ============================================================
            frame_anomalies = detect_cluster_anomalies(current_clusters, state_bank, frame_num)
            
            if any(frame_anomalies.values()):
                logger.warning("=" * 60)
                logger.warning(f"[OK] ANOMALIES DETECTED IN FRAME {frame_num}")
                
                if frame_anomalies['size_swaps']:
                    logger.error(f"   SIZE SWAPS: {len(frame_anomalies['size_swaps'])}")
                    for swap in frame_anomalies['size_swaps']:
                        logger.error(f"     {swap['description']}")
                
                if frame_anomalies['size_jumps']:
                    logger.warning(f"   SIZE JUMPS: {len(frame_anomalies['size_jumps'])}")
                    for jump in frame_anomalies['size_jumps']:
                        logger.warning(f"     {jump['uuid'][:8]}: {jump['type']} "
                                      f"({jump['prev_points']} [OK] {jump['curr_points']})")
                
                if frame_anomalies['centroid_jumps']:
                    logger.warning(f"   CENTROID JUMPS: {len(frame_anomalies['centroid_jumps'])}")
                    for jump in frame_anomalies['centroid_jumps']:
                        logger.warning(f"     {jump['uuid'][:8]}: {jump['distance']:.1f}cm")
                
                # Store anomalies for debugging
                if not hasattr(state_bank, 'frame_anomalies'):
                    state_bank.frame_anomalies = {}
                state_bank.frame_anomalies[frame_num] = frame_anomalies
                
                logger.warning("=" * 60)
            
            # ============================================================
            # ENHANCED ANTI-MERGE PROTECTION WITH VOLUME VALIDATION
            # ============================================================
            if len(current_clusters) >= 2:
                cluster_list = list(current_clusters.items())
                volume_tracker = getattr(state_bank, 'volume_tracker', None)
                
                for i in range(len(cluster_list)):
                    for j in range(i+1, len(cluster_list)):
                        cluster1_key, cluster1_data = cluster_list[i]
                        cluster2_key, cluster2_data = cluster_list[j]
                        
                        uuid1 = cluster1_data.get('uuid')
                        uuid2 = cluster2_data.get('uuid')
                        
                        if uuid1 and uuid2:
                            centroid1 = np.array(cluster1_data.get('centroid', [0,0,0]))
                            centroid2 = np.array(cluster2_data.get('centroid', [0,0,0]))
                            distance = np.linalg.norm(centroid1 - centroid2)
                            
                            if distance < 30.0:
                                pair_key = tuple(sorted([uuid1, uuid2]))
                                volume_validation_blocked = False
                                
                                if volume_tracker is not None:
                                    vol1 = volume_tracker.get_stable_volume(uuid1)
                                    vol2 = volume_tracker.get_stable_volume(uuid2)
                                    
                                    if vol1 is not None and vol2 is not None:
                                        expected_sum = vol1 + vol2
                                        
                                        pts1 = cluster1_data.get('points')
                                        pts2 = cluster2_data.get('points')
                                        
                                        if pts1 is not None and pts2 is not None:
                                            merged_pts = np.vstack([pts1, pts2])
                                            vol_est_obj, _ = volume_tracker.update_volume('_temp_merge_check', merged_pts, frame_num)
                                            merged_vol_est = vol_est_obj.volume_cm3 if vol_est_obj is not None else None
                                            
                                            if merged_vol_est is not None and expected_sum > 0:
                                                ratio = merged_vol_est / expected_sum
                                                
                                                logger.info(f"[OK] BEATING HEART VALIDATION: {uuid1[:8]} + {uuid2[:8]}")
                                                logger.info(f"   vol1={vol1:.0f} + vol2={vol2:.0f} = {expected_sum:.0f} cm3")
                                                logger.info(f"   merged={merged_vol_est:.0f} cm3 (ratio={ratio:.2f})")
                                                
                                                if 0.7 < ratio < 1.4:
                                                    logger.warning(f"   OCCLUSION DETECTED - NOT A REAL MERGE!")
                                                    volume_validation_blocked = True
                                                    
                                                    if not hasattr(state_bank, 'volume_merge_blocks'):
                                                        state_bank.volume_merge_blocks = []
                                                    state_bank.volume_merge_blocks.append({
                                                        'frame': frame_num,
                                                        'uuid1': uuid1,
                                                        'uuid2': uuid2,
                                                        'vol1': vol1,
                                                        'vol2': vol2,
                                                        'merged_est': merged_vol_est,
                                                        'ratio': ratio,
                                                        'reason': 'volume_sum_preserved'
                                                    })
                                
                                if pair_key in state_bank.merge_prevention_pairs or volume_validation_blocked:
                                    logger.error(f"[OK] MERGE PREVENTION CRITICAL: {uuid1[:8]} and {uuid2[:8]} "
                                               f"are {distance:.1f}cm apart but MUST NEVER MERGE!")
                                    
                                    if not hasattr(state_bank, 'merge_prevention_events'):
                                        state_bank.merge_prevention_events = []
                                    
                                    state_bank.merge_prevention_events.append({
                                        'frame': frame_num,
                                        'uuid1': uuid1,
                                        'uuid2': uuid2,
                                        'distance': distance,
                                        'prevented': True,
                                        'volume_blocked': volume_validation_blocked
                                    })
                                else:
                                    state_bank.merge_prevention_pairs.add(pair_key)
                                    logger.warning(f"NEW MERGE RISK: Adding prevention for "
                                                 f"{uuid1[:8]} <-> {uuid2[:8]} at {distance:.1f}cm")
            
            # ============================================================
            # CREATE/UPDATE HISTORICAL LOCKS
            # SKIPPED when inventory is active — inventory is the authority
            # ============================================================
            _inventory_active = (hasattr(state_bank, 'object_inventory') and
                                 len(state_bank.object_inventory) > 0)
            if not _inventory_active:
                for uuid_str in state_bank.permanent_uuid_registry.keys():
                    if uuid_str not in state_bank.historical_locks:
                        registry_data = state_bank.permanent_uuid_registry[uuid_str]
                        
                        state_bank.historical_locks[uuid_str] = {
                            'locked_at_frame': state_bank.uuid_created_frame.get(uuid_str, frame_num),
                            'position_index': registry_data['position_index'],
                            'last_seen_frame': frame_num if uuid_str in state_bank.last_frame_assignments else -1,
                            'disappearance_count': 0 if uuid_str in state_bank.last_frame_assignments else 1,
                            'initial_centroid': registry_data['initial_centroid'],
                            'typical_size': registry_data['typical_size'],
                            'permanent': True
                        }
                        logger.warning(f"[OK] Historical Lock created: {uuid_str[:8]}")
                    else:
                        lock = state_bank.historical_locks[uuid_str]
                        if uuid_str in state_bank.last_frame_assignments:
                            lock['last_seen_frame'] = frame_num
                            lock['disappearance_count'] = 0
                        else:
                            lock['disappearance_count'] += 1
            
            # Store in frame clusters
            state_bank.frame_clusters[frame_num] = {}
            for cluster_key, cluster_data in current_clusters.items():
                cluster_uuid = cluster_data.get('uuid')
                if cluster_uuid:
                    state_bank.frame_clusters[frame_num][cluster_uuid] = cluster_data.copy()
            
            # Verification logging
            logger.info(f"UUID Assignment Complete for Frame {frame_num}:")
            
            for cluster_key, cluster_data in current_clusters.items():
                uuid_val = cluster_data.get('uuid', 'NONE')
                points_count = cluster_data.get('total_points', 0)
                
                if uuid_val in state_bank.permanent_uuid_registry:
                    pos = state_bank.permanent_uuid_registry[uuid_val]['position_index']
                    logger.warning(f"  {cluster_key}: Position {pos} = {uuid_val[:8]}... ({points_count} points)")
                else:
                    logger.info(f"  {cluster_key}: {uuid_val[:8]}... ({points_count} points)")
            
            # Check for duplicate UUIDs
            assigned_uuids = [c.get('uuid') for c in current_clusters.values() if c.get('uuid')]
            if len(assigned_uuids) != len(set(assigned_uuids)):
                logger.error("CRITICAL: Duplicate UUIDs detected!")
                from collections import Counter
                uuid_counts = Counter(assigned_uuids)
                for uuid_val, count in uuid_counts.items():
                    if count > 1:
                        logger.error(f"  UUID {uuid_val[:8]}... assigned to {count} clusters!")
                        
        else:
            logger.error("No state_bank available - emergency UUID assignment!")
            for cluster_key, cluster_data in current_clusters.items():
                cluster_data['uuid'] = str(uuid_module.uuid4())
        
        # ============================================================
        # STEP 8: UPDATE STATE BANK
        # ============================================================
        if state_bank is not None:
            logger.info(f"Updating state_bank with {len(current_clusters)} UUID-assigned clusters")
            
            state_bank.frame_clusters[frame_num] = {}
            sorted_cluster_keys = sorted(current_clusters.keys())
            
            for cluster_key, cluster_data in current_clusters.items():
                if not isinstance(cluster_data, dict):
                    logger.error(f"SKIP: {cluster_key} is not a dict: {type(cluster_data)}")
                    continue
                    
                cluster_uuid = cluster_data.get('uuid')
                if not cluster_uuid:
                    logger.error(f"SKIP: {cluster_key} has no UUID")
                    continue
                
                try:
                    position_index = int(cluster_key.replace('cluster_', ''))
                except ValueError:
                    position_index = sorted_cluster_keys.index(cluster_key)
                
                cluster_data['position_index'] = position_index
                state_bank.frame_clusters[frame_num][cluster_uuid] = cluster_data.copy()
                
                if cluster_uuid not in state_bank.global_cluster_registry:
                    state_bank.global_cluster_registry[cluster_uuid] = {
                        'first_seen': frame_num,
                        'last_seen': frame_num,
                        'frames_present': [frame_num],
                        'total_points': cluster_data.get('total_points', 0),
                        'latest_centroid': cluster_data.get('centroid', [0, 0, 0]),
                        'bbox': cluster_data.get('bbox'),
                        'latest_bbox': cluster_data.get('bbox'),
                        'typical_size': cluster_data.get('point_count', cluster_data.get('total_points', 0)),
                        'position_index': position_index
                    }
                else:
                    registry_entry = state_bank.global_cluster_registry[cluster_uuid]
                    registry_entry['last_seen'] = frame_num
                    if frame_num not in registry_entry['frames_present']:
                        registry_entry['frames_present'].append(frame_num)
                    registry_entry['latest_centroid'] = cluster_data.get('centroid', [0, 0, 0])
                    registry_entry['position_index'] = position_index
                    
                    if 'bbox' in cluster_data and cluster_data['bbox']:
                        registry_entry['latest_bbox'] = cluster_data['bbox']
                        if not registry_entry.get('bbox'):
                            registry_entry['bbox'] = cluster_data['bbox']
                    
                    point_count = cluster_data.get('point_count', cluster_data.get('total_points', 0))
                    if point_count > 0:
                        current_typical = registry_entry.get('typical_size', 0)
                        if current_typical == 0:
                            registry_entry['typical_size'] = point_count
                        else:
                            registry_entry['typical_size'] = int(0.7 * current_typical + 0.3 * point_count)
                
                frames_tracked = len(state_bank.global_cluster_registry[cluster_uuid]['frames_present'])
                if frames_tracked == 5 and not state_bank.has_historical_lock(cluster_uuid) and not _inventory_active:
                    bbox = cluster_data.get('bbox', state_bank.global_cluster_registry[cluster_uuid].get('latest_bbox'))
                    centroid = cluster_data.get('centroid', [0, 0, 0])
                    
                    y_span = None
                    if bbox and isinstance(bbox, dict) and 'min' in bbox and 'max' in bbox:
                        y_span = [bbox['min'][1], bbox['max'][1]]
                    else:
                        centroid_y = centroid[1] if len(centroid) > 1 else 0
                        y_span = [centroid_y - 50, centroid_y + 50]
                    
                    state_bank.historical_locks[cluster_uuid] = {
                        'locked_at_frame': frame_num,
                        'frames_tracked': frames_tracked,
                        'last_seen_frame': frame_num,
                        'disappearance_count': 0,
                        'initial_centroid': centroid,
                        'centroid': centroid,
                        'position_index': position_index,
                        'bbox': bbox,
                        'y_span': y_span,
                        'typical_size': cluster_data.get('point_count', cluster_data.get('total_points', 0)),
                        'permanent': True,
                        'voxel_metadata': cluster_data.get('voxel_data', {}),
                        'voxel_indices': cluster_data.get('voxel_indices', [])
                    }
                    
                    logger.warning(f"[OK] NEW Historical Lock: {cluster_uuid[:8]} at position {position_index}")
            
            state_bank.update_clusters(
                current_clusters=current_clusters,
                current_labels=labels,
                points=points,
                frame_num=frame_num,
                pose_3d=pose_3d,
                pose_2d_confidence=pose_2d.get('scores', [0])[0] if pose_2d and isinstance(pose_2d, dict) and 'scores' in pose_2d else None
            )
        
        # ============================================================
        # STEP 9: ADD TO TEMPORAL BUFFER
        # ============================================================
        if frame_buffer is not None and not skip_temporal:
            frame_data = {
                'frame_num': frame_num,
                'points': points,
                'labels': labels,
                'n_clusters': n_clusters,
                'clusters': current_clusters,
                'voxel_metadata': {},
                'pose_2d': pose_2d.tolist() if isinstance(pose_2d, np.ndarray) else pose_2d,
                'pose_3d': pose_3d.tolist() if isinstance(pose_3d, np.ndarray) else pose_3d,
                'person_cluster_uuid': person_cluster_uuid,
                'pose_predicted': pose_was_predicted,
                'consecutive_rejections': consecutive_rejections,
                'segment_validation_applied': segment_validation_applied,
                # ===================== NEW =====================
                'facing_info': facing_info  # Add facing for temporal tracking
                # ===============================================
            }
            frame_buffer.add_frame(frame_data)
            logger.info(f"Added frame {frame_num} to temporal buffer (buffer size: {len(frame_buffer.frames)})")
            
            if frame_buffer.is_full() and state_bank is not None:
                logger.info(f"Buffer full ({len(frame_buffer.frames)} frames), processing temporal clusters...")
                
                try:
                    buffer_points, frame_indices, point_to_frame_mapping, voxel_metadata = frame_buffer.get_buffer_points_for_clustering()
                    
                    temporal_clusters = {}
                    for cluster_key, cluster_data in current_clusters.items():
                        temporal_clusters[cluster_key] = cluster_data.copy()
                    
                    matched_clusters, frame_assignments = state_bank.process_temporal_clusters(
                        temporal_clusters=temporal_clusters,
                        frames=frame_buffer.frames,
                        frame_indices=frame_indices,
                        point_to_frame_mapping=point_to_frame_mapping
                    )
                    
                    logger.info(f"Temporal processing complete: {len(matched_clusters)} matched clusters")
                    
                    if len(matched_clusters) == 0:
                        # Try inventory first, historical_locks as fallback
                        _resurrection_source = {}
                        if _inventory_active:
                            _resurrection_source = {uid: inv for uid, inv in state_bank.object_inventory.items()}
                            logger.warning("No clusters but inventory has objects - RESURRECTING from inventory!")
                        elif state_bank.historical_locks:
                            _resurrection_source = state_bank.historical_locks
                            logger.warning("No clusters but Historical Locks exist - RESURRECTING!")

                        for locked_uuid, lock_info in _resurrection_source.items():
                            _cent = lock_info.get('last_centroid', lock_info.get('centroid', [0, 0, 0]))
                            if locked_uuid in state_bank.cluster_histories and state_bank.cluster_histories[locked_uuid]:
                                last_hist = state_bank.cluster_histories[locked_uuid][-1]
                                _cent = last_hist.get('centroid', _cent)
                            
                            resurrected_cluster = {
                                'uuid': locked_uuid,
                                'centroid': _cent,
                                'point_count': 0,
                                'original_points': 0,
                                'dummy_points': 0,
                                'resurrected': True
                            }
                            
                            matched_clusters[locked_uuid] = resurrected_cluster
                            logger.warning(f"  [OK] RESURRECTED: {locked_uuid[:8]}")
                    
                    for cluster_uuid, cluster_data in matched_clusters.items():
                        if cluster_uuid not in state_bank.global_cluster_registry:
                            state_bank.global_cluster_registry[cluster_uuid] = {
                                'first_seen': frame_num,
                                'last_seen': frame_num,
                                'frames_present': [frame_num],
                                'total_points': cluster_data.get('point_count', 0),
                                'uuid': cluster_uuid
                            }
                        else:
                            registry_entry = state_bank.global_cluster_registry[cluster_uuid]
                            registry_entry['last_seen'] = frame_num
                            if frame_num not in registry_entry['frames_present']:
                                registry_entry['frames_present'].append(frame_num)
                        
                        state_bank.active_clusters[cluster_uuid] = state_bank.global_cluster_registry[cluster_uuid]
                        state_bank._xz_footprints[cluster_uuid] = state_bank.calculate_xz_footprint(cluster_data)
                        
                        history_entry = {
                            'frame': frame_num,
                            'centroid': cluster_data.get('centroid', [0, 0, 0]),
                            'point_count': cluster_data.get('point_count', 0),
                            'confidence': 1.0,
                            'resurrected': cluster_data.get('resurrected', False)
                        }
                        state_bank.cluster_histories[cluster_uuid].append(history_entry)
                    
                    for cluster_uuid in matched_clusters.keys():
                        if not _inventory_active:
                            state_bank.check_and_create_historical_lock(cluster_uuid, frame_num)
                    
                    MIN_CLUSTER_SIZE = 500
                    filtered_matched = {}
                    
                    for cluster_uuid, cluster_data in matched_clusters.items():
                        max_points = cluster_data.get('point_count', 0)
                        
                        _in_inventory = (_inventory_active and
                                         cluster_uuid in state_bank.object_inventory)
                        if _in_inventory:
                            filtered_matched[cluster_uuid] = cluster_data
                            logger.warning(f"[OK] INVENTORY preserves {cluster_uuid[:8]}")
                        elif state_bank.has_historical_lock(cluster_uuid):
                            filtered_matched[cluster_uuid] = cluster_data
                            logger.warning(f"[OK] HISTORICAL LOCK preserves {cluster_uuid[:8]}")
                        elif max_points >= MIN_CLUSTER_SIZE:
                            filtered_matched[cluster_uuid] = cluster_data
                        else:
                            logger.warning(f"REJECTED: {cluster_uuid[:8]} ({max_points} < {MIN_CLUSTER_SIZE})")
                    
                    matched_clusters = filtered_matched
                    
                    uuid_to_label = {}
                    sorted_clusters = sorted(matched_clusters.items(),
                                           key=lambda x: x[1].get('centroid', [0,0,0])[0])
                    
                    for idx, (uuid, cluster_data) in enumerate(sorted_clusters):
                        cluster_label = f"cluster_{idx:02d}"
                        uuid_to_label[uuid] = cluster_label
                        cluster_data['uuid'] = uuid
                        cluster_data['cluster_id'] = cluster_label
                    
                    current_clusters = {}
                    for uuid, cluster_label in uuid_to_label.items():
                        current_clusters[cluster_label] = matched_clusters[uuid].copy()
                    
                    logger.info(f"[OK] Temporal processing complete: {len(current_clusters)} clusters ready")

                    # ==========================================================
                    # PERSON-LABEL-REMAP FIX (frame 45 "missing head" root cause)
                    # The rename loop above rewrites cluster labels by centroid-X
                    # sort.  `person_cluster_uuid` still holds the OLD label that
                    # MMPose returned earlier this frame — which can now point at
                    # a completely different cluster (e.g. the chair).
                    #
                    # Concrete case: frame 45 had 3 clusters (75-pt stub, person,
                    # chair).  MMPose set person_cluster_uuid='cluster_01' (the
                    # real person).  Then the stub got filtered (75 < 500), and
                    # the rename re-sorted the remaining 2 clusters by centroid
                    # X: person (X=-17) → cluster_00, chair (X=+51) → cluster_01.
                    # Stale person_cluster_uuid='cluster_01' now indexes the
                    # CHAIR in current_clusters and in results_data[clusters_info],
                    # which is why skeleton_fitting ran on the chair's 208 voxels
                    # producing a mannequin at chair-Y, leaving the real person
                    # with no skeleton → "missing head" in the visualization.
                    #
                    # Fix: use the stable person_actual_uuid (captured at MMPose
                    # time and refreshed at sticky match) to look up the person's
                    # NEW label in uuid_to_label, and replace person_cluster_uuid.
                    # ==========================================================
                    if (person_actual_uuid is not None
                            and person_actual_uuid in uuid_to_label):
                        _new_person_label = uuid_to_label[person_actual_uuid]
                        if _new_person_label != person_cluster_uuid:
                            logger.warning(
                                f"[PERSON-LABEL-REMAP] Frame {frame_num}: "
                                f"person label updated "
                                f"{person_cluster_uuid} -> {_new_person_label} "
                                f"(uuid={person_actual_uuid[:8]} preserved)")
                            person_cluster_uuid = _new_person_label
                    elif person_actual_uuid is not None:
                        # Person UUID survived sticky match but the cluster was
                        # filtered out (e.g. rejected for low point count).  This
                        # frame has no visible person — log it so downstream
                        # skeleton fitting knows to skip or use fallback.
                        logger.warning(
                            f"[PERSON-LABEL-REMAP] Frame {frame_num}: person "
                            f"uuid={person_actual_uuid[:8]} NOT in final "
                            f"current_clusters (filtered out or missing) - "
                            f"person_cluster_uuid='{person_cluster_uuid}' may "
                            f"point at wrong cluster downstream")
                    
                except Exception as e:
                    logger.error(f"Error in temporal cluster processing: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
        
        # ============================================================
        # STEP 9.5: OBJECT INVENTORY RECONCILIATION (replaces inject_historical_lock_clusters)
        # The inventory is the SINGLE authority for cluster persistence.
        # It assigns UUIDs, recovers missing objects, splits merges,
        # and promotes new candidates after 5 frames.
        # ============================================================
        logger.warning("=" * 5)
        logger.warning("[OK] INVENTORY RECONCILIATION - the inventory is the authority")
        logger.warning("=" * 5)

        if state_bank is not None and hasattr(state_bank, 'object_inventory'):
            if frame_num == 1 or not state_bank.object_inventory:
                state_bank.inventory_init_frame1(
                    current_clusters, person_cluster_uuid)
            else:
                current_clusters = state_bank.reconcile_clusters_with_inventory(
                    current_clusters, frame_num,
                    voxel_grid=voxel_grid, points=points,
                    person_context=person_context)
            # YOLO label channel: bind cached detections to inventory entries.
            if hasattr(state_bank, 'apply_yolo_labels'):
                try:
                    from mmpose_integration import get_yolo_seg_result
                    _yolo_res = get_yolo_seg_result(None, frame_num)
                    if _yolo_res:
                        state_bank.apply_yolo_labels(
                            frame_num, _yolo_res,
                            image_size=getattr(args, 'image_size', (480, 864)))
                except Exception as _yolo_lbl_e:
                    logger.debug(f"[YOLO] label apply skipped: {_yolo_lbl_e}")

            # ISSUE-3 anti-merge: once a rigid object's class is LOCKED by the
            # YOLO channel, fit its OBB primitive ONCE from its voxels.  Read-only
            # (just stores the primitive on state_bank.rigid_primitives); the
            # actual voxel claim is gated separately by rigid_claim_enabled, so
            # this hook changes NO clustering behaviour by itself — it only makes
            # [RIGID-FIT] appear in the log so the lock can be verified first.
            if (hasattr(state_bank, 'object_inventory')
                    and hasattr(state_bank, 'rigid_primitives')):
                _RIGID_CLASSES = {'chair', 'couch', 'bench', 'table',
                                  'dining table', 'tv', 'refrigerator',
                                  'potted plant', 'suitcase'}
                for _ck, _cd in current_clusters.items():
                    _u = _cd.get('uuid', _cd.get('cluster_uuid'))
                    if not _u or _u in state_bank.rigid_primitives:
                        continue
                    _inv = state_bank.object_inventory.get(_u)
                    if not _inv or not _inv.get('_class_locked'):
                        continue
                    if _inv.get('class_label') not in _RIGID_CLASSES:
                        continue
                    _pts = _cd.get('points')
                    if _pts is None or len(_pts) == 0:
                        continue
                    _prim = fit_rigid_primitive(
                        _pts, class_label=_inv.get('class_label'),
                        frame_locked=frame_num)
                    if _prim is not None:
                        state_bank.rigid_primitives[_u] = _prim
                        logger.info(f"[RIGID-FIT] frame {frame_num}: locked "
                                    f"{_prim} for uuid={str(_u)[:8]}")

        logger.warning("[OK] INVENTORY RECONCILIATION COMPLETE")
        logger.warning("=" * 5)
        
        # =================================================================
        # LOGGER ITEM 8: Report synthetic/dummy points injected per cluster
        # =================================================================
        for _ck, _cd in current_clusters.items():
            _dummy = _cd.get('dummy_points', 0)
            _total = _cd.get('total_points', _cd.get('point_count', 0))
            _uuid = _cd.get('uuid', _cd.get('cluster_uuid', _ck))[:8]
            _is_injected = _cd.get('injected', False) or _cd.get('from_lock', False)
            if _is_injected or _dummy > 0:
                logger.info(f"  [SYNTHETIC-8] cluster {_uuid}: "
                           f"total_pts={_total} synthetic_pts={_dummy} "
                           f"injected={_is_injected}")
        
        # ============================================================
        # STEP 9.6: MERGE DUPLICATE UUID CLUSTERS
        # ============================================================
        # When VOLUME SWAP assigns a Historical Lock UUID to a better candidate,
        # but inject_historical_lock_clusters also created a "bluff" cluster with
        # the same UUID, we end up with DUPLICATES. Merge them here.
        #
        # SACRED RULE: A cluster whose UUID is a historical lock can NEVER be
        # removed — both copies must be preserved (they represent different
        # real-world objects that happen to share a UUID due to a pipeline bug).
        # In that case we reassign a fresh UUID to the smaller copy instead of
        # deleting it.
        # ============================================================
        
        # Collect all sacred UUIDs — from inventory (primary) or historical locks (fallback)
        _sacred_uuids = set()
        if state_bank is not None and hasattr(state_bank, 'object_inventory') and state_bank.object_inventory:
            _sacred_uuids = set(state_bank.object_inventory.keys())
        elif state_bank is not None and hasattr(state_bank, 'historical_locks'):
            _sacred_uuids = set(state_bank.historical_locks.keys())

        # Build UUID -> cluster_keys mapping
        uuid_to_keys = {}
        for cluster_key, cluster_data in current_clusters.items():
            uuid_str = cluster_data.get('cluster_uuid', cluster_data.get('uuid'))
            if uuid_str:
                if uuid_str not in uuid_to_keys:
                    uuid_to_keys[uuid_str] = []
                uuid_to_keys[uuid_str].append(cluster_key)
        
        # Find and resolve duplicates
        duplicates_found = False
        keys_to_remove = []
        
        for uuid_str, cluster_keys in uuid_to_keys.items():
            if len(cluster_keys) > 1:
                duplicates_found = True
                logger.warning(f"[OK] DUPLICATE UUID DETECTED: {uuid_str[:8]} appears in {len(cluster_keys)} clusters!")
                
                # Get all clusters with this UUID and their sizes
                clusters_with_sizes = []
                for key in cluster_keys:
                    cdata = current_clusters[key]
                    size = cdata.get('total_points', cdata.get('point_count', cdata.get('original_points', 0)))
                    clusters_with_sizes.append((key, size, cdata))
                    logger.warning(f"   {key}: {size} points")
                
                # Sort by size (largest first)
                clusters_with_sizes.sort(key=lambda x: x[1], reverse=True)
                
                # SACRED CHECK: if this UUID is a historical lock, DO NOT delete
                # any cluster — reassign a fresh UUID to all but the largest
                if uuid_str in _sacred_uuids:
                    keeper_key, keeper_size, keeper_data = clusters_with_sizes[0]
                    logger.warning(f"[SACRED] UUID {uuid_str[:8]} is historical lock - REASSIGNING smaller copies, not deleting")
                    for key, size, cdata in clusters_with_sizes[1:]:
                        fresh_uuid = str(uuid_module.uuid4())
                        cdata['cluster_uuid'] = fresh_uuid
                        cdata['uuid'] = fresh_uuid
                        logger.warning(f"[SACRED] Reassigned {key} ({size} pts) -> new UUID {fresh_uuid[:8]}")
                else:
                    # Keep the LARGEST, mark others for removal
                    keeper_key, keeper_size, keeper_data = clusters_with_sizes[0]
                    logger.warning(f"   [OK] KEEPING: {keeper_key} ({keeper_size} points)")
                    for key, size, cdata in clusters_with_sizes[1:]:
                        logger.warning(f"   [FAIL] REMOVING: {key} ({size} points) - merging into {keeper_key}")
                        keys_to_remove.append(key)
        
        # Remove the duplicate clusters
        for key in keys_to_remove:
            del current_clusters[key]
            logger.warning(f"[OK] Removed duplicate cluster: {key}")
        
        if duplicates_found:
            # Re-index cluster keys to maintain sequential naming
            remaining_clusters = list(current_clusters.values())
            remaining_clusters.sort(key=lambda c: c.get('centroid', [0, 0, 0])[0])
            
            # Rebuild current_clusters dict with sequential keys
            new_current_clusters = {}
            for idx, cdata in enumerate(remaining_clusters):
                new_key = f"cluster_{idx:02d}"
                new_current_clusters[new_key] = cdata
            
            current_clusters = new_current_clusters
            n_clusters = len(current_clusters)
            logger.warning(f"[OK] MERGE COMPLETE: Now {n_clusters} clusters after removing duplicates")
        
        # ============================================================
        # STEP 9.65: CLUSTER FLAT-PROJECTION DIAGNOSTICS
        # 
        # For every cluster in the current frame, compute its flat XZ
        # footprint (collapse Y) and flat YZ footprint (collapse X).
        # Compare each cluster's footprint cell-by-cell against the live
        # CoP flat XZ / flat YZ from this same frame to surface three
        # diagnostic signals:
        #
        #   1. Per-cluster connected-component count on each projection.
        #      A cluster whose own XZ silhouette has 2 disjoint blobs is
        #      internally separable (the doll-on-chair signature) — the
        #      thin-slice gate cannot see this geometry but the flat
        #      projection makes it explicit.
        #
        #   2. Per-cluster coverage % of CoP-XZ and CoP-YZ.
        #      How much of the live CoP silhouette this cluster owns.
        #      Frames where the person cluster covers <15% of CoP-XZ
        #      while another cluster covers <15% means significant CoP
        #      mass is unowned.
        #
        #   3. UNOWNED-CoP cell count.
        #      Cells in the live CoP-XZ / CoP-YZ that NO cluster's
        #      footprint covers.  This is the smoking gun for missing
        #      or undersized clusters — frames where this is >30% are
        #      where clustering is failing to claim the geometry.
        #
        # ZERO clustering decisions change here.  This is purely
        # diagnostic — like the FLAT-ORACLE block in step 1.  Future
        # zoning of the flat projections (planned next) will hook in
        # at the same boundaries this block establishes.
        # ============================================================
        if _ctrl_origin is not None and _ctrl_res is not None and current_clusters:
            try:
                # Build the LIVE CoP flat projections for THIS frame only.
                # (The historical XZ/YZ from step 1 are different — they
                # accumulate across frames.  Step 3 compares per-frame
                # cluster footprints to per-frame CoP; cross-frame
                # comparison is a later step.)
                _ox = float(_ctrl_origin[0])
                _oy = float(_ctrl_origin[1])
                _oz = float(_ctrl_origin[2])
                _r  = float(_ctrl_res)

                _cop_xz_live = set()
                _cop_yz_live = set()
                for _cp in points:
                    _xi = int((float(_cp[0]) - _ox) / _r)
                    _yi = int((float(_cp[1]) - _oy) / _r)
                    _zi = int((float(_cp[2]) - _oz) / _r)
                    _cop_xz_live.add((_xi, _zi))
                    _cop_yz_live.add((_yi, _zi))

                # 4-connected components on a sparse cell set, BFS over
                # a stack.  No scipy dependency, ~25 lines.
                def _cc4_count(cells):
                    if not cells:
                        return 0
                    remaining = set(cells)
                    n_comp = 0
                    while remaining:
                        seed = next(iter(remaining))
                        stack = [seed]
                        n_comp += 1
                        while stack:
                            c = stack.pop()
                            if c not in remaining:
                                continue
                            remaining.remove(c)
                            ca, cb = c
                            for nb in ((ca + 1, cb), (ca - 1, cb),
                                       (ca, cb + 1), (ca, cb - 1)):
                                if nb in remaining:
                                    stack.append(nb)
                    return n_comp

                _all_cluster_xz = set()
                _all_cluster_yz = set()
                _per_cluster = []  # collected for the summary log

                for _ckey, _cdata in current_clusters.items():
                    _vi = _cdata.get('voxel_indices')
                    if not _vi:
                        # Some clusters store voxel data only in
                        # 'voxel_data' as string keys "(x,y,z)".  Fall
                        # back to that if voxel_indices is missing.
                        _vd = _cdata.get('voxel_data')
                        if _vd:
                            _vi = []
                            for _vk in _vd.keys():
                                try:
                                    _clean = _vk.strip("() ")
                                    _parts = _clean.split(",")
                                    _vi.append((int(_parts[0].strip()),
                                                int(_parts[1].strip()),
                                                int(_parts[2].strip())))
                                except Exception:
                                    pass
                    if not _vi:
                        # Last-ditch: derive voxel cells from raw points
                        # (which may not align perfectly with the voxel
                        # grid but at least gives a footprint).
                        _pts = _cdata.get('points')
                        if _pts is not None and len(_pts) > 0:
                            _vi = []
                            for _pp in _pts:
                                _vi.append((int((float(_pp[0]) - _ox) / _r),
                                            int((float(_pp[1]) - _oy) / _r),
                                            int((float(_pp[2]) - _oz) / _r)))
                    if not _vi:
                        _per_cluster.append((_ckey, _cdata, set(), set(), 0, 0, 0, 0))
                        continue

                    # Project this cluster's voxels onto XZ and YZ.
                    _xz_set = set()
                    _yz_set = set()
                    for _v in _vi:
                        _vt = tuple(_v) if isinstance(_v, list) else _v
                        if len(_vt) >= 3:
                            _xz_set.add((_vt[0], _vt[2]))
                            _yz_set.add((_vt[1], _vt[2]))

                    _xz_blobs = _cc4_count(_xz_set)
                    _yz_blobs = _cc4_count(_yz_set)

                    # Coverage of LIVE CoP by THIS cluster (cells the
                    # cluster owns that the CoP also has).
                    _xz_cov = len(_xz_set & _cop_xz_live)
                    _yz_cov = len(_yz_set & _cop_yz_live)

                    _per_cluster.append(
                        (_ckey, _cdata, _xz_set, _yz_set,
                         _xz_blobs, _yz_blobs, _xz_cov, _yz_cov))
                    _all_cluster_xz |= _xz_set
                    _all_cluster_yz |= _yz_set

                    # Stash on the cluster dict so future steps can
                    # consume without recomputing.  Keys chosen so they
                    # don't collide with any existing cluster fields.
                    _cdata['_flat_xz_footprint'] = _xz_set
                    _cdata['_flat_yz_footprint'] = _yz_set
                    _cdata['_flat_xz_blobs']     = _xz_blobs
                    _cdata['_flat_yz_blobs']     = _yz_blobs

                # UNOWNED CoP territory: cells in CoP no cluster covers
                _unowned_xz = _cop_xz_live - _all_cluster_xz
                _unowned_yz = _cop_yz_live - _all_cluster_yz
                _cop_xz_n = max(1, len(_cop_xz_live))
                _cop_yz_n = max(1, len(_cop_yz_live))

                # === Per-frame summary log ===
                logger.warning(
                    f"[CLUSTER-PROJ] Frame {frame_num}: {len(current_clusters)} clusters; "
                    f"CoP-XZ={len(_cop_xz_live)} cells, "
                    f"CoP-YZ={len(_cop_yz_live)} cells")

                # === Per-cluster detail logs ===
                for (_ckey, _cdata, _xz_set, _yz_set,
                     _xz_blobs, _yz_blobs, _xz_cov, _yz_cov) in _per_cluster:
                    _is_person = (
                        person_cluster_uuid is not None
                        and (_cdata.get('uuid') == person_cluster_uuid
                             or _ckey == person_cluster_uuid))
                    _tag = "PERSON" if _is_person else "OBJ"
                    _uuid_short = (_cdata.get('uuid') or _ckey)[:8]
                    _xz_pct = 100.0 * _xz_cov / _cop_xz_n
                    _yz_pct = 100.0 * _yz_cov / _cop_yz_n
                    _split_flag = "[WARN]SPLITTABLE" if (_xz_blobs >= 2 or _yz_blobs >= 2) else ""
                    logger.warning(
                        f"[CLUSTER-PROJ]   {_ckey} ({_tag} {_uuid_short}): "
                        f"XZ={len(_xz_set)} cells ({_xz_blobs} blob"
                        f"{'s' if _xz_blobs != 1 else ''}), "
                        f"YZ={len(_yz_set)} cells ({_yz_blobs} blob"
                        f"{'s' if _yz_blobs != 1 else ''}); "
                        f"covers {_xz_cov}/{len(_cop_xz_live)}={_xz_pct:.0f}% of CoP-XZ, "
                        f"{_yz_cov}/{len(_cop_yz_live)}={_yz_pct:.0f}% of CoP-YZ"
                        f"{_split_flag}")

                # === Unowned-territory summary ===
                _unowned_xz_pct = 100.0 * len(_unowned_xz) / _cop_xz_n
                _unowned_yz_pct = 100.0 * len(_unowned_yz) / _cop_yz_n
                _unowned_flag = ""
                if _unowned_xz_pct > 30.0 or _unowned_yz_pct > 30.0:
                    _unowned_flag = "[WARN]UNOWNED-HIGH"
                logger.warning(
                    f"[CLUSTER-PROJ] Frame {frame_num} unowned CoP: "
                    f"XZ={len(_unowned_xz)} cells ({_unowned_xz_pct:.0f}%), "
                    f"YZ={len(_unowned_yz)} cells ({_unowned_yz_pct:.0f}%)"
                    f"{_unowned_flag}")

                # === Save per-frame projection sidecar JSON ===
                # Sidecar JSON write moved to STEP 10 (late-state) so the
                # per-frame projection data matches the clusters that
                # frame_results.json actually ships.  STEP 9.65 keeps the
                # per-frame log lines for diagnostic visibility but no
                # longer writes the JSON here.
            except Exception as _proj_e:
                logger.warning(f"[CLUSTER-PROJ] failed: {_proj_e}")
        
        # ============================================================
        # STEP 9.7: BLANKET ALGORITHM — shell extraction + ICP
        # Extract surface voxels from person cluster, register against
        # previous frame's surface voxels via ICP, store for next frame.
        # ============================================================
        if state_bank is not None and person_cluster_uuid and voxel_grid is not None:
            try:
                # Find person cluster's voxel indices
                _person_voxels = set()
                for _ck, _cd in current_clusters.items():
                    _cuuid = _cd.get('uuid', '')
                    if _cuuid == person_cluster_uuid or _ck == person_cluster_uuid:
                        _vi = _cd.get('voxel_indices')
                        if _vi:
                            _person_voxels = set(tuple(v) if isinstance(v, list) else v for v in _vi)
                        break

                if len(_person_voxels) > 20:
                    # Extract surface voxels
                    curr_shell = voxel_grid.extract_shell_voxels(_person_voxels)
                    
                    if len(curr_shell) > 10:
                        # ICP against previous frame
                        prev_shell = state_bank.get_previous_shell_voxels(person_cluster_uuid)
                        icp_result = None
                        if prev_shell is not None and len(prev_shell) > 10:
                            icp_result = voxel_grid.register_shell_voxels_icp(
                                prev_shell, curr_shell, max_distance=10.0)
                            if icp_result is not None:
                                state_bank.store_icp_transform(person_cluster_uuid, icp_result)
                                logger.info(f"Frame {frame_num}: [BLANKET] ICP: "
                                           f"rot={icp_result['rotation_deg']:.1f}deg, "
                                           f"trans={np.linalg.norm(icp_result['translation_cm']):.1f}cm, "
                                           f"fitness={icp_result['fitness']:.3f}")
                        
                        # Store current shell for next frame's ICP
                        state_bank.store_shell_voxels(person_cluster_uuid, curr_shell)
                        logger.info(f"Frame {frame_num}: [BLANKET] Stored {len(curr_shell)} "
                                   f"shell voxels for {person_cluster_uuid[:8]}")
            except Exception as _blanket_err:
                logger.warning(f"Frame {frame_num}: [BLANKET] Shell extraction failed: {_blanket_err}")

        # ============================================================
        # (STEP 9.9 removed — inventory reconciliation moved to STEP 9.5)
        # ============================================================

        # ============================================================
        # STEP 9.95: INVENTORY-AUTHORITATIVE PERSON ID
        # ============================================================
        if (state_bank is not None
                and hasattr(state_bank, 'primary_subject_uuid')
                and state_bank.primary_subject_uuid):
            _inv_person_uuid = state_bank.primary_subject_uuid
            _inv_person_label = None
            for _ck, _cd in current_clusters.items():
                _cluster_uuid = _cd.get('uuid', _cd.get('cluster_uuid', ''))
                if _cluster_uuid == _inv_person_uuid:
                    _inv_person_label = _ck
                    break
            if _inv_person_label is not None and _inv_person_label != person_cluster_uuid:
                logger.warning(
                    f"[PERSON-INV-OVERRIDE] Frame {frame_num}: "
                    f"inventory says person={_inv_person_label} "
                    f"(uuid={_inv_person_uuid[:8]}) but "
                    f"person_cluster_uuid={person_cluster_uuid} - "
                    f"overriding to {_inv_person_label}")
                person_cluster_uuid = _inv_person_label
                person_actual_uuid = _inv_person_uuid
                if frame_buffer is not None:
                    for _buf_entry in frame_buffer.frames:
                        if _buf_entry.get('frame_num') == frame_num:
                            _buf_entry['person_cluster_uuid'] = person_cluster_uuid
                            break

        # ============================================================
        # STEP 10: SAVE FRAME RESULTS
        # ============================================================
        results_dir = os.path.join(args.output, "results")
        os.makedirs(results_dir, exist_ok=True)
        results_file = os.path.join(results_dir, f"frame_{frame_num:03d}_results.json")

        # ------------------------------------------------------------
        # LATE-STATE PROJECTION SIDECAR  (frame_NNN_proj.json)
        #
        # Recompute flat XZ/YZ projections from `current_clusters` AT THIS
        # POINT (after every clustering decision, after STEP 9.5 inventory
        # reconciliation, after STEP 9.6 duplicate merge, after STEP 9.7
        # blanket).  Cluster set here is byte-identical to what
        # frame_results.json ships (line 3571 below) and what the 3D
        # Pose visualization panel reads via clusters_by_frame.
        #
        # Earlier STEP 9.65 wrote the same file from MID-pipeline state,
        # which let it disagree with the 3D Pose panel (panel showed 2
        # clusters at F13 while the sidecar showed 3).  This block
        # eliminates that discrepancy.
        #
        # Sidecar fields:
        #   frame_num, origin, res            : grid metadata
        #   cop_xz, cop_yz                    : live CoP cells (flat int)
        #   unowned_xz, unowned_yz            : CoP cells no cluster owns
        #   unowned_xz_pct, unowned_yz_pct    : numeric coverage gap
        #   cluster_count_dict                : len(current_clusters)
        #   cluster_count_visualized          : computed for table parity
        #                                       (see below — uses same
        #                                       voxel-projection logic as
        #                                       visualization.py)
        #   cluster_uuids                     : list of UUIDs in stable
        #                                       cluster_key order, for
        #                                       history-match column
        #   clusters[]                        : per-cluster footprint &
        #                                       blob counts
        # ------------------------------------------------------------
        # Helper function to sort voxel_data by Y, X, Z
        def sort_voxel_data_by_yxz(voxel_data_dict):
            if not voxel_data_dict:
                return {}
            def parse_key(k):
                clean = k.strip("() ").strip()
                if clean.startswith("cf_"):   # capsule-fill voxel key: cf_<ix>_<iy>_<iz>
                    parts = clean[3:].split("_")
                else:                          # normal voxel key: "(x, y, z)" / "x,y,z"
                    parts = clean.split(",")
                return (int(parts[0].strip()), int(parts[1].strip()), int(parts[2].strip()))
            sorted_keys = sorted(voxel_data_dict.keys(), 
                                 key=lambda k: (parse_key(k)[1], parse_key(k)[0], parse_key(k)[2]))
            return {k: voxel_data_dict[k] for k in sorted_keys}
        
        # Helper function to round floats to 1 decimal place
        def round_floats(obj, decimals=1):
            if isinstance(obj, float):
                return round(obj, decimals)
            elif isinstance(obj, dict):
                return {k: round_floats(v, decimals) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [round_floats(x, decimals) for x in obj]
            elif isinstance(obj, np.ndarray):
                return [round_floats(x, decimals) for x in obj.tolist()]
            elif isinstance(obj, (np.float32, np.float64)):
                return round(float(obj), decimals)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            return obj
        
        # Calculate voxel indices for pose_3d keypoints
        pose_3d_with_voxels = None
        if pose_3d is not None:
            kp_list = pose_3d.tolist() if isinstance(pose_3d, np.ndarray) else pose_3d
            voxel_indices = []
            grid_res = args.grid_resolution if hasattr(args, 'grid_resolution') else 2.0
            
            # Get grid origin from voxel_grid bounds
            grid_origin = voxel_grid.bounds[0] if (voxel_grid and hasattr(voxel_grid, 'bounds') and voxel_grid.bounds is not None) else np.array([0.0, 0.0, 0.0])
            
            for kp in kp_list:
                if kp is not None and len(kp) == 3 and not all(v == 0 for v in kp):
                    # Correct formula: (point - origin) / resolution
                    vx = int((kp[0] - grid_origin[0]) / grid_res)
                    vy = int((kp[1] - grid_origin[1]) / grid_res)
                    vz = int((kp[2] - grid_origin[2]) / grid_res)
                    voxel_indices.append([vx, vy, vz])
                else:
                    voxel_indices.append(None)
            pose_3d_with_voxels = {
                "keypoints": round_floats(kp_list),
                "voxel_indices": voxel_indices
            }
        
        # Clean pose_2d - remove keypoints_raw, round floats
        pose_2d_clean = None
        if pose_2d and isinstance(pose_2d, dict):
            pose_2d_clean = {}
            for k, v in pose_2d.items():
                if k == 'keypoints_raw':
                    continue  # Skip keypoints_raw
                if isinstance(v, np.ndarray):
                    pose_2d_clean[k] = round_floats(v.tolist())
                else:
                    pose_2d_clean[k] = round_floats(v)
        
        # ==================================================================
        # FIX-E REBUILD: rebuild person_bbox_3d from CURRENT-FRAME data
        # ----------------------------------------------------------------
        # The early FIX-E block (line ~1196) had to run BEFORE clustering
        # for the split veto, so it pulled data from state_bank.inventory
        # (which can be stale or zeroed) and from _early_pose_2d_bbox
        # (which often falls back to a previous frame's bbox when early
        # MMPose detection fails).  Two corruption modes for two frames:
        #   - frame 38: pose_2d_bbox FALLBACK from frame 37
        #   - frame 100: inventory last_3D_box = [0,0,0] -> Y-span [0,0]
        #
        # By the time we reach this point we have:
        #   - current_clusters[person_cluster_uuid] -> the actual person
        #     cluster's points for THIS frame
        #   - pose_2d -> the FRESH 2D pose from the main MMPose call
        #   - args camera params
        #
        # So rebuild the tunnel here from those, using the minimum Y of
        # the actual CoP points belonging to the person (per Michael:
        # "take the minimal y-wall in the CoP").  No inventory, no
        # centroid filter, no stale bbox.
        # ==================================================================
        _rebuilt_person_bbox_3d = None
        _rebuilt_pose_2d_bbox = None
        try:
            # ---- 1. Fresh 2D bbox from CURRENT pose_2d -------------------
            if pose_2d is not None and isinstance(pose_2d, dict):
                _kps_raw = pose_2d.get('keypoints')
                if _kps_raw is not None:
                    _kps_arr = np.asarray(_kps_raw, dtype=float)
                    if _kps_arr.ndim == 2 and _kps_arr.shape[0] >= 5:
                        if _kps_arr.shape[1] >= 3:
                            _vmask = _kps_arr[:, 2] > 0.3
                        else:
                            _vmask = (_kps_arr[:, 0] > 0) & (_kps_arr[:, 1] > 0)
                        if int(_vmask.sum()) >= 5:
                            _vk = _kps_arr[_vmask]
                            _rebuilt_pose_2d_bbox = [
                                float(_vk[:, 0].min()),
                                float(_vk[:, 1].min()),
                                float(_vk[:, 0].max()),
                                float(_vk[:, 1].max()),
                                1.0,
                            ]

            # ---- 2. Person cluster's actual points THIS FRAME ------------
            _person_cluster_pts = None
            if person_cluster_uuid and current_clusters:
                _pcd = current_clusters.get(person_cluster_uuid)
                if _pcd is None:
                    # Some paths key by 'cluster_NN' label, look up by uuid field
                    for _ck, _cd in current_clusters.items():
                        if _cd.get('uuid') == person_cluster_uuid:
                            _pcd = _cd
                            break
                if _pcd is not None:
                    _pp = _pcd.get('points')
                    if _pp is not None:
                        _person_cluster_pts = np.asarray(_pp)

            # ---- 2b. Y-CONTINUITY SANITY GUARD ---------------------------
            # Frame 45 bug: UUID→cluster mapping can get corrupted after
            # FORCEFUL RECOVERY + UUID reassignment steps. In that frame
            # current_clusters[person_cluster_uuid].points returned the
            # CHAIR's 11436 points at Y≈367cm even though the real person
            # cluster (cluster_01) had 18347 points at Y≈643cm and that is
            # the one that was sticky-matched to person_cluster_uuid.
            # Rebuild with 11436 chair-Y points produced tunnel Y=[354,392]
            # on frame 45, corrupting inventory for downstream frames.
            #
            # Fix: before trusting _person_cluster_pts, verify its centroid Y
            # is within _MAX_Y_JUMP_CM of the previous frame's person Y. If
            # not, reject the rebuild so downstream falls back to the early
            # FIX-E person_context (which is derived from state_bank and
            # remains at the correct Y even when current_clusters gets
            # corrupted).
            _MAX_Y_JUMP_CM = 50.0
            _prev_person_y = getattr(
                process_single_frame, '_prev_person_cop_y', None)
            if (_person_cluster_pts is not None
                    and _person_cluster_pts.ndim == 2
                    and _person_cluster_pts.shape[1] >= 3
                    and len(_person_cluster_pts) >= 30
                    and _prev_person_y is not None):
                _cluster_y_cent = float(np.mean(_person_cluster_pts[:, 1]))
                _y_delta = abs(_cluster_y_cent - _prev_person_y)
                if _y_delta > _MAX_Y_JUMP_CM:
                    # Try inventory as a second opinion before rejecting
                    _inv_y_cent = None
                    if (state_bank is not None
                            and hasattr(state_bank, 'object_inventory')
                            and person_cluster_uuid
                            and person_cluster_uuid in state_bank.object_inventory):
                        _inv_box = state_bank.object_inventory[
                            person_cluster_uuid].get('last_3D_box', {})
                        if 'min' in _inv_box and 'max' in _inv_box:
                            _inv_y_cent = (_inv_box['min'][1]
                                           + _inv_box['max'][1]) / 2.0
                    _inv_delta = (abs(_cluster_y_cent - _inv_y_cent)
                                  if _inv_y_cent is not None else None)
                    # If BOTH prev frame and inventory disagree with the
                    # cluster's Y, the cluster dict is corrupted. Reject.
                    if _inv_delta is None or _inv_delta > _MAX_Y_JUMP_CM:
                        logger.warning(
                            f"[FIX-E REBUILD] Frame {frame_num}: REJECTED - "
                            f"person cluster points Y={_cluster_y_cent:.0f}cm "
                            f"jumped {_y_delta:.0f}cm from prev "
                            f"(Y={_prev_person_y:.0f}cm)"
                            + (f", inv Y={_inv_y_cent:.0f}cm "
                               f"(delta={_inv_delta:.0f}cm)"
                               if _inv_y_cent is not None else ", no inv")
                            + "UUID->cluster mapping corrupted, "
                            "falling back to early FIX-E")
                        _person_cluster_pts = None

            # ---- 3. CoP min/max Y from those points ----------------------
            if (_person_cluster_pts is not None
                    and _person_cluster_pts.ndim == 2
                    and _person_cluster_pts.shape[1] >= 3
                    and len(_person_cluster_pts) >= 30
                    and _rebuilt_pose_2d_bbox is not None):

                _cop_y_front = float(np.min(_person_cluster_pts[:, 1]))
                _cop_y_back  = float(np.max(_person_cluster_pts[:, 1]))
                _cop_body_depth = max(20.0, min(120.0,
                                                (_cop_y_back - _cop_y_front)
                                                + 2.0 * args.grid_resolution))

                _rb_cam_pos  = getattr(args, 'camera_position', [-47.0, 28.0, -20.0])
                _rb_cam_tgt  = getattr(args, 'camera_target',   [-25.1, 123.8, -28.3])
                _rb_cam_fov  = getattr(args, 'field_of_view',   66.0)
                _rb_img_size = getattr(args, 'image_size',      (480, 864))

                _rebuilt_person_bbox_3d = _bbox_tunnel_to_3d(
                    _rebuilt_pose_2d_bbox,
                    _cop_y_front - args.grid_resolution,  # one step closer
                    _cop_body_depth,
                    _rb_cam_pos, _rb_cam_tgt, _rb_cam_fov, _rb_img_size,
                )

                if _rebuilt_person_bbox_3d is not None:
                    _actual_z_min = float(np.min(_person_cluster_pts[:, 2]))
                    _actual_z_max = float(np.max(_person_cluster_pts[:, 2]))
                    # Z-EXTENT: project 2D bbox to world Z, take max(cluster,proj)
                    try:
                        _um=0.5*(_rebuilt_pose_2d_bbox[0]+_rebuilt_pose_2d_bbox[2])
                        _ptt=_pixel_to_XZ_at_Y(_um,float(_rebuilt_pose_2d_bbox[1]),
                            max(10.,_cop_y_front),_rb_cam_pos,_rb_cam_tgt,_rb_cam_fov,_rb_img_size)
                        _ptb=_pixel_to_XZ_at_Y(_um,float(_rebuilt_pose_2d_bbox[3]),
                            max(10.,_cop_y_front),_rb_cam_pos,_rb_cam_tgt,_rb_cam_fov,_rb_img_size)
                        _pz_max=max(_ptt[1],_ptb[1]); _pz_min=min(_ptt[1],_ptb[1])
                        if _pz_max>_actual_z_max+10.:
                            logger.warning(f"[FIX-E REBUILD] Frame {frame_num}: "
                                f"Z_max {_actual_z_max:.0f}->{_pz_max:.0f}cm")
                            _actual_z_max=_pz_max
                        if _pz_min<_actual_z_min-10.: _actual_z_min=_pz_min
                    except Exception as _ze: logger.debug(f"Z-proj failed:{_ze}")

                    # ── FIX-E Z historical rejection ──
                    # Skip rejection on first 3 frames — no reliable baseline yet.
                    # Frame 1's Z_max can be off by 10-15cm; rejecting frame 2
                    # based on that bad baseline clips the head.
                    _MAX_Z_JUMP_CM = 8.0
                    _fxe_prev_zmin = getattr(process_single_frame, '_fixe_prev_z_min', None)
                    _fxe_prev_zmax = getattr(process_single_frame, '_fixe_prev_z_max', None)
                    _z_rejected = False
                    if (frame_num > 3
                            and _fxe_prev_zmin is not None
                            and _fxe_prev_zmax is not None):
                        _dz_min = abs(_actual_z_min - _fxe_prev_zmin)
                        _dz_max = abs(_actual_z_max - _fxe_prev_zmax)
                        if _dz_min > _MAX_Z_JUMP_CM or _dz_max > _MAX_Z_JUMP_CM:
                            logger.warning(
                                f"[FIX-E REBUILD] Frame {frame_num}: Z REJECTED - "
                                f"deltaz_min={_dz_min:.0f} deltaz_max={_dz_max:.0f} "
                                f"-> using prev Z=[{_fxe_prev_zmin:.0f},{_fxe_prev_zmax:.0f}]")
                            _actual_z_min = _fxe_prev_zmin
                            _actual_z_max = _fxe_prev_zmax
                            _z_rejected = True
                    # Always update baseline (even on early frames)
                    if not _z_rejected:
                        process_single_frame._fixe_prev_z_min = _actual_z_min
                        process_single_frame._fixe_prev_z_max = _actual_z_max

                    _rebuilt_person_bbox_3d['min'][2]=_actual_z_min
                    _rebuilt_person_bbox_3d['max'][2]=_actual_z_max

                    # CRITICAL: write corrected bbox back to person_bbox_3d
                    # and the cluster dict — otherwise results_data and the
                    # visualization 3D-CAL still show the uncorrected box.
                    person_bbox_3d = _rebuilt_person_bbox_3d
                    if (person_cluster_uuid is not None
                            and person_cluster_uuid in current_clusters):
                        current_clusters[person_cluster_uuid]['bbox'] = \
                            _rebuilt_person_bbox_3d

                    # Store authoritative person Y for next-frame continuity guard
                    process_single_frame._prev_person_cop_y=(
                        (_cop_y_front+_cop_y_back)/2.0)

                    # STEP 4: update person-specific flat history with validated cluster
                    if (_flat_xz_person_history is not None
                            and _flat_yz_person_history is not None
                            and _ctrl_origin is not None and _ctrl_res is not None):
                        try:
                            _ph_ox=float(_ctrl_origin[0]); _ph_oy=float(_ctrl_origin[1])
                            _ph_oz=float(_ctrl_origin[2]); _ph_r=float(_ctrl_res)
                            for _pp in _person_cluster_pts:
                                _xi=int((float(_pp[0])-_ph_ox)/_ph_r)
                                _yi=int((float(_pp[1])-_ph_oy)/_ph_r)
                                _zi=int((float(_pp[2])-_ph_oz)/_ph_r)
                                _flat_xz_person_history.add((_xi,_zi))
                                _flat_yz_person_history.add((_yi,_zi))
                            process_single_frame._flat_xz_person_set=_flat_xz_person_history
                            process_single_frame._flat_yz_person_set=_flat_yz_person_history
                            logger.info(f"[PERSON-HISTORY] Frame {frame_num}: "
                                f"XZ={len(_flat_xz_person_history)} cells, "
                                f"YZ={len(_flat_yz_person_history)} cells")
                        except Exception as _ph_e:
                            logger.debug(f"[PERSON-HISTORY] failed:{_ph_e}")

                    # STEP 3: post-clustering validation
                    if _flat_xz_person_history and _ctrl_origin is not None:
                        try:
                            _ph_ox=float(_ctrl_origin[0]); _ph_oz=float(_ctrl_origin[2])
                            _ph_r=float(_ctrl_res)
                            _cl_xz={( int((float(_pp[0])-_ph_ox)/_ph_r),
                                      int((float(_pp[2])-_ph_oz)/_ph_r))
                                    for _pp in _person_cluster_pts}
                            _iou=(100.*len(_cl_xz&_flat_xz_person_history)
                                  /max(1,len(_cl_xz|_flat_xz_person_history)))
                            _zs=(float(_rebuilt_person_bbox_3d['max'][2])
                                 -float(_rebuilt_person_bbox_3d['min'][2]))
                            _bz=(abs(_az['z_max']-_az['z_min']) if _az else 0.)
                            _zr=_zs/max(1.,_bz)
                            _fl=[]
                            if _iou<50. and len(_flat_xz_person_history)>200:
                                _fl.append(f"LOW-XZ-IOU={_iou:.0f}%")
                            if _zr<0.65 and _bz>20.: _fl.append(f"LOW-Z={_zr:.2f}")
                            logger.warning(f"[STEP3-VALIDATE] Frame {frame_num}: "
                                f"xz_iou={_iou:.0f}% z_ratio={_zr:.2f} "
                                f"{' '.join(_fl)}")
                        except Exception as _s3e:
                            logger.debug(f"[STEP3] failed:{_s3e}")

                    # ── Fix 2: X from actual cluster points (not projection) ──
                    _actual_x_min = float(np.min(_person_cluster_pts[:, 0]))
                    _actual_x_max = float(np.max(_person_cluster_pts[:, 0]))
                    _rebuilt_person_bbox_3d['min'][0] = _actual_x_min
                    _rebuilt_person_bbox_3d['max'][0] = _actual_x_max

                    # ── Fix 2b: Clamp X span to ≤70 cm (anatomically plausible). ──
                    # The raw cluster bbox can extend to the chair's X position
                    # when the person cluster's bounding box bleeds across (the
                    # F18/F20 pattern: cluster spans X=[-53,+74] = 127cm because
                    # the chair at X=+54 sits inside the bbox even though it is a
                    # separate cluster).  Cap matches the early-FIX-E threshold.
                    _MAX_PERSON_X_SPAN_CM = 70.0
                    _rebuild_x_span = _actual_x_max - _actual_x_min
                    if _rebuild_x_span > _MAX_PERSON_X_SPAN_CM:
                        _used_inv_x = False
                        # Prefer inventory X when its own span is plausible
                        if (state_bank is not None
                                and hasattr(state_bank, 'object_inventory')
                                and person_cluster_uuid
                                and person_cluster_uuid in state_bank.object_inventory):
                            _inv2 = state_bank.object_inventory[person_cluster_uuid]
                            _inv2_box = _inv2.get('last_3D_box', {})
                            if 'min' in _inv2_box and 'max' in _inv2_box:
                                _inv2_x_span = (_inv2_box['max'][0]
                                                - _inv2_box['min'][0])
                                if _inv2_x_span <= _MAX_PERSON_X_SPAN_CM:
                                    _actual_x_min = _inv2_box['min'][0]
                                    _actual_x_max = _inv2_box['max'][0]
                                    _rebuilt_person_bbox_3d['min'][0] = _actual_x_min
                                    _rebuilt_person_bbox_3d['max'][0] = _actual_x_max
                                    _used_inv_x = True
                                    logger.info(
                                        f"[FIX-E REBUILD] X clamped from inventory "
                                        f"(cluster span {_rebuild_x_span:.0f}cm "
                                        f"> {_MAX_PERSON_X_SPAN_CM:.0f}cm): "
                                        f"X=[{_actual_x_min:.0f},{_actual_x_max:.0f}]cm")
                        if not _used_inv_x:
                            # Fallback: cap symmetrically around cluster centroid X
                            _cx = float(np.mean(_person_cluster_pts[:, 0]))
                            _actual_x_min = _cx - _MAX_PERSON_X_SPAN_CM / 2.0
                            _actual_x_max = _cx + _MAX_PERSON_X_SPAN_CM / 2.0
                            _rebuilt_person_bbox_3d['min'][0] = _actual_x_min
                            _rebuilt_person_bbox_3d['max'][0] = _actual_x_max
                            logger.info(
                                f"[FIX-E REBUILD] X clamped +/-{_MAX_PERSON_X_SPAN_CM/2:.0f}cm "
                                f"around centroid (cluster span {_rebuild_x_span:.0f}cm "
                                f"> {_MAX_PERSON_X_SPAN_CM:.0f}cm): "
                                f"X=[{_actual_x_min:.0f},{_actual_x_max:.0f}]cm")
                    # ── END Fix 2b ────────────────────────────────────────────

                    logger.info(
                        f"[OK] FIX-E REBUILD frame {frame_num}: "
                        f"CoP Y=[{_cop_y_front:.0f},{_cop_y_back:.0f}] "
                        f"({len(_person_cluster_pts)} cluster pts), "
                        f"fresh 2D bbox=[{_rebuilt_pose_2d_bbox[0]:.0f},"
                        f"{_rebuilt_pose_2d_bbox[1]:.0f},"
                        f"{_rebuilt_pose_2d_bbox[2]:.0f},"
                        f"{_rebuilt_pose_2d_bbox[3]:.0f}], "
                        f"tunnel X=[{_actual_x_min:.0f},{_actual_x_max:.0f}] "
                        f"Y=[{_rebuilt_person_bbox_3d['min'][1]:.0f},"
                        f"{_rebuilt_person_bbox_3d['max'][1]:.0f}] "
                        f"Z=[{_actual_z_min:.0f},{_actual_z_max:.0f}]")

                    # Store authoritative person Y for next-frame continuity
                    # guard (see Y-CONTINUITY SANITY GUARD above).
                    process_single_frame._prev_person_cop_y = (
                        (_cop_y_front + _cop_y_back) / 2.0)
        except Exception as _e_rebuild:
            logger.warning(f"[FIX-E REBUILD] Frame {frame_num}: failed ({_e_rebuild}) "
                           f"-- falling back to early FIX-E person_context")

        # Pick the rebuilt values when available, else fall back to the
        # early FIX-E person_context values (preserves old behavior).
        # When falling back to projection-only tunnel, add 10cm X padding.
        _final_person_bbox_3d = _rebuilt_person_bbox_3d
        if _final_person_bbox_3d is None and person_context is not None:
            _final_person_bbox_3d = person_context.get('person_bbox_3d')
            if _final_person_bbox_3d is not None:
                # Projection-only fallback: add 10cm padding to each X side
                _final_person_bbox_3d = dict(_final_person_bbox_3d)
                _final_person_bbox_3d['min'] = list(_final_person_bbox_3d['min'])
                _final_person_bbox_3d['max'] = list(_final_person_bbox_3d['max'])
                _final_person_bbox_3d['min'][0] -= 10.0
                _final_person_bbox_3d['max'][0] += 10.0
                logger.info(
                    f"[TUNNEL] Frame {frame_num}: fallback projection + 10cm X padding -> "
                    f"X=[{_final_person_bbox_3d['min'][0]:.0f},{_final_person_bbox_3d['max'][0]:.0f}]")

        _final_pose_2d_bbox = _rebuilt_pose_2d_bbox
        if _final_pose_2d_bbox is None and person_context is not None:
            _final_pose_2d_bbox = person_context.get('pose_2d_bbox')

        # If rebuild was rejected (or failed) but we have a fallback bbox,
        # seed _prev_person_cop_y from it so the NEXT frame still has a
        # reference Y to continuity-check against. Don't overwrite if the
        # rebuild already stored its own (more accurate) value.
        if (not hasattr(process_single_frame, '_prev_person_cop_y')
                or getattr(process_single_frame, '_prev_person_cop_y', None)
                is None
                or _rebuilt_person_bbox_3d is None):
            if (_final_person_bbox_3d is not None
                    and 'min' in _final_person_bbox_3d
                    and 'max' in _final_person_bbox_3d):
                try:
                    _fb_y = (float(_final_person_bbox_3d['min'][1])
                             + float(_final_person_bbox_3d['max'][1])) / 2.0
                    # Sanity: only accept plausible person-height Y values
                    if 200.0 <= _fb_y <= 900.0:
                        process_single_frame._prev_person_cop_y = _fb_y
                except (TypeError, IndexError, ValueError):
                    pass

        # ==================================================================
        # ISSUE #4 — FILL-TO-COP (iterative stop condition)
        # The clustering loop and Y-INJECT have run.  If the person
        # cluster's XZ or YZ projection still falls short of the
        # dense-CoP loop in the person's territory, we keep going:
        # walk every raw CoP point inside the acceptance zone, and
        # inject any point whose XZ or YZ projection cell is a
        # dense-CoP cell that the cluster has not yet claimed.
        # The cluster terminates ONLY when both projections have
        # <= 5% unclaimed dense-CoP cells, or when an iteration
        # makes no progress.  Single-pass was insufficient (a single
        # pass leaves the same 60-70% unclaimed that the *first*
        # measurement reported).
        # Uses the SAME elevated density threshold (>=8 pts/cell)
        # and small-blob filter as the projection videos, so the
        # "unclaimed" set excludes the wall-stripe ghost.
        # ==================================================================
        try:
            if (person_cluster_uuid is not None
                    and person_cluster_uuid in current_clusters
                    and person_context is not None
                    and person_context.get('acceptance_zone')
                    and person_context['acceptance_zone'].get('active')
                    and voxel_grid is not None
                    and hasattr(voxel_grid, 'bounds')
                    and voxel_grid.bounds
                    and _final_person_bbox_3d is not None):

                _pc_obj   = current_clusters[person_cluster_uuid]
                _pc_pts   = _pc_obj.get('points')
                _f_az     = person_context['acceptance_zone']
                _f_origin = voxel_grid.bounds[0]
                _f_res    = float(getattr(voxel_grid, 'resolution', 2.0))
                _f_ox = float(_f_origin[0])
                _f_oy = float(_f_origin[1])
                _f_oz = float(_f_origin[2])

                def _pt_to_idx(_px, _py, _pz):
                    return (int((_px - _f_ox) / _f_res),
                            int((_py - _f_oy) / _f_res),
                            int((_pz - _f_oz) / _f_res))

                # Person territory: AZ widened by 10cm on X
                _ft_x_lo = float(min(_final_person_bbox_3d['min'][0],
                                     _f_az['x_min'])) - 10.0
                _ft_x_hi = float(max(_final_person_bbox_3d['max'][0],
                                     _f_az['x_max'])) + 10.0
                _ft_y_lo = float(_f_az['y_min'])
                _ft_y_hi = float(_f_az['y_max'])
                _ft_z_lo = float(_f_az['z_min'])
                _ft_z_hi = float(_f_az['z_max'])

                # ── CRITICAL: collect every OTHER cluster's voxel
                #    projection cells.  These are off-limits to
                #    FILL-TO-COP — they belong to another body, not
                #    the person.  Without this, FILL-TO-COP pulls
                #    chair-area dense-CoP into the person cluster
                #    and visualization renders two overlapping
                #    masses claiming the same physical territory.
                _other_xz = set(); _other_yz = set(); _other_xyz = set()
                for _ock, _ocd in current_clusters.items():
                    if _ock == person_cluster_uuid:
                        continue
                    # Voxel-data first (authoritative for what the
                    # renderer will draw)
                    _ovd = _ocd.get('voxel_data')
                    if isinstance(_ovd, dict) and _ovd:
                        for _ovk, _ovv in _ovd.items():
                            _oc = _ovv.get('centroid') if isinstance(_ovv, dict) else None
                            if isinstance(_oc, (list, tuple)) and len(_oc) >= 3:
                                _oxi, _oyi, _ozi = _pt_to_idx(
                                    float(_oc[0]), float(_oc[1]), float(_oc[2]))
                                _other_xz.add((_oxi, _ozi))
                                _other_yz.add((_oyi, _ozi))
                                _other_xyz.add((_oxi, _oyi, _ozi))
                    # Fallback: raw points
                    _opts = _ocd.get('points')
                    if _opts is not None and len(_opts) > 0:
                        try:
                            _opts_np = (np.asarray(_opts, dtype=np.float64)
                                        if not isinstance(_opts, np.ndarray)
                                        else _opts.astype(np.float64))
                            for _op in _opts_np:
                                _oxi, _oyi, _ozi = _pt_to_idx(
                                    float(_op[0]), float(_op[1]), float(_op[2]))
                                _other_xz.add((_oxi, _ozi))
                                _other_yz.add((_oyi, _ozi))
                                _other_xyz.add((_oxi, _oyi, _ozi))
                        except Exception:
                            pass
                _other_count = len(_other_xyz)

                # ── Build dense-CoP cells within person territory ──
                # Same threshold (>=8) as the projection video so what
                # the renderer draws and what we fill are identical.
                # Also drops tiny blobs (<6 cells) so isolated wall
                # specks at fixed depth don't enter the target set.
                _F_DENSE = 8  # overridden adaptively after _ft_pts is populated
                _ft_xz_cnt = {}
                _ft_yz_cnt = {}
                # Snapshot the raw CoP slice inside territory ONCE.
                _ft_pts = []
                for _pp in points:
                    _ppx = float(_pp[0]); _ppy = float(_pp[1]); _ppz = float(_pp[2])
                    if not (_ft_x_lo <= _ppx <= _ft_x_hi): continue
                    if not (_ft_y_lo <= _ppy <= _ft_y_hi): continue
                    if not (_ft_z_lo <= _ppz <= _ft_z_hi): continue
                    _xi, _yi, _zi = _pt_to_idx(_ppx, _ppy, _ppz)
                    _kxz = (_xi, _zi); _kyz = (_yi, _zi)
                    _ft_xz_cnt[_kxz] = _ft_xz_cnt.get(_kxz, 0) + 1
                    _ft_yz_cnt[_kyz] = _ft_yz_cnt.get(_kyz, 0) + 1
                    _ft_pts.append((_ppx, _ppy, _ppz, _xi, _yi, _zi))
                # Adaptive recompute: now _ft_pts and _ft_xz_cnt are populated
                _ft_avg_density = len(_ft_pts) / max(1, len(_ft_xz_cnt))
                _F_DENSE = max(3, min(8, int(_ft_avg_density * 0.6)))  # matches sidecar adaptive threshold
                _ft_xz_dense_raw = {k for k, n in _ft_xz_cnt.items() if n >= _F_DENSE}
                _ft_yz_dense_raw = {k for k, n in _ft_yz_cnt.items() if n >= _F_DENSE}

                # Drop dense-CoP specks (<6 cells, 4-connected)
                def _ftc_drop_small(cells_set, min_size=6):
                    if not cells_set:
                        return set()
                    rem = set(cells_set); kept = set()
                    while rem:
                        seed = next(iter(rem))
                        stk = [seed]; comp = set()
                        while stk:
                            c = stk.pop()
                            if c not in rem: continue
                            rem.remove(c); comp.add(c)
                            ca, cb = c
                            for nb in ((ca+1, cb), (ca-1, cb),
                                       (ca, cb+1), (ca, cb-1)):
                                if nb in rem: stk.append(nb)
                        if len(comp) >= min_size:
                            kept |= comp
                    return kept
                _ft_xz_dense = _ftc_drop_small(_ft_xz_dense_raw, 6)
                _ft_yz_dense = _ftc_drop_small(_ft_yz_dense_raw, 6)

                # ── Subtract OTHER clusters' projection cells ──
                # If a cell is already claimed by the chair (or any
                # other non-person cluster), it is NOT a person-cluster
                # gap.  This prevents FILL-TO-COP from pulling chair
                # material into the person cluster.
                _ft_xz_dense -= _other_xz
                _ft_yz_dense -= _other_yz

                if _ft_xz_dense or _ft_yz_dense:
                    # ── Initial cluster projection cells ──
                    # _cl_xz / _cl_yz are the 2D silhouette footprints
                    # (gap detection).  _cl_xyz is the 3D claimed-voxel
                    # set used by the proximity guard below so the fill
                    # can only grow OUTWARD FROM THE BODY, never teleport
                    # a far floor/shadow speck into the person just
                    # because it plugs a 2D projection gap.
                    _cl_xz = set(); _cl_yz = set(); _cl_xyz = set()
                    if _pc_pts is not None and len(_pc_pts) > 0:
                        _pc_pts_np = (np.asarray(_pc_pts, dtype=np.float64)
                                      if not isinstance(_pc_pts, np.ndarray)
                                      else _pc_pts.astype(np.float64))
                        for _cp in _pc_pts_np:
                            _xi, _yi, _zi = _pt_to_idx(
                                float(_cp[0]), float(_cp[1]), float(_cp[2]))
                            _cl_xz.add((_xi, _zi))
                            _cl_yz.add((_yi, _zi))
                            _cl_xyz.add((_xi, _yi, _zi))
                    else:
                        _pc_pts_np = np.empty((0, 3), dtype=np.float64)

                    _start_pts = int(len(_pc_pts_np))

                    # ── Iterative fill loop ──
                    _DONE_PCT     = 5.0   # stop at <= 5% unclaimed (both axes)
                    _MAX_ITERS    = 6
                    _STALL_LIMIT  = 2     # 2 iters with no new pts -> stop
                    _stall_count  = 0
                    _iter_log     = []
                    # 3D proximity guard: a candidate may only be claimed
                    # if its voxel is within _FILL_MAX_GAP voxels (Chebyshev)
                    # of an already-claimed voxel.  4 voxels ≈ 8cm at the
                    # 2cm grid — bridges real silhouette holes but blocks
                    # the ~50-voxel (100cm) jump to a floor speck that
                    # merely shares a projection column with the body.
                    _FILL_MAX_GAP = 4
                    _fill_rejected_far = 0

                    def _fill_near_body(_qx, _qy, _qz, _claimed, _gap):
                        for _ddx in range(-_gap, _gap + 1):
                            for _ddy in range(-_gap, _gap + 1):
                                for _ddz in range(-_gap, _gap + 1):
                                    if (_qx + _ddx, _qy + _ddy,
                                            _qz + _ddz) in _claimed:
                                        return True
                        return False

                    for _it in range(_MAX_ITERS):
                        _unclaimed_xz = _ft_xz_dense - _cl_xz
                        _unclaimed_yz = _ft_yz_dense - _cl_yz
                        _pct_xz = 100.0 * len(_unclaimed_xz) / max(1, len(_ft_xz_dense))
                        _pct_yz = 100.0 * len(_unclaimed_yz) / max(1, len(_ft_yz_dense))
                        _iter_log.append((_it, _pct_xz, _pct_yz, len(_pc_pts_np)))

                        if _pct_xz <= _DONE_PCT and _pct_yz <= _DONE_PCT:
                            break
                        if not _unclaimed_xz and not _unclaimed_yz:
                            break

                        # Find every raw CoP point in territory whose
                        # XZ or YZ cell is still unclaimed.  No per-iter
                        # cap: we want the loop to terminate by
                        # coverage, not by point count.
                        _add = []
                        for (_ppx, _ppy, _ppz, _xi, _yi, _zi) in _ft_pts:
                            # Reject points whose 3D voxel index is
                            # already claimed by another cluster.
                            # Without this guard, an "unclaimed
                            # projection cell" might still correspond
                            # to a 3D voxel owned by the chair (e.g.
                            # a column where chair material and
                            # FILL gap overlap at different Y depths).
                            if (_xi, _yi, _zi) in _other_xyz:
                                continue
                            if ((_xi, _zi) in _unclaimed_xz
                                    or (_yi, _zi) in _unclaimed_yz):
                                # 3D proximity guard: only accept if the
                                # candidate is attached to the body (or
                                # to material claimed earlier this fill).
                                # Blocks far floor/shadow voxels that fill
                                # a 2D gap but sit 100cm away in depth.
                                if not _fill_near_body(_xi, _yi, _zi,
                                                       _cl_xyz, _FILL_MAX_GAP):
                                    _fill_rejected_far += 1
                                    continue
                                _add.append((_ppx, _ppy, _ppz))

                        if not _add:
                            _stall_count += 1
                            if _stall_count >= _STALL_LIMIT:
                                break
                            continue

                        _add_np = np.asarray(_add, dtype=np.float64)
                        _pc_pts_np = np.vstack([_pc_pts_np, _add_np])
                        for (_xx, _yy, _zz) in _add:
                            _xi, _yi, _zi = _pt_to_idx(_xx, _yy, _zz)
                            _cl_xz.add((_xi, _zi))
                            _cl_yz.add((_yi, _zi))
                            _cl_xyz.add((_xi, _yi, _zi))
                        _stall_count = 0

                    _added = int(len(_pc_pts_np) - _start_pts)
                    if _added > 0:
                        _pc_obj['points']       = _pc_pts_np
                        _pc_obj['point_count']  = int(len(_pc_pts_np))
                        _pc_obj['total_points'] = int(len(_pc_pts_np))
                        _pc_obj['centroid']     = _pc_pts_np.mean(axis=0).tolist()
                        _pc_obj['bbox'] = {
                            'min': _pc_pts_np.min(axis=0).tolist(),
                            'max': _pc_pts_np.max(axis=0).tolist(),
                        }
                        _final_person_bbox_3d = {
                            'min': [float(_pc_obj['bbox']['min'][i]) for i in range(3)],
                            'max': [float(_pc_obj['bbox']['max'][i]) for i in range(3)],
                        }

                        # ── CRITICAL: also update voxel_data ──
                        # visualization.py reconstructs the cluster from
                        # voxel_data (utils.reconstruct_from_voxel_metadata),
                        # NOT from cluster['points'].  If we only update
                        # points, the renderer reads stale voxel_data and
                        # the filled material never appears on screen.
                        # Add every injected point's voxel index to the
                        # cluster's voxel_data dict with a minimal valid
                        # entry (centroid + total_voxels bump).  Key
                        # format matches clustering.py: str((vx, vy, vz)).
                        _vd = _pc_obj.get('voxel_data')
                        if not isinstance(_vd, dict):
                            _vd = {}
                        _injected_pts_for_vd = _pc_pts_np[_start_pts:]
                        # Aggregate injected pts by their voxel index to
                        # compute per-voxel centroids.
                        _vox_agg = {}
                        for _ip in _injected_pts_for_vd:
                            _ipx = float(_ip[0]); _ipy = float(_ip[1]); _ipz = float(_ip[2])
                            _vxi, _vyi, _vzi = _pt_to_idx(_ipx, _ipy, _ipz)
                            _vkey = str((_vxi, _vyi, _vzi))
                            if _vkey in _vox_agg:
                                _agg = _vox_agg[_vkey]
                                _agg[0] += _ipx; _agg[1] += _ipy; _agg[2] += _ipz
                                _agg[3] += 1
                            else:
                                _vox_agg[_vkey] = [_ipx, _ipy, _ipz, 1, _vxi, _vyi, _vzi]
                        _vd_new_count = 0
                        for _vkey, _agg in _vox_agg.items():
                            _vcx = _agg[0] / _agg[3]
                            _vcy = _agg[1] / _agg[3]
                            _vcz = _agg[2] / _agg[3]
                            if _vkey in _vd:
                                # Update centroid (weighted toward new
                                # points by their count) — keep any
                                # existing y_plane_* metadata.
                                _existing = _vd[_vkey]
                                _ec = _existing.get('centroid')
                                if isinstance(_ec, (list, tuple)) and len(_ec) >= 3:
                                    _existing['centroid'] = [
                                        round((_ec[0] + _vcx) / 2.0, 2),
                                        round((_ec[1] + _vcy) / 2.0, 2),
                                        round((_ec[2] + _vcz) / 2.0, 2),
                                    ]
                                else:
                                    _existing['centroid'] = [
                                        round(_vcx, 2), round(_vcy, 2), round(_vcz, 2)]
                            else:
                                _vd[_vkey] = {
                                    'voxel_idx': (int(_agg[4]), int(_agg[5]), int(_agg[6])),
                                    'centroid': [
                                        round(_vcx, 2), round(_vcy, 2), round(_vcz, 2)],
                                    'point_count': int(_agg[3]),
                                    'source': 'fill_to_cop',
                                }
                                _vd_new_count += 1
                        _pc_obj['voxel_data']    = _vd
                        _pc_obj['total_voxels']  = len(_vd)

                        _final_xz_pct = _iter_log[-1][1] if _iter_log else 0.0
                        _final_yz_pct = _iter_log[-1][2] if _iter_log else 0.0
                        _trace = " -> ".join(
                            f"i{i}:XZ{xz:.0f}%/YZ{yz:.0f}%"
                            for (i, xz, yz, _n) in _iter_log)
                        logger.warning(
                            f"[FILL-TO-COP] Frame {frame_num}: "
                            f"injected {_added} pts + {_vd_new_count} new voxels "
                            f"across {len(_iter_log)} iters ({_trace}); "
                            f"cluster pts {_start_pts} -> {len(_pc_pts_np)}, "
                            f"voxels -> {len(_vd)}; "
                            f"other_clusters_excluded={_other_count} cells; "
                            f"far_rejected={_fill_rejected_far}; "
                            f"final unclaimed XZ={_final_xz_pct:.0f}% "
                            f"YZ={_final_yz_pct:.0f}%")
                    else:
                        _final_xz_pct = _iter_log[-1][1] if _iter_log else 0.0
                        _final_yz_pct = _iter_log[-1][2] if _iter_log else 0.0
                        logger.info(
                            f"[FILL-TO-COP] Frame {frame_num}: cluster "
                            f"already fills dense-CoP "
                            f"(XZ unclaimed {_final_xz_pct:.0f}%, "
                            f"YZ unclaimed {_final_yz_pct:.0f}%) - no injection")
        except Exception as _e_fill:
            logger.debug(f"[FILL-TO-COP] failed: {_e_fill}")

        results_data = {
            "frame": frame_num,
            "total_points": len(points),
            "n_clusters": len(current_clusters),
            "clusters_info": {},
            "pose_2d": round_floats({k: v.tolist() if isinstance(v, np.ndarray) else v 
                       for k, v in pose_2d.items()}) if pose_2d and isinstance(pose_2d, dict) else None,
            "pose_3d": round_floats(pose_3d.tolist() if isinstance(pose_3d, np.ndarray) else pose_3d),
            "person_cluster_uuid": person_cluster_uuid,
            "pose_predicted": pose_was_predicted,
            "consecutive_rejections": consecutive_rejections,
            "segment_validation_applied": segment_validation_applied,
            "iccs": {
                "rotation_angle": round(iccs.get('rotation', {}).get('angle'), 1) if iccs and iccs.get('rotation', {}).get('angle') is not None else None,
                "facing": iccs.get('rotation', {}).get('facing') if iccs else None,
                "segment_count": iccs.get('segment_count', 0) if iccs else 0
            } if iccs else None,
            # ===================== NEW: Add facing_info =====================
            "facing_info": {
                "facing": facing_info.get('facing', 'toward_camera'),
                "angle": round(facing_info.get('angle', 0), 1),
                "confidence": round(facing_info.get('confidence', 0.5), 2),
                "stable": facing_info.get('stable', False),
                "frames_at_current": facing_info.get('frames_at_current', 0),
                # 24-state facing fields (from upgraded detect_facing_from_2d_pose)
                "body_yaw_deg": round(facing_info.get('body_yaw_deg', 0), 1) if facing_info.get('body_yaw_deg') is not None else None,
                "facing_24": facing_info.get('facing_24'),
                "fitting_path": facing_info.get('fitting_path'),
                "torso_twist_deg": round(facing_info.get('torso_twist_deg', 0), 1) if facing_info.get('torso_twist_deg') is not None else None,
                "head_twist_deg": round(facing_info.get('head_twist_deg', 0), 1) if facing_info.get('head_twist_deg') is not None else None,
                "total_twist_deg": round(facing_info.get('total_twist_deg', 0), 1) if facing_info.get('total_twist_deg') is not None else None,
                "width_ratios": facing_info.get('width_ratios'),
                "source": facing_info.get('source'),
            } if facing_info else None,
            # ================================================================
            # TUNNEL BBOX + 2D BBOX — for middle panel visualization
            # Rebuilt above from current-frame CoP min Y + fresh pose_2d
            # (FIX-E REBUILD).  Falls back to early person_context values
            # only if the rebuild failed.
            "person_bbox_3d": round_floats(_final_person_bbox_3d) if _final_person_bbox_3d else None,
            "pose_2d_bbox": round_floats(_final_pose_2d_bbox) if _final_pose_2d_bbox else None,
            # MP33 wrist landmarks for skeleton fitting arm extension
            "mp33_wrists": _mp33_wrists if '_mp33_wrists' in dir() else None,
        }

        # ==================================================================
        # ISSUE 1+2: COMPUTE BODY_YAW + BUILD ROTATED MANNEQUIN (Phase 1)
        # Uses chord_angles from per-Z slice analysis + MMPose facing info.
        # Mannequin is built, rotated, and written to results.json HERE —
        # not in a separate Phase 2 pass an hour later.
        # ==================================================================
        _mannequin_joints = None
        _body_yaw_from_chord = None
        _person_ck = None
        _person_cd = None

        # Find person cluster
        for _ck, _cd in current_clusters.items():
            if _cd.get('uuid') == person_cluster_uuid or _ck == person_cluster_uuid:
                _person_ck = _ck
                _person_cd = _cd
                break

        if _person_cd is not None:
            _chord_angles = _person_cd.get('chord_angles', [])
            _mean_chord = sum(_chord_angles) / len(_chord_angles) if _chord_angles else 0.0
            if not _chord_angles:
                logger.warning(
                    f"[ISSUE-1] Frame {frame_num}: person cluster {_person_ck} found "
                    f"but chord_angles EMPTY (keys: {[k for k in _person_cd.keys() if 'chord' in k or 'slice' in k]})")

            # Issue 1 Step 4: Quadrant from MMPose facing
            _trio_score = 0.0
            _side_bias = 0.0
            if pose_2d is not None:
                # pose_2d can be dict with 'keypoints' or raw array/list
                _raw_kps = pose_2d.get('keypoints') if isinstance(pose_2d, dict) else pose_2d
                if _raw_kps is not None:
                    _kps_arr = np.array(_raw_kps)
                    if _kps_arr.ndim == 2 and _kps_arr.shape[0] >= 17 and _kps_arr.shape[1] >= 3:
                        # trio_score: average confidence of nose(0), L_eye(1), R_eye(2)
                        _trio_score = float(np.mean(_kps_arr[:3, 2]))
                        # side_bias: L_shoulder - R_shoulder + L_hip - R_hip
                        _side_bias = (float(_kps_arr[5, 2]) - float(_kps_arr[6, 2]) +
                                      float(_kps_arr[11, 2]) - float(_kps_arr[12, 2]))

            # Issue 1 Step 5: body_yaw = base + side_sign × mean_chord_angle
            _base = 0.0 if _trio_score >= 0.78 else 180.0
            _side_sign = 1.0 if _side_bias > 0 else -1.0
            _body_yaw_from_chord = _base + _side_sign * _mean_chord

            # Normalize to [0, 360)
            _body_yaw_from_chord = _body_yaw_from_chord % 360.0

            logger.info(
                f"[ISSUE-1] Frame {frame_num}: body_yaw={_body_yaw_from_chord:.1f}deg "
                f"(base={_base:.0f}deg + sign={_side_sign:+.0f} x chord={_mean_chord:.1f}deg, "
                f"trio={_trio_score:.2f}, side_bias={_side_bias:.2f}, "
                f"slices={len(_chord_angles)})")

            # ── POSE-DB LATERAL override for body_yaw ──
            if (_early_pose_strategy is not None
                    and _early_pose_strategy.get('ok')
                    and _early_pose_strategy.get('depth_axis') == 'LATERAL'):
                _lat_yaw = None
                try:
                    _pose_3d_raw = results_data.get('pose_3d') if 'results_data' in dir() else None
                    if _pose_3d_raw is not None:
                        _p3d = np.array(_pose_3d_raw)
                        if _p3d.ndim == 2 and _p3d.shape[0] >= 7:
                            _lat_yaw = 90.0 if float(_p3d[5, 1]) < float(_p3d[6, 1]) else 270.0
                except Exception:
                    pass
                if _lat_yaw is None:
                    _lat_side = _utils_pose.derive_lateral_facing(pose_2d)
                    if _lat_side == 'side_left':
                        _lat_yaw = 90.0
                    elif _lat_side == 'side_right':
                        _lat_yaw = 270.0
                if _lat_yaw is not None:
                    logger.info(
                        f"[ISSUE-1] Frame {frame_num}: POSE-DB=LATERAL - "
                        f"overriding chord body_yaw {_body_yaw_from_chord:.1f}deg->{_lat_yaw:.1f}deg")
                    _body_yaw_from_chord = _lat_yaw

            # =====================================================================
            # ISSUE-2 ROTATION SOURCE: 2D-PIXEL-GEOMETRY (the camera image itself)
            # The 3D pipeline's depth signals (PCA, hip vector WCS, _calculate_body
            # _orientation_safe) all collapse near-constant due to MiDaS half-shell
            # geometry — every visible voxel is at roughly the same camera-facing
            # depth. The signal that actually varies continuously through body
            # rotation is the 2D MMPose pixel geometry from the camera image.
            #
            # Continuous side_factor (0=front/back, 1=full side) from shoulder pixel
            # width over person pixel height. Hemisphere (front vs back) from trio.
            # Side disambiguation (which side facing camera) from nose horizontal
            # offset relative to shoulder center.
            #
            # Canonical mannequin geometry: at θ=0, the canonical's raised left arm
            # projects to camera-LEFT (negative world X), which means the canonical
            # represents a back-facing body at θ=0 and a front-facing body at θ=180.
            # =====================================================================
            _body_yaw_for_mannequin = None
            _yaw_source = 'chord'
            _side_factor = None
            _side_sign_2d = None
            try:
                if pose_2d is not None:
                    _kps = None
                    if isinstance(pose_2d, dict):
                        _kps = pose_2d.get('keypoints', pose_2d.get('keypoints_raw'))
                    elif isinstance(pose_2d, (list, np.ndarray)):
                        _kps = pose_2d
                    if _kps is not None:
                        _kps_a = np.asarray(_kps)
                        if _kps_a.ndim == 2 and _kps_a.shape[0] >= 17 and _kps_a.shape[1] >= 2:
                            _l_sh = _kps_a[5][:2]
                            _r_sh = _kps_a[6][:2]
                            _nose = _kps_a[0][:2]
                            _l_ank = _kps_a[15][:2]
                            _r_ank = _kps_a[16][:2]
                            _shoulder_width_px = abs(float(_r_sh[0]) - float(_l_sh[0]))
                            _shoulder_cy = (float(_l_sh[1]) + float(_r_sh[1])) / 2.0
                            _ankle_cy = (float(_l_ank[1]) + float(_r_ank[1])) / 2.0
                            _person_height_px = abs(_ankle_cy - float(_nose[1]))
                            if _person_height_px > 20.0:
                                # Empirical front/back ratio ≈ 0.27 (shoulder span ~27% of
                                # person height in pixels for full front/back). Below that
                                # the body is rotating toward side-on.
                                _ratio_2d = _shoulder_width_px / _person_height_px
                                _side_factor = max(0.0, min(1.0, 1.0 - _ratio_2d / 0.27))
                                _shoulder_cx = (float(_l_sh[0]) + float(_r_sh[0])) / 2.0
                                _nose_offset_px = float(_nose[0]) - _shoulder_cx
                                # Pos = nose right of shoulder center = body's anatomical
                                # right is closer to camera = body rotated toward camera-right.
                                # In the canonical with face at +Y (back-facing default),
                                # rotating toward camera-right means rotating positive θ.
                                _side_sign_2d = 1.0 if _nose_offset_px >= 0 else -1.0
                                _trio_high = (_trio_score >= 0.78)
                                # Hemisphere baseline:
                                #   trio low  (back to camera)  → θ = 0  (canonical default)
                                #   trio high (face to camera)  → θ = 180 (canonical flipped)
                                _hemi_base = 180.0 if _trio_high else 0.0
                                # Side displacement: ±90° × side_factor, signed by which side
                                # of the body is leading toward the camera. Sign convention:
                                # for a back-facing body rotating right-shoulder-toward-camera,
                                # body Y-axis sweeps from +Y (away) toward +X (camera-right) →
                                # rotation θ increases (counter-clockwise viewed from above).
                                _side_offset = _side_sign_2d * 90.0 * _side_factor
                                _body_yaw_for_mannequin = (_hemi_base + _side_offset) % 360.0
                                _yaw_source = '2d_pixels'
            except Exception as _pix_e:
                logger.debug(f"[ISSUE-2] 2d_pixels derivation failed: {_pix_e}")
                _body_yaw_for_mannequin = None
            if _body_yaw_for_mannequin is None:
                _body_yaw_for_mannequin = _body_yaw_from_chord

            # ── POSE-DB LATERAL override for mannequin yaw ──
            if (_early_pose_strategy is not None
                    and _early_pose_strategy.get('ok')
                    and _early_pose_strategy.get('depth_axis') == 'LATERAL'):
                _lat_mannequin_yaw = _body_yaw_from_chord
                if abs(_body_yaw_for_mannequin - _lat_mannequin_yaw) > 30.0:
                    logger.warning(
                        f"[ISSUE-2] Frame {frame_num}: POSE-DB=LATERAL - "
                        f"overriding mannequin yaw "
                        f"{_body_yaw_for_mannequin:.1f}deg->{_lat_mannequin_yaw:.1f}deg")
                    _body_yaw_for_mannequin = _lat_mannequin_yaw
                    _yaw_source = '2d_pixels_lateral_override'

            # Issue 2: Build mannequin from cluster bbox, rotate by body_yaw
            _p_bbox = _person_cd.get('bbox')
            if _p_bbox is not None:
                # Apply locked height from inventory if available
                if state_bank and hasattr(state_bank, 'object_inventory'):
                    _p_uuid = _person_cd.get('uuid')
                    _inv = state_bank.object_inventory.get(_p_uuid, {})
                    _locked_h = _inv.get('locked_z_span')
                    if _locked_h is not None and _locked_h > 50:
                        _p_bbox = dict(_p_bbox)  # copy
                        _p_bbox['max'] = list(_p_bbox['max'])
                        _p_bbox['max'][2] = _p_bbox['min'][2] + _locked_h
                        logger.info(f"[ISSUE-2] Using locked height {_locked_h:.1f}cm")

                from visualization import (build_skeleton21_from_cluster_bbox,
                                           rotate_skeleton21_by_yaw,
                                           articulate_skeleton21_to_pose)
                _raw_mannequin = build_skeleton21_from_cluster_bbox(_p_bbox)
                if _raw_mannequin is not None:
                    # Articulate the canonical limbs onto the detected pose
                    # FIRST (relative limb bend, orientation-invariant), THEN
                    # apply the rigid yaw for global facing.  Without this the
                    # mannequin stays frozen in the canonical pose and only
                    # rotates as a sculpture.  pose_3d is the detected COCO
                    # 17-joint world pose; a missing/unusable pose_3d makes
                    # articulate_skeleton21_to_pose() a no-op (canonical kept).
                    # mp33_wrists supplies reliable 2D-landmark arm elevation
                    # (depth-axis-free), and _yellow_cred lets frames where the
                    # yellow silhouette is more credible than the 3D pose trust
                    # the landmarks more.
                    _art_landmarks_2d = (results_data.get('mp33_wrists')
                                         if 'results_data' in dir() else None)
                    _yellow_cred = 1.0
                    _articulated_mannequin = articulate_skeleton21_to_pose(
                        _raw_mannequin, pose_3d,
                        landmarks_2d=_art_landmarks_2d,
                        yellow_credibility=_yellow_cred)
                    _was_articulated = not np.allclose(
                        _articulated_mannequin, _raw_mannequin)
                    _mannequin_joints = rotate_skeleton21_by_yaw(
                        _articulated_mannequin, _body_yaw_for_mannequin)
                    _sf_str = (f"{_side_factor:.2f}"
                               if _side_factor is not None else "N/A")
                    _ss_str = (f"{_side_sign_2d:+.0f}"
                               if _side_sign_2d is not None else "N/A")
                    logger.info(
                        f"[ISSUE-2] Frame {frame_num}: mannequin built + "
                        f"{'articulated + ' if _was_articulated else ''}rotated "
                        f"({_body_yaw_for_mannequin:.1f}deg, source={_yaw_source}, "
                        f"articulated={_was_articulated}, "
                        f"side_factor={_sf_str}, side_sign={_ss_str}, "
                        f"trio={_trio_score:.2f}, "
                        f"chord_fallback={_body_yaw_from_chord:.1f}deg), "
                        f"21 joints written to results.json")

        # Add mannequin data to results
        results_data["rotated_mannequin_joints"] = (
            round_floats(_mannequin_joints.tolist()) if _mannequin_joints is not None else None)
        results_data["body_yaw_from_chord_deg"] = (
            round(_body_yaw_from_chord, 1) if _body_yaw_from_chord is not None else None)
        results_data["mannequin_height_cm"] = (
            float(_person_cd['bbox']['max'][2] - _person_cd['bbox']['min'][2])
            if _person_cd and _person_cd.get('bbox') else None)
        
        for cluster_key, cluster_data in current_clusters.items():
            cluster_uuid = cluster_data.get('uuid', 'UNKNOWN')
            
            # Sort voxel_data by Y, X, Z
            raw_voxel_data = cluster_data.get('voxel_data', {})
            sorted_voxel_data = sort_voxel_data_by_yxz(raw_voxel_data)
            sorted_voxel_data = round_floats(sorted_voxel_data)
            
            cluster_info = {
                "cluster_uuid": cluster_uuid,
                "total_points": cluster_data.get('total_points', 0),
                "original_points": cluster_data.get('original_points', 0),
                "dummy_points": cluster_data.get('dummy_points', 0),
                "chord_voxels": cluster_data.get('chord_voxels', 0),
                "total_voxels": cluster_data.get('total_voxels', 0),
                "voxel_data": sorted_voxel_data,
                "centroid": round_floats(cluster_data.get('centroid', [0, 0, 0])),
                "bbox": round_floats(cluster_data.get('bbox', {"min": [0, 0, 0], "max": [0, 0, 0]})),
                "is_historical_lock": state_bank.has_historical_lock(cluster_uuid) if state_bank else False,
                "volume_confidence": cluster_data.get('volume_confidence'),
                "volume_reason": cluster_data.get('volume_reason'),
                "xy_slice_splits": cluster_data.get('xy_slice_splits', 0),
                "xy_slice_total": cluster_data.get('xy_slice_total', 0),
                "yz_slice_splits": cluster_data.get('yz_slice_splits', 0),
                "yz_slice_total": cluster_data.get('yz_slice_total', 0),
                "chord_angles": round_floats(cluster_data.get('chord_angles', [])),
                "xy_ratios": round_floats(cluster_data.get('xy_ratios', [])),
                "is_person_region": cluster_data.get('is_person_region', False),
                "xy_slice_details": cluster_data.get('xy_slice_details', []),
                "yz_slice_details": cluster_data.get('yz_slice_details', []),
            }

            # ── YOLO label channel — from object_inventory (read-only) ──────────
            if state_bank and hasattr(state_bank, 'object_inventory'):
                _inv_e = state_bank.object_inventory.get(cluster_uuid, {})
                cluster_info['yolo_class_label'] = _inv_e.get('class_label')
                cluster_info['yolo_class_conf'] = round(
                    float(_inv_e.get('class_conf', 0.0)), 2)
                cluster_info['yolo_class_locked'] = bool(
                    _inv_e.get('_class_locked', False))

            results_data["clusters_info"][cluster_key] = cluster_info

        # ── Rigid-object OBBs (YOLO-locked non-person objects) ──────────────
        if hasattr(state_bank, 'rigid_primitives') and state_bank.rigid_primitives:
            results_data['rigid_primitives'] = {
                str(_uid): _prim.to_dict()
                for _uid, _prim in state_bank.rigid_primitives.items()
            }
        
        try:
            # Round ALL floats in results_data to 1 decimal place
            results_data = round_floats(results_data)
            
            # Write JSON with hybrid formatting:
            # - Top-level and cluster-level keys: readable with newlines
            # - voxel_data entries and leaf values: compact single-line
            def write_compact_json_fixed(data, filepath):
                # Readable hybrid layout: dicts indented one-per-line, numeric
                # arrays kept on ONE line ([33, 18, 61]) -- economical but NOT a
                # single giant line.  Normalize numpy/enum types to primitives
                # first (arrays -> lists so they get the one-line treatment),
                # then format via the module-level pretty-printer.
                import json
                _norm = json.loads(json.dumps(
                    data,
                    default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o)))
                with open(filepath, 'w') as f:
                    f.write(_format_compact_json(_norm))
            
            write_compact_json_fixed(results_data, results_file)
            logger.info(f"[OK] SAVED frame {frame_num} results to {results_file}")

            # ==============================================================
            # SIMULTANEOUS: results_s.json (skeleton fitting)
            # Run skeleton fitting for THIS frame immediately after
            # results.json — not in a separate pass an hour later.
            # ==============================================================
            try:
                import skeleton_fitting as _sf

                _surfaces_dir = os.path.join(args.output, "surfaces")
                _cam_params = {
                    'camera_position': getattr(args, 'camera_position', [-47.0, 28.0, -20.0]),
                    'camera_target':   getattr(args, 'camera_target', [-25.1, 123.8, -28.3]),
                    'focal_length':    getattr(args, 'focal_length', 27.5),
                    'field_of_view':   getattr(args, 'field_of_view', 66.0),
                    'panel_width':     getattr(args, 'image_size', (480, 864))[0],
                    'panel_height':    getattr(args, 'image_size', (480, 864))[1],
                }

                # Initialize MovementIndexEngine once (store on function attr)
                if not hasattr(process_single_frame, '_mi_engine'):
                    from movement_index import MovementIndexEngine, extract_limits_from_segments
                    _mi_limits = extract_limits_from_segments()
                    _fps = float(getattr(args, 'fps', 12.0))
                    process_single_frame._mi_engine = MovementIndexEngine(
                        joint_limits=_mi_limits, buffer_size=30, fps=_fps)
                    process_single_frame._sf_uuid_first_seen = {}
                    process_single_frame._sf_grid_origin = None

                # Grid origin (compute once)
                if process_single_frame._sf_grid_origin is None and voxel_grid is not None:
                    if hasattr(voxel_grid, 'bounds') and voxel_grid.bounds is not None:
                        _b = voxel_grid.bounds[0]
                        process_single_frame._sf_grid_origin = [float(_b[i]) for i in range(3)]

                # Find person cluster
                _sf_person_uuid, _sf_cluster_data = _sf._extract_person_cluster(results_data)
                if _sf_person_uuid is not None:
                    _sf_voxel_indices = _sf._extract_cluster_voxel_indices(_sf_cluster_data)
                    _sf_bbox = _sf._extract_cluster_bbox(_sf_cluster_data)
                    _sf_height = _sf._height_from_bbox(_sf_bbox)
                    _sf_facing = _sf._parse_facing(results_data.get('facing_info'))

                    # Geometric facing override
                    _sf_kp3d = _sf._build_keypoints_3d_mapping(results_data)
                    _sf_kp2d = _sf._build_keypoints_2d_mapping(results_data)
                    _sf_geo = _sf._geometric_facing(_sf_kp3d, results_data.get('pose_2d'))
                    if _sf_geo and _sf_geo != _sf_facing:
                        _sf_facing = _sf_geo

                    # PLY paths
                    _sf_poisson, _sf_balloon_path = _sf._resolve_ply_paths(
                        _surfaces_dir, _sf_person_uuid, frame_num)
                    _sf_balloon = _sf._load_trimesh_ply(_sf_balloon_path)

                    # Get/create skeleton
                    # BUG-5 FIX: Always use primary_subject_uuid as the canonical
                    # skeleton key.  When the UUID changes between frames due to a
                    # tracking failure, using the transient UUID creates a new
                    # skeleton chain with frame_count=0, producing multiple competing
                    # chains with wrong velocity/Kuramoto deltas (frames 50, 61, 79+
                    # in the reference run showed frame_count=76, 77, 18 simultaneously).
                    # The skeleton must always live under the invariant
                    # primary_subject_uuid so there is exactly one chain per person.
                    _sf_canonical_uuid = _sf_person_uuid
                    if state_bank is not None:
                        _psu = getattr(state_bank, 'primary_subject_uuid', None)
                        if _psu is not None:
                            _sf_canonical_uuid = _psu
                    _sf_is_first = _sf_canonical_uuid not in process_single_frame._sf_uuid_first_seen
                    process_single_frame._sf_uuid_first_seen[_sf_canonical_uuid] = True
                    _sf_skeleton, _sf_is_f1 = _sf._get_or_init_skeleton(
                        state_bank, _sf_canonical_uuid, _sf_height, frame_num, _sf_is_first)

                    # Body yaw from facing_info
                    _sf_body_yaw = None
                    _fi = results_data.get('facing_info')
                    if _fi and isinstance(_fi, dict):
                        _sf_body_yaw = _fi.get('body_yaw_deg')

                    # ── ISSUE-1 Defect-2: per-UUID TEMPORAL BODY_YAW LOCK ──────────
                    # _sf_body_yaw is the ONLY value that orients the skeleton's ICCS
                    # frame (and thus every DoF axis / ROM clamp).  chord/trio/POSE-DB
                    # corrupt it on the bad frames:
                    #   • POSE-DB false LATERAL -> hardcoded 90/270  (frames 36-48)
                    #   • chord_angles EMPTY    -> 'default' 0.0     (65-66, 73-109)
                    # So: accept the candidate only when RELIABLE and velocity-clamp it
                    # (a single bad frame can't snap the frame; a real turn still
                    # tracks over a few frames); otherwise HOLD the last good yaw.
                    # chord/POSE-DB are demoted to bootstrap-only (first sighting of a
                    # UUID + forced reseat via state_bank.yaw_force_bootstrap).
                    _YAW_MAX_RATE = 45.0   # deg/frame: real turns track, snaps clamped
                    _YAW_CONF_MIN = 0.30   # below this the facing estimate isn't trusted
                    if state_bank is not None:
                        if not hasattr(state_bank, 'last_good_body_yaw'):
                            state_bank.last_good_body_yaw = {}
                            state_bank.yaw_force_bootstrap = set()
                        _yaw_uuid = _sf_canonical_uuid
                        _yaw_src = (_fi.get('source', 'default')
                                    if (_fi and isinstance(_fi, dict)) else 'none')
                        _yaw_conf = (float(_fi.get('confidence', 0.5))
                                     if (_fi and isinstance(_fi, dict)) else 0.0)
                        _yaw_last = state_bank.last_good_body_yaw.get(_yaw_uuid)
                        _yaw_force_bs = _yaw_uuid in getattr(state_bank,
                                                             'yaw_force_bootstrap', set())
                        _yaw_unreliable = (_sf_body_yaw is None
                                           or _yaw_src in ('default', 'pose_db_override')
                                           or _yaw_conf < _YAW_CONF_MIN)
                        if _yaw_last is None or _yaw_force_bs:
                            # BOOTSTRAP: seed from injected value (chord/POSE-DB ok here)
                            _yaw_locked = float(_sf_body_yaw) if _sf_body_yaw is not None else 0.0
                            _yaw_reason = 'bootstrap'
                            state_bank.yaw_force_bootstrap.discard(_yaw_uuid)
                        elif _yaw_unreliable:
                            # HOLD: an unreliable yaw must not move the ICCS frame
                            _yaw_locked = float(_yaw_last)
                            _yaw_reason = f"hold(src={_yaw_src},conf={_yaw_conf:.2f})"
                        else:
                            # RELIABLE: follow, velocity-clamped (rejects instant snaps)
                            _yaw_step = (float(_sf_body_yaw) - float(_yaw_last) + 180.0) % 360.0 - 180.0
                            _yaw_clamped = max(-_YAW_MAX_RATE, min(_YAW_MAX_RATE, _yaw_step))
                            _yaw_locked = (float(_yaw_last) + _yaw_clamped) % 360.0
                            _yaw_reason = ('track' if abs(_yaw_step) <= _YAW_MAX_RATE
                                           else f"slew({_yaw_step:+.1f}->{_yaw_clamped:+.1f})")
                        state_bank.last_good_body_yaw[_yaw_uuid] = _yaw_locked
                        logger.info(
                            f"[YAW-LOCK] Frame {frame_num} uuid={str(_yaw_uuid)[:8]}: "
                            f"candidate={'None' if _sf_body_yaw is None else f'{float(_sf_body_yaw):.1f}deg'} "
                            f"src={_yaw_src} conf={_yaw_conf:.2f} -> "
                            f"locked={_yaw_locked:.1f}deg ({_yaw_reason})")
                        _sf_body_yaw = _yaw_locked
                    # ───────────────────────────────────────────────────────────────

                    # Spine curve from previous frame
                    _sf_spine = None
                    if state_bank and _sf_person_uuid:
                        _sf_spine = state_bank.get_previous_spine_curve(_sf_person_uuid)

                    # ── MP33 arm extension detection ──────────────────
                    _sf_arm_ext = False
                    _sf_mp33w = results_data.get('mp33_wrists')
                    if _sf_mp33w and isinstance(_sf_mp33w, dict):
                        _MP33_COCO = {
                            'L_wrist': 9, 'R_wrist': 10,
                            'L_elbow': 7, 'R_elbow': 8,
                        }
                        _sf_ov = 0
                        for _mn, _ci in _MP33_COCO.items():
                            _mp = _sf_mp33w.get(_mn)
                            if _mp is None:
                                continue
                            _mpx, _mpy = _mp['px'], _mp['py']
                            _mm = _sf_kp2d[_ci] if _ci < len(_sf_kp2d) else {}
                            _mmpx = _mm.get('middle_panel_pixel')
                            _do = _mmpx is None
                            if not _do:
                                _dd = ((_mpx - _mmpx[0])**2
                                       + (_mpy - _mmpx[1])**2)**0.5
                                _do = _dd > 25
                            if _do:
                                _sf_kp2d[_ci] = {
                                    'middle_panel_pixel': (_mpx, _mpy),
                                    'confidence': _mp.get('vis', 0.5),
                                    'source': 'mp33'}
                                if _ci < len(_sf_kp3d):
                                    _sf_kp3d[_ci] = {'world_pos': None}
                                _sf_ov += 1
                        if _sf_ov > 0:
                            logger.info(
                                f"[SIMULTANEOUS] Frame {frame_num}: "
                                f"[MP33-ARM] overrode {_sf_ov} keypoints")
                        _sf_is_lat = ('side' in str(_sf_facing).lower()
                                      or 'lateral' in str(_sf_facing).lower())
                        if _sf_is_lat and ('L_wrist' in _sf_mp33w
                                           or 'R_wrist' in _sf_mp33w):
                            _sf_arm_ext = True
                            logger.info(
                                f"[SIMULTANEOUS] Frame {frame_num}: "
                                f"[MP33-ARM] LATERAL arm extension - "
                                f"Option 4 will prefer p_near")

                    # ── Pose DoF from matched POSE_DB entry ──────────
                    _sf_pose_strat = results_data.get('pose_strategy')
                    _sf_pose_dof = {}
                    if _sf_pose_strat and isinstance(_sf_pose_strat, dict):
                        try:
                            from utils import pose_to_dof as _ptd
                            _sf_pose_dof = _ptd(_sf_pose_strat)
                            if _sf_pose_dof:
                                logger.info(
                                    f"[SIMULTANEOUS] Frame {frame_num}: "
                                    f"[POSE-DOF] {len(_sf_pose_dof)} segments "
                                    f"from pose #{_sf_pose_strat.get('db_row_id')} "
                                    f"\"{_sf_pose_strat.get('db_row_name', '?')}\"")
                        except Exception as _ptd_e:
                            logger.warning(f"[SIMULTANEOUS] pose_to_dof failed: {_ptd_e}")

                    # FIT
                    _sf_result = _sf.fit_frame(
                        skeleton=_sf_skeleton,
                        voxel_grid=voxel_grid,
                        keypoints_2d_mapping=_sf_kp2d,
                        keypoints_3d_mapping=_sf_kp3d,
                        cluster_voxel_indices=_sf_voxel_indices,
                        facing_direction=_sf_facing,
                        camera_params=_cam_params,
                        balloon_mesh=_sf_balloon,
                        poisson_ply_path=_sf_poisson,
                        frame_num=frame_num,
                        body_yaw_deg=_sf_body_yaw,
                        spine_curve=_sf_spine,
                        mp33_arm_extended=_sf_arm_ext,
                        pose_dof=_sf_pose_dof,
                    )

                    # Store spine curve
                    _sf_new_spine = _sf_result.get('spine_curve')
                    if _sf_new_spine is not None and state_bank and _sf_person_uuid:
                        state_bank.store_spine_curve(_sf_person_uuid, _sf_new_spine)

                    # Movement index
                    _sf_mi = _sf._compute_movement_index(
                        skeleton=_sf_skeleton, fit_result=_sf_result,
                        frame_num=frame_num, mi_engine=process_single_frame._mi_engine,
                        cluster_bbox=_sf_bbox, icp_transform=None)

                    # ── CONTROL FRAME OVERRIDE ──────────────────────────────
                    # Frames listed in _CONTROL_KP_WORLD replace the fitted
                    # result with manually verified world-21 positions and
                    # recalibrate bone_lengths on the skeleton so every
                    # subsequent frame inherits correct proportions.
                    # Updated on Sun Jun 14 19:16:08 2026
                    _ctrl_kp21 = _CONTROL_KP_WORLD.get(frame_num)
                    if _ctrl_kp21 is not None:
                        _sf_result['fitted_keypoints_world_21'] = _ctrl_kp21
                        _sf_result['success']     = True
                        _sf_result['avg_error_cm'] = 0.0
                        _sf_result['method']      = 'control_frame_manual'
                        if _sf_skeleton is not None:
                            try:
                                import numpy as _np_cf
                                _c = _np_cf.array(_ctrl_kp21)
                                def _cf_d(a, b):
                                    return float(_np_cf.linalg.norm(_c[a] - _c[b]))
                                _sf_skeleton.bone_lengths['shoulder_width'] = _cf_d(5,  6)
                                _sf_skeleton.bone_lengths['hip_width']      = _cf_d(11, 12)
                                _sf_skeleton.bone_lengths['thigh_l']        = _cf_d(11, 13)
                                _sf_skeleton.bone_lengths['thigh_r']        = _cf_d(12, 14)
                                _sf_skeleton.bone_lengths['shin_l']         = _cf_d(13, 15)
                                _sf_skeleton.bone_lengths['shin_r']         = _cf_d(14, 16)
                                _sf_skeleton.bone_lengths['upper_arm_l']    = _cf_d(5,  7)
                                _sf_skeleton.bone_lengths['upper_arm_r']    = _cf_d(6,  8)
                                _sf_skeleton.bone_lengths['forearm_l']      = _cf_d(7,  9)
                                _sf_skeleton.bone_lengths['forearm_r']      = _cf_d(8,  10)
                                _sf_skeleton.bone_lengths['lower_spine']    = _cf_d(19, 20)
                                _sf_skeleton.bone_lengths['upper_spine']    = _cf_d(20, 18)
                            except Exception as _cf_e:
                                logger.warning(
                                    f"[CONTROL-FRAME] Frame {frame_num}: "
                                    f"bone_lengths recalibration failed: {_cf_e}")
                        logger.warning(
                            f"[CONTROL-FRAME] Frame {frame_num}: "
                            f"world-21 manually injected, bone_lengths recalibrated "
                            f"(sho={_sf_skeleton.bone_lengths.get('shoulder_width',0):.1f}cm "
                            f"hip={_sf_skeleton.bone_lengths.get('hip_width',0):.1f}cm "
                            f"thigh={_sf_skeleton.bone_lengths.get('thigh_l',0):.1f}cm "
                            f"shin={_sf_skeleton.bone_lengths.get('shin_l',0):.1f}cm)")
                    # ── END CONTROL FRAME OVERRIDE ───────────────────────────

                    # Nearest cluster voxels
                    _sf_fitted21 = _sf_result.get('fitted_keypoints_world_21')
                    _sf_ncv = _sf._compute_nearest_cluster_voxels(
                        np.array(_sf_fitted21) if _sf_fitted21 is not None else np.zeros((21, 3)),
                        voxel_grid, _sf_voxel_indices)

                    # Pose angles
                    _sf_angles = _sf._extract_pose_angles(_sf_skeleton) if _sf_result.get('success') else {}

                    # WRITE results_s.json
                    _sf_ok = _sf._write_results_s(
                        results_dir=results_dir, frame_num=frame_num,
                        skeleton=_sf_skeleton, fit_result=_sf_result,
                        facing_direction=_sf_facing, voxel_grid=voxel_grid,
                        cluster_voxel_indices=_sf_voxel_indices,
                        mi_snapshot=_sf_mi,
                        pose_3d_fallback=results_data.get('pose_3d'),
                        grid_origin=process_single_frame._sf_grid_origin,
                        nearest_cluster_voxels=_sf_ncv,
                        pose_angles=_sf_angles,
                        cluster_bbox=_sf_bbox,
                        body_yaw_deg=_sf_body_yaw,
                        state_bank=state_bank)
                    if _sf_ok:
                        logger.info(f"[SIMULTANEOUS] Frame {frame_num}: results_s.json written")
                else:
                    logger.warning(f"[SIMULTANEOUS] Frame {frame_num}: no person UUID - results_s.json skipped")
            except ImportError:
                logger.warning(f"[SIMULTANEOUS] skeleton_fitting not available - results_s.json skipped")
            except Exception as _sf_exc:
                logger.warning(f"[SIMULTANEOUS] Frame {frame_num}: skeleton fitting failed: {_sf_exc}")

            # ==============================================================
            # SIMULTANEOUS: body_descriptor.json
            # Backward-looking 5-frame window (current + previous 4).
            # Uses the REAL body_descriptor functions (not _compute_descriptor
            # which doesn't exist).
            # ==============================================================
            try:
                import body_descriptor as _bd

                # Load last 4 frames + current from disk
                _bd_window_fns = [fn for fn in range(max(1, frame_num - 4), frame_num + 1)]
                _bd_frames = {}
                for _bfn in _bd_window_fns:
                    _bfd = _bd.load_frame_data(results_dir, _bfn)
                    if _bfd is not None:
                        _bd_frames[_bfn] = _bfd

                if _bd_frames and frame_num in _bd_frames:
                    # Build Y-wall stats for each frame in window
                    _bd_window_stats = []
                    for _bfn in sorted(_bd_frames.keys()):
                        _bvi = _bd._parse_voxel_keys(
                            _bd_frames[_bfn].get('voxel_data', {}))
                        _bgo = _bd_frames[_bfn].get('grid_origin')
                        _bres = _bd_frames[_bfn].get('grid_resolution_cm', 2.0)
                        _bd_window_stats.append(
                            _bd._compute_y_wall_stats_from_voxels(_bvi, _bgo, _bres))

                    # Compute descriptor using the real functions
                    _bd_metrics = _bd.compute_body_metrics(_bd_window_stats)
                    _bd_pose = _bd.classify_pose(_bd_window_stats, _bd_metrics)
                    _bd_movement = _bd.classify_movement(_bd_window_stats)
                    _bd_subclusters = _bd.compute_limb_subclusters(
                        _bd_window_stats, _bd_metrics)

                    _bd_center_fd = _bd_frames[frame_num]
                    _bd_center_yw = _bd._compute_y_wall_stats_from_voxels(
                        _bd._parse_voxel_keys(_bd_center_fd.get('voxel_data', {})),
                        _bd_center_fd.get('grid_origin'),
                        _bd_center_fd.get('grid_resolution_cm', 2.0))

                    _bd_ywall_cmp = _bd.count_ywall_comparison(
                        _bd_window_stats,
                        _bd_center_fd.get('joints', []),
                        _bd_metrics)

                    _bd_descriptor = {
                        'frame': frame_num,
                        'window_frames': sorted(_bd_frames.keys()),
                        'window_size': len(_bd_frames),
                        'body_metrics': _bd_metrics,
                        'pose_class': _bd_pose,
                        'movement_class': _bd_movement,
                        'limb_subclusters': _bd_subclusters,
                        'ywall_comparison': _bd_ywall_cmp,
                        'shell_ywall_count': _bd_center_yw.get('y_wall_count', 0),
                        'grid_origin': _bd_center_fd.get('grid_origin'),
                    }

                    _bd_ok = _bd._write_descriptor(results_dir, frame_num, _bd_descriptor)
                    if _bd_ok:
                        logger.info(f"[SIMULTANEOUS] Frame {frame_num}: body_descriptor.json written "
                                    f"(pose={_bd_pose}, move={_bd_movement})")

                        # Store in state_bank
                        if state_bank is not None and hasattr(state_bank, 'body_descriptors_by_frame'):
                            state_bank.body_descriptors_by_frame[frame_num] = _bd_descriptor
            except ImportError:
                logger.warning(f"[SIMULTANEOUS] body_descriptor module not available")
            except Exception as _bd_exc:
                logger.warning(f"[SIMULTANEOUS] Frame {frame_num}: body descriptor failed: {_bd_exc}")

            # ==============================================================
            # SIMULTANEOUS: results_p.json
            # Project fitted 3D keypoints to 2D middle-panel pixels using
            # the SAME project_3d_to_2d that draws the yellow contour.
            # ==============================================================
            try:
                _rp_path = os.path.join(results_dir, f"frame_{frame_num:03d}_results_p.json")
                _rp_fitted21 = None
                try:
                    _rp_fitted21 = _sf_result.get('fitted_keypoints_world_21')
                except NameError:
                    pass

                if _rp_fitted21 is not None:
                    from opencv_integration import project_3d_to_2d as _rp_proj
                    _rp_cam_pos = _cam_params['camera_position']
                    _rp_cam_tgt = _cam_params['camera_target']
                    _rp_fov = _cam_params['field_of_view']
                    _rp_img_size = (_cam_params['panel_width'], _cam_params['panel_height'])

                    _rp_pts_3d = np.array(_rp_fitted21)
                    _rp_pts_2d = _rp_proj(
                        _rp_pts_3d,
                        camera_position=_rp_cam_pos,
                        camera_target=_rp_cam_tgt,
                        field_of_view=_rp_fov,
                        image_size=_rp_img_size)

                    # =================================================================
                    # PHASE D v2 — 2D-anchor + temporal validator (last word, idempotent)
                    #
                    # Per KP, force every 3D 17-KP placement to satisfy two invariants:
                    #   (a) project(world_3d[i])  within  2 px  of  pose_2d[i]   (MMPose anchor)
                    #   (b) |world_3d_t[i] - world_3d_(t-1)[i]|  within  V cm/frame
                    #
                    # CHANGES vs v1:
                    #   - snap radius ALSO 2 px (was 3) — matches anchor tolerance
                    #   - snap picks voxel CLOSEST TO CAMERA (smallest distance from
                    #     camera origin) when multiple voxels project under the pixel,
                    #     so we land on the body's camera-facing surface, NOT on
                    #     chair voxels behind the body
                    #   - corrected XYZ is written BACK into the LOCAL `pose_3d` variable
                    #     AND _results.json is re-written so downstream visualization
                    #     (which reads pose_3d, NOT _p.json) sees corrected data
                    #
                    # Touches indices 0..16 only (the 17 COCO joints).  Indices 17..20
                    # are derived (head_center etc.) — left as-is.
                    # Re-running on already-fixed data is a no-op (every KP already
                    # passes the 2 px test, no modification fires).
                    # =================================================================
                    # Define a sentinel exception type so the gate can short-circuit
                    # silently without polluting the log with "validator failed: ..."
                    class _ValidatorDisabledSentinel(Exception):
                        pass

                    try:
                        # =============================================================
                        # GATE: PHASE D v2 here is DISABLED.  It anchored to the RAW
                        # MMPose pixel — wrong target.  The middle-panel magenta is
                        # drawn at  MMPose × FLAT_scale + (dx, dy)  using a per-frame
                        # FLAT transform fitted from yellow/red contours, which only
                        # exists inside visualization.py.  The validator MUST anchor
                        # to that FLAT-mapped target, not the raw MMPose pixel.
                        # The validator now lives in visualization.py at the right
                        # place (after kp_middle_by_coco is built, before STEP 8 blue
                        # draw) and rewrites _p.json from there so the temporal chain
                        # is anchored to the correct target on every frame.
                        # All variable names below are preserved for downstream code
                        # but the body short-circuits.
                        # =============================================================
                        _VALIDATOR_DISABLED_HERE = True
                        if _VALIDATOR_DISABLED_HERE:
                            logger.info(f"[VALIDATOR] Frame {frame_num}: PHASE D v2 here is DISABLED "
                                        f"- anchor lives in visualization.py (FLAT-mapped target)")
                            raise _ValidatorDisabledSentinel
                        _ANCHOR_PX_TOL = 2.0          # strict 2-pixel ceiling (HARD)
                        _SNAP_PX_RADIUS = 2.0          # voxel snap must be within this many px of MMPose pixel
                        _MIN_CONF = 0.30               # KPs with conf below this are not validated
                        # Local COCO_NAMES (the outer one is declared below the validator).
                        COCO_NAMES = [
                            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
                            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
                            'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
                            'head_center', 'shoulder_center', 'pelvis_center', 'spine_mid']
                        # Per-joint world-cm velocity caps (per frame, ~12 fps walk):
                        _VELOCITY_LIMITS_CM = {
                            0: 15, 1: 15, 2: 15, 3: 15, 4: 15,           # head group
                            5: 15, 6: 15,                                # shoulders
                            7: 22, 8: 22,                                # elbows
                            9: 30, 10: 30,                               # wrists
                            11: 15, 12: 15,                              # hips
                            13: 22, 14: 22,                              # knees
                            15: 30, 16: 30,                              # ankles
                        }

                        # Build the inverse-projection helper using the same camera
                        # parameters that produced _rp_pts_2d.
                        _vfwd = np.asarray(_rp_cam_tgt, dtype=float) - np.asarray(_rp_cam_pos, dtype=float)
                        _vfwd = _vfwd / max(np.linalg.norm(_vfwd), 1e-9)
                        _wup = np.array([0.0, 0.0, 1.0])
                        _vrt = np.cross(_vfwd, _wup)
                        if np.linalg.norm(_vrt) < 1e-9:
                            _wup = np.array([0.0, 1.0, 0.0])
                            _vrt = np.cross(_vfwd, _wup)
                        _vrt = _vrt / max(np.linalg.norm(_vrt), 1e-9)
                        _vup = np.cross(_vrt, _vfwd)
                        _vup = _vup / max(np.linalg.norm(_vup), 1e-9)
                        _img_w = float(_rp_img_size[0])
                        _img_h = float(_rp_img_size[1])
                        _fov_rad = float(_rp_fov) * np.pi / 180.0
                        _f_pix = (_img_h / 2.0) / np.tan(_fov_rad / 2.0)
                        _cam_origin = np.asarray(_rp_cam_pos, dtype=float)

                        def _project_one(world_xyz):
                            """Same camera as _rp_proj — project a single 3D point."""
                            v = np.asarray(world_xyz, dtype=float) - _cam_origin
                            x_l = float(np.dot(v, _vrt))
                            y_l = float(np.dot(v, _vup))
                            z_l = float(np.dot(v, _vfwd))
                            if z_l <= 1e-6:
                                return None
                            u = (x_l / z_l) * _f_pix + _img_w / 2.0
                            vv = -(y_l / z_l) * _f_pix + _img_h / 2.0
                            return (u, vv)

                        def _ray_point_at_y(u_tgt, v_tgt, y_target):
                            """Cast ray through pixel and return point with world Y = y_target."""
                            x_l = (u_tgt - _img_w / 2.0) / _f_pix
                            y_l = -(v_tgt - _img_h / 2.0) / _f_pix
                            z_l = 1.0
                            ray_dir_world = x_l * _vrt + y_l * _vup + z_l * _vfwd
                            ray_dir_world = ray_dir_world / max(np.linalg.norm(ray_dir_world), 1e-9)
                            denom = float(ray_dir_world[1])
                            if abs(denom) < 1e-6:
                                return None
                            t = (y_target - float(_cam_origin[1])) / denom
                            if t <= 0:
                                return None
                            P = _cam_origin + t * ray_dir_world
                            return (float(P[0]), float(P[1]), float(P[2]))

                        # Pre-project all cluster voxels once.
                        _voxel_proj_list = []
                        try:
                            if _sf_voxel_indices and voxel_grid is not None and \
                                    hasattr(voxel_grid, 'bounds') and voxel_grid.bounds is not None and \
                                    hasattr(voxel_grid, 'resolution') and voxel_grid.resolution:
                                _bmin = np.asarray(voxel_grid.bounds[0], dtype=float)
                                _vres = float(voxel_grid.resolution)
                                _voxel_iter = list(_sf_voxel_indices)
                                if len(_voxel_iter) > 8000:
                                    _step = max(1, len(_voxel_iter) // 8000)
                                    _voxel_iter = _voxel_iter[::_step]
                                _vox_world = np.array(
                                    [[_bmin[0] + (v[0] + 0.5) * _vres,
                                      _bmin[1] + (v[1] + 0.5) * _vres,
                                      _bmin[2] + (v[2] + 0.5) * _vres] for v in _voxel_iter],
                                    dtype=float)
                                _vox_proj = _rp_proj(
                                    _vox_world,
                                    camera_position=_rp_cam_pos,
                                    camera_target=_rp_cam_tgt,
                                    field_of_view=_rp_fov,
                                    image_size=_rp_img_size)
                                # Per-voxel distance from camera (for "closest to camera" tie-break)
                                _vox_camdist = np.linalg.norm(_vox_world - _cam_origin, axis=1)
                                _voxel_proj_list = (_vox_world, _vox_proj, _vox_camdist)
                        except Exception as _vox_err:
                            logger.debug(f"[VALIDATOR] Frame {frame_num}: voxel pre-project failed: {_vox_err}")
                            _voxel_proj_list = []

                        # Load previous frame's _p.json for temporal anchor (if exists).
                        # Note: previous _p.json was already corrected by validator on
                        # previous frame, so this history IS trustworthy.
                        _prev_world_3d = {}
                        try:
                            _prev_path = os.path.join(results_dir, f"frame_{frame_num - 1:03d}_results_p.json")
                            if frame_num > 1 and os.path.exists(_prev_path):
                                import json as _json_prev
                                with open(_prev_path, 'r') as _pf:
                                    _prev_data = _json_prev.load(_pf)
                                for _entry in _prev_data.get('keypoints_2d_mapping', []):
                                    _idx = _entry.get('keypoint_idx')
                                    _w3 = _entry.get('world_3d')
                                    if isinstance(_idx, int) and _w3 and len(_w3) == 3:
                                        _prev_world_3d[_idx] = (float(_w3[0]), float(_w3[1]), float(_w3[2]))
                        except Exception as _pe:
                            logger.debug(f"[VALIDATOR] Frame {frame_num}: prev _p.json load failed: {_pe}")
                            _prev_world_3d = {}

                        # Pull MMPose 2D keypoints (same camera as _rp_proj per log line:
                        # "MMPose initialized with camera: pos=..., target=..., fov=...").
                        _mm_kps = None
                        try:
                            _pose2d_block = results_data.get('pose_2d') if isinstance(results_data, dict) else None
                            if _pose2d_block and isinstance(_pose2d_block, dict):
                                _mm_kps = _pose2d_block.get('keypoints')
                        except Exception:
                            _mm_kps = None

                        # Per-KP validation loop (only the 17 COCO joints)
                        _n_corrected = 0
                        _n_skipped_lowconf = 0
                        _n_skipped_no_anchor = 0
                        _n_already_ok = 0
                        _corrected_indices = []
                        if _mm_kps is not None and len(_mm_kps) >= 17:
                            for _ki in range(17):
                                _kp_mm = _mm_kps[_ki]
                                if not (isinstance(_kp_mm, (list, tuple)) and len(_kp_mm) >= 2):
                                    _n_skipped_no_anchor += 1
                                    continue
                                _u_mm = float(_kp_mm[0]); _v_mm = float(_kp_mm[1])
                                _conf = float(_kp_mm[2]) if len(_kp_mm) >= 3 else 1.0
                                if _conf < _MIN_CONF:
                                    _n_skipped_lowconf += 1
                                    continue

                                _u_3d = float(_rp_pts_2d[_ki][0])
                                _v_3d = float(_rp_pts_2d[_ki][1])
                                _du = _u_3d - _u_mm
                                _dv = _v_3d - _v_mm
                                _gap = (_du * _du + _dv * _dv) ** 0.5

                                if _gap <= _ANCHOR_PX_TOL:
                                    _n_already_ok += 1
                                    continue

                                # Need fix.  Seed depth = current Y.
                                _y_seed = float(_rp_pts_3d[_ki][1])

                                # Step 4: voxel snap with "closest to camera" tie-break
                                _candidate = None
                                _src = None
                                if isinstance(_voxel_proj_list, tuple) and len(_voxel_proj_list) == 3:
                                    _vw, _vp, _vd = _voxel_proj_list
                                    _du_v = _vp[:, 0] - _u_mm
                                    _dv_v = _vp[:, 1] - _v_mm
                                    _d2 = _du_v * _du_v + _dv_v * _dv_v
                                    _under_mask = _d2 <= (_SNAP_PX_RADIUS * _SNAP_PX_RADIUS)
                                    if np.any(_under_mask):
                                        _under_world = _vw[_under_mask]
                                        _under_camdist = _vd[_under_mask]
                                        # Camera-facing surface: smallest distance from camera
                                        _best = int(np.argmin(_under_camdist))
                                        _candidate = (float(_under_world[_best, 0]),
                                                      float(_under_world[_best, 1]),
                                                      float(_under_world[_best, 2]))
                                        _src = 'snap'

                                # Step 3 fallback: ray at seed Y
                                if _candidate is None:
                                    _ray_pt = _ray_point_at_y(_u_mm, _v_mm, _y_seed)
                                    if _ray_pt is not None:
                                        _candidate = _ray_pt
                                        _src = 'ray'

                                if _candidate is None:
                                    logger.warning(
                                        f"[VALIDATOR] Frame {frame_num} KP{_ki} ({COCO_NAMES[_ki]}): "
                                        f"cannot construct candidate (Y_seed={_y_seed:.1f})")
                                    continue

                                # Step 5: temporal anchor
                                _vlim = _VELOCITY_LIMITS_CM.get(_ki, 25)
                                if _ki in _prev_world_3d:
                                    _pw = _prev_world_3d[_ki]
                                    _step_dist = ((_candidate[0] - _pw[0]) ** 2 +
                                                  (_candidate[1] - _pw[1]) ** 2 +
                                                  (_candidate[2] - _pw[2]) ** 2) ** 0.5
                                    if _step_dist > _vlim:
                                        _blended = (0.7 * _pw[0] + 0.3 * _candidate[0],
                                                    0.7 * _pw[1] + 0.3 * _candidate[1],
                                                    0.7 * _pw[2] + 0.3 * _candidate[2])
                                        _candidate = _blended
                                        _src = _src + '+blend'

                                # Step 6: must reproject within 2 px.  If not, fall back
                                # to pure ray (mathematically guaranteed sub-pixel).
                                _proj_check = _project_one(_candidate)
                                _final = _candidate
                                if _proj_check is None:
                                    _ray_pt = _ray_point_at_y(_u_mm, _v_mm, _y_seed)
                                    if _ray_pt is not None:
                                        _final = _ray_pt
                                        _src = 'ray-fallback'
                                        _proj_check = _project_one(_final)
                                if _proj_check is not None:
                                    _resid = ((_proj_check[0] - _u_mm) ** 2 +
                                              (_proj_check[1] - _v_mm) ** 2) ** 0.5
                                    if _resid > _ANCHOR_PX_TOL:
                                        _ray_pt = _ray_point_at_y(_u_mm, _v_mm, _y_seed)
                                        if _ray_pt is not None:
                                            _final = _ray_pt
                                            _src = (_src or 'ray') + '/ray-resid'
                                            _proj_check = _project_one(_final)

                                # Commit
                                if _final is not None and _proj_check is not None:
                                    _rp_pts_3d[_ki, 0] = float(_final[0])
                                    _rp_pts_3d[_ki, 1] = float(_final[1])
                                    _rp_pts_3d[_ki, 2] = float(_final[2])
                                    _rp_pts_2d[_ki, 0] = float(_proj_check[0])
                                    _rp_pts_2d[_ki, 1] = float(_proj_check[1])
                                    _resid_after = ((_proj_check[0] - _u_mm) ** 2 +
                                                    (_proj_check[1] - _v_mm) ** 2) ** 0.5
                                    _n_corrected += 1
                                    _corrected_indices.append((_ki, _final))
                                    logger.info(
                                        f"[VALIDATOR] Frame {frame_num} KP{_ki:2d} ({COCO_NAMES[_ki]:14s}) "
                                        f"src={_src:20s} before delta=({_du:+6.1f},{_dv:+6.1f}) |delta|={_gap:5.1f}  "
                                        f"after |delta|={_resid_after:4.1f}")
                        else:
                            _n_skipped_no_anchor = 17

                        logger.info(
                            f"[VALIDATOR] Frame {frame_num}: corrected={_n_corrected} "
                            f"already_ok={_n_already_ok} lowconf={_n_skipped_lowconf} "
                            f"no_anchor={_n_skipped_no_anchor}")

                        # =============================================================
                        # WRITEBACK: propagate corrections to LOCAL pose_3d AND to disk
                        # so downstream visualization (which reads pose_3d, not _p.json)
                        # renders the corrected blue 3D 17-KP skeleton.
                        # =============================================================
                        if _n_corrected > 0:
                            try:
                                # 1) Update the local `pose_3d` variable (used by frame_data
                                #    and any subsequent in-process consumer)
                                if pose_3d is not None:
                                    if isinstance(pose_3d, np.ndarray):
                                        for _ki, _final in _corrected_indices:
                                            if _ki < len(pose_3d):
                                                pose_3d[_ki, 0] = float(_final[0])
                                                pose_3d[_ki, 1] = float(_final[1])
                                                pose_3d[_ki, 2] = float(_final[2])
                                    elif isinstance(pose_3d, list):
                                        for _ki, _final in _corrected_indices:
                                            if _ki < len(pose_3d):
                                                pose_3d[_ki] = [float(_final[0]),
                                                                 float(_final[1]),
                                                                 float(_final[2])]
                                    logger.info(f"[VALIDATOR] Frame {frame_num}: "
                                                f"updated local pose_3d ({_n_corrected} KPs)")

                                # 2) Update results_data['pose_3d'] in-place
                                if isinstance(results_data, dict) and 'pose_3d' in results_data:
                                    _rd_p3 = results_data['pose_3d']
                                    if isinstance(_rd_p3, list):
                                        for _ki, _final in _corrected_indices:
                                            if _ki < len(_rd_p3):
                                                _rd_p3[_ki] = [round(float(_final[0]), 1),
                                                                round(float(_final[1]), 1),
                                                                round(float(_final[2]), 1)]
                                        logger.info(f"[VALIDATOR] Frame {frame_num}: "
                                                    f"updated results_data['pose_3d']")

                                # 3) Re-write _results.json so visualization (which loads
                                #    from disk via frame_data_provider OR direct read)
                                #    sees corrected pose_3d
                                try:
                                    _rfile = os.path.join(results_dir,
                                                          f"frame_{frame_num:03d}_results.json")
                                    if os.path.exists(_rfile):
                                        # Load existing file, patch pose_3d, write back.
                                        # Preserves all other fields (clusters, iccs, facing_info, ...).
                                        import json as _json_rw
                                        with open(_rfile, 'r') as _rf:
                                            _disk_data = _json_rw.load(_rf)
                                        if 'pose_3d' in _disk_data and isinstance(_disk_data['pose_3d'], list):
                                            for _ki, _final in _corrected_indices:
                                                if _ki < len(_disk_data['pose_3d']):
                                                    _disk_data['pose_3d'][_ki] = [
                                                        round(float(_final[0]), 1),
                                                        round(float(_final[1]), 1),
                                                        round(float(_final[2]), 1)]
                                            with open(_rfile, 'w') as _rwf:
                                                _rwf.write(_utils_pose._utils_format_compact_json(_disk_data, 0))
                                            logger.info(f"[VALIDATOR] Frame {frame_num}: "
                                                        f"re-wrote {_rfile} with corrected pose_3d")
                                except Exception as _rwe:
                                    logger.warning(f"[VALIDATOR] Frame {frame_num}: "
                                                   f"results.json re-write failed: {_rwe}")
                            except Exception as _wb_err:
                                logger.warning(f"[VALIDATOR] Frame {frame_num}: writeback failed: {_wb_err}")
                                import traceback as _tb_wb
                                logger.debug(_tb_wb.format_exc())
                    except _ValidatorDisabledSentinel:
                        # Gate fired — validator is intentionally off here.  Already
                        # logged the disable message; nothing further to do.
                        pass
                    except Exception as _val_err:
                        logger.warning(f"[VALIDATOR] Frame {frame_num}: validator failed: {_val_err}")
                        import traceback as _tb_v
                        logger.debug(_tb_v.format_exc())

                    COCO_NAMES = [
                        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
                        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
                        'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
                        'head_center', 'shoulder_center', 'pelvis_center', 'spine_mid']

                    _rp_2d_mapping = []
                    _rp_3d_mapping = []
                    for _ki in range(min(21, len(_rp_pts_3d))):
                        _rp_2d_mapping.append({
                            'keypoint_idx': _ki,
                            'keypoint_name': COCO_NAMES[_ki] if _ki < len(COCO_NAMES) else f'kp{_ki}',
                            'middle_panel_pixel': [round(float(_rp_pts_2d[_ki][0]), 1),
                                                    round(float(_rp_pts_2d[_ki][1]), 1)]
                                                   if _ki < len(_rp_pts_2d) else None,
                            'world_3d': [round(float(_rp_pts_3d[_ki][j]), 1) for j in range(3)],
                        })
                        _rp_3d_mapping.append({
                            'keypoint_idx': _ki,
                            'keypoint_name': COCO_NAMES[_ki] if _ki < len(COCO_NAMES) else f'kp{_ki}',
                            'position_3d': [round(float(_rp_pts_3d[_ki][j]), 1) for j in range(3)],
                        })

                    import json as _json_rp
                    _rp_data = {
                        'frame': frame_num,
                        'keypoints_2d_mapping': _rp_2d_mapping,
                        'keypoints_3d_mapping': _rp_3d_mapping,
                    }
                    with open(_rp_path, 'w') as _rpf:
                        _rpf.write(_utils_pose._utils_format_compact_json(_rp_data, 0))
                    logger.info(f"[SIMULTANEOUS] Frame {frame_num}: results_p.json written")
            except Exception as _rp_exc:
                logger.debug(f"[SIMULTANEOUS] Frame {frame_num}: results_p failed: {_rp_exc}")
            
            if frame_num > 1 and state_bank:
                prev_frame = frame_num - 1
                if prev_frame in state_bank.frame_clusters:
                    prev_uuids = set(state_bank.frame_clusters[prev_frame].keys())
                    curr_uuids = set(cluster_data.get('uuid', '') for cluster_data in current_clusters.values())
                    
                    maintained = prev_uuids & curr_uuids
                    lost = prev_uuids - curr_uuids
                    new = curr_uuids - prev_uuids
                    
                    logger.info(f"UUID Consistency Check (Frame {prev_frame}->{frame_num}):")
                    logger.info(f"  Maintained: {len(maintained)} clusters")
                    if lost:
                        logger.warning(f"  Lost: {len(lost)} clusters: {[u[:8] for u in lost]}")
                    if new:
                        logger.info(f"  New: {len(new)} clusters: {[u[:8] for u in new]}")
                    
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
        
        # ============================================================
        # STEP 11: RETURN FRAME RESULTS
        if voxel_grid is not None and hasattr(voxel_grid, 'bounds') \
                and voxel_grid.bounds is not None and current_clusters:
            try:
                _ls_origin = np.asarray(voxel_grid.bounds[0])
                _ls_res = float(getattr(voxel_grid, 'resolution', 2.0))
                _ls_ox = float(_ls_origin[0])
                _ls_oy = float(_ls_origin[1])
                _ls_oz = float(_ls_origin[2])
                _ls_r = _ls_res

                # Live CoP with density counting for lasso
                _ls_cop_xz=set(); _ls_cop_yz=set()
                _ls_cop_xz_cnt={}; _ls_cop_yz_cnt={}
                for _cp in points:
                    _xi=int((float(_cp[0])-_ls_ox)/_ls_r)
                    _yi=int((float(_cp[1])-_ls_oy)/_ls_r)
                    _zi=int((float(_cp[2])-_ls_oz)/_ls_r)
                    _ls_cop_xz.add((_xi,_zi)); _ls_cop_yz.add((_yi,_zi))
                    _kx=(_xi,_zi); _ky=(_yi,_zi)
                    _ls_cop_xz_cnt[_kx]=_ls_cop_xz_cnt.get(_kx,0)+1
                    _ls_cop_yz_cnt[_ky]=_ls_cop_yz_cnt.get(_ky,0)+1
                # Density threshold (Issue: ghost stripe).  5 pts/cell
                # admits a column of wall/door background voxels at a
                # fixed depth as "dense", producing the tall thin green
                # stripe seen in the projection videos.  Raise to 8 and
                # additionally drop tiny dense-CoP blobs (<6 cells), so
                # only real bodies stay in the dense-CoP set.
                _avg_cop_density = len(points) / max(1, len(_ls_cop_xz))
                _LD = max(3, min(8, int(_avg_cop_density * 0.6)))  # adaptive: scales [3..8] with pts/cell
                _ls_cop_xz_dense_raw = {k for k, n in _ls_cop_xz_cnt.items() if n >= _LD}
                _ls_cop_yz_dense_raw = {k for k, n in _ls_cop_yz_cnt.items() if n >= _LD}

                # 4-connected blob filter — drop specks (<6 cells)
                def _ls_drop_small(cells_set, min_size=6):
                    if not cells_set:
                        return set()
                    rem = set(cells_set); kept = set()
                    while rem:
                        seed = next(iter(rem))
                        stk = [seed]; comp = set()
                        while stk:
                            c = stk.pop()
                            if c not in rem: continue
                            rem.remove(c); comp.add(c)
                            ca, cb = c
                            for nb in ((ca+1, cb), (ca-1, cb),
                                       (ca, cb+1), (ca, cb-1)):
                                if nb in rem: stk.append(nb)
                        if len(comp) >= min_size:
                            kept |= comp
                    return kept
                _ls_cop_xz_dense = _ls_drop_small(_ls_cop_xz_dense_raw, min_size=6)
                _ls_cop_yz_dense = _ls_drop_small(_ls_cop_yz_dense_raw, min_size=6)

                # 4-connected components helper (same as STEP 9.65)
                def _ls_cc4(cells):
                    if not cells:
                        return 0
                    rem = set(cells); n = 0
                    while rem:
                        seed = next(iter(rem)); stk = [seed]; n += 1
                        while stk:
                            c = stk.pop()
                            if c not in rem: continue
                            rem.remove(c)
                            ca, cb = c
                            for nb in ((ca+1,cb), (ca-1,cb),
                                       (ca,cb+1), (ca,cb-1)):
                                if nb in rem: stk.append(nb)
                    return n

                def _ls_voxel_indices(cdata):
                    _vi = cdata.get('voxel_indices')
                    if _vi:
                        return _vi
                    _vd = cdata.get('voxel_data')
                    if _vd:
                        out = []
                        for vk in _vd.keys():
                            try:
                                clean = vk.strip("() ")
                                parts = clean.split(",")
                                out.append((int(parts[0].strip()),
                                            int(parts[1].strip()),
                                            int(parts[2].strip())))
                            except Exception:
                                pass
                        return out
                    _pts = cdata.get('points')
                    if _pts is not None and len(_pts) > 0:
                        out = []
                        for pp in _pts:
                            out.append((
                                int((float(pp[0]) - _ls_ox) / _ls_r),
                                int((float(pp[1]) - _ls_oy) / _ls_r),
                                int((float(pp[2]) - _ls_oz) / _ls_r)))
                        return out
                    return []

                _ls_all_cluster_xz = set()
                _ls_all_cluster_yz = set()
                _ls_clusters_payload = []
                _ls_uuids_in_order = []
                _ls_visualized_count = 0  # parity with visualization.py

                for _ls_ck, _ls_cd in current_clusters.items():
                    _vi = _ls_voxel_indices(_ls_cd)
                    if not _vi:
                        continue
                    _xz_set = set()
                    _yz_set = set()
                    for _v in _vi:
                        _vt = tuple(_v) if isinstance(_v, list) else _v
                        if len(_vt) >= 3:
                            _xz_set.add((_vt[0], _vt[2]))
                            _yz_set.add((_vt[1], _vt[2]))
                    _xz_blobs = _ls_cc4(_xz_set)
                    _yz_blobs = _ls_cc4(_yz_set)
                    _xz_cov = len(_xz_set & _ls_cop_xz)
                    _yz_cov = len(_yz_set & _ls_cop_yz)

                    _is_person = (
                        person_cluster_uuid is not None
                        and (_ls_cd.get('uuid') == person_cluster_uuid
                             or _ls_ck == person_cluster_uuid))

                    # Visualization parity: visualization.py counts a
                    # cluster as "visible" when its projected points
                    # render successfully — which depends on having
                    # voxel_data/voxel_metadata that survives projection.
                    # Approximate this here: a cluster is "visualized"
                    # when its XZ footprint has at least 5 cells (i.e.
                    # the projection is large enough to render as a
                    # blob, not a stray dot).
                    if len(_xz_set) >= 5:
                        _ls_visualized_count += 1

                    _ls_clusters_payload.append({
                        'key':        _ls_ck,
                        'uuid':       _ls_cd.get('uuid', ''),
                        'is_person':  bool(_is_person),
                        'xz':         [int(v) for c in _xz_set for v in c],
                        'yz':         [int(v) for c in _yz_set for v in c],
                        'xz_blobs':   int(_xz_blobs),
                        'yz_blobs':   int(_yz_blobs),
                        'xz_cov':     int(_xz_cov),
                        'yz_cov':     int(_yz_cov),
                        'xz_cell_count': int(len(_xz_set)),
                        'yz_cell_count': int(len(_yz_set)),
                    })
                    _ls_all_cluster_xz |= _xz_set
                    _ls_all_cluster_yz |= _yz_set
                    _ls_uuids_in_order.append(_ls_cd.get('uuid', '') or _ls_ck)

                _ls_unowned_xz = _ls_cop_xz - _ls_all_cluster_xz
                _ls_unowned_yz = _ls_cop_yz - _ls_all_cluster_yz
                _ls_cop_xz_n = max(1, len(_ls_cop_xz))
                _ls_cop_yz_n = max(1, len(_ls_cop_yz))
                _ls_unowned_xz_pct = 100.0 * len(_ls_unowned_xz) / _ls_cop_xz_n
                _ls_unowned_yz_pct = 100.0 * len(_ls_unowned_yz) / _ls_cop_yz_n

                _ls_proj_payload={
                    'frame_num':int(frame_num),'origin':[_ls_ox,_ls_oy,_ls_oz],'res':_ls_r,
                    'cop_xz':[int(v) for c in _ls_cop_xz for v in c],
                    'cop_yz':[int(v) for c in _ls_cop_yz for v in c],
                    'cop_xz_dense':[int(v) for c in _ls_cop_xz_dense for v in c],
                    'cop_yz_dense':[int(v) for c in _ls_cop_yz_dense for v in c],
                    'unowned_xz':[int(v) for c in _ls_unowned_xz for v in c],
                    'unowned_yz':[int(v) for c in _ls_unowned_yz for v in c],
                    'unowned_xz_pct':float(_ls_unowned_xz_pct),
                    'unowned_yz_pct':float(_ls_unowned_yz_pct),
                    'cluster_count':int(len(current_clusters)),
                    'cluster_count_dict':int(len(current_clusters)),
                    'cluster_count_visualized':int(_ls_visualized_count),
                    'cluster_uuids':_ls_uuids_in_order,
                    'person_cluster_uuid':person_cluster_uuid or '',
                    'clusters':_ls_clusters_payload,
                }
                _ls_proj_path=os.path.join(results_dir,f"frame_{frame_num:03d}_proj.json")
                write_compact_json(_ls_proj_payload,_ls_proj_path)

                logger.info(
                    f"[CLUSTER-PROJ] Late-state sidecar: dict={len(current_clusters)}, "
                    f"visualized={_ls_visualized_count}, "
                    f"unowned XZ={_ls_unowned_xz_pct:.0f}%, "
                    f"YZ={_ls_unowned_yz_pct:.0f}%")
            except Exception as _ls_e:
                logger.warning(f"[CLUSTER-PROJ] Late-state sidecar failed: {_ls_e}")

        # ============================================================
        return {
            'frame': frame_num,
            'clusters': current_clusters,
            'n_clusters': n_clusters,
            'method': method_name,
            'voxel_metadata_preserved': True,
            'voxel_grid': voxel_grid,
            'pose_2d': pose_2d,
            'pose_3d': pose_3d,
            'person_cluster_uuid': person_cluster_uuid,
            'pose_predicted': pose_was_predicted,
            'consecutive_rejections': consecutive_rejections,
            'segment_validation_applied': segment_validation_applied,
            'iccs': iccs,
            'facing_info': facing_info,                                                         # POST-ICCS facing info (legacy)
            'mannequin_body_yaw_deg': _body_yaw_for_mannequin if '_body_yaw_for_mannequin' in dir() else None,  # ISSUE-2 xy_pca yaw applied to canonical mannequin — also used for PLY rotation_angle
            'anomalies_detected': any(frame_anomalies.values()) if 'frame_anomalies' in dir() else False
        }
        
    except Exception as e:
        logger.error(f"Error processing frame: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def process_multiple_frames(file_pattern, args):
    """
    Process multiple frames with complete voxel metadata preservation.
    FIXED: Builds voxel_grids_by_frame and properly saves temporal data.
    """
    
    # ============== COMPACT JSON HELPERS (MODULE LEVEL) ==============

    def compact_reconstruction_data(obj):
        """Round floats to 1 decimal and keep ONLY reconstruction-essential fields"""
        if isinstance(obj, dict):
            result = {}
            # Skip non-essential fields INCLUDING voxel_indices
            skip_fields = {'normal', 'method', 'pattern_summary', 'y1_voxels', 'y2_voxels', 
                        'both_y_voxels', 'x_span', 'y_span', 'z_span', 'density', 'voxel_indices'}
            for key, value in obj.items():
                # Convert tuple keys to strings
                if isinstance(key, tuple):
                    key = str(key)
                if key in skip_fields:
                    continue
                result[key] = compact_reconstruction_data(value)
            return result
        elif isinstance(obj, list):
            return [round(x, 1) if isinstance(x, float) else compact_reconstruction_data(x) for x in obj]
        elif isinstance(obj, float):
            return round(obj, 1)
        elif isinstance(obj, np.ndarray):
            return [round(float(x), 1) for x in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return round(float(obj), 1)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, tuple):
            return str(obj)
        elif hasattr(obj, 'name'):
            return obj.name
        elif hasattr(obj, 'value'):
            return obj.value
        else:
            return obj

    def write_compact_json(data, filepath):
        """Write JSON with compact formatting - ALL numeric arrays on single lines"""
        
        def format_value(val, indent=0):
            indent_str = "  " * indent
            
            if isinstance(val, dict):
                if not val:
                    return "{}"
                
                items = []
                current_line = ""
                result_lines = ["{"]
                
                for i, (k, v) in enumerate(val.items()):
                    # Handle tuple keys
                    if isinstance(k, tuple):
                        k = str(k)
                    key_str = f'"{k}": '
                    val_str = format_value(v, indent + 1)
                    
                    item = key_str + val_str
                    if i < len(val) - 1:
                        item += ","
                    
                    # Check if we should break line
                    if current_line and len(current_line) + len(item) + 1 > 100:
                        result_lines.append(indent_str + "  " + current_line)
                        current_line = item
                    else:
                        if current_line:
                            current_line += " " + item
                        else:
                            current_line = item
                
                if current_line:
                    result_lines.append(indent_str + "  " + current_line)
                
                result_lines.append(indent_str + "}")
                return "\n".join(result_lines)
                
            elif isinstance(val, list):
                if not val:
                    return "[]"
                
                # Keep ALL numeric arrays on one line (this is the key fix!)
                if all(isinstance(x, (int, float)) for x in val):
                    return "[" + ", ".join(str(round(x, 1) if isinstance(x, float) else x) for x in val) + "]"
                
                # Handle nested arrays (like lasso points) - keep on single lines
                if all(isinstance(x, list) for x in val):
                    inner_strs = []
                    for item in val:
                        if isinstance(item, list) and all(isinstance(v, (int, float)) for v in item):
                            inner_strs.append("[" + ", ".join(str(round(v, 1) if isinstance(v, float) else v) for v in item) + "]")
                        else:
                            inner_strs.append(format_value(item, indent + 1))
                    return "[" + ", ".join(inner_strs) + "]"
                
                # Other lists - try to keep compact
                items = [format_value(item, indent + 1) for item in val]
                one_line = "[" + ", ".join(items) + "]"
                if len(one_line) < 100:
                    return one_line
                
                # Break into lines if too long
                return "[\n" + indent_str + "  " + (",\n" + indent_str + "  ").join(items) + "\n" + indent_str + "]"
                
            elif isinstance(val, str):
                return f'"{val}"'
            elif isinstance(val, bool):
                return str(val).lower()
            elif isinstance(val, float):
                return str(round(val, 1))
            elif isinstance(val, int):
                return str(val)
            elif val is None:
                return "null"
            elif hasattr(val, 'name'):
                return f'"{val.name}"'
            else:
                return f'"{str(val)}"'
        
        formatted = format_value(data, 0)
        
        with open(filepath, 'w') as f:
            f.write(formatted)

    # Find matching files
    logger.info(f"Looking for files matching pattern: {file_pattern}")
    files = sorted(glob.glob(file_pattern))
    
    if not files:
        logger.error(f"No files found matching pattern: {file_pattern}")
        return {}
    
    logger.info(f"Found {len(files)} files matching pattern")
    
    # Create folder structure
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(os.path.join(args.output, "visualizations"), exist_ok=True)
    os.makedirs(os.path.join(args.output, "results"), exist_ok=True)
    
    # ==================================================================
    # TIMING FIX: Delete stale results_s.json from PREVIOUS runs.
    # Phase 1 video pass runs BEFORE Phase 2 generates new results_s.
    # If old results_s.json exist, they contain wrong body_yaw values
    # that override the correct Phase 1 facing_info from frame_results.
    # ==================================================================
    _stale_dir = os.path.join(args.output, "results")
    _stale_count = 0
    for _sf in os.listdir(_stale_dir):
        if _sf.endswith('_results_s.json'):
            os.remove(os.path.join(_stale_dir, _sf))
            _stale_count += 1
    if _stale_count > 0:
        logger.warning(f"[TIMING] Deleted {_stale_count} stale results_s.json from previous run")
    
    # -----------------------------------------------------------------------
    # FIX (blocker): state_bank and frame_buffer always initialized.
    # See matching fix in main() for full explanation.
    # -----------------------------------------------------------------------
    from temporal_consistency import FrameBuffer, ClusterStateBank
    frame_buffer = FrameBuffer(max_size=5)
    state_bank   = ClusterStateBank()
    logger.info("Initialized temporal consistency components (always-on)")

    # ===================== NEW: 2D Pose Stabilization =====================
    pose_2d_history = Pose2DHistory(buffer_size=5)
    facing_history = FacingHistory(required_consistent_frames=5)
    logger.info("Initialized 2D pose stabilization (Pose2DHistory, FacingHistory)")
    # ======================================================================

    if hasattr(args, 'use_temporal') and args.use_temporal:
        logger.info("--use_temporal: extended temporal-consistency reporting enabled")
    
    
    # Initialize ICCS
    cluster_coords = ClusterCoordinateSystem()
    logger.info("Initialized Cluster Inner Coordinate System (ICCS)")
    
    # ============================================================
    # INITIALIZE SURFACE ACCUMULATION COMPONENTS FIRST
    # ============================================================
    surface_accumulator = None
    volume_tracker = None
    keypoint_validator = None
    
    if SURFACE_ACCUMULATION_AVAILABLE:
        surface_accumulator = SurfaceAccumulator(
            voxel_resolution=args.grid_resolution if hasattr(args, 'grid_resolution') else 2.0,
            overlap_threshold=3.0,
            min_overlap_ratio=0.3,
            output_dir=args.output,
            y_plane_spacing=1.0
        )
        volume_tracker = VolumeTracker(
            stability_window=10,
            volume_tolerance=0.25,
            rigid_tolerance=0.10
        )
        keypoint_validator = KeypointValidator(snap_distance=5.0)
        logger.info("Initialized surface accumulation: SurfaceAccumulator, VolumeTracker, KeypointValidator")
    
    # ============================================================
    # INITIALIZE ICCS - PASS THE SURFACE ACCUMULATOR TO IT!
    # ============================================================
    cluster_coords = ClusterCoordinateSystem(
        enable_surface=SURFACE_ACCUMULATION_AVAILABLE,
        surface_accumulator=surface_accumulator,      # â† PASS IT IN!
        volume_tracker=volume_tracker,                # â† PASS IT IN!
        keypoint_validator=keypoint_validator         # â† PASS IT IN!
    )
    logger.info("Initialized Cluster Inner Coordinate System (ICCS) with shared surface components")
    
    # ============================================================
    # STORE IN state_bank FOR process_single_frame ACCESS
    # ============================================================
    if state_bank is not None:
        # Store the SAME instances everywhere
        state_bank.surface_accumulator = surface_accumulator
        state_bank.volume_tracker = volume_tracker
        state_bank.keypoint_validator = keypoint_validator
        state_bank.cluster_coords = cluster_coords  # â† ALSO STORE cluster_coords!
        logger.info("Surface components and ICCS stored in state_bank")
        
        # Set camera params for inventory 2D projection
        if hasattr(state_bank, 'set_inventory_camera_params'):
            state_bank.set_inventory_camera_params({
                'camera_position': args.camera_position,
                'camera_target': args.camera_target,
                'field_of_view': args.field_of_view,
                'image_size': args.image_size,
            })
        # Link volume tracker to state bank for volume validation in Historical Lock matching
        if volume_tracker is not None:
            state_bank.set_volume_tracker(volume_tracker)
            logger.info("Linked VolumeTracker to ClusterStateBank for volume-based matching")

    # Track clusters and voxel grids across frames
    clusters_by_frame = {}
    voxel_grids_by_frame = {}  # CRITICAL: Store grids for metadata extraction
    consolidated_y_wall_index = {}
    
    # Log camera parameters
    logger.info(f"Camera parameters for all frames:")
    logger.info(f"  Camera position: {args.camera_position}")
    logger.info(f"  Camera target: {args.camera_target}")
    logger.info(f"  Focal length: {args.focal_length}mm")
    logger.info(f"  Field of view: {args.field_of_view} degrees")
    logger.info(f"  Image size: {args.image_size} pixels")
    
    start_time = time.time()
    
    try:
        for i, file_path in enumerate(files):
            frame_num = point_cloud.extract_frame_number(file_path)
            if frame_num is None:
                frame_num = i + 1
            
            logger.info(f"\nProcessing frame {frame_num} ({i+1}/{len(files)}): {os.path.basename(file_path)}")
            
            # ============================================================================
            # SUPERSEDED: old "check previous frame anomaly" block removed.
            # RC-1 (inside process_single_frame, ~line 1352) detects and reprocesses
            # under-clustered frames immediately, BEFORE mesh/surface work, using the
            # correct file that is already loaded for that frame's iteration.
            # The old block here reloaded files[i-1] at the start of the NEXT frame's
            # iteration, causing:
            #   (a) a redundant second reprocessing of the previous frame,
            #   (b) confusing log entries showing frame N loading frame N-1's file,
            #   (c) potential buffer corruption if the forced-split result differed
            #       from what RC-1 had already stored.
            # Both bugs were confirmed in the frame-13/14 anomaly (2026-04-10 run).
            # ============================================================================
            # ============================================================================
            
            existing_clusters = None
            if args.maintain_consistent_ids and not args.use_temporal and frame_num-1 in clusters_by_frame:
                existing_clusters = clusters_by_frame[frame_num-1].get('clusters', {})
            
            # Process frame (normal processing)
            frame_results = process_single_frame(
                file_path, args, frame_num, existing_clusters,
                frame_buffer, state_bank
            )
            
            if frame_results is not None:
                clusters_by_frame[frame_num] = frame_results
                
                # CRITICAL: Store the voxel grid for this frame
                if 'voxel_grid' in frame_results and frame_results['voxel_grid'] is not None:
                    voxel_grids_by_frame[frame_num] = frame_results['voxel_grid']
                    logger.info(f"  [OK] Stored voxel grid for frame {frame_num}")
                
                # Extract pose data from frame results (if available)
                pose_2d = frame_results.get('pose_2d')
                pose_3d = frame_results.get('pose_3d')
                person_cluster_uuid = frame_results.get('person_cluster_uuid')
                facing_info = frame_results.get('facing_info')                              # POST-ICCS info (fallback)
                mannequin_body_yaw_deg = frame_results.get('mannequin_body_yaw_deg')        # ISSUE-2 xy_pca yaw — primary PLY rotation source
                
                # NOTE: frame_buffer.add_frame() is already called inside process_single_frame
                if frame_buffer is not None:
                    logger.info(f"  [OK] Frame {frame_num} already in buffer (buffer has {len(frame_buffer.frames)} frames)")
                
                # Verify voxel data preservation
                if frame_results.get('voxel_metadata_preserved'):
                    logger.info(f"  [OK] Frame {frame_num}: Voxel metadata preserved")
                else:
                    logger.warning(f"  [FAIL] Frame {frame_num}: Voxel metadata NOT preserved")
                
                # ===== ESTABLISH ICCS FOR ALL CLUSTERS =====
                if 'clusters' in frame_results and frame_results['clusters']:
                    current_clusters = frame_results['clusters']
                    voxel_grid = frame_results.get('voxel_grid')
                    
                    logger.debug(f"Frame {frame_num}: Processing ICCS for {len(current_clusters)} clusters")
                    
                    for cluster_id, cluster_data in current_clusters.items():
                        iccs = None
                        
                        # Check if this is the person cluster
                        is_person_cluster = False
                        if person_cluster_uuid:
                            is_person_cluster = (cluster_id == person_cluster_uuid or 
                                               cluster_data.get('uuid') == person_cluster_uuid)
                        
                        # Try MMPose-based ICCS for person cluster
                        if (args.use_mmpose and is_person_cluster and pose_2d is not None):
                            logger.debug(f"Attempting MMPose ICCS for cluster {cluster_id}")
                            
                            keypoints = None
                            if isinstance(pose_2d, dict) and 'keypoints' in pose_2d:
                                keypoints = np.array(pose_2d['keypoints'])
                            elif isinstance(pose_2d, list):
                                keypoints = np.array(pose_2d)
                            
                            if keypoints is not None:
                                # Pass facing_info if available from earlier detection
                                _fi_for_iccs = None
                                try:
                                    _fi_for_iccs = facing_info
                                except NameError:
                                    pass
                                iccs = cluster_coords.establish_iccs_from_mmpose(
                                    keypoints,
                                    facing_info=_fi_for_iccs
                                )
                                
                                if iccs:
                                    logger.info(f"Frame {frame_num}: Established ICCS from MMPose for {cluster_id}")
                                    logger.info(f"Rotation: {iccs['rotation']['angle']:.1f}deg, "
                                              f"Facing: {iccs['rotation']['facing']}")
                        
                        # Fallback to voxel-based ICCS for non-person clusters
                        if iccs is None:
                            cluster_points = cluster_data.get('points')
                            
                            if cluster_points is not None and len(cluster_points) > 10:
                                logger.debug(f"Attempting voxel-based ICCS for cluster {cluster_id}")
                                iccs = cluster_coords.establish_iccs_from_voxels(
                                    voxel_grid=voxel_grid,
                                    cluster_points=cluster_points
                                )
                                if iccs:
                                    logger.debug(f"Frame {frame_num}: Established voxel ICCS for {cluster_id}")
                        
                        # Update coordinate system if established
                        if iccs is not None:
                            tracking_id = cluster_data.get('uuid', cluster_id)
                            
                            cluster_coords.update_cluster_iccs(tracking_id, iccs, frame_num)
                            cluster_data['iccs'] = iccs
                            
                            rotation_info = cluster_coords.track_rotation(tracking_id)
                            cluster_data['rotation_tracking'] = rotation_info
                            
                            if rotation_info['rotating']:
                                logger.warning(f">>> ROTATION DETECTED: Frame {frame_num}, "
                                             f"Cluster {cluster_id} (UUID: {tracking_id[:8]})")
                                logger.warning(f"    Direction: {rotation_info['direction']}")
                                logger.warning(f"Total rotation: {rotation_info['total_rotation']:.1f}deg")
                                
                                cluster_data['force_synthetic'] = True
                                cluster_data['rotation_detected'] = True
                                
                                logger.warning(f"    FORCING synthetic points for rotating cluster")
                                
                                point_count = cluster_data.get('point_count', 0)
                                if point_count < 500:
                                    logger.warning(f"    Sparse rotating cluster: only {point_count} points!")
                            
                            # ========== SURFACE ACCUMULATION UPDATE ==========
                            if SURFACE_ACCUMULATION_AVAILABLE and surface_accumulator is not None:
                                cluster_points = cluster_data.get('points')
                                if cluster_points is None and 'voxel_metadata' in cluster_data:
                                    cluster_points = reconstruct_from_voxel_metadata(
                                        cluster_data['voxel_metadata'],
                                        resolution=args.grid_resolution if hasattr(args, 'grid_resolution') else 2.0
                                    )
                                
                                if cluster_points is not None and len(cluster_points) > 10:
                                    is_person = (tracking_id == person_cluster_uuid) if person_cluster_uuid else False
                                    object_type = 'human' if is_person else 'rigid'
                                    
                                    # Extract viewing angle from iccs if available
                                    viewing_angle = None
                                    if iccs and 'rotation' in iccs:
                                        viewing_angle = iccs['rotation'].get('angle', 0)
                                    
                                    # =====================================================================
                                    # NEW: Calculate rotation_angle for ICCS transformation
                                    # =====================================================================
                                    rotation_angle = None
                                    
                                    # For person cluster: prefer mannequin_body_yaw_deg (ISSUE-2 XY-PCA
                                    # yaw — same value applied to canonical mannequin). Keeps PLY mesh
                                    # reference_rotation in sync with the visual mannequin orientation.
                                    if is_person and mannequin_body_yaw_deg is not None:
                                        try:
                                            rotation_angle = float(mannequin_body_yaw_deg)
                                            logger.debug(f"Frame {frame_num}: PLY rotation_angle = "
                                                         f"{rotation_angle:.1f}deg (source=mannequin_body_yaw_deg, xy_pca)")
                                        except Exception:
                                            rotation_angle = None
                                    
                                    # Fallback: pose_3d shoulder/hip-derived rotation_z
                                    if is_person and rotation_angle is None and pose_3d is not None:
                                        body_orientation = _calculate_body_orientation_safe(pose_3d)
                                        
                                        if body_orientation and body_orientation.get('valid'):
                                            rotation_angle = body_orientation['rotation_z']
                                            logger.debug(f"Frame {frame_num}: Body rotation_z = {rotation_angle:.1f}deg (fallback)")
                                        else:
                                            logger.debug(f"Frame {frame_num}: Could not calculate body orientation")
                                    
                                    # For non-person clusters, use ICCS rotation if available
                                    if rotation_angle is None and iccs and 'rotation' in iccs:
                                        rotation_angle = iccs['rotation'].get('angle')
                                        logger.debug(f"Frame {frame_num}: Using ICCS rotation = {rotation_angle:.1f}deg")
                                    
                                    # Get cluster centroid from cluster_data
                                    cluster_centroid = cluster_data.get('centroid')
                                    if cluster_centroid is None and cluster_points is not None:
                                        cluster_centroid = np.mean(cluster_points, axis=0)
                                        logger.debug(f"Frame {frame_num}: Computed centroid from points")
                                    
                                    # =====================================================================
                                    # UPDATED: Pass rotation_angle and cluster_centroid to accumulate_frame
                                    # =====================================================================
                                    surface_state, overlap_result = surface_accumulator.accumulate_frame(
                                        uuid=tracking_id,
                                        new_points=cluster_points,
                                        frame_num=frame_num,
                                        viewing_angle=viewing_angle,
                                        iccs=iccs,
                                        voxel_data=cluster_data.get('voxel_data'),
                                        cluster_centroid=cluster_centroid,      # NEW - for proper centering
                                        rotation_angle=rotation_angle            # NEW - for ICCS rotation correction!
                                    )
                                    
                                    # Log rotation correction status
                                    if rotation_angle is not None and surface_state.reference_rotation is not None:
                                        delta = rotation_angle - surface_state.reference_rotation
                                        logger.info(f"Frame {frame_num} [{tracking_id[:8]}]: "
                                                   f"rotation={rotation_angle:.1f}deg, "
                                                   f"reference={surface_state.reference_rotation:.1f}deg, "
                                                   f"delta={delta:.1f}deg")
                                        
                                        if abs(delta) > 0.5:
                                            logger.info(f"[OK] Applied inverse rotation correction: {-delta:.1f}deg")
                                    
                                    # Continue with volume tracking...
                                    volume_est, volume_valid = volume_tracker.update_volume(
                                        uuid=tracking_id,
                                        points=cluster_points,
                                        frame_num=frame_num,
                                        object_type=object_type
                                    )
                    # Add ICCS data to frame results for JSON output
                    iccs_data = {}
                    for cluster_id in current_clusters:
                        tracking_id = current_clusters[cluster_id].get('uuid', cluster_id)
                        current_iccs = cluster_coords.get_cluster_iccs(tracking_id)
                        if current_iccs:
                            iccs_data[cluster_id] = current_iccs
                    
                    frame_results['cluster_iccs'] = iccs_data
                    
                    # Log rotation events
                    rotation_events = []
                    for cluster_id, cluster_data in current_clusters.items():
                        if cluster_data.get('rotation_detected', False):
                            rotation_events.append({
                                'cluster_id': cluster_id,
                                'frame': frame_num,
                                'rotation_info': cluster_data.get('rotation_tracking', {})
                            })
                    
                    if rotation_events:
                        frame_results['rotation_events'] = rotation_events
                        logger.info(f"Frame {frame_num}: {len(rotation_events)} rotation events detected")
            
            logger.info(f"Processed {len(clusters_by_frame)} frames so far.")

            # ==============================================================
            # PER-FRAME SKELETON FITTING (replaces separate Phase 2 pass)
            # Runs immediately after surface accumulation so results_s.json
            # is written seconds after results.json — not an hour later.
            # ==============================================================
            try:
                import skeleton_fitting as _sf_module
                _results_dir_pf = os.path.join(args.output, "results")
                _surfaces_dir_pf = os.path.join(args.output, "surfaces")
                if os.path.isdir(_results_dir_pf):
                    _cam_params_pf = {
                        'camera_position': getattr(args, 'camera_position', [-47.0, 28.0, -20.0]),
                        'camera_target':   getattr(args, 'camera_target', [-25.1, 123.8, -28.3]),
                        'focal_length':    getattr(args, 'focal_length', 27.5),
                        'field_of_view':   getattr(args, 'field_of_view', 66.0),
                        'panel_width':     args.image_size[0] if hasattr(args, 'image_size') else 480,
                        'panel_height':    args.image_size[1] if hasattr(args, 'image_size') else 864,
                    }
                    # Initialize fitting engine once
                    if not hasattr(process_multiple_frames, '_fit_engine_init'):
                        process_multiple_frames._fit_engine_init = True
                        from movement_index import MovementIndexEngine, extract_limits_from_segments
                        _mi_limits = extract_limits_from_segments()
                        _fps = float(getattr(args, 'fps', 12.0))
                        process_multiple_frames._mi_engine = MovementIndexEngine(
                            joint_limits=_mi_limits, buffer_size=30, fps=_fps)
                        process_multiple_frames._uuid_first_seen = {}
                        # Compute grid origin once
                        process_multiple_frames._grid_origin = None
                        if voxel_grid is not None:
                            _go = _sf_module._compute_grid_origin(voxel_grid)
                            if _go is not None:
                                process_multiple_frames._grid_origin = _go

                    # Run single-frame fit
                    _sf_result = _sf_module.run_fitting_pass(
                        results_dir=_results_dir_pf,
                        surfaces_dir=_surfaces_dir_pf,
                        state_bank=state_bank,
                        voxel_grids_by_frame={frame_num: voxel_grid} if voxel_grid else {},
                        args=args,
                        camera_params=_cam_params_pf,
                        frame_nums_override=[frame_num],  # fit THIS frame only
                    )
                    _ok = _sf_result.get(frame_num, False) if _sf_result else False
                    logger.info(f"[PHASE-1+2] Frame {frame_num}: skeleton fitting {'OK' if _ok else 'FAILED'}")
            except ImportError:
                pass  # skeleton_fitting not available
            except Exception as _sf_err:
                logger.debug(f"[PHASE-1+2] Frame {frame_num}: fitting error: {_sf_err}")
    
    except KeyboardInterrupt:
        logger.warning("\nProcessing interrupted by user!")
    except Exception as e:
        logger.error(f"\nError during processing: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
    
    # Save comprehensive summary
    if clusters_by_frame:
        summary_file = os.path.join(args.output, "results", "clustering_summary.json")
        
        frames_with_voxel_data = sum(
            1 for frame_data in clusters_by_frame.values()
            if frame_data.get('voxel_metadata_preserved', False)
        )
        
        summary_data = {
            "total_frames": len(clusters_by_frame),
            "frames_with_voxel_data": frames_with_voxel_data,
            "voxel_grids_stored": len(voxel_grids_by_frame),
            "frames": {}
        }
        
        for frame_num, frame_data in clusters_by_frame.items():
            summary_data["frames"][str(frame_num)] = {
                "clusters": frame_data.get('n_clusters', 0),
                "voxel_metadata": frame_data.get('voxel_metadata_preserved', False)
            }
        
        try:
            compact_summary = compact_reconstruction_data(summary_data)
            write_compact_json(compact_summary, summary_file)
            logger.info(f"Saved compact clustering summary to {summary_file}")
            logger.info(f"  - Voxel grids stored: {len(voxel_grids_by_frame)}")
        except Exception as e:
            logger.error(f"Error saving summary: {str(e)}")
        
        # Save temporal consistency data - BUFFER FRAMES ONLY (last 5)
        if state_bank is not None:
            temporal_file = os.path.join(args.output, "results", "temporal_consistency_data.json")
            
            try:
                all_frame_nums = sorted(state_bank.frame_clusters.keys())
                buffer_frame_nums = all_frame_nums[-5:] if len(all_frame_nums) > 5 else all_frame_nums
                
                logger.info(f"Saving temporal data for buffer frames only: {len(buffer_frame_nums)} frames")
                logger.info(f"  Buffer frames: {buffer_frame_nums}")
                
                original_frame_clusters = state_bank.frame_clusters.copy()
                state_bank.frame_clusters = {
                    frame_num: original_frame_clusters[frame_num]
                    for frame_num in buffer_frame_nums
                    if frame_num in original_frame_clusters
                }
                
                temporal_data = {
                    "buffer_size": 5,
                    "total_frames": len(all_frame_nums),
                    "frames": {}
                }
                
                logger.info("Rebuilding cluster_registry from filtered frame_clusters")
                cluster_registry = {}
                
                for frame_num in sorted(state_bank.frame_clusters.keys()):
                    frame_clusters = state_bank.frame_clusters[frame_num]
                    
                    for cluster_uuid, cluster_data in frame_clusters.items():
                        if cluster_uuid not in cluster_registry:
                            cluster_registry[cluster_uuid] = {
                                "first_seen": frame_num,
                                "last_seen": frame_num,
                                "frames_present": [frame_num],
                                "total_points": cluster_data.get('point_count', 0)
                            }
                        else:
                            cluster_registry[cluster_uuid]["last_seen"] = frame_num
                            cluster_registry[cluster_uuid]["frames_present"].append(frame_num)
                            cluster_registry[cluster_uuid]["total_points"] += cluster_data.get('point_count', 0)
                
                temporal_data["cluster_registry"] = cluster_registry
                
                logger.info(f"Rebuilt cluster_registry with {len(cluster_registry)} unique clusters")
                for uuid, reg in cluster_registry.items():
                    logger.info(f"  {uuid[:8]}: frames {reg['first_seen']}-{reg['last_seen']} ({len(reg['frames_present'])} frames)")
                                
                for frame_num in all_frame_nums:
                    if frame_num not in state_bank.frame_clusters:
                        continue
                    
                    frame_clusters = state_bank.frame_clusters[frame_num]
                    frame_data = {
                        "n_clusters": len(frame_clusters),
                        "clusters": {}
                    }
                    
                    # Extract pose data from frame_buffer
                    pose_2d = None
                    pose_3d = None
                    person_cluster_uuid = None
                    facing_info = None
                    mannequin_body_yaw_deg = None
                    
                    if frame_buffer is not None:
                        logger.info(f"Frame {frame_num}: Searching for pose data in buffer ({len(frame_buffer.frames)} frames)")
                        for buffered_frame in frame_buffer.frames:
                            if buffered_frame['frame_num'] == frame_num:
                                pose_2d = buffered_frame.get('pose_2d')
                                pose_3d = buffered_frame.get('pose_3d')
                                person_cluster_uuid = buffered_frame.get('person_cluster_uuid')
                                facing_info = buffered_frame.get('facing_info')
                                mannequin_body_yaw_deg = buffered_frame.get('mannequin_body_yaw_deg')
                                
                                logger.info(f"Frame {frame_num}: Found pose data in buffer - pose_2d={pose_2d is not None}, pose_3d={pose_3d is not None}, person_cluster={person_cluster_uuid}")
                                break
                        
                        if pose_2d is None and pose_3d is None:
                            logger.warning(f"Frame {frame_num}: NO pose data found in buffer!")
                    
                    # ========== KEYPOINT VALIDATION AGAINST SURFACE ==========
                    if (SURFACE_ACCUMULATION_AVAILABLE and 
                        keypoint_validator is not None and 
                        surface_accumulator is not None and
                        person_cluster_uuid is not None and
                        pose_3d is not None):
                        
                        accumulated_pts = surface_accumulator.get_accumulated_points(person_cluster_uuid)
                        
                        if accumulated_pts is not None and len(accumulated_pts) > 50:
                            pose_3d_array = np.array(pose_3d) if not isinstance(pose_3d, np.ndarray) else pose_3d
                            
                            if len(pose_3d_array) > 0:
                                validated_kps, kp_confidences = keypoint_validator.validate_skeleton(
                                    pose_3d_array,
                                    accumulated_pts
                                )
                                
                                snapped_count = np.sum(kp_confidences < 0.8)
                                if snapped_count > 0:
                                    logger.info(f"Frame {frame_num}: Snapped {snapped_count} keypoints to surface shell")
                                
                                pose_3d = validated_kps
                                
                                frame_data["keypoint_confidences"] = kp_confidences.tolist()
                                frame_data["keypoints_snapped"] = int(snapped_count)
                                
                                logger.debug(f"Frame {frame_num}: Keypoint validation complete, "
                                           f"mean confidence: {np.mean(kp_confidences):.2f}")
                    # ========== END KEYPOINT VALIDATION ==========
                            
                    # Convert numpy arrays to plain Python lists
                    if pose_2d and isinstance(pose_2d, dict):
                        pose_2d = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in pose_2d.items()}
                        
                    if pose_3d is not None and isinstance(pose_3d, np.ndarray):
                        pose_3d = pose_3d.tolist()
                    
                    frame_data["pose_2d"] = pose_2d
                    frame_data["pose_3d"] = pose_3d
                    frame_data["person_cluster_uuid"] = person_cluster_uuid
                    
                    logger.info(f"Frame {frame_num}: Added pose data to temporal JSON - pose_2d={pose_2d is not None}, pose_3d={pose_3d is not None}")
                    
                    voxel_grid = voxel_grids_by_frame.get(frame_num)
                    
                    for cluster_key, cluster_info in frame_clusters.items():
                        voxel_metadata = {}
                        
                        if 'voxel_data' in cluster_info:
                            voxel_metadata = cluster_info['voxel_data']
                        elif voxel_grid and 'voxel_indices' in cluster_info:
                            for voxel_idx in cluster_info['voxel_indices']:
                                voxel_key = str(voxel_idx)
                                voxel_info = {}
                                
                                if hasattr(voxel_grid, 'get_complete_voxel_metadata'):
                                    voxel_info = voxel_grid.get_complete_voxel_metadata(voxel_idx)
                                elif hasattr(voxel_grid, 'cell_metadata') and voxel_idx in voxel_grid.cell_metadata:
                                    metadata = voxel_grid.cell_metadata[voxel_idx]
                                    voxel_info['centroid'] = metadata.get('centroid', [0, 0, 0])
                                    
                                    if hasattr(voxel_grid, 'voxel_patterns') and voxel_idx in voxel_grid.voxel_patterns:
                                        pattern_obj = voxel_grid.voxel_patterns[voxel_idx]
                                        
                                        if pattern_obj.y_plane_1:
                                            y1_data = {
                                                'pattern_id': pattern_obj.y_plane_1.pattern_id.name if hasattr(pattern_obj.y_plane_1.pattern_id, 'name') else str(pattern_obj.y_plane_1.pattern_id),
                                                'point_count': pattern_obj.y_plane_1.point_count,
                                                'centroid_offset': pattern_obj.y_plane_1.centroid_offset.tolist() if hasattr(pattern_obj.y_plane_1.centroid_offset, 'tolist') else list(pattern_obj.y_plane_1.centroid_offset)
                                            }
                                            
                                            shape_key = 'y_plane_1_shape_bounds'
                                            if shape_key in metadata:
                                                y1_data['shape_bounds'] = metadata[shape_key]
                                            
                                            voxel_info['y_plane_1'] = y1_data
                                        
                                        if pattern_obj.y_plane_2:
                                            y2_data = {
                                                'pattern_id': pattern_obj.y_plane_2.pattern_id.name if hasattr(pattern_obj.y_plane_2.pattern_id, 'name') else str(pattern_obj.y_plane_2.pattern_id),
                                                'point_count': pattern_obj.y_plane_2.point_count,
                                                'centroid_offset': pattern_obj.y_plane_2.centroid_offset.tolist() if hasattr(pattern_obj.y_plane_2.centroid_offset, 'tolist') else list(pattern_obj.y_plane_2.centroid_offset)
                                            }
                                            
                                            shape_key = 'y_plane_2_shape_bounds'
                                            if shape_key in metadata:
                                                y2_data['shape_bounds'] = metadata[shape_key]
                                            
                                            voxel_info['y_plane_2'] = y2_data
                                
                                if voxel_info:
                                    voxel_metadata[voxel_key] = voxel_info
                        
                        cluster_data = {
                            "cluster_uuid": cluster_info.get('cluster_uuid', cluster_key),
                            "total_points": cluster_info.get('total_points', 0),
                            "total_voxels": cluster_info.get('total_voxels', 0),
                            "centroid": cluster_info.get('centroid', [0, 0, 0]),
                            "bbox": cluster_info.get('bbox', {'min': [0, 0, 0], 'max': [0, 0, 0]}),
                            "voxel_metadata": voxel_metadata
                        }
                        
                        frame_data["clusters"][cluster_key] = cluster_data
                    
                    temporal_data["frames"][str(frame_num)] = frame_data
                
                compact_temporal = compact_reconstruction_data(temporal_data)
                write_compact_json(compact_temporal, temporal_file)
                
                logger.info(f"[OK] Saved compact temporal consistency data to {temporal_file}")
                logger.info(f"  - Buffer frames: {len(buffer_frame_nums)}")
                logger.info(f"  - Total frames: {len(all_frame_nums)}")
                
                # Verify
                with open(temporal_file, 'r') as f:
                    temp_data = json.load(f)
                    
                    total_frames = len(temp_data.get('frames', {}))
                    total_clusters = len(temp_data.get('cluster_registry', {}))
                    
                    total_voxels_with_metadata = 0
                    for frame_num, frame_data in temp_data.get('frames', {}).items():
                        for cluster_id, cluster_info in frame_data.get('clusters', {}).items():
                            if 'voxel_metadata' in cluster_info:
                                total_voxels_with_metadata += len(cluster_info['voxel_metadata'])
                    
                    if total_voxels_with_metadata > 0:
                        logger.info(f"[OK] TEMPORAL DATA SAVED SUCCESSFULLY:")
                        logger.info(f"  - {total_frames} frames")
                        logger.info(f"  - {total_clusters} unique clusters")
                        logger.info(f"  - {total_voxels_with_metadata} voxels with complete metadata")
                    else:
                        logger.error("[FAIL] WARNING: Temporal data saved but no voxel metadata found!")
                        
            except Exception as e:
                logger.error(f"Error saving temporal data: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
        
        # ============================================================
        # SAVE SURFACE STATES
        # ============================================================
        if SURFACE_ACCUMULATION_AVAILABLE and surface_accumulator is not None:
            try:
                surface_file = os.path.join(args.output, "results", "surface_states.json")
                save_surface_states(surface_accumulator.surfaces, surface_file)
                logger.info(f"[OK] Saved surface states to {surface_file}")
            except Exception as e:
                logger.error(f"Error saving surface states: {str(e)}")
        
        # ================================================================
        # SKELETON + MOVEMENT-INDEX PASS
        #
        # create_video_from_frames is the only entry-point that runs
        # shell-fitting and the MovementIndexEngine.  It must execute
        # whenever ANY video output is requested — not only when
        # --create_video is set.  Previously, running with
        # --create_overlay_video (but NOT --create_video) silently
        # skipped the entire skeleton / MI layer, leaving every
        # frame_XXX_results_s.json with joints_in_cluster=0 and no
        # movement_index block at all.
        #
        # FIX: run create_video_from_frames when EITHER flag is set.
        # The three-panel video file is always written (it is fast to
        # render and gives a useful second visual pass alongside the
        # overlay).  If only --create_overlay_video was requested the
        # caller can ignore the _video.mp4 file; the _p/_s JSON outputs
        # are the important artefacts.
        # ================================================================
        _run_viz_pass = args.create_video or getattr(args, 'create_overlay_video', False)

        if _run_viz_pass:
            video_dir = os.path.join(args.output, "videos")
            os.makedirs(video_dir, exist_ok=True)

            method_name = "Grid" if args.use_grid else "DBSCAN" if args.use_dbscan else "HDBSCAN"
            if args.opencv_enhance:
                method_name = "OpenCV+" + method_name

            # ==============================================================
            # SKIP first video pass. It runs BEFORE Phase 2, reads stale
            # results_s.json from previous runs, produces wrong PLYs/panels.
            # The post-Phase-2 video pass (below, after skeleton_fitting)
            # is the ONLY pass that has correct body_yaw, height, facing.
            # ==============================================================
            logger.info("[SYNC] Skipping pre-Phase-2 video pass - "
                       "will render AFTER Phase 2 with correct data")

            # Create overlay video (separate pass — no skeleton work here)
            if getattr(args, 'create_overlay_video', False):
                overlay_video_file = os.path.join(video_dir, f"{method_name}_overlay_video.mp4")
                overlay_files = sorted(glob.glob(file_pattern))
                create_overlay_video(clusters_by_frame, overlay_files, args.frames_dir, overlay_video_file, args, args.fps)
    else:
        logger.warning("No frames were successfully processed")
    
    total_time = time.time() - start_time
    logger.info(f"\nSuccessfully processed {len(clusters_by_frame)} frames in {total_time:.2f} seconds")
    if len(clusters_by_frame) > 0:
        logger.info(f"Average time per frame: {total_time/len(clusters_by_frame):.2f} seconds")
    
    # Save final ICCS state
    iccs_output = os.path.join(args.output, 'cluster_iccs_final.json')
    cluster_coords.save_state(iccs_output)
    logger.info(f"Saved final ICCS state to {iccs_output}")
    
    # Log summary
    total_rotations = 0
    for uuid in cluster_coords.rotation_history:
        history = cluster_coords.rotation_history[uuid]
        if len(history) > 0:
            total_rotations += len(history)
            logger.info(f"Cluster {uuid[:8]}: {len(history)} rotation events")
    
    if total_rotations == 0:
        logger.warning(">>> NO ROTATIONS DETECTED IN ENTIRE SEQUENCE!")
    else:
        logger.info(f">>> Total rotation events detected: {total_rotations}")
    
    return clusters_by_frame, frame_buffer, state_bank, voxel_grids_by_frame

# def save_temporal_data(state_bank, output_file):
#     """
#     Save temporal consistency data to file with consistent cluster IDs.
    
#     Args:
#         state_bank: ClusterStateBank instance
#         output_file: Path to output JSON file
#     """
#     # Use the state_bank's built-in save method which includes all the new data
#     state_bank.save_to_file(output_file)
    
#     # Log additional information
#     logger.info(f"Saved temporal consistency data to {output_file}")
#     logger.info(f"  Total clusters in registry: {len(state_bank.global_cluster_registry)}")
#     logger.info(f"  Frames with clusters: {len(state_bank.frame_clusters)}")
    
#     # Optionally create a simplified summary for quick viewing
#     summary_file = output_file.replace('.json', '_summary.json')
#     try:
#         summary_data = {
#             "cluster_count": len(state_bank.global_cluster_registry),
#             "frame_count": len(state_bank.frame_clusters),
#             "clusters": {}
#         }
        
#         # Add brief info for each cluster
#         for cluster_id, registry_data in state_bank.global_cluster_registry.items():
#             summary_data["clusters"][cluster_id] = {
#                 "first_seen": registry_data.get('first_seen'),
#                 "last_seen": registry_data.get('last_seen'),
#                 "frames_present": registry_data.get('frames_present', []),
#                 "total_points": registry_data.get('total_points', 0)
#             }
        
#         with open(summary_file, 'w') as f:
#             json.dump(summary_data, f, indent=2, default=str)
#         logger.info(f"Saved temporal summary to {summary_file}")
#     except Exception as e:
#         logger.error(f"Error saving temporal summary: {str(e)}")

def track_ywall_clusters_temporally(frame_y_wall_results, state_bank=None, min_overlap=0.3):
    """
    Track Y-wall clusters across frames using centroid distance and size similarity.
    FIXED: Sort clusters by centroid X position for consistent ordering across frames.
    """
    tracked_clusters = {}
    clusters_by_frame = {}
    
    # Track historical statistics for anomaly detection
    historical_cluster_counts = []
    historical_cluster_sizes = []
    merge_warnings = []
    
    # First, extract just the clusters from each frame's Y-wall results
    temporal_frame_clusters = {}
    for frame_num, y_wall_result in frame_y_wall_results.items():
        if isinstance(y_wall_result, dict) and 'clusters' in y_wall_result:
            temporal_frame_clusters[frame_num] = y_wall_result['clusters']
        else:
            logger.warning(f"Invalid Y-wall result structure for frame {frame_num}")
            temporal_frame_clusters[frame_num] = {}
    
    frame_nums = sorted(temporal_frame_clusters.keys())
    
    # Initialize with first frame - check for existing clusters first
    if frame_nums:
        first_frame = frame_nums[0]
        clusters_by_frame[first_frame] = {}
        
        # CHECK: Does this frame already have clusters from a previous window?
        if state_bank and first_frame in state_bank.frame_clusters:
            logger.info(f"Frame {first_frame} already processed in previous window, matching clusters by centroid")
            existing_clusters = state_bank.frame_clusters[first_frame]
            used_existing_ids = set()
            
            # CRITICAL FIX: Sort clusters by centroid X position for consistent ordering
            sorted_clusters = sorted(
                temporal_frame_clusters[first_frame].items(),
                key=lambda x: x[1].get('centroid', [0, 0, 0])[0]  # Sort by X coordinate
            )
            
            # Match each new cluster to existing ones by centroid
            for local_id, cluster_info in sorted_clusters:
                new_centroid = cluster_info.get('centroid', [0, 0, 0])
                best_match_id = None
                best_distance = float('inf')
                
                # Find best matching existing cluster
                for existing_id, existing_info in existing_clusters.items():
                    if existing_id in used_existing_ids:
                        continue  # Already matched
                        
                    existing_centroid = existing_info.get('centroid', [0, 0, 0])
                    distance = np.linalg.norm(np.array(new_centroid) - np.array(existing_centroid))
                    
                    if distance < best_distance and distance < 15.0:  # 15cm tolerance for movement
                        best_distance = distance
                        best_match_id = existing_id
                
                if best_match_id:
                    # Reuse existing UUID
                    cluster_uuid = best_match_id
                    used_existing_ids.add(best_match_id)
                    logger.info(f"  Matched to existing {cluster_uuid} (centroid moved {best_distance:.1f}cm)")
                    
                    # Update tracked_clusters with existing UUID
                    if cluster_uuid in tracked_clusters:
                        # Update existing tracking
                        tracked_clusters[cluster_uuid]['last_seen'] = first_frame
                        tracked_clusters[cluster_uuid]['voxel_counts'][first_frame] = cluster_info.get('voxel_count', 0)
                        tracked_clusters[cluster_uuid]['centroid_history'][first_frame] = cluster_info.get('centroid', [0, 0, 0])
                        tracked_clusters[cluster_uuid]['size_history'][first_frame] = cluster_info.get('point_count', 0)
                    else:
                        # Initialize tracking with existing UUID
                        tracked_clusters[cluster_uuid] = {
                            'first_seen': first_frame,
                            'last_seen': first_frame,
                            'frames': [first_frame],
                            'voxel_history': {first_frame: cluster_info.get('voxel_metadata', {})},
                            'voxel_counts': {first_frame: cluster_info.get('voxel_count', 0)},
                            'pattern_history': {first_frame: cluster_info.get('pattern_distribution', {})},
                            'centroid_history': {first_frame: cluster_info.get('centroid', [0, 0, 0])},
                            'size_history': {first_frame: cluster_info.get('point_count', 0)},
                            'xz_footprint_history': {first_frame: calculate_xz_footprint(cluster_info)}
                        }
                else:
                    # No match found - this is a new cluster
                    cluster_uuid = f"cluster_{len(tracked_clusters):02d}"
                    logger.warning(f"  No match for cluster with centroid {new_centroid}, assigning new ID: {cluster_uuid}")
                    
                    # Initialize new tracking
                    tracked_clusters[cluster_uuid] = {
                        'first_seen': first_frame,
                        'last_seen': first_frame,
                        'frames': [first_frame],
                        'voxel_history': {first_frame: cluster_info.get('voxel_metadata', {})},
                        'voxel_counts': {first_frame: cluster_info.get('voxel_count', 0)},
                        'pattern_history': {first_frame: cluster_info.get('pattern_distribution', {})},
                        'centroid_history': {first_frame: cluster_info.get('centroid', [0, 0, 0])},
                        'size_history': {first_frame: cluster_info.get('point_count', 0)},
                        'xz_footprint_history': {first_frame: calculate_xz_footprint(cluster_info)}
                    }
                
                # Store in frame mapping
                clusters_by_frame[first_frame][cluster_uuid] = cluster_info.copy()
                clusters_by_frame[first_frame][cluster_uuid]['uuid'] = cluster_uuid
                
                # Track size for anomaly detection
                historical_cluster_sizes.append(cluster_info.get('point_count', 0))
                
                logger.info(f"Frame {first_frame}: Cluster {cluster_uuid} has {cluster_info.get('point_count', 0)} points")
        
        else:
            # First time processing this frame - generate new UUIDs
            logger.info(f"Frame {first_frame}: First time processing, generating new cluster IDs")
            
            # CRITICAL FIX: Sort clusters by centroid X position for consistent ordering
            sorted_clusters = sorted(
                temporal_frame_clusters[first_frame].items(),
                key=lambda x: x[1].get('centroid', [0, 0, 0])[0]  # Sort by X coordinate
            )
            
            for idx, (local_id, cluster_info) in enumerate(sorted_clusters):
                # Generate a stable UUID for this cluster - now based on sorted order!
                cluster_uuid = f"cluster_{idx:02d}"  # Simple, stable ID based on X position
                
                # Store cluster WITH its voxel data
                tracked_clusters[cluster_uuid] = {
                    'first_seen': first_frame,
                    'last_seen': first_frame,
                    'frames': [first_frame],
                    'voxel_history': {first_frame: cluster_info.get('voxel_metadata', {})},
                    'voxel_counts': {first_frame: cluster_info.get('voxel_count', 0)},
                    'pattern_history': {first_frame: cluster_info.get('pattern_distribution', {})},
                    'centroid_history': {first_frame: cluster_info.get('centroid', [0, 0, 0])},
                    'size_history': {first_frame: cluster_info.get('point_count', 0)},
                    'xz_footprint_history': {first_frame: calculate_xz_footprint(cluster_info)}
                }
                
                # Store in frame mapping
                clusters_by_frame[first_frame][cluster_uuid] = cluster_info.copy()
                clusters_by_frame[first_frame][cluster_uuid]['uuid'] = cluster_uuid
                
                # Track size for anomaly detection
                historical_cluster_sizes.append(cluster_info.get('point_count', 0))
                
                logger.info(f"Frame {first_frame}: Created cluster {cluster_uuid} with {cluster_info.get('point_count', 0)} points at X={cluster_info.get('centroid', [0,0,0])[0]:.1f}")
        
        # Track cluster count
        historical_cluster_counts.append(len(clusters_by_frame[first_frame]))
    
    # Process subsequent frames - TRACK AND DETECT ANOMALIES
    for i in range(1, len(frame_nums)):
        curr_frame = frame_nums[i]
        prev_frame = frame_nums[i-1]
        
        # Calculate historical averages for anomaly detection
        avg_cluster_count = np.mean(historical_cluster_counts) if historical_cluster_counts else 0
        avg_cluster_size = np.mean(historical_cluster_sizes) if historical_cluster_sizes else 0
        
        # Check for sudden cluster count drop (potential merge)
        current_cluster_count = len(temporal_frame_clusters[curr_frame])
        
        if avg_cluster_count > 0 and current_cluster_count < avg_cluster_count * 0.7:
            logger.warning(f"=" * 60)
            logger.warning(f"QUESTIONABLE CLUSTERING DETECTED IN FRAME {curr_frame}!")
            logger.warning(f"  Current: {current_cluster_count} clusters")
            logger.warning(f"  Historical average: {avg_cluster_count:.1f} clusters")
            logger.warning(f"  This might indicate incorrectly merged clusters")
            
            # Check for suspiciously large clusters
            for local_id, cluster_info in temporal_frame_clusters[curr_frame].items():
                cluster_size = cluster_info.get('point_count', 0)
                if avg_cluster_size > 0 and cluster_size > avg_cluster_size * 1.8:
                    logger.warning(f"  Cluster {local_id}: {cluster_size} points "
                                 f"({cluster_size/avg_cluster_size:.1f}x average)")
                    
                    # Check if this cluster's centroid is between two previous clusters
                    curr_centroid = np.array(cluster_info.get('centroid', [0, 0, 0]))
                    prev_centroids = []
                    for prev_uuid, prev_cluster in clusters_by_frame[prev_frame].items():
                        prev_centroids.append({
                            'uuid': prev_uuid,
                            'centroid': np.array(prev_cluster.get('centroid', [0, 0, 0])),
                            'size': prev_cluster.get('point_count', 0)
                        })
                    
                    # Check if current large cluster is near multiple previous clusters
                    nearby_prev_clusters = []
                    for prev_data in prev_centroids:
                        distance = np.linalg.norm(curr_centroid - prev_data['centroid'])
                        if distance < 100:  # Within 100cm
                            nearby_prev_clusters.append({
                                'uuid': prev_data['uuid'],
                                'distance': distance,
                                'size': prev_data['size']
                            })
                    
                    if len(nearby_prev_clusters) > 1:
                        logger.warning(f"    This cluster is near {len(nearby_prev_clusters)} previous clusters:")
                        for nearby in nearby_prev_clusters:
                            logger.warning(f"      - {nearby['uuid']}: {nearby['distance']:.1f}cm away, "
                                         f"{nearby['size']} points")
                        
                        merge_warnings.append({
                            'frame': curr_frame,
                            'cluster_id': local_id,
                            'size': cluster_size,
                            'nearby_previous': nearby_prev_clusters,
                            'likely_merged': True
                        })
                        
                        # Mark cluster as potentially merged
                        cluster_info['potentially_merged'] = True
                        cluster_info['merge_candidates'] = [n['uuid'] for n in nearby_prev_clusters]
            
            logger.warning(f"  RECOMMENDATION: Check XZ scanning results for frame {curr_frame}")
            logger.warning(f"=" * 60)
        
        clusters_by_frame[curr_frame] = {}
        matched_current = set()
        matched_previous = set()
        
        # Build match matrix between previous and current frame clusters
        match_matrix = []
        
        # Get previous frame clusters with their UUIDs
        prev_clusters_list = []
        for uuid, cluster_data in clusters_by_frame[prev_frame].items():
            prev_clusters_list.append((uuid, cluster_data))
        
        # CRITICAL FIX: Sort current frame clusters by centroid X before processing
        sorted_curr_clusters = sorted(
            temporal_frame_clusters[curr_frame].items(),
            key=lambda x: x[1].get('centroid', [0, 0, 0])[0]  # Sort by X coordinate
        )
        
        # Get current frame clusters (no UUIDs yet) - now sorted!
        curr_clusters_list = []
        for local_id, cluster_data in sorted_curr_clusters:
            curr_clusters_list.append((local_id, cluster_data))
        
        # Calculate match scores for all pairs
        for prev_idx, (prev_uuid, prev_cluster) in enumerate(prev_clusters_list):
            prev_centroid = np.array(prev_cluster.get('centroid', [0, 0, 0]))
            prev_size = prev_cluster.get('point_count', 0)
            prev_xz_footprint = calculate_xz_footprint(prev_cluster)
            
            for curr_idx, (curr_local_id, curr_cluster) in enumerate(curr_clusters_list):
                curr_centroid = np.array(curr_cluster.get('centroid', [0, 0, 0]))
                curr_size = curr_cluster.get('point_count', 0)
                curr_xz_footprint = calculate_xz_footprint(curr_cluster)
                
                # Calculate spatial distance
                distance = np.linalg.norm(curr_centroid - prev_centroid)
                
                # Calculate size similarity (0 to 1)
                if max(prev_size, curr_size) > 0:
                    size_ratio = min(prev_size, curr_size) / max(prev_size, curr_size)
                else:
                    size_ratio = 0
                
                # Calculate XZ footprint overlap
                xz_overlap = calculate_xz_overlap(prev_xz_footprint, curr_xz_footprint)
                
                # Score based on distance, size, and XZ overlap
                distance_score = max(0, 1.0 - distance / 50.0)  # 50cm max reasonable distance
                
                # Weighted combination
                combined_score = (distance_score * 0.5) + (size_ratio * 0.3) + (xz_overlap * 0.2)
                
                # Penalize if current cluster is marked as potentially merged
                if curr_cluster.get('potentially_merged', False):
                    # Check if this previous cluster is one of the merge candidates
                    if prev_uuid in curr_cluster.get('merge_candidates', []):
                        combined_score *= 1.2  # Boost score for likely match
                    else:
                        combined_score *= 0.8  # Reduce score for other matches
                
                match_matrix.append({
                    'prev_idx': prev_idx,
                    'prev_uuid': prev_uuid,
                    'curr_idx': curr_idx, 
                    'curr_local_id': curr_local_id,
                    'score': combined_score,
                    'distance': distance,
                    'size_ratio': size_ratio,
                    'xz_overlap': xz_overlap
                })
        
        # Sort matches by score (best first)
        match_matrix.sort(key=lambda x: x['score'], reverse=True)
        
        # Assign matches greedily (best score first)
        for match in match_matrix:
            # Skip if already matched
            if match['prev_idx'] in matched_previous or match['curr_idx'] in matched_current:
                continue
            
            # Accept match if score is good enough
            if match['score'] >= 0.4:  # Threshold for accepting a match
                # REUSE THE UUID FROM PREVIOUS FRAME!
                existing_uuid = match['prev_uuid']
                curr_cluster = temporal_frame_clusters[curr_frame][match['curr_local_id']]
                
                # Check for sudden size increase (potential merge)
                prev_size = clusters_by_frame[prev_frame][existing_uuid].get('point_count', 0)
                curr_size = curr_cluster.get('point_count', 0)
                
                if prev_size > 0 and curr_size > prev_size * 1.5:
                    logger.warning(f"Frame {curr_frame}: Cluster {existing_uuid} grew significantly "
                                 f"({prev_size} -> {curr_size} points, {curr_size/prev_size:.1f}x)")
                    if curr_cluster.get('potentially_merged', False):
                        logger.warning(f"  This cluster likely absorbed another object!")
                
                # Update tracked cluster with new frame data
                tracked_clusters[existing_uuid]['last_seen'] = curr_frame
                tracked_clusters[existing_uuid]['frames'].append(curr_frame)
                tracked_clusters[existing_uuid]['voxel_history'][curr_frame] = curr_cluster.get('voxel_metadata', {})
                tracked_clusters[existing_uuid]['voxel_counts'][curr_frame] = curr_cluster.get('voxel_count', 0)
                tracked_clusters[existing_uuid]['pattern_history'][curr_frame] = curr_cluster.get('pattern_distribution', {})
                tracked_clusters[existing_uuid]['centroid_history'][curr_frame] = curr_cluster.get('centroid', [0, 0, 0])
                tracked_clusters[existing_uuid]['size_history'][curr_frame] = curr_cluster.get('point_count', 0)
                tracked_clusters[existing_uuid]['xz_footprint_history'][curr_frame] = calculate_xz_footprint(curr_cluster)
                
                # Mark if potentially merged
                if curr_cluster.get('potentially_merged', False):
                    if 'merge_events' not in tracked_clusters[existing_uuid]:
                        tracked_clusters[existing_uuid]['merge_events'] = []
                    tracked_clusters[existing_uuid]['merge_events'].append({
                        'frame': curr_frame,
                        'candidates': curr_cluster.get('merge_candidates', [])
                    })
                
                # Store in frame mapping WITH SAME UUID
                clusters_by_frame[curr_frame][existing_uuid] = curr_cluster.copy()
                clusters_by_frame[curr_frame][existing_uuid]['uuid'] = existing_uuid
                
                # Mark as matched
                matched_previous.add(match['prev_idx'])
                matched_current.add(match['curr_idx'])
                
                logger.info(f"Frame {curr_frame}: Matched cluster {existing_uuid} "
                           f"(dist={match['distance']:.1f}cm, size_ratio={match['size_ratio']:.2f}, "
                           f"xz_overlap={match['xz_overlap']:.2f})")
        
        # Handle unmatched current clusters (new objects or splits)
        # CRITICAL FIX: Ensure new cluster IDs are assigned consistently
        next_cluster_id = len(tracked_clusters)
        for curr_idx, (curr_local_id, curr_cluster) in enumerate(curr_clusters_list):
            if curr_idx not in matched_current:
                # Check if this might be a split from merged cluster
                is_split = False
                if curr_cluster.get('was_split', False):
                    logger.info(f"Frame {curr_frame}: Detected split cluster from XZ scanning")
                    is_split = True
                
                # Generate new UUID for truly new cluster
                new_uuid = f"cluster_{next_cluster_id:02d}"
                next_cluster_id += 1
                
                # Initialize new tracked cluster
                tracked_clusters[new_uuid] = {
                    'first_seen': curr_frame,
                    'last_seen': curr_frame,
                    'frames': [curr_frame],
                    'voxel_history': {curr_frame: curr_cluster.get('voxel_metadata', {})},
                    'voxel_counts': {curr_frame: curr_cluster.get('voxel_count', 0)},
                    'pattern_history': {curr_frame: curr_cluster.get('pattern_distribution', {})},
                    'centroid_history': {curr_frame: curr_cluster.get('centroid', [0, 0, 0])},
                    'size_history': {curr_frame: curr_cluster.get('point_count', 0)},
                    'xz_footprint_history': {curr_frame: calculate_xz_footprint(curr_cluster)},
                    'was_split': is_split,
                    'split_info': curr_cluster.get('split_type', None) if is_split else None
                }
                
                # Store in frame mapping
                clusters_by_frame[curr_frame][new_uuid] = curr_cluster.copy()
                clusters_by_frame[curr_frame][new_uuid]['uuid'] = new_uuid
                
                if is_split:
                    logger.info(f"Frame {curr_frame}: New cluster {new_uuid} (SPLIT) with {curr_cluster.get('point_count', 0)} points at X={curr_cluster.get('centroid', [0,0,0])[0]:.1f}")
                else:
                    logger.info(f"Frame {curr_frame}: New cluster {new_uuid} with {curr_cluster.get('point_count', 0)} points at X={curr_cluster.get('centroid', [0,0,0])[0]:.1f}")
        
        # Log disappeared clusters (potential merge victims)
        disappeared_clusters = []
        for prev_idx, (prev_uuid, prev_cluster) in enumerate(prev_clusters_list):
            if prev_idx not in matched_previous:
                disappeared_clusters.append({
                    'uuid': prev_uuid,
                    'last_centroid': prev_cluster.get('centroid', [0, 0, 0]),
                    'last_size': prev_cluster.get('point_count', 0)
                })
                logger.info(f"Frame {curr_frame}: Cluster {prev_uuid} disappeared")
        
        # Check if disappeared clusters might have been merged
        if disappeared_clusters and current_cluster_count < len(prev_clusters_list):
            logger.warning(f"Frame {curr_frame}: {len(disappeared_clusters)} clusters disappeared, "
                         f"possible merge event")
            
            # Find which current cluster might have absorbed them
            for curr_uuid, curr_cluster in clusters_by_frame[curr_frame].items():
                if curr_cluster.get('potentially_merged', False):
                    logger.warning(f"  Cluster {curr_uuid} likely absorbed the disappeared clusters")
        
        # Update historical statistics
        historical_cluster_counts.append(current_cluster_count)
        if len(historical_cluster_counts) > 5:  # Keep last 5 frames
            historical_cluster_counts.pop(0)
        
        for cluster_data in temporal_frame_clusters[curr_frame].values():
            historical_cluster_sizes.append(cluster_data.get('point_count', 0))
        if len(historical_cluster_sizes) > 20:  # Keep last 20 cluster sizes
            historical_cluster_sizes = historical_cluster_sizes[-20:]
    
    # Summary
    persistent_clusters = [uuid for uuid, data in tracked_clusters.items() 
                          if len(data['frames']) > 1]
    
    logger.info(f"=" * 60)
    logger.info(f"Tracking complete: {len(tracked_clusters)} total clusters, {len(persistent_clusters)} persistent")
    
    # Log tracking summary
    for cluster_uuid, data in tracked_clusters.items():
        frames = data['frames']
        if len(frames) > 1:
            merge_info = ""
            if 'merge_events' in data:
                merge_info = f" [MERGE EVENTS: {len(data['merge_events'])}]"
            logger.info(f"  {cluster_uuid}: present in frames {min(frames)}-{max(frames)} "
                       f"({len(frames)} frames){merge_info}")
    
    # Report merge warnings
    if merge_warnings:
        logger.warning(f"=" * 60)
        logger.warning(f"MERGE WARNINGS SUMMARY: {len(merge_warnings)} potential incorrect merges detected")
        for warning in merge_warnings:
            logger.warning(f"  Frame {warning['frame']}: Cluster likely contains merged objects")
        logger.warning(f"=" * 60)
    
    return {
        'clusters_by_frame': clusters_by_frame,
        'tracked_clusters': tracked_clusters,
        'persistent_count': len(persistent_clusters),
        'total_count': len(tracked_clusters),
        'merge_warnings': merge_warnings
    }

def calculate_xz_footprint(cluster_info):
    """Calculate XZ footprint of a cluster for better tracking."""
    centroid = cluster_info.get('centroid', [0, 0, 0])
    
    # Use bounding box if available, otherwise estimate from centroid
    if 'bbox' in cluster_info:
        bbox = cluster_info['bbox']
        return {
            'x_min': bbox['min'][0],
            'x_max': bbox['max'][0],
            'z_min': bbox['min'][2],
            'z_max': bbox['max'][2],
            'center_x': centroid[0],
            'center_z': centroid[2]
        }
    else:
        # Estimate footprint from centroid with tolerance
        tolerance = 30  # 30cm tolerance
        return {
            'x_min': centroid[0] - tolerance,
            'x_max': centroid[0] + tolerance,
            'z_min': centroid[2] - tolerance,
            'z_max': centroid[2] + tolerance,
            'center_x': centroid[0],
            'center_z': centroid[2]
        }


def calculate_xz_overlap(footprint1, footprint2):
    """Calculate overlap ratio between two XZ footprints."""
    # Calculate intersection
    x_overlap = max(0, min(footprint1['x_max'], footprint2['x_max']) - 
                    max(footprint1['x_min'], footprint2['x_min']))
    z_overlap = max(0, min(footprint1['z_max'], footprint2['z_max']) - 
                    max(footprint1['z_min'], footprint2['z_min']))
    
    if x_overlap <= 0 or z_overlap <= 0:
        return 0.0
    
    intersection = x_overlap * z_overlap
    
    # Calculate areas
    area1 = (footprint1['x_max'] - footprint1['x_min']) * (footprint1['z_max'] - footprint1['z_min'])
    area2 = (footprint2['x_max'] - footprint2['x_min']) * (footprint2['z_max'] - footprint2['z_min'])
    
    # Use Jaccard index (IoU)
    union = area1 + area2 - intersection
    
    if union > 0:
        return intersection / union
    return 0.0

# Update save_temporal_data to handle the new structure
def save_temporal_data(state_bank, output_file, frame_buffer=None, voxel_grids_by_frame=None):
    """
    Save temporal consistency data to file with complete voxel metadata for reconstruction.
    FIXED: Now saves all voxel patterns and metadata needed to reconstruct synthetic points.
    
    Args:
        state_bank: ClusterStateBank instance with frame clusters
        output_file: Path to output JSON file
        frame_buffer: Optional FrameBuffer for additional context
        voxel_grids_by_frame: Dict mapping frame_num to enhanced_grid instance
    """
    logger.info(f"Saving temporal consistency data to {output_file}")
    
    # Verify we have data to save
    if not state_bank.frame_clusters:
        logger.error("ERROR: state_bank.frame_clusters is EMPTY! No data to save!")
        logger.error("  This means add_frame_clusters() was never called after clustering")
        # Create empty structure so the file isn't completely broken
        data = {
            "metadata": {
                "version": "5.0",
                "format": "frame-centric-with-voxel-metadata",
                "total_frames": 0,
                "total_unique_clusters": 0,
                "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                "error": "No frame clusters were added to state_bank"
            },
            "frames": {},
            "cluster_registry": {}
        }
        with open(output_file, 'w') as f:
            f.write(_format_compact_json(data))
        return
    
    data = {
        "metadata": {
            "version": "5.0",  # New version for metadata-complete format
            "format": "frame-centric-with-voxel-metadata",
            "total_frames": len(state_bank.frame_clusters),
            "total_unique_clusters": len(state_bank.global_cluster_registry),
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "voxel_resolution": 2.0,  # 2cm voxels
            "reconstruction_enabled": True
        },
        "frames": {},
        "cluster_registry": {}
    }
    
    # Process each frame
    for frame_num in sorted(state_bank.frame_clusters.keys()):
        frame_clusters = state_bank.frame_clusters[frame_num]
        
        data["frames"][str(frame_num)] = {
            "cluster_count": len(frame_clusters),
            "clusters": {}
        }
        
        # Process each cluster in this frame
        for cluster_uuid, cluster_data in sorted(frame_clusters.items()):
            
            # Extract complete voxel metadata for reconstruction
            voxel_metadata_dict = {}
            
            # Get voxel indices for this cluster
            voxel_indices = cluster_data.get('voxel_indices', [])
            
            # If we have the enhanced grid for this frame, extract full metadata
            if voxel_grids_by_frame and frame_num in voxel_grids_by_frame:
                voxel_grid = voxel_grids_by_frame[frame_num]
                
                for voxel_idx in voxel_indices:
                    voxel_key = str(voxel_idx)
                    voxel_info = {}
                    
                    # Get basic metadata
                    if hasattr(voxel_grid, 'cell_metadata') and voxel_idx in voxel_grid.cell_metadata:
                        metadata = voxel_grid.cell_metadata[voxel_idx]
                        voxel_info['centroid'] = metadata.get('centroid', [0, 0, 0])
                        voxel_info['density'] = metadata.get('density', 0)
                        voxel_info['y_continuity'] = metadata.get('y_continuity', {})
                    
                    # Get VoxelPattern with y-plane data
                    if hasattr(voxel_grid, 'voxel_patterns') and voxel_idx in voxel_grid.voxel_patterns:
                        pattern = voxel_grid.voxel_patterns[voxel_idx]
                        
                        # Store overall pattern
                        if hasattr(pattern, 'surface_type'):
                            voxel_info['surface_type'] = pattern.surface_type
                        if hasattr(pattern, 'normal') and pattern.normal is not None:
                            voxel_info['normal'] = pattern.normal if isinstance(pattern.normal, list) else pattern.normal.tolist()
                        
                        # Store y_plane_1 data with pattern for filtering
                        if pattern.y_plane_1:
                            y1_data = {}
                            if hasattr(pattern.y_plane_1, 'pattern_id'):
                                # It's a YPlaneData object
                                y1_data['pattern_id'] = pattern.y_plane_1.pattern_id.name if hasattr(pattern.y_plane_1.pattern_id, 'name') else str(pattern.y_plane_1.pattern_id)
                                y1_data['point_count'] = pattern.y_plane_1.point_count
                                y1_data['centroid_offset'] = pattern.y_plane_1.centroid_offset.tolist() if hasattr(pattern.y_plane_1.centroid_offset, 'tolist') else list(pattern.y_plane_1.centroid_offset)
                            
                            # Get shape bounds if stored separately
                            shape_key = f'{voxel_idx}_y_plane_1_shape_bounds'
                            if hasattr(voxel_grid, 'cell_metadata') and voxel_idx in voxel_grid.cell_metadata:
                                if 'y_plane_1_shape_bounds' in voxel_grid.cell_metadata[voxel_idx]:
                                    y1_data['shape_bounds'] = voxel_grid.cell_metadata[voxel_idx]['y_plane_1_shape_bounds']
                            
                            voxel_info['y_plane_1'] = y1_data
                        
                        # Store y_plane_2 data with pattern for filtering
                        if pattern.y_plane_2:
                            y2_data = {}
                            if hasattr(pattern.y_plane_2, 'pattern_id'):
                                # It's a YPlaneData object
                                y2_data['pattern_id'] = pattern.y_plane_2.pattern_id.name if hasattr(pattern.y_plane_2.pattern_id, 'name') else str(pattern.y_plane_2.pattern_id)
                                y2_data['point_count'] = pattern.y_plane_2.point_count
                                y2_data['centroid_offset'] = pattern.y_plane_2.centroid_offset.tolist() if hasattr(pattern.y_plane_2.centroid_offset, 'tolist') else list(pattern.y_plane_2.centroid_offset)
                            
                            # Get shape bounds if stored separately
                            if hasattr(voxel_grid, 'cell_metadata') and voxel_idx in voxel_grid.cell_metadata:
                                if 'y_plane_2_shape_bounds' in voxel_grid.cell_metadata[voxel_idx]:
                                    y2_data['shape_bounds'] = voxel_grid.cell_metadata[voxel_idx]['y_plane_2_shape_bounds']
                            
                            voxel_info['y_plane_2'] = y2_data
                    
                    # Determine if this voxel should be filtered during reconstruction
                    should_skip = False
                    if 'y_plane_1' in voxel_info and voxel_info['y_plane_1'].get('pattern_id') in ['SPARSE', 'Y_CONTINUOUS_SPARSE']:
                        should_skip = True
                    if 'y_plane_2' in voxel_info and voxel_info['y_plane_2'].get('pattern_id') in ['SPARSE', 'Y_CONTINUOUS_SPARSE']:
                        should_skip = True
                    voxel_info['skip_reconstruction'] = should_skip
                    
                    # ONLY add if it has y_plane data
                    if 'y_plane_1' in voxel_info or 'y_plane_2' in voxel_info:
                        voxel_metadata_dict[voxel_key] = voxel_info
            
            else:
                # Fallback: try to extract from cluster_data if it was stored there
                voxel_metadata_dict = cluster_data.get('voxel_metadata', {})
                if not voxel_metadata_dict:
                    logger.warning(f"Frame {frame_num}, Cluster {cluster_uuid}: No voxel metadata available!")
            
            # Build complete cluster info with metadata for reconstruction
            cluster_info = {
                "point_count": cluster_data.get('point_count', 0),
                "voxel_count": len(voxel_indices),
                "voxel_indices": voxel_indices,  # List of voxel indices
                "voxel_metadata": voxel_metadata_dict,  # Complete metadata for reconstruction!
                "pattern_summary": cluster_data.get('pattern_distribution', {}),
                "y1_voxels": cluster_data.get('y1_voxels', 0),
                "y2_voxels": cluster_data.get('y2_voxels', 0),
                "both_y_voxels": cluster_data.get('both_y_voxels', 0),
                "centroid": cluster_data.get('centroid', [0, 0, 0]),
                "x_span": cluster_data.get('x_span', 0),
                "y_span": cluster_data.get('y_span', 0),
                "z_span": cluster_data.get('z_span', 0),
                "method": cluster_data.get('method', 'unknown')
            }
            
            data["frames"][str(frame_num)]["clusters"][cluster_uuid] = cluster_info
    
    # Add cluster registry for tracking
    for cluster_uuid, registry_info in state_bank.global_cluster_registry.items():
        data["cluster_registry"][cluster_uuid] = {
            "first_seen": registry_info.get('first_seen'),
            "last_seen": registry_info.get('last_seen'),
            "frames_present": registry_info.get('frames_present', []),
            "total_points": registry_info.get('total_points', 0),
            "stable": registry_info.get('stable', False)
        }
    
    # Custom JSON encoder for numpy arrays and enums
    class MetadataEncoder(json.JSONEncoder):
        def default(self, obj):
            # Handle numpy arrays
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            # Handle numpy integers
            if isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            # Handle numpy floats
            if isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            # Handle numpy booleans
            if isinstance(obj, np.bool_):
                return bool(obj)
            # Handle enums by name
            if hasattr(obj, 'name'):
                return obj.name
            # Handle enum values
            if hasattr(obj, 'value'):
                return obj.value
            # Default fallback
            return super().default(obj)
    
    # Save to file
    try:
        # Normalize numpy types to Python primitives via the encoder
        # (json.loads(json.dumps(...)) round-trip), then write compactly
        # with arrays of numbers on one line.
        _norm_text = json.dumps(data, cls=MetadataEncoder)
        _norm_data = json.loads(_norm_text)
        with open(output_file, 'w') as f:
            f.write(_format_compact_json(_norm_data))
        
        # Verify what we saved
        total_voxels_with_metadata = 0
        frames_with_metadata = 0
        clusters_with_metadata = set()
        sparse_voxels_marked = 0
        
        for frame_num, frame_data in data["frames"].items():
            frame_has_metadata = False
            for cluster_uuid, cluster_info in frame_data["clusters"].items():
                voxel_metadata = cluster_info.get('voxel_metadata', {})
                if voxel_metadata:
                    total_voxels_with_metadata += len(voxel_metadata)
                    clusters_with_metadata.add(cluster_uuid)
                    frame_has_metadata = True
                    
                    # Count voxels marked for skipping
                    for voxel_key, voxel_info in voxel_metadata.items():
                        if voxel_info.get('skip_reconstruction', False):
                            sparse_voxels_marked += 1
                            
            if frame_has_metadata:
                frames_with_metadata += 1
        
        logger.info(f"[OK] Saved temporal consistency data with FULL METADATA:")
        logger.info(f"  - {len(data['frames'])} frames total")
        # logger.info(f"  - {len(data['cluster_registry'])} unique clusters tracked")
        logger.info(f"  - {frames_with_metadata} frames have voxel metadata")
        logger.info(f"  - {len(clusters_with_metadata)} clusters have voxel metadata")
        logger.info(f"  - {total_voxels_with_metadata} total voxels with reconstruction metadata")
        logger.info(f"  - {sparse_voxels_marked} sparse voxels marked for filtering")
        
        if total_voxels_with_metadata == 0:
            logger.error("[OK] WARNING: No voxels have metadata - reconstruction will fail!")
            logger.error("  Check that voxel_grids_by_frame is being passed correctly")
        
    except Exception as e:
        logger.error(f"Failed to save temporal data: {e}")
        import traceback
        logger.error(traceback.format_exc())

def compare_methods(file_pattern, args, frame_buffer=None, state_bank=None):
   """
   Compare different clustering methods on the same data.
   
   Args:
       file_pattern: File pattern for point cloud files
       args: Command-line arguments
   """
   logger.info("=================================================================")
   logger.info("                STANDARD CLUSTERING METHOD                        ")
   logger.info("=================================================================")
   
   # Create copy of args for standard clustering
   standard_args = argparse.Namespace(**vars(args))
   standard_args.opencv_enhance = False
   standard_args.use_grid = False
   standard_args.refine_boundaries = False
   standard_args.use_contour_alignment = False
   standard_args.clusters_dir = None
   standard_args.clusters_by_label_dir = None
   standard_args.output = os.path.join(args.output, "standard")
   
   # Process with standard clustering
   standard_clusters, _, _, _ = process_multiple_frames(file_pattern, standard_args)
   
   # Conditionally compare with contour alignment method
   if args.use_contour_alignment:
       logger.info("\n\n=================================================================")
       logger.info("                CONTOUR ALIGNMENT CLUSTERING METHOD                 ")
       logger.info("=================================================================")
       
       # Create copy of args for contour alignment clustering
       alignment_args = argparse.Namespace(**vars(args))
       alignment_args.use_contour_alignment = True
       alignment_args.output = os.path.join(args.output, "contour_aligned")
       
       # Process with contour alignment clustering
       alignment_clusters, _, _, _ = process_multiple_frames(file_pattern, alignment_args)
   
   logger.info("\n\n=================================================================")
   logger.info("                    COMPARISON COMPLETE                           ")
   logger.info("=================================================================")
   
   # Print summary of results
   if standard_clusters:
       logger.info(f"Standard clustering results: {standard_args.output}/")
       logger.info(f"  Frames: {len(standard_clusters)}")
       avg_clusters = sum(len(clusters) for clusters in standard_clusters.values()) / len(standard_clusters)
       logger.info(f"  Average clusters per frame: {avg_clusters:.1f}")
   
   if args.use_contour_alignment and 'alignment_clusters' in locals():
       logger.info(f"Contour alignment clustering results: {alignment_args.output}/")
       logger.info(f"  Frames: {len(alignment_clusters)}")
       avg_clusters = sum(len(clusters) for clusters in alignment_clusters.values()) / len(alignment_clusters)
       logger.info(f"  Average clusters per frame: {avg_clusters:.1f}")

def main():
    """Main entry point for clustering script with camera optimization"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='3D Point Cloud Clustering with Pose Estimation')
    #parser = argparse.ArgumentParser(description='RO-NOUS Point Cloud Clustering')
    
    # Input/output parameters
    parser.add_argument('--path', type=str, default=None, 
                        help='Path to point cloud file or pattern for multiple files')
    parser.add_argument('--frames_dir', type=str, default=None, 
                        help='Directory containing original video frames')
    parser.add_argument('--clusters_dir', type=str, default=None,
                        help='Directory containing existing clusters by ID')
    parser.add_argument('--clusters_by_label_dir', type=str, default=None,
                        help='Directory containing existing clusters by label')
    parser.add_argument('--output', type=str, default='clustering_output', 
                        help='Output folder for files')
    parser.add_argument('--log_file', type=str, default=None,
                        help='File to write logs to (in addition to console)')
    parser.add_argument('--log_level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level')
    
    # Method selection
    parser.add_argument('--use_dbscan', action='store_true', 
                        help='Use DBSCAN instead of HDBSCAN')
    parser.add_argument('--opencv_enhance', action='store_true', 
                        help='Use OpenCV features to enhance clustering')
    parser.add_argument('--use_grid', action='store_true',
                        help='Use 3D occupancy grid for clustering')
    parser.add_argument('--compare_methods', action='store_true',
                        help='Compare different clustering methods')
    parser.add_argument('--refine_boundaries', action='store_true',
                        help='Use existing clusters and edge detection to refine results')
    
    # Camera optimization
    parser.add_argument('--optimize_camera', action='store_true',
                        help='Run camera parameter optimization on first frame')
    
    # Manual alignment parameters
    parser.add_argument('--use_alignment', action='store_true',
                        help='Use manual alignment with specific parameters')
    parser.add_argument('--alignment_params', type=str, default=None,
                        help='Manual alignment parameters as "dx,dy,scale_x,scale_y"')
    
    # Consistency tracking
    parser.add_argument('--maintain_consistent_ids', action='store_true', default=True,
                        help='Maintain consistent cluster IDs across frames')
    parser.add_argument('--overlap_threshold', type=float, default=0.5,
                        help='Overlap threshold for maintaining consistent cluster IDs')
    
    # Point filtering
    parser.add_argument('--filter_points', action='store_true', default=True,
                        help='Filter points but track them for potential reclamation')
    
    # DBSCAN parameters
    parser.add_argument('--eps', type=float, default=2.7, 
                        help='DBSCAN eps parameter')
    
    # HDBSCAN parameters
    parser.add_argument('--min_cluster_size', type=int, default=50, 
                        help='HDBSCAN min_cluster_size parameter')
    parser.add_argument('--min_samples', type=int, default=20, 
                        help='HDBSCAN/DBSCAN min_samples parameter')
    parser.add_argument('--cluster_selection_epsilon', type=float, default=5.0, 
                        help='HDBSCAN cluster_selection_epsilon parameter')
    parser.add_argument('--allow_single_cluster', action='store_true', default=True, 
                        help='Allow a single cluster to form in HDBSCAN')
    
    # Grid parameters
    parser.add_argument('--grid_resolution', type=float, default=1.0,
                        help='Resolution (cell size) for 3D occupancy grid')
    parser.add_argument('--min_points_per_cell', type=int, default=3,
                        help='Minimum points per cell to be considered occupied')
    parser.add_argument('--connectivity', type=int, default=26, choices=[6, 18, 26],
                        help='Connectivity type for grid-based clustering')
    parser.add_argument('--filter_noise', action='store_true',
                        help='Filter noise cells from grid')
    parser.add_argument('--visualize_grid', action='store_true',
                        help='Visualize the 3D grid')
    
    # Camera parameters
    parser.add_argument('--camera_position', type=str, default="47,28,-20.0", 
                        help='Camera position for projection (x,y,z)')
    parser.add_argument('--camera_target', type=str, default="-25.1,123.8,-28.3",
                        help='Camera target/look-at point (x,y,z)')
    parser.add_argument('--focal_length', type=float, default=27.5,
                        help='Camera focal length in mm')
    parser.add_argument('--field_of_view', type=float, default=66,
                        help='Camera field of view in degrees')
    parser.add_argument('--image_size', type=str, default="480,864",
                        help='Size of camera view images (width,height)')
    
    # Filtering parameters
    parser.add_argument('--min_percentage', type=float, default=3.0, 
                        help='Minimum percentage of points for significant clusters')
    
    # Visualization parameters
    parser.add_argument('--visualize', action='store_true', default=True, 
                        help='Generate visualizations')
    parser.add_argument('--max_points', type=int, default=10000, 
                        help='Maximum points to visualize')
    parser.add_argument('--sample', type=float, default=1.0, 
                        help='Sample fraction of points to use (0.0-1.0)')
    
    # Export parameters
    parser.add_argument('--export_clusters', action='store_true', default=True, 
                        help='Export clusters to separate files')
    parser.add_argument('--minimal_output', action='store_true', default=True,
                        help='Generate only essential outputs')
    
    # Video parameters
    parser.add_argument('--create_video', action='store_true', default=False, 
                        help='Create MP4 video from frames')
    parser.add_argument('--create_overlay_video', action='store_true', default=False,
                        help='Create video with point cloud overlaid on original frames')
    parser.add_argument('--fps', type=int, default=10, 
                        help='Frames per second for video output')
    
    parser.add_argument('--use_temporal', action='store_true',
                    help='Enable temporal clustering with 5-frame buffer')
    
    parser = add_enhanced_arguments(parser)
    
    args = parser.parse_args()
    
    # Initialize pose completion pipeline if enabled
    pose_pipeline = None
    if args.use_mmpose or args.use_intelligent_completion:
        pose_pipeline = PoseCompletionPipeline(args)
        print("[OK] Enhanced pose pipeline initialized")
    args.pose_pipeline = pose_pipeline
    
    # Initialize logging
    log_level = getattr(logging, args.log_level.upper())
    utils.setup_logging(log_level=log_level, log_file=args.log_file)
    
    # Parse camera parameters
    args.camera_position = utils.parse_camera_position(args.camera_position)
    args.camera_target = utils.parse_camera_target(args.camera_target)
    
    # Parse image size
    try:
        width, height = map(int, args.image_size.split(','))
        args.image_size = (width, height)
    except:
        logger.warning(f"Invalid image size format: {args.image_size}. Using default (480,864).")
        args.image_size = (480, 864)
    
    # Default path if none provided
    if args.path is None:
        args.path = "outputs/flesh/dummy_flesh_cop_frame_001.txt"
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Log command line parameters (minimal)
    logger.info("Starting clustering with the following parameters:")
    for arg, value in sorted(vars(args).items()):
        # Only log important parameters
        if arg in ['path', 'frames_dir', 'output', 'use_dbscan', 'use_grid', 
                   'grid_resolution', 'min_points_per_cell', 'connectivity', 'filter_noise',
                   'use_alignment', 'alignment_params', 'optimize_camera',
                   'refine_boundaries', 'opencv_enhance', 'create_video', 'create_overlay_video']:
            logger.info(f"  {arg}: {value}")
    
    # CAMERA OPTIMIZATION - Run before processing if requested
    # CAMERA OPTIMIZATION - Run before processing if requested
    if args.optimize_camera:
        logger.info("="*70)
        logger.info("CAMERA PARAMETER OPTIMIZATION USING CONTOUR MATCHING")
        logger.info("="*70)
        logger.info("Initial parameters will be used as starting point for iterative refinement")
        
        # Find first file
        is_pattern = '*' in args.path or '?' in args.path
        if is_pattern:
            files = sorted(glob.glob(args.path))
            if not files:
                logger.error("No files found for camera optimization")
                sys.exit(1)
            first_file = files[0]
        else:
            first_file = args.path
        
        # Load first frame CoP
        cop_points = point_cloud.load_point_cloud(first_file)
        if cop_points is None:
            logger.error("Could not load CoP for camera optimization")
            sys.exit(1)
        
        # Find corresponding frame
        frame_path = point_cloud.find_matching_frame(first_file, args.frames_dir)
        if frame_path and os.path.exists(frame_path):
            import cv2
            frame = cv2.imread(frame_path)
            
            if frame is not None:
                # Log initial parameters
                logger.info(f"Initial camera position: {args.camera_position}")
                logger.info(f"Initial camera target: {args.camera_target}")
                logger.info(f"Initial FOV: {args.field_of_view}deg")
                
                # Create initial params structure
                initial_params = {
                    'camera_position': args.camera_position,
                    'camera_target': args.camera_target,
                    'focal_length': args.focal_length,
                    'field_of_view': args.field_of_view
                }
                
                # Calculate initial score to compare against
                logger.info("Calculating initial parameters score...")
                edges = image_alignment.extract_edges(frame, method='combined')
                
                # Helper function to calculate score
                def calculate_params_score(params, cop_points, edges, image_size):
                    """Calculate overlap pixel count with visibility check"""
                    
                    # Project CoP points with given parameters
                    projected = opencv_integration.project_3d_to_2d(cop_points, **params, image_size=image_size)
                    
                    # Dilate edges for more forgiving scoring
                    kernel = np.ones((5,5), np.uint8)
                    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
                    
                    # Create mask from projected points
                    mask = np.zeros((image_size[1], image_size[0]), dtype=np.uint8)
                    visible_count = 0
                    
                    for x, y in projected:
                        px, py = int(round(x)), int(round(y))
                        if 0 <= px < image_size[0] and 0 <= py < image_size[1]:
                            cv2.circle(mask, (px, py), 3, 255, -1)
                            visible_count += 1
                    
                    # Calculate visibility percentage
                    visibility_percentage = (visible_count / len(cop_points)) * 100 if len(cop_points) > 0 else 0
                    
                    # Calculate overlap pixel count
                    overlap_pixels = int(np.sum((mask > 0) & (dilated_edges > 0)))
                    
                    return overlap_pixels, visibility_percentage, overlap_pixels
                
                initial_score, initial_vis, initial_align = calculate_params_score(
                    initial_params, cop_points, edges, args.image_size
                )
                logger.info(f"Initial alignment: {initial_align} overlap pixels")
                
                # Run contour-matching optimization
                best_params, score = opencv_integration.optimize_camera_parameters(
                    cop_points, frame, initial_params)
                
                # Check if optimization actually improved things
                if best_params is not None:
                    # Calculate the actual score of optimized parameters
                    opt_score, opt_vis, opt_align = calculate_params_score(
                        best_params, cop_points, edges, args.image_size
                    )
                    logger.info(f"Visibility: {opt_vis:.1f}% | Overlap: {opt_align} pixels")
                    
                    # Only use optimized parameters if they're actually better
                    if opt_score > initial_score + 5:
                        logger.info("="*70)
                        logger.info(f"OPTIMIZATION SUCCESSFUL - improved by {opt_score - initial_score} pixels")
                        logger.info("="*70)
                        
                        # Use optimized parameters
                        args.camera_position = best_params['camera_position']
                        args.camera_target = best_params['camera_target']
                        args.field_of_view = best_params['field_of_view']
                    else:
                        logger.warning("="*70)
                        logger.warning(f"Optimization did not improve results (only {opt_score:.1f} pix vs initial {initial_score:.1f} pix)")
                        logger.warning("Keeping initial parameters")
                        logger.warning("="*70)
        
        # Camera optimization complete - now continue with normal processing
        logger.info("="*70)
        logger.info("CAMERA OPTIMIZATION COMPLETE - PROCEEDING WITH FULL PROCESSING")
        logger.info("="*70)
    
    # Initialize temporal consistency components AFTER camera optimization
    frame_buffer = None
    # -----------------------------------------------------------------------
    # FIX (blocker): state_bank and frame_buffer must ALWAYS be initialized.
    # Previously they were only created when --use_temporal was passed.
    # Without state_bank, visualization.create_video_from_frames() receives
    # None every frame, recreates the skeleton from scratch each time, and
    # loses all temporal state: facing direction, pose history, bone-length
    # locks, and velocity clamping.  --use_temporal now only controls whether
    # the extra temporal-consistency reporting pass runs; the core objects are
    # unconditional.
    # -----------------------------------------------------------------------
    from temporal_consistency import FrameBuffer, ClusterStateBank
    frame_buffer = FrameBuffer(max_size=5)
    state_bank   = ClusterStateBank()
    logger.info("Initialized temporal consistency components (always-on)")

    # ===================== NEW: 2D Pose Stabilization =====================
    # Initialize pose stabilization components (always, not just with temporal)
    pose_2d_history = Pose2DHistory(buffer_size=5)
    facing_history = FacingHistory(required_consistent_frames=5)
    logger.info("Initialized 2D pose stabilization (Pose2DHistory, FacingHistory)")
    # ======================================================================

    if hasattr(args, 'use_temporal') and args.use_temporal:
        logger.info("--use_temporal: extended temporal-consistency reporting enabled")
        
    
        
    # ============== ADD THIS NEW CODE ==============
    # Initialize cluster coordinate system
    # cluster_coords = ClusterCoordinateSystem()
    # logger.info("Initialized Cluster Inner Coordinate System (ICCS)")
    
    # Add simplified visualization function if not exists
    if not hasattr(visualization, 'visualize_clusters_simple'):
        def visualize_clusters_simple(points, labels, original_frame_path=None, max_points=10000, filename=None):
            """Create a single, simple visualization of clusters."""
            import matplotlib.pyplot as plt
            import cv2
            import numpy as np
            
            # Subsample points if needed
            if len(points) > max_points:
                indices = np.random.choice(len(points), max_points, replace=False)
                vis_points = points[indices]
                vis_labels = labels[indices]
            else:
                vis_points = points
                vis_labels = labels
            
            # Create figure with subplots
            if original_frame_path and os.path.exists(original_frame_path):
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
                
                # Show original frame
                try:
                    frame = cv2.imread(original_frame_path)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    ax1.imshow(frame)
                    ax1.set_title("Original Frame")
                    ax1.axis('off')
                except Exception as e:
                    logger.error(f"Error loading frame: {str(e)}")
                    ax1.text(0.5, 0.5, "Error loading frame", ha='center', va='center')
                    ax1.axis('off')
            else:
                fig, ax2 = plt.subplots(figsize=(8, 6))
            
            # Get unique cluster labels
            unique_labels = np.unique(vis_labels)
            unique_labels = unique_labels[unique_labels >= 0]
            
            # Create color map
            colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels) + 1))
            
            # Plot noise points
            noise_mask = (vis_labels == -1)
            if np.any(noise_mask):
                noise_points = vis_points[noise_mask]
                ax2.scatter(noise_points[:, 0], noise_points[:, 1], c='gray', marker='.', s=1, alpha=0.3, label='Noise')
            
            # Plot each cluster
            for i, label in enumerate(unique_labels):
                cluster_mask = (vis_labels == label)
                cluster_points = vis_points[cluster_mask]
                ax2.scatter(cluster_points[:, 0], cluster_points[:, 1], c=[colors[i]], marker='.', s=5, alpha=0.7, label=f'Cluster {label}')
            
            # Set labels and title
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            ax2.set_title(f"Clusters ({len(unique_labels)} clusters)")
            ax2.grid(True, alpha=0.3)
            
            # Add legend with small size
            if len(unique_labels) <= 10:
                ax2.legend(loc='upper right', fontsize='x-small')
            
            plt.tight_layout()
            
            # Save figure if filename provided
            if filename:
                plt.savefig(filename, dpi=150)
                logger.info(f"Saved visualization to {filename}")
            
            plt.close(fig)
        
        # Add the function to the visualization module
        visualization.visualize_clusters_simple = visualize_clusters_simple
        logger.info("Added simplified visualization function")
    
    # Add grid visualization function if not exists
    if not hasattr(visualization, 'visualize_grid') and args.visualize_grid:
        def visualize_grid(grid_obj, threshold=1, filename=None, max_cells=10000):
            """Visualize 3D occupancy grid."""
            import matplotlib.pyplot as plt
            import numpy as np
            from mpl_toolkits.mplot3d import Axes3D
            
            occupied_cells = grid_obj.get_occupied_cells(threshold)
            
            if len(occupied_cells) > max_cells:
                logger.info(f"Subsampling grid from {len(occupied_cells)} to {max_cells} cells")
                indices = np.random.choice(len(occupied_cells), max_cells, replace=False)
                vis_cells = [occupied_cells[i] for i in indices]
            else:
                vis_cells = occupied_cells
            
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            if not vis_cells:
                ax.text(0, 0, 0, "No occupied cells", size=14, ha='center')
            else:
                cell_coords = np.array([grid_obj._cell_to_point(cell) for cell in vis_cells])
                occupancy_values = np.array([grid_obj.grid[cell] for cell in vis_cells])
                
                if np.max(occupancy_values) > np.min(occupancy_values):
                    normalized_occ = (occupancy_values - np.min(occupancy_values)) / (np.max(occupancy_values) - np.min(occupancy_values))
                else:
                    normalized_occ = np.ones(len(occupancy_values))
                
                sc = ax.scatter(cell_coords[:, 0], cell_coords[:, 1], cell_coords[:, 2], 
                               c=normalized_occ, marker='s', cmap='viridis', alpha=0.8)
                
                cbar = plt.colorbar(sc, ax=ax, shrink=0.7)
                cbar.set_label('Occupancy')
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'3D Grid (resolution={grid_obj.resolution}, threshold={threshold})')
            ax.view_init(elev=30, azim=45)
            
            plt.tight_layout()
            
            if filename:
                plt.savefig(filename, dpi=150)
                logger.info(f"Saved grid visualization to {filename}")
            
            plt.close(fig)
        
        visualization.visualize_grid = visualize_grid
        logger.info("Added grid visualization function")
    
    # Check if using multiple frames
    is_pattern = '*' in args.path or '?' in args.path
    
    # Run appropriate processing
    if args.compare_methods:
        if is_pattern:
            compare_methods(args.path, args, frame_buffer, state_bank)
        else:
            logger.warning("Method comparison requires multiple frames. Please use a file pattern.")
            process_single_frame(args.path, args, frame_buffer=frame_buffer, state_bank=state_bank)
    else:
        # Process frames normally
        if is_pattern:
            logger.info(f"Processing multiple frames using pattern: {args.path}")
            # Before calling process_multiple_frames, add:
        
            clusters_by_frame, frame_buffer, state_bank, voxel_grids_by_frame = process_multiple_frames(args.path, args)
            
            # ================================================================
            # SKELETON + MOVEMENT-INDEX PASS (main() call site)
            #
            # Mirror of the fix in process_multiple_frames: run
            # create_video_from_frames whenever ANY video output is
            # requested so skeleton shell-fitting and the
            # MovementIndexEngine always execute and write _p/_s JSON.
            # ================================================================
            _run_viz_pass_main = (args.create_video or getattr(args, 'create_overlay_video', False)) and len(clusters_by_frame) > 1

            # ================================================================
            # PHASE 2: Skeleton Fitting Pass
            # Runs AFTER Phase 1 (clustering + JSON writing) and BEFORE
            # Phase 3 (create_video_from_frames).  Reads frame JSONs + PLYs
            # written by Phase 1, fits 21-joint skeleton, writes _s JSON.
            # Wrapped in try/except so a missing skeleton_fitting module or
            # any fitting error falls back gracefully to inline fitting in
            # Phase 3 (visualization.py).
            # ================================================================
            if _run_viz_pass_main:
                # ==============================================================
                # CRITICAL: Delete stale results_s.json from PREVIOUS run.
                # The first video pass runs BEFORE Phase 2 and reads whatever
                # results_s files are on disk.  Stale files from a previous run
                # have wrong body_yaw values that contaminate PLYs and panels.
                # Phase 2 will write fresh results_s files immediately after.
                #
                # GATE: only delete if inline Phase 1 fitting did NOT already
                # write fresh results_s in the current process.  If it did,
                # the files on disk are this run's authoritative output and
                # must not be touched -- otherwise Phase 3 (visualization)
                # will see an empty results dir, take its fallback writer
                # path, and overwrite every frame_NNN_results_s.json with
                # its own lower-quality version (timestamp drift symptom).
                # ==============================================================
                _inline_fit_ran = hasattr(process_multiple_frames, '_fit_engine_init')
                if _inline_fit_ran:
                    logger.info("[SYNC] Skipping stale results_s cleanup -- inline "
                                "Phase 1 fitting already wrote fresh files this run")
                else:
                    _results_dir_clean = os.path.join(
                        getattr(args, 'output', None) or '.', 'results')
                    if os.path.isdir(_results_dir_clean):
                        import glob as _glob_clean
                        _stale = _glob_clean.glob(os.path.join(
                            _results_dir_clean, 'frame_*_results_s.json'))
                        if _stale:
                            for _sf in _stale:
                                os.remove(_sf)
                            logger.info(f"[SYNC] Deleted {len(_stale)} stale results_s.json "
                                       f"from previous run (prevents first-pass contamination)")

                try:
                    import skeleton_fitting
                    _results_dir = os.path.join(
                        getattr(args, 'output', None) or '.', 'results')
                    _surfaces_dir = os.path.join(
                        getattr(args, 'output', None) or '.', 'surfaces')
                    _voxel_grids = voxel_grids_by_frame if voxel_grids_by_frame else {}
                    _cam_params = {
                        'camera_position': args.camera_position,
                        'camera_target':   args.camera_target,
                        'focal_length':    args.focal_length,
                        'field_of_view':   args.field_of_view,
                        'panel_width':     args.image_size[0] if hasattr(args, 'image_size') else 480,
                        'panel_height':    args.image_size[1] if hasattr(args, 'image_size') else 864,
                    }
                    # Skip Phase 2 if per-frame fitting already ran in the loop
                    if hasattr(process_multiple_frames, '_fit_engine_init'):
                        logger.info("[PHASE 2] SKIPPED - per-frame fitting already ran in Phase 1 loop")
                    else:
                        logger.info("[PHASE 2] Starting skeleton fitting pass...")
                        _fit_results = skeleton_fitting.run_fitting_pass(
                            results_dir          = _results_dir,
                            surfaces_dir         = _surfaces_dir,
                            state_bank           = state_bank,
                            voxel_grids_by_frame = _voxel_grids,
                            args                 = args,
                            camera_params        = _cam_params,
                        )
                        _n_ok = sum(1 for v in _fit_results.values() if v)
                        logger.info(f"[PHASE 2] Skeleton fitting complete: {_n_ok}/{len(_fit_results)} frames OK")
                except ImportError:
                    logger.warning("[PHASE 2] skeleton_fitting module not found - skipping, inline fitting in Phase 3 will run")
                except Exception as _e:
                    logger.warning(f"[PHASE 2] Skeleton fitting failed ({_e}) - falling back to inline fitting in Phase 3")

            # ================================================================
            # PHASE 2.5: Body Descriptor Pass
            # Skip if body_descriptor.json files already written inline
            # during Phase 1 (simultaneous orchestration).
            # ================================================================
            if _run_viz_pass_main:
                _bd_results_dir = os.path.join(
                    getattr(args, 'output', None) or '.', 'results')
                # Check if inline Phase 1 already wrote body_descriptor files
                _bd_existing = glob.glob(os.path.join(
                    _bd_results_dir, 'frame_*_body_descriptor.json'))
                if len(_bd_existing) >= len(clusters_by_frame) * 0.8:
                    logger.info(f"[PHASE 2.5] SKIPPED - {len(_bd_existing)} body_descriptor.json "
                                f"already written by Phase 1 inline")
                else:
                    try:
                        import body_descriptor
                        _bd_voxel_grids = voxel_grids_by_frame if voxel_grids_by_frame else {}
                        logger.info("[PHASE 2.5] Starting body descriptor pass...")
                        _bd_results = body_descriptor.run_descriptor_pass(
                            results_dir          = _bd_results_dir,
                            voxel_grids_by_frame = _bd_voxel_grids,
                            state_bank           = state_bank,
                        )
                        _bd_ok = sum(1 for v in _bd_results.values() if v)
                        logger.info(f"[PHASE 2.5] Body descriptor complete: {_bd_ok}/{len(_bd_results)} frames")
                    except ImportError:
                        logger.warning("[PHASE 2.5] body_descriptor module not found - skipping")
                    except Exception as _e:
                        logger.warning(f"[PHASE 2.5] Body descriptor failed ({_e}) - continuing without descriptors")

            if _run_viz_pass_main:
                logger.info(f"Creating visualization video with {args.fps} fps...")
                
                # Find files matching pattern
                files = sorted(glob.glob(args.path))
                
                # Create video directory
                video_dir = os.path.join(args.output, "videos")
                os.makedirs(video_dir, exist_ok=True)
                
                # Generate video file name based on method
                method_name = "Grid" if args.use_grid else "DBSCAN" if args.use_dbscan else "HDBSCAN"
                if args.opencv_enhance:
                    method_name = "OpenCV+" + method_name
                if args.use_alignment:
                    method_name += "+Aligned"
                
                video_file = os.path.join(video_dir, f"{method_name}_video.mp4")
                
                # Create video
                if hasattr(visualization, 'create_video_from_frames'):
                    visualization.create_video_from_frames(
                        clusters_by_frame, 
                        files, 
                        args.frames_dir, 
                        video_file, 
                        fps=args.fps,
                        camera_position=args.camera_position,
                        camera_target=args.camera_target,
                        focal_length=args.focal_length,
                        field_of_view=args.field_of_view,
                        image_size=args.image_size,
                        state_bank=state_bank
                    )
                    logger.info(f"Video created: {video_file}")
                else:
                    logger.error("create_video_from_frames function not found in visualization module")
            
            # Create overlay video if requested
            if args.create_overlay_video and len(clusters_by_frame) > 1:
                logger.info("Creating overlay video...")
                
                # Create video directory
                video_dir = os.path.join(args.output, "videos")
                os.makedirs(video_dir, exist_ok=True)
                
                # Generate overlay video file name
                method_name = "Grid" if args.use_grid else "DBSCAN" if args.use_dbscan else "HDBSCAN"
                if args.opencv_enhance:
                    method_name = "OpenCV+" + method_name
                if args.use_alignment:
                    method_name += "+Aligned"
                
                overlay_video_file = os.path.join(video_dir, f"{method_name}_overlay_video.mp4")
                
                # Find files matching pattern
                files = sorted(glob.glob(args.path))
                
                # Create overlay video
                create_overlay_video(clusters_by_frame, files, args.frames_dir, overlay_video_file, args, args.fps)
        else:
            logger.info(f"Processing single frame: {args.path}")
            process_single_frame(args.path, args, None, None, frame_buffer=frame_buffer, state_bank=state_bank)
    
    # Create flat-projection diagnostic videos (XZ and YZ) at end of run.
    # Renders whenever per-frame `frame_NNN_proj.json` sidecars exist (i.e.
    # whenever STEP 9.65 ran successfully).  Independent of --create_video
    # and --create_overlay_video — no new CLI flag.
    try:
        create_flat_projection_videos(args, fps=getattr(args, 'fps', 10))
    except Exception as _fpv_e:
        logger.warning(f"[FLAT-VIDEO] generation failed: {_fpv_e}")
    
    logger.info("Clustering completed successfully")

# =============================================================================
# FLAT-PROJECTION VIDEOS
# Reads per-frame `frame_NNN_proj.json` sidecars saved by STEP 9.65 and renders
# two MP4s: top-down (XZ) and side-view (YZ).  Visualization-only — does not
# affect any clustering decision.
# =============================================================================

def create_flat_projection_videos(args, fps=10):
    """Render flat XZ and YZ projection MP4s from per-frame sidecar JSONs.

    Layout (per frame, per axis):
      - Black canvas, fixed scale (3 px/cm, see PX_PER_CM below).
      - Live CoP cells → gray dots.
      - Per-cluster footprint cells → colored dots (color stable across
        frames via UUID hashing into the CLUSTER_COLORS palette).
      - Unowned CoP cells (CoP minus union of cluster footprints) →
        red dots.  These are the diagnostic signal: any red cell is
        territory the live CoP saw but no cluster claimed.
      - Bottom-left text overlay: frame number, cluster count,
        unowned %, and a UNOWNED-HIGH warning when >30%.
    """
    try:
        import cv2
        import numpy as np
        import glob
        import json
        import os
    except ImportError as _e:
        logger.error(f"[FLAT-VIDEO] required packages missing: {_e}")
        return False

    # Pull all sidecars
    proj_dir = os.path.join(args.output, "results")
    proj_files = sorted(glob.glob(os.path.join(proj_dir, "frame_*_proj.json")))
    if not proj_files:
        logger.warning(f"[FLAT-VIDEO] No projection sidecars in {proj_dir}; "
                       f"skipping flat-projection videos")
        return False

    # ---------- First pass: scan all frames to determine canvas extents.
    # Canvas size is fixed across the whole video so the dots don't jump
    # frame-to-frame; we use the union of all (CoP ∪ cluster) cell ranges
    # padded to a margin.
    PX_PER_CM = 3
    MARGIN_CELLS = 10  # extra padding around the canvas

    xz_x_min = xz_z_min = yz_y_min = yz_z_min = float('inf')
    xz_x_max = xz_z_max = yz_y_max = yz_z_max = float('-inf')

    for pf in proj_files:
        try:
            with open(pf, 'r') as f:
                d = json.load(f)
        except Exception:
            continue

        def _scan_flat(flat_list, mins, maxs):
            # flat_list = [a0,b0,a1,b1,...]
            for i in range(0, len(flat_list), 2):
                a, b = flat_list[i], flat_list[i + 1]
                if a < mins[0]: mins[0] = a
                if a > maxs[0]: maxs[0] = a
                if b < mins[1]: mins[1] = b
                if b > maxs[1]: maxs[1] = b

        xz_mins = [xz_x_min, xz_z_min]; xz_maxs = [xz_x_max, xz_z_max]
        yz_mins = [yz_y_min, yz_z_min]; yz_maxs = [yz_y_max, yz_z_max]
        _scan_flat(d.get('cop_xz', []), xz_mins, xz_maxs)
        _scan_flat(d.get('cop_yz', []), yz_mins, yz_maxs)
        for c in d.get('clusters', []):
            _scan_flat(c.get('xz', []), xz_mins, xz_maxs)
            _scan_flat(c.get('yz', []), yz_mins, yz_maxs)
        xz_x_min, xz_z_min = xz_mins; xz_x_max, xz_z_max = xz_maxs
        yz_y_min, yz_z_min = yz_mins; yz_y_max, yz_z_max = yz_maxs

    if xz_x_min == float('inf'):
        logger.warning("[FLAT-VIDEO] All sidecars empty - skipping")
        return False

    # Per-axis canvas dimensions.  Z axis grows DOWNWARD on canvas (low Z =
    # ground = bottom of frame), so we flip the Z mapping when blitting.
    res_per_pixel = 1  # one cell per pixel-block; we apply PX_PER_CM at draw time
    xz_w_cells = xz_x_max - xz_x_min + 1 + 2 * MARGIN_CELLS
    xz_h_cells = xz_z_max - xz_z_min + 1 + 2 * MARGIN_CELLS
    yz_w_cells = yz_y_max - yz_y_min + 1 + 2 * MARGIN_CELLS
    yz_h_cells = yz_z_max - yz_z_min + 1 + 2 * MARGIN_CELLS

    # Per-cell pixel size — each cell becomes a small block on canvas.
    # We pick a per-cell pixel scale so the resulting video isn't tiny.
    # Target ≈ 480x720 canvas, but keep the ASPECT RATIO of the cell range.
    def _scale_to_target(cells_w, cells_h, target_max=1080):
        # Pick block-pixel scale so max(width_px, height_px) ≈ target_max.
        scale_w = target_max / max(1, cells_w)
        scale_h = target_max / max(1, cells_h)
        scale = min(scale_w, scale_h)
        scale = max(2, int(round(scale)))  # at least 2 px per cell
        return scale

    xz_scale = _scale_to_target(xz_w_cells, xz_h_cells, target_max=1080)
    yz_scale = _scale_to_target(yz_w_cells, yz_h_cells, target_max=1080)

    xz_canvas_w = xz_w_cells * xz_scale
    xz_canvas_h = xz_h_cells * xz_scale
    yz_canvas_w = yz_w_cells * yz_scale
    yz_canvas_h = yz_h_cells * yz_scale

    # Reserve bottom strip for text
    TEXT_STRIP_H = 60
    xz_frame_h = xz_canvas_h + TEXT_STRIP_H
    yz_frame_h = yz_canvas_h + TEXT_STRIP_H

    # Cluster colour palette (BGR) — 20 colours, UUID-hashed for stability.
    # All entries are saturated/bright so clusters never blend into the
    # dark-gray CoP backdrop.
    PALETTE_BGR = [
        (255, 200,   0), (  0, 255, 255), (255,   0, 255),
        (  0, 255,   0), (255, 128,   0), (128, 255, 128),
        (255, 128, 255), (128, 128, 255), (  0, 128, 255),
        (255,   0, 128), (255, 255, 128), (128, 255, 255),
        (  0, 165, 255), (255, 192, 128), (128, 192, 255),
        (255, 128, 192), (192, 255, 128), (128, 255, 192),
        (192, 128, 255), (255, 192, 192),
    ]

    def _color_for_uuid(uuid_str, ckey, is_person):
        # Person always gets the first palette entry (cyan/teal) so it's
        # visually consistent with the existing main video.
        if is_person:
            return PALETTE_BGR[0]
        # Hash UUID into palette index, falling back to ckey.
        seed = uuid_str if uuid_str else ckey
        if not seed:
            return PALETTE_BGR[1]
        # Simple deterministic hash — sum of bytes mod len(palette).
        h = sum(ord(c) for c in seed) % len(PALETTE_BGR)
        # Skip index 0 (reserved for person)
        if h == 0:
            h = 1
        return PALETTE_BGR[h]

    # Output paths
    video_dir = os.path.join(args.output, "videos")
    os.makedirs(video_dir, exist_ok=True)
    method_name = "Grid" if getattr(args, 'use_grid', False) else \
                  "DBSCAN" if getattr(args, 'use_dbscan', False) else "HDBSCAN"
    xz_video_path = os.path.join(video_dir, f"{method_name}_flat_xz.mp4")
    yz_video_path = os.path.join(video_dir, f"{method_name}_flat_yz.mp4")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    xz_writer = cv2.VideoWriter(xz_video_path, fourcc, fps,
                                (xz_canvas_w, xz_frame_h))
    yz_writer = cv2.VideoWriter(yz_video_path, fourcc, fps,
                                (yz_canvas_w, yz_frame_h))

    if not xz_writer.isOpened() or not yz_writer.isOpened():
        logger.error(f"[FLAT-VIDEO] VideoWriter failed to open "
                     f"(xz={xz_writer.isOpened()}, yz={yz_writer.isOpened()})")
        return False

    logger.info(f"[FLAT-VIDEO] Writing flat XZ ({xz_canvas_w}x{xz_frame_h}) "
                f"and YZ ({yz_canvas_w}x{yz_frame_h}) at {fps} fps "
                f"from {len(proj_files)} sidecar JSONs")

    def _flat_to_pairs(flat):
        for i in range(0, len(flat), 2):
            yield flat[i], flat[i+1]

    def _draw_lasso(canvas, cells_iter, axis_min_a, axis_min_b, scale,
                    color, thickness=2, cop_dense_iter=None,
                    single_loop=None):
        """Render a contour around `cells_iter`.

        Two modes (auto-selected from `cop_dense_iter`, or forced by
        `single_loop`):

        ── CLUSTER MODE (single_loop=True, default when cop_dense_iter
           is given) ─────────────────────────────────────────────
        One outer contour per call.  The cluster (mask M) is grown
        aggressively to bridge all islands within the dense-CoP hull
        (D), then the SINGLE contour with the largest overlap with M
        is drawn.  Floating contours that don't contain any original
        cluster cell are dropped — they are CoP-clip artifacts, not
        body pieces.

        ── DENSE-COP MODE (single_loop=False, default when
           cop_dense_iter is None) ────────────────────────────────
        Multiple contours OK — the dense-CoP itself naturally has one
        loop per body present in the scene (person, chair, etc.).
        Each natural blob is closed cleanly with morphological close
        before contour extraction.

        Both modes use the SAME closing/bridging step so the cluster
        loop fits the dense-CoP loop snugly (Issue #3).
        """
        # Resolve mode
        if single_loop is None:
            single_loop = (cop_dense_iter is not None)

        ch = canvas.shape[0] // scale
        cw = canvas.shape[1] // scale
        cells = list(cells_iter)
        if not cells:
            return

        # ── 1. Build cluster mask ──
        mask = np.zeros((ch, cw), dtype=np.uint8)
        for a, b in cells:
            col = a - axis_min_a + MARGIN_CELLS
            row = ch - 1 - (b - axis_min_b + MARGIN_CELLS)
            if 0 <= col < cw and 0 <= row < ch:
                mask[row, col] = 255

        # ── 2. Build dense-CoP hull mask (when provided) ──
        cop_hull = None
        if cop_dense_iter is not None:
            cop = np.zeros((ch, cw), dtype=np.uint8)
            for a, b in cop_dense_iter:
                col = a - axis_min_a + MARGIN_CELLS
                row = ch - 1 - (b - axis_min_b + MARGIN_CELLS)
                if 0 <= col < cw and 0 <= row < ch:
                    cop[row, col] = 255
            cop_hull = cv2.dilate(cop, np.ones((3, 3), np.uint8),
                                  iterations=2)

        if single_loop:
            # ── CLUSTER MODE ─────────────────────────────────────
            # Grow cluster aggressively to bridge ALL islands; the
            # dense-CoP hull will rein it back in so it cannot
            # escape this body's territory.
            grown = cv2.dilate(mask, np.ones((5, 5), np.uint8),
                               iterations=3)
            if cop_hull is not None:
                grown = cv2.bitwise_and(grown, cop_hull)
            grown = cv2.bitwise_or(grown, mask)
            grown = cv2.morphologyEx(
                grown, cv2.MORPH_CLOSE,
                np.ones((3, 3), np.uint8), iterations=2)

            cnts, _ = cv2.findContours(
                grown, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                return

            # Keep the contour with the largest overlap with the
            # original cluster mask.  Floating CoP-clip artifacts
            # (no cluster cell inside) are dropped.
            best_cnt = None; best_score = -1
            for cnt in cnts:
                blob = np.zeros((ch, cw), dtype=np.uint8)
                cv2.drawContours(blob, [cnt], -1, 255, cv2.FILLED)
                ovl = int(cv2.countNonZero(cv2.bitwise_and(blob, mask)))
                if ovl > best_score:
                    best_score = ovl; best_cnt = cnt
            if best_cnt is None or best_score <= 0:
                best_cnt = max(cnts, key=cv2.contourArea)
            cv2.drawContours(canvas, [best_cnt * scale], -1,
                             color, thickness, cv2.LINE_AA)
        else:
            # ── DENSE-COP MODE ─────────────────────────────────
            # One contour per natural body in the dense-CoP; close
            # small internal gaps so each body is a clean loop.
            closed = cv2.morphologyEx(
                mask, cv2.MORPH_CLOSE,
                np.ones((3, 3), np.uint8), iterations=2)
            cnts, _ = cv2.findContours(
                closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                return
            # Drop tiny noise specks (< 6 cells area), keep everything
            # else so each real body has its own loop.
            kept = [c for c in cnts if cv2.contourArea(c) >= 6]
            if not kept:
                kept = [max(cnts, key=cv2.contourArea)]
            cv2.drawContours(
                canvas, [c * scale for c in kept], -1,
                color, thickness, cv2.LINE_AA)

    # ---------- Per-frame render
    for pf in proj_files:
        try:
            with open(pf, 'r') as f:
                d = json.load(f)
        except Exception as _e:
            logger.warning(f"[FLAT-VIDEO] failed to read {pf}: {_e}")
            continue

        xz_canvas=np.zeros((xz_canvas_h,xz_canvas_w,3),dtype=np.uint8)
        _xzch=xz_canvas_h//xz_scale
        for a,b in _flat_to_pairs(d.get('cop_xz',[])):
            col=(a-xz_x_min+MARGIN_CELLS)*xz_scale+xz_scale//2
            row=(_xzch-1-(b-xz_z_min+MARGIN_CELLS))*xz_scale+xz_scale//2
            if 0<=col<xz_canvas_w and 0<=row<xz_canvas_h: xz_canvas[row,col]=(80,80,80)
        _draw_lasso(xz_canvas,_flat_to_pairs(d.get('cop_xz_dense',[])),
                    xz_x_min,xz_z_min,xz_scale,(0,220,0),2)
        for c in d.get('clusters',[]):
            col=_color_for_uuid(c.get('uuid',''),c.get('key',''),c.get('is_person',False))
            _draw_lasso(xz_canvas,_flat_to_pairs(c.get('xz',[])),
                        xz_x_min,xz_z_min,xz_scale,col,2,
                        cop_dense_iter=_flat_to_pairs(d.get('cop_xz_dense',[])))
        yz_canvas=np.zeros((yz_canvas_h,yz_canvas_w,3),dtype=np.uint8)
        _yzch=yz_canvas_h//yz_scale
        for a,b in _flat_to_pairs(d.get('cop_yz',[])):
            col=(a-yz_y_min+MARGIN_CELLS)*yz_scale+yz_scale//2
            row=(_yzch-1-(b-yz_z_min+MARGIN_CELLS))*yz_scale+yz_scale//2
            if 0<=col<yz_canvas_w and 0<=row<yz_canvas_h: yz_canvas[row,col]=(80,80,80)
        _draw_lasso(yz_canvas,_flat_to_pairs(d.get('cop_yz_dense',[])),
                    yz_y_min,yz_z_min,yz_scale,(0,220,0),2)
        for c in d.get('clusters',[]):
            col=_color_for_uuid(c.get('uuid',''),c.get('key',''),c.get('is_person',False))
            _draw_lasso(yz_canvas,_flat_to_pairs(c.get('yz',[])),
                        yz_y_min,yz_z_min,yz_scale,col,2,
                        cop_dense_iter=_flat_to_pairs(d.get('cop_yz_dense',[])))
        def _mts(w,fn,nc,ax):
            s=np.zeros((TEXT_STRIP_H,w,3),dtype=np.uint8)
            cv2.putText(s,f"Frame {fn}  axis={ax}  clusters={nc}",
                (10,25),cv2.FONT_HERSHEY_SIMPLEX,0.6,(200,200,200),1,cv2.LINE_AA)
            cv2.putText(s,"gray=CoP  green=dense lasso  color=cluster lasso",
                (10,50),cv2.FONT_HERSHEY_SIMPLEX,0.45,(160,160,160),1,cv2.LINE_AA)
            return s
        fn=d.get('frame_num',0); nc=d.get('cluster_count_dict',d.get('cluster_count',0))
        xz_strip=_mts(xz_canvas_w,fn,nc,'XZ'); yz_strip=_mts(yz_canvas_w,fn,nc,'YZ')

        xz_full = np.vstack([xz_canvas, xz_strip])
        yz_full = np.vstack([yz_canvas, yz_strip])

        xz_writer.write(xz_full)
        yz_writer.write(yz_full)

    xz_writer.release()
    yz_writer.release()
    logger.info(f"[FLAT-VIDEO] Wrote {xz_video_path}")
    logger.info(f"[FLAT-VIDEO] Wrote {yz_video_path}")
    return True


# Add the create_overlay_video function to support the --create_overlay_video parameter
def create_overlay_video(clusters_by_frame, files, frames_dir, output_file, args, fps=10):
    """
    Create overlay video using saved frame results and reconstructed synthetic points.
    FIXED: Uses saved voxel metadata instead of regenerating everything from scratch.
    UPDATED: Mask is transparent and scaled 140% from bottom-center anchor.
    """
    try:
        import cv2
        import matplotlib.cm as cm
        import numpy as np
        import tempfile
        import shutil
        import os
        import glob
        import json
        from ro_nous_clustering import point_cloud
    except ImportError as e:
        logger.error(f"Required packages not available: {e}")
        return False
    
    logger.info(f"Creating fitted overlay video at {output_file}")
    temp_dir = tempfile.mkdtemp()
    
    # Define the same cluster colors as visualization.py
    CLUSTER_COLORS = [
        (255, 200, 0),   # Cyan - Label 0
        (0, 255, 255),   # Yellow - Label 1
        (255, 0, 255),   # Magenta - Label 2
        (0, 255, 0),     # Green - Label 3
        (255, 128, 0),   # Blue-ish - Label 4
        (128, 255, 128), # Light green - Label 5
        (255, 128, 255), # Light magenta - Label 6
        (128, 128, 255), # Light red - Label 7
        (0, 128, 255),   # Orange - Label 8
        (255, 0, 128),   # Purple-ish - Label 9
        (255, 255, 128), # Light yellow - Label 10
        (128, 255, 255), # Light cyan - Label 11
        (192, 192, 192), # Silver - Label 12
        (255, 192, 128), # Peach - Label 13
        (128, 192, 255), # Sky blue - Label 14
        (255, 128, 192), # Pink - Label 15
        (192, 255, 128), # Lime - Label 16
        (128, 255, 192), # Mint - Label 17
        (192, 128, 255), # Lavender - Label 18
        (255, 192, 192), # Light pink - Label 19
    ]
    
    # Overlay parameters
    MASK_SCALE = 1.0       # 140% scale from bottom-center
    MASK_ALPHA = 0.6       # Transparency (0=invisible, 1=opaque)
   
    
    # Handle different key formats in clusters_by_frame
    logger.info(f"clusters_by_frame keys: {list(clusters_by_frame.keys())[:5]}...")
    
    frame_mapping = {}
    for key in clusters_by_frame.keys():
        try:
            if isinstance(key, int):
                frame_num = key
            elif isinstance(key, str):
                if key.isdigit():
                    frame_num = int(key)
                elif key.startswith('frame_'):
                    frame_num = int(key.replace('frame_', ''))
                else:
                    frame_num = int(key)
            else:
                logger.warning(f"Unexpected key type: {type(key)} for key {key}")
                continue
            
            frame_mapping[frame_num] = key
        except Exception as e:
            logger.warning(f"Could not parse frame number from key: {key}, error: {e}")
            continue
    
    frame_nums = sorted(frame_mapping.keys())
    logger.info(f"Processing {len(frame_nums)} frames: {frame_nums}")
    
    if len(frame_nums) == 0:
        logger.error("No valid frame numbers found")
        return False
    
    # Process each frame
    for frame_num in frame_nums:
        logger.info(f"Processing overlay for frame {frame_num}")
        
        # Load original frame
        original_frame_path = None
        if frames_dir:
            original_frame_path = point_cloud.find_matching_frame(
                f"frame_{frame_num:03d}", frames_dir
            )
            if original_frame_path:
                logger.info(f"Found exact matching frame: {os.path.basename(original_frame_path)}")
        
        if not original_frame_path:
            # Try alternative naming patterns
            possible_names = [
                f"frame_{frame_num}.jpg",
                f"frame_{frame_num:02d}.jpg", 
                f"frame_{frame_num:03d}.jpg",
                f"frame{frame_num}.jpg"
            ]
            for name in possible_names:
                test_path = os.path.join(frames_dir, name) if frames_dir else name
                if os.path.exists(test_path):
                    original_frame_path = test_path
                    logger.info(f"Found frame at: {original_frame_path}")
                    break
        
        if not original_frame_path:
            logger.warning(f"No original frame found for frame {frame_num}, creating blank")
            original_frame = np.zeros((480, 864, 3), dtype=np.uint8)
        else:
            original_frame = cv2.imread(original_frame_path)
            if original_frame is None:
                original_frame = np.zeros((480, 864, 3), dtype=np.uint8)
        
        logger.info(f"Original frame dimensions: {original_frame.shape[:2]}")
        
        # ===============================================
        # CRITICAL FIX: LOAD SAVED FRAME RESULTS
        # ===============================================
        
        # Try to load the saved frame results
        results_file = os.path.join(args.output, "results", f"frame_{frame_num:03d}_results.json")
        
        if os.path.exists(results_file):
            logger.info(f"Loading saved results from: {results_file}")
            
            try:
                with open(results_file, 'r') as f:
                    frame_results = json.load(f)
                
                # Reconstruct synthetic points from saved clusters
                all_points = []
                all_cluster_labels = []
                
                clusters_info = frame_results.get('clusters_info', {})
                
                if not clusters_info:
                    logger.warning(f"No clusters_info in saved results for frame {frame_num}")
                    cv2.imwrite(os.path.join(temp_dir, f"overlay_{frame_num:05d}.png"), original_frame)
                    continue
                
                # Process each cluster
                for cluster_key, cluster_data in clusters_info.items():
                    # Skip empty clusters
                    if cluster_data.get('total_points', 0) == 0:
                        continue
                    
                    # Reconstruct synthetic points from voxel metadata.
                    # RC-3: after utils.py patch, centroid-fallback returns a
                    # tagged dict.  Skip ghost 125-pt blobs before rendering.
                    _rc3_recon = utils.reconstruct_from_voxel_metadata(cluster_data, voxel_size=2.0)
                    if isinstance(_rc3_recon, dict):
                        if _rc3_recon.get('reconstruction_method') == 'centroid_fallback':
                            logger.warning(
                                f"[RC-3 SKIP] ghost cluster {cluster_key} "
                                f"- centroid fallback only, skipping overlay render"
                            )
                            continue
                        synthetic_points = _rc3_recon['points']
                    else:
                        synthetic_points = _rc3_recon

                    if len(synthetic_points) > 0:
                        all_points.extend(synthetic_points)
                        # Use cluster_key as label
                        all_cluster_labels.extend([cluster_key] * len(synthetic_points))
                        logger.info(f"Reconstructed {len(synthetic_points)} points for {cluster_key}")
                
                if not all_points:
                    logger.warning(f"No points reconstructed for frame {frame_num}")
                    cv2.imwrite(os.path.join(temp_dir, f"overlay_{frame_num:05d}.png"), original_frame)
                    continue
                
                # Convert to arrays
                all_points = np.array(all_points)
                logger.info(f"Total reconstructed: {len(all_points)} synthetic points from saved metadata")
                
                # Count points per cluster
                unique_clusters = list(set(all_cluster_labels))
                cluster_counts = {}
                for cluster in unique_clusters:
                    cluster_counts[cluster] = all_cluster_labels.count(cluster)
                logger.info(f"Cluster distribution: {cluster_counts}")
                
            except Exception as e:
                logger.error(f"Error loading saved results: {e}")
                # Fall back to old method
                logger.warning("Falling back to clusters_by_frame data")
                
                # Get clusters using the correct key format
                cluster_key = frame_mapping[frame_num]
                current_clusters = clusters_by_frame[cluster_key]
                
                if not current_clusters:
                    logger.warning(f"No clusters for frame {frame_num}")
                    cv2.imwrite(os.path.join(temp_dir, f"overlay_{frame_num:05d}.png"), original_frame)
                    continue
                
                # Use old method of extracting points from clusters
                all_points = []
                all_cluster_labels = []
                
                for cluster_id, cluster_info in current_clusters.items():
                    if 'points' in cluster_info and len(cluster_info['points']) > 0:
                        points = cluster_info['points']
                        all_points.extend(points)
                        all_cluster_labels.extend([cluster_id] * len(points))
                
                if not all_points:
                    logger.warning(f"No points available for frame {frame_num}")
                    cv2.imwrite(os.path.join(temp_dir, f"overlay_{frame_num:05d}.png"), original_frame)
                    continue
                
                all_points = np.array(all_points)
                logger.info(f"Using {len(all_points)} original points (not synthetic)")
        else:
            logger.warning(f"No saved results found for frame {frame_num}: {results_file}")
            cv2.imwrite(os.path.join(temp_dir, f"overlay_{frame_num:05d}.png"), original_frame)
            continue
        
        # ===============================================
        # PROJECT POINTS TO 2D
        # ===============================================
        
        from ro_nous_clustering.opencv_integration import project_3d_to_2d
        
        projected_points = project_3d_to_2d(
            all_points,
            camera_position=args.camera_position if hasattr(args, 'camera_position') else [-47.0, 28.0, -20.0],
            camera_target=args.camera_target if hasattr(args, 'camera_target') else [-25.1,123.8,-28.3],
            focal_length=args.focal_length if hasattr(args, 'focal_length') else 27.5,
            field_of_view=args.field_of_view if hasattr(args, 'field_of_view') else 66.0,
            image_size=(480, 864),  # FIX: Use panel dimensions!
            original_frame=original_frame  # Pass for alignment
        )
        
        logger.info(f"Projected {len(projected_points)} points to 2D")
        
        # ===============================================
        # CREATE COLOR MAPPING
        # ===============================================
        
        # Create consistent color mapping for clusters
        unique_clusters = list(set(all_cluster_labels))
        unique_clusters.sort(key=str)
        
        cluster_color_map = {}
        for cluster_label in unique_clusters:
            # Extract cluster number from label (e.g., "cluster_00" -> 0)
            try:
                if isinstance(cluster_label, str) and 'cluster_' in cluster_label:
                    cluster_num = int(cluster_label.split('_')[1])
                else:
                    cluster_num = int(cluster_label)
            except:
                cluster_num = unique_clusters.index(cluster_label)
            
            # Use consistent color based on cluster number
            color_bgr = CLUSTER_COLORS[cluster_num % len(CLUSTER_COLORS)]
            cluster_color_map[cluster_label] = color_bgr
        
        logger.info(f"Drawing {len(unique_clusters)} clusters with colors: {list(cluster_color_map.keys())}")
        
        # ===============================================
        # SCALE POINTS 140% FROM BOTTOM-CENTER ANCHOR
        # ===============================================
        
        h, w = original_frame.shape[:2]
        anchor_x = w / 2.0      # Bottom-center X
        anchor_y = float(h)     # Bottom-center Y (bottom of image)
        
        scaled_points = []
        for (x, y) in projected_points:
            # Scale from bottom-center anchor
            new_x = anchor_x + (x - anchor_x) * MASK_SCALE
            new_y = anchor_y + (y - anchor_y) * MASK_SCALE
            scaled_points.append((new_x, new_y))
        
        logger.info(f"Scaled {len(scaled_points)} points by {MASK_SCALE}x from bottom-center")
        
        # ===============================================
        # DRAW POINTS ON TRANSPARENT MASK LAYER
        # ===============================================
        
        # Create a separate mask layer (BGRA for alpha)
        mask_layer = np.zeros((h, w, 3), dtype=np.uint8)
        alpha_mask = np.zeros((h, w), dtype=np.float32)
        
        points_drawn = 0
        for i, (x, y) in enumerate(scaled_points):
            px, py = int(round(x)), int(round(y))
            
            # Check bounds
            if 0 <= px < w and 0 <= py < h:
                cluster_label = all_cluster_labels[i]
                cv_color = cluster_color_map.get(cluster_label, (255, 255, 255))
                cv2.circle(mask_layer, (px, py), 2, cv_color, -1)
                cv2.circle(alpha_mask, (px, py), 2, 1.0, -1)
                points_drawn += 1
        
        logger.info(f"Drew {points_drawn} points on mask layer")
        
        # ===============================================
        # ALPHA BLEND MASK ONTO ORIGINAL FRAME
        # ===============================================
        
        # Expand alpha_mask to 3 channels for blending
        alpha_3ch = np.stack([alpha_mask] * 3, axis=-1)
        
        # Blend: overlay = original * (1 - alpha * mask) + mask_layer * (alpha * mask)
        overlay = original_frame.astype(np.float32)
        mask_float = mask_layer.astype(np.float32)
        
        blend_factor = alpha_3ch * MASK_ALPHA
        overlay = overlay * (1.0 - blend_factor) + mask_float * blend_factor
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        
        # ===============================================
        # ADD TEXT OVERLAYS (on top of blended result)
        # ===============================================
        
        # Add frame number text
        cv2.putText(overlay, f"Frame {frame_num}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Add cluster legend
        y_pos = 60
        for cluster_label in sorted(unique_clusters, key=str):
            color = cluster_color_map[cluster_label]
            count = all_cluster_labels.count(cluster_label)
            text = f"{cluster_label}: {count} pts"
            cv2.rectangle(overlay, (10, y_pos-15), (30, y_pos), color, -1)
            cv2.putText(overlay, text, (40, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_pos += 25
        
        # Save overlay frame
        overlay_path = os.path.join(temp_dir, f"overlay_{frame_num:05d}.png")
        cv2.imwrite(overlay_path, overlay)
        logger.info(f"Saved overlay frame {frame_num} with transparent scaled mask")
    
    # ===============================================
    # CREATE VIDEO FROM FRAMES
    # ===============================================
    
    overlay_files = sorted(glob.glob(os.path.join(temp_dir, "*.png")))
    
    if not overlay_files:
        logger.error("No overlay frames generated")
        shutil.rmtree(temp_dir)
        return False
    
    logger.info(f"Creating video from {len(overlay_files)} overlay frames")
    
    # Get dimensions from first frame
    first = cv2.imread(overlay_files[0])
    if first is None:
        logger.error("Could not read first overlay frame")
        shutil.rmtree(temp_dir)
        return False
    
    h, w = first.shape[:2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_file, fourcc, fps, (w, h))
    
    if not video.isOpened():
        logger.error("Could not create video writer")
        shutil.rmtree(temp_dir)
        return False
    
    # Write all frames
    frames_written = 0
    for f in overlay_files:
        frame = cv2.imread(f)
        if frame is not None:
            video.write(frame)
            frames_written += 1
        else:
            logger.warning(f"Could not read frame: {f}")
    
    video.release()
    logger.info(f"Overlay video created: {output_file}")
    logger.info(f"Frames written: {frames_written}/{len(overlay_files)}")
    
    # Clean up
    shutil.rmtree(temp_dir)
    logger.info("Cleaned up temporary files")
    
    return True

def generate_synthetic_points_voxel(centroid, point_count, voxel_size=2.0):
    """
    Generate synthetic points around a centroid using 2cm voxel grid.
    
    Args:
        centroid: [x, y, z] center position in cm
        point_count: Number of points to generate
        voxel_size: Size of voxels in cm (2.0 for 2cm grid)
    
    Returns:
        numpy array of synthetic points
    """
    import numpy as np
    import logging
    logger = logging.getLogger(__name__)
    
    if point_count <= 0:
        point_count = 10  # Minimum points for visibility
    
    # Estimate cluster radius based on point count
    # Typical density: 5-10 points per 2cm voxel
    points_per_voxel = 7
    num_voxels = max(1, point_count // points_per_voxel)
    
    # Calculate radius to encompass required voxels
    # Use cube root of volume to get radius
    voxel_volume = voxel_size ** 3
    total_volume = num_voxels * voxel_volume
    radius = (3 * total_volume / (4 * np.pi)) ** (1/3)
    
    # Ensure reasonable radius (10-50cm for human body parts)
    radius = np.clip(radius, 10.0, 50.0)
    
    synthetic_points = []
    
    # Generate voxel grid around centroid
    x_steps = int(2 * radius / voxel_size) + 1
    y_steps = int(2 * radius / voxel_size) + 1
    z_steps = int(2 * radius / voxel_size) + 1
    
    x_range = np.linspace(centroid[0] - radius, centroid[0] + radius, x_steps)
    y_range = np.linspace(centroid[1] - radius, centroid[1] + radius, y_steps)
    z_range = np.linspace(centroid[2] - radius, centroid[2] + radius, z_steps)
    
    # Create points in voxels within sphere
    for x in x_range:
        for y in y_range:
            for z in z_range:
                # Check if voxel center is within sphere
                dist = np.sqrt((x - centroid[0])**2 + (y - centroid[1])**2 + (z - centroid[2])**2)
                if dist <= radius * 0.9:  # Use 90% of radius for better shape
                    # Add points within this voxel
                    for _ in range(points_per_voxel):
                        # Random offset within voxel
                        offset = np.random.uniform(-voxel_size/2, voxel_size/2, 3)
                        point = [x + offset[0], y + offset[1], z + offset[2]]
                        synthetic_points.append(point)
    
    if not synthetic_points:
        # Fallback: create points directly at centroid
        logger.warning(f"No voxels within radius, creating points at centroid")
        synthetic_points = np.random.randn(point_count, 3) * voxel_size + centroid
    else:
        synthetic_points = np.array(synthetic_points)
        
        # Adjust to exact point count
        if len(synthetic_points) > point_count:
            # Randomly sample to get exact count
            indices = np.random.choice(len(synthetic_points), point_count, replace=False)
            synthetic_points = synthetic_points[indices]
        elif len(synthetic_points) < point_count:
            # Add more points with Gaussian distribution
            shortage = point_count - len(synthetic_points)
            extra_points = np.random.randn(shortage, 3) * (radius/3) + centroid
            synthetic_points = np.vstack([synthetic_points, extra_points])
    
    logger.debug(f"Generated {len(synthetic_points)} points in sphere r={radius:.1f}cm using {voxel_size}cm voxels")
    
    return synthetic_points

def create_point_overlay(original_frame, projected_points, labels, clusters, cluster_colors, image_size):
    """
    Create an overlay of point cloud on original frame using projected points.
    
    Args:
        original_frame: Original frame image
        projected_points: 2D projections of points
        labels: Cluster labels for each point
        clusters: Dictionary of clusters
        cluster_colors: Dictionary mapping cluster IDs to colors
        image_size: Image size (width, height)
        
    Returns:
        Overlay image
    """
    try:
        import cv2
        import numpy as np
        
        # Create a copy of the original frame
        overlay = original_frame.copy()
        
        # Draw points colored by cluster
        for cluster_id, cluster_info in clusters.items():
            # Find points for this cluster using the label
            label = cluster_info['original_label']
            points_mask = (labels == label)
            cluster_points = projected_points[points_mask]
            
            # Get color for this cluster
            color = cluster_colors.get(cluster_id, (1.0, 1.0, 1.0))
            # Convert color from matplotlib format (0-1) to OpenCV format (0-255)
            cv_color = (int(color[0]*255), int(color[1]*255), int(color[2]*255))
            
            # Draw each point
            for x, y in cluster_points:
                # Convert to integer coordinates
                px, py = int(round(x)), int(round(y))
                
                # Skip if out of bounds
                if px < 0 or px >= image_size[0] or py < 0 or py >= image_size[1]:
                    continue
                
                # Draw point
                cv2.circle(overlay, (px, py), 2, cv_color, -1)
        
        return overlay
    except Exception as e:
        logger.error(f"Error creating point overlay: {str(e)}")
        return None

def create_cluster_overlay(original_frame, clusters, cluster_colors, args):
    """
    Create an overlay of clusters on original frame by projecting cluster points.
    
    Args:
        original_frame: Original frame image
        clusters: Dictionary of clusters for this frame
        cluster_colors: Dictionary mapping cluster IDs to colors
        args: Command-line arguments with projection parameters
        
    Returns:
        Overlay image
    """
    try:
        import cv2
        import numpy as np
        from ro_nous_clustering import opencv_integration
        
        # Create a copy of the original frame
        overlay = original_frame.copy()
        
        # Process each cluster
        for cluster_id, cluster_info in clusters.items():
            # Get color for this cluster
            color = cluster_colors.get(cluster_id, (1.0, 1.0, 1.0))
            # Convert color from matplotlib format (0-1) to OpenCV format (0-255)
            cv_color = (int(color[0]*255), int(color[1]*255), int(color[2]*255))
            
            # Get points for this cluster
            points = cluster_info['points']
            
            # Only process if we have points
            if len(points) > 0:
                # Project points to 2D
                projected_points = opencv_integration.project_3d_to_2d(
                    points,
                    camera_position=args.camera_position,
                    camera_target=args.camera_target,
                    focal_length=args.focal_length,
                    field_of_view=args.field_of_view,
                    image_size=(original_frame.shape[1], original_frame.shape[0])
                )
                
                # Draw each point
                for x, y in projected_points:
                    # Convert to integer coordinates
                    px, py = int(round(x)), int(round(y))
                    
                    # Skip if out of bounds
                    if px < 0 or px >= original_frame.shape[1] or py < 0 or py >= original_frame.shape[0]:
                        continue
                    
                    # Draw point
                    cv2.circle(overlay, (px, py), 2, cv_color, -1)
        
        return overlay
    except Exception as e:
        logger.error(f"Error creating cluster overlay: {str(e)}")
        return None

if __name__ == "__main__":
   main()