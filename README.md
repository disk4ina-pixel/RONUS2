# RO-NOUS Point Cloud Clustering System

A sophisticated 3D point cloud clustering system with integrated pose estimation, temporal consistency tracking, and multi-modal sensor fusion for human subject analysis.

## Overview

This system processes sequential 3D point cloud data (Center of Pressure - CoP) to identify, track, and reconstruct human subjects and rigid objects in a scene. It combines multiple clustering algorithms, pose estimation frameworks, and temporal filtering to produce robust 3D reconstructions.

## Key Features

### Core Capabilities
- **Multi-method clustering**: DBSCAN, HDBSCAN, and 3D occupancy grid-based clustering
- **Temporal consistency**: Maintains cluster identity across frames with historical locking
- **Pose estimation integration**: MMPose (2D/3D) and MediaPipe Pose (MP33) support
- **YOLO segmentation**: Object class labeling and rigid object primitive fitting
- **Voxel-based reconstruction**: Detailed voxel metadata with pattern information for surface reconstruction
- **Camera parameter optimization**: Iterative refinement using contour matching
- **Capsule-guided cluster fill**: Densifies thin person shells into complete body representations

### Advanced Features
- **Tunnel widening**: Expands acceptance zones to capture more body points during movement
- **Turn-tunnel expand-only override**: Prevents tunnel collapse during turning motions
- **Temporal height anchor**: Converges person stature to tallest clean capture
- **Rigid object primitives**: Gravity-aligned oriented bounding boxes for furniture (chairs, tables)
- **Fill-to-CoP 3D proximity guard**: Prevents stray voxel injection from distant floor/shadow points
- **Feature-A capsule fill**: Optional densification using mannequin capsule guidance
- **Body yaw temporal lock**: Velocity-clamped facing direction with unreliable-frame rejection
- **Control frame override**: Ground-truth keypoint injection for calibration frames

## Installation

### Prerequisites
- Python 3.8+
- NumPy
- SciPy
- OpenCV
- trimesh (optional, for rigid object fitting)
- torch
- joblib

### Optional Dependencies
- **MMPose**: For 2D/3D pose estimation
- **MediaPipe**: For MP33 pose detection
- **Ultralytics YOLOv8**: For instance segmentation and object classification
- **Trimesh**: For oriented bounding box fitting

```bash
pip install numpy scipy opencv-python torch joblib
pip install mmpose mmcv  # Optional
pip install mediapipe  # Optional
pip install ultralytics  # Optional
pip install trimesh  # Optional
```

## Usage

### Basic Usage

```bash
python run_clustering.py --path "outputs/flesh/dummy_flesh_cop_frame_*.txt" --frames_dir "data/clip_frames"
```

### Common Command-Line Options

#### Input/Output
- `--path`: Path to point cloud file(s) or glob pattern
- `--frames_dir`: Directory containing original video frames
- `--output`: Output folder for results (default: `clustering_output`)
- `--log_file`: Optional log file path
- `--log_level`: DEBUG, INFO, WARNING, ERROR

#### Clustering Methods
- `--use_dbscan`: Use DBSCAN instead of HDBSCAN
- `--use_grid`: Use 3D occupancy grid clustering
- `--compare_methods`: Compare different clustering methods
- `--refine_boundaries`: Refine existing cluster boundaries

#### Clustering Parameters
- `--eps`: DBSCAN epsilon (default: 2.7)
- `--min_cluster_size`: HDBSCAN minimum cluster size (default: 50)
- `--min_samples`: Minimum samples for clustering (default: 20)
- `--cluster_selection_epsilon`: HDBSCAN cluster selection epsilon (default: 5.0)
- `--grid_resolution`: Grid cell size in cm (default: 1.0)
- `--connectivity`: Grid connectivity (6, 18, or 26; default: 26)

#### Camera Parameters
- `--camera_position`: Camera position as "x,y,z" (default: "47,28,-20.0")
- `--camera_target`: Look-at target as "x,y,z" (default: "-25.1,123.8,-28.3")
- `--focal_length`: Focal length in mm (default: 27.5)
- `--field_of_view`: FOV in degrees (default: 66)
- `--image_size`: Image dimensions as "width,height" (default: "480,864")
- `--optimize_camera`: Run automatic camera parameter optimization

#### Temporal Processing
- `--use_temporal`: Enable temporal clustering with 5-frame buffer
- `--maintain_consistent_ids`: Maintain cluster IDs across frames (default: True)
- `--overlap_threshold`: IoU threshold for ID consistency (default: 0.5)

#### Pose Estimation
- `--use_mmpose`: Enable MMPose 2D/3D pose estimation
- `--use_intelligent_completion`: Enable intelligent pose completion

#### Visualization & Export
- `--visualize`: Generate visualization frames (default: True)
- `--create_video`: Create MP4 video from output frames
- `--create_overlay_video`: Create video with point cloud overlay on original frames
- `--fps`: Video frame rate (default: 10)
- `--export_clusters`: Export individual cluster files (default: True)

### Example Commands

#### Process with Grid Clustering and Temporal Tracking
```bash
python run_clustering.py \
    --path "data/cop_frame_*.txt" \
    --frames_dir "data/frames" \
    --use_grid \
    --grid_resolution 1.0 \
    --use_temporal \
    --visualize \
    --create_video
```

#### Process with Pose Estimation and YOLO Labels
```bash
python run_clustering.py \
    --path "data/cop_frame_*.txt" \
    --frames_dir "data/frames" \
    --use_mmpose \
    --use_intelligent_completion \
    --output "results_with_pose"
```

#### Optimize Camera Parameters
```bash
python run_clustering.py \
    --path "data/cop_frame_001.txt" \
    --optimize_camera \
    --camera_position "47,28,-20.0" \
    --camera_target "-25.1,123.8,-28.3"
```

## Input Format

### Point Cloud Files
Text files with one 3D point per line (X, Y, Z coordinates in cm):
```
-45.2 120.5 -15.3
-44.8 121.0 -15.1
...
```

File naming convention: `dummy_flesh_cop_frame_XXX.txt` where XXX is the frame number.

### Frame Images (Optional)
Video frames for visualization and 2D pose estimation should be placed in the `--frames_dir` directory.

## Output Structure

The output directory contains:

```
clustering_output/
├── frame_XXX_results.json      # Per-frame clustering results
├── frame_XXX_visualization.png # Visualization frames
├── clusters/                    # Individual cluster exports
│   ├── cluster_XXX_points.npy
│   └── ...
├── output_video.mp4            # Compiled video (if --create_video)
└── overlay_video.mp4           # Overlay video (if --create_overlay_video)
```

### Results JSON Format
```json
{
  "frame_number": 42,
  "clusters": {
    "person_uuid_1": {
      "points_count": 1250,
      "bbox_3d": {...},
      "voxel_data": [...],
      "yolo_class_label": "person",
      "skeleton": {...},
      "body_yaw_deg": 15.3
    }
  },
  "rigid_primitives": {
    "chair_uuid_1": {
      "obb_center": [...],
      "obb_dims": [...],
      "obb_orientation": [...]
    }
  }
}
```

## Architecture

### Main Components

1. **run_clustering.py**: Main processing script
   - Point cloud loading and preprocessing
   - Clustering algorithm execution
   - Temporal consistency management
   - Voxel grid generation
   - Result export and visualization

2. **ro_nous_clustering package** (imported module):
   - `point_cloud`: Point cloud I/O and manipulation
   - `clustering`: DBSCAN/HDBSCAN implementations
   - `grid`: 3D occupancy grid operations
   - `visualization`: 3D rendering and video generation
   - `utils`: Utility functions and logging
   - `opencv_integration`: Camera projection utilities

3. **Integration Modules**:
   - `mmpose_integration`: MMPose 2D/3D pose estimation
   - `temporal_consistency`: Frame buffering and state management
   - `pose_completion_integration`: Intelligent pose completion
   - `image_alignment`: Camera parameter optimization
   - `cluster_coordinates`: Cluster coordinate systems
   - `pattern_classifier`: Voxel pattern classification

### Processing Pipeline

1. **Load Frame**: Read point cloud from file
2. **Preprocess**: Filter noise, apply historical locks
3. **Cluster**: Run selected clustering algorithm
4. **Refine**: Boundary refinement, outlier removal
5. **Label**: Apply YOLO class labels (if available)
6. **Fit Poses**: MMPose/MP33 skeleton fitting
7. **Reconstruct**: Generate voxel metadata with patterns
8. **Export**: Save results and generate visualizations

## Advanced Configuration

### Feature Flags (Runtime Toggles)

Several features can be enabled/disabled via code flags:

- `_FEATURE_A_CAPSULE_FILL=False`: Capsule-guided cluster densification
- `state_bank.rigid_claim_enabled=False`: Rigid object voxel claiming
- `_CONTROL_KP_WORLD`: Ground-truth keypoint overrides for specific frames

### Log Tags

The system uses structured log tags for debugging:
- `[TUNNEL]`: Tunnel/acceptance zone modifications
- `[MP33-WRIST]`: MediaPipe wrist repair operations
- `[RIGID-FIT]`: Rigid object primitive fitting
- `[CAPSULE-FILL]`: Capsule fill operations
- `[WARN]NARROW`: Acceptance zone warnings
- `[OK]` / `[FAIL]`: Operation status

## Known Issues & Limitations

1. **Partial MiDaS Shell Dropout**: On frames where the Center of Pressure lacks head points, the reconstructed shell may be shorter than actual stature. The temporal height anchor mitigates this by converging to the tallest clean capture.

2. **Person-on-Chair Contact**: Touching surfaces between person and rigid objects are not geometrically separable without YOLO-instance/per-voxel temporal layer.

3. **Background Contamination**: Approximately 6 frames may show background contamination when tunnel Z expansion approaches wall projection limits.

4. **Inventory Lag**: During rapid turns, the inventory bbox may lag behind current pose, addressed by union-only override.

## Changelog

### 2026 Updates

- **Jun 25**: Tunnel widening (+16cm X span, +40cm Y depth, +20cm Z height); Turn-tunnel expand-only override
- **Jun 24**: Temporal height anchor with running maximum convergence
- **Jun 23**: Feature-A capsule fill toggle; FEATURE-A regression fix for cf_ prefix parsing
- **Jun 23**: Capsule-guided cluster fill (2nd pass, in-frame)
- **Jun 23**: Density-robust _body_depth measurement from raw CoP points
- **Jun 19**: Mojibake purge (ASCII normalization)
- **Jun 17**: Gravity-aligned rigid primitives; Voxel data preservation fixes
- **Jun 16**: Fill-to-CoP 3D proximity guard
- **Jun 14**: Control frame ground-truth override
- **Jun 13**: Rigid object primitives; YOLO inventory persistence; Anti-merge for rigid objects
- **Jun 13**: Body yaw temporal lock with velocity clamping
- **Jun 07**: MP33 wrist repair (silhouette-based)
- **Jun 06**: MP33 wrist/arm repair at source
- **Jun 02**: JSON readable layout (indented dicts)

### 2025 Updates

- **Apr 24**: Initial creation

## Author

Chaim

## License

[Specify license if applicable]

## Contributing

[Specify contribution guidelines if applicable]
