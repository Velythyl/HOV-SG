# HOV-SG
[![Static Badge](https://img.shields.io/badge/-arXiv-B31B1B?logo=arxiv)](https://arxiv.org/abs/2403.17846)
[![Static Badge](https://img.shields.io/badge/Project-Page-a)](https://hovsg.github.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Static Badge](https://img.shields.io/badge/-Video-FF0000?logo=youtube)](https://hovsg.github.io/static/images/hovsg_rss_final.mp4)



This repository is the official implementation of the paper:

> **Hierarchical Open-Vocabulary 3D Scene Graphs for Language-Grounded Robot Navigation**
>
> [Abdelrhman Werby]()&ast;, [Chenguang Huang](http://www2.informatik.uni-freiburg.de/~huang/)&ast;, [Martin Büchner](https://rl.uni-freiburg.de/people/buechner)&ast;, [Abhinav Valada](https://rl.uni-freiburg.de/people/valada), and [Wolfram Burgard](https://www.utn.de/person/wolfram-burgard/). <br>
> &ast;Equal contribution. <br> 
> 
> *arXiv preprint arXiv:2403.17846*, 2024 <br>
> (Accepted for *Robotics: Science and Systems (RSS), Delft, Netherlands*, 2024.)

<p align="center">
  <img src="media/teaser-hovsg-white.png" alt="HOV-SG allows the construction of accurate, open-vocabulary 3D scene graphs for large-scale and multi-story environments and enables robots to effectively navigate in them with language instructions." width="600" />
</p>

## RAGMAP adapter (fork addition)

This fork ([Velythyl/HOV-SG](https://github.com/Velythyl/HOV-SG)) packages the
**original, unmodified HOV-SG graph construction** as a container that RAGMAP
runs as an `object_mapping` stage. Only adapters were added; nothing under
`hovsg/` or `application/` was changed:

| Path | What it is |
|---|---|
| `ragmap_adapter/dataset.py` | HOV-SG dataset loader (`RGBDDataset` subclass) for the RAGMAP input contract, including the up-axis mapping |
| `ragmap_adapter/run.py` | `ragmap-run` entry point: runs the steps of `application/create_graph.py` (Hydra-composed config), CLIP room naming, export |
| `ragmap_adapter/export.py` | Writes `objects.jsonl` etc. in the input world frame |
| `ragmap_adapter/weights.py` | Download-on-first-run of the checkpoints into the mounted weights dir |
| `ragmap_adapter/fast_merge.py`, `ragmap_adapter/exact_dbscan.py` | **Exact** replacement of the sequential merge's two hot spots (per-mask DBSCAN denoise, brute-force faiss overlap); the merged masks are byte-identical to upstream's. On by default (`ragmap.fast_merge`); Replica office0 at stride 10 goes from ~11 h to ~1.5 h. `ragmap.fast_merge=false` runs upstream's own functions |
| `ragmap_adapter/cpu_shim.py` | Redirects upstream's hard-coded `.cuda()` to CPU **only when no GPU is visible** (CI smoke) |
| `config/ragmap.yaml` | `create_graph.yaml` + a `ragmap:` section; every upstream key keeps its upstream default |
| `Dockerfile`, `docker/requirements.txt` | CUDA 12.1 / PyTorch 2.3.1 / Python 3.9 image, **without habitat-sim** (only needed to re-render HM3DSem walks) |
| `.github/workflows/ghcr.yml` | Builds and publishes `ghcr.io/velythyl/hovsg`, then runs a CPU smoke test of the published image |
| `smoke/` | Synthetic scene generator, output checker, `run_smoke.sh` |

### License

HOV-SG is released by its authors under the MIT license **for academic usage**;
for any commercial purpose contact the authors (see [License](#-license)).

### Running it

```bash
podman run --rm --device nvidia.com/gpu=all \
  -v /path/to/scene:/input:ro,Z \
  -v /path/to/out:/output:Z \
  -v $HOME/.cache/hovsg-weights:/weights:Z \
  ghcr.io/velythyl/hovsg:latest \
  --input /input --output /output [hydra overrides, e.g. pipeline.skip_frames=5]
```

**Weights** are not baked into the image. Mount a persistent directory at
`/weights` (or point `HOVSG_WEIGHTS` elsewhere). On first use the runner
downloads into it, atomically and under a file lock so concurrent containers
can share it:

* `laion2b_s32b_b79k.bin`: OpenCLIP ViT-H-14 (laion2B-s32B-b79K), 3.9 GB, from Hugging Face (`HF_ENDPOINT` is respected)
* `sam_vit_h_4b8939.pth`: SAM ViT-H, 2.4 GB (`sam_vit_b_01ec64.pth` / `sam_vit_l_0b3195.pth` if `models.sam.type` is changed)

Set `HOVSG_OFFLINE=1` to fail instead of downloading. Explicit paths can be given with
`models.clip.checkpoint=...` / `models.sam.checkpoint=...`.

### Input contract (`/input`)

* `meta.json`: `{"fx","fy","cx","cy","width","height","depth_scale","up_axis","frame"}`.
  `depth_scale` is depth-PNG units per metre, `up_axis` is `"z"`, `"y"` or `"-y"`.
* `frames.jsonl`: one `{"frame_index","observation_id","rgb","depth","pose"}` per line.
  `pose` is 16 floats, row-major 4x4 camera-to-world, OpenCV camera convention;
  `depth` is a uint16 PNG. Frames whose pose is non-finite are skipped and counted in `run.json`.

**Up axis.** HOV-SG hard-codes +Y as vertical. `Graph.segment_floors` builds its
height histogram on `points[:, 1]`, rooms are sliced on Y and projected onto X/Z,
and camera height is read from `pose[1, 3]`. That is the Habitat world convention;
upstream's HM3DSem loader only flips the camera from OpenGL to OpenCV and keeps the
world Y-up. The loader therefore left-multiplies every pose by a proper rotation `A`
that sends the declared up axis to +Y:
`z: (x,y,z)->(x,z,-y)`, `y: identity`, `-y: (x,y,z)->(x,-y,-z)`. Every exported
coordinate is mapped back with `A^T`. The upstream-format graph in
`/output/hovsg/` is left in HOV-SG's frame, and `run.json` records `A`.

### Output contract (`/output`)

* `objects.jsonl`: one object per line:
  `{"id","label","caption","crop","centroid","bbox_min","bbox_max","pointcloud","frame_indices","floor","room","extra"}`.
  * `id`: HOV-SG object id `<floor>_<room>_<n>`.
  * `label`: HOV-SG's CLIP label over `pipeline.obj_labels` (default `HM3DSEM_LABELS`).
  * `caption`: always `null` (HOV-SG does not caption).
  * `floor`: HOV-SG floor id.
  * `room`: HOV-SG room name from `Graph.generate_room_names(generate_method="view_embedding")`, which votes with CLIP over `ragmap.room_types`. The default list is upstream's `visualize_query_graph.py` list.
  * `pointcloud`: `objects/<id>.ply`, with observed colours restored. Upstream's `save_masked_pcds` paints each mask a random colour.
  * `frame_indices` and `crop` come from reprojecting the object points into every input frame with a depth-consistency test. The crop is taken from the frame with the most visible points.
  * `extra` holds `hovsg_room_id`, `num_points`, `best_frame_index`, `clip_feature_row` (a row of `object_clip_feats.npy`, HOV-SG's 1024-d object embedding) and `label_vocabulary`.
* `run.json`: status, the resolved config, counts (floors/rooms/objects), per-phase timings, frame counts, the up-axis matrix and the navigation-graph status. On failure it holds `status: failed` with the traceback, and the exit code is 1.
* `rooms.jsonl`, `floors.jsonl`, `object_clip_feats.npy`: extras. `hovsg/`: upstream's native output (`graph/`, `full_pcd.ply`, `masked_pcd.ply`, ...).

Adapter knobs live under `ragmap.*` in `config/ragmap.yaml`:
* `nav_graph`: runs upstream's Voronoi navigation graph. A failure there is recorded instead of aborting.
* `save_feature_map`: also writes upstream's `full_feats.pt`, which is N_points x 1024 floats and can be many GB.
* `visibility.*`: the `frame_indices` and crop thresholds.
* `room_types`, `seed`.

### Hosted models

The graph **build** path calls no hosted model. `Graph.__init__`,
`create_feature_map`, `build_graph` and `generate_room_names("view_embedding")`
use only local SAM and OpenCLIP. `hovsg/utils/llm_utils.py` imports `openai` at
module level, so the package is installed, but nothing in the build path calls
it. OpenAI is used only at **query time**, with the key read from `OPENAI_KEY`:
* `Graph.query_hierarchy` calls `parse_hier_query` (`gpt-3.5-turbo`) to split a query into floor, room and object. `application/visualize_query_graph.py` uses it.
* `Graph.query_floor(query_method="gpt")` calls `infer_floor_id_from_query` (`gpt-3.5-turbo-instruct`).
* `parse_floor_room_object_gpt35` (`gpt-3.5-turbo`).
* `Room.infer_room_type_from_objects(infer_method="label")`. The adapter does not use it, and its `llm.llm_utils` import is broken upstream.

### Smoke test

```bash
smoke/run_smoke.sh ghcr.io/velythyl/hovsg:latest podman                    # CPU profile
GPU_ARGS="--device nvidia.com/gpu=all" smoke/run_smoke.sh ghcr.io/velythyl/hovsg:latest podman
UP_AXIS=y smoke/run_smoke.sh ...                                           # other conventions
```

The script renders a ray-cast box room with four boxes, runs the image, and
checks the output against the contract. It fails if any object centroid falls
outside the room in the **input** frame, which catches a wrong up-axis mapping.
CI runs the CPU profile against every published image.

## 📰 Major Updates
- **[29 Aug 2024]** **We added `hm3dsem_walks` dataset generation and hierarchical scene graph evaluation code.** <br>
Please review the updated code structure and newly added dependencies for dataset construction. <br><br>
- [01 Jul 2024] Initial release of HOV-SG including mapping and graph construction engine.

## 🏗 Setup
1. Clone and set up the HOV-SG repository
```bash
git clone https://github.com/hovsg/HOV-SG.git
cd HOV-SG

# set up virtual environment and install habitat-sim afterwards separately to avoid errors.
conda env create -f environment.yaml
conda activate hovsg
conda install habitat-sim -c conda-forge -c aihabitat

# set up the HOV-SG python package
pip install -e .
```

### OpenCLIP
HOV-SG uses the Open CLIP model to extract features from RGB-D frames. To download the Open CLIP model checkpoint `CLIP-ViT-H-14-laion2B-s32B-b79K` please refer to [Open CLIP](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K).
```bash
mkdir checkpoints
wget https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/resolve/main/open_clip_pytorch_model.bin?download=true -O checkpoints/temp_open_clip_pytorch_model.bin && mv checkpoints/temp_open_clip_pytorch_model.bin checkpoints/laion2b_s32b_b79k.bin
```
Another option is to use the OVSeg fine-tuned Open CLIP model, which is available under [here](https://github.com/facebookresearch/ov-seg):
```bash
pip install gdown
gdown --fuzzy https://drive.google.com/file/d/17C9ACGcN7Rk4UT4pYD_7hn3ytTa3pFb5/view -O checkpoints/ovseg_clip.pth
```

### SAM
HOV-SG uses [SAM](https://github.com/facebookresearch/segment-anything) to generate class-agnostic masks for the RGB-D frames. To download the SAM model checkpoint `sam_v2` execute the following:
```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O checkpoints/sam_vit_h_4b8939.pth
```

## 🖼️ Dataset Preparation

### Habitat Matterport 3D Semantics
HOV-SG takes posed RGB-D sequences as input. In order to produce hierarchical multi-story scenes we make use of the Habitat 3D Semantics dataset ([HM3DSem](https://aihabitat.org/datasets/hm3d-semantics/)). 

- Download the [Habitat Matterport 3D Semantics](https://github.com/matterport/habitat-matterport-3dresearch) dataset. More specifically, download through the links corresponding to these filenames: [hm3d-val-habitat-v0.2.tar](https://api.matterport.com/resources/habitat/hm3d-val-habitat-v0.2.tar), [hm3d-val-semantic-annots-v0.2.tar](https://api.matterport.com/resources/habitat/hm3d-val-semantic-annots-v0.2.tar), [hm3d-val-semantic-configs-v0.2.tar](	https://api.matterport.com/resources/habitat/hm3d-val-semantic-configs-v0.2.tar).
    <details>
    <summary>Make sure that the raw HM3D dataset has the following structure:</summary>
    
    ```
    ├── hm3d
    │   ├── hm3d_annotated_basis.scene_dataset_config.json # this file is necessary
    │   ├── val
    │   │   └── 00824-Dd4bFSTQ8gi
    │   │         ├── Dd4bFSTQ8gi.basis.glb
    │   │         ├── Dd4bFSTQ8gi.basis.navmesh
    │   │         ├── Dd4bFSTQ8gi.glb
    │   │         ├── Dd4bFSTQ8gi.semantic.glb
    │   │         └── Dd4bFSTQ8gi.semantic.txt
            ...
        ...
    ...
    ```

    </details>
We used the following scenes from the Habitat Matterport 3D Semantics dataset in our evaluation:
<details>
  <summary>Show Scenes ID</summary>
  
  1. `00824-Dd4bFSTQ8gi`
  2. `00829-QaLdnwvtxbs`
  3. `00843-DYehNKdT76V`
  4. `00861-GLAQ4DNUx5U`
  5. `00862-LT9Jq6dN3Ea`
  6. `00873-bxsVRursffK`
  7. `00877-4ok3usBNeis`
  8. `00890-6s7QHgap2fW`

</details>

1. Our method requires posed input data. Because of that, we recorded trajectories for each sequence we evaluate on. We provide a script (`hovsg/data/hm3dsem/gen_hm3dsem_walks_from_poses.py`) that turns a set of camera poses (`hovsg/data/hm3dsem/metadata/poses`) into a sequence of RGB-D observations using the [habitat-sim](https://github.com/facebookresearch/habitat-sim) simulator. The output includes RGB, depth, poses and frame-wise semantic/panoptic ground truth:
```bash
  python data/habitat/gen_hm3dsem_from_poses.py --dataset_dir <hm3dsem_dir> --save_dir data/hm3dsem_walks/
```

2. Secondly, we construct a new hierarchical graph-structured dataset that is called `hm3dsem_walks` that includes ground truth based on all observations recorded. To produce this ground-truth data please execute the following: First, define the following config paths: `main.package_path`, `main.dataset_path`, `main.raw_data_path`, and `main.save_path` under `config/create_graph.yaml`. For each scene, define the `main.scene_id`, `main.split`. Next, execute the following to obtain floor-, region-, and object-level ground truth data per scene. We utilize every recorded frame without skipping (see parameter `dataset.hm3dsem.gt_skip_frames`) and recommend 128 GB of RAM to compile this as the scenes differ in size:
```bash
cd HOV-SG
python hovsg/data/hm3dsem/create_hm3dsem_walks_gt.py
```

To evaluate semantic segmentation cababilities, we used [ScanNet](http://www.scan-net.org/) and [Replica](https://github.com/facebookresearch/Replica-Dataset).
### ScanNet
To get an RGBD sequence for ScanNet, download the ScanNet dataset from the [official website](http://www.scan-net.org/). The dataset contains RGB-D frames compressed as .sens files. To extract the frames, use the [SensReader/python](https://github.com/ScanNet/ScanNet/blob/master/SensReader/python).
We used the following scenes from the ScanNet dataset:

<details>
  <summary>Show Scenes ID</summary>

  1. `scene0011_00`
  2. `scene0050_00`
  2. `scene0231_00`
  3. `scene0378_00`
  4. `scene0518_00`
</details>

### Replica
To get an RGBD sequence for Replica, Instead of the original Replica dataset, download the scanned RGB-D trajectories of the Replica dataset provided by [Nice-SLAM](https://github.com/cvg/nice-slam). It contains rendered trajectories using the mesh models provided by the original Replica datasets. 
Download the Replica RGB-D scan dataset using the downloading [script](https://github.com/cvg/nice-slam/blob/master/scripts/download_replica.sh) in [Nice-SLAM](https://github.com/cvg/nice-slam#replica-1).

```bash
wget https://cvg-data.inf.ethz.ch/nice-slam/data/Replica.zip -O data/Replica.zip && unzip data/Replica.zip -d data/Replica_RGBD && rm data/Replica.zip 
```

To evaluate against the ground truth semantics labels, you also need also to download the original Replica dataset from the [Replica](https://github.com/facebookresearch/Replica-Dataset) as it contains the ground truth semantics labels as .ply files.
```bash
git clone https://github.com/facebookresearch/Replica-Dataset.git data/Replica-Dataset
chmod +x data/Replica-Dataset/download.sh && data/Replica-Dataset/download.sh data/Replica_original
```
We only used the following scenes from the Replica dataset:
<details>
  <summary>Show Scenes ID</summary>
  
  1. `office0`
  2. `office1`
  3. `office2`
  4. `office3`
  5. `office4`
  6. `room0`
  7. `room1`
  8. `room2`

</details>

## 📂 Datasets file strutcre
The Data folder should have the following structure:

<details>
  <summary>Show data folder structure</summary>
  
```
├── hm3dsem_walks
│   ├── val
│   │   ├── 00824-Dd4bFSTQ8gi
│   │   │   ├── depth
│   │   │   │   ├── Dd4bFSTQ8gi-000000.png
│   │   │   │   ├── ...
│   │   │   ├── rgb
│   │   │   │   ├── Dd4bFSTQ8gi-000000.png
│   │   │   │   ├── ...
│   │   │   ├── semantic
│   │   │   │   ├── Dd4bFSTQ8gi-000000.png
│   │   │   │   ├── ...
│   │   │   ├── pose
│   │   │   │   ├── Dd4bFSTQ8gi-000000.png
│   │   │   │   ├── ...
|   |   ├── 00829-QaLdnwvtxbs
|   |   ├── ..
├── Replica
│   ├── office0
│   │   ├── results
│   │   │   ├── depth0000.png
│   │   │   ├── ...
│   │   |   ├── rgb0000.png
│   │   |   ├── ...
│   │   ├── traj.txt
│   ├── office1
│   ├── ...
├── ScanNet
│   ├── scans
│   │   ├── scene0011_00
│   │   │   ├── color
│   │   │   │   ├── 0.jpg
│   │   │   │   ├── ...
│   │   │   ├── depth
│   │   │   │   ├── 0.png
│   │   │   │   ├── ...
│   │   │   ├── poses
│   │   │   │   ├── 0.txt
│   │   │   │   ├── ...
│   │   │   ├── internsics
│   │   │   │   ├── intrinsics_color.txt
│   │   │   │   ├── intrinsics_depth.txt
│   │   ├── ..
```

</details>



## :rocket: Run 

### Create scene graphs (only for Habitat Matterport 3D Semantics):
```bash
python application/create_graph.py main.dataset=hm3dsem main.dataset_path=data/hm3dsem_walks/val/00824-Dd4bFSTQ8gi/ main.save_path=data/scene_graphs/00824-Dd4bFSTQ8gi
```
<details>
  <summary>This will generate a scene graph for the specified RGB-D sequence and save it. The following files are generated:</summary>

```
├── graph
│   ├── floors
│   │   ├── 0.json
│   │   ├── 0.ply
│   │   ├── 1.json
│   │   ├── ...
│   ├── rooms
│   │   ├── 0_0.json
│   │   ├── 0_0.ply
│   │   ├── 0_1.json
│   │   ├── ...
│   ├── objects
│   │   ├── 0_0_0.json
│   │   ├── 0_0_0.ply
│   │   ├── 0_0_1.json
│   │   ├── ...
│   ├── nav_graph
├── tmp
├── full_feats.pt
├── mask_feats.pt
├── full_pcd.ply
├── masked_pcd.ply
```
The `graph` folder contains the generated scene graph hierarchy, the first number in the file name represents the floor number, the second number represents the room number, and the third number represents the object number. The `tmp` folder holds intermediate results obtained throughout graph construction. The `full_feats.pt` and `mask_feats.pt` contain the features extracted from the RGBD frames using the Open CLIP and SAM models. the former contains per point features and the latter contains the features for the object masks. The `full_pcd.ply` and `masked_pcd.ply` contain the point cloud representation of the RGB-D frames and the instance masks of all objects, respectively.

</details>

### Visualize scene graph
```bash
python application/visualize_graph.py graph_path=data/scene_graphs/hm3dsem/00824-Dd4bFSTQ8gi/graph
```
![hovsg_graph_vis](media/hovsg_graph_vis.gif)

### Interactive visualization of scene graphs and natural language queries

#### Setup OpenAI
In order to test graph queries with HOV-SG, you need to setup an OpenAI API account with the following steps:
1. [Sign up an OpenAI account](https://openai.com/blog/openai-api), login your account, and bind your account with at least one payment method.
2. [Get you OpenAI API keys](https://platform.openai.com/account/api-keys), copy it.
3. Open your `~/.bashrc` file, paste a new line `export OPENAI_KEY=<your copied key>`, save the file, and source it with command `source ~/.bashrc`. Another way would be to run `export OPENAI_KEY=<your copied key>` in the teminal where you want to run the query code.

#### Evaluate query against pre-built hierarchical scene graph 
```bash
python application/visualize_query_graph.py main.graph_path=data/scene_graphs/hm3dsem/00824-Dd4bFSTQ8gi/graph
```
After launching the code, you will be asked to input the hierarchical query. An example is `chair in the living room on floor 0`. You can see the visualization of the top 5 target objects and the room it lies in.
![hovsg_graph_query](media/hovsg_graph_query.gif)

### Extract feature map for semantic segmentation (only ScanNet and Replica)
```bash
python application/semantic_segmentation.py main.dataset=replica main.dataset_path=Replica/office0 main.save_path=data/sem_seg/office0
```

### Evaluate semantic segmentation (only ScanNet and Replica)
```bash
python application/eval/evaluate_sem_seg.py dataset=replica scene_name=office0 feature_map_path=data/sem_seg/office0
```

### Evaluate predicted scene graphs (only Habitat 3D Semantics)
- Define the scene identifiers and paths of ground truth and the predicted scene graph in the `config/eval_graph.yaml`.
- Run the graph evaluation method:
```bash
python application/eval/evaluate_graph.py 
```

## 📔 Abstract

Recent open-vocabulary robot mapping methods enrich dense geometric maps with pre-trained visual-language features. While these maps allow for the prediction of point-wise saliency maps when queried for a certain language concept, largescale environments and abstract queries beyond the object level still pose a considerable hurdle, ultimately limiting languagegrounded robotic navigation. In this work, we present HOVSG, a hierarchical open-vocabulary 3D scene graph mapping approach for language-grounded indoor robot navigation. Leveraging open-vocabulary vision foundation models, we first obtain state-of-the-art open-vocabulary segment-level maps in 3D and subsequently construct a 3D scene graph hierarchy consisting of floor, room, and object concepts, each enriched with openvocabulary features. Our approach is able to represent multistory buildings and allows robotic traversal of those using a cross-floor Voronoi graph. HOV-SG is evaluated on three distinct datasets and surpasses previous baselines in open-vocabulary semantic accuracy on the object, room, and floor level while producing a 75% reduction in representation size compared to dense open-vocabulary maps. In order to prove the efficacy and generalization capabilities of HOV-SG, we showcase successful long-horizon language-conditioned robot navigation within realworld multi-story environments. 

If you find our work useful, please consider citing our paper:
```
@article{werby23hovsg,
Author = {Abdelrhman Werby and Chenguang Huang and Martin Büchner and Abhinav Valada and Wolfram Burgard},
Title = {Hierarchical Open-Vocabulary 3D Scene Graphs for Language-Grounded Robot Navigation},
Year = {2024},
journal = {Robotics: Science and Systems},
} 
```

## 👩‍⚖️  License

For academic usage, the code is released under the [MIT](https://opensource.org/licenses/MIT) license.
For any commercial purpose, please contact the authors.


## 🙏 Acknowledgment

This work was funded by the German Research Foundation
(DFG) Emmy Noether Program grant number 468878300, the
BrainLinks-BrainTools Center of the University of Freiburg,
and an academic grant from NVIDIA.
