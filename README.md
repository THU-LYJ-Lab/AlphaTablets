![title](https://github.com/user-attachments/assets/ac7675cb-3a2e-4e22-8316-da7c420ba69e)

# AlphaTablets

<p align="center">
    📃 <a href="https://arxiv.org/abs/2411.19950" target="_blank">Paper</a> • 🌐 <a href="https://hyzcluster.github.io/alphatablets" target="_blank">Project Page</a>
</p>

> **AlphaTablets: A Generic Plane Representation for 3D Planar Reconstruction from Monocular Videos**
>
> [Yuze He](https://hyzcluster.github.io), [Wang Zhao](https://github.com/thuzhaowang), [Shaohui Liu](http://b1ueber2y.me/), [Yubin Hu](https://github.com/AlbertHuyb), [Yushi Bai](https://bys0318.github.io/), Yu-Hui Wen, Yong-Jin Liu
>
> NeurIPS 2024

**AlphaTablets** is a novel and generic representation of 3D planes that features continuous 3D surface and precise boundary delineation. By representing 3D planes as rectangles with alpha channels, AlphaTablets combine the advantages of current 2D and 3D plane representations, enabling accurate, consistent and flexible modeling of 3D planes.

We propose a novel bottom-up pipeline for 3D planar reconstruction from monocular videos. Starting with 2D superpixels and geometric cues from pre-trained models, we initialize 3D planes as AlphaTablets and optimize them via differentiable rendering. An effective merging scheme is introduced to facilitate the growth and refinement of AlphaTablets. Through iterative optimization and merging, we reconstruct complete and accurate 3D planes with solid surfaces and clear boundaries.

<img src="https://hyzcluster.github.io/alphatablets/static/images/pipeline.png">



## Quick Start

### 1. Clone the Repository

Make sure to clone the repository along with its submodules:

```bash
git clone --recursive https://github.com/THU-LYJ-Lab/AlphaTablets
```

### 2. Install Dependencies

Set up a Python environment and install the required packages:

```bash
conda create -n alphatablets python=3.9
conda activate alphatablets

# Install PyTorch based on your machine configuration
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt
```

### 3. Download Pretrained Weights

#### Monocular Normal Estimation Weights

Download **Omnidata** pretrained weights:

- File: `omnidata_dpt_normal_v2.ckpt`
- Link: [Download Here](https://www.dropbox.com/scl/fo/348s01x0trt0yxb934cwe/h?rlkey=a96g2incso7g53evzamzo0j0y&e=2&dl=0)

Place the file in the directory:

```plaintext
./recon/third_party/omnidata/omnidata_tools/torch/pretrained_models
```

#### Depth Estimation Weights

Download **Metric3D** pretrained weights:

- File: `metric_depth_vit_giant2_800k.pth`
- Link: [Download Here](https://huggingface.co/JUGGHM/Metric3D/blob/main/metric_depth_vit_giant2_800k.pth)

Place the file in the directory:

```plaintext
./recon/third_party/metric3d/weight
```

### 4. Running Demos

#### ScanNet Demo

1. Download the `scene0684_01` demo scene from [here](https://drive.google.com/drive/folders/13rYkek_CQuOk_N5erJL08R26B1BkYmwD?usp=sharing) and extract it to `./data/`.
2. Run the demo with the following command:

```bash
python run.py --job scene0684_01
```

#### Replica Demo

1. Download the `office0` demo scene from [here](https://drive.google.com/drive/folders/13rYkek_CQuOk_N5erJL08R26B1BkYmwD?usp=sharing) and extract it to `./data/`.
2. Run the demo using the specified configuration:

```bash
python run.py --config configs/replica.yaml --job office0
```

### Tips

- **Out-of-Memory (OOM):** Reduce `batch_size` if you encounter memory issues.
- **Low Frame Rate Sequences:** Increase `weight_decay`, or set it to `-1` for an automatic decay. The default value is `0.9` (works well for ScanNet and Replica), but it can go up to larger values (no more than `1.0`).
- **Scene Scaling Issues:** If the scene scale differs significantly from real-world dimensions, adjust merging parameters such as `dist_thres` (maximum allowable distance for tablet merging).



## Evaluation on the ScanNet v2 dataset

1. **Download and Extract ScanNet**: 
   Follow the instructions provided on the [ScanNet website](http://www.scan-net.org/) to download and extract the dataset.

2. **Prepare the Data**: 
   Use the data preparation script to parse the raw ScanNet data into a processed pickle format and generate ground truth planes using code modified from [PlaneRCNN](https://github.com/NVlabs/planercnn/blob/master/data_prep/parse.py) and [PlanarRecon](https://github.com/neu-vi/PlanarRecon/tree/main).

   Run the following command under the PlanarRecon environment:

   ```bash
   python tools/generate_gt.py --data_path PATH_TO_SCANNET --save_name planes_9/ --window_size 9 --n_proc 2 --n_gpu 1
   python tools/prepare_inst_gt_txt.py --val_list PATH_TO_SCANNET/scannetv2_val.txt --plane_mesh_path ./planes_9
   ```

3. **Process Scenes in the Validation Set**: 
    You can use the following command to process each scene in the validation set. Update `scene????_??` with the specific scene name. Train/val/test split information is available [here](https://github.com/ScanNet/ScanNet/tree/master/Tasks/Benchmark):

   ```bash
   python run.py --job scene????_?? --input_dir PATH_TO_SCANNET/scene????_??
   ```

4. **Run the Test Script**: 
   Finally, execute the test script to evaluate the processed data:

   ```bash
   python test.py
   ```



## Citation

If you find our work useful, please kindly cite:

```
@article{he2024alphatablets,
  title={AlphaTablets: A Generic Plane Representation for 3D Planar Reconstruction from Monocular Videos}, 
  author={Yuze He and Wang Zhao and Shaohui Liu and Yubin Hu and Yushi Bai and Yu-Hui Wen and Yong-Jin Liu},
  journal={arXiv preprint arXiv:2411.19950},
  year={2024}
}
```



## Acknowledgements

Some of the test code and installation guide in this repo is borrowed from [NeuralRecon](https://github.com/zju3dv/NeuralRecon), [PlanarRecon](https://github.com/neu-vi/PlanarRecon/tree/main) and [ParticleSfM](https://github.com/bytedance/particle-sfm)! We sincerely thank them all.
