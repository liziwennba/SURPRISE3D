---

<div align='center'>

<h2><a href="https://arxiv.org/abs/your_arxiv_id">Surprise3D: A Dataset for Spatial Understanding and Reasoning in Complex 3D Scenes</a></h2>

[Ziwen Li](https://github.com/liziwennba/)<sup>1*</sup>, [Hanlve Zhang](https://scholar.google.com/citations?user=example)<sup>1*</sup>, [Jiaxin Huang](https://scholar.google.com/citations?user=example)<sup>1</sup>, [Yu-Shen Liu](https://yushen-liu.github.io/)<sup>2</sup>, [Tiejun Huang](https://scholar.google.com/citations?user=knvEK4AAAAAJ&hl=en)<sup>1,3</sup>, [Xinlong Wang](https://www.xloong.wang/)<sup>1</sup>
 
<sup>1</sup>[BAAI](https://www.baai.ac.cn/english.html), <sup>2</sup>[THU](https://www.tsinghua.edu.cn/en/), <sup>3</sup>[PKU](https://english.pku.edu.cn/) <br><sup>*</sup> Equal Contribution
 
CVPR 2025 (Highlight)

[[Project Page](https://img.shields.io/badge/Project-Page-green)](https://mbzuai-liziwen.github.io/Surprise3D/)
[[arXiv](https://img.shields.io/badge/arXiv-Paper-red)](https://arxiv.org/abs/your_arxiv_id)
[[PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/surprise3d-a-dataset-for-spatial/3d-visual-grounding)](https://paperswithcode.com/sota/3d-visual-grounding?p=surprise3d-a-dataset-for-spatial)

</div>

<p align="center">
    <img src="assets/task.png" alt="overview" width="800" />
</p>

We introduce **Surprise3D**, a novel dataset designed to evaluate **language-guided spatial reasoning segmentation** in complex 3D scenes. Unlike existing datasets that often mix semantic cues (e.g., object names) with spatial context, **Surprise3D** emphasizes **spatial reasoning** by crafting queries that exclude object names, thus mitigating shortcut biases.

The dataset includes:
- **200k+ vision-language pairs** across **900+ indoor scenes** from **ScanNet++**.
- **89k+ human-annotated spatial queries** that focus on spatial relationships without object names.
- **2.8k unique object classes**, providing rich diversity for spatial reasoning tasks.

Surprise3D covers a wide range of **spatial reasoning skills**, including:
- **Relative position** (e.g., "Find the object behind the chair."),
- **Narrative perspective** (e.g., "Locate the object visible from the sofa."),
- **Parametric perspective** (e.g., "Select the object 2 meters to the left of the table."),
- **Absolute distance reasoning** (e.g., "Identify the object exactly 3 meters in front of you.").

We modified the code obtained from [Intent3D](https://github.com/WeitaiKang/Intent3D) and [Reason3D](https://github.com/KuanchihHuang/Reason3D) to support running on our proposed dataset, we thank their dedicated efforts in their impressives works. You may find modified versions [here](./Models). 
