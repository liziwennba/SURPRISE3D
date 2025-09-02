
<div align='center'>

<h2><a href="https://arxiv.org/abs/2507.07781">Surprise3D: A Dataset for Spatial Understanding and Reasoning in Complex 3D Scenes</a></h2>

Jiaxin Huang, Ziwen Li, Hanlue Zhang, Runnan Chen, Zhengqing Gao, Xiao He, Yandong Guo, Wenping Wang, Tongliang Liu, Mingming Gong
 
MBZUAI, AI2Robotics

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

---
## 🔍 Data Analysis

<p align="center">
    <img src="assets/data_analysis.png" alt="Data Analysis" width="800" />
</p>

We provide a detailed analysis of the dataset:
1. **Augmentation for Low-Frequency Objects**: Boosting the number of questions targeting rarely occurring objects to improve model robustness.
2. **Object Frequency (%) by Question Type (Top 15 Objects)**: Examining how frequently the top 15 objects are referenced across different question types.
3. **Distribution of Question Types**: Visualizing the proportion of questions across various reasoning categories.

Our dataset ensures a balanced distribution of reasoning types and incorporates augmentation techniques to reduce biases caused by object frequency disparities. This analysis supports the development of models that generalize better across diverse reasoning tasks.

---

## 📥 Download Annotations

You can download the **Surprise3D annotation data** from Hugging Face using the following link:

[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Dataset-blue)](https://huggingface.co/datasets/hhllzz/surprise-3d)

---

## ⚙️ Train and Evaluation

We have modified parts of the **Reason3D** codebase to support training and testing on our **Surprise3D** dataset. These modifications enable the preprocessing of **ScanNet++** data and the use of **Reason3D** for segmentation tasks on **Surprise3D**.

   - Please refer to **[here](./Models/reason3d)** for scripts to preprocess the **ScanNet++** data required for the **Surprise3D** dataset and training and evaluation for **Reason3D**.

These updates allow us to leverage the powerful capabilities of **Reason3D** while ensuring compatibility with the unique structure and annotations of **Surprise3D**.

---

We thank the authors of **[Reason3D](https://github.com/KuanchihHuang/Reason3D)** for their outstanding work, which served as the foundation for our modifications.
