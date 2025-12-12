
<div align="center">

# **SpaceDrive**: Infusing Spatial Awareness into VLM-based Autonomous Driving

[Peizheng Li](https://edwardleelpz.github.io/)<sup>* 1,2</sup>, [Zhenghao Zhang](https://zhenghao2519.github.io/)<sup>* 1,4</sup>, [David Holtz](https://scholar.google.com/citations?user=gf09DbwAAAAJ&hl=en&oi=ao)<sup>1</sup>, [Hang Yu](https://scholar.google.com/citations?view_op=search_authors&mauthors=Hang+Yu+mercedes&hl=en&oi=ao)<sup>1,5</sup>, [Yutong Yang](https://scholar.google.com/citations?hl=en&user=kg9OvU0AAAAJ)<sup>1,6</sup>, [Yuzhi Lai](https://scholar.google.com/citations?user=9Z6Gjo4AAAAJ&hl=en)<sup>2</sup>, [Rui Song](https://rruisong.github.io/)<sup>7</sup>, [Andreas Geiger](https://www.cvlibs.net/)<sup>2,3</sup>, [Andreas Zell](https://uni-tuebingen.de/fakultaeten/mathematisch-naturwissenschaftliche-fakultaet/fachbereiche/informatik/lehrstuehle/kognitive-systeme/the-chair/staff/prof-dr-andreas-zell/) <sup>2</sup>

<sup>1</sup> Mercedes-Benz AG, <sup>2</sup> University of Tübingen, <sup>3</sup> Tübingen AI Center, <sup>4</sup> TU Munich, <sup>5</sup> Karlsruhe Institute of Technology, <sup>6</sup> University of Stuttgart, <sup>7</sup> UCLA

(*) Equal contribution


[![arXiv](https://img.shields.io/badge/arXiv-2504.10117-red?logo=arXiv&logoColor=red)](https://arxiv.org/abs/2512.10719)
[![Project_Page](https://img.shields.io/badge/Project_Page-SpaceDrive-blue?logo=pagekit&logoColor=blue)](https://zhenghao2519.github.io/SpaceDrive_Page/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

</div>

##

End-to-end autonomous driving methods built on vision language models (VLMs) have undergone rapid development driven by their universal visual understanding and strong reasoning capabilities obtained from the large-scale pretraining. However, we find that current VLMs struggle to understand fine-grained 3D spatial relationships which is a fundamental requirement for systems interacting with the physical world.To address this issue, we propose SpaceDrive, a spatial-aware VLM-based driving framework that treats spatial information as explicit positional encodings (PEs) instead of textual digit tokens, enabling joint reasoning over semantic and spatial representations. SpaceDrive employs a universal positional encoder to all 3D coordinates derived from multi-view depth estimation, historical ego-states, and text prompts. These 3D PEs are first superimposed to augment the corresponding 2D visual tokens. Meanwhile, they serve as a task-agnostic coordinate representation, replacing the digit-wise numerical tokens as both inputs and outputs for the VLM. This mechanism enables the model to better index specific visual semantics in spatial reasoning and directly regress trajectory coordinates rather than generating digit-by-digit, thereby enhancing planning accuracy. Extensive experiments validate that SpaceDrive achieves state-of-the-art open-loop performance on the nuScenes dataset and the second-best Driving Score of 78.02 on the Bench2Drive closed-loop benchmark over existing VLM-based methods.

![](./assets/architecture.png)

## 📰 News

- [2025/12/11] Paper is released on [arXiv](https://arxiv.org/abs/2512.10719).

## ⌨️ Code
The code is currently under the internal review process and will be released soon.

## 🎥 Visualizations

<table width="100%">
  <tr>
    <td width="33%">
      <video src="https://github.com/user-attachments/assets/71630ab6-53b3-49d8-93a6-6feaa206bc52" controls width="100%"></video>
    </td>
    <td width="33%">
      <video src="https://github.com/user-attachments/assets/a6152c57-49c6-4197-b7f2-976934353e4d" controls width="100%"></video>
    </td>
    <td width="33%">
      <video src="https://github.com/user-attachments/assets/fad7808b-7c26-4088-bf58-52201bfcea05" controls width="100%"></video>
    </td>
  </tr>
  
  <tr>
    <td width="33%">
      <video src="https://github.com/user-attachments/assets/30145da4-1e84-4db0-99bb-4d1137841fb0" controls width="100%"></video>
    </td>
    <td width="33%">
      <video src="https://github.com/user-attachments/assets/48568d5d-15b4-4a19-8b67-556641e01471" controls width="100%"></video>
    </td>
    <td width="33%">
      <video src="https://github.com/user-attachments/assets/a3a8dce3-8b9d-4e01-a3f1-ae09c5011f6b" controls width="100%"></video>
    </td>
  </tr>
</table>

## 📜 License
SpaceDrive is released under the MIT license. Please see the [LICENSE](LICENSE) file for more information.

## 🔗 Citation -->
```
@article{li2025spacedrive,
  title={SpaceDrive: Infusing Spatial Awareness into VLM-based Autonomous Driving},
  author={Li, Peizheng and Zhang, Zhenghao and Holtz, David and Yu, Hang and Yang, Yutong and Lai, Yuzhi and Song, Rui and Geiger, Andreas and Zell, Andreas},
  journal={arXiv preprint arXiv:2512.10719},
  year={2025}
}
```