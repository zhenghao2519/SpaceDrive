
<div align="center">

# **SpaceDrive**: Infusing Spatial Awareness into VLM-based Autonomous Driving

[Peizheng Li<sup>* 1,2</sup>](https://edwardleelpz.github.io/), [Zhenghao Zhang<sup>* 1,4</sup>](https://zhenghao2519.github.io/), [David Holtz<sup>1</sup>](https://scholar.google.com/citations?user=gf09DbwAAAAJ&hl=en&oi=ao), [Hang Yu<sup>1,5</sup>](https://scholar.google.com/citations?view_op=search_authors&mauthors=Hang+Yu+mercedes&hl=en&oi=ao), [Yutong Yang<sup>1,6</sup>](https://scholar.google.com/citations?hl=en&user=kg9OvU0AAAAJ), [Yuzhi Lai<sup>2</sup>](https://scholar.google.com/citations?user=9Z6Gjo4AAAAJ&hl=en), [Rui Song<sup>7</sup>](https://rruisong.github.io/), [Andreas Geiger<sup>2,3</sup>](https://www.cvlibs.net/), [Andreas Zell<sup>2</sup>](https://uni-tuebingen.de/fakultaeten/mathematisch-naturwissenschaftliche-fakultaet/fachbereiche/informatik/lehrstuehle/kognitive-systeme/the-chair/staff/prof-dr-andreas-zell/) 

<sup>1</sup> Mercedes-Benz AG, <sup>2</sup> University of Tübingen, <sup>3</sup> Tübingen AI Center, <sup>4</sup> TU Munich, <sup>5</sup> Karlsruhe Institute of Technology, <sup>6</sup> University of Stuttgart, <sup>7</sup> UCLA

(*) Equal contribution


<!-- [![arXiv](https://img.shields.io/badge/arXiv-2504.10117-red?logo=arXiv&logoColor=red)](https://arxiv.org/abs/2504.10117) -->
[![License: MIT](https://img.shields.io/github/license/hustvl/GaussTR)](LICENSE)

</div>

##

End-to-end autonomous driving methods built on vision language models (VLMs) have undergone rapid development driven by their universal visual understanding and strong reasoning capabilities obtained from the large-scale pretraining. However, we find that current VLMs struggle to understand fine-grained 3D spatial relationships which is a fundamental requirement for systems interacting with the physical world.To address this issue, we propose SpaceDrive, a spatial-aware VLM-based driving framework that treats spatial information as explicit positional encodings (PEs) instead of textual digit tokens, enabling joint reasoning over semantic and spatial representations. SpaceDrive employs a universal positional encoder to all 3D coordinates derived from multi-view depth estimation, historical ego-states, and text prompts. These 3D PEs are first superimposed to augment the corresponding 2D visual tokens. Meanwhile, they serve as a task-agnostic coordinate representation, replacing the digit-wise numerical tokens as both inputs and outputs for the VLM. This mechanism enables the model to better index specific visual semantics in spatial reasoning and directly regress trajectory coordinates rather than generating digit-by-digit, thereby enhancing planning accuracy. Extensive experiments validate that SpaceDrive achieves state-of-the-art open-loop performance on the nuScenes dataset and the second-best Driving Score of 78.02 on the Bench2Drive closed-loop benchmark over existing VLM-based methods.

![](./assets/architecture.png)

<!-- ## 📰 News -->

## 📰 Code
The code is currently under the internal review process and will be released soon.

## 🎥 Visualizations

<table width="100%">
  <tr>
    <td width="33%"><video src="https://github.com/user-attachments/assets/8abe3871-d570-4253-9c07-50eb8194d44c"></video></td>
    <td width="33%"><video src="https://github.com/user-attachments/assets/ef1ab357-e7bc-46f2-9365-efd67f25bbfb"></video></td>
    <td width="33%"><video src="https://github.com/user-attachments/assets/004cbf80-143e-46d4-b0dd-06815aeaa763"></video></td>
  </tr>
</table>

<table width="100%">
  <tr>
    <td width="33%"><video src="https://github.com/user-attachments/assets/390ae4e0-1909-42b5-a0c2-ad387a2d0530"></video></td>
    <td width="33%"><video src="https://github.com/user-attachments/assets/ef88e799-14e5-4c36-81fb-13b7c6a47221"></video></td>
    <td width="33%"><video src="https://github.com/user-attachments/assets/6e0ffebf-b991-4cfa-aef2-a67d0afd4896"></video></td>
  </tr>
</table>

## 📜 License
SpaceDrive is released under the MIT license. Please see the [LICENSE](LICENSE) file for more information.

<!-- ## 🔗 Citation -->
<!-- ```
@article{li2025spacedrive,
  title={SpaceDrive: Infusing Spatial Awareness into VLM-based Autonomous Driving},
  author={Li, Peizheng and Zhang, Zhenghao and Holtz, David and Yu, Hang and Yang, Yutong and Lai, Yuzhi and Song, Rui and Geiger, Andreas and Zell, Andreas},
  year={2025}
}
``` -->