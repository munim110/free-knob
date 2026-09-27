# Third-party notices

This repository evaluates public datasets, libraries, and released
model checkpoints documented in `README.md` and `REPRODUCE.md`. Those external
assets are not redistributed here and remain governed by their upstream terms.

`common/models/rainnet.py` is a PyTorch implementation based on the public
RainNet architecture by Georgy Ayzel. The upstream RainNet repository is
distributed under the MIT License; its license text is included at
`licenses/RainNet-MIT.txt`.

`crowd/shanghaitech/models/csrnet.py` implements the CSRNet architecture from
Yuhong Li, Xiaofan Zhang, and Deming Chen, "CSRNet: Dilated Convolutional Neural
Networks for Understanding the Highly Congested Scenes," CVPR 2018
(https://openaccess.thecvf.com/content_cvpr_2018/html/Li_CSRNet_Dilated_Convolutional_CVPR_2018_paper.html).
It is an independent implementation using a batch-normalised VGG-16 frontend;
the upstream PyTorch repository and its weights are not redistributed.

All other code and result artefacts in this repository are released under the
MIT License; see `LICENSE`.
