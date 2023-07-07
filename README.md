# OmniFM-DR

## News
We now provide a pretrained OmniFM-DR on June 9, 2023! 

## Online Demo
Click the image to have a try with OmniFM-DR around the chest DR images

<!-- Insert a pipeline of your algorithm here if got one -->
<div align="center">
    <a href="https://medhk23.github.io/OmniFM-DR-demo/"><img width="600px" height="auto" src="demo.gif"></a>
</div>


## Key Features

This repository provides the official implementation of OmniFM-DR

key feature bulletin points here
- First foundation model for multi-task analysis of chest DR image
- The largest full labeled chest DR dataset
- Supoort 5 tpyes of downstream tasks
    - Report Generation
    - Disease Localization
    - Segmentation
    - Classification
    - Visual Question Answering

## Links

- [Paper](https://)
- [Model](https://)
- [Code](https://) 
<!-- [Code] may link to your project at your institute>


<!-- give a introduction of your project -->
## Details

 We have built a multimodal multitask model for DR data, aiming to solve all tasks in this field with one model, such as report generation, disease detection, disease question answering, and even segmentation. Without any fine-tuning, our model has achieved satisfactory results in report generation, disease detection and question answering.

<!-- Insert a pipeline of your algorithm here if got one -->
<div align="center">
    <a href=""><img width="1000px" height="auto" src="omnifmdr.png"></a>
</div>

More intro text here.


## Dataset Links

We utilize 10 public and 6 private datasets for pre-training and provide the download via the following links:

Public dataset: 

- [MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.0.0/)
- [VinDR](https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection/data)
- [ChestX-Det-Dataset](https://github.com/Deepwise-AILab/ChestX-Det-Dataset)
- [ChestX-ray14]( https://nihcc.app.box.com/v/ChestXray-NIHCC)
- [CheXpert]( https://stanfordmlgroup.github.io/competitions/chexpert/)
- [TBX11K](https://www.kaggle.com/datasets/vbookshelf/tbx11k-simplified)
- [object-CXR]( https://github.com/hlk-1135/object-CXR)
- [JSRT Database]( http://db.jsrt.or.jp/eng.php)
- [Shenzhen chest X-ray Set]( https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4256233/)
- [Montgomery County chest X-ray Set]( https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4256233/)

## Get Started

**Main Requirements**  

- python 3.7.4
- pytorch 1.11.0
- torchvision 0.12.1


**Installation**
```bash
git clone https://github.com/MedHK23/OmniFM-DR.git
git clone --depth 1 --filter=blob:none https://github.com/OFA-Sys/OFA.git fairseq
pip install -e ./fairseq/
pip install -r requirements.txt
```

**Download Model**

You can download from [Huggingface](https://huggingface.co/OpenMEDLab/OmniFM-DR/blob/main/omnifmdr.pt). Put it under folder checkpoints.

**Testing**
```bash
python demo.py  # result can be checked in logs
```

## 🛡️ License

This project is under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgement

A lot of code is modified from [OFA-Sys](https://github.com/OFA-Sys/OFA). Thanks them for releasing their codes.

## 
## 📝 Citation

If you find this repository useful, please consider citing this paper:
```
@article{John2023,
  title={paper},
  author={John},
  journal={arXiv preprint arXiv:},
  year={2023}
}
```
