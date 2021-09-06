# Lightweight OpenPose

reference: https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch  
  
Modify some content to fit needed.

## Training

### Test environment
* Python 3.8.8
* PyTorch 1.6.0, PyTorch 1.8.1

### Prerequisites

1. Download COCO 2017 dataset: [http://cocodataset.org/#download](http://cocodataset.org/#download) (train, val, annotations) and unpack it to `<COCO_HOME>` folder.
2. Install PyTorch by the command from [pytorch website](https://pytorch.org/get-started/previous-versions/).
3. Install requirements `pip install -r requirements.txt`.

   #### NOTE
   * Pytorch與cuda版本的搭配及對應的安裝指令可以參考[pytorch網站](https://pytorch.org/get-started/previous-versions/)，直接安裝有問題也可以嘗試用pytorch網站的安裝指令
   * 若在linux環境使用pip 安裝pycocotools有報錯，可以改用conda安裝 `conda install -c conda-forge pycocotools`
   * 建議使用opencv-python-headless取代opencv-python `pip install opencv-python-headless`，避免造成import cv2時出錯

### Training Steps

#### Common Steps
1. Download pre-trained model from [GoogleDrive](https://drive.google.com/drive/folders/1SKKtiK1EeoID0j5H_6xZ-NjC2aEMP4Cr?usp=sharing).

2. Convert train annotations in internal format. Run `python scripts/prepare_train_labels.py --labels <COCO_HOME>/annotations/person_keypoints_train2017.json`. It will produce `prepared_train_annotation.pkl` with converted in internal format annotations.

   [OPTIONAL] For fast validation it is recommended to make *subset* of validation dataset. Run `python scripts/make_val_subset.py --labels <COCO_HOME>/annotations/person_keypoints_val2017.json`. It will produce `val_subset.json` with annotations just for 250 random images (out of 5000).

#### Train from MobileNet weights
* `python train.py --train-images-folder <COCO_HOME>/train2017/ --prepared-train-labels prepared_train_annotation.pkl --val-labels val_subset.json --val-images-folder <COCO_HOME>/val2017/ --checkpoint-path <path_to>/mobilenet_sgd_68.848.pth.tar --from-mobilenet`

#### Train from checkpoint from previous step
* `python train.py --train-images-folder <COCO_HOME>/train2017/ --prepared-train-labels prepared_train_annotation.pkl --val-labels val_subset.json --val-images-folder <COCO_HOME>/val2017/ --checkpoint-path <path_to>/checkpoint_iter_420000.pth --weights-only`

   ##### NOTE
   1. 若有多張顯卡可以設定`--gpu n`，代表使用第n張顯卡訓練
   2. 若執行後有"CUDA out of memory."或類似ram不足的錯誤的話可以透過`--batch-size n`將batch size調小，預設為80，反之若記憶體占用不高，可以調大加快訓練
   3. 預設為5000個epoch儲存checkpoint，若覺得太久可以透過`--checkpoint-after n`調整
   4. 不同stage的模型checkpoint不能共用，沒有checkpoint的話只能從mobilenet參數訓練，參考"Train from MobileNet weights"
   5. 若要調整stage數目`--num-refinement-stages n`預設n為1，代表基本一定要有的第一層加refinement-stages 1層，總共兩層，這次增加到4層stage也就是這邊參數設3

## Validation

* `python val.py --labels <COCO_HOME>/annotations/person_keypoints_val2017.json --images-folder <COCO_HOME>/val2017 --checkpoint-path <CHECKPOINT>`

## Python Demo

* `python demo.py --checkpoint-path <path_to>/checkpoint_iter_370000.pth --video 0`

## Convert to onnx

* `python save_checkpoint_to_onnx.py --num-refinement-stages 3 --size 120 160 --checkpoint-path <path_to_checkpoint> --output-name lw_pose_4stage.onnx`, --num-refinement-stages defalut 1, --output-name default 'lw_pose.onnx'
