## Data

### Locations
- Input PDFs: `data/pdfs/`
- Converted images: `data/raw/images/`
- YOLO splits: `data/raw/{train,valid,test}/{images,labels}`
- COCO annotations (RFDETR): `_annotations.coco.json` in each split folder

### Annotation format (YOLO)
Each line per box: `class_id cx cy w h` with values normalized to [0,1].

### Converting YOLO → COCO
```bash
python src/yolo_to_coco.py
```

### DVC tracking
- Track inputs and splits: `python -m dvc add data/pdfs data/raw/train data/raw/valid data/raw/test`
- Commit `.dvc` files with Git.





