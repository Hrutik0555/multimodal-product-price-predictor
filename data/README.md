# Data

Place your files here (they are git-ignored):

| File        | Required columns                                          |
|-------------|-----------------------------------------------------------|
| `train.csv` | `sample_id`, `catalog_content`, `image_link`, `price`     |
| `test.csv`  | `sample_id`, `catalog_content`, `image_link`              |

Column names are configurable in `src/config.py`
(`ID_COL`, `TEXT_COL`, `IMAGE_URL_COL`, `TARGET_COL`).

- `catalog_content` is free text. If it contains fragments such as
  `Item Name: ...`, `Value: 12.0`, `Unit: Ounce` or `Pack of 6`, they are
  parsed with regular expressions into numeric features.
- `image_link` is a public image URL. Images are downloaded to `data/images/`
  and cached by URL hash.
