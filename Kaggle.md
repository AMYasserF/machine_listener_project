How to Train on Kaggle (Free GPU)
Step 1 — Upload your source code as a Kaggle Dataset
Go to kaggle.com/datasets → click "+ New Dataset"
Name it machine-listener-src
Upload your entire src/ folder (containing preprocess.py, features.py, dataset.py, model.py, train.py, __init__.py)
Click Create
Step 2 — Create a Kaggle Notebook
Go to kaggle.com/code → click "+ New Notebook"
On the right panel → Settings → Accelerator → select GPU T4 x2
"+ Add Data" → search alieldinalaa/nn-cmp27-dataset → Add it
"+ Add Data" again → go to "Your Datasets" → add machine-listener-src
Step 3 — Paste the notebook cells
Open kaggle_train.py and paste each cell block (marked with CELL 1, CELL 2, etc.) into separate cells in the Kaggle notebook.

Step 4 — Run & Download
Run all cells — training takes ~30-60 min on T4 GPU
When done, click the "Output" tab on the right panel
Download model.onnx and best_model.pt
Place them in your local d:\CMP\NN\project\models\ folder
Step 5 — Local inference
bash
python infer.py data/ models/model.onnx
That's it! The notebook auto-discovers both datasets and handles everything.