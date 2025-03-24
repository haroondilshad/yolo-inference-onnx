# GitHub Repository Setup Instructions

## 1. Create a New Repository on GitHub

1. Go to [GitHub](https://github.com/) and log in to your account
2. Click on the "+" icon in the top-right corner and select "New repository"
3. Enter a name for your repository (e.g., "yolo-inference-onnx")
4. Add a description (optional): "YOLO model inference and ONNX conversion project"
5. Choose visibility (public or private)
6. DO NOT initialize with README, .gitignore, or license as we already have those files
7. Click "Create repository"

## 2. Push Your Local Repository to GitHub

After creating the repository, GitHub will show instructions for pushing an existing repository. Follow these steps:

```bash
# Replace YOUR_USERNAME and YOUR_REPO_NAME with your GitHub username and repository name
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git branch -M main
git push -u origin main
```

## 3. Include Data Files (Optional)

The large data files are currently ignored by .gitignore:
- YOLO model: yolo11n.pt
- ONNX model: yolo11n.onnx
- Image: image.png

If you want to include these, you have a few options:

### Option 1: Add them directly (not recommended for large files)
```bash
git add image.png yolo11n.pt
git commit -m "Add data files"
git push origin main
```

### Option 2: Use Git LFS (Large File Storage) for the model files

1. Install Git LFS: https://git-lfs.com/
2. Set up Git LFS:
```bash
git lfs install
git lfs track "*.pt" "*.onnx"
git add .gitattributes
git commit -m "Configure Git LFS"
git push origin main
# Then add the files
git add yolo11n.pt image.png
git commit -m "Add model and test image via Git LFS"
git push origin main
```

### Option 3: Share download instructions in the README

- Update the README to include download links or instructions for obtaining the model files
- This is often the best approach for large model files

## 4. Verify Your Repository

- Go to your GitHub repository page to verify all files were uploaded correctly
- Check that the README and other documentation display properly 