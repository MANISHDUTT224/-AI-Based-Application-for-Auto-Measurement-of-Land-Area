# -AI-Based-Application-for-Auto-Measurement-of-Land-Area

🌍 Cloud Removal & Land Segmentation using CycleGAN & UNET Land Segmentation**  
🚀 A deep learning-based application for *cloud removal* from satellite images and *land segmentation* using COCO segmentation.  

## 📌 *Features*  
✅ *Cloud Removal* using a trained *CycleGAN* model.  
✅ *Land Segmentation* using a *COCO segmentation model*.  
✅ *Flask-based Backend* to handle image processing.  
✅ *React/JavaScript-based Frontend* for user-friendly interaction.  
✅ *Git LFS Support* to handle large files efficiently.  
## 🛠 *Tech Stack*  
- *Frontend*: HTML, CSS, JavaScript (React/VanillaJS)  
- *Backend*: Flask (Python)  
- *Machine Learning Models*: PyTorch (CycleGAN), COCO segmentation  
- *Database*: N/A (File-based storage)  

  📂 *Project Structure
```
├── backend/                    # Flask Backend
│   ├── app.py                   # Main Flask app
│   ├── cloud_removal.py         # Cloud Removal Model
│   ├── coco_segmentation.py     # Land Segmentation Model
│   ├── static/                  # Static Files (CSS, JS)
│   ├── templates/               # HTML Templates
│   ├── uploads/                 # Input Images Folder
│   ├── outputs/                 # Processed Images Folder
├── frontend/                   # Frontend Code
│   ├── index.html               # Frontend UI
│   ├── app.css                  # Stylesheet
│   ├── scripts.js               # JavaScript Logic
├── dataset/                    # Large Dataset Files (Handled by Git LFS)
├── models/                     # Pretrained ML Models (Handled by Git LFS)
├── .gitattributes               # Git LFS Configurations
├── .gitignore                   # Ignored Files
├── requirements.txt             # Python Dependencies
├── README.md                    # Project Documentation
```
🚀 *Installation & Setup*  

*1. Clone the Repository*
bash
```
git clone https://github.com/KABILESH77/-AI-Based-Application-for-Auto-Measurement-of-Land-Area
```

```
cd -AI-Based-Application-for-Auto-Measurement-of-Land-Area
```

🔹 *2. Install Dependencies*  
Make sure you have Python 3.x installed. Then, install the required libraries:  
bash
pip install -r requirements.txt


3. Setup Git LFS (For Large Files)*
bash
git lfs install
git lfs pull


 *4. Run the Flask Backend*
bash
python app.py

The backend should now be running on http://127.0.0.1:5000/.

 *5. Open Frontend*
- Open index.html in your browser, or  
- If using React, run:
bash
npm install
npm start

## 📸 *Usage*  
1️⃣ Upload a *satellite image* (cloudy image).  
2️⃣ Click "Process" → The *CycleGAN* model removes clouds.  
3️⃣ The output is *land-segmented* using *COCO segmentation*.  
4️⃣ The final processed image is displayed & *automatically downloaded*.  

## 🎯 *Results & Improvements*  
✅ *Cloud-Free, High-Resolution Images*  
✅ *Accurate Terrain Segmentation*  
✅ *Fast Processing with Flask API*  

🔹 *Future Work*:  
- Improve CycleGAN model for better color correction.  
- Enhance frontend with React-based UI.  

---

## 📜 *License*  
🔓 This project is *open-source* under the *MIT License*.  

---

## 🙌 *Contributors*  
👤 *Kabilesh Kumar* - Developer  
🔗 *GitHub*: [KABILESH77](-AI-Based-Application-for-Auto-Measurement-of-Land-Area)
