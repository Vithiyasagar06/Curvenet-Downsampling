3D Point Cloud Classification with CurveNet
An end-to-end 3D geometric deep learning pipeline implementing the CurveNet architecture on the ModelNet40 dataset using PyTorch.

Overview
1. Data Ingestion & Cleaning: Reads raw 3D .off CAD files into spatial point coordinate sets, removing degenerate samples and validating geometry.

2. Preprocessing & Standardization: Resamples shapes to a uniform N=1,024 points and centers/scales them within a unit sphere.

3. Geometric Curve Learning: Exploits local surface curvature and guided walks across k-nearest neighbors (k=20) to extract continuous curve segments.

4. Architecture: Official CurveNet classification network (curvenet_cls.py) utilizing multiscale curve feature extraction and global shape aggregation.

5. Data Augmentation: Real-time 3D SO(2) random rotation around the gravity axis and coordinate jittering paired with Cosine Annealing learning rate scheduling.

6. Evaluation & Inference: Evaluates Overall Accuracy (OA), Mean Class Accuracy (mAcc), and confusion matrices, accompanied by an interactive 3D visual prediction engine for unseen CAD files.
