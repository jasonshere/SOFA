# Project Name

Session-based Recommender Systems (SBRSs) aim at timely predicting the next likely item by capturing users’ current preferences in sessions. Existing SBRSs research only focuses on maximizing session utilities, and little has been done on the fairness issue in SBRSs, which is vital but different from the same issue in traditional Recommender Systems (RSs). To fill in this gap, we define a novel concept of session-oriented fairness to enforce individual items to have the same exposure accumulated within each single session, which is flexible enough to provide opportunities to achieve different fairness goals. Then, we devise a Session-Oriented Fairness-Aware algorithm (SOFA) with a dual Temporal Convolutional Networks (TCN) architecture: one is SOUP (Session-Oriented Utility Promoter) and the other is SODA (Session-Oriented Disparity Alleviator). Benefit from the collaborative learning of SOUP and SODA for the evolution of accumulated exposure in sessions, SOFA is effective to maximize session-oriented fairness while maintaining high session utilities. To the best of our knowledge, this research is the first to solve fairness issues in SBRSs. Extensive experiments on real-world datasets demonstrate that SOFA outperforms the state-of-the-art approaches in terms of both utility and fairness.

---

## Installation & Setup

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/jasonshere/SOFA.git
   cd SOFA
   ```

2. **Create a Virtual Environment**  
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Linux/Mac
   .venv\Scripts\activate     # On Windows
   ```

3. **Install Dependencies**  
   ```bash
   pip3 install -r requirements.txt
   ```

---

## Data Preprocessing

Run the preprocessing script to prepare your datasets:

```bash
python3 data/pre_process.py
```

This script will load raw data, filter it, and generate processed files that will be used for training and evaluation.

---

## Training

To train the model, simply run:

```bash
python3 train.py
```

This command will:
- Load the processed dataset
- Initialize and train your model
- Print logs showing training progress and metrics

---

## Citation

If you find this work useful, please cite the following paper:

```
@ARTICLE{10772009,
  author={Li, Jie and Deng, Ke and Li, Jianxin and Ren, Yongli},
  journal={IEEE Transactions on Knowledge and Data Engineering}, 
  title={Session-Oriented Fairness-Aware Recommendation via Dual Temporal Convolutional Networks}, 
  year={2025},
  volume={37},
  number={2},
  pages={923-935},
  keywords={Recommender systems;Vectors;Real-time systems;Convolutional neural networks;Training;Convolution;Accuracy;Predictive models;Prediction algorithms;Nearest neighbor methods;Fairness;exposure;recommendations},
  doi={10.1109/TKDE.2024.3509454}
}
```

---

## Notes

- Ensure GPU drivers are properly installed if you want to train on GPU.
- If you face environment conflicts, recreate the virtual environment or re-install dependencies.
- Feel free to open issues or contact the repository owner if you have any questions.
