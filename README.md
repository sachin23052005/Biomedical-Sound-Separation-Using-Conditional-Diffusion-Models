# Biomedical Sound Separation Using Conditional Diffusion Models

> **Deep Learning | Generative AI | Audio Signal Processing | Biomedical AI | Source Separation**

A deep learning system for separating mixed biomedical auscultation recordings into their **heart** and **lung** sound components using a **conditional diffusion model** operating on time-frequency spectrogram representations.

The project creates controlled synthetic mixtures of heart and lung recordings, trains a conditional U-Net diffusion model to recover the individual sources, reconstructs the separated audio signals, and evaluates separation quality using objective audio metrics such as **SI-SDR, SI-SDR improvement, SNR, MSE, MAE, and spectrogram similarity**.

---

## 📌 Project Overview

Biomedical auscultation recordings can contain multiple overlapping physiological sounds. In particular, heart and lung sounds may occur simultaneously, making it difficult to analyze each signal independently.

Traditional signal-processing techniques can struggle when the component signals overlap heavily in time and frequency.

This project investigates a **generative AI approach** to this problem by using a **conditional diffusion model** to learn the underlying structure of heart and lung acoustic signals.

The system takes a mixed biomedical audio recording as input and attempts to generate two separated outputs:

* ❤️ **Heart sound**
* 🫁 **Lung sound**

The model operates in the spectrogram domain rather than directly predicting raw waveforms.

### High-Level Pipeline

```text
Heart Sound ─────┐
                 ├──► Synthetic Mixture ──► Spectrogram
Lung Sound ──────┘                              │
                                               ▼
                                  Conditional Diffusion Model
                                               │
                                ┌──────────────┴──────────────┐
                                ▼                             ▼
                         Heart Spectrogram              Lung Spectrogram
                                │                             │
                                └──────────────┬──────────────┘
                                               ▼
                                      Audio Reconstruction
                                               │
                                ┌──────────────┴──────────────┐
                                ▼                             ▼
                          Heart Audio                     Lung Audio
```

---

# 🎯 Objectives

The main objectives of this project are:

1. Develop a machine learning pipeline for biomedical audio source separation.
2. Combine heart and lung recordings to create controlled synthetic mixtures.
3. Transform biomedical audio signals into time-frequency representations.
4. Train a conditional diffusion model to estimate clean heart and lung spectrograms.
5. Reconstruct separated audio signals from the predicted spectrograms.
6. Evaluate the quality of the separation using objective signal-processing metrics.
7. Provide an interactive demonstration through a Gradio interface.
8. Develop an end-to-end workflow that can run using GPU acceleration in Google Colab.

---

# 🧠 Why Diffusion Models?

Diffusion models have become an important class of generative models capable of learning complex data distributions.

Instead of directly predicting a clean signal in one step, a diffusion model learns to reverse a gradual noise-addition process.

For this project, the model learns to predict the noise that was added to clean heart and lung spectrograms.

During training:

```text
Clean Heart/Lung Spectrogram
          │
          ▼
   Add Gaussian Noise
          │
          ▼
   Noisy Target
          │
          ├──────────────┐
          │              │
          ▼              ▼
   Mixed Spectrogram   Timestep
          │              │
          └──────┬───────┘
                 ▼
       Conditional U-Net
                 │
                 ▼
          Predicted Noise
```

During inference, the process is reversed. The model starts from noise and progressively generates the estimated heart and lung spectrograms conditioned on the mixed recording.

---

# 🏗️ System Architecture

The project consists of several major components.

## 1. Audio Data Preparation

The project uses two types of biomedical audio:

* **Heart sound recordings**
* **Respiratory/lung sound recordings**

The notebook is designed around:

* **PhysioNet heart sound WAV recordings**
* **ICBHI respiratory sound WAV recordings**

The heart and lung recordings are independently loaded and processed before being combined.

---

## 2. Synthetic Mixture Generation

Obtaining perfectly separated ground-truth recordings for real-world mixed auscultation recordings can be difficult.

To create a supervised learning setup, this project generates synthetic mixtures.

A heart recording and a lung recording are:

1. Loaded.
2. Converted to a common sampling rate.
3. Converted to mono.
4. Normalized.
5. Randomly segmented.
6. RMS-normalized.
7. Added together.

Conceptually:

```text
Mixture(t) = Heart(t) + Lung(t)
```

The clean heart and lung signals remain available as ground truth.

This provides a controlled training environment where:

```text
Input  → Mixed Heart + Lung
Target → Clean Heart + Clean Lung
```

---

# 🎵 Audio Preprocessing

The audio pipeline uses a target sampling rate of:

```text
4,000 Hz
```

Each training example contains:

```text
8 seconds
```

of audio.

The preprocessing pipeline includes:

### Resampling

All recordings are converted to a common sampling rate.

### Mono Conversion

Audio is loaded as a single-channel waveform.

### Amplitude Normalization

The signals are normalized to prevent excessively large amplitudes.

### Random Segmentation

Random 8-second segments are extracted from the recordings.

### RMS Normalization

Heart and lung signals are scaled to controlled RMS levels before mixing.

### Peak Normalization

If the resulting mixture exceeds the allowed amplitude range, the mixture and source signals are scaled accordingly.

---

# 📊 Time-Frequency Representation

The project converts waveform signals into spectrogram representations using the **Short-Time Fourier Transform (STFT)**.

The STFT configuration used in the notebook is:

| Parameter        |     Value |
| ---------------- | --------: |
| Sampling Rate    |  4,000 Hz |
| Segment Duration | 8 seconds |
| FFT Size         |       512 |
| Hop Length       |       128 |
| Window Length    |       512 |
| Window           |      Hann |

The magnitude spectrogram is converted into a logarithmic representation:

```text
log(1 + magnitude)
```

This produces the representation used by the diffusion model.

---

# 🤖 Conditional Diffusion Model

The core of the project is a **conditional U-Net diffusion model**.

The model receives three pieces of information:

### 1. Noisy Target Spectrogram

Two channels:

```text
Channel 1 → Heart
Channel 2 → Lung
```

### 2. Mixture Spectrogram

One channel containing the mixed biomedical recording.

### 3. Diffusion Timestep

A timestep embedding tells the model how much noise is currently present.

Therefore, the effective model input contains:

```text
2 noisy target channels
+
1 mixture conditioning channel
+
timestep embedding
```

---

# 🧩 U-Net Architecture

The diffusion network follows a U-Net-style encoder-decoder structure.

The architecture contains:

```text
Input
  │
  ▼
Downsampling Block 1
  │
  ▼
Downsampling Block 2
  │
  ▼
Downsampling Block 3
  │
  ▼
Middle Block
  │
  ▼
Upsampling Block 3
  │
  ▼
Upsampling Block 2
  │
  ▼
Upsampling Block 1
  │
  ▼
Output
```

Skip connections are used between corresponding encoder and decoder levels.

The model also incorporates timestep information through sinusoidal time embeddings.

---

# ⏱️ Diffusion Configuration

The diffusion process uses:

| Parameter          |              Value |
| ------------------ | -----------------: |
| Diffusion Steps    |                200 |
| Beta Start         |             0.0001 |
| Beta End           |               0.02 |
| Noise Schedule     |             Linear |
| Training Objective |   Noise Prediction |
| Loss Function      | Mean Squared Error |

During training, Gaussian noise is added to the clean target spectrogram at a randomly selected diffusion timestep.

The model then learns to predict the noise that was added.

The training objective is:

```text
L = MSE(predicted_noise, actual_noise)
```

---

# 🏋️ Training

The training pipeline uses:

```text
Batch Size: 8
Epochs: 30
Learning Rate: 2 × 10⁻⁴
Optimizer: Adam
Gradient Clipping: 1.0
```

A synthetic dataset of approximately **1,500 generated examples per epoch** is used in the notebook.

The generated dataset is divided into:

```text
90% → Training
10% → Validation
```

A fixed random seed of:

```text
42
```

is used for reproducibility.

The notebook automatically saves:

```text
last.pt
best.pt
```

The `best.pt` checkpoint is selected based on the lowest validation loss.

---

# 🔄 Inference Pipeline

After training, a mixed WAV file can be passed to the model.

The inference pipeline is:

```text
Input WAV
   │
   ▼
Resampling
   │
   ▼
8-second Segment
   │
   ▼
STFT
   │
   ▼
Log-Magnitude Spectrogram
   │
   ▼
Conditional Diffusion Sampling
   │
   ├───────────────┐
   ▼               ▼
Heart Spectrum   Lung Spectrum
   │               │
   └───────┬───────┘
           ▼
   Spectrogram → Waveform
           │
     ┌─────┴─────┐
     ▼           ▼
Heart WAV     Lung WAV
```

---

# 🔊 Audio Reconstruction

The predicted spectrograms are converted back into waveform audio using inverse STFT.

Because the model predicts magnitude information, the project uses the **phase of the original mixture** during reconstruction.

The resulting files are saved as:

```text
*_mix_segment.wav
*_heart_pred.wav
*_lung_pred.wav
```

---

# 📈 Evaluation

Because the project creates synthetic mixtures from known clean sources, the predicted signals can be compared against the original heart and lung signals.

This makes objective evaluation possible.

The notebook evaluates several metrics.

---

## SI-SDR

**Scale-Invariant Signal-to-Distortion Ratio (SI-SDR)** measures the quality of the estimated source while being less sensitive to overall scale differences.

Higher values generally indicate better separation quality.

The project reports:

```text
Heart SI-SDR
Lung SI-SDR
Overall SI-SDR
```

---

## SI-SDR Improvement

The project also compares the separated signal against the original mixture.

```text
SI-SDR Improvement
=
Separated Signal SI-SDR
-
Mixture SI-SDR
```

This provides a more meaningful measure of whether the separation process improved the source estimate relative to simply using the mixture.

---

## Signal-to-Noise Ratio

SNR is calculated for both predicted sources.

The project reports:

```text
Heart SNR
Lung SNR
Overall SNR
```

Higher values indicate a stronger signal relative to reconstruction error.

---

## Mean Squared Error

MSE is calculated for both:

* Spectrogram predictions
* Reconstructed waveforms

```text
MSE = Mean((Reference - Prediction)²)
```

Lower values indicate smaller reconstruction error.

---

## Mean Absolute Error

MAE is also calculated for the predicted spectrograms.

```text
MAE = Mean(|Reference - Prediction|)
```

---

## Spectrogram Similarity

The notebook additionally calculates an accuracy-like percentage using cosine similarity between predicted and reference spectrogram representations.

This is intended as a similarity indicator rather than a conventional classification accuracy.

---

# 🧪 Evaluation Outputs

The evaluation pipeline generates:

```text
detailed_metrics.csv
summary_metrics.csv
```

The detailed file stores metrics for individual evaluation samples.

The summary file contains mean values across the evaluated examples.

The notebook can report:

```text
Overall Accuracy-like Similarity
Heart Similarity
Lung Similarity

Overall SI-SDR
Heart SI-SDR
Lung SI-SDR

Overall SI-SDR Improvement
Heart SI-SDR Improvement
Lung SI-SDR Improvement

Overall SNR

Heart Spectrogram MSE
Lung Spectrogram MSE
```

---

# 🖥️ Interactive Demo

The project includes an optional **Gradio** interface.

The interface allows a user to upload a mixed biomedical WAV recording and receive:

```text
Input Audio Segment
        │
        ├──► Predicted Heart Sound
        │
        └──► Predicted Lung Sound
```

The interface is created using:

```text
Gradio
```

and can be launched directly from Google Colab.

---

# 🛠️ Technologies Used

### Programming

* Python

### Deep Learning

* PyTorch
* Conditional Diffusion Models
* U-Net
* Neural Networks

### Audio Processing

* Librosa
* SoundFile
* STFT
* Inverse STFT

### Data Science

* NumPy
* Pandas

### Visualization

* Matplotlib

### Evaluation

* SI-SDR
* SNR
* MSE
* MAE
* RMSE
* Cosine Similarity

### Deployment / Demo

* Gradio

### Development Environment

* Google Colab
* Google Drive
* CUDA-enabled GPU

---

# 📁 Project Structure

A recommended GitHub structure for this project is:

```text
Biomedical-Sound-Separation-Diffusion/
│
├── README.md
│
├── notebooks/
│   └── Biomedical_Sound_Separation_Diffusion_Colab.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── dataset.py
│   ├── diffusion.py
│   ├── model.py
│   ├── inference.py
│   └── evaluation.py
│
├── models/
│   └── best.pt
│
├── metrics/
│   ├── detailed_metrics.csv
│   └── summary_metrics.csv
│
├── demo/
│   └── gradio_demo.py
│
├── requirements.txt
│
└── .gitignore
```

> The current implementation is provided primarily as a Google Colab notebook. The structure above represents a recommended production-style organization for further development.

---

# 📦 Dataset

The notebook is designed around two biomedical audio sources:

### PhysioNet Heart Sound Data

Used as the source of heart recordings.

### ICBHI Respiratory Sound Database

Used as the source of lung/respiratory recordings.

The project combines recordings from these sources to generate synthetic mixed examples.

The raw datasets are **not included in this repository**.

Users should obtain the datasets from their respective official sources and configure the dataset paths in the notebook.

---

# 🚀 Running the Project

## 1. Open the Notebook

Open the notebook in:

**Google Colab**

GPU runtime is recommended.

---

## 2. Install Dependencies

The notebook installs:

```bash
pip install librosa soundfile torch torchaudio matplotlib tqdm mir_eval gradio
```

---

## 3. Configure Dataset Paths

Update the dataset directories:

```python
DATA_ROOT = "/content/drive/MyDrive/data/mixed"

TRAIN_HEART_DIR = f"{DATA_ROOT}/train/heart"
TRAIN_LUNG_DIR = f"{DATA_ROOT}/train/lung"
TRAIN_MIXED_DIR = f"{DATA_ROOT}/train/mixed"

VAL_DIR = f"{DATA_ROOT}/val"
TEST_DIR = f"{DATA_ROOT}/test"
```

---

## 4. Start Training

Run the training section of the notebook.

The model will:

1. Load heart and lung recordings.
2. Generate synthetic mixtures.
3. Convert signals into spectrograms.
4. Add diffusion noise.
5. Train the conditional U-Net.
6. Calculate validation loss.
7. Save model checkpoints.

---

## 5. Load the Best Model

The best checkpoint is stored as:

```text
best.pt
```

The notebook automatically loads the best-performing checkpoint for inference.

---

## 6. Run Separation

Provide a mixed WAV file to the separation function:

```python
separate_wav("path/to/mixed_audio.wav")
```

The model generates:

```text
heart_pred.wav
lung_pred.wav
```

---

## 7. Run Evaluation

The evaluation pipeline can be executed using:

```python
evaluate_diffusion_separation(n_examples=5)
```

For more reliable estimates, increase the number of evaluation examples.

For example:

```python
evaluate_diffusion_separation(n_examples=50)
```

---

## 8. Launch the Demo

The optional Gradio interface can be launched from the notebook.

Users can upload a WAV file and listen to the predicted heart and lung components.

---

# 💡 Key Engineering Decisions

### Synthetic Supervised Learning

Real mixed biomedical recordings with clean source-level ground truth are difficult to obtain.

Synthetic mixing provides a controlled supervised learning problem where the original sources are known.

### Spectrogram-Based Modeling

Rather than directly modeling raw waveforms, the project uses time-frequency representations.

This allows the model to learn patterns across both:

```text
Frequency
Time
```

### Conditional Generation

The mixture spectrogram is supplied as conditioning information.

The diffusion model therefore does not generate arbitrary biomedical sounds; it attempts to generate source components consistent with the observed mixture.

### U-Net Architecture

The encoder-decoder structure and skip connections allow the network to preserve useful local time-frequency information while processing higher-level representations.

### Objective Evaluation

Because the synthetic test mixtures have known ground truth, the system can be evaluated quantitatively instead of relying only on subjective listening.

---

# ⚠️ Limitations

This project is a research/experimental prototype and has several limitations.

### 1. Synthetic Mixtures

The model is primarily trained and evaluated using artificially generated mixtures.

Real-world recordings may contain:

* Environmental noise
* Recording-device artifacts
* Different microphone characteristics
* Variable patient anatomy
* Background speech
* Room acoustics
* Different recording locations

Therefore, performance on synthetic mixtures does not guarantee equivalent performance on clinical recordings.

### 2. Mixture Phase Reconstruction

The reconstruction process uses the phase information from the mixture.

This can limit waveform reconstruction quality because the predicted sources do not receive independently estimated phase information.

### 3. Limited Training Configuration

The current notebook uses a relatively compact diffusion U-Net and a limited number of diffusion steps compared with large-scale generative audio systems.

### 4. Computational Cost

Diffusion sampling requires multiple denoising iterations.

Consequently, inference is significantly more computationally expensive than a single forward-pass source separation model.

### 5. Short Audio Segments

The current pipeline operates on approximately 8-second segments.

Long recordings would require a suitable chunking and overlap/reconstruction strategy.

### 6. Evaluation Size

The notebook's example evaluation uses a small number of synthetic samples.

A larger evaluation set should be used before drawing strong conclusions about generalization.

---

# 🔮 Future Improvements

Several improvements could make the system more robust and closer to a production/research-grade biomedical audio system.

### Audio and Dataset Improvements

* Train on a larger number of biomedical recordings.
* Include more diverse heart and respiratory sound conditions.
* Introduce realistic background noise.
* Simulate different recording environments.
* Add microphone/device variability.
* Use real mixed recordings when ground truth is available.

### Model Improvements

* Use a larger U-Net architecture.
* Experiment with different diffusion schedules.
* Explore classifier-free or alternative conditioning strategies.
* Compare diffusion against Conv-TasNet, Demucs, Wave-U-Net, and spectrogram masking approaches.
* Investigate latent diffusion approaches for faster inference.

### Audio Reconstruction Improvements

* Predict complex spectrograms instead of magnitude only.
* Estimate source-specific phase.
* Explore neural vocoders.
* Use overlap-add processing for long recordings.

### Evaluation Improvements

* Increase the evaluation dataset size.
* Compare against classical source separation baselines.
* Compare against conventional deep-learning separation models.
* Perform cross-dataset testing.
* Conduct perceptual listening evaluations.
* Evaluate robustness to noise and recording conditions.

### Deployment Improvements

* Convert the model into a standalone inference service.
* Create a web application.
* Add batch processing.
* Add visualization of input and separated spectrograms.
* Optimize diffusion sampling for faster inference.

---

# 📊 Research Significance

The project demonstrates how **generative AI can be applied to biomedical signal processing**, specifically the separation of overlapping physiological sounds.

Instead of treating the problem purely as conventional audio denoising, the system formulates source separation as a **conditional generation problem**.

The model learns:

```text
Mixed Biomedical Audio
          ↓
Time-Frequency Representation
          ↓
Conditional Generative Modeling
          ↓
Estimated Physiological Sources
```

This provides an experimental foundation for exploring generative models in applications such as:

* Biomedical auscultation
* Respiratory sound analysis
* Cardiac sound analysis
* Digital stethoscope systems
* Clinical audio preprocessing
* Computer-assisted diagnosis
* Physiological signal enhancement

> **Important:** This project is an academic/research prototype and is not intended for clinical diagnosis or medical decision-making.

---

# 📌 Project Highlights

* ❤️ Biomedical heart and lung sound separation
* 🫁 Respiratory audio processing
* 🤖 Conditional diffusion model
* 🧠 U-Net architecture
* 🎵 STFT-based spectrogram processing
* 🔊 Audio source reconstruction
* 📈 SI-SDR evaluation
* 📊 SNR, MSE, MAE and RMSE evaluation
* 🧪 Synthetic supervised dataset generation
* ⚡ GPU-enabled Google Colab training
* 🖥️ Interactive Gradio demonstration
* 💾 Automatic model checkpointing
* 📁 CSV-based evaluation reports

---

# 👨‍💻 Author

**Sachin Anil Prasad**

Computer Science Engineering Student

Interested in:

* Artificial Intelligence
* Generative AI
* Machine Learning
* Cybersecurity
* Data Structures & Algorithms
* Biomedical AI
* Deep Learning

---

# ⭐ Acknowledgements

This project builds upon publicly available biomedical audio resources and open-source Python and deep-learning libraries.

Special acknowledgment goes to the communities maintaining biomedical audio datasets and open-source machine-learning frameworks that make experimentation in biomedical AI possible.

---

# 📜 Disclaimer

This project is developed for **educational and research purposes**.

The generated heart and lung signals should not be interpreted as medically validated signals, and the system should not be used to make clinical diagnoses or medical decisions.

Performance reported on synthetic mixtures should not be interpreted as clinical performance.

---


See the repository's `LICENSE` file for the complete license terms.

