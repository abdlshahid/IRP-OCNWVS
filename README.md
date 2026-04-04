# 🌊 Wave Spectrum Reconstruction using Physics-Informed Machine Learning

## Overview
This project investigates the reconstruction of 2D directional ocean wave spectra \(S(f,\theta)\) using analytical methods, machine learning models, and physics-informed approaches.

## Methods
- Baselines: Fourier (1st & 2nd harmonic), NDBC, MEM  
- ML Models: MLP, ResNet (1D & 2D CNN), FNO  
- Physics: Energy conservation, moment matching, KL divergence  
- Residual learning: baseline + ML correction  

## Data
- NDBC buoy data (initial approach)  
- ERA5 wave reanalysis (main dataset)  

## Key Results
- MEM provides best accuracy  
- Residual learning significantly improves ML models  
- Physics + residual models perform best overall  
- ML models struggle with cross-region generalization  

## Run
```bash
pip install -r requirements.txt 