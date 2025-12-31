# VolSurf • Heston Volatility Engine

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.24+-013243?style=for-the-badge&logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-1.10+-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white)
![Three.js](https://img.shields.io/badge/Three.js-r160-000000?style=for-the-badge&logo=three.js&logoColor=white)

**Production-grade Heston stochastic volatility calibration for NIFTY 50 options**

*Calibrate. Price. Trade.*

[Live Demo](#quick-start) • [API Docs](#api-endpoints) • [Mathematical Framework](#mathematical-framework)

</div>

---

## 🎯 Overview

VolSurf is a real-time volatility surface calibration engine that fits the **Heston Stochastic Volatility Model** to live NSE option chain data. Built with institutional-grade numerical methods:

| Feature | Implementation |
|---------|---------------|
| **Pricing Engine** | COS Fourier Method (Fang-Oosterlee 2008) |
| **Optimizer** | Levenberg-Marquardt with Trust Region |
| **Residual Weighting** | Vega-weighted (ATM options prioritized) |
| **Numerical Stability** | "Little Heston Trap" formulation |
| **Data Source** | Live Upstox API |
| **Typical RMSE** | < 10 basis points |

## 📐 Mathematical Framework

### The Heston Model

The Heston model describes asset price dynamics with stochastic variance:

$$dS_t = \mu S_t \, dt + \sqrt{v_t} \, S_t \, dW_t^S$$

$$dv_t = \kappa(\theta - v_t) \, dt + \sigma \sqrt{v_t} \, dW_t^v$$

$$\text{Corr}(dW_t^S, dW_t^v) = \rho \, dt$$

### Parameters

| Symbol | Name | Typical Range | Description |
|--------|------|---------------|-------------|
| $v_0$ | Initial Variance | 0.01 - 0.25 | Current volatility level |
| $\kappa$ | Mean Reversion | 0.5 - 10 | Speed of variance reversion |
| $\theta$ | Long-term Variance | 0.01 - 0.25 | Equilibrium variance |
| $\sigma$ | Vol of Vol | 0.1 - 1.0 | Controls kurtosis/smile curvature |
| $\rho$ | Correlation | -0.9 - 0.3 | Leverage effect (negative for equity) |

### Feller Condition

For the variance process to remain strictly positive:

$$2\kappa\theta > \sigma^2$$

### COS Method Pricing

Option prices are computed via Fourier-cosine series expansion:

$$C = e^{-rT} \sum_{k=0}^{N-1} \text{Re}\left[\phi\left(\frac{k\pi}{b-a}\right) \cdot V_k \cdot e^{ik\pi\frac{x-a}{b-a}}\right]$$

Where $\phi(u)$ is the Heston characteristic function with the "little Heston trap" formulation for numerical stability.

## 🏗️ Architecture

```
volsurf/
├── backend/
│   ├── __init__.py
│   ├── main.py             # FastAPI REST API
│   ├── cos_pricer.py       # COS Fourier pricing engine
│   ├── calibrator.py       # Levenberg-Marquardt optimizer
│   └── data_ingestion.py   # Upstox API integration
├── css/
│   └── styles.css          # Glassmorphism UI
├── js/
│   ├── main.js             # Application logic + insights
│   └── three-scene.js      # 3D background
├── index.html              # Frontend SPA
├── requirements.txt
├── run.py                  # Quick start script
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Upstox API token ([Get one here](https://upstox.com/developer/api/))

### Installation

```bash
# Clone the repository
git clone https://github.com/Amanmohammed1/volsurf.git
cd volsurf

# Install dependencies
pip install -r requirements.txt

# Set your Upstox token
echo "UPSTOX_ACCESS_TOKEN=your_token_here" > .env
```

### Running

```bash
# One command start
python run.py

# Or manually:
# Terminal 1: Backend
python -m uvicorn backend.main:app --port 5001

# Terminal 2: Frontend
python -m http.server 8080

# Open http://localhost:8080
```

## 📡 API Endpoints

### `GET /api/health`
Server health check with uptime and features.

### `GET /api/market-data`
Fetch live NIFTY 50 option chain.

**Response:**
```json
{
  "success": true,
  "data": {
    "spot": 26129.6,
    "expiry": "06-Jan-2026",
    "T": 0.0137,
    "r": 0.065,
    "options": [...]
  }
}
```

### `POST /api/calibrate`
Calibrate Heston model to market data.

**Response:**
```json
{
  "success": true,
  "result": {
    "params": {
      "v0": 0.0058,
      "kappa": 2.05,
      "theta": 0.0414,
      "sigma": 0.41,
      "rho": -0.62
    },
    "loss": 0.00077,
    "lossBps": 7.7,
    "converged": true,
    "fellerSatisfied": true,
    "modelIvs": [...]
  }
}
```

### `POST /api/price`
Price a single option with Greeks.

### `POST /api/smile`
Generate model volatility smile.

## 📊 Sample Output

After calibration, the dashboard shows:

- **Calibrated Parameters** with volatility interpretations
- **Volatility Smile Chart** comparing market vs model
- **Market Analysis** with regime detection and mispriced options
- **Model Quality** assessment

### Typical Results (NIFTY 50)

| Parameter | Value | Interpretation |
|-----------|-------|----------------|
| $v_0$ | 0.006 | 7.8% spot volatility |
| $\kappa$ | 2.0 | ~4 month vol half-life |
| $\theta$ | 0.04 | 20% long-term volatility |
| $\sigma$ | 0.4 | Moderate vol-of-vol |
| $\rho$ | -0.65 | Strong leverage effect |
| **RMSE** | <10 bps | Institutional quality |

## 🔬 Technical Details

### Numerical Methods

1. **COS Method** (Fang-Oosterlee 2008)
   - Exponential convergence O(N^-N)
   - 300x faster than quadrature
   - Stable via put-first pricing

2. **Levenberg-Marquardt** (Cui et al. 2017)
   - Optimal for Heston calibration
   - Trust Region Reflective bounds
   - Vega-weighted residuals

3. **Little Heston Trap** (Albrecher 2007)
   - Avoids branch cut issues
   - Stable for all parameters

### Data Pipeline

- Upstox WebSocket/REST for live quotes
- Automatic nearest expiry selection
- Volume/moneyness filtering
- Risk-free rate from RBI repo (6.5%)

## 📚 References

1. Fang, F. & Oosterlee, C.W. (2008). *"A Novel Pricing Method for European Options Based on Fourier-Cosine Series Expansions"*
2. Cui, Y. et al. (2017). *"Full and fast calibration of the Heston stochastic volatility model"*
3. Albrecher, H. et al. (2007). *"The Little Heston Trap"*
4. Gatheral, J. (2006). *"The Volatility Surface: A Practitioner's Guide"*

## 👤 Author

**Aman Mohammed**

- GitHub: [@Amanmohammed1](https://github.com/Amanmohammed1)
- LinkedIn: [Aman Mohammed](https://www.linkedin.com/in/aman-mohammed-2182b51b9/)

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

<div align="center">

**Built for the Quant Community** 🚀

</div>
