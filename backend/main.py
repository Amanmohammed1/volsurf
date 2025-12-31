"""
FastAPI Backend - Heston Model Calibration

Production-grade REST API with:
- Async endpoints for high performance
- Automatic OpenAPI/Swagger documentation
- Comprehensive error handling
- Request validation with Pydantic
- CORS configuration for frontend

Run with: uvicorn backend.main:app --reload --port 5001
Docs at: http://localhost:5001/docs
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import numpy as np
from datetime import datetime
import time
import traceback
import os

# Load environment variables
from dotenv import load_dotenv
load_dotenv()


def sanitize_for_json(obj):
    """
    Recursively sanitize a data structure for JSON serialization.
    Replaces inf/-inf with None and NaN with None.
    """
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    elif isinstance(obj, (np.ndarray,)):
        return [sanitize_for_json(v) for v in obj.tolist()]
    elif isinstance(obj, (float, np.floating)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, (int, np.integer)):
        return int(obj)
    elif isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    elif obj is None:
        return None
    else:
        return obj

# Import our production modules
from .data_ingestion import (
    get_nifty_chain, 
    calculate_time_to_expiry, 
    get_risk_free_rate,
    DataSourceInfo
)
from .calibrator import calibrate_heston, check_feller_condition
from .cos_pricer import (
    cos_heston_call, 
    cos_heston_put,
    cos_heston_iv,
    heston_delta,
    heston_gamma,
    heston_vega
)


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class DataSourceResponse(BaseModel):
    source: str
    is_live: bool
    data_date: str
    is_stale: bool
    warning: Optional[str] = None


class OptionData(BaseModel):
    strike: float
    moneyness: float
    CE_price: float
    PE_price: float
    CE_iv: float
    PE_iv: float
    CE_volume: int
    PE_volume: int


class MarketDataPayload(BaseModel):
    spot: float
    expiry: str
    T: float
    r: float
    fetchedAt: str
    fetchTimeMs: Optional[int] = None
    dataSource: DataSourceResponse
    options: List[Dict[str, Any]]


class CalibrationRequest(BaseModel):
    """Request body for calibration endpoint."""
    strikes: Optional[List[float]] = Field(None, description="Strike prices")
    ivs: Optional[List[float]] = Field(None, description="Market implied volatilities (decimal)")
    spot: Optional[float] = Field(None, description="Spot price")
    T: Optional[float] = Field(None, description="Time to expiry (years)")
    r: Optional[float] = Field(0.065, description="Risk-free rate")
    use_vega_weighting: bool = Field(True, description="Use vega-weighted residuals")


class HestonParams(BaseModel):
    v0: float = Field(..., description="Initial variance")
    kappa: float = Field(..., description="Mean reversion speed")
    theta: float = Field(..., description="Long-term variance")
    sigma: float = Field(..., description="Volatility of volatility")
    rho: float = Field(..., description="Spot-variance correlation")


class PriceRequest(BaseModel):
    """Request body for option pricing."""
    spot: float
    strike: float
    T: float
    r: float = 0.065
    v0: float = 0.04
    kappa: float = 2.0
    theta: float = 0.04
    sigma: float = 0.3
    rho: float = -0.7
    option_type: str = "call"


# =============================================================================
# FASTAPI APP
# =============================================================================

app = FastAPI(
    title="Heston Calibration API",
    description="""
    Production-grade Heston stochastic volatility model calibration.
    
    ## Features
    - Real-time NIFTY 50 option chain data (via Upstox API)
    - COS method pricing (Fang-Oosterlee 2008) - 300x faster than quadrature
    - Levenberg-Marquardt calibration (optimal per Cui et al. 2017)
    - Vega-weighted residuals for professional-grade calibration
    
    ## Mathematical Model
    The Heston model describes asset dynamics with stochastic volatility:
    - dS = μS dt + √v S dW¹
    - dv = κ(θ - v) dt + σ√v dW²
    - corr(dW¹, dW²) = ρ dt
    """,
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS - allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Track server start time
SERVER_START_TIME = datetime.now()


# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/api/health")
async def health_check():
    """
    Health check endpoint.
    
    Returns server status, version, and uptime.
    """
    uptime = (datetime.now() - SERVER_START_TIME).total_seconds()
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0",
        "uptime_seconds": round(uptime),
        "environment": os.getenv("ENVIRONMENT", "development"),
        "features": {
            "cos_method": True,
            "lm_calibration": True,
            "greeks": True
        }
    }


@app.get("/api/market-data")
async def get_market_data():
    """
    Fetch live NIFTY option chain from Upstox.
    
    - Requires UPSTOX_ACCESS_TOKEN environment variable
    
    Returns spot price, expiry, time to expiry, risk-free rate, and option chain.
    """
    try:
        start = time.perf_counter()
        
        # Fetch live data from Upstox
        df, spot, expiry, source_info = get_nifty_chain()
        
        T = calculate_time_to_expiry(expiry)
        r = get_risk_free_rate()
        
        # Prepare options data
        options = []
        for _, row in df.iterrows():
            options.append({
                'strike': float(row['strikePrice']),
                'moneyness': float(row['moneyness']),
                'CE_price': float(row['CE_lastPrice']),
                'PE_price': float(row['PE_lastPrice']),
                'CE_iv': float(row['CE_impliedVolatility']) / 100,  # Convert to decimal
                'PE_iv': float(row['PE_impliedVolatility']) / 100,
                'CE_volume': int(row['CE_totalTradedVolume']),
                'PE_volume': int(row['PE_totalTradedVolume']),
            })
        
        elapsed = (time.perf_counter() - start) * 1000
        
        return {
            'success': True,
            'data': {
                'spot': spot,
                'expiry': expiry,
                'T': T,
                'r': r,
                'fetchedAt': datetime.now().isoformat(),
                'fetchTimeMs': round(elapsed),
                'dataSource': source_info.to_dict(),
                'options': options
            }
        }
        
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={
                'success': False,
                'error': str(e),
                'hint': 'Ensure UPSTOX_ACCESS_TOKEN is set in .env'
            }
        )


@app.post("/api/calibrate")
async def calibrate(request: CalibrationRequest):
    """
    Calibrate Heston model to market data.
    
    If no data provided in request body, fetches live data from Upstox.
    
    Uses:
    - COS method for option pricing (Fang-Oosterlee 2008)
    - Levenberg-Marquardt optimization (Cui et al. 2017)
    - Vega-weighted residuals (optional, default True)
    
    Returns calibrated parameters, model implied volatilities, and diagnostics.
    """
    try:
        start = time.perf_counter()
        
        # Get data
        if request.strikes is None or request.ivs is None:
            # Fetch live data
            df, spot, expiry, _ = get_nifty_chain()
            T = calculate_time_to_expiry(expiry)
            r = get_risk_free_rate()
            
            # Use call options with valid IVs
            call_opts = df[(df['CE_lastPrice'] > 5) & (df['CE_impliedVolatility'] > 0)]
            strikes = np.array(call_opts['strikePrice'])
            ivs = np.array(call_opts['CE_impliedVolatility']) / 100
        else:
            strikes = np.array(request.strikes)
            ivs = np.array(request.ivs)
            spot = request.spot
            T = request.T
            r = request.r or 0.065
        
        # Run calibration
        result = calibrate_heston(
            ivs, strikes, spot, T, r,
            use_vega_weighting=request.use_vega_weighting,
            verbose=True
        )
        
        # Prepare response
        feller = 2 * result.kappa * result.theta - result.sigma ** 2
        
        elapsed = (time.perf_counter() - start) * 1000
        
        # Sanitize model IVs for JSON (replace inf/nan with None)
        if result.model_ivs is not None:
            model_ivs_clean = [
                float(iv) if np.isfinite(iv) else None 
                for iv in result.model_ivs
            ]
        else:
            model_ivs_clean = []
        
        # Build and sanitize response
        response_data = {
            'success': True,
            'result': {
                'params': {
                    'v0': result.v0,
                    'kappa': result.kappa,
                    'theta': result.theta,
                    'sigma': result.sigma,
                    'rho': result.rho
                },
                'loss': result.loss,
                'lossPercent': result.loss * 100,
                'lossBps': result.loss * 10000,
                'iterations': result.iterations,
                'converged': result.success,
                'feller': feller,
                'fellerSatisfied': feller > 0,
                'timeMs': round(elapsed),
                'calibrationTimeMs': round(result.time_ms),
                'modelIvs': model_ivs_clean,
                'diagnostics': result.diagnostics
            }
        }
        
        # Sanitize entire response for JSON (handles inf/nan in diagnostics etc.)
        return sanitize_for_json(response_data)
        
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={'success': False, 'error': str(e)}
        )


@app.post("/api/price")
async def price_option(request: PriceRequest):
    """
    Price a single option using the Heston model.
    
    Uses COS method (Fang-Oosterlee 2008) for fast, accurate pricing.
    Returns call and put prices, plus Greeks.
    """
    try:
        S = request.spot
        K = request.strike
        T = request.T
        r = request.r
        kappa = request.kappa
        theta = request.theta
        sigma = request.sigma
        rho = request.rho
        v0 = request.v0
        
        # Price
        call_price = cos_heston_call(S, K, T, r, kappa, theta, sigma, rho, v0)
        put_price = cos_heston_put(S, K, T, r, kappa, theta, sigma, rho, v0)
        
        # Greeks
        delta = heston_delta(S, K, T, r, kappa, theta, sigma, rho, v0, request.option_type)
        gamma = heston_gamma(S, K, T, r, kappa, theta, sigma, rho, v0)
        vega = heston_vega(S, K, T, r, kappa, theta, sigma, rho, v0)
        
        # IV
        iv = cos_heston_iv(S, K, T, r, kappa, theta, sigma, rho, v0)
        
        return {
            'success': True,
            'result': {
                'call_price': float(call_price),
                'put_price': float(put_price),
                'iv': float(iv),
                'greeks': {
                    'delta': float(delta),
                    'gamma': float(gamma),
                    'vega': float(vega)
                }
            }
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={'success': False, 'error': str(e)}
        )


@app.post("/api/smile")
async def compute_smile(
    spot: float,
    T: float,
    r: float = 0.065,
    v0: float = 0.04,
    kappa: float = 2.0,
    theta: float = 0.04,
    sigma: float = 0.3,
    rho: float = -0.7,
    strike_min: Optional[float] = None,
    strike_max: Optional[float] = None,
    num_strikes: int = 50
):
    """
    Compute model volatility smile for given Heston parameters.
    
    Returns array of strikes and corresponding implied volatilities.
    """
    try:
        if strike_min is None:
            strike_min = spot * 0.85
        if strike_max is None:
            strike_max = spot * 1.15
        
        strikes = np.linspace(strike_min, strike_max, num_strikes)
        ivs = cos_heston_iv(spot, strikes, T, r, kappa, theta, sigma, rho, v0)
        
        return {
            'success': True,
            'result': {
                'strikes': strikes.tolist(),
                'impliedVols': ivs.tolist(),
                'impliedVolsPercent': (ivs * 100).tolist()
            }
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={'success': False, 'error': str(e)}
        )


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 5001))
    
    print("=" * 60)
    print("HESTON CALIBRATION API")
    print("=" * 60)
    print(f"  Server: http://localhost:{port}")
    print(f"  Docs:   http://localhost:{port}/docs")
    print(f"  ReDoc:  http://localhost:{port}/redoc")
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=port)
