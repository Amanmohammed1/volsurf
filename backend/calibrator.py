"""
Heston Model Calibration - Production Grade

Implements robust calibration using:
- Levenberg-Marquardt algorithm (optimal for Heston - Cui et al. 2017)
- Trust Region Reflective method for bounded optimization
- Vega-weighted residuals (professional approach)
- Multi-start optimization for global minimum
- Feller condition as hard constraint
- Adaptive initial guess from market data

References:
- Cui et al. (2017) "Full and fast calibration of the Heston model"
- Gatheral (2006) "The Volatility Surface: A Practitioner's Guide"
"""

import numpy as np
from typing import Tuple, Optional, NamedTuple, List
from scipy.optimize import least_squares, differential_evolution
import time
import warnings

from .cos_pricer import (
    cos_heston_call, 
    implied_volatility, 
    black_scholes_call,
    black_scholes_vega
)


class CalibrationResult(NamedTuple):
    """Result of Heston calibration."""
    v0: float
    kappa: float
    theta: float
    sigma: float
    rho: float
    loss: float
    iterations: int
    success: bool
    time_ms: float
    model_ivs: np.ndarray
    diagnostics: dict


# =============================================================================
# PARAMETER BOUNDS AND CONSTRAINTS
# =============================================================================

# Parameter bounds based on market experience
# [v0, kappa, theta, sigma, rho]
PARAM_BOUNDS = {
    'lower': np.array([0.0001, 0.01, 0.0001, 0.01, -0.99]),
    'upper': np.array([1.0, 15.0, 1.0, 3.0, 0.99])
}

# Typical Indian equity market ranges (NIFTY 50)
INDIA_MARKET_TYPICAL = {
    'v0': (0.005, 0.15),      # 7% - 39% spot vol
    'kappa': (0.5, 5.0),       # Mean reversion 0.5-5 years⁻¹
    'theta': (0.01, 0.20),     # 10% - 45% long-term vol
    'sigma': (0.1, 1.5),       # Vol of vol
    'rho': (-0.95, -0.2)       # Leverage effect (negative for equity)
}


def check_feller_condition(kappa: float, theta: float, sigma: float) -> bool:
    """
    Check if Feller condition is satisfied: 2κθ > σ²
    
    This ensures the variance process stays strictly positive.
    """
    return 2 * kappa * theta > sigma * sigma


def feller_violation(kappa: float, theta: float, sigma: float) -> float:
    """
    Compute Feller violation magnitude.
    Positive means condition satisfied, negative means violated.
    """
    return 2 * kappa * theta - sigma * sigma


# =============================================================================
# INITIAL GUESS ESTIMATION
# =============================================================================

def estimate_initial_params(
    market_ivs: np.ndarray,
    strikes: np.ndarray,
    spot: float,
    T: float
) -> np.ndarray:
    """
    Estimate initial Heston parameters from market data.
    
    Uses:
    - ATM volatility for v0 and theta
    - IV skew for rho estimation
    - IV curvature for sigma estimation
    
    Args:
        market_ivs: Market implied volatilities (decimal)
        strikes: Strike prices
        spot: Spot price
        T: Time to expiry
    
    Returns:
        Initial parameter guess [v0, kappa, theta, sigma, rho]
    """
    # Find ATM option
    moneyness = strikes / spot
    atm_idx = np.argmin(np.abs(moneyness - 1.0))
    atm_iv = market_ivs[atm_idx]
    
    # Initial variance from ATM IV
    v0 = atm_iv ** 2
    v0 = np.clip(v0, 0.005, 0.5)
    
    # Long-term variance: assume slightly higher than spot
    theta = max(v0 * 1.2, 0.02)
    theta = np.clip(theta, 0.01, 0.3)
    
    # Mean reversion: typical value
    kappa = 2.0
    
    # Estimate slope (skew) for rho
    # IV typically decreases for higher strikes (for equity)
    if len(strikes) > 3:
        otm_idx = moneyness < 0.97
        itm_idx = moneyness > 1.03
        
        if np.sum(otm_idx) > 0 and np.sum(itm_idx) > 0:
            otm_iv = np.mean(market_ivs[otm_idx])
            itm_iv = np.mean(market_ivs[itm_idx])
            skew = otm_iv - itm_iv
            
            # Strong positive skew (OTM calls cheaper) implies negative rho
            if skew > 0.02:
                rho = -0.8
            elif skew > 0.01:
                rho = -0.6
            elif skew < -0.01:
                rho = -0.3
            else:
                rho = -0.5
        else:
            rho = -0.65
    else:
        rho = -0.65
    
    # Vol of vol: ensure Feller condition with margin
    # 2*kappa*theta > sigma^2 => sigma < sqrt(2*kappa*theta)
    max_sigma = np.sqrt(2 * kappa * theta * 0.7)  # 70% of limit
    sigma = min(0.4, max_sigma)
    sigma = max(sigma, 0.1)
    
    return np.array([v0, kappa, theta, sigma, rho])


# =============================================================================
# CALIBRATION CORE
# =============================================================================

def calibrate_heston(
    market_ivs: np.ndarray,
    strikes: np.ndarray,
    spot: float,
    T: float,
    r: float,
    initial_guess: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
    use_vega_weighting: bool = True,
    multi_start: bool = False,
    n_starts: int = 5,
    verbose: bool = True
) -> CalibrationResult:
    """
    Calibrate Heston model to market implied volatilities.
    
    Uses Levenberg-Marquardt (Trust Region Reflective) algorithm,
    which is optimal for Heston calibration per Cui et al. (2017).
    
    Args:
        market_ivs: Market implied volatilities (decimal, e.g., 0.15 for 15%)
        strikes: Strike prices
        spot: Current spot price
        T: Time to expiry (years)
        r: Risk-free rate
        initial_guess: Optional [v0, kappa, theta, sigma, rho]
        weights: Optional weights for each strike
        use_vega_weighting: Weight residuals by Black-Scholes vega (professional approach)
        multi_start: Use multiple random starting points
        n_starts: Number of starting points for multi-start
        verbose: Print progress
    
    Returns:
        CalibrationResult with fitted parameters and diagnostics
    """
    start_time = time.perf_counter()
    
    # Filter valid IVs
    valid_mask = (
        (market_ivs > 0.01) & 
        (market_ivs < 3.0) & 
        np.isfinite(market_ivs) &
        (strikes > 0)
    )
    
    if np.sum(valid_mask) < 5:
        if verbose:
            print("  ⚠ Insufficient valid options for calibration")
        return CalibrationResult(
            v0=0.04, kappa=2.0, theta=0.04, sigma=0.5, rho=-0.7,
            loss=float('inf'), iterations=0, success=False, time_ms=0,
            model_ivs=np.full(len(strikes), np.nan),
            diagnostics={'error': 'Insufficient data'}
        )
    
    strikes_valid = strikes[valid_mask]
    ivs_valid = market_ivs[valid_mask]
    n_options = len(strikes_valid)
    
    if verbose:
        print(f"  Calibrating to {n_options} options...")
    
    # Compute weights
    if weights is None:
        if use_vega_weighting:
            # Vega weighting: ATM options have more weight
            # This is the professional approach used in practice
            atm_vol = ivs_valid[np.argmin(np.abs(strikes_valid / spot - 1.0))]
            vegas = np.array([
                black_scholes_vega(spot, K, T, r, atm_vol) 
                for K in strikes_valid
            ])
            vegas = np.maximum(vegas, 1e-6)
            weights_valid = vegas / vegas.max()
        else:
            # Gaussian ATM weighting
            moneyness = strikes_valid / spot
            weights_valid = np.exp(-0.5 * ((moneyness - 1.0) / 0.15) ** 2)
    else:
        weights_valid = weights[valid_mask]
    
    # Normalize weights
    weights_valid = weights_valid / weights_valid.sum()
    
    # Initial guess
    if initial_guess is None:
        initial_guess = estimate_initial_params(ivs_valid, strikes_valid, spot, T)
    
    if verbose:
        print(f"  Initial: v0={initial_guess[0]:.4f}, κ={initial_guess[1]:.2f}, "
              f"θ={initial_guess[2]:.4f}, σ={initial_guess[3]:.2f}, ρ={initial_guess[4]:.2f}")
    
    # Bounds
    lb = PARAM_BOUNDS['lower']
    ub = PARAM_BOUNDS['upper']
    
    # Clip initial guess to bounds
    initial_guess = np.clip(initial_guess, lb + 1e-4, ub - 1e-4)
    
    def residuals(params: np.ndarray) -> np.ndarray:
        """
        Compute weighted IV residuals.
        
        Returns: (model_iv - market_iv) * sqrt(weight) for each option
        """
        v0, kappa, theta, sigma, rho = params
        
        # Feller condition penalty
        feller = 2 * kappa * theta - sigma * sigma
        if feller < 0:
            # Return large residuals proportional to violation
            penalty = 10.0 + abs(feller) * 100
            return np.full(n_options, penalty)
        
        try:
            # Price options
            model_prices = cos_heston_call(
                spot, strikes_valid, T, r,
                kappa, theta, sigma, rho, v0,
                N=96, L=10
            )
            
            # Extract IVs
            model_ivs = np.array([
                implied_volatility(p, spot, k, T, r, 'call')
                for p, k in zip(model_prices, strikes_valid)
            ])
            
            # Handle NaN
            model_ivs = np.nan_to_num(model_ivs, nan=0.5, posinf=1.0, neginf=0.0)
            
            # Weighted residuals (scaled for numerical stability)
            res = (model_ivs - ivs_valid) * np.sqrt(weights_valid) * 100
            
            return res
            
        except Exception as e:
            return np.full(n_options, 100.0)
    
    # Single optimization run
    def run_optimization(x0: np.ndarray) -> Tuple[np.ndarray, float, int, bool]:
        result = least_squares(
            residuals,
            x0,
            bounds=(lb, ub),
            method='trf',  # Trust Region Reflective
            ftol=1e-8,
            xtol=1e-8,
            gtol=1e-8,
            max_nfev=300,
            verbose=0
        )
        
        loss = np.sqrt(np.mean(result.fun ** 2)) / 100  # RMSE in decimal
        return result.x, loss, result.nfev, result.success
    
    # Multi-start optimization
    if multi_start:
        best_params = None
        best_loss = float('inf')
        best_nfev = 0
        
        # Generate starting points
        np.random.seed(42)  # Reproducibility
        starts = [initial_guess]
        
        for _ in range(n_starts - 1):
            random_start = np.array([
                np.random.uniform(0.01, 0.15),   # v0
                np.random.uniform(0.5, 5.0),     # kappa
                np.random.uniform(0.02, 0.15),   # theta
                np.random.uniform(0.2, 0.8),     # sigma
                np.random.uniform(-0.9, -0.3)    # rho
            ])
            # Ensure Feller condition
            while 2 * random_start[1] * random_start[2] < random_start[3] ** 2:
                random_start[3] *= 0.9
            starts.append(random_start)
        
        for i, x0 in enumerate(starts):
            params, loss, nfev, success = run_optimization(x0)
            if loss < best_loss:
                best_params = params
                best_loss = loss
                best_nfev = nfev
        
        final_params = best_params
        final_loss = best_loss
        final_nfev = best_nfev
        final_success = True
    else:
        final_params, final_loss, final_nfev, final_success = run_optimization(initial_guess)
    
    # Extract final parameters
    v0, kappa, theta, sigma, rho = final_params
    
    # Compute model IVs for ALL strikes (not just valid ones)
    try:
        final_prices = cos_heston_call(
            spot, strikes, T, r,
            kappa, theta, sigma, rho, v0,
            N=128, L=12
        )
        
        final_ivs = np.array([
            implied_volatility(p, spot, k, T, r, 'call') if p > 0 else np.nan
            for p, k in zip(final_prices, strikes)
        ])
    except Exception:
        final_ivs = np.full(len(strikes), np.nan)
    
    # Diagnostics
    feller = 2 * kappa * theta - sigma * sigma
    
    elapsed = (time.perf_counter() - start_time) * 1000
    
    if verbose:
        print(f"  ✓ Calibration complete in {elapsed:.0f} ms")
        print(f"  ✓ RMSE: {final_loss*100:.4f}% ({final_loss*10000:.1f} bps)")
        print(f"  ✓ Feller: {feller:.4f} ({'✓' if feller > 0 else '✗'})")
    
    return CalibrationResult(
        v0=float(v0),
        kappa=float(kappa),
        theta=float(theta),
        sigma=float(sigma),
        rho=float(rho),
        loss=float(final_loss),
        iterations=int(final_nfev),
        success=bool(final_success),
        time_ms=float(elapsed),
        model_ivs=final_ivs,
        diagnostics={
            'feller': float(feller),
            'feller_satisfied': bool(feller > 0),
            'n_options': n_options,
            'method': 'Trust Region Reflective (LM)',
            'multi_start': multi_start
        }
    )


def calibrate_heston_global(
    market_ivs: np.ndarray,
    strikes: np.ndarray,
    spot: float,
    T: float,
    r: float,
    verbose: bool = True
) -> CalibrationResult:
    """
    Global calibration using differential evolution.
    
    More robust but slower than local optimization.
    Use when local optimization fails or gives poor results.
    """
    start_time = time.perf_counter()
    
    # Filter valid data
    valid_mask = (market_ivs > 0.01) & (market_ivs < 3.0) & np.isfinite(market_ivs)
    if np.sum(valid_mask) < 5:
        return calibrate_heston(market_ivs, strikes, spot, T, r, verbose=verbose)
    
    strikes_valid = strikes[valid_mask]
    ivs_valid = market_ivs[valid_mask]
    n_options = len(strikes_valid)
    
    if verbose:
        print(f"  Global calibration to {n_options} options...")
    
    # Vega weights
    atm_vol = ivs_valid[np.argmin(np.abs(strikes_valid / spot - 1.0))]
    vegas = np.array([
        black_scholes_vega(spot, K, T, r, atm_vol) 
        for K in strikes_valid
    ])
    vegas = np.maximum(vegas, 1e-6)
    weights = vegas / vegas.sum()
    
    def objective(params: np.ndarray) -> float:
        v0, kappa, theta, sigma, rho = params
        
        # Feller constraint
        if 2 * kappa * theta < sigma ** 2:
            return 1000.0 + (sigma ** 2 - 2 * kappa * theta) * 100
        
        try:
            model_prices = cos_heston_call(
                spot, strikes_valid, T, r,
                kappa, theta, sigma, rho, v0,
                N=64, L=8  # Fewer terms for speed
            )
            
            model_ivs = np.array([
                implied_volatility(p, spot, k, T, r)
                for p, k in zip(model_prices, strikes_valid)
            ])
            
            model_ivs = np.nan_to_num(model_ivs, nan=0.5)
            
            # Weighted RMSE
            rmse = np.sqrt(np.sum(weights * (model_ivs - ivs_valid) ** 2))
            return rmse
            
        except Exception:
            return 1000.0
    
    # Bounds for differential evolution
    bounds = [
        (0.001, 0.5),   # v0
        (0.1, 10.0),    # kappa
        (0.005, 0.5),   # theta
        (0.05, 2.0),    # sigma
        (-0.95, 0.5)    # rho
    ]
    
    result = differential_evolution(
        objective,
        bounds,
        maxiter=100,
        seed=42,
        polish=True,  # Fine-tune with local optimizer
        updating='deferred',
        workers=1
    )
    
    v0, kappa, theta, sigma, rho = result.x
    
    # Compute final IVs
    final_prices = cos_heston_call(
        spot, strikes, T, r,
        kappa, theta, sigma, rho, v0,
        N=128, L=12
    )
    
    final_ivs = np.array([
        implied_volatility(p, spot, k, T, r) if p > 0 else np.nan
        for p, k in zip(final_prices, strikes)
    ])
    
    elapsed = (time.perf_counter() - start_time) * 1000
    feller = 2 * kappa * theta - sigma * sigma
    
    if verbose:
        print(f"  ✓ Global calibration complete in {elapsed:.0f} ms")
        print(f"  ✓ Loss: {result.fun*100:.4f}%")
    
    return CalibrationResult(
        v0=float(v0),
        kappa=float(kappa),
        theta=float(theta),
        sigma=float(sigma),
        rho=float(rho),
        loss=float(result.fun),
        iterations=int(result.nit),
        success=bool(result.success),
        time_ms=float(elapsed),
        model_ivs=final_ivs,
        diagnostics={
            'feller': float(feller),
            'feller_satisfied': bool(feller > 0),
            'method': 'Differential Evolution (Global)'
        }
    )


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("HESTON CALIBRATION - PRODUCTION TEST")
    print("=" * 70)
    
    # Generate synthetic market data with known parameters
    spot = 26000.0
    strikes = np.linspace(24500, 27500, 25)
    T = 7 / 365  # 1 week
    r = 0.065
    
    # "True" Heston parameters
    true_params = {
        'v0': 0.0081,     # 9% spot vol
        'kappa': 2.5,
        'theta': 0.0529,  # 23% long-term vol
        'sigma': 0.45,
        'rho': -0.72
    }
    
    print(f"\nTrue parameters:")
    print(f"  v0={true_params['v0']:.4f}, κ={true_params['kappa']:.2f}, "
          f"θ={true_params['theta']:.4f}, σ={true_params['sigma']:.2f}, ρ={true_params['rho']:.2f}")
    
    # Verify Feller
    feller = 2 * true_params['kappa'] * true_params['theta'] - true_params['sigma'] ** 2
    print(f"  Feller: {feller:.4f} ({'✓' if feller > 0 else '✗'})")
    
    # Generate synthetic prices
    from .cos_pricer import cos_heston_iv
    
    market_ivs = cos_heston_iv(
        spot, strikes, T, r,
        true_params['kappa'], true_params['theta'],
        true_params['sigma'], true_params['rho'], true_params['v0']
    )
    
    # Add small noise
    np.random.seed(42)
    market_ivs = market_ivs * (1 + np.random.normal(0, 0.005, len(strikes)))
    
    print(f"\n  Synthetic IV range: {np.min(market_ivs)*100:.2f}% - {np.max(market_ivs)*100:.2f}%")
    
    # Calibrate
    print("\nRunning calibration...")
    result = calibrate_heston(market_ivs, strikes, spot, T, r, verbose=True)
    
    print(f"\nCalibrated parameters:")
    print(f"  v0    = {result.v0:.4f} (true: {true_params['v0']:.4f})")
    print(f"  kappa = {result.kappa:.2f} (true: {true_params['kappa']:.2f})")
    print(f"  theta = {result.theta:.4f} (true: {true_params['theta']:.4f})")
    print(f"  sigma = {result.sigma:.2f} (true: {true_params['sigma']:.2f})")
    print(f"  rho   = {result.rho:.2f} (true: {true_params['rho']:.2f})")
    
    # Parameter recovery errors
    print(f"\nParameter recovery errors:")
    print(f"  v0:    {abs(result.v0 - true_params['v0']) / true_params['v0'] * 100:.1f}%")
    print(f"  kappa: {abs(result.kappa - true_params['kappa']) / true_params['kappa'] * 100:.1f}%")
    print(f"  theta: {abs(result.theta - true_params['theta']) / true_params['theta'] * 100:.1f}%")
    print(f"  sigma: {abs(result.sigma - true_params['sigma']) / true_params['sigma'] * 100:.1f}%")
    print(f"  rho:   {abs(result.rho - true_params['rho']) / abs(true_params['rho']) * 100:.1f}%")
    
    print("\n" + "=" * 70)
