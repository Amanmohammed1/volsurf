"""
Data Ingestion: LIVE Option Chain from Upstox API
NO HARDCODED FALLBACKS - Real market data only

For this to work, you need:
1. UPSTOX_ACCESS_TOKEN environment variable set
2. Market must be open (9:15 AM - 3:30 PM IST, Mon-Fri)
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime
from typing import Tuple, Optional, Dict, Any
import os


# Upstox API Configuration
UPSTOX_BASE_URL = "https://api.upstox.com/v2"
NIFTY_INSTRUMENT_KEY = "NSE_INDEX|Nifty 50"


class DataSourceInfo:
    """Metadata about the data source."""
    def __init__(self, source: str, is_live: bool, data_date: str, 
                 is_stale: bool = False, warning: str = None):
        self.source = source
        self.is_live = is_live
        self.data_date = data_date
        self.is_stale = is_stale
        self.warning = warning
    
    def to_dict(self) -> dict:
        return {
            'source': self.source,
            'is_live': self.is_live,
            'data_date': self.data_date,
            'is_stale': self.is_stale,
            'warning': self.warning
        }


def get_upstox_token() -> Optional[str]:
    """Get Upstox access token from environment or config."""
    # First check environment
    token = os.environ.get('UPSTOX_ACCESS_TOKEN')
    if token:
        return token
    
    # Check for config file
    config_path = os.path.join(os.path.dirname(__file__), '..', '.env')
    if os.path.exists(config_path):
        with open(config_path) as f:
            for line in f:
                if line.startswith('UPSTOX_ACCESS_TOKEN='):
                    return line.split('=', 1)[1].strip().strip('"\'')
    
    return None


def get_expiry_dates(token: str) -> list:
    """Fetch available NIFTY option expiry dates from Upstox."""
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'Authorization': f'Bearer {token}'
    }
    
    try:
        url = f"{UPSTOX_BASE_URL}/option/contract"
        params = {'instrument_key': NIFTY_INSTRUMENT_KEY}
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        resp.raise_for_status()
        
        data = resp.json()
        if data.get('status') == 'success':
            contracts = data.get('data', [])
            expiries = sorted(set(c.get('expiry') for c in contracts if c.get('expiry')))
            # Filter to only future expiries
            today = datetime.now().strftime('%Y-%m-%d')
            return [e for e in expiries if e >= today]
        return []
    except Exception as e:
        print(f"Error fetching expiries: {e}")
        return []


def fetch_option_chain(token: str, expiry_date: str) -> Dict[str, Any]:
    """
    Fetch NIFTY option chain from Upstox API.
    
    Returns dict with 'success', 'spot', 'expiry', 'options' on success
    or 'error' message on failure.
    """
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'Authorization': f'Bearer {token}'
    }
    
    try:
        url = f"{UPSTOX_BASE_URL}/option/chain"
        params = {
            'instrument_key': NIFTY_INSTRUMENT_KEY,
            'expiry_date': expiry_date
        }
        
        resp = requests.get(url, params=params, headers=headers, timeout=15)
        resp.raise_for_status()
        
        data = resp.json()
        
        if data.get('status') != 'success':
            return {'error': data.get('message', 'Unknown API error')}
        
        chain = data.get('data', [])
        if not chain:
            return {'error': 'Empty option chain returned'}
        
        # Extract spot price
        spot = chain[0].get('underlying_spot_price', 0)
        
        # Process options data
        options = []
        for item in chain:
            strike = item.get('strike_price')
            
            call = item.get('call_options', {})
            put = item.get('put_options', {})
            
            call_mkt = call.get('market_data', {})
            put_mkt = put.get('market_data', {})
            call_greeks = call.get('option_greeks', {})
            put_greeks = put.get('option_greeks', {})
            
            options.append({
                'strikePrice': strike,
                'CE_lastPrice': call_mkt.get('ltp', 0),
                'PE_lastPrice': put_mkt.get('ltp', 0),
                'CE_impliedVolatility': call_greeks.get('iv', 0),  # Already in %
                'PE_impliedVolatility': put_greeks.get('iv', 0),
                'CE_totalTradedVolume': call_mkt.get('volume', 0),
                'PE_totalTradedVolume': put_mkt.get('volume', 0),
                'CE_openInterest': call_mkt.get('oi', 0),
                'PE_openInterest': put_mkt.get('oi', 0),
                'CE_delta': call_greeks.get('delta', 0),
                'CE_gamma': call_greeks.get('gamma', 0),
                'CE_theta': call_greeks.get('theta', 0),
                'CE_vega': call_greeks.get('vega', 0),
                'PE_delta': put_greeks.get('delta', 0),
                'PE_gamma': put_greeks.get('gamma', 0),
                'PE_theta': put_greeks.get('theta', 0),
                'PE_vega': put_greeks.get('vega', 0),
                'pcr': item.get('pcr', 0),
            })
        
        return {
            'success': True,
            'spot': spot,
            'expiry': expiry_date,
            'options': options
        }
        
    except requests.exceptions.HTTPError as e:
        if e.response is not None:
            try:
                err = e.response.json()
                return {'error': err.get('errors', [{}])[0].get('message', str(e))}
            except:
                pass
        return {'error': str(e)}
    except Exception as e:
        return {'error': str(e)}


def get_nifty_chain() -> Tuple[pd.DataFrame, float, str, DataSourceInfo]:
    """
    Fetch NIFTY option chain from Upstox API.
    
    This is the ONLY data source - no hardcoded fallbacks.
    If API fails, it returns an error that must be handled.
    
    Returns:
        tuple: (DataFrame, spot_price, expiry_date, DataSourceInfo)
        
    Raises:
        RuntimeError: If unable to fetch data (no token, API error, etc.)
    """
    print("\n" + "=" * 50)
    print("FETCHING LIVE NIFTY OPTION CHAIN")
    print("=" * 50)
    
    # Check for token
    token = get_upstox_token()
    if not token:
        raise RuntimeError(
            "UPSTOX_ACCESS_TOKEN not set! "
            "Set environment variable or add to .env file. "
            "Get token from Upstox Developer Portal."
        )
    
    print(f"✓ Token found: {token[:20]}...")
    
    # Get available expiry dates
    expiries = get_expiry_dates(token)
    if not expiries:
        raise RuntimeError("Could not fetch expiry dates from Upstox")
    
    nearest_expiry = expiries[0]
    print(f"✓ Nearest expiry: {nearest_expiry}")
    
    # Fetch option chain
    result = fetch_option_chain(token, nearest_expiry)
    
    if 'error' in result:
        raise RuntimeError(f"Upstox API error: {result['error']}")
    
    # Convert to DataFrame
    df = pd.DataFrame(result['options'])
    spot = result['spot']
    
    # Calculate moneyness
    df['moneyness'] = df['strikePrice'] / spot
    
    # Filter by volume > 500 and moneyness 0.93-1.07 (Liquid Core)
    # Note: Deep wings (>7% OTM) excluded - require jump-diffusion model for proper fitting
    df = df[(df['CE_totalTradedVolume'] > 500) | (df['PE_totalTradedVolume'] > 500)]
    df = df[(df['moneyness'] >= 0.93) & (df['moneyness'] <= 1.07)]
    df = df.sort_values('strikePrice').reset_index(drop=True)
    
    print(f"✓ Fetched {len(df)} options (filtered)")
    print(f"✓ Spot: ₹{spot:,.2f}")
    
    # Convert expiry format for display
    expiry_dt = datetime.strptime(nearest_expiry, '%Y-%m-%d')
    expiry_display = expiry_dt.strftime('%d-%b-%Y')
    
    info = DataSourceInfo(
        source='UPSTOX_LIVE',
        is_live=True,
        data_date=datetime.now().strftime('%d-%b-%Y %H:%M'),
        is_stale=False,
        warning=None
    )
    
    return df, spot, expiry_display, info


def calculate_time_to_expiry(expiry_date_str: str, data_date: datetime = None) -> float:
    """Calculate time to expiry in years."""
    try:
        expiry_date = datetime.strptime(expiry_date_str, "%d-%b-%Y")
    except ValueError:
        try:
            expiry_date = datetime.strptime(expiry_date_str, "%Y-%m-%d")
        except:
            raise ValueError(f"Invalid expiry format: {expiry_date_str}")
    
    reference_date = data_date if data_date else datetime.now()
    days_to_expiry = (expiry_date - reference_date).days
    
    if days_to_expiry <= 0:
        days_to_expiry = 1  # Minimum 1 day (expiry day)
    
    return days_to_expiry / 365.0


def get_risk_free_rate() -> float:
    """Indian risk-free rate (RBI Repo Rate as of 2024)."""
    return 0.065  # 6.5%


if __name__ == "__main__":
    try:
        df, spot, expiry, info = get_nifty_chain()
        
        print("\n" + "=" * 50)
        print("DATA SUMMARY")
        print("=" * 50)
        print(f"Source: {info.source}")
        print(f"📈 Is Live: {info.is_live}")
        print(f"📅 Data Date: {info.data_date}")
        print(f"💰 Spot: ₹{spot:,.2f}")
        print(f"📆 Expiry: {expiry}")
        print(f"📋 Options: {len(df)} strikes")
        
        T = calculate_time_to_expiry(expiry)
        print(f"⏱️  Time to Expiry: {T:.6f} years ({T*365:.1f} days)")
        
    except RuntimeError as e:
        print(f"\n❌ ERROR: {e}")
        print("\nTo fix this:")
        print("1. Set UPSTOX_ACCESS_TOKEN environment variable")
        print("   export UPSTOX_ACCESS_TOKEN='your_token_here'")
        print("2. Or create .env file in project root with:")
        print("   UPSTOX_ACCESS_TOKEN=your_token_here")
