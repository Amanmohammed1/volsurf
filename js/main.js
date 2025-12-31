/**
 * Main Application Logic
 * Handles API calls, calibration, and chart rendering
 */

// API Configuration - Production uses Railway, local dev uses localhost
const API_BASE = window.location.hostname === 'localhost'
    ? 'http://localhost:5001/api'
    : 'https://web-production-d4bb1.up.railway.app/api';

// State
let marketData = null;
let calibrationResult = null;

// DOM Elements
const elements = {
    marketStatus: document.getElementById('market-status'),
    spotPrice: document.getElementById('spot-price'),
    expiryDate: document.getElementById('expiry-date'),
    timeToExpiry: document.getElementById('time-to-expiry'),
    btnCalibrate: document.getElementById('btn-calibrate'),
    optionsTableBody: document.getElementById('options-table-body'),
    optionsCount: document.getElementById('options-count'),
    loadingOverlay: document.getElementById('loading-overlay'),

    // Parameters
    paramV0: document.getElementById('param-v0'),
    paramV0Vol: document.getElementById('param-v0-vol'),
    paramKappa: document.getElementById('param-kappa'),
    paramTheta: document.getElementById('param-theta'),
    paramThetaVol: document.getElementById('param-theta-vol'),
    paramSigma: document.getElementById('param-sigma'),
    paramRho: document.getElementById('param-rho'),
    rhoBar: document.getElementById('rho-bar'),
    paramLoss: document.getElementById('param-loss'),
    calibrationStatus: document.getElementById('calibration-status'),
    fellerCheck: document.getElementById('feller-check')
};

/**
 * Format numbers nicely
 */
function formatNumber(num, decimals = 4) {
    if (num === null || num === undefined || isNaN(num)) return '--';
    return Number(num).toFixed(decimals);
}

function formatPercent(num) {
    if (num === null || num === undefined || isNaN(num)) return '--';
    return (num * 100).toFixed(2) + '%';
}

function formatCurrency(num) {
    if (num === null || num === undefined || isNaN(num)) return '--';
    return '₹' + Number(num).toLocaleString('en-IN', { maximumFractionDigits: 2 });
}

/**
 * Show/hide loading overlay
 */
function setLoading(loading, message = 'Calibrating...') {
    if (loading) {
        elements.loadingOverlay.querySelector('.loader-text').textContent = message;
        elements.loadingOverlay.classList.add('active');
    } else {
        elements.loadingOverlay.classList.remove('active');
    }
}

/**
 * Update status display
 */
function updateStatus(connected, dataSource = null) {
    const statusDot = elements.marketStatus.querySelector('.status-dot');

    if (connected) {
        // Check if live or archived
        const isLive = dataSource?.is_live;
        const isStale = dataSource?.is_stale;

        if (isLive) {
            elements.marketStatus.innerHTML = '<span class="status-dot connected"></span>Live';
        } else if (isStale) {
            elements.marketStatus.innerHTML = '<span class="status-dot" style="background: #ff6b6b;"></span>Archived (Stale)';
        } else {
            elements.marketStatus.innerHTML = '<span class="status-dot" style="background: #ffc107;"></span>Archived';
        }
    } else {
        elements.marketStatus.innerHTML = '<span class="status-dot"></span>Disconnected';
    }
}

/**
 * Fetch market data from API
 */
async function fetchMarketData() {
    try {
        setLoading(true, 'Fetching market data...');

        const response = await fetch(`${API_BASE}/market-data`);
        const result = await response.json();

        if (!result.success) {
            throw new Error(result.error || 'Failed to fetch market data');
        }

        marketData = result.data;
        updateStatus(true, marketData.dataSource);
        displayMarketData();

    } catch (error) {
        console.error('Error fetching market data:', error);
        updateStatus(false);
        elements.optionsTableBody.innerHTML = `
            <tr>
                <td colspan="6" class="loading-cell" style="color: var(--accent-error);">
                    Error: ${error.message}. Make sure the backend server is running.
                </td>
            </tr>
        `;
    } finally {
        setLoading(false);
    }
}

/**
 * Display market data in UI
 */
function displayMarketData() {
    if (!marketData) return;

    // Update status bar
    elements.spotPrice.textContent = formatCurrency(marketData.spot);
    elements.expiryDate.textContent = marketData.expiry;
    elements.timeToExpiry.textContent = `${(marketData.T * 365).toFixed(1)} days`;

    // Show data source warning if applicable
    const dataSource = marketData.dataSource;
    if (dataSource?.warning) {
        // Add warning banner if not exists
        let warningBanner = document.getElementById('data-warning');
        if (!warningBanner) {
            warningBanner = document.createElement('div');
            warningBanner.id = 'data-warning';
            warningBanner.style.cssText = `
                background: linear-gradient(135deg, rgba(255,107,107,0.1), rgba(255,193,7,0.1));
                border: 1px solid rgba(255,193,7,0.3);
                border-radius: 8px;
                padding: 12px 16px;
                margin: 16px 0;
                font-size: 0.9rem;
                color: #ffc107;
            `;
            const dashboard = document.getElementById('dashboard');
            dashboard.insertBefore(warningBanner, dashboard.firstChild.nextSibling);
        }
        warningBanner.innerHTML = `
            <strong>⚠️ Data Source: ${dataSource.source}</strong><br>
            ${dataSource.warning}
            ${!dataSource.is_live ? '<br><br><em>💡 For live data, set UPSTOX_ACCESS_TOKEN environment variable.</em>' : ''}
        `;
    }

    // Update table
    const options = marketData.options;
    elements.optionsCount.textContent = `${options.length} options`;

    let tableHTML = '';
    options.forEach(opt => {
        const marketIV = opt.CE_iv > 0 ? opt.CE_iv : opt.PE_iv;
        tableHTML += `
            <tr data-strike="${opt.strike}">
                <td>${formatCurrency(opt.strike)}</td>
                <td>${formatNumber(opt.moneyness, 3)}</td>
                <td>${formatCurrency(opt.CE_price)}</td>
                <td>${formatPercent(marketIV)}</td>
                <td class="model-iv">--</td>
                <td class="iv-error">--</td>
            </tr>
        `;
    });

    elements.optionsTableBody.innerHTML = tableHTML;

    // Initial chart with market data only
    renderSmileChart();
}

/**
 * Run Heston calibration
 */
async function calibrate() {
    if (!marketData) {
        alert('Please fetch market data first');
        return;
    }

    try {
        elements.btnCalibrate.classList.add('loading');
        elements.btnCalibrate.disabled = true;
        setLoading(true, 'Calibrating Heston model...');

        // Prepare calibration data
        const callOptions = marketData.options.filter(opt => opt.CE_price > 0 && opt.CE_iv > 0);

        const payload = {
            strikes: callOptions.map(opt => opt.strike),
            ivs: callOptions.map(opt => opt.CE_iv),
            spot: marketData.spot,
            T: marketData.T,
            r: marketData.r
        };

        const response = await fetch(`${API_BASE}/calibrate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const result = await response.json();

        if (!result.success) {
            throw new Error(result.error || 'Calibration failed');
        }

        calibrationResult = result.result;
        displayCalibrationResult(callOptions);

    } catch (error) {
        console.error('Calibration error:', error);
        alert('Calibration failed: ' + error.message);
    } finally {
        elements.btnCalibrate.classList.remove('loading');
        elements.btnCalibrate.disabled = false;
        setLoading(false);
    }
}

/**
 * Display calibration results
 */
function displayCalibrationResult(callOptions) {
    if (!calibrationResult) return;

    const { params, loss, iterations, converged, modelIvs } = calibrationResult;

    // Update parameters display
    elements.paramV0.textContent = formatNumber(params.v0);
    elements.paramV0Vol.textContent = `σ₀ = ${formatPercent(Math.sqrt(params.v0))}`;

    elements.paramKappa.textContent = formatNumber(params.kappa, 2);
    elements.paramTheta.textContent = formatNumber(params.theta);
    elements.paramThetaVol.textContent = `σ∞ = ${formatPercent(Math.sqrt(params.theta))}`;

    elements.paramSigma.textContent = formatNumber(params.sigma, 2);
    elements.paramRho.textContent = formatNumber(params.rho, 2);

    // Rho bar (map -1 to 1 -> 0% to 100%)
    const rhoPercent = ((params.rho + 1) / 2) * 100;
    elements.rhoBar.style.width = `${rhoPercent}%`;

    elements.paramLoss.textContent = loss.toExponential(4);
    elements.calibrationStatus.textContent = converged
        ? `Converged in ${iterations} iterations`
        : 'Did not converge';
    elements.calibrationStatus.style.color = converged
        ? 'var(--accent-success)'
        : 'var(--accent-warning)';

    // Feller condition: 2κθ > σ²
    const feller = 2 * params.kappa * params.theta - params.sigma ** 2;
    elements.fellerCheck.textContent = feller > 0 ? `✓ (${formatNumber(feller)})` : `✗ (${formatNumber(feller)})`;
    elements.fellerCheck.className = 'feller-value ' + (feller > 0 ? 'satisfied' : 'violated');

    // Update table with model IVs
    callOptions.forEach((opt, i) => {
        const row = document.querySelector(`tr[data-strike="${opt.strike}"]`);
        if (row && modelIvs[i] !== undefined) {
            const modelIV = modelIvs[i];
            const error = modelIV - opt.CE_iv;

            row.querySelector('.model-iv').textContent = formatPercent(modelIV);

            const errorCell = row.querySelector('.iv-error');
            errorCell.textContent = (error * 10000).toFixed(1) + ' bps';
            errorCell.className = 'iv-error ' + (error >= 0 ? 'error-positive' : 'error-negative');
        }
    });

    // Update chart
    renderSmileChart(callOptions, modelIvs);

    // Generate analysis insights
    generateInsights(callOptions, modelIvs);
}

/**
 * Generate market analysis insights based on calibration
 */
function generateInsights(callOptions, modelIvs) {
    if (!calibrationResult || !marketData) return;

    const { params, loss } = calibrationResult;
    const spot = marketData.spot;

    // Calculate key metrics
    const spotVol = Math.sqrt(params.v0) * 100;
    const longTermVol = Math.sqrt(params.theta) * 100;
    const halfLife = 0.693 / params.kappa;  // ln(2)/kappa in years
    const halfLifeDays = halfLife * 365;
    const lossInBps = loss * 10000;

    // Find mispriced options (largest model vs market discrepancy)
    const mispricings = [];
    callOptions.forEach((opt, i) => {
        if (modelIvs[i] !== undefined && modelIvs[i] !== null) {
            const error = (modelIvs[i] - opt.CE_iv) * 10000;  // in bps
            mispricings.push({
                strike: opt.strike,
                moneyness: opt.moneyness,
                marketIV: opt.CE_iv * 100,
                modelIV: modelIvs[i] * 100,
                errorBps: error
            });
        }
    });

    // Sort by absolute error
    mispricings.sort((a, b) => Math.abs(b.errorBps) - Math.abs(a.errorBps));
    const topMispriced = mispricings.slice(0, 3);

    // Determine market regime
    let marketRegime = '';
    let regimeColor = '';
    if (params.rho < -0.7 && params.sigma > 0.5) {
        marketRegime = 'High Fear / Crash Risk';
        regimeColor = '#ff6b6b';
    } else if (params.rho < -0.5) {
        marketRegime = 'Normal Equity Skew';
        regimeColor = '#00d4ff';
    } else if (params.rho > -0.3) {
        marketRegime = 'Unusual / Low Skew';
        regimeColor = '#ffc107';
    } else {
        marketRegime = 'Typical Market';
        regimeColor = '#4ade80';
    }

    // Generate headline
    const volDirection = spotVol < longTermVol ? 'below' : 'above';
    const volDiff = Math.abs(spotVol - longTermVol).toFixed(1);

    // Generate story elements
    const volStory = spotVol < longTermVol
        ? `Volatility is currently <strong>compressed</strong>. The market is calm now but expects vol to rise toward ${longTermVol.toFixed(0)}% over the next ${halfLifeDays.toFixed(0)} days.`
        : `Volatility is <strong>elevated</strong>. Current vol of ${spotVol.toFixed(0)}% should decay toward ${longTermVol.toFixed(0)}% equilibrium.`;

    const skewStory = params.rho < -0.6
        ? `Strong negative skew (ρ=${params.rho.toFixed(2)}) indicates the market is pricing in significant downside protection. Puts are expensive relative to calls.`
        : `Moderate skew suggests balanced sentiment without extreme fear or complacency.`;

    // Actionable insight
    const topRich = topMispriced.filter(m => m.marketIV > m.modelIV)[0];
    const actionable = topRich
        ? `The <strong>₹${topRich.strike.toLocaleString()}</strong> strike is trading at ${topRich.marketIV.toFixed(0)}% IV but fair value is only ${topRich.modelIV.toFixed(0)}%. That's a ${Math.abs(topRich.errorBps).toFixed(0)} bps premium — consider selling volatility at this strike.`
        : 'Model is tightly aligned with market — no significant arbitrage detected.';

    // Build insights HTML with story structure
    let insightsHTML = `
        <!-- Headline Takeaway -->
        <div class="insight-headline">
            <div class="headline-badge">Key Insight</div>
            <h3>Volatility is ${volDiff}% ${volDirection} long-term equilibrium</h3>
            <p>${volStory}</p>
        </div>
        
        <!-- Metrics Row -->
        <div class="insights-metrics">
            <div class="metric-item">
                <div class="metric-label">Now</div>
                <div class="metric-value">${spotVol.toFixed(1)}%</div>
                <div class="metric-bar">
                    <div class="metric-fill current" style="width: ${Math.min(spotVol / 30 * 100, 100)}%"></div>
                </div>
            </div>
            <div class="metric-arrow">→</div>
            <div class="metric-item">
                <div class="metric-label">Long-term</div>
                <div class="metric-value">${longTermVol.toFixed(1)}%</div>
                <div class="metric-bar">
                    <div class="metric-fill target" style="width: ${Math.min(longTermVol / 30 * 100, 100)}%"></div>
                </div>
            </div>
            <div class="metric-item">
                <div class="metric-label">Half-life</div>
                <div class="metric-value">${halfLifeDays.toFixed(0)}d</div>
                <div class="metric-subtext">for 50% reversion</div>
            </div>
            <div class="metric-item regime" style="border-color: ${regimeColor}">
                <div class="metric-label">Regime</div>
                <div class="metric-value" style="color: ${regimeColor}">${marketRegime}</div>
            </div>
        </div>
        
        <div class="insights-divider"></div>
        
        <!-- Market Story -->
        <div class="insight-story">
            <h4>What the Smile Tells Us</h4>
            <p>${skewStory}</p>
            
            <div class="param-pills">
                <span class="param-pill">ρ = ${params.rho.toFixed(2)} <small>leverage effect</small></span>
                <span class="param-pill">σ = ${params.sigma.toFixed(2)} <small>vol-of-vol</small></span>
                <span class="param-pill">κ = ${params.kappa.toFixed(2)} <small>mean reversion</small></span>
            </div>
        </div>
        
        <div class="insights-divider"></div>
        
        <!-- Trading Signals -->
        <div class="insight-signals">
            <h4>Trading Signals</h4>
            <p class="signal-intro">Options where market price deviates most from Heston fair value:</p>
            
            <div class="signal-cards">
                ${topMispriced.slice(0, 3).map((m, i) => {
        const isRich = m.marketIV > m.modelIV;
        const premium = Math.abs(m.errorBps);
        return `
                <div class="signal-card ${isRich ? 'sell' : 'buy'}">
                    <div class="signal-rank">#${i + 1}</div>
                    <div class="signal-content">
                        <div class="signal-strike">₹${m.strike.toLocaleString()}</div>
                        <div class="signal-comparison">
                            <span class="iv-market">Mkt ${m.marketIV.toFixed(1)}%</span>
                            <span class="iv-vs">vs</span>
                            <span class="iv-model">Fair ${m.modelIV.toFixed(1)}%</span>
                        </div>
                        <div class="signal-premium">${premium.toFixed(0)} bps ${isRich ? 'rich' : 'cheap'}</div>
                    </div>
                    <div class="signal-action ${isRich ? 'sell' : 'buy'}">${isRich ? 'SELL' : 'BUY'}</div>
                </div>
                    `;
    }).join('')}
            </div>
        </div>
        
        <div class="insights-divider"></div>
        
        <!-- Actionable Summary -->
        <div class="insight-action">
            <h4>Actionable Takeaway</h4>
            <p>${actionable}</p>
        </div>
        
        <!-- Calibration Quality -->
        <div class="insight-quality">
            <span class="quality-label">Model Fit</span>
            <span class="quality-value">${lossInBps.toFixed(1)} bps RMSE</span>
            <span class="quality-badge ${lossInBps < 15 ? 'excellent' : lossInBps < 30 ? 'good' : 'fair'}">${getQualityAssessment(lossInBps).split('(')[0].trim()}</span>
        </div>
    `;

    // Show insights section
    const insightsSection = document.getElementById('insights-section');
    const insightsContent = document.getElementById('insights-content');

    if (insightsSection && insightsContent) {
        insightsContent.innerHTML = insightsHTML;
        insightsSection.style.display = 'block';
    }
}

function getRhoInterpretation(rho) {
    if (rho < -0.8) return "Extreme negative leverage effect. Market heavily prices crash protection.";
    if (rho < -0.6) return "Strong negative correlation typical for equity indices. Volatility spikes on downmoves.";
    if (rho < -0.4) return "Moderate negative correlation. Standard equity market behavior.";
    if (rho < -0.2) return "Mild negative correlation. Less pronounced leverage effect.";
    return "Unusual for equity. May indicate special market conditions.";
}

function getSigmaInterpretation(sigma) {
    if (sigma > 0.8) return "Very high vol-of-vol. Expect significant realized volatility clustering.";
    if (sigma > 0.5) return "High vol-of-vol. Market anticipates fat tails and vol surprises.";
    if (sigma > 0.3) return "Moderate vol-of-vol. Standard market uncertainty.";
    return "Low vol-of-vol. Stable volatility regime.";
}

function getKappaInterpretation(kappa) {
    const halfLife = Math.round(0.693 / kappa * 12 * 30); // in days (approx)
    if (kappa > 5) return `Fast mean reversion (~${Math.round(halfLife / 30)} months). Vol shocks dissipate quickly.`;
    if (kappa > 2) return `Moderate mean reversion (~${Math.round(halfLife / 30)} months). Standard decay speed.`;
    if (kappa > 1) return `Slow mean reversion (${Math.round(halfLife / 30)}+ months). Vol shocks persist.`;
    return "Very slow reversion. Volatility regimes are sticky.";
}

function getQualityAssessment(lossBps) {
    if (lossBps < 5) return "Exceptional fit (institutional grade)";
    if (lossBps < 15) return "Excellent fit (production quality)";
    if (lossBps < 30) return "Good fit (acceptable for trading)";
    if (lossBps < 50) return "Moderate fit (use with caution)";
    return "Poor fit (model may not capture market dynamics)";
}

/**
 * Render volatility smile chart using Plotly
 */
function renderSmileChart(callOptions = null, modelIvs = null) {
    if (!marketData) return;

    const options = callOptions || marketData.options.filter(opt => opt.CE_iv > 0);
    const strikes = options.map(opt => opt.strike);
    const marketIVs = options.map(opt => opt.CE_iv * 100);

    const traces = [
        {
            x: strikes,
            y: marketIVs,
            mode: 'markers',
            type: 'scatter',
            name: 'Market IV',
            marker: {
                color: '#ff6b9d',
                size: 10,
                symbol: 'circle',
                line: {
                    color: '#ffffff',
                    width: 1
                }
            }
        }
    ];

    // Add model curve if available
    if (modelIvs) {
        traces.push({
            x: strikes,
            y: modelIvs.map(iv => iv * 100),
            mode: 'lines',
            type: 'scatter',
            name: 'Heston Model',
            line: {
                color: '#00d4ff',
                width: 3,
                shape: 'spline'
            }
        });
    }

    const layout = {
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: {
            family: 'Inter, sans-serif',
            color: '#a0a0b8'
        },
        xaxis: {
            title: 'Strike Price (₹)',
            gridcolor: 'rgba(255,255,255,0.05)',
            linecolor: 'rgba(255,255,255,0.1)',
            tickformat: ',.0f',
            tickprefix: '₹'
        },
        yaxis: {
            title: 'Implied Volatility (%)',
            gridcolor: 'rgba(255,255,255,0.05)',
            linecolor: 'rgba(255,255,255,0.1)',
            ticksuffix: '%'
        },
        margin: { t: 20, r: 20, b: 60, l: 60 },
        showlegend: true,
        legend: {
            x: 0.02,
            y: 0.98,
            bgcolor: 'rgba(15,15,25,0.8)',
            bordercolor: 'rgba(255,255,255,0.1)',
            borderwidth: 1
        },
        hovermode: 'x unified'
    };

    const config = {
        responsive: true,
        displayModeBar: false
    };

    Plotly.newPlot('smile-chart', traces, layout, config);
}

/**
 * Event Listeners
 */
document.addEventListener('DOMContentLoaded', () => {
    // Fetch data on load
    fetchMarketData();

    // Calibrate button
    elements.btnCalibrate.addEventListener('click', calibrate);

    // Smooth scroll for CTA
    document.querySelector('.cta-button')?.addEventListener('click', (e) => {
        e.preventDefault();
        document.getElementById('dashboard').scrollIntoView({ behavior: 'smooth' });
    });
});

// Refresh data every 5 minutes
setInterval(fetchMarketData, 5 * 60 * 1000);
