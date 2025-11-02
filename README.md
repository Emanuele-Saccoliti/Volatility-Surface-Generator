# Repository Description

- This reposority provides a Python-based engine leveraging JAX for automatic differentiation to price European options, calculate Greeks, and compute implied volatilities via Newton-Raphson optimization from market data.
  
- Extracts market data using Yahoo Finance and processes option chains across multiple expirations and strikes.
  
- The tool displays a 3D volatility surface with color gradients showing how implied volatility changes across strike prices and expirations, providing a clear visualization of the volatility smile and term structure.


# 🔍 Key Objectives
* Generate volatility smiles & surfaces consistent with market conditions
* Provide a fast and differentiable pipeline for pricing, Greeks, and implied vol computation
* Visualize implied volatility surfaces and smiles to study market structure
* Build a flexible framework extendable to stochastic volatility and exotic pricing
  

# 📌 Key Takeaways
* High-performance computation: Iterative root-finding can be computationally expensive; using vectorized JAX operations accelerates pricing and gradient evaluation.
* Efficient Greek computation: Greeks involve derivatives of the pricing function. Leveraging JAX for automatic differentiation avoids manual derivation, reduces numerical error, and improves computational speed.
Reliable implied volatility extraction: Computing implied volatilities requires solving a nonlinear equation. Using Newton–Raphson with JAX-based automatic differentiation ensures fast and stable convergence, even for strikes with low liquidity or poor quoting quality.


# ⚠️ Challenges
* Handling real-world option data: Market option chains often contain noise, missing data, and arbitrage inconsistencies. Cleaning data and filtering strikes/maturities is essential for producing a smooth implied volatility surface.
* Robust volatility surface construction: Building a 3D implied volatility surface across strike/maturity dimensions requires careful interpolation to capture smiles and term structures without introducing artifacts.
