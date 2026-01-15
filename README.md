# Can We Predict Revenue & Venue Capacity Utilization?

Data exploration and predictive modeling of Ticketmaster tour data.

**Jupyter Notebook:** [prompt_V.ipynb](https://github.com/cheeseinvert/tour-data-capstone/blob/main/prompt_V.ipynb)

## Research Question

Using available live event ticket sales data, can we create a predictive model to allow live event tour planners to accurately predict tour revenue and/or venue capacity utilization for new tours?

## Data Source

Ticketmaster US and Canada live event tour data including Artist Name, Event Date, Venue, City, Country, Revenue, Estimated Capacity, and Tickets Sold.

- **Source:** [touringdata.org](https://www.patreon.com/c/touringdata) (2024-2025 data)
- **Records:** 7,774 events after cleaning
- **Enrichment:** State lookups via Google API, artist genres via Spotify API

## Key Findings

### Revenue Prediction: Success ✅
| Model | R² Score | RMSE |
|-------|----------|------|
| Linear Regression | 0.81 | $1.4M |
| Ridge + Polynomial | 0.81 | $1.4M |
| Decision Tree | 0.76 | $1.6M |
| **Neural Network** | **0.84** | **$1.3M** |

The neural network model explains **84% of variance** in tour revenue using venue capacity, timing (month, day of week), number of shows, state, and genre as features.

### Capacity Utilization Prediction: Limited Success ⚠️
| Model | R² Score |
|-------|----------|
| Linear Regression | 0.12 |
| Ridge + Polynomial | 0.12 |
| Decision Tree | 0.11 |
| Neural Network | 0.15 |

All models struggled to predict capacity utilization (best R² = 0.15), suggesting this metric depends heavily on factors not captured in the dataset—such as artist popularity/social media presence, ticket pricing strategy, local market conditions, and competing events.

## Actionable Recommendations for Tour Planners

1. **Revenue is predictable:** Use venue capacity and timing features to estimate expected revenue with reasonable accuracy (~84% variance explained).

2. **Capacity utilization requires additional data:** Consider supplementing Ticketmaster data with artist social media metrics, historical fan engagement, or market-specific demand indicators.

3. **Feature importance:** Venue capacity and number of shows are the strongest predictors of revenue—book appropriately-sized venues for your artist's draw.

## Techniques Used

- Feature correlation analysis
- One-hot encoding for categorical variables (State, Genre)
- Multiple regression approaches: Linear, Ridge, Polynomial, Decision Tree, Neural Network
- Hyperparameter tuning via GridSearchCV with cross-validation
- Evaluation metrics: R², RMSE, MAE

## Next Steps

1. Incorporate artist popularity metrics (Spotify followers, social media engagement)
2. Add ticket pricing data if available
3. Explore ensemble methods (Random Forest, XGBoost) for potential improvement
4. Consider time-series forecasting for seasonal patterns
