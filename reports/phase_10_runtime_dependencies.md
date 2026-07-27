# Phase 10 Runtime Dependency Audit

The development environment keeps using `requirements.txt`. The production
FastAPI image uses `requirements-api.txt`.

## Included

| Dependency | Runtime reason |
|---|---|
| FastAPI | HTTP application and schemas |
| Uvicorn | ASGI production process |
| Pydantic | response models, LLM schemas, validated settings |
| python-dotenv | local `.env` support; production still uses process environment |
| pandas | prepared feature-table loading and filtering |
| NumPy | model/tool numerical operations |
| scikit-learn | saved Logistic Regression pipeline and nearest neighbors |
| joblib | saved model artifact loading |
| OpenAI SDK | selectable OpenAI router provider |
| Google GenAI SDK | selectable Gemini router provider |
| LangGraph | alternative graph endpoint |

Both provider SDKs are currently required because `src/agent/router.py`
imports both eagerly. Removing the unused provider through lazy imports could
reduce a future image further, but is outside the minimum dependency split.

LangGraph installs its required LangChain Core packages transitively; the API
does not import a separate LangChain agent.

## Excluded

| Dependency | Why the API does not need it |
|---|---|
| KaggleHub | raw-data download only |
| Matplotlib and Seaborn | report/training visualization |
| Jupyter | notebook development |
| MLflow | experiment tracking |
| XGBoost | training comparison; production winner is Logistic Regression |
| pytest | test runner, not runtime |
| requests | Streamlit's HTTP client, not the FastAPI backend |
| Streamlit | separate UI process |

The saved production model is a scikit-learn pipeline, so XGBoost is not
required to deserialize it.

## Required runtime artifacts

Only these data/model artifacts are needed:

- `models/best_delivery_risk_model.joblib`
- `models/model_features.json`
- `data/processed/delivery_features.csv`

`delivery_dataset.csv`, the baseline model, raw data, MLflow storage, reports,
notebooks, training modules, and screenshots are not needed by the API.
