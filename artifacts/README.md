# Runtime artifacts

`runtime_artifacts.tar.gz` contains the exact verified inference artifacts:

- `models/best_delivery_risk_model.joblib`
- `models/model_features.json`
- `data/processed/delivery_features.csv`

The archive exists because those generated directories are intentionally
ignored during normal data-science work, but a clean CI runner and Git-based
container deployment still need immutable runtime inputs.

The archive does not contain `.env`, provider credentials, raw Olist files,
customer names, notebooks, reports, MLflow data, or training code.

It does not retrain or alter the model. Docker extracts these exact bytes, and
CI verifies the archive checksum before testing.

To restore the runtime files from a clean checkout:

```powershell
tar -xzf artifacts/runtime_artifacts.tar.gz
```

Archive SHA-256:

```text
60422B1303ACA4DCB68AE2DE663EDB3D60F8C7FE40793FA871FB27EFE6D03663
```

Contained source checksums at packaging time:

```text
best_delivery_risk_model.joblib  AD6C2B24E92CA1D2AFE39C90AEE890B94B740247C31B8579BED6658952EC3588
model_features.json              EFFEC0504BE6281D9E7CD9AEE57E8452165B9169221442A6A73CF49E75892B69
delivery_features.csv            FCB599F79459A0B279D1E84200CAAFFD54368B769A35F08D3E0FD45FEE7A3C27
```
