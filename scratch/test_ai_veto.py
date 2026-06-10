import sys
import pandas as pd
from unittest.mock import MagicMock

# Import config and verify threshold
import config
print(f"Configured AI_VETO_THRESHOLD: {config.AI_VETO_THRESHOLD}")
assert config.AI_VETO_THRESHOLD == 0.45, f"Expected 0.45, got {config.AI_VETO_THRESHOLD}"
print("✅ Configuration threshold check passed.")

# Import modules.smc_engine
import modules.smc_engine as smc

# Save original rf_model
original_model = smc.rf_model

try:
    # Create a mock model
    mock_model = MagicMock()
    smc.rf_model = mock_model

    # We will test three scenarios:
    # 1. win_prob = 0.41 (should be vetoed, i.e., approved = False)
    # 2. win_prob = 0.44 (should be vetoed, i.e., approved = False)
    # 3. win_prob = 0.45 (should be approved, i.e., approved = True)
    # 4. win_prob = 0.50 (should be approved, i.e., approved = True)

    features = pd.DataFrame() # dummy features

    for prob in [0.41, 0.44, 0.45, 0.50]:
        # Mock predict_proba to return [[1 - prob, prob]]
        mock_model.predict_proba.return_value = [[1.0 - prob, prob]]
        
        prob_win, approved = smc._ai_evaluate(
            symbol="R_75",
            direction="BUY",
            features=features,
            entry_price=28000.0,
            is_duplicate=False
        )
        
        expected_approved = prob >= 0.45
        print(f"Prob: {prob:.2f} | Approved: {approved} (Expected: {expected_approved}) | Returned Prob: {prob_win}")
        assert approved == expected_approved, f"Assertion failed for probability {prob}"

    print("✅ AI evaluation logic gate tests passed successfully!")

finally:
    # Restore original model
    smc.rf_model = original_model
