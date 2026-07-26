"""Orchestrator for the Telco churn pipeline.

Runs the stages in order, each reading the previous stage's output. The three
core stages (clean, select, train) run by default; the slower or optional
stages sit behind toggles. Optional stages are wrapped so a missing dependency
(shap, xgboost) or a missing input file skips that stage with a message rather
than crashing the whole pipeline.

Run everything: python run.py
Iterate on cleaning/encoding only: set RUN_TRAINING = False and rerun.
"""

import data_cleaning
import feature_selection
import model_training

# Stage toggles. The three core stages always run in order. These control the
# slower and optional extras.
RUN_TRAINING = False    # the slow stage: four models, grids, calibration
RUN_EXPLAIN = True    # SHAP charts, needs shap + xgboost installed
RUN_GENERATE = False   # synthetic customers for predict to score
RUN_PREDICT = False    # batch scoring, needs an input file of new customers


def banner(text):
    line = "=" * 64
    print(f"\n{line}\n  {text}\n{line}")


def main():
    banner("STAGE 1: DATA CLEANING")
    data_cleaning.run()

    banner("STAGE 2: FEATURE SELECTION")
    feature_selection.run()

    if RUN_TRAINING:
        banner("STAGE 3: MODEL TRAINING")
        model_training.run()
    else:
        banner("STAGE 3: MODEL TRAINING (skipped, RUN_TRAINING is False)")

    if RUN_EXPLAIN:
        banner("STAGE 4: SHAP EXPLANATION")
        try:
            import explain_model
            explain_model.run()
        except ImportError as e:
            print(f"[skip] explanation needs a missing package: {e}")
        except FileNotFoundError as e:
            print(f"[skip] explanation input not found: {e}")

    if RUN_GENERATE:
        banner("STAGE 5: GENERATE SYNTHETIC CUSTOMERS")
        import generate_test_data
        generate_test_data.run()

    if RUN_PREDICT:
        banner("STAGE 6: BATCH PREDICTION")
        try:
            import predict
            predict.run()
        except FileNotFoundError as e:
            print(f"[skip] no prediction input found: {e}")
            print("[hint] set RUN_GENERATE = True to create a synthetic batch first")

    banner("PIPELINE COMPLETE")


if __name__ == "__main__":
    main()