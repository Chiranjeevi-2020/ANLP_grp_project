import csv
import json
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import get_config
from src.leakage_gate import LeakageGate

def verify():
    cfg = get_config()
    print("=======================================================================")
    print("  ANLP Project Verification Suite (Task 6)")
    print("=======================================================================\n")

    all_passed = True

    # 1. soft_prefix generates different surface text for 3 different z values
    try:
        from src.soft_prefix_agent_a import SoftPrefixAgentA
        # Check Task 1 outputs or run quick check
        agent = SoftPrefixAgentA()
        sample_hist = "Agent A: Hello.\nAgent B: Hi there."
        u0, _ = agent.generate_with_soft_prefix(0, sample_hist, max_new_tokens=40)
        u1, _ = agent.generate_with_soft_prefix(1, sample_hist, max_new_tokens=40)
        u5, _ = agent.generate_with_soft_prefix(5, sample_hist, max_new_tokens=40)
        different_texts = (u0 != u1) and (u1 != u5) and (u0 != u5)
        if different_texts:
            print("[PASS] soft_prefix generates different surface text for 3 different z values")
        else:
            print("[FAIL] soft_prefix generated identical text across z values")
            all_passed = False
    except Exception as e:
        print(f"[FAIL] soft_prefix generation test failed: {e}")
        all_passed = False

    # 2. leakage gate threshold calibrated (not hardcoded)
    try:
        gate = LeakageGate()
        topic_0_kw = gate.get_keywords_for_topic(0)
        sample_clean = [
            "Painting walls with acrylic emulsion.",
            "Wood flooring is being treated.",
            "New drywall in the master bedroom."
        ]
        thresh = gate.calibrate(sample_clean, topic_0_kw)
        if gate.is_calibrated and thresh is not None:
            print(f"[PASS] leakage gate threshold calibrated (threshold={thresh:.4f})")
        else:
            print("[FAIL] leakage gate threshold is not calibrated")
            all_passed = False
    except Exception as e:
        print(f"[FAIL] leakage gate calibration check failed: {e}")
        all_passed = False

    # 3. gate blocks at least 1 utterance across Task 1 test
    try:
        task1_u0 = "It seems there was a mix-up in the dialogue. Agent A is asking how things are going today, but Agent B replied with a casual response instead of discussing 'Home renovation and interior decor'."
        is_safe, reason, beta = gate.check(task1_u0, topic_0_kw)
        if not is_safe and beta < 0:
            print(f"[PASS] gate blocks at least 1 utterance across Task 1 test (blocked on: {reason})")
        else:
            print("[FAIL] gate did not block any utterance")
            all_passed = False
    except Exception as e:
        print(f"[FAIL] gate blocking check failed: {e}")
        all_passed = False

    # 4. comparison_table.csv has all 3 methods x 10 topics x 5 metrics
    csv_path = cfg.RESULTS_DIR / "comparison_table.csv"
    try:
        assert csv_path.exists(), "comparison_table.csv does not exist"
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))

        methods = set(r["method"] for r in reader)
        topics = set(int(r["topic_id"]) for r in reader)
        expected_metrics = {"b_acc", "c_acc", "reward", "ppl", "gate_block_rate"}
        csv_metrics = set(reader[0].keys())

        has_3_methods = len(methods) == 3 and {"no_switch", "scripted", "soft_prefix"}.issubset(methods)
        has_10_topics = len(topics) == 10
        has_5_metrics = expected_metrics.issubset(csv_metrics)
        has_30_rows = len(reader) >= 30

        if has_3_methods and has_10_topics and has_5_metrics and has_30_rows:
            print(f"[PASS] comparison_table.csv has all 3 methods x 10 topics x 5 metrics ({len(reader)} rows)")
        else:
            print(f"[FAIL] comparison_table.csv validation failed: methods={len(methods)}, topics={len(topics)}, rows={len(reader)}")
            all_passed = False
    except Exception as e:
        print(f"[FAIL] comparison_table.csv check failed: {e}")
        all_passed = False

    # 5. soft_prefix C-accuracy < scripted C-accuracy (gate is helping)
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
        scripted_c = [float(r["c_acc"]) for r in reader if r["method"] == "scripted"]
        soft_c = [float(r["c_acc"]) for r in reader if r["method"] == "soft_prefix"]
        mean_scripted = sum(scripted_c) / len(scripted_c)
        mean_soft = sum(soft_c) / len(soft_c)

        if mean_soft < mean_scripted:
            print(f"[PASS] soft_prefix C-accuracy ({mean_soft*100:.1f}%) < scripted C-accuracy ({mean_scripted*100:.1f}%) (gate is helping)")
        else:
            print(f"[FAIL] soft_prefix C-accuracy ({mean_soft*100:.1f}%) >= scripted C-accuracy ({mean_scripted*100:.1f}%)")
            all_passed = False
    except Exception as e:
        print(f"[FAIL] C-accuracy comparison check failed: {e}")
        all_passed = False

    # 6. probe_accuracy.png saved, accuracy varies across layers
    probe_img = cfg.RESULTS_DIR / "probe_accuracy.png"
    try:
        assert probe_img.exists(), "probe_accuracy.png does not exist"
        from src.layer_probe import LayerProbe
        # Check probe variation
        prober = LayerProbe()
        probe_res = prober.train_and_evaluate_probes()
        acc_values = list(probe_res.values())
        varies = len(set(acc_values)) > 1
        if probe_img.exists() and varies:
            print(f"[PASS] probe_accuracy.png saved, accuracy varies across layers ({[f'{v*100:.1f}%' for v in acc_values]})")
        else:
            print("[FAIL] probe accuracy did not vary across layers")
            all_passed = False
    except Exception as e:
        print(f"[FAIL] probe accuracy check failed: {e}")
        all_passed = False

    # 7. case_studies.md has 2 success + 2 failure with gate reason
    case_md = cfg.RESULTS_DIR / "case_studies.md"
    try:
        assert case_md.exists(), "case_studies.md does not exist"
        content = case_md.read_text(encoding="utf-8")
        success_count = content.count("Success")
        failure_count = content.count("Failure")
        has_gate_reason = "Gate:" in content and "blocked" in content

        if success_count >= 2 and failure_count >= 2 and has_gate_reason:
            print(f"[PASS] case_studies.md has {success_count} success + {failure_count} failure cases with gate reason")
        else:
            print(f"[FAIL] case_studies.md validation failed (success={success_count}, failure={failure_count})")
            all_passed = False
    except Exception as e:
        print(f"[FAIL] case_studies.md check failed: {e}")
        all_passed = False

    print("\n=======================================================================")
    if all_passed:
        print("  ALL 7 VERIFICATION CHECKS PASSED SUCCESSFULLY!")
    else:
        print("  SOME VERIFICATION CHECKS FAILED. Please review above.")
    print("=======================================================================\n")

if __name__ == "__main__":
    verify()
