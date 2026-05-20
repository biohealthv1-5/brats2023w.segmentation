# -*- coding: utf-8 -*-
"""
Step 14: Whole-Slice vs Patch-Based 최종 비교 보고서
=====================================================
모든 평가 결과를 종합하여 비교 요약 보고서를 자동 생성합니다.
(Markdown + JSON 출력)

사용법:
    python code/step14_comparison.py
"""

import sys, json
from pathlib import Path
from datetime import datetime

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

PROJECT_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_DIR / "outputs" / "logs"
GRADCAM_OLD = PROJECT_DIR / "outputs" / "figures" / "gradcam" / "gradcam_analysis.json"
GRADCAM_NEW = PROJECT_DIR / "outputs" / "figures" / "patch_gradcam" / "patch_gradcam_analysis.json"


def safe_load(path):
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def main():
    print("=" * 60)
    print("  Step 14: 최종 비교 보고서 생성")
    print("=" * 60)

    # ─── 데이터 로드 ─────────────────────────────────────────
    ws_eval = safe_load(LOG_DIR / "evaluation_results.json")
    ws_train = safe_load(LOG_DIR / "training_history.json")
    pa_eval = safe_load(LOG_DIR / "step12_patch_eval.json")
    pa_train = safe_load(LOG_DIR / "step11_train_log.json")
    patch_stats = safe_load(LOG_DIR / "step8_stats.json")
    gcam_old = safe_load(GRADCAM_OLD)
    gcam_new = safe_load(GRADCAM_NEW)

    if not ws_eval or not pa_eval:
        print("  [ERROR] 평가 결과 파일이 부족합니다.")
        print(f"    whole-slice: {LOG_DIR / 'evaluation_results.json'} → {'있음' if ws_eval else '없음'}")
        print(f"    patch:       {LOG_DIR / 'step12_patch_eval.json'} → {'있음' if pa_eval else '없음'}")
        return

    # best 집계 방법
    best_method = pa_eval.get("best_aggregation", "max")
    pa_slice = pa_eval["slice_level"].get(best_method, {})
    pa_patch = pa_eval["patch_level"]

    # Grad-CAM IoU
    old_iou = 0.0
    new_iou = 0.0
    if gcam_old:
        old_iou = gcam_old.get("TP_평균_IoU", gcam_old.get("TP_\ud3c9\uade0_IoU", 0))
    if gcam_new:
        new_iou = gcam_new.get("patch_tp_mean_iou", 0)

    # ─── Markdown 보고서 생성 ────────────────────────────────
    lines = []
    L = lines.append

    L(f"# 🔬 Whole-Slice vs Patch-Based 최종 비교 보고서")
    L(f"")
    L(f"> 생성일: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    L(f"")
    L(f"---")
    L(f"")

    # 데이터 규모
    L(f"## 1. 데이터 규모 비교")
    L(f"")
    L(f"| 항목 | Whole-Slice | Patch-Based |")
    L(f"|------|:-----------:|:-----------:|")
    L(f"| 입력 크기 | 224×224 | 64×64 |")
    if patch_stats:
        L(f"| 총 데이터 수 | 166,626 슬라이스 | {patch_stats['total_patches_extracted']:,} 패치 |")
        L(f"| Positive 비율 | 48.8% | {patch_stats['positive_patches']/patch_stats['total_patches_extracted']*100:.1f}% |")
    L(f"| 모델 | ResNet-18 (11.2M) | PatchCNN (~95K) |")
    L(f"")

    # 학습 설정
    L(f"## 2. 학습 설정 비교")
    L(f"")
    L(f"| 항목 | Whole-Slice | Patch-Based |")
    L(f"|------|:-----------:|:-----------:|")
    L(f"| 모델 | ResNet-18 Pretrained | 경량 CNN (3-block) |")
    L(f"| 학습 전략 | 2-Phase (FC→Full) | 단일 Phase |")
    if ws_train:
        L(f"| 총 Epochs | {len(ws_train['train_loss'])} | {pa_train['hyperparams']['epochs_run'] if pa_train else '?'} |")
    if pa_train:
        L(f"| Best Epoch | 9 | {pa_train['best_epoch']} |")
        L(f"| 학습 시간 | - | {pa_train['total_time_min']:.0f}분 |")
    L(f"| Batch Size | 32 | 128 |")
    L(f"| 클래스 균형 | pos_weight | 서브샘플 1:3 + WeightedSampler |")
    L(f"")

    # 핵심 성능 비교
    L(f"## 3. 핵심 성능 비교 (슬라이스 단위)")
    L(f"")
    L(f"| 지표 | Whole-Slice | Patch ({best_method}) | 차이 |")
    L(f"|------|:-----------:|:---------------------:|:----:|")

    metrics = [
        ("Accuracy", ws_eval["accuracy"], pa_slice.get("accuracy", 0)),
        ("Precision", ws_eval["precision"], pa_slice.get("precision", 0)),
        ("Recall", ws_eval["recall_sensitivity"], pa_slice.get("recall", 0)),
        ("Specificity", ws_eval["specificity"], pa_slice.get("specificity", 0)),
        ("F1-Score", ws_eval["f1_score"], pa_slice.get("f1", 0)),
        ("AUC-ROC", ws_eval["auc_roc"]*100, pa_slice.get("auc_roc", 0)*100),
    ]
    for name, w, p in metrics:
        d = p - w
        arrow = "↑" if d > 0 else "↓" if d < 0 else "="
        L(f"| {name} | {w:.2f}% | {p:.2f}% | {arrow}{abs(d):.2f}% |")
    L(f"")

    # FN/FP 비교
    ws_cm = ws_eval["confusion_matrix"]
    pa_cm = pa_slice.get("cm", {})

    L(f"## 4. 오류 분석 비교")
    L(f"")
    L(f"| 항목 | Whole-Slice | Patch ({best_method}) | 변화 |")
    L(f"|------|:-----------:|:---------------------:|:----:|")

    ws_fn, ws_fp = ws_cm["FN"], ws_cm["FP"]
    pa_fn = pa_cm.get("FN", 0)
    pa_fp = pa_cm.get("FP", 0)
    fn_d = pa_fn - ws_fn
    fp_d = pa_fp - ws_fp
    L(f"| FN (놓친 종양) | {ws_fn:,} | {pa_fn:,} | {'↓' if fn_d<0 else '↑'}{abs(fn_d):,} |")
    L(f"| FP (오탐) | {ws_fp:,} | {pa_fp:,} | {'↓' if fp_d<0 else '↑'}{abs(fp_d):,} |")
    L(f"| TN | {ws_cm['TN']:,} | {pa_cm.get('TN',0):,} | |")
    L(f"| TP | {ws_cm['TP']:,} | {pa_cm.get('TP',0):,} | |")
    L(f"")

    # Grad-CAM 비교
    L(f"## 5. Grad-CAM 해석성 비교")
    L(f"")
    L(f"| 지표 | Whole-Slice | Patch-Based | 변화 |")
    L(f"|------|:-----------:|:-----------:|:----:|")
    iou_d = new_iou - old_iou
    L(f"| TP 평균 IoU | {old_iou:.4f} | {new_iou:.4f} | {'↑' if iou_d>0 else '↓'}{abs(iou_d):.4f} |")
    if old_iou > 0:
        L(f"| IoU 개선율 | - | - | {iou_d/old_iou*100:+.1f}% |")
    L(f"")

    L(f"**Whole-Slice Grad-CAM 문제:**")
    L(f"- 히트맵이 뇌 중심에 고정된 blob → 종양 위치 무시")
    L(f"- TP/FP 동일 패턴 → 판별력 없음")
    L(f"- 소종양(FN 92%) feature map에서 소실")
    L(f"")

    L(f"**Patch Grad-CAM 개선:**")
    L(f"- 패치 단위로 국소 영역에 집중")
    L(f"- 히트맵 재조립 → 종양 위치 추적 가능")
    L(f"- IoU {old_iou:.3f} → {new_iou:.3f}")
    L(f"")

    # 패치 단위 성능
    L(f"## 6. 패치 단위 성능 (참고)")
    L(f"")
    L(f"| 지표 | 값 |")
    L(f"|------|:---:|")
    L(f"| Accuracy | {pa_patch['accuracy']:.2f}% |")
    L(f"| Precision | {pa_patch['precision']:.2f}% |")
    L(f"| Recall | {pa_patch['recall']:.2f}% |")
    L(f"| F1-Score | {pa_patch['f1']:.2f}% |")
    L(f"| AUC-ROC | {pa_patch['auc_roc']:.4f} |")
    L(f"| 총 테스트 패치 | {pa_patch['total_patches']:,} |")
    L(f"")

    # 집계 방법 비교
    L(f"## 7. 슬라이스 집계 방법 비교")
    L(f"")
    L(f"| 방법 | Accuracy | F1 | AUC | FN | FP |")
    L(f"|:----:|:--------:|:--:|:---:|:--:|:--:|")
    for m in ["max", "mean", "count"]:
        if m in pa_eval["slice_level"]:
            r = pa_eval["slice_level"][m]
            c = r.get("cm", {})
            star = " ★" if m == best_method else ""
            L(f"| {m}{star} | {r['accuracy']:.2f}% | {r['f1']:.2f}% | {r['auc_roc']:.4f} | {c.get('FN',0)} | {c.get('FP',0)} |")
    L(f"")

    # 결론
    L(f"## 8. 결론")
    L(f"")

    # 자동 판정
    f1_improved = pa_slice.get("f1", 0) > ws_eval["f1_score"]
    iou_improved = new_iou > old_iou
    fn_reduced = pa_fn < ws_fn

    if f1_improved and iou_improved:
        verdict = "Patch-Based 모델이 성능과 해석성 모두에서 개선을 보임"
    elif iou_improved:
        verdict = "Patch-Based 모델이 해석성(IoU)에서 개선되었으나 전체 성능은 유사/하락"
    elif f1_improved:
        verdict = "Patch-Based 모델이 분류 성능은 개선되었으나 해석성은 추가 검증 필요"
    else:
        verdict = "현재 설정에서는 Whole-Slice가 우세. Patch 모델 추가 튜닝 필요"

    L(f"**종합 판정: {verdict}**")
    L(f"")
    L(f"| 관점 | 결과 |")
    L(f"|------|------|")
    L(f"| 분류 성능 (F1) | {'개선 ✅' if f1_improved else '하락/유사 ⚠️'} |")
    L(f"| 해석성 (IoU) | {'개선 ✅' if iou_improved else '하락/유사 ⚠️'} |")
    L(f"| FN 감소 | {'감소 ✅' if fn_reduced else '증가 ⚠️'} |")
    L(f"| 모델 크기 | PatchCNN이 ~120배 작음 ✅ |")
    L(f"")

    # 보고서 저장
    md_path = PROJECT_DIR / "comparison_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n  ✓ Markdown 보고서: {md_path}")

    # JSON 저장
    summary = {
        "whole_slice": {
            "accuracy": ws_eval["accuracy"],
            "f1": ws_eval["f1_score"],
            "auc": ws_eval["auc_roc"],
            "fn": ws_fn, "fp": ws_fp,
            "gradcam_iou": old_iou,
        },
        "patch_based": {
            "accuracy": pa_slice.get("accuracy", 0),
            "f1": pa_slice.get("f1", 0),
            "auc": pa_slice.get("auc_roc", 0),
            "fn": pa_fn, "fp": pa_fp,
            "gradcam_iou": new_iou,
            "aggregation": best_method,
        },
        "verdict": verdict,
    }
    json_path = LOG_DIR / "step14_comparison.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  ✓ JSON 요약: {json_path}")

    # 콘솔 요약
    print(f"\n{'═' * 60}")
    print(f"  최종 비교 요약")
    print(f"{'═' * 60}")
    for name, w, p in metrics:
        d = p - w
        arrow = "↑" if d > 0 else "↓" if d < 0 else "="
        print(f"  {name:<15s} {w:>8.2f}% → {p:>8.2f}% ({arrow}{abs(d):.2f}%)")
    print(f"  {'FN':<15s} {ws_fn:>8,d} → {pa_fn:>8,d}")
    print(f"  {'FP':<15s} {ws_fp:>8,d} → {pa_fp:>8,d}")
    print(f"  {'IoU':<15s} {old_iou:>8.4f} → {new_iou:>8.4f}")
    print(f"{'─' * 60}")
    print(f"  판정: {verdict}")
    print(f"{'═' * 60}")


if __name__ == "__main__":
    main()
