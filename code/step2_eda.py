# -*- coding: utf-8 -*-
"""
Step 2: 탐색적 데이터 분석 (EDA)
==================================
전처리된 슬라이스 데이터의 분포를 분석하고 시각화합니다.

사용법:
    python code/step2_eda.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from pathlib import Path
from collections import Counter

# Windows 콘솔 UTF-8 출력 설정
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# 한글 폰트 설정
matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False

# ─── 경로 설정 ───────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent
LABELS_CSV = PROJECT_DIR / "processed" / "labels.csv"
SLICE_DIR = PROJECT_DIR / "processed" / "slices"
FIGURES_DIR = PROJECT_DIR / "outputs" / "figures"


def plot_class_distribution(df: pd.DataFrame):
    """클래스 분포 시각화"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. 전체 클래스 분포 (바 차트)
    class_counts = df["label"].value_counts().sort_index()
    colors = ["#2ecc71", "#e74c3c"]
    labels_text = ["Negative\n(종양 없음)", "Positive\n(종양 존재)"]

    bars = axes[0].bar(labels_text, class_counts.values, color=colors, edgecolor="white", linewidth=1.5)
    for bar, count in zip(bars, class_counts.values):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 500,
            f"{count:,}\n({count/len(df)*100:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )
    axes[0].set_title("전체 슬라이스 클래스 분포", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("슬라이스 수", fontsize=12)
    axes[0].spines[["top", "right"]].set_visible(False)

    # 2. 파이 차트
    axes[1].pie(
        class_counts.values,
        labels=labels_text,
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 12},
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    axes[1].set_title("클래스 비율", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "01_class_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ 01_class_distribution.png 저장 완료")


def plot_patient_analysis(df: pd.DataFrame):
    """환자별 분석"""
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    patient_stats = df.groupby("patient_id").agg(
        total_slices=("label", "count"),
        positive_slices=("label", "sum"),
    )
    patient_stats["positive_ratio"] = (
        patient_stats["positive_slices"] / patient_stats["total_slices"]
    )

    # 1. 환자당 전체 슬라이스 수 분포
    axes[0].hist(
        patient_stats["total_slices"],
        bins=30,
        color="#3498db",
        edgecolor="white",
        alpha=0.8,
    )
    axes[0].axvline(
        patient_stats["total_slices"].median(),
        color="#e74c3c",
        linestyle="--",
        linewidth=2,
        label=f"중앙값: {patient_stats['total_slices'].median():.0f}",
    )
    axes[0].set_title("환자당 슬라이스 수 분포", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("슬라이스 수", fontsize=12)
    axes[0].set_ylabel("환자 수", fontsize=12)
    axes[0].legend(fontsize=11)
    axes[0].spines[["top", "right"]].set_visible(False)

    # 2. 환자당 종양 슬라이스 수 분포
    axes[1].hist(
        patient_stats["positive_slices"],
        bins=30,
        color="#e74c3c",
        edgecolor="white",
        alpha=0.8,
    )
    axes[1].axvline(
        patient_stats["positive_slices"].median(),
        color="#2c3e50",
        linestyle="--",
        linewidth=2,
        label=f"중앙값: {patient_stats['positive_slices'].median():.0f}",
    )
    axes[1].set_title("환자당 종양 슬라이스 수 분포", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("종양 슬라이스 수", fontsize=12)
    axes[1].set_ylabel("환자 수", fontsize=12)
    axes[1].legend(fontsize=11)
    axes[1].spines[["top", "right"]].set_visible(False)

    # 3. 환자당 종양 비율 분포
    axes[2].hist(
        patient_stats["positive_ratio"],
        bins=30,
        color="#9b59b6",
        edgecolor="white",
        alpha=0.8,
    )
    axes[2].axvline(
        patient_stats["positive_ratio"].median(),
        color="#e74c3c",
        linestyle="--",
        linewidth=2,
        label=f"중앙값: {patient_stats['positive_ratio'].median():.2f}",
    )
    axes[2].set_title("환자당 종양 슬라이스 비율 분포", fontsize=14, fontweight="bold")
    axes[2].set_xlabel("종양 비율", fontsize=12)
    axes[2].set_ylabel("환자 수", fontsize=12)
    axes[2].legend(fontsize=11)
    axes[2].spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "02_patient_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ 02_patient_analysis.png 저장 완료")


def plot_slice_position_distribution(df: pd.DataFrame):
    """슬라이스 위치별 종양 분포"""
    fig, ax = plt.subplots(figsize=(14, 5))

    # 슬라이스 위치별 종양 비율
    slice_stats = df.groupby("slice_idx").agg(
        total=("label", "count"),
        positive=("label", "sum"),
    )
    slice_stats["positive_ratio"] = slice_stats["positive"] / slice_stats["total"]

    ax.fill_between(
        slice_stats.index,
        slice_stats["positive_ratio"],
        alpha=0.3,
        color="#e74c3c",
    )
    ax.plot(
        slice_stats.index,
        slice_stats["positive_ratio"],
        color="#e74c3c",
        linewidth=2,
        label="종양 존재 비율",
    )
    ax.set_title(
        "슬라이스 위치별 종양 존재 비율 (Axial)", fontsize=14, fontweight="bold"
    )
    ax.set_xlabel("슬라이스 인덱스 (Z축)", fontsize=12)
    ax.set_ylabel("종양 존재 비율", fontsize=12)
    ax.legend(fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "03_slice_position_distribution.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()
    print("  ✓ 03_slice_position_distribution.png 저장 완료")


def plot_sample_slices(df: pd.DataFrame):
    """샘플 슬라이스 시각화 (Positive vs Negative)"""
    import cv2

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle(
        "샘플 슬라이스 예시", fontsize=16, fontweight="bold", y=1.02
    )

    def imread_unicode(filepath):
        """한글 경로를 지원하는 이미지 로드 함수"""
        img_array = np.fromfile(str(filepath), dtype=np.uint8)
        return cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)

    # Positive 샘플 5개
    positive_samples = df[df["label"] == 1].sample(n=5, random_state=42)
    for idx, (_, row) in enumerate(positive_samples.iterrows()):
        img = imread_unicode(SLICE_DIR / row["filename"])
        if img is not None:
            axes[0, idx].imshow(img, cmap="gray")
            axes[0, idx].set_title(
                f"Positive\n{row['patient_id']}\nz={row['slice_idx']}",
                fontsize=9,
                color="#e74c3c",
                fontweight="bold",
            )
        axes[0, idx].axis("off")

    # Negative 샘플 5개
    negative_samples = df[df["label"] == 0].sample(n=5, random_state=42)
    for idx, (_, row) in enumerate(negative_samples.iterrows()):
        img = imread_unicode(SLICE_DIR / row["filename"])
        if img is not None:
            axes[1, idx].imshow(img, cmap="gray")
            axes[1, idx].set_title(
                f"Negative\n{row['patient_id']}\nz={row['slice_idx']}",
                fontsize=9,
                color="#2ecc71",
                fontweight="bold",
            )
        axes[1, idx].axis("off")

    # 행 라벨
    axes[0, 0].set_ylabel("Positive\n(종양 존재)", fontsize=12, fontweight="bold", color="#e74c3c")
    axes[1, 0].set_ylabel("Negative\n(종양 없음)", fontsize=12, fontweight="bold", color="#2ecc71")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "04_sample_slices.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ 04_sample_slices.png 저장 완료")


def plot_brain_fraction_distribution(df: pd.DataFrame):
    """뇌 영역 비율 분포"""
    fig, ax = plt.subplots(figsize=(10, 5))

    df["brain_fraction"] = df["brain_fraction"].astype(float)

    for label, color, name in [(0, "#2ecc71", "Negative"), (1, "#e74c3c", "Positive")]:
        subset = df[df["label"] == label]["brain_fraction"]
        ax.hist(subset, bins=50, alpha=0.6, color=color, label=name, edgecolor="white")

    ax.set_title("뇌 영역 비율 분포 (라벨별)", fontsize=14, fontweight="bold")
    ax.set_xlabel("뇌 영역 비율", fontsize=12)
    ax.set_ylabel("슬라이스 수", fontsize=12)
    ax.legend(fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "05_brain_fraction.png", dpi=150, bbox_inches="tight"
    )
    plt.close()
    print("  ✓ 05_brain_fraction.png 저장 완료")


def main():
    print("=" * 60)
    print("  Step 2: 탐색적 데이터 분석 (EDA)")
    print("=" * 60)

    # 라벨 CSV 확인
    if not LABELS_CSV.exists():
        print(f"\n[ERROR] 라벨 파일이 없습니다: {LABELS_CSV}")
        print("  step1_preprocess.py를 먼저 실행하세요.")
        return

    # 출력 디렉토리 생성
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 데이터 로드
    df = pd.read_csv(LABELS_CSV)
    print(f"\n  총 슬라이스 수: {len(df):,}")
    print(f"  환자 수: {df['patient_id'].nunique()}")
    print(f"  Positive: {(df['label']==1).sum():,} ({(df['label']==1).mean()*100:.1f}%)")
    print(f"  Negative: {(df['label']==0).sum():,} ({(df['label']==0).mean()*100:.1f}%)")

    # 기본 통계
    print(f"\n{'─' * 60}")
    print("  환자별 통계:")
    patient_stats = df.groupby("patient_id")["label"].agg(["count", "sum", "mean"])
    patient_stats.columns = ["total", "positive", "positive_ratio"]
    print(f"    슬라이스/환자 - 평균: {patient_stats['total'].mean():.1f}, "
          f"중앙값: {patient_stats['total'].median():.0f}, "
          f"범위: [{patient_stats['total'].min()}, {patient_stats['total'].max()}]")
    print(f"    종양슬라이스/환자 - 평균: {patient_stats['positive'].mean():.1f}, "
          f"중앙값: {patient_stats['positive'].median():.0f}")
    print(f"    종양비율/환자 - 평균: {patient_stats['positive_ratio'].mean():.3f}")

    # 시각화
    print(f"\n{'─' * 60}")
    print("  시각화 생성 중...")
    print(f"{'─' * 60}")

    plot_class_distribution(df)
    plot_patient_analysis(df)
    plot_slice_position_distribution(df)
    plot_sample_slices(df)
    plot_brain_fraction_distribution(df)

    print(f"\n{'=' * 60}")
    print(f"  EDA 완료! 시각화 저장 경로: {FIGURES_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
