import streamlit as st
import pandas as pd
import numpy as np
import math
import os

# =========================================================
# 基本設定
# =========================================================
st.set_page_config(
    page_title="大底初動AIスクリーナー",
    page_icon="🌱",
    layout="wide"
)

st.title("🌱 大底初動AIスクリーナー")
st.caption(
    "52週安値圏 × 月足反転 × 週足MACD反転 × 日足初動を中心に、"
    "大底からの上昇初動候補を100点満点で評価します。"
)


# =========================================================
# データ読み込み
# =========================================================
@st.cache_data(ttl=3600 * 12)
def load_data():
    try:
        # ---------------------------------------------
        # 株価
        # ---------------------------------------------
        prices_df = pd.read_csv(
            "prices.csv",
            index_col=0,
            parse_dates=True
        )

        prices_df.index = pd.to_datetime(prices_df.index)
        prices_df = prices_df.sort_index()

        # ---------------------------------------------
        # 銘柄名
        # ---------------------------------------------
        names_df = pd.read_csv(
            "names.csv",
            dtype={"コード": str}
        )

        name_dict = dict(
            zip(
                names_df["コード"].astype(str),
                names_df["会社名"]
            )
        )

        # ---------------------------------------------
        # 出来高（任意）
        # volume.csv
        #
        # 日付 × 銘柄コード
        # prices.csvと同じ構造を想定
        # ---------------------------------------------
        volume_df = None

        if os.path.exists("volume.csv"):
            volume_df = pd.read_csv(
                "volume.csv",
                index_col=0,
                parse_dates=True
            )

            volume_df.index = pd.to_datetime(volume_df.index)
            volume_df = volume_df.sort_index()

        # ---------------------------------------------
        # ファンダメンタルズ（任意）
        # fundamentals.csv
        #
        # 想定列：
        # コード
        # PER
        # PBR
        # EPS成長率
        # 自社株買い
        # 業種平均PER（任意）
        # ---------------------------------------------
        fundamentals_df = None

        if os.path.exists("fundamentals.csv"):
            fundamentals_df = pd.read_csv(
                "fundamentals.csv",
                dtype={"コード": str}
            )

            fundamentals_df["コード"] = (
                fundamentals_df["コード"]
                .astype(str)
                .str.replace(".0", "", regex=False)
                .str.replace(".T", "", regex=False)
            )

        # ---------------------------------------------
        # 需給（任意）
        # supply.csv
        #
        # 想定列：
        # コード
        # 信用買い残変化率
        # 空売り比率
        # 浮動株比率
        # 信用倍率
        # ---------------------------------------------
        supply_df = None

        if os.path.exists("supply.csv"):
            supply_df = pd.read_csv(
                "supply.csv",
                dtype={"コード": str}
            )

            supply_df["コード"] = (
                supply_df["コード"]
                .astype(str)
                .str.replace(".0", "", regex=False)
                .str.replace(".T", "", regex=False)
            )

        return (
            prices_df,
            name_dict,
            volume_df,
            fundamentals_df,
            supply_df
        )

    except Exception as e:
        st.error(
            "⚠️ データの読み込みに失敗しました。\n\n"
            f"詳細: {e}"
        )

        return None, None, None, None, None


# =========================================================
# テクニカル計算
# =========================================================
def calc_macd(series, fast=12, slow=26, signal=9):

    ema_fast = series.ewm(
        span=fast,
        adjust=False
    ).mean()

    ema_slow = series.ewm(
        span=slow,
        adjust=False
    ).mean()

    macd = ema_fast - ema_slow

    macd_signal = macd.ewm(
        span=signal,
        adjust=False
    ).mean()

    return macd, macd_signal


def calc_rsi(series, period=14):

    delta = series.diff()

    gain = (
        delta.where(delta > 0, 0)
        .rolling(period)
        .mean()
    )

    loss = (
        -delta.where(delta < 0, 0)
        .rolling(period)
        .mean()
    )

    rs = gain / loss

    rsi = 100 - (100 / (1 + rs))

    return rsi


def safe_float(value, default=np.nan):
    try:
        if pd.isna(value):
            return default

        return float(value)

    except Exception:
        return default


# =========================================================
# データ読み込み
# =========================================================
(
    prices_df,
    info_dict,
    volume_df,
    fundamentals_df,
    supply_df
) = load_data()


# =========================================================
# サイドバー
# =========================================================
with st.sidebar:

    st.header("⚙️ スクリーニング設定")

    st.write(
        "現在の設定は「大底 → 反転初動」を狙う仕様です。"
    )

    min_score = st.slider(
        "最低総合スコア",
        min_value=40,
        max_value=90,
        value=60,
        step=5
    )

    max_results = st.slider(
        "表示銘柄数",
        min_value=10,
        max_value=100,
        value=30,
        step=10
    )

    st.divider()

    st.subheader("🚫 ハード除外")

    exclude_month_decline = st.checkbox(
        "月足MACDが3か月連続悪化 → 除外",
        value=True
    )

    exclude_week_decline = st.checkbox(
        "週足ヒストグラムが2週連続悪化 → 除外",
        value=True
    )

    exclude_overheat = st.checkbox(
        "RSI 70超 → 除外",
        value=True
    )

    st.divider()

    st.subheader("📁 オプションデータ")

    st.write(
        "以下のCSVが存在する場合、自動的にスコアへ反映します。"
    )

    volume_status = (
        "✅ volume.csv"
        if volume_df is not None
        else "⚪ volume.csvなし"
    )

    fundamentals_status = (
        "✅ fundamentals.csv"
        if fundamentals_df is not None
        else "⚪ fundamentals.csvなし"
    )

    supply_status = (
        "✅ supply.csv"
        if supply_df is not None
        else "⚪ supply.csvなし"
    )

    st.write(volume_status)
    st.write(fundamentals_status)
    st.write(supply_status)

    run_btn = st.button(
        "🚀 スクリーニング実行",
        type="primary",
        use_container_width=True
    )


# =========================================================
# スクリーニング
# =========================================================
if run_btn and prices_df is not None:

    with st.spinner(
        "🌱 大底初動AI解析中..."
    ):

        raw_candidates = []

        total_symbols = len(prices_df.columns)
        screened_symbols = 0
        excluded_symbols = 0

        # =================================================
        # 全銘柄ループ
        # =================================================
        for t in prices_df.columns:

            try:

                # -----------------------------------------
                # 株価データ
                # -----------------------------------------
                close_d = (
                    prices_df[t]
                    .dropna()
                    .astype(float)
                    .sort_index()
                )

                if len(close_d) < 500:
                    continue

                curr_price = float(
                    close_d.iloc[-1]
                )

                # =================================================
                # ① 52週位置
                # 最大15点
                # =================================================
                recent_245d = close_d.tail(245)

                high_52w = float(
                    recent_245d.max()
                )

                low_52w = float(
                    recent_245d.min()
                )

                if high_52w > low_52w:

                    position_ratio = (
                        curr_price - low_52w
                    ) / (
                        high_52w - low_52w
                    )

                else:

                    position_ratio = 1.0

                if position_ratio <= 0.10:

                    score_52w = 15

                elif position_ratio <= 0.20:

                    score_52w = 12

                elif position_ratio <= 0.30:

                    score_52w = 8

                elif position_ratio <= 0.40:

                    score_52w = 4

                else:

                    score_52w = 0

                distance_from_52w_low = (
                    (curr_price / low_52w) - 1
                ) * 100 if low_52w > 0 else np.nan


                # =================================================
                # ② 月足
                # 最大15点
                # =================================================
                try:

                    df_m = (
                        close_d
                        .resample("ME")
                        .last()
                        .dropna()
                    )

                except Exception:

                    df_m = (
                        close_d
                        .resample("M")
                        .last()
                        .dropna()
                    )

                if len(df_m) < 30:
                    continue

                macd_m, sig_m = calc_macd(df_m)

                ma25_m = (
                    df_m
                    .rolling(25)
                    .mean()
                )

                curr_m_macd = float(
                    macd_m.iloc[-1]
                )

                prev_m_macd = float(
                    macd_m.iloc[-2]
                )

                prev2_m_macd = float(
                    macd_m.iloc[-3]
                )

                curr_m_sig = float(
                    sig_m.iloc[-1]
                )

                curr_m_ma25 = float(
                    ma25_m.iloc[-1]
                )

                prev_m_ma25 = float(
                    ma25_m.iloc[-2]
                )

                # 月足MACD改善
                monthly_macd_improving = (
                    curr_m_macd > prev_m_macd
                )

                # 月足MACDシグナル上
                monthly_macd_above_signal = (
                    curr_m_macd > curr_m_sig
                )

                # 月足25MA傾き
                monthly_ma25_up = (
                    curr_m_ma25 > prev_m_ma25
                )

                score_monthly = 0

                if curr_m_macd < 0 and monthly_macd_improving:
                    score_monthly += 7

                elif curr_m_macd < 0:
                    score_monthly += 3

                if monthly_macd_above_signal:
                    score_monthly += 4

                if monthly_ma25_up:
                    score_monthly += 4

                score_monthly = min(
                    score_monthly,
                    15
                )

                # -----------------------------------------
                # 月足MACD 3か月連続悪化
                # -----------------------------------------
                monthly_3_consecutive_decline = (
                    curr_m_macd < prev_m_macd
                    and
                    prev_m_macd < prev2_m_macd
                )

                if (
                    exclude_month_decline
                    and monthly_3_consecutive_decline
                ):
                    excluded_symbols += 1
                    continue


                # =================================================
                # ③ 週足MACD
                # 最大40点
                # =================================================
                df_w = (
                    close_d
                    .resample("W-FRI")
                    .last()
                    .dropna()
                )

                if len(df_w) < 120:
                    continue

                macd_w, sig_w = calc_macd(df_w)

                hist_w = (
                    macd_w - sig_w
                )

                curr_w_macd = float(
                    macd_w.iloc[-1]
                )

                curr_w_sig = float(
                    sig_w.iloc[-1]
                )

                curr_hist = float(
                    hist_w.iloc[-1]
                )

                prev_hist = float(
                    hist_w.iloc[-2]
                )

                hist_2w = float(
                    hist_w.iloc[-3]
                )

                # -----------------------------------------
                # 週足MACD深度
                # 過去104週でどれくらい深いか
                # 最大10点
                # -----------------------------------------
                macd_104 = (
                    macd_w
                    .tail(104)
                    .dropna()
                )

                current_rank_percentile = (
                    macd_104
                    .rank(pct=True)
                    .iloc[-1]
                )

                if current_rank_percentile <= 0.10:

                    score_macd_depth = 10

                elif current_rank_percentile <= 0.20:

                    score_macd_depth = 8

                elif current_rank_percentile <= 0.30:

                    score_macd_depth = 5

                elif current_rank_percentile <= 0.50:

                    score_macd_depth = 2

                else:

                    score_macd_depth = 0


                # -----------------------------------------
                # 週足MACD反転力
                # 最大20点
                # -----------------------------------------
                macd_changes = (
                    macd_w.diff()
                )

                consecutive_improvement = 0

                for i in range(1, 5):

                    if (
                        len(macd_changes) > i
                        and
                        macd_changes.iloc[-i] > 0
                    ):
                        consecutive_improvement += 1
                    else:
                        break

                if consecutive_improvement >= 4:

                    score_macd_reversal = 20

                elif consecutive_improvement == 3:

                    score_macd_reversal = 17

                elif consecutive_improvement == 2:

                    score_macd_reversal = 13

                elif consecutive_improvement == 1:

                    score_macd_reversal = 7

                else:

                    score_macd_reversal = 0

                # -----------------------------------------
                # ヒストグラム改善を追加評価
                # -----------------------------------------
                histogram_improving = (
                    curr_hist > prev_hist
                )

                if histogram_improving:

                    score_macd_reversal += 2

                # 上限20点
                score_macd_reversal = min(
                    score_macd_reversal,
                    20
                )


                # -----------------------------------------
                # 週足GC検出
                # -----------------------------------------
                w_gc_weeks_ago = -1

                for i in range(1, 9):

                    if (
                        macd_w.iloc[-i]
                        >
                        sig_w.iloc[-i]
                        and
                        macd_w.iloc[-(i + 1)]
                        <=
                        sig_w.iloc[-(i + 1)]
                    ):

                        w_gc_weeks_ago = i - 1

                        break


                # -----------------------------------------
                # GC距離
                # 最大10点
                # -----------------------------------------
                gc_distance_pct = (
                    (
                        curr_w_sig
                        -
                        curr_w_macd
                    )
                    / curr_price
                ) * 100

                score_gc = 0

                gc_status = "GCなし"

                if w_gc_weeks_ago == 0:

                    score_gc = 10
                    gc_status = "今週GC🔥"

                elif 1 <= w_gc_weeks_ago <= 2:

                    if w_gc_weeks_ago == 1:
                        score_gc = 9
                    else:
                        score_gc = 7

                    gc_status = (
                        f"{w_gc_weeks_ago}週前GC✨"
                    )

                elif 3 <= w_gc_weeks_ago <= 4:

                    score_gc = 4

                    gc_status = (
                        f"{w_gc_weeks_ago}週前GC"
                    )

                else:

                    # GC前の場合
                    if (
                        curr_w_macd < curr_w_sig
                        and
                        curr_w_macd > float(
                            macd_w.iloc[-2]
                        )
                    ):

                        if gc_distance_pct <= 0.20:

                            score_gc = 10
                            gc_status = "GC超直前🔥"

                        elif gc_distance_pct <= 0.50:

                            score_gc = 8
                            gc_status = "GC直前🔥"

                        elif gc_distance_pct <= 1.00:

                            score_gc = 6
                            gc_status = "GC接近"

                        elif gc_distance_pct <= 2.00:

                            score_gc = 3
                            gc_status = "GC待ち"

                        else:

                            score_gc = 0
                            gc_status = "GCまで距離あり"

                    else:

                        score_gc = 0
                        gc_status = "GCなし"

                # -----------------------------------------
                # 週足落ちるナイフ判定
                # -----------------------------------------
                weekly_hist_2_consecutive_decline = (
                    curr_hist < prev_hist
                    and
                    prev_hist < hist_2w
                )

                if (
                    exclude_week_decline
                    and
                    weekly_hist_2_consecutive_decline
                ):
                    excluded_symbols += 1
                    continue


                # =================================================
                # ④ 日足25MA
                # 最大10点
                # =================================================
                ma25_d = (
                    close_d
                    .rolling(25)
                    .mean()
                )

                curr_ma25 = float(
                    ma25_d.iloc[-1]
                )

                prev_ma25 = float(
                    ma25_d.iloc[-2]
                )

                price_above_ma25 = (
                    curr_price > curr_ma25
                )

                ma25_slope_up = (
                    curr_ma25 > prev_ma25
                )

                score_d_trend = 0

                d_trend_status = "25日線下"

                # 25MA突破タイミング
                breakout_days_ago = -1

                lookback_breakout = min(
                    10,
                    len(close_d) - 2
                )

                for j in range(
                    1,
                    lookback_breakout + 1
                ):

                    price_today = float(
                        close_d.iloc[-j]
                    )

                    price_before = float(
                        close_d.iloc[-j - 1]
                    )

                    ma_today = float(
                        ma25_d.iloc[-j]
                    )

                    ma_before = float(
                        ma25_d.iloc[-j - 1]
                    )

                    if (
                        price_today > ma_today
                        and
                        price_before <= ma_before
                    ):

                        breakout_days_ago = j - 1
                        break

                if breakout_days_ago == 0:

                    score_d_trend = 10
                    d_trend_status = "本日25MA突破🔥"

                elif 1 <= breakout_days_ago <= 3:

                    score_d_trend = 8
                    d_trend_status = (
                        f"{breakout_days_ago}日前25MA突破"
                    )

                elif 4 <= breakout_days_ago <= 7:

                    score_d_trend = 5
                    d_trend_status = (
                        f"{breakout_days_ago}日前25MA突破"
                    )

                elif price_above_ma25:

                    score_d_trend = 3
                    d_trend_status = "25MA上"

                else:

                    score_d_trend = 0
                    d_trend_status = "25MA下"

                # MA25上向きなら少し補強
                if (
                    ma25_slope_up
                    and
                    price_above_ma25
                ):

                    score_d_trend += 1

                score_d_trend = min(
                    score_d_trend,
                    10
                )


                # =================================================
                # ⑤ RSI
                # 最大5点
                # =================================================
                rsi_series = calc_rsi(
                    close_d
                )

                curr_rsi = float(
                    rsi_series.iloc[-1]
                )

                if 40 <= curr_rsi <= 50:

                    score_rsi = 5

                elif 50 < curr_rsi <= 55:

                    score_rsi = 4

                elif 55 < curr_rsi <= 65:

                    score_rsi = 2

                elif 30 <= curr_rsi < 40:

                    score_rsi = 1

                elif curr_rsi < 30:

                    score_rsi = 0

                else:

                    score_rsi = -5

                # RSI70超ハード除外
                if (
                    exclude_overheat
                    and
                    curr_rsi > 70
                ):
                    excluded_symbols += 1
                    continue


                # =================================================
                # ⑥ 出来高
                # 最大5点
                # =================================================
                score_volume = 0
                volume_ratio = np.nan

                if (
                    volume_df is not None
                    and
                    t in volume_df.columns
                ):

                    volume_series = (
                        volume_df[t]
                        .dropna()
                        .astype(float)
                        .sort_index()
                    )

                    if len(volume_series) >= 25:

                        current_volume = float(
                            volume_series.iloc[-1]
                        )

                        avg_volume_20 = float(
                            volume_series
                            .tail(20)
                            .mean()
                        )

                        if avg_volume_20 > 0:

                            volume_ratio = (
                                current_volume
                                /
                                avg_volume_20
                            )

                            if volume_ratio >= 2.0:

                                score_volume = 5

                            elif volume_ratio >= 1.5:

                                score_volume = 4

                            elif volume_ratio >= 1.2:

                                score_volume = 2

                            else:

                                score_volume = 0


                # =================================================
                # ⑦ ファンダメンタルズ
                # 最大5点
                # =================================================
                score_fundamentals = 0

                per_value = np.nan
                pbr_value = np.nan
                eps_growth = np.nan
                buyback_value = None

                code = (
                    t
                    .replace(".T", "")
                    .replace(".JP", "")
                )

                if fundamentals_df is not None:

                    fund_rows = fundamentals_df[
                        fundamentals_df["コード"] == code
                    ]

                    if not fund_rows.empty:

                        fund = fund_rows.iloc[0]

                        per_value = safe_float(
                            fund.get("PER", np.nan)
                        )

                        pbr_value = safe_float(
                            fund.get("PBR", np.nan)
                        )

                        eps_growth = safe_float(
                            fund.get("EPS成長率", np.nan)
                        )

                        buyback_value = fund.get(
                            "自社株買い",
                            None
                        )

                        industry_per = safe_float(
                            fund.get(
                                "業種平均PER",
                                np.nan
                            )
                        )

                        # EPS成長
                        if (
                            not np.isnan(eps_growth)
                            and
                            eps_growth > 0
                        ):

                            score_fundamentals += 2

                        # PER
                        if (
                            not np.isnan(per_value)
                            and
                            per_value > 0
                        ):

                            if (
                                not np.isnan(industry_per)
                                and
                                industry_per > 0
                                and
                                per_value < industry_per
                            ):

                                score_fundamentals += 1

                            elif per_value <= 15:

                                score_fundamentals += 1

                        # PBR
                        if (
                            not np.isnan(pbr_value)
                            and
                            pbr_value <= 1.0
                        ):

                            score_fundamentals += 1

                        # 自社株買い
                        if buyback_value is not None:

                            buyback_text = str(
                                buyback_value
                            ).lower()

                            if buyback_text in [
                                "1",
                                "true",
                                "yes",
                                "あり",
                                "有",
                                "実施"
                            ]:

                                score_fundamentals += 1

                score_fundamentals = min(
                    score_fundamentals,
                    5
                )


                # =================================================
                # ⑧ 需給
                # 最大5点
                # =================================================
                score_supply = 0

                credit_change = np.nan
                short_ratio = np.nan
                float_ratio = np.nan
                credit_ratio = np.nan

                if supply_df is not None:

                    supply_rows = supply_df[
                        supply_df["コード"] == code
                    ]

                    if not supply_rows.empty:

                        supply = supply_rows.iloc[0]

                        credit_change = safe_float(
                            supply.get(
                                "信用買い残変化率",
                                np.nan
                            )
                        )

                        short_ratio = safe_float(
                            supply.get(
                                "空売り比率",
                                np.nan
                            )
                        )

                        float_ratio = safe_float(
                            supply.get(
                                "浮動株比率",
                                np.nan
                            )
                        )

                        credit_ratio = safe_float(
                            supply.get(
                                "信用倍率",
                                np.nan
                            )
                        )

                        # 信用買い残減少
                        if (
                            not np.isnan(credit_change)
                            and
                            credit_change < 0
                        ):

                            score_supply += 2

                        # 空売り
                        if (
                            not np.isnan(short_ratio)
                            and
                            short_ratio >= 3
                        ):

                            score_supply += 1

                        # 浮動株
                        if (
                            not np.isnan(float_ratio)
                            and
                            float_ratio <= 30
                        ):

                            score_supply += 1

                        # 信用倍率
                        if (
                            not np.isnan(credit_ratio)
                            and
                            credit_ratio < 3
                        ):

                            score_supply += 1

                score_supply = min(
                    score_supply,
                    5
                )


                # =================================================
                # 総合スコア
                # =================================================
                total_score = (
                    score_52w
                    +
                    score_monthly
                    +
                    score_macd_depth
                    +
                    score_macd_reversal
                    +
                    score_gc
                    +
                    score_d_trend
                    +
                    score_rsi
                    +
                    score_volume
                    +
                    score_fundamentals
                    +
                    score_supply
                )

                total_score = round(
                    float(total_score),
                    1
                )


                # =================================================
                # Stage判定
                # =================================================
                if (
                    (
                        gc_status == "今週GC🔥"
                        or
                        gc_status == "GC超直前🔥"
                    )
                    and
                    price_above_ma25
                    and
                    (
                        volume_ratio >= 1.5
                        if not np.isnan(volume_ratio)
                        else True
                    )
                ):

                    stage = "🚀 Stage 4：上昇確認"

                elif (
                    "GC" in gc_status
                    and
                    (
                        "直前" in gc_status
                        or
                        "接近" in gc_status
                        or
                        "今週" in gc_status
                    )
                ):

                    stage = "🔥 Stage 3：GC直前～GC"

                elif (
                    consecutive_improvement >= 2
                    and
                    monthly_macd_improving
                ):

                    stage = "🌱 Stage 2：反転初動"

                else:

                    stage = "🌑 Stage 1：大底形成"


                # =================================================
                # スコア評価
                # =================================================
                if total_score >= 80:

                    rating = "⭐⭐⭐ 最注目"

                elif total_score >= 70:

                    rating = "⭐⭐ 有力監視"

                elif total_score >= 60:

                    rating = "⭐ 候補"

                else:

                    rating = "監視"


                # =================================================
                # データ充足率
                # =================================================
                available_optional = 0

                if volume_df is not None:
                    available_optional += 5

                if fundamentals_df is not None:
                    available_optional += 5

                if supply_df is not None:
                    available_optional += 5

                if available_optional == 0:

                    data_coverage = 85

                else:

                    data_coverage = (
                        (
                            85
                            +
                            available_optional
                        )
                        / 100
                    ) * 100

                screened_symbols += 1


                # =================================================
                # 結果保存
                # =================================================
                raw_candidates.append({

                    "コード":
                        code,

                    "会社名":
                        info_dict.get(
                            code,
                            code
                        ),

                    "株価":
                        f"¥{curr_price:,.0f}",

                    "総合スコア":
                        total_score,

                    "評価":
                        rating,

                    "Stage":
                        stage,

                    # --------
                    # スコア内訳
                    # --------
                    "52週位置":
                        score_52w,

                    "月足トレンド":
                        score_monthly,

                    "週足MACD深度":
                        score_macd_depth,

                    "週足MACD反転":
                        score_macd_reversal,

                    # ★修正: "GC距離" と重複していたキーを
                    # "GCスコア"（数値・0〜10点）に変更。
                    # 下の「GC距離」は距離を示す文字列(%)として別キーで保持する。
                    "GCスコア":
                        score_gc,

                    "日足25MA":
                        score_d_trend,

                    "RSI":
                        score_rsi,

                    "出来高":
                        score_volume,

                    "ファンダ":
                        score_fundamentals,

                    "需給":
                        score_supply,

                    # --------
                    # 状態
                    # --------
                    "月足サイン":
                        (
                            "MACD改善"
                            if monthly_macd_improving
                            else "MACD悪化"
                        ),

                    "週足MACD":
                        gc_status,

                    "日足トレンド":
                        d_trend_status,

                    "日足RSI":
                        round(
                            curr_rsi,
                            1
                        ),

                    "52週安値から":
                        (
                            f"{distance_from_52w_low:.1f}%"
                            if not np.isnan(
                                distance_from_52w_low
                            )
                            else "-"
                        ),

                    "出来高倍率":
                        (
                            f"{volume_ratio:.2f}倍"
                            if not np.isnan(
                                volume_ratio
                            )
                            else "-"
                        ),

                    # ★修正: score_gc と重複していたキー。
                    # こちらはGC(ゴールデンクロス)までの距離(%)を示す文字列専用のキー。
                    "GC距離":
                        (
                            f"{gc_distance_pct:.3f}%"
                            if not np.isnan(
                                gc_distance_pct
                            )
                            else "-"
                        ),

                    "PER":
                        (
                            round(per_value, 1)
                            if not np.isnan(per_value)
                            else "-"
                        ),

                    "PBR":
                        (
                            round(pbr_value, 2)
                            if not np.isnan(pbr_value)
                            else "-"
                        ),

                    "EPS成長率":
                        (
                            f"{eps_growth:.1f}%"
                            if not np.isnan(
                                eps_growth
                            )
                            else "-"
                        ),

                    "データ充足":
                        f"{data_coverage:.0f}%"
                })

            except Exception:
                # 個別銘柄で異常データがあっても
                # 全体処理を止めない
                continue


        # =========================================================
        # スコア順
        # =========================================================
        raw_candidates.sort(
            key=lambda x: x["総合スコア"],
            reverse=True
        )

        # 最低点以上のみ
        filtered_candidates = [
            x
            for x in raw_candidates
            if x["総合スコア"] >= min_score
        ]

        top_candidates = (
            filtered_candidates[:max_results]
        )


        # =========================================================
        # 結果
        # =========================================================
        if top_candidates:

            st.success(
                f"✨ スクリーニング完了："
                f"{len(top_candidates)}銘柄を抽出しました。"
            )

            # ---------------------------------------------
            # サマリー
            # ---------------------------------------------
            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "対象銘柄",
                f"{total_symbols:,}"
            )

            col2.metric(
                "スコア条件通過",
                f"{len(filtered_candidates):,}"
            )

            col3.metric(
                "ハード除外",
                f"{excluded_symbols:,}"
            )

            col4.metric(
                "最高スコア",
                f"{top_candidates[0]['総合スコア']:.0f}"
            )


            st.divider()

            # ---------------------------------------------
            # メイン一覧
            # ---------------------------------------------
            display_columns = [
                "コード",
                "会社名",
                "株価",
                "総合スコア",
                "評価",
                "Stage",
                "52週位置",
                "月足トレンド",
                "週足MACD深度",
                "週足MACD反転",
                "GCスコア",
                "日足25MA",
                "RSI",
                "出来高",
                "ファンダ",
                "需給",
                "週足MACD",
                "日足トレンド",
                "日足RSI",
                "出来高倍率"
            ]

            res_df = pd.DataFrame(
                top_candidates
            )

            res_df = (
                res_df[
                    display_columns
                ]
                .set_index("コード")
            )


            # ---------------------------------------------
            # スコアを色付け
            # ---------------------------------------------
            styled_df = (
                res_df.style
                .background_gradient(
                    subset=["総合スコア"],
                    cmap="Oranges"
                )
            )

            st.dataframe(
                styled_df,
                use_container_width=True,
                height=700
            )


            # =================================================
            # 上位銘柄の詳細
            # =================================================
            st.divider()
            st.subheader(
                "🔎 上位銘柄のスコア詳細"
            )

            for candidate in top_candidates[:10]:

                with st.expander(
                    f"{candidate['コード']} "
                    f"{candidate['会社名']} "
                    f"｜ {candidate['総合スコア']:.0f}点 "
                    f"｜ {candidate['Stage']}"
                ):

                    col1, col2, col3, col4 = st.columns(4)

                    col1.metric(
                        "総合",
                        f"{candidate['総合スコア']:.0f} / 100"
                    )

                    col2.metric(
                        "52週位置",
                        f"{candidate['52週位置']} / 15"
                    )

                    col3.metric(
                        "週足MACD",
                        f"{candidate['週足MACD反転']} / 20"
                    )

                    col4.metric(
                        "GC",
                        f"{candidate['GCスコア']} / 10"
                    )


                    detail_df = pd.DataFrame({

                        "評価項目": [
                            "52週位置",
                            "月足トレンド",
                            "週足MACD深度",
                            "週足MACD反転",
                            "GCスコア",
                            "日足25MA",
                            "RSI",
                            "出来高",
                            "ファンダ",
                            "需給"
                        ],

                        "得点": [
                            candidate["52週位置"],
                            candidate["月足トレンド"],
                            candidate["週足MACD深度"],
                            candidate["週足MACD反転"],
                            candidate["GCスコア"],
                            candidate["日足25MA"],
                            candidate["RSI"],
                            candidate["出来高"],
                            candidate["ファンダ"],
                            candidate["需給"]
                        ],

                        "最大点": [
                            15,
                            15,
                            10,
                            20,
                            10,
                            10,
                            5,
                            5,
                            5,
                            5
                        ]
                    })

                    detail_df["達成率"] = (
                        detail_df["得点"]
                        /
                        detail_df["最大点"]
                        * 100
                    ).round(0).astype(int).astype(str) + "%"

                    st.dataframe(
                        detail_df,
                        use_container_width=True,
                        hide_index=True
                    )

                    st.write(
                        f"**週足MACD：** "
                        f"{candidate['週足MACD']}"
                    )

                    st.write(
                        f"**月足：** "
                        f"{candidate['月足サイン']}"
                    )

                    st.write(
                        f"**日足：** "
                        f"{candidate['日足トレンド']}"
                    )

                    st.write(
                        f"**RSI：** "
                        f"{candidate['日足RSI']}"
                    )

                    st.write(
                        f"**出来高：** "
                        f"{candidate['出来高倍率']}"
                    )

                    st.write(
                        f"**GCまでの距離：** "
                        f"{candidate['GC距離']}"
                    )

                    st.write(
                        f"**52週安値からの上昇率：** "
                        f"{candidate['52週安値から']}"
                    )

                    st.write(
                        f"**データ充足率：** "
                        f"{candidate['データ充足']}"
                    )


            # ---------------------------------------------
            # CSVダウンロード
            # ---------------------------------------------
            csv_data = (
                pd.DataFrame(
                    top_candidates
                )
                .to_csv(
                    index=False,
                    encoding="utf-8-sig"
                )
            )

            st.download_button(
                "📥 スクリーニング結果をCSV保存",
                data=csv_data,
                file_name="大底初動スクリーニング結果.csv",
                mime="text/csv",
                use_container_width=True
            )

        else:

            st.warning(
                "❌ 現在、設定した条件を満たす反転銘柄がありませんでした。"
            )

            st.info(
                "💡 最低総合スコアを60→55→50と下げると、"
                "監視候補を広げられます。"
            )
