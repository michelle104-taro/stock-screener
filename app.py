import streamlit as st
import pandas as pd
import numpy as np
import math

# =========================================================
# 基本設定
# =========================================================
st.set_page_config(
    page_title="大底初動AIスクリーナー",
    page_icon="🌱",
    layout="wide"
)

st.title("🌱 真・大底初動AIスクリーナー")
st.caption(
    "ダマシ（落ちるナイフ）を排除し、長期下落トレンドを明確に脱出した「真の上昇初動」を100点満点で評価します。"
)

# =========================================================
# データ読み込み
# =========================================================
@st.cache_data(ttl=3600 * 12)
def load_data():
    try:
        # 株価
        prices_df = pd.read_csv("prices.csv", index_col=0, parse_dates=True)
        prices_df.index = pd.to_datetime(prices_df.index)
        prices_df = prices_df.sort_index()

        # 銘柄名
        names_df = pd.read_csv("names.csv", dtype={"コード": str})
        name_dict = dict(zip(names_df["コード"].astype(str), names_df["会社名"]))

        # 出来高
        volume_df = pd.read_csv("volume.csv", index_col=0, parse_dates=True)
        volume_df.index = pd.to_datetime(volume_df.index)
        volume_df = volume_df.sort_index()

        # ファンダメンタルズ
        fundamentals_df = pd.read_csv("fundamentals.csv", dtype={"コード": str})
        fundamentals_df["コード"] = (
            fundamentals_df["コード"]
            .astype(str)
            .str.replace(".0", "", regex=False)
            .str.replace(".T", "", regex=False)
        )

        return prices_df, name_dict, volume_df, fundamentals_df

    except Exception as e:
        st.error(f"⚠️ データの読み込みに失敗しました。\n\n詳細: {e}")
        return None, None, None, None

# =========================================================
# テクニカル計算
# =========================================================
def calc_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    return macd, macd_signal

def calc_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0).rolling(period).mean())
    loss = (-delta.where(delta < 0, 0).rolling(period).mean())
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
# データ取得
# =========================================================
(
    prices_df,
    info_dict,
    volume_df,
    fundamentals_df
) = load_data()

# =========================================================
# サイドバー
# =========================================================
with st.sidebar:
    st.header("⚙️ スクリーニング設定")
    st.write("大底のダマシを避け、反転初動を狙うロジックです。")

    min_score = st.slider("最低総合スコア", min_value=40, max_value=90, value=50, step=5)
    max_results = st.slider("表示銘柄数", min_value=10, max_value=100, value=30, step=10)

    st.divider()
    st.subheader("🚫 ハード除外")
    exclude_overheat = st.checkbox("日足RSI 80超 → 除外（高値掴み防止）", value=True)

    st.divider()
    st.subheader("📁 評価ウエイト (100点満点)")
    st.write("① 長期トレンド転換: 25点")
    st.write("② 底値からの浮上度: 15点")
    st.write("③ 出来高(買い集め): 20点")
    st.write("④ 中長期MACD改善: 20点")
    st.write("⑤ 日足の健全性: 10点")
    st.write("⑥ ファンダメンタルズ: 10点")

    # Streamlit非推奨警告対策
    run_btn = st.button("🚀 スクリーニング実行", type="primary")

# =========================================================
# スクリーニング実行
# =========================================================
if run_btn and prices_df is not None:
    with st.spinner("🌱 真・上昇トレンドAI解析中..."):
        raw_candidates = []
        total_symbols = len(prices_df.columns)
        screened_symbols = 0
        excluded_symbols = 0

        for t in prices_df.columns:
            try:
                close_d = prices_df[t].dropna().astype(float).sort_index()
                if len(close_d) < 500:
                    continue

                curr_price = float(close_d.iloc[-1])

                # 移動平均線
                ma25 = close_d.rolling(25).mean()
                ma75 = close_d.rolling(75).mean()
                ma200 = close_d.rolling(200).mean()

                curr_ma25 = float(ma25.iloc[-1])
                prev_ma25 = float(ma25.iloc[-2])
                curr_ma75 = float(ma75.iloc[-1])
                curr_ma200 = float(ma200.iloc[-1])
                prev_ma200 = float(ma200.iloc[-2])

                # =================================================
                # ① 長期トレンド転換 (最大25点)
                # =================================================
                score_trend = 0
                trend_status = "下落トレンド"
                
                # 落ちるナイフ（パーフェクトオーダー下落）判定
                if curr_price < curr_ma25 and curr_ma25 < curr_ma75 and curr_ma75 < curr_ma200:
                    score_trend = 0
                    trend_status = "完全下落(要注意)"
                # 200日線突破 ＋ 200日線上向き（最強の転換サイン）
                elif curr_price > curr_ma200 and curr_ma200 >= prev_ma200:
                    score_trend = 25
                    trend_status = "200MA突破・上昇転換🔥"
                # 200日線は下向きだが価格は上抜けた
                elif curr_price > curr_ma200:
                    score_trend = 20
                    trend_status = "200MA上抜け✨"
                # 75日線は上抜けて底打ち感が出ている
                elif curr_price > curr_ma75:
                    score_trend = 15
                    trend_status = "75MA上抜け(初動候補)"
                # 25日線のみ上抜け
                elif curr_price > curr_ma25:
                    score_trend = 5
                    trend_status = "25MA上抜け(打診)"

                # =================================================
                # ② 大底からの浮上度 (最大15点)
                # =================================================
                recent_245d = close_d.tail(245)
                low_52w = float(recent_245d.min())
                high_52w = float(recent_245d.max())
                
                rise_pct = ((curr_price / low_52w) - 1) * 100 if low_52w > 0 else 0
                
                score_52w = 0
                if 10 <= rise_pct <= 30:
                    score_52w = 15  # 初動としてベストな浮上位置
                elif 5 <= rise_pct < 10 or 30 < rise_pct <= 40:
                    score_52w = 10  # 悪くない位置
                elif 0 <= rise_pct < 5:
                    score_52w = 5   # まだ底割れリスクあり
                else:
                    score_52w = 0   # 高値圏すぎる、または異常値

                # =================================================
                # ③ 出来高・買い集め判定 (最大20点)
                # =================================================
                score_vol = 0
                volume_ratio = np.nan
                acc_ratio = np.nan

                if volume_df is not None and t in volume_df.columns:
                    vol_s = volume_df[t].dropna().astype(float).sort_index()
                    if len(vol_s) >= 20:
                        curr_vol = float(vol_s.iloc[-1])
                        avg_vol_20 = float(vol_s.tail(20).mean())
                        
                        # 単純な出来高増加（最大10点）
                        if avg_vol_20 > 0:
                            volume_ratio = curr_vol / avg_vol_20
                            if volume_ratio >= 2.0: score_vol += 10
                            elif volume_ratio >= 1.5: score_vol += 5
                            elif volume_ratio >= 1.2: score_vol += 2
                        
                        # 買い集め判定（Accumulation）（最大10点）
                        # 過去20日の「上昇日の出来高合計」÷「下落日の出来高合計」
                        vol_20 = vol_s.tail(20)
                        diff_20 = close_d.tail(20).diff()
                        up_vol = vol_20[diff_20 > 0].sum()
                        down_vol = vol_20[diff_20 <= 0].sum()
                        
                        if down_vol > 0:
                            acc_ratio = up_vol / down_vol
                            if acc_ratio >= 1.5: score_vol += 10
                            elif acc_ratio >= 1.1: score_vol += 5

                # =================================================
                # ④ 中長期MACD改善 (最大20点)
                # =================================================
                score_macd = 0
                
                # 週足MACD
                df_w = close_d.resample("W-FRI").last().dropna()
                macd_w, sig_w = calc_macd(df_w)
                curr_w_macd = float(macd_w.iloc[-1])
                prev_w_macd = float(macd_w.iloc[-2])
                curr_w_sig = float(sig_w.iloc[-1])
                
                # マイナス圏でのGC、またはマイナス圏で上向き（最大10点）
                if curr_w_macd < 0:
                    if curr_w_macd > curr_w_sig and prev_w_macd <= float(sig_w.iloc[-2]):
                        score_macd += 10 # 今週GC
                    elif curr_w_macd > curr_w_sig:
                        score_macd += 5  # すでにGC済みで上昇中
                    elif curr_w_macd > prev_w_macd:
                        score_macd += 3  # まだGCしていないが改善中
                        
                # 月足MACDヒストグラム
                try: df_m = close_d.resample("ME").last().dropna()
                except Exception: df_m = close_d.resample("M").last().dropna()
                
                macd_m, sig_m = calc_macd(df_m)
                hist_m = macd_m - sig_m
                if len(hist_m) >= 2:
                    curr_m_hist = float(hist_m.iloc[-1])
                    prev_m_hist = float(hist_m.iloc[-2])
                    if curr_m_hist > prev_m_hist:
                        score_macd += 10 # 月足レベルで底打ちサイン

                # =================================================
                # ⑤ 日足の健全性 (最大10点)
                # =================================================
                score_daily = 0
                
                # RSIが健全なトレンド範囲か（最大5点）
                rsi_series = calc_rsi(close_d)
                curr_rsi = float(rsi_series.iloc[-1])
                if exclude_overheat and curr_rsi > 80:
                    excluded_symbols += 1
                    continue
                    
                if 45 <= curr_rsi <= 65:
                    score_daily += 5
                elif 40 <= curr_rsi < 45 or 65 < curr_rsi <= 75:
                    score_daily += 2
                    
                # 25日線が上向きか（最大5点）
                if curr_price > curr_ma25 and curr_ma25 > prev_ma25:
                    score_daily += 5

                # =================================================
                # ⑥ ファンダメンタルズ (最大10点)
                # =================================================
                score_funda = 0
                per_value = np.nan
                pbr_value = np.nan
                eps_growth = np.nan

                code = t.replace(".T", "").replace(".JP", "")

                if fundamentals_df is not None:
                    fund_rows = fundamentals_df[fundamentals_df["コード"] == code]
                    if not fund_rows.empty:
                        fund = fund_rows.iloc[0]
                        per_value = safe_float(fund.get("PER", np.nan))
                        pbr_value = safe_float(fund.get("PBR", np.nan))
                        eps_growth = safe_float(fund.get("EPS成長率", np.nan))
                        industry_per = safe_float(fund.get("業種平均PER", np.nan))

                        if not np.isnan(eps_growth) and eps_growth > 0:
                            score_funda += 4 # 成長している企業を優遇
                        if not np.isnan(per_value) and per_value > 0 and not np.isnan(industry_per) and industry_per > 0:
                            if per_value < industry_per:
                                score_funda += 3 # 割安
                        if not np.isnan(pbr_value) and pbr_value < 1.0:
                            score_funda += 3 # PBR1倍割れ

                # =================================================
                # 総合スコア
                # =================================================
                total_score = (
                    score_trend + score_52w + score_vol + score_macd + score_daily + score_funda
                )
                total_score = round(float(total_score), 1)

                # Stage判定
                if "200MA突破" in trend_status:
                    stage = "🚀 Stage 3：トレンド転換"
                elif "75MA上抜け" in trend_status:
                    stage = "🔥 Stage 2：反転初動"
                else:
                    stage = "🌑 Stage 1：底練り"

                # 評価
                if total_score >= 70: rating = "⭐⭐⭐ 最注目"
                elif total_score >= 60: rating = "⭐⭐ 有力監視"
                elif total_score >= 50: rating = "⭐ 候補"
                else: rating = "監視"

                screened_symbols += 1

                # 結果保存
                raw_candidates.append({
                    "コード": code,
                    "会社名": info_dict.get(code, code),
                    "株価": f"¥{curr_price:,.0f}",
                    "総合スコア": total_score,
                    "評価": rating,
                    "Stage": stage,
                    "トレンド": score_trend,
                    "底値脱出": score_52w,
                    "出来高": score_vol,
                    "中長期MACD": score_macd,
                    "日足健全性": score_daily,
                    "ファンダ": score_funda,
                    "トレンド詳細": trend_status,
                    "安値から上昇率": f"{rise_pct:.1f}%",
                    "出来高倍率": f"{volume_ratio:.2f}倍" if not np.isnan(volume_ratio) else "-",
                    "買集め比率": f"{acc_ratio:.2f}" if not np.isnan(acc_ratio) else "-",
                    "日足RSI": round(curr_rsi, 1),
                    "PER": round(per_value, 1) if not np.isnan(per_value) else "-",
                    "PBR": round(pbr_value, 2) if not np.isnan(pbr_value) else "-",
                    "EPS成長率": f"{eps_growth:.1f}%" if not np.isnan(eps_growth) else "-"
                })

            except Exception:
                continue

        # =========================================================
        # 表示準備と結果出力
        # =========================================================
        raw_candidates.sort(key=lambda x: x["総合スコア"], reverse=True)
        filtered_candidates = [x for x in raw_candidates if x["総合スコア"] >= min_score]
        top_candidates = filtered_candidates[:max_results]

        if top_candidates:
            st.success(f"✨ スクリーニング完了：真の上昇初動候補を {len(top_candidates)}銘柄 抽出しました。")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("対象銘柄", f"{total_symbols:,}")
            col2.metric("スコア条件通過", f"{len(filtered_candidates):,}")
            col3.metric("過熱感除外(RSI>80)", f"{excluded_symbols:,}")
            col4.metric("最高スコア", f"{top_candidates[0]['総合スコア']:.0f}")

            st.divider()

            display_columns = [
                "コード", "会社名", "株価", "総合スコア", "評価", "Stage",
                "トレンド", "底値脱出", "出来高", "中長期MACD", "日足健全性", "ファンダ",
                "トレンド詳細", "安値から上昇率", "出来高倍率", "日足RSI"
            ]

            # 順番入れ替え（列指定 → set_index）
            res_df = pd.DataFrame(top_candidates)[display_columns].set_index("コード")
            styled_df = res_df.style.background_gradient(subset=["総合スコア"], cmap="Oranges")

            st.dataframe(styled_df, use_container_width=True, height=700)

            st.divider()
            st.subheader("🔎 上位銘柄のスコア詳細")

            for candidate in top_candidates[:10]:
                with st.expander(
                    f"{candidate['コード']} {candidate['会社名']} ｜ {candidate['総合スコア']:.0f}点 ｜ {candidate['Stage']}"
                ):
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("総合スコア", f"{candidate['総合スコア']:.0f} / 100")
                    c2.metric("トレンド転換", f"{candidate['トレンド']} / 25")
                    c3.metric("買い集め(出来高)", f"{candidate['出来高']} / 20")
                    c4.metric("中長期MACD", f"{candidate['中長期MACD']} / 20")

                    detail_df = pd.DataFrame({
                        "評価項目": [
                            "① 長期トレンド転換", "② 底値からの浮上度", "③ 出来高(買い集め)", 
                            "④ 中長期MACD改善", "⑤ 日足の健全性", "⑥ ファンダメンタルズ"
                        ],
                        "得点": [
                            candidate["トレンド"], candidate["底値脱出"], candidate["出来高"],
                            candidate["中長期MACD"], candidate["日足健全性"], candidate["ファンダ"]
                        ],
                        "最大点": [25, 15, 20, 20, 10, 10]
                    })
                    detail_df["達成率"] = (detail_df["得点"] / detail_df["最大点"] * 100).round(0).astype(int).astype(str) + "%"

                    st.dataframe(detail_df, use_container_width=True, hide_index=True)

                    st.write(f"**トレンド状況：** {candidate['トレンド詳細']}")
                    st.write(f"**安値からの上昇率：** {candidate['安値から上昇率']}")
                    st.write(f"**出来高倍率：** {candidate['出来高倍率']} （買集め比率: {candidate['買集め比率']}）")
                    st.write(f"**日足RSI：** {candidate['日足RSI']}")
                    st.write(f"**PER / PBR：** {candidate['PER']} / {candidate['PBR']}")
                    st.write(f"**EPS成長率：** {candidate['EPS成長率']}")

            csv_data = pd.DataFrame(top_candidates).to_csv(index=False, encoding="utf-8-sig")
            st.download_button(
                "📥 スクリーニング結果をCSV保存",
                data=csv_data,
                file_name="真・大底初動スクリーニング結果.csv",
                mime="text/csv"
            )
        else:
            st.warning("❌ 現在、設定した条件を満たす反転銘柄がありませんでした。")
            st.info("💡 最低総合スコアを下げると、監視候補を広げられます。")