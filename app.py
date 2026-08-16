import streamlit as st
import pandas as pd
import numpy as np
import math

st.set_page_config(page_title="大底初動AIスクリーナー", page_icon="🌱", layout="wide")
st.title("🌱 究極版・大底初動先回りスクリーナー")
st.caption("毎日深夜に自動収集された最新データを使用し、大底初動銘柄を超高速抽出します。")

# 🚀 ファイルを読み込むだけの爆速処理！
@st.cache_data(ttl=3600*12)
def load_data():
    try:
        # 日付をインデックスにして株価CSVを読み込み
        prices_df = pd.read_csv("prices.csv", index_col=0, parse_dates=True)
        # 社名CSVを読み込み
        names_df = pd.read_csv("names.csv", dtype={"コード": str})
        name_dict = dict(zip(names_df["コード"], names_df["会社名"]))
        return prices_df, name_dict
    except Exception as e:
        st.error("⚠️ データの読み込みに失敗しました。裏側のデータ更新が完了するまでお待ちください。")
        return None, None

def calc_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    return macd, macd_signal

def calc_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

prices_df, info_dict = load_data()

with st.sidebar:
    st.header("⚙️ 設定")
    st.write("※ 対象銘柄の変更は現在バックエンド一括取得のため固定化されています。")
    run_btn = st.button("🚀 超高速スクリーニング実行", type="primary", use_container_width=True)

if run_btn and prices_df is not None:
    with st.spinner("🌱 AI解析中...（数秒で終わります！）"):
        raw_candidates = []
        # 保存されている全銘柄（列）に対してループ
        for t in prices_df.columns:
            try:
                close_d = prices_df[t].dropna()
                if len(close_d) < 245: continue
                
                curr_price = float(close_d.iloc[-1])
                recent_245d = close_d.tail(245)
                high_52w, low_52w = float(recent_245d.max()), float(recent_245d.min())
                position_ratio = (curr_price - low_52w) / (high_52w - low_52w) if high_52w > low_52w else 1.0

                score_pos = 30.0 if position_ratio < 0.20 else (20.0 if position_ratio < 0.40 else (10.0 if position_ratio < 0.60 else -30.0))

                try:
                    df_m = close_d.resample('ME').last().dropna()
                except Exception:
                    df_m = close_d.resample('M').last().dropna()

                if len(df_m) < 26: continue
                macd_m, _ = calc_macd(df_m)
                curr_m_macd, prev_m_macd = float(macd_m.iloc[-1]), float(macd_m.iloc[-2])
                score_m = 25.0 if (curr_m_macd < 0 and curr_m_macd > prev_m_macd) else (10.0 if curr_m_macd < 0 else 0.0)

                df_w = close_d.resample('W-MON').last().dropna()
                if len(df_w) < 26: continue
                macd_w, sig_w = calc_macd(df_w)
                
                curr_w_macd, curr_w_sig = float(macd_w.iloc[-1]), float(sig_w.iloc[-1])
                prev_w_macd, prev_w_sig = float(macd_w.iloc[-2]), float(sig_w.iloc[-2])

                w_gc_weeks_ago = -1
                for i in range(1, 8):
                    if macd_w.iloc[-i] > sig_w.iloc[-i] and macd_w.iloc[-(i+1)] <= sig_w.iloc[-(i+1)]:
                        w_gc_weeks_ago = i - 1
                        break

                curr_hist_w = curr_w_macd - curr_w_sig
                prev_hist_w = prev_w_macd - prev_w_sig
                
                if curr_hist_w < 0 and curr_hist_w < prev_hist_w:
                    score_w_macd, gc_status = -30.0, "⚠️ 落ちるナイフ状態"
                else:
                    is_imminent = (curr_w_macd < curr_w_sig) and (curr_w_macd > prev_w_macd) and ((curr_w_sig - curr_w_macd) / curr_price < 0.01)
                    if is_imminent:
                        score_w_macd, gc_status = 25.0, "GC直前🔥"
                    elif 0 <= w_gc_weeks_ago <= 2:
                        score_w_macd, gc_status = 20.0, f"{w_gc_weeks_ago}週前GC✨"
                    elif 3 <= w_gc_weeks_ago <= 4:
                        score_w_macd, gc_status = 10.0, f"{w_gc_weeks_ago}週前GC"
                    else:
                        score_w_macd, gc_status = 0.0, "GCなし/反転待ち"

                ma25_d = close_d.rolling(25).mean()
                score_d_trend = 15.0 if curr_price > float(ma25_d.iloc[-1]) else 0.0
                d_trend_status = "25日線突破⤴" if score_d_trend == 15.0 else "下降トレンド"

                curr_rsi = float(calc_rsi(close_d).iloc[-1])
                score_d_rsi = 10.0 if 40 <= curr_rsi <= 55 else (5.0 if 55 < curr_rsi <= 65 else (-10.0 if curr_rsi > 65 else 0.0))

                total_score = score_pos + score_m + score_w_macd + score_d_trend + score_d_rsi
                if total_score < 60.0: continue

                code = t.replace(".T", "")
                raw_candidates.append({
                    "コード": code,
                    "会社名": info_dict.get(code, code),
                    "株価": f"¥{curr_price:,.0f}",
                    "総合スコア": float(f"{math.floor(total_score * 10) / 10:.1f}"),
                    "月足サイン": "底打ちフック発生" if score_m == 25 else "大底揉み合い",
                    "週足MACD": gc_status,
                    "日足トレンド": d_trend_status,
                    "日足RSI": f"{math.floor(curr_rsi * 10) / 10:.1f}"
                })
            except Exception: pass

    raw_candidates.sort(key=lambda x: x["総合スコア"], reverse=True)
    top_candidates = raw_candidates[:30]

    if top_candidates:
        st.success("✨ 【爆速完了】計算が完了しました！")
        res_df = pd.DataFrame(top_candidates).set_index("コード")
        st.dataframe(
            res_df.style.background_gradient(
                subset=['総合スコア'], cmap='Oranges', gmap=res_df['総合スコア']
            ),
            use_container_width=True, height=600
        )
    else:
        st.warning("❌ 現在、厳格な条件を満たす反転銘柄は見つかりませんでした。")