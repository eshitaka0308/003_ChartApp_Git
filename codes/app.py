import streamlit as st
import yfinance as yf
import plotly.express as px
import pandas as pd
import os
import xml.etree.ElementTree as ET
import io

## Streamlit: https://essy-003-chartapp-git.streamlit.app/

# セッション状態初期化
if 'watchlists' not in st.session_state:
    st.session_state.watchlists = {}
if 'current_watchlist' not in st.session_state:
    st.session_state.current_watchlist = None
if 'reference_tickers' not in st.session_state:
    st.session_state.reference_tickers = {}
if 'hidden_tickers' not in st.session_state:
    st.session_state.hidden_tickers = {}
if 'use_log_scale' not in st.session_state:
    st.session_state.use_log_scale = False

# ローカル保存ファイル
SAVE_FILE = 'watchlists.xml'

# ローカル保存/読み込み（XML）
@st.cache_data
def load_watchlists():
    if os.path.exists(SAVE_FILE):
        try:
            tree = ET.parse(SAVE_FILE)
            root = tree.getroot()
            result = {}
            ref_tickers = {}
            # 保存時にルート要素に付与した選択中ウォッチリスト属性を取得
            selected = root.get('selected', '')
            for wl in root.findall('watchlist'):
                name = wl.get('name')
                tickers = [t.text for t in wl.findall('ticker') if t.text]
                reference_ticker = wl.get('reference_ticker', '')
                result[name] = tickers
                if reference_ticker:
                    ref_tickers[name] = reference_ticker
            return result, ref_tickers, selected
        except Exception as e:
            st.error(f"ウォッチリスト読み込みエラー: {e}")
            return {}, {}, ''
    return {}, {}, ''


def save_watchlists(watchlists, reference_tickers=None):
    from xml.dom import minidom
    if reference_tickers is None:
        reference_tickers = {}
    # 現在選択中のウォッチリストを属性として保存（存在しない/なし の場合は空文字）
    selected = ''
    try:
        selected = st.session_state.get('current_watchlist') or ''
    except Exception:
        selected = ''
    if selected == "なし":
        selected = ''
    root = ET.Element('watchlists', {'selected': selected})
    for name, tickers in watchlists.items():
        ref_ticker = reference_tickers.get(name, '')
        wl = ET.SubElement(root, 'watchlist', {'name': name, 'reference_ticker': ref_ticker})
        for t in tickers:
            te = ET.SubElement(wl, 'ticker')
            te.text = t
    try:
        # XML を整形して保存
        rough_string = ET.tostring(root, encoding='utf-8')
        reparsed = minidom.parseString(rough_string)
        pretty_xml = reparsed.toprettyxml(indent="  ")
        # XML 宣言の重複を避けるため、minidom の宣言を削除してから自分で追加
        pretty_xml_lines = pretty_xml.split('\n')[1:]  # 最初の XML 宣言行をスキップ
        with open(SAVE_FILE, 'w', encoding='utf-8') as f:
            f.write('<?xml version="1.0" encoding="utf-8"?>\n')
            f.write('\n'.join(pretty_xml_lines))
    except Exception as e:
        st.error(f"ウォッチリスト保存エラー: {e}")

def watchlists_to_xml_bytes(watchlists, reference_tickers=None):
    from xml.dom import minidom
    if reference_tickers is None:
        reference_tickers = {}
    # 現在選択中のウォッチリストを属性として含める
    selected = ''
    try:
        selected = st.session_state.get('current_watchlist') or ''
    except Exception:
        selected = ''
    if selected == "なし":
        selected = ''
    root = ET.Element('watchlists', {'selected': selected})
    for name, tickers in watchlists.items():
        ref_ticker = reference_tickers.get(name, '')
        wl = ET.SubElement(root, 'watchlist', {'name': name, 'reference_ticker': ref_ticker})
        for t in tickers:
            te = ET.SubElement(wl, 'ticker')
            te.text = t
    # XML を整形してバイト列で返す
    rough_string = ET.tostring(root, encoding='utf-8')
    reparsed = minidom.parseString(rough_string)
    pretty_xml = reparsed.toprettyxml(indent="  ")
    # XML 宣言の重複を避けるため調整
    pretty_xml_lines = pretty_xml.split('\n')[1:]
    result = '<?xml version="1.0" encoding="utf-8"?>\n' + '\n'.join(pretty_xml_lines)
    return result.encode('utf-8')

def load_watchlists_from_bytes(bytes_data):
    try:
        root = ET.fromstring(bytes_data)
        result = {}
        ref_tickers = {}
        selected = root.get('selected', '')
        for wl in root.findall('watchlist'):
            name = wl.get('name')
            tickers = [t.text for t in wl.findall('ticker') if t.text]
            reference_ticker = wl.get('reference_ticker', '')
            result[name] = tickers
            if reference_ticker:
                ref_tickers[name] = reference_ticker
        return result, ref_tickers, selected
    except Exception as e:
        raise

# ティッカー検証
def validate_ticker(ticker):
    try:
        info = yf.Ticker(ticker).info
        return bool(info)
    except Exception:
        return False


# ティッカー名取得（キャッシュ付き）
@st.cache_data
def get_ticker_name(ticker):
    try:
        info = yf.Ticker(ticker).info
        name = info.get('shortName') or info.get('longName') or ''
        return name
    except Exception:
        return ''

# データ取得
def get_data(tickers, period='1y'):
    try:
        data = yf.download(tickers, period=period)['Close']
        # 単一ティッカーのとき Series になることがあるため DataFrame に変換
        if isinstance(data, pd.Series):
            data = data.to_frame(name=tickers if isinstance(tickers, str) else tickers[0])
        # MultiIndex 修正（複数銘柄で返回される場合）
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]
        return data
    except Exception as e:
        st.error(f"データ取得エラー: {e}")
        return pd.DataFrame()

# 正規化
def normalize_data(data, normalize_method='left_edge_with_reference', reference_ticker=None):
    """
    正規化関数
    
    Args:
        data: DataFrame
        normalize_method: 
            - 'left_edge': 各ティッカーの左端を100%にする
            - 'reference_ticker': 指定ティッカーの左端を100%にする
            - 'left_edge_with_reference': 左端を100%にした後、指定ティッカーが100%になるようスケーリング
        reference_ticker: reference_tickerメソッドで基準とするティッカー名
    
    Returns:
        正規化されたDataFrame
    """
    if data.empty:
        return data
    
    norm = data.copy()
    
    if normalize_method == 'left_edge':
        # 各ティッカーをそれぞれ左端（最初の有効値）を100%とする正規化
        for col in norm.columns:
            series = norm[col].dropna()
            if series.empty:
                continue
            first = series.iloc[0]
            try:
                if first == 0:
                    continue
                norm[col] = (norm[col] / first) * 100.0
            except Exception:
                # 変換できない場合は元のままにする
                continue
    
    elif normalize_method == 'reference_ticker':
        # 毎データポイント、指定ティッカーで正規化（指定ティッカーが常に100%になる）
        if reference_ticker not in data.columns:
            st.error(f"ティッカー '{reference_ticker}' がデータに存在しません")
            return data
        
        reference_series = norm[reference_ticker]
        if reference_series.empty:
            st.error(f"ティッカー '{reference_ticker}' のデータがありません")
            return data
        
        try:
            # 各列を指定ティッカーの値で割る（毎データポイントで正規化）
            for col in norm.columns:
                norm[col] = (norm[col] / reference_series) * 100.0
        except Exception as e:
            st.error(f"正規化エラー: {e}")
            return data
    
    elif normalize_method == 'left_edge_with_reference':
        # 指定ティッカーが100%になるようスケーリングしてから、
        # 各ティッカーをそれぞれ左端を100%で正規化
        if reference_ticker not in data.columns:
            st.error(f"ティッカー '{reference_ticker}' がデータに存在しません")
            return data
        
        reference_series = norm[reference_ticker]
        if reference_series.empty:
            st.error(f"ティッカー '{reference_ticker}' のデータがありません")
            return data
        
        try:
            # 各列を指定ティッカーの値で割る（毎データポイントで正規化）
            for col in norm.columns:
                norm[col] = (norm[col] / reference_series) * 100.0
        except Exception as e:
            st.error(f"正規化エラー: {e}")
            return data
        
        # ステップ2: 各ティッカーを左端を100%で正規化
        for col in norm.columns:
            series = norm[col].dropna()
            if series.empty:
                continue
            first = series.iloc[0]
            try:
                if first == 0:
                    continue
                # 最初の値で割って、次以降のnorm[col]を計算
                norm[col] = norm[col] / first * 100.0
            except Exception:
                continue
    
    return norm

# 初期読み込み（セッションが空ならファイルから読み込む）
if not st.session_state.watchlists:
    watchlists_data, ref_tickers_data, selected = load_watchlists()
    st.session_state.watchlists = watchlists_data
    st.session_state.reference_tickers = ref_tickers_data
    # 保存時に選択していたウォッチリストを復元（存在しない/空ならそのまま）
    if selected:
        st.session_state.current_watchlist = selected

# メインUI
st.title("株式ウォッチリスト・チャートアプリ")

# サイドバー: ウォッチリスト管理
st.sidebar.header("ウォッチリスト管理")

# 新規作成
new_name = st.sidebar.text_input("新規ウォッチリスト名")
if st.sidebar.button("作成") and new_name:
    if new_name not in st.session_state.watchlists:
        st.session_state.watchlists[new_name] = []
        save_watchlists(st.session_state.watchlists, st.session_state.reference_tickers)
        st.sidebar.success(f"{new_name} を作成しました")
# ウォッチリスト名変更（リネーム）
rename_input = st.sidebar.text_input("ウォッチリスト名を変更", key="rename_input")
if st.sidebar.button("名前変更"):
    old = st.session_state.current_watchlist
    new = rename_input.strip()
    if old is None or old == "なし":
        st.sidebar.error("先にウォッチリストを選択してください")
    elif not new:
        st.sidebar.error("新しい名前を入力してください")
    elif new in st.session_state.watchlists:
        st.sidebar.error("同名のウォッチリストが既に存在します")
    else:
        # 名前を変更（データと参照ティッカー両方）
        st.session_state.watchlists[new] = st.session_state.watchlists.pop(old)
        # reference_tickers があれば移動
        if old in st.session_state.reference_tickers:
            st.session_state.reference_tickers[new] = st.session_state.reference_tickers.pop(old)
        else:
            # 新しいキーが存在しない場合は空にしておく
            st.session_state.reference_tickers.pop(new, None)
        # 保存して表示を更新
        save_watchlists(st.session_state.watchlists, st.session_state.reference_tickers)
        st.session_state.current_watchlist = new
        st.sidebar.success(f"{old} を {new} に名前変更しました")
        st.rerun()

# リスト表示/選択
watchlist_names = list(st.session_state.watchlists.keys())
previous_watchlist = st.session_state.current_watchlist
st.session_state.current_watchlist = st.sidebar.selectbox("ウォッチリストを選択", ["なし"] + watchlist_names)

# ウォッチリスト選択変更時に右側ウィンドウを更新
if st.session_state.current_watchlist != previous_watchlist:
    # ウォッチリスト変更時にログスケール状態をリセット
    st.session_state.use_log_scale = False
    st.rerun()

if st.session_state.current_watchlist != "なし":
    selected_list = st.session_state.watchlists[st.session_state.current_watchlist]
    # ティッカーをYahoo Financeへのリンクに変換
    ticker_links = [f"[{ticker}](https://finance.yahoo.co.jp/quote/{ticker}/chart)" for ticker in selected_list]
    st.sidebar.markdown(f"ティッカー: {', '.join(ticker_links)}")

# ファイル操作: 任意の場所からの読み込みと任意の場所への保存（ダウンロード）
st.sidebar.header("ファイル操作")
# アップロード（任意の場所から読み込み）
uploaded = st.sidebar.file_uploader("XMLファイルを読み込む", type=['xml'])
if uploaded is not None:
    try:
        data = uploaded.read()
        loaded, loaded_refs, loaded_selected = load_watchlists_from_bytes(data)
        st.session_state.watchlists = loaded
        st.session_state.reference_tickers = loaded_refs
        # サーバー側にも保存しておく（オプション）
        save_watchlists(st.session_state.watchlists, st.session_state.reference_tickers)
        # 読み込み後、XMLに保存されていた選択ウォッチリストがあれば復元、なければ最初のウォッチリストを自動選択
        if loaded_selected:
            st.session_state.current_watchlist = loaded_selected
        elif loaded:
            first_watchlist = list(loaded.keys())[0]
            st.session_state.current_watchlist = first_watchlist
        st.sidebar.success("ウォッチリストを読み込みました")
        st.rerun()
    except Exception as e:
        st.sidebar.error(f"読み込みエラー: {e}")

# サーバーに上書き保存（ローカルファイルを更新）
if st.sidebar.button("上書き保存（サーバー）"):
    try:
        save_watchlists(st.session_state.watchlists, st.session_state.reference_tickers)
        st.sidebar.success(f"{SAVE_FILE} に上書き保存しました")
    except Exception as e:
        st.sidebar.error(f"保存エラー: {e}")

# ダウンロード（任意の場所へ保存）
xml_bytes = watchlists_to_xml_bytes(st.session_state.watchlists, st.session_state.reference_tickers)
st.sidebar.download_button("ダウンロード (XML)", data=xml_bytes, file_name="watchlists.xml", mime="application/xml")

# ティッカー登録（マニュアルのみ）
st.sidebar.header("ティッカー登録")
manual_ticker = st.sidebar.text_input("ティッカーを入力（例: AAPL, 7203.T）")

if st.sidebar.button("登録") and st.session_state.current_watchlist != "なし":
    ticker_to_add = manual_ticker.strip()
    if not ticker_to_add:
        st.sidebar.error("ティッカーを入力してください")
    elif ticker_to_add in st.session_state.watchlists[st.session_state.current_watchlist]:
        st.sidebar.info("既に登録されています")
    elif validate_ticker(ticker_to_add):
        st.session_state.watchlists[st.session_state.current_watchlist].append(ticker_to_add)
        save_watchlists(st.session_state.watchlists, st.session_state.reference_tickers)
        st.sidebar.success(f"{ticker_to_add} を登録しました")
    else:
        st.sidebar.error("無効なティッカーです")

# 削除
if st.session_state.current_watchlist != "なし":
    to_remove = st.sidebar.multiselect("削除するティッカー", st.session_state.watchlists[st.session_state.current_watchlist])
    if st.sidebar.button("削除"):
        for t in to_remove:
            if t in st.session_state.watchlists[st.session_state.current_watchlist]:
                st.session_state.watchlists[st.session_state.current_watchlist].remove(t)
        save_watchlists(st.session_state.watchlists, st.session_state.reference_tickers)
        st.rerun()

# メイン: チャート表示
if st.session_state.current_watchlist != "なし" and len(st.session_state.watchlists[st.session_state.current_watchlist]) > 0:
    tickers = st.session_state.watchlists[st.session_state.current_watchlist]
    periods = ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']
    
    # チェックボックス状態変更時のコールバック関数
    def toggle_visibility(ticker):
        hidden_list = st.session_state.hidden_tickers[st.session_state.current_watchlist]
        if ticker in hidden_list:
            hidden_list.remove(ticker)
        else:
            hidden_list.append(ticker)
    
    col1, col2 = st.columns(2)
    with col1:
        period = st.selectbox("表示期間", periods, index=periods.index('1y'), key="period_select")
    
    with col2:
        normalize_method = st.selectbox(
            "正規化方法",
            ["左端を100%にし、基準ティッカーを選択", "左端を100%にする", "基準ティッカーを選択"],
            index=0,
            key="normalize_method_select"
        )
    
    data = get_data(tickers, period)
    if not data.empty:
        # 非表示ティッカーを取得（ウォッチリストがなければ空リスト）
        hidden_list = st.session_state.hidden_tickers.get(st.session_state.current_watchlist, [])
        # 表示対象のティッカーをフィルタ
        display_tickers = [t for t in data.columns if t not in hidden_list]
        
        # 表示対象がない場合の処理
        if not display_tickers:
            st.warning("表示対象のティッカーがありません。チェックボックスで表示するティッカーを選択してください。")
        else:
            # フィルタ後のデータのみを使用
            data = data[display_tickers]
        # 正規化方法に応じて処理 
        if normalize_method == "左端を100%にする":
            normalized_data = normalize_data(data, normalize_method='left_edge')
            chart_title = f"{st.session_state.current_watchlist} のチャート (左端を100%で正規化)"
        
        elif normalize_method == "基準ティッカーを選択":
            # 参照ティッカーを選択（前回の選択を復元）
            current_ref = st.session_state.reference_tickers.get(st.session_state.current_watchlist, tickers[0])
            # 基準ティッカーが現在のリストに存在しない場合は最初のティッカーを使用
            if current_ref not in tickers:
                current_ref = tickers[0]
            st.write("基準とするティッカー")
            reference_ticker = st.radio("", tickers, index=tickers.index(current_ref), key="reference_ticker_radio_1", horizontal=True)
            # 選択内容を保存
            if reference_ticker != current_ref:
                st.session_state.reference_tickers[st.session_state.current_watchlist] = reference_ticker
                save_watchlists(st.session_state.watchlists, st.session_state.reference_tickers)
            normalized_data = normalize_data(data, normalize_method='reference_ticker', reference_ticker=reference_ticker)
            chart_title = f"{st.session_state.current_watchlist} のチャート ({reference_ticker} で毎回正規化)"
        
        else:  # "左端を100%にし、基準ティッカーを選択"
            # 参照ティッカーを選択（前回の選択を復元）
            current_ref = st.session_state.reference_tickers.get(st.session_state.current_watchlist, tickers[0])
            # 基準ティッカーが現在のリストに存在しない場合は最初のティッカーを使用
            if current_ref not in tickers:
                current_ref = tickers[0]
            st.write("基準とするティッカー")
            reference_ticker = st.radio("", tickers, index=tickers.index(current_ref), key="reference_ticker_radio_2", horizontal=True)
            # 選択内容を保存
            if reference_ticker != current_ref:
                st.session_state.reference_tickers[st.session_state.current_watchlist] = reference_ticker
                save_watchlists(st.session_state.watchlists, st.session_state.reference_tickers)
            normalized_data = normalize_data(data, normalize_method='left_edge_with_reference', reference_ticker=reference_ticker)
            chart_title = f"{st.session_state.current_watchlist} のチャート (左端を100%→{reference_ticker}を100%に調整)"
        
        fig = px.line(normalized_data, x=normalized_data.index, y=normalized_data.columns,
                      title=chart_title,
                      labels={'value': '変化 (%)', 'variable': 'ティッカー'})
        fig.update_layout(xaxis_title="日付", yaxis_title="変化 (%)")
        fig.update_xaxes(rangeslider_visible=True)

        # ログスケール切り替えチェックボックス
        st.session_state.use_log_scale = st.checkbox("縦軸をログスケールにする", value=st.session_state.use_log_scale)
        if st.session_state.use_log_scale:
            fig.update_yaxes(type='log')
        else:
            fig.update_yaxes(type='linear')
        st.plotly_chart(fig, use_container_width=True)

        # チャート下にティッカーと名称を表示（チェックボックス付き）
        try:
            # 現在のウォッチリスト用の hidden_tickers を初期化
            if st.session_state.current_watchlist not in st.session_state.hidden_tickers:
                st.session_state.hidden_tickers[st.session_state.current_watchlist] = []
            
            hidden_list = st.session_state.hidden_tickers[st.session_state.current_watchlist]
            
            st.markdown("**ティッカー一覧（表示/非表示）**")
            
            # チェックボックスをコラム配置で表示
            cols = st.columns(3)  # 3列のレイアウト
            col_idx = 0
            
            for t in tickers:
                col = cols[col_idx % 3]
                
                is_hidden = t in hidden_list
                
                with col:
                    # チェックボックス：チェック = 表示, 非チェック = 非表示
                    new_state = st.checkbox(
                        t,
                        value=(not is_hidden),  # 非表示なら False, 表示なら True
                        key=f"ticker_visibility_{st.session_state.current_watchlist}_{t}",
                        on_change=lambda ticker=t: toggle_visibility(ticker)
                    )
                    
                    # ティッカー情報を表示
                    name = get_ticker_name(t)
                    gf_link = f"[GF](https://www.google.com/finance/quote/{t})"
                    yf_link = f"[YF](https://finance.yahoo.co.jp/quote/{t}/chart)"
                    yfc_link = f"[YFC](https://finance.yahoo.com/quote/{t})"
                    if name:
                        st.caption(f"{name} ({gf_link}, {yf_link}, {yfc_link})")
                    else:
                        st.caption(f"({gf_link}, {yf_link}, {yfc_link})")
                
                col_idx += 1
        except Exception:
            pass
    else:
        st.warning("データを取得できませんでした。")
else:
    st.info("ウォッチリストを作成/選択してチャートを表示してください。")