@echo off
chcp 65001 > nul
echo ========================================================
echo  AI顔画像 出身地＆ルーツ判定 Webアプリケーション起動中...
echo ========================================================
python -m streamlit run app.py --server.port 8501
pause
