@echo off
echo ===========================================
echo Starting EdgeAuth Liveness Challenge
echo ===========================================
cd ai_engine
python liveness_challenge.py
cd ..
pause
