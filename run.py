import os
import sys

# Change directory to ai_engine to ensure paths work correctly
os.chdir(os.path.join(os.path.dirname(__file__), 'ai_engine'))

print("===========================================")
print("Starting EdgeAuth Liveness Challenge")
print("===========================================")

# Execute the liveness script
os.system(f"{sys.executable} liveness_challenge.py")
