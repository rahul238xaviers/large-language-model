import mlx.core as mx

try:
    print("mx.argwhere:", hasattr(mx, "argwhere"))
except Exception as e:
    print("Error:", e)
