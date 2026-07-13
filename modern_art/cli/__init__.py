"""命令行前端:引擎的薄封装。支持人机混战与 AI 自对弈。

运行:
    python -m modern_art.cli.play                 # 交互式配置座位
    python -m modern_art.cli.play --seats human,heuristic,heuristic
    python -m modern_art.cli.play --auto --seats heuristic,heuristic,random --seed 1
"""
