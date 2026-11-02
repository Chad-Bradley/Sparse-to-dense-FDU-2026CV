"""单元测试。

运行（从 s2d_baseline/ 目录）：

    python -m unittest discover -s tests -v

测试原则：
    - 不依赖 GPU
    - 不依赖真实数据集；需要数据的测试在数据不存在时 SkipTest
    - 每个测试都能在几秒内跑完
"""
