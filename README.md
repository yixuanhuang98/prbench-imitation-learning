# Imitation Learning Baselines for PRBench

![workflow](https://github.com/yixuanhuang98/prbench-imitation-learning/actions/workflows/ci.yml/badge.svg)

## Installation

1. Recommended: create and source a virtualenv (perhaps with [uv](https://github.com/astral-sh/uv))
2. Clone this repo with submodules: `git clone --recurse-submodules https://github.com/yixuanhuang98/prbench-imitation-learning.git` 
3. Install this repo: `pip install -e ".[develop]"`
4. Install the submodules:
    - `pip install -e third-party/prbench`
    - `pip install -e third-party/prbench-bilevel-planning`
    - `pip install -e third-party/prbench-bilevel-planning/third-party/prbench-models`
    - `pip install -e third-party/prbench-bilevel-planning/third-party/bilevel-planning`