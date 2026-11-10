---
status: accepted
date: 2026-11-08
---

# 모든 호출은 temperature 0

파이프라인 간 차이가 샘플링 잡음이 아니라 하네스 차이에서 나오도록, 모든 파이프라인의 모든 호출을 temperature 0으로 고정한다. 정확도가 라운드마다 거의 같아지므로 라운드 반복은 시간·호출 수 측정의 변동을 보기 위한 것이 되고, 절제 묶음을 1라운드로 시작할 수 있는 근거가 된다([ADR-0005](0005-round-based-measurement.md)).

## Consequences

- 여러 번 샘플링해 투표하는 방식(self-consistency 등)은 쓰지 않는다. 호출 수가 늘어 파레토 개선 목표와도 맞지 않는다.
- temperature 0이어도 GPU·Ollama 버전에 따라 출력이 미세하게 달라질 수 있으므로, 라운드마다 GPU 정보와 Ollama 버전을 기록한다.
