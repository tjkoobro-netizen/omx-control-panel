# 관제 미션 손실·과부하 감시 (omx_live_monitor.py)

11/24 TK 요청으로 만들었다. 카메라를 모두 켠 관제 미션에서 과부하와 데이터 손실(카메라·추론 통신)을 기록한다. 보고 싶을 때는 실시간으로 보고, 아니면 나중에 확인한다.
제어 PC(tkoo-pc)에서 패널과 따로 돌고, 읽기만 한다. 로봇·패널·평가 프로세스에는 손대지 않는다.

## 부담 (11/24 실측, 1분)
- 제어 PC: 코어 하나의 1.6%(8코어 전체로 약 0.2%)이고 nice 10이다. 1초에 한 번 /proc 카운터와 패널이 이미 쓰는 파일만 읽는다. 영상은 열지 않는다.
- GPU 노트북: 0.06%이고 nice 19다. 기본(local)은 미션 중 통신이 없다. 그 노트북 파일에만 쓰고, 미션이 끝난 뒤 `--collect`로 가져온다.
- stream 모드(실시간 GPU 경고)는 1초에 약 0.4 KB를 보낸다. 추론 데이터(약 1 MB/s)의 0.04%다.

## 쓰는 법
패널 3.27.0부터 프로그램은 패널 안(`omx_tools/omx_live_monitor.py`)에 있고, 제어 PC 패널의 평가 탭 **손실 감시**에서 켜고 끈다.
- 켜기·끄기 스위치, 상태(켜짐·꺼짐·멈춤), 오늘 경고 수, 마지막 갱신, 마지막 회차 요약
- "패널이 켜질 때 자동으로 켜기"(기본): 재부팅 뒤 패널이 켜질 때 함께 켜진다. 감시는 `setsid`로 패널과 따로 돌아 패널을 다시 시작해도 끊기지 않는다.
- "미션 뒤 보고서": `--collect` 뒤 `--report`. 평가·미션 중에는 막힌다.
- 패널의 끄기는 이 PC의 감시만 끈다(인자를 확인한 뒤 PID로). GPU 쪽 작은 기록은 그대로 둔다.

손으로 쓸 때 (패널 설치 폴더 기준):
```bash
T=~/il_ws/omx_control_panel_v3/omx_tools
# 실시간으로 보기 (다른 창, 원할 때만)
python3 $T/omx_live_monitor.py --watch
tail -f ~/il_ws/omx_loss_tracker/live/$(date +%Y%m%d)_warn.log        # 경고만
# 미션 뒤: GPU 쪽 기록 가져오기 → 그날 평가 회차별 요약
python3 $T/omx_live_monitor.py --collect --gpu omx-gpu
python3 $T/omx_live_monitor.py --report $(date +%Y%m%d)    # live/report_YYYYMMDD.md
# GPU 쪽 기록 끄기 (필요할 때만)
python3 $T/omx_live_monitor.py --gpu-stop --gpu omx-gpu
```

## 기록 폴더
- 기록은 프로그램 위치와 상관없이 `~/il_ws/omx_loss_tracker/live/`에 쌓인다. 환경변수 `OMX_LOSS_DIR`로 바꿀 수 있다.
- 패널 기록 폴더 기본값은 `~/il_ws/omx_control_panel_v3/logs`이고, `OMX_PANEL_LOGS`로 바꿀 수 있다.
- 패널이 켤 때는 `OMX_LOSS_DIR`(설정 `"loss_monitor"`의 `live_dir`)과 `OMX_PANEL_LOGS`(패널 기록 폴더)를 넘긴다. 같은 기록 폴더에 쓰는 감시가 이미 돌고 있으면 또 켜지 않는다.
- 패널을 다시 설치해 도구 폴더가 바뀌어도 기록은 지워지지 않는다.

## 기록되는 것 (live/)
- `YYYYMMDD.jsonl`: 평가 중에는 1초마다, 쉬는 중에는 30초마다 한 줄
  - 카메라별 실제 fps (패널의 /dev/shm/omx_live_*stats.json)
  - CPU 전체와 가장 바쁜 코어, load, 메모리, omx_run CPU
  - 네트워크 장치 오류·버림, UDP InErrors·RcvbufErrors·SndbufErrors, omx_run UDP 소켓 drops
  - 커널 USB·카메라 메시지 수
  - 평가 로그: 원격 추론 왕복·GPU 추론 ms, 보낸 영상 KB, "응답 없어 다시 보냄" 누적, 루프 느림 줄, Corrupt JPEG, 카메라 읽기 실패
- `YYYYMMDD_warn.log`: 경고 줄. 경고 조건은 아래와 같다.
  - 카메라 24 fps 미만 또는 멈춤
  - CPU 코어 90% 이상
  - 버림 발생
  - 재전송 증가
  - 커널 USB 오류
- `YYYYMMDD_runs.jsonl`: 평가가 끝날 때마다 그 회차 요약
- `report_YYYYMMDD.md`: `--report` 결과. 회차별 카메라 fps 최소·중앙, 버림 수(제어 PC·GPU), 재전송, 경고 수
- `status.json`: 지금 상태. `--watch`가 이 파일을 읽는다.
- `gpu/gpu_YYYYMMDD.jsonl`: `--collect`로 가져온 GPU 쪽 1초 기록. UDP 카운터, 5600 포트 소켓 drops, 네트워크 장치, load가 들어 있다.

## 패널·AI 서버 안의 지표 (배포되면 자동으로 들어감)
- 패널(Admin 제안): 평가 로그에 5초마다 남기는 줄
  - `[omx] cams:` 카메라별 fps, 프레임 나이 p50/p95, 같은 프레임, 오류
  - `[omx] udp:` 요청, 응답, 타임아웃, 늦은 응답, 왕복
  - `[omx] loop5s:` 최저 Hz, 33ms 초과
  - 감시기가 바로 읽어서 경고를 내고, 회차 요약에 합을 넣는다.
- AI 서버(Guide_3): GPU `~/il_ws/omx_control_panel_v3/logs/udp_server.log`의 `[omx-udp] stats {json}` 줄. 10초마다 남기고 서버 시작부터 누적이다.
  - `--collect`가 이 파일도 가져온다.
  - 보고서에 회차 구간 증가분이 들어간다: 조각 받음·끝내 안 옴, 조각 모자라 시간 초과, 새 요청에 밀림, CRC·MAC 실패, 계산 건너뜀, 응답 보냄, 추론 p95.
  - 서버가 재시작해 started 값이 바뀌면 구간을 나눠 계산한다.

## 한계
- UDP 조각 손실과 재조립 실패 수는 infer_link 안에서만 알 수 있다. 지금 밖에서 보이는 것은 "응답 없어 다시 보냄"과 커널 버림뿐이다.
- 카메라 프레임 지연(찍힌 뒤 루프가 쓰기까지 ms)과 읽기 타임아웃 수는 omx_run 안에서만 알 수 있다.
- 카메라 fps는 패널 라이브 보기 파일이 갱신될 때만 보인다. 평가 중에는 0.5초마다 갱신된다.
