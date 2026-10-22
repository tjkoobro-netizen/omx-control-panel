# OMX 제어판 v1.6.1 설치 매뉴얼

OMX-AI 로봇팔(리더·팔로워)이 연결된 **GPU 노트북**을, 각자 들고 다니는 **개인 PC**의 브라우저에서 클릭으로 다루는 도구입니다. 데이터 수집(텔레옵·녹화·다시보기), 학습(ACT·Diffusion), 평가(모델로 로봇 실행·분석)를 한 화면에서 합니다. 학원 밖에서도 **Tailscale**로 GPU 노트북에 접속합니다.

- 대상: Ubuntu 22.04/24.04, LeRobot v0.4.4, OMX-AI (OpenRB-150 보드, 팔로워 모터 ID 11~16, 리더 1~6)
- 걸리는 시간: Tailscale·SSH 20분, GPU 노트북 설치 10분 (LeRobot이 없으면 +30분), 개인 PC 설치 5분

---

## 0. 한눈에 보기

```
 개인 PC (각자)                         인터넷 / 학원 Wi-Fi                     GPU 노트북 (팀마다 1대)
 ───────────────                                                                ─────────────────────────
 브라우저 http://127.0.0.1:<포트>                                                제어판 서버 (계정마다 1개)
   ▲                                                                              ├─ 텔레옵·녹화·평가 → 로봇, 카메라
 OMX 제어판 아이콘 / omx-panel  ── SSH 터널 ──▶ Tailscale (100.x.x.x) ──▶  SSH ──┤─ 학습 → GPU
                                                                                 └─ 데이터·모델 폴더
```

제어판은 GPU 노트북 안에서만(127.0.0.1) 열리고, 개인 PC는 SSH 터널로 들어갑니다. 그래서 같은 Wi-Fi의 다른 사람은 제어판에 접근할 수 없습니다.

### 어디서 무엇을 하나

| 단계 | 어디서 | 누가 | sudo | 내용 |
|---|---|---|---|---|
| [1](#1-tailscale로-연결하기) | GPU 노트북 + 개인 PC | 관리자 → 팀원 | 설치할 때 | Tailscale로 두 컴퓨터 연결 |
| [2](#2-gpu-노트북-기본-준비-관리자-한-번) | **GPU 노트북** | 관리자 | **필요** | SSH 서버, 계정, 패키지, (선택) 공유 폴더 |
| [3](#3-gpu-노트북에-제어판-설치-팀원마다) | **GPU 노트북** | 팀원마다 | 불필요 | `install_gpu.sh` (LeRobot 확인, 로봇·카메라 감지) |
| [4](#4-개인-pc에-제어판-연결-설치-팀원마다) | **개인 PC** | 팀원마다 | Tailscale 설치 때만 | `install_laptop.sh` (SSH 키, 아이콘) |
| [5](#5-처음-사용하기) | 개인 PC 브라우저 | 팀원 | - | 점검 → 수집 → 학습 → 평가 |

**관리자**는 GPU 노트북에 sudo 권한이 있는 사람(보통 노트북 주인)입니다. 팀원은 GPU 노트북에 **자기 계정**이 있어야 합니다.

### 패키지 구성 (`omx_panel_v1.6.1.zip`)

| 파일 | 어디서 쓰나 | 역할 |
|---|---|---|
| `INSTALL.md` | - | 이 매뉴얼 |
| `CHANGELOG.md` | - | 버전 기록 |
| `install_gpu.sh` | GPU 노트북 | 제어판 설치·설정 |
| `install_laptop.sh` | 개인 PC | Tailscale·SSH·아이콘 설정, GPU 쪽 원격 설치 |
| `omx_tools/` | GPU 노트북 (`omx-control.sh`만 개인 PC) | 제어판 프로그램 (안전 시작·정지 `omx_safe_start.py` 포함) |

---

## 1. Tailscale로 연결하기

### 1-1. 왜 필요한가
GPU 노트북은 학원 공유기 뒤에 있어서 밖에서 주소로 찾아갈 수 없습니다. Tailscale은 같은 계정 그룹(**tailnet**)에 로그인한 기기끼리 어디서든 `100.x.x.x` 주소로 서로 보이게 해줍니다. **GPU 노트북과 팀원들의 개인 PC가 모두 같은 tailnet에 있어야** 합니다.

### 1-2. [GPU 노트북 · 관리자 · sudo] tailnet 만들고 GPU 노트북 등록 (팀마다 한 번)
GPU 노트북 터미널에서 실행합니다.
```bash
curl -fsSL https://tailscale.com/install.sh | sh     # Tailscale 설치
sudo tailscale up                                     # 출력된 주소를 브라우저로 열어 로그인 → tailnet이 만들어짐
sudo systemctl enable --now tailscaled                # 재부팅 후에도 자동 연결
tailscale ip -4                                       # GPU 노트북 주소 (예: 100.87.250.14) 기록해 두기
```
그다음 [Tailscale 관리 화면](https://login.tailscale.com/admin/machines)에서 할 일이 있습니다.
- **Machines → GPU 노트북 → ⋯ → Disable key expiry**를 누릅니다. 이걸 하지 않으면 몇 달 뒤 인증이 만료돼 원격 접속이 끊기고, GPU 노트북 앞에서 다시 로그인해야 합니다.
- 기기 이름을 알아보기 쉽게 바꿔도 됩니다(예: `team3-gpu`).

### 1-3. [관리자] 팀원 초대
- 관리 화면 **Users → Invite users**에서 팀원 이메일을 넣어 초대합니다.
- 팀원은 초대 메일의 링크로 **같은 tailnet**에 들어옵니다.
- 무료 요금제는 사용자 수 제한이 있습니다. 인원이 많으면 관리 화면의 요금제 안내를 확인하세요.
- **팀원이 이미 자기 tailnet을 쓰고 있다면**, 초대 대신 **Machines → GPU 노트북 → ⋯ → Share**로 GPU 노트북 한 대만 공유해도 됩니다.

### 1-4. [개인 PC · 팀원] Tailscale 설치·로그인
4단계의 `install_laptop.sh`가 자동으로 해줍니다. 직접 하려면 이렇게 합니다.
```bash
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up            # 초대받은(또는 공유받은) 계정으로 로그인
tailscale status             # 목록에 GPU 노트북이 보여야 함
tailscale ping <GPU 노트북 이름 또는 100.x 주소>
```

### 1-5. Tailscale 문제 해결
| 증상 | 해결 |
|---|---|
| `tailscale status`에 GPU 노트북이 없음 | 다른 tailnet에 로그인한 것입니다. `sudo tailscale logout` 후 초대받은 계정으로 `sudo tailscale up` |
| GPU 노트북이 `offline` | GPU 노트북이 꺼졌거나 잠자기 상태, 또는 Wi-Fi가 끊긴 상태입니다(2-6 절전 끄기). 키 만료(1-2) 여부도 확인 |
| `tailscale ping`은 되는데 `ssh`가 안 됨 | GPU 노트북의 SSH 서버(2-1)와 방화벽을 확인합니다: `sudo ufw allow in on tailscale0 to any port 22 proto tcp` |
| 학교·회사망에서 느림 | 직접 연결이 막혀 중계 서버(DERP)를 거치는 상태입니다. 동작은 하지만 영상이 느릴 수 있습니다 |

---

## 2. GPU 노트북 기본 준비 (관리자, 한 번)

> **[GPU 노트북 · 관리자 · sudo]** 이 단계는 모두 GPU 노트북에서 sudo 권한으로 합니다. `install_gpu.sh`가 무엇이 빠졌는지 알려주므로, 3단계를 먼저 돌려보고 안내된 것만 해도 됩니다.

### 2-1. SSH 서버
```bash
sudo apt update && sudo apt install -y openssh-server
sudo systemctl enable --now ssh
```

### 2-2. 팀원 계정과 그룹
팀원마다 계정을 만들고 로봇·카메라 그룹에 넣습니다. 계정을 나누면 데이터가 사람별 폴더로 저장되고, 제어판 서버도 사람마다 따로 돕니다.
```bash
sudo adduser <팀원계정>
sudo usermod -aG dialout,video <팀원계정>     # dialout = 로봇 포트, video = 카메라
```
관리자 자신의 계정도 `dialout`, `video`에 넣으세요. 그룹을 바꾼 뒤에는 **다시 로그인**해야 적용됩니다.

### 2-3. 패키지
```bash
sudo apt install -y git v4l-utils acl curl python3.10 python3.10-venv python3.10-dev
```
Ubuntu 24.04에 `python3.10`이 없으면 먼저 저장소를 추가합니다: `sudo add-apt-repository ppa:deadsnakes/ppa -y && sudo apt update`

### 2-4. GPU 드라이버 확인
```bash
nvidia-smi
```
GPU 이름이 나오면 정상입니다. **"couldn't communicate with the NVIDIA driver"**가 나오면 커널이 업데이트됐는데 드라이버 모듈이 따라오지 못한 경우가 많습니다.
```bash
sudo apt install linux-modules-nvidia-<버전>-open-generic-hwe-$(lsb_release -rs)    # 예: 595, 24.04
sudo reboot
```
설치된 버전은 `dpkg -l | grep nvidia-driver`로 확인합니다.

### 2-5. (선택) 팀 공유 폴더
여러 계정이 **같은 데이터와 모델을 함께 쓰려면** 만듭니다. 혼자 쓰면 건너뛰어도 되고, 그러면 `~/omx_data`를 씁니다.
```bash
sudo groupadd -f ilteam
sudo usermod -aG ilteam <각 팀원계정>
sudo mkdir -p /srv/il/lerobot_data /srv/il/checkpoints
sudo chgrp -R ilteam /srv/il && sudo chmod -R 2775 /srv/il
sudo setfacl -R -d -m g:ilteam:rwX /srv/il
```
- `/srv/il/lerobot_data`: 녹화 데이터, 캘리브레이션
- `/srv/il/checkpoints`: 학습 결과, 학습 기록 표 `INDEX.md`
- 로봇 사용 잠금 파일도 `/srv/il`에 생기므로, 두 사람이 동시에 로봇을 움직이는 것을 막아줍니다.

### 2-6. 절전 끄기 (원격으로 쓰려면 필수)
```bash
sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target
sudo sed -i 's/^#\?HandleLidSwitch=.*/HandleLidSwitch=ignore/' /etc/systemd/logind.conf
sudo sed -i 's/^#\?HandleLidSwitchExternalPower=.*/HandleLidSwitchExternalPower=ignore/' /etc/systemd/logind.conf
sudo reboot
```
전원 어댑터를 꽂아두고, Wi-Fi는 "자동 연결"과 "모든 사용자가 사용 가능"으로 저장합니다.

### 2-7. 로봇과 카메라 연결
- 팔로워: USB + **12V 전원**. 리더: USB만 연결합니다.
- 카메라: **front**(작업대 전체가 보이는 고정 카메라), **wrist**(팔로워 그리퍼에 달린 카메라)
- 확인:
  ```bash
  ls /dev/serial/by-id/                  # OpenRB-150 두 개
  v4l2-ctl --list-devices                # 카메라 목록
  ```
- (선택) udev로 `/dev/omx_follower`, `/dev/omx_leader` 고정 이름을 만들어도 됩니다(강의자료 방식). **안 만들어도** 설치 스크립트가 `/dev/serial/by-id/…` 고정 경로를 찾아 씁니다.

---

## 3. GPU 노트북에 제어판 설치 (팀원마다)

> **[GPU 노트북 · 내 계정 · sudo 불필요]** 제어판을 쓸 **각 계정으로** 한 번씩 합니다.
> **4단계의 `install_laptop.sh`가 이 단계를 개인 PC에서 원격으로 대신 실행해줄 수 있습니다.** GPU 노트북 앞에 있다면 여기서 직접 해도 됩니다.

### 3-1. 패키지를 GPU 노트북으로 옮기기
GPU 노트북에서 zip을 받아 풀거나, 개인 PC에서 복사합니다.
```bash
# 개인 PC에서 (Tailscale 연결 후)
scp omx_panel_v1.6.1.zip <내계정>@<GPU 노트북 100.x 주소>:~
# GPU 노트북에서
unzip -o ~/omx_panel_v1.6.1.zip -d ~ && cd ~/omx_panel_v1.6.1
```

### 3-2. 설치 실행
```bash
bash install_gpu.sh            # 질문에 답하며 설치
bash install_gpu.sh --yes      # 감지한 값으로 자동 설치 (다시 설치·업데이트할 때 편함)
```

| 단계 | 하는 일 | 내가 할 일 |
|---|---|---|
| 1 환경 점검 | OS, GPU, SSH 서버, 그룹, Tailscale 확인 | 빠진 것은 마지막에 sudo 명령으로 안내됩니다. 관리자에게 전달하세요 |
| 2 LeRobot | 가상환경(`~/venv/il`)에 LeRobot이 있는지 확인. **없으면 설치할지 묻습니다**(약 4GB, 20~40분) | 이미 다른 위치에 있으면 그 경로를 입력 |
| 3 파일 복사 | `~/il_ws/src/omx_tools`에 설치, `omx-panel` 명령 등록 | - |
| 4 로봇 포트 | 각 포트에 모터 신호만 보내서 ID 11~16이면 팔로워, 1~6이면 리더로 자동 구분. **로봇은 움직이지 않습니다** | 못 찾으면 USB·12V 전원을 확인하고, 다른 프로그램(텔레옵 등)을 끈 뒤 다시 실행 |
| 5 카메라 | 연결된 카메라 목록을 보여줍니다. 노트북 내장 카메라는 표시됩니다 | **front, wrist 번호를 직접 확인해서 고르세요.** 자동으로 앞뒤를 알 수 없습니다. 헷갈리면 `v4l2-ctl --list-devices`로 하나씩 뽑아보며 확인 |
| 6 폴더 | `/srv/il`(2-5)이 있으면 팀 공유, 없으면 개인 폴더 `~/omx_data` | 팀 공유를 원하면 관리자가 2-5를 한 뒤 다시 실행 |
| 7 저장·시작 | `~/.config/omx/config.json`에 설정을 저장하고 제어판 서버를 시작 | 마지막에 나오는 **포트 번호**를 확인 |

- 다시 실행하면 **이전 설정이 기본값**이 됩니다. 그래서 업데이트할 때 `--yes`로 돌려도 됩니다.
- 이전 설정에 있던 카메라가 잠깐 빠져 있어도 설정은 지우지 않고 경고만 합니다.
- 녹화·텔레옵·학습이 실행 중이면 서버를 재시작하지 않습니다.

### 3-3. 설정 확인과 수정
```bash
~/venv/il/bin/python ~/il_ws/src/omx_tools/omx_config.py    # 실제로 쓰는 설정과 내 포트
nano ~/.config/omx/config.json                              # 직접 고친 뒤: omx-panel --restart
```

| 항목 | 뜻 | 예 |
|---|---|---|
| `gpu_name` | 제어판 상단에 보이는 이름 | `team16-PC` |
| `data_home` | 녹화 데이터·캘리브레이션 (LeRobot 홈) | `/srv/il/lerobot_data` |
| `checkpoints` | 학습 결과·학습 기록 표 | `/srv/il/checkpoints` |
| `shared_root` | 로봇·학습 잠금 파일 위치 (모든 계정이 쓸 수 있어야 함) | `/srv/il` |
| `team_group` | 녹화·모델에 읽기·쓰기를 줄 그룹, 빈칸이면 권한을 바꾸지 않음 | `ilteam` |
| `follower_port`, `leader_port` | 로봇 포트 | `/dev/serial/by-id/usb-ROBOTIS_OpenRB-150_…-if00` |
| `cameras` | 이름 → 카메라 경로. 이름이 데이터 키(`observation.images.<이름>`)가 됨 | `{"front": "/dev/v4l/by-id/…", "wrist": "…"}` |
| `fixed_fps_camera` | 어두워도 30fps를 유지하도록 설정할 카메라 | `wrist` |
| `venv` | LeRobot 가상환경 | `~/venv/il` |
| `port_base`, `uid_base` | 계정별 포트 = `port_base + uid - uid_base` | `8765`, `1001` |
| `safe_start` | 안전 시작·정지 세부값 (없으면 기본값). `enabled`, `pwm_ratio`(시작할 때 출력 제한, 0.3), `lift_speed`·`approach_speed`·`catch_up_speed`·`stop_speed`(초당 단위, 12·15·60·12), `leader_move`(이만큼 리더가 움직이면 원래 출력으로, 3), `max_gap`(이보다 차이가 크면 시작 거부, 90 ≈ 160°), `gripper_close`(47), `stop_ramp_s`(3초) | `{"pwm_ratio": 0.3}` |

> **주의**: 카메라 이름(`front`, `wrist`)은 학습한 모델의 입력 이름과 같아야 합니다. 이미 학습한 모델이 있으면 이름을 바꾸지 마세요.

---

## 4. 개인 PC에 제어판 연결 설치 (팀원마다)

> **[개인 PC · 내 계정]** Ubuntu 개인 PC 기준입니다. Windows·macOS는 [부록 A](#부록-a-windows--macos-개인-pc에서-연결)를 보세요.

```bash
unzip -o omx_panel_v1.6.1.zip && cd omx_panel_v1.6.1
bash install_laptop.sh
```

| 단계 | 하는 일 | 내가 할 일 |
|---|---|---|
| 1 Tailscale | 없으면 설치하고, 로그인 전이면 `sudo tailscale up` | **sudo 비밀번호** 입력, 브라우저에서 초대받은 계정으로 로그인 |
| 2 GPU 노트북 고르기 | 같은 tailnet의 기기 목록을 켜짐·꺼짐 상태와 함께 보여줌 | GPU 노트북 번호 선택, **GPU 노트북의 내 계정 이름** 입력 |
| 3 SSH 키 | 키가 없으면 만들고 GPU 노트북에 등록 | **GPU 노트북 내 계정의 비밀번호**를 한 번 입력 |
| 4 별칭 | `~/.ssh/config`에 `omx-gpu` 등록. 다른 설정은 건드리지 않음 | - |
| 5 GPU 쪽 설치 | GPU 노트북에 제어판이 없으면 **여기서 원격으로 `install_gpu.sh`를 실행**할지 묻습니다 | 원격으로 하면 3단계의 질문이 이 터미널에 나옵니다 |
| 6 실행기 | 바탕화면 **"OMX 제어판"** 아이콘, `omx-panel` 명령 | - |
| 7 Rerun (선택) | 녹화·평가 중 관절 그래프를 보는 뷰어 | 필요할 때만 |

설치가 끝나면 이렇게 접속을 확인합니다.
```bash
ssh omx-gpu 'hostname'       # 비밀번호 없이 GPU 노트북 이름이 나오면 성공
```

---

## 5. 처음 사용하기

1. **[개인 PC]** 바탕화면의 **OMX 제어판**을 더블클릭하거나 터미널에서 `omx-panel`을 실행합니다.
   - 처음엔 아이콘을 우클릭해 **실행 허용(Allow Launching)**을 눌러야 할 수 있습니다.
   - "연결하는 중" 알림 뒤 크롬 새 창이 열립니다. 상단에 **GPU 노트북 이름과 내 계정**이 맞는지 확인하세요.
2. **① 데이터 수집** 탭
   - **점검하기**: 로봇 포트, 모터 12개, 카메라가 모두 정상인지 봅니다.
   - **카메라 켜기**: 영상을 확인합니다. 기존 모델로 평가할 거라면 **카메라 위치 맞추기**로 학습 때 구도와 맞춥니다.
   - **시작 자세·끝 자세 저장(처음 한 번, 로봇마다)**: 텔레옵 칸의 "안전 시작 · 안전 정지"에서 저장합니다. 리더암을 손으로 **팔을 든 자세**로 잡고 "시작 자세로 저장", **팔을 접어 내려놓은 자세**로 잡고 "끝 자세로 저장"을 누릅니다. 리더 위치만 읽고 로봇은 움직이지 않습니다.
   - 안전 확인을 체크하고 **텔레옵 시작**으로 동작을 확인합니다.
   - **녹화 시작**으로 학습 데이터를 모읍니다. 데이터셋 이름은 날짜·시간 `MMDD_HHMM`으로 자동 입력됩니다.
3. **② 학습** 탭: 데이터 고르기 → 영상을 보며 실패한 에피소드 빼기 → ACT(추천) → 바로 시작 (여러 개를 이어서 하려면 계획에 추가 → 계획 실행)
4. **③ 평가** 탭: 모델 고르기 → 평가 시작 → 시도마다 성공·실패 표시 → 분석 자료 만들기

각 기능 설명은 제어판 오른쪽 위 **사용법**이나 각 칸의 **?** 버튼으로 봅니다.

> **안전 시작·정지**: 텔레옵·녹화·평가를 시작하면 팔로워가 바로 리더 자세로 뛰어가지 않습니다. ① 시작 자세로 팔을 들고 ② 그리퍼를 닫은 뒤 ③ 리더 자세로 천천히(출력 30%) 다가갑니다. 다가가는 중에 리더를 잡고 움직이면 바로 원래 출력으로 따라잡습니다. 정지하면 ① 끝 자세로 천천히 옮기고 ② 출력을 3초에 걸쳐 줄인 뒤 토크를 꺼서 팔이 떨어지지 않습니다. 다가가는 몇 초 동안은 리더를 크게 움직이지 마세요.
>
> **안전**: 텔레옵·녹화·평가는 로봇을 움직입니다. **로봇 옆에 사람이 있을 때만** 하고, 팔로워 12V 전원을 바로 뽑을 수 있게 두세요. 원격에서는 카메라 화면만으로 주변을 다 볼 수 없습니다.

---

## 6. 여러 사람이 같은 GPU 노트북을 쓸 때

| 항목 | 동작 |
|---|---|
| 제어판 서버 | **계정마다 따로**, 포트도 다릅니다(예: uid 1001 → 8765, 1002 → 8766). 녹화 데이터는 `<데이터 폴더>/<계정>/`에 저장됩니다 |
| 로봇 사용 | 공유 잠금으로 **한 번에 한 사람만**. 다른 사람이 쓰는 중이면 "다른 사람이 로봇을 쓰고 있습니다"와 이름이 표시됩니다 |
| GPU 학습 | 팀 전체에서 **한 번에 한 학습만**. 다른 학습이 돌면 시작 버튼이 잠깁니다 |
| 공유 폴더(2-5) | 데이터·모델·학습 기록 표를 팀이 함께 봅니다. 남의 데이터는 다시보기·학습에 쓸 수 있지만, 이어서 녹화할 수는 없습니다 |
| Hugging Face 업로드 | 계정마다 GPU 노트북에서 `~/venv/il/bin/hf auth login`을 한 번 해야 합니다(Write 토큰). 토큰은 대화창 등에 붙여넣지 마세요 |

---

## 7. 업데이트와 제거

### 업데이트 (새 버전 zip을 받았을 때)
**[개인 PC]** 압축을 풀고 다시 실행하면, 5단계에서 GPU 노트북 쪽도 업데이트할지 묻습니다.
```bash
bash install_laptop.sh
```
GPU 노트북에서 직접 할 때는 이렇게 합니다.
```bash
bash install_gpu.sh --yes        # 이전 설정 유지, 서버 재시작
```
실행 중인 녹화·학습이 있으면 서버 재시작을 건너뜁니다. 끝난 뒤 `omx-panel --restart`를 실행하세요.

### 제거
```bash
# GPU 노트북 (내 계정)
pkill -u $USER -f "python omx_web.py"; rm -rf ~/il_ws/src/omx_tools ~/.config/omx ~/.local/bin/omx-panel
# 개인 PC
rm -rf ~/.local/share/omx ~/.config/omx ~/.local/bin/omx-panel ~/.local/share/applications/omx-control.desktop ~/Desktop/omx-control.desktop
# (~/.ssh/config 의 Host omx-gpu 블록은 직접 지우기)
```
녹화 데이터와 모델(`data_home`, `checkpoints`)은 지우지 않습니다.

---

## 8. 문제 해결

| 증상 | 확인할 것 |
|---|---|
| 아이콘을 눌러도 반응 없음 | 우클릭 → 실행 허용. 터미널에서 `omx-panel`을 실행해 메시지를 확인 |
| "Tailscale이 연결되어 있지 않습니다" | 개인 PC에서 `sudo tailscale up` |
| "GPU 노트북에서 제어판 서버를 켜지 못했습니다" | `ssh omx-gpu`가 비밀번호 없이 되는지 확인. GPU 노트북의 `~/il_ws/omx_logs/web.log` 마지막 줄 확인 |
| 제어판 상단의 계정이 내 계정이 아님 | `~/.ssh/config`의 `Host omx-gpu`에 있는 `User`가 맞는지 확인 |
| 점검: 포트 없음 | USB·12V 전원 확인. 포트 이름이 바뀌었으면 `install_gpu.sh`를 다시 실행 |
| 점검: 모터 "응답 없음" | 12V 전원, 케이블 확인. 텔레옵 등 다른 프로그램이 포트를 쓰고 있지 않은지 확인 |
| 텔레옵 시작 때 팔로워가 이상한 방향으로 들림 | 저장한 시작 자세가 부적절합니다. 리더를 팔을 든 자세로 잡고 "시작 자세로 저장"을 다시 누릅니다. 공유 폴더의 `omx_start_pose.json`을 지우면 들어올리기를 건너뜁니다 |
| 정지할 때 팔이 끝 자세로 가다 어딘가에 걸림 | 끝 자세를 다시 저장하거나 `omx_rest_pose.json`을 지웁니다. 지우면 그 자리에서 출력만 서서히 줄입니다 |
| 1축(어깨 회전)이 잘 안 돌거나 1축에 하드웨어 에러 | 리더나 팔로워 1축을 전원이 켜진 채 경계 너머로 돌리면 한 바퀴(360°) 어긋나게 읽힙니다. v1.4.1부터 자동 보정합니다. 이미 에러가 났다면 팔을 내려둔 채 **모터 초기화** |
| 학습이 중간에 멈추고 로그에 `DataLoader worker ... killed by signal: Killed` (화면·소리가 같이 꺼지기도 함) | GPU가 아니라 **시스템 RAM 부족**입니다. 학습 탭의 예상 RAM이 노랑·빨강이면 **데이터 로딩 수**를 먼저 줄이고, 그래도 부족하면 배치를 줄이세요 (**권장값 적용** 버튼). Diffusion은 배치 16·로딩 4(약 8GB)를 권장합니다 |
| 평가 중 로봇이 뚝뚝 끊기고 "제어 속도 ○Hz / 목표 30Hz" 경고 | CPU를 다른 작업이 쓰고 있거나 스레드가 너무 많습니다. 학습을 멈추고, 평가 설정의 **CPU 스레드 수**를 기본값 4(또는 2·8)로 두세요. 24로 두면 약 23Hz로 떨어집니다 |
| "리더와 팔로워의 ○○ 차이가 너무 큽니다"로 시작 안 됨 | 리더를 팔로워와 비슷한 자세로 맞춘 뒤 다시 시작합니다 |
| 점검: 모터 "하드웨어 에러" | 과부하로 보호 정지된 상태입니다. **팔을 내려둔 상태에서** 복구 칸의 **모터 초기화**를 누르거나, 팔로워 전원을 껐다 켭니다 |
| 카메라 "검은 화면" | 렌즈 캡, 가림, USB 확인. 카메라를 다시 꽂았다면 경로가 바뀌지 않았는지 `v4l2-ctl --list-devices`로 확인 |
| `failed to set fps=30` | 카메라가 해당 해상도에서 30fps를 못 냅니다. 설정의 `camera_width/height`를 낮추거나 다른 카메라를 씁니다 |
| 녹화 영상이 다른 계정에서 안 열림 (PermissionError) | `team_group`이 설정돼 있는지 확인. 기존 데이터는 `setfacl -R -m g::rwX,g:<그룹>:rwX,m::rwX <폴더>`로 고침 |
| 학습: GPU를 안 씀 / `cuda False` | 2-4 GPU 드라이버 |
| 평가: 로봇이 어색하게 움직이거나 그리퍼를 덜 벌림 | 카메라 위치(학습 때와 같은지), **부드럽게 실행** 옵션, 평가 탭의 **분석 자료 만들기**로 확인 |
| 평가를 **즉시 중단**으로 끝낸 뒤 관절 기록이 없음 | 즉시 중단은 저장을 건너뜁니다. 평소에는 **평가 끝내기**를 쓰세요 |
| 브라우저에 "제어판 서버와 연결이 끊겼습니다" | 개인 PC가 잠들었다 깨면 SSH 터널이 끊깁니다. 아이콘을 다시 누르세요 |

로그 위치는 다음과 같습니다.
- GPU 노트북 `~/il_ws/omx_logs/`: `web.log`(서버), 실행할 때마다 생기는 `*_teleop.log`, `*_record.log`, `*_eval.log`, `train_*.log`
- 브라우저: 각 탭의 로그 칸

---

## 부록 A. Windows / macOS 개인 PC에서 연결
설치 스크립트는 Ubuntu용이지만, 아래처럼 하면 같은 제어판을 쓸 수 있습니다.
1. [tailscale.com/download](https://tailscale.com/download)에서 앱을 설치하고, 초대받은 계정으로 로그인합니다.
2. 터미널(Windows는 PowerShell)에서 SSH 키를 만들어 등록합니다.
   ```bash
   ssh-keygen -t ed25519
   # Windows: type $env:USERPROFILE\.ssh\id_ed25519.pub | ssh <계정>@<100.x 주소> "cat >> ~/.ssh/authorized_keys"
   ssh-copy-id <계정>@<100.x 주소>       # macOS
   ```
3. 제어판을 열 때마다 이렇게 합니다.
   ```bash
   ssh <계정>@<100.x 주소> "bash il_ws/src/omx_tools/omx-panel-server.sh --port-only"   # 포트 번호가 출력됨 (예: 8766)
   ssh -N -L 8766:127.0.0.1:8766 <계정>@<100.x 주소>                                  # 이 창은 열어둠
   ```
   브라우저에서 `http://127.0.0.1:8766`을 엽니다.
4. **VS Code Remote-SSH**를 쓰면 더 간단합니다. GPU 노트북에 접속한 뒤 터미널에서 `omx-panel`을 실행하고, **포트(Ports)** 탭에서 지구본 아이콘을 누릅니다.

## 부록 B. 이 도구가 대신 처리하는 LeRobot v0.4.4 문제
| 문제 | 처리 |
|---|---|
| 녹화가 끝날 때 영상 저장 단계에서 멈춤 (카메라 2대 동시 인코딩) | 녹화 래퍼가 안전한 방식(spawn)으로 인코딩 |
| 리더 없이 정책을 평가할 때 원위치 시간이 끝나지 않음 | 래퍼가 원위치 시간을 직접 계산 |
| 원위치 시간에 녹화를 끝내면 빈 에피소드 저장 에러 | 빈 에피소드는 건너뜀 |
| 텔레옵·녹화를 시작하면 팔로워가 리더 자세로 전속력·최대 힘으로 이동 (손목 링크 파손 사례) | 안전 시작: 시작 자세로 들기 → 그리퍼 닫기 → 출력 40%로 천천히 접근 |
| 정지하면 토크가 바로 꺼져 팔이 떨어짐 | 안전 정지: 끝 자세로 이동 → 출력을 서서히 줄인 뒤 토크 끄기 |
| `--dataset.episodes`로 일부만 학습하면 Diffusion이 깨짐 | 뺄 에피소드를 지운 학습용 데이터셋을 새로 만들어 학습 |
| 녹화 영상이 600 권한이라 다른 계정이 못 읽음 | 녹화 후 팀 그룹 읽기·쓰기 권한 설정 |
| SSH·원격 실행에서 방향키 단축키가 안 됨 | 제어판 버튼, 또는 터미널에 `next`/`rerecord`/`stop` 입력 |

## 부록 C. 터미널에서 직접 평가할 때 (제어판 없이)
```bash
source ~/venv/il/bin/activate && cd ~/il_ws/src/lerobot
python ~/il_ws/src/omx_tools/omx_record.py \
  --robot.type=omx_follower --robot.port=<팔로워 포트> --robot.id=omx_follower_arm \
  --robot.cameras="$(~/venv/il/bin/python -c 'import sys; sys.path.insert(0, "'$HOME'/il_ws/src/omx_tools"); import omx_config; print(omx_config.cameras_arg())')" \
  --policy.path=<모델 폴더>/pretrained_model \
  --dataset.repo_id=$USER/eval_test_$(date +%m%d_%H%M) --dataset.single_task="<학습 때 작업 문장>" \
  --dataset.num_episodes=5 --dataset.episode_time_s=30 --dataset.reset_time_s=15 --dataset.push_to_hub=false
```
실행 중에는 `next`, `rerecord`, `stop`을 입력하고 Enter를 누릅니다.
