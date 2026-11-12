# OMX Control Panel v3.14.4 설치 가이드

처음 설치하는 팀원과 GPU 노트북 관리자를 위한 안내입니다. Ubuntu 22.04/24.04 기준이며, 전체는 처음이면 1시간 남짓 걸립니다(대부분 LeRobot 내려받기).

## 1. 먼저 내 경우 고르기

| 내 경우 | 할 일 |
|---|---|
| 이미 설치된 팀 GPU 노트북을 같이 쓰고, 내 PC에 설치한다 | [2](#2-누가-무엇을-해-주어야-하나)의 준비물을 받고 [4](#4-팀원-내-pc에-설치-제어-pc)로 |
| 새 GPU 노트북을 팀에 추가한다, 또는 내 GPU 노트북으로 혼자 쓴다 | [3](#3-관리자-새-gpu-노트북-준비)부터 |
| GPU 노트북을 다른 것으로 바꾼다 | [6](#6-gpu-노트북이-다를-때) |
| 이미 설치했고 새 버전으로 올린다 | [8](#8-업데이트) |

전체 구조는 이렇습니다. 로봇은 **내 PC나 GPU 노트북 어느 쪽에 꽂아도** 됩니다. 아이콘을 누르면 패널이 로봇 위치를 보고 맞게 열립니다.

```
 내 PC (제어 PC)                                   GPU 노트북
 ───────────────                                   ──────────
 브라우저 ─ 제어판                ── Tailscale ──▶  학습, 팀 공유 데이터
 (로봇을 꽂으면 이쪽에서 구동)       + SSH          (로봇을 꽂으면 이쪽에서 구동)
                                 ── 같은 공유기 ──▶  AI 추론 (UDP 서버, GPU 제어판이 켬)
                                    UDP
```

AI 추론(평가 중 동작 계산)은 **UDP로만** 주고받습니다. 두 PC가 같은 공유기에 있어야 하고, 같은 키 파일이 양쪽에 있어야 합니다(5단계 8번).

## 2. 누가 무엇을 해 주어야 하나

### 담당별 권한

| 담당 | 필요한 권한 | 하는 일 |
|---|---|---|
| **Tailscale 관리자** | 팀 Tailscale(tailnet)의 Owner 또는 Admin | 팀원 초대, 새 기기 승인(켜 둔 경우), GPU 노트북의 키 만료 끄기 |
| **GPU 노트북 관리자** | GPU 노트북의 `sudo` | 팀원 계정 만들기, 그룹 추가, SSH 서버, 공유 폴더, 장치 이름 규칙, 절전 끄기, 방화벽(ufw)을 쓰면 팀원 PC → 그 계정의 AI 추론 UDP 포트 허용 |
| **팀원 본인** | 내 PC의 `sudo` | 내 PC 설치, 내 GPU 계정 비밀번호 변경, Hugging Face 로그인(업로드할 때만) |
| **저장소 관리자** | GitHub 저장소 쓰기 권한 | 코드 수정과 배포. 받는 쪽은 권한이 필요 없습니다(공개 저장소) |

한 사람이 여러 역할을 맡아도 됩니다.

### 팀원이 설치 전에 받아야 할 것

아래를 관리자에게 요청합니다. 그대로 복사해 보내도 됩니다.

> OMX 제어판을 설치하려고 합니다.
> 1. **Tailscale 관리자님**: `<내 이메일>`을 팀 Tailscale에 초대해 주세요.
> 2. **GPU 노트북 관리자님**: GPU 노트북에 `<계정 이름>` 계정을 만들고 `dialout`, `video`, `ilteam` 그룹에 넣어 주세요. 초기 비밀번호는 저에게 직접 전달해 주세요.

체크 목록:
- [ ] Tailscale 초대 메일을 받았다
- [ ] GPU 노트북의 내 계정 이름과 초기 비밀번호를 받았다
- [ ] 내 PC에서 `sudo`를 쓸 수 있다
- [ ] 내 PC에 디스크 여유가 10GB 이상 있다

비밀번호와 토큰은 공개 저장소, 단체 채팅, AI 대화창에 적지 마세요.

## 3. 관리자: 새 GPU 노트북 준비

GPU 노트북마다 **한 번만** 합니다. 이미 쓰고 있는 노트북이면 [3-3](#3-3-팀원마다-계정-만들기)만 팀원마다 합니다.

### 3-1. Tailscale (Tailscale 관리자 + GPU 노트북 관리자)

GPU 노트북 터미널에서:
```bash
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up          # 나온 주소를 브라우저로 열어 팀 Tailscale 계정으로 로그인
sudo systemctl enable --now tailscaled
tailscale ip -4            # 100.x.x.x 주소 기록
```
그다음 [Tailscale 관리 화면](https://login.tailscale.com/admin/machines)에서:
1. **Machines → GPU 노트북 → ⋯ → Disable key expiry**. 이걸 안 하면 몇 달 뒤 인증이 만료돼 원격 접속이 끊깁니다.
2. 기기 승인을 켜 둔 팀이면 **Approve**를 누릅니다.
3. 알아보기 쉽게 이름을 바꿉니다. 이름에 `gpu`나 `team`이 있으면 설치 프로그램이 기본값으로 제안합니다.

**팀원 초대**: **Users → Invite users**에 이메일을 넣습니다. 사용자 수 제한은 요금제 화면에서 확인합니다.
팀원이 이미 자기 Tailscale을 쓰고 계속 쓰고 싶다면, 초대 대신 **Machines → GPU 노트북 → ⋯ → Share**로 이 노트북만 공유해도 됩니다.

### 3-2. 기본 설정 (GPU 노트북 관리자, `sudo`)

```bash
sudo apt update
sudo apt install -y openssh-server git curl rsync v4l-utils acl python3.12 python3.12-venv python3.12-dev
sudo systemctl enable --now ssh
nvidia-smi                 # GPU 이름이 나와야 합니다
```
Ubuntu 22.04는 `python3.12`가 없어 먼저 `sudo add-apt-repository -y ppa:deadsnakes/ppa && sudo apt update`를 합니다.
`nvidia-smi`가 실패하면 NVIDIA 드라이버부터 설치합니다. NVIDIA GPU가 없으면 학습과 AI 추론을 할 수 없습니다.

**공유 폴더** (여럿이 같은 데이터와 모델을 쓸 때):
```bash
sudo groupadd -f ilteam
sudo mkdir -p /srv/il/lerobot_data /srv/il/checkpoints
sudo chgrp -R ilteam /srv/il && sudo chmod -R 2775 /srv/il
sudo setfacl -R -d -m g:ilteam:rwX /srv/il
```

**절전 끄기** (원격으로 쓰려면 필수):
```bash
sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target
sudo sed -i 's/^#\?HandleLidSwitch=.*/HandleLidSwitch=ignore/' /etc/systemd/logind.conf
sudo sed -i 's/^#\?HandleLidSwitchExternalPower=.*/HandleLidSwitchExternalPower=ignore/' /etc/systemd/logind.conf
sudo reboot
```
충전기를 꽂아 두고, Wi-Fi는 "자동 연결"과 "모든 사용자가 사용 가능"으로 저장합니다.

**로봇을 이 노트북에 꽂을 때만, 장치 이름 규칙** (저장소를 받은 폴더에서):
```bash
sudo cp udev/99-omx.rules /etc/udev/rules.d/ && sudo udevadm control --reload-rules && sudo udevadm trigger --subsystem-match=tty
ls -l /dev/omx_*           # omx_follower_N, omx_leader_N이 보여야 합니다
```
규칙에 없는 새 보드는 이름이 생기지 않습니다. [6](#6-gpu-노트북이-다를-때)을 보세요.

### 3-3. 팀원마다 계정 만들기 (GPU 노트북 관리자, `sudo`)

```bash
sudo adduser <팀원계정>
sudo usermod -aG dialout,video,ilteam <팀원계정>
id <팀원계정>              # dialout, video, ilteam이 모두 보여야 합니다
```

| 그룹 | 없으면 |
|---|---|
| `dialout` | 로봇 포트를 열 수 없음 |
| `video` | 카메라를 열 수 없음 |
| `ilteam` | 공유 폴더의 데이터와 모델을 쓸 수 없음 |

그룹을 바꾼 뒤에는 **그 팀원이 다시 로그인**해야 적용됩니다. `sudo` 그룹은 관리자가 아니면 넣지 않아도 됩니다.
팀원은 처음 접속한 뒤 `passwd`로 비밀번호를 바꿉니다.

## 4. 팀원: 내 PC에 설치 (제어 PC)

### 4-1. Tailscale 초대 수락
초대 메일의 링크를 열어 **팀 Tailscale**에 들어갑니다. 개인 Tailscale에 로그인되어 있으면 GPU 노트북이 보이지 않습니다.

### 4-2. 받아서 설치
```bash
sudo apt install -y git
git clone https://github.com/tjkoobro-netizen/omx-control-panel.git ~/omx-control-panel
cd ~/omx-control-panel && bash install_control.sh
```

### 4-3. 설치 중 묻는 것

| 단계 | 하는 일 | 내가 할 일 |
|---|---|---|
| 1/8 시스템 패키지 | 빠진 프로그램 설치 | 내 PC `sudo` 비밀번호 입력 |
| 2/8 Tailscale | 없으면 설치, 로그인 | 나온 주소를 브라우저로 열어 **초대받은 계정**으로 로그인 |
| 3/8 GPU 노트북 | 기기 목록에서 선택, SSH 키 등록 | GPU 노트북 번호 선택, **GPU 계정 이름**과 **GPU 계정 비밀번호** 한 번 입력 |
| 4/8 GPU 쪽 설치 | GPU의 내 계정에 패널이 없거나 버전이 다르면 원격 설치 | **Y**. 카메라 등 질문은 기본값으로 두고, 나중에 패널에서 고칩니다. 공유 폴더가 있으면 자동으로 씁니다. LeRobot이 없으면 20~40분 걸립니다 |
| 5/8 LeRobot | 내 PC에 LeRobot 0.6.1 설치 (약 6GB, 20~40분) | Y |
| 6/8 장치 이름과 USB 권한 | 장치 이름 규칙 설치, `dialout` 추가 | `sudo` 비밀번호 입력 |
| 7/8 패널 설치 | 설정 저장 | 없음 |
| 8/8 아이콘 | 바탕화면 **OMX Control Panel v3** | 없음 |

`dialout` 권한이 새로 추가됐다고 나오면 **로그아웃 후 다시 로그인**합니다.

### 4-4. 설치 확인
```bash
tailscale status | head -5    # GPU 노트북이 목록에 있고 offline이 아니어야 합니다
ssh omx-gpu hostname          # 비밀번호 없이 GPU 노트북 이름이 나와야 합니다
```

### 4-5. 팀 이름으로 표시 (선택)

패널의 번호는 기본으로 보드 번호를 따릅니다. 우리 팀 호칭(인도장 팔 = OMX 1, 2)으로 보려면, 로봇을 꽂는 쪽에서 한 번 실행합니다. 패널이 돌 때 실행하면 켜진 패널이 다시 시작됩니다. 녹화, 텔레옵, 평가 중에는 하지 마세요.
```bash
# 로봇을 내 PC에 꽂을 때
~/venv/il3/bin/python ~/il_ws/omx_control_panel_v3/omx_tools/omx_team_names.py
bash ~/il_ws/omx_control_panel_v3/omx_tools/omx-panel-server.sh --restart

# 로봇을 GPU 노트북에 꽂을 때
ssh omx-gpu '~/venv/il3/bin/python ~/il_ws/omx_control_panel_v3/omx_tools/omx_team_names.py && bash ~/il_ws/omx_control_panel_v3/omx_tools/omx-panel-server.sh --restart'
```

| 팀 이름 | 보드 장치 |
|---|---|
| OMX 1 | `/dev/omx_follower_2`, `/dev/omx_leader_2` |
| OMX 2 | `/dev/omx_follower_3`, `/dev/omx_leader_3` |
| OMX 3 | `/dev/omx_follower_1`, `/dev/omx_leader_1` |

이름만 바뀌고 팔은 그대로입니다. 설정을 손으로 고치지 마세요. 숫자를 잘못 맞추면 다른 팔이 움직일 수 있습니다. 시뮬레이터 화면과 장치 이름은 보드 번호를 그대로 씁니다.

## 5. 처음 실행과 동작 확인

**로봇이 움직이는 시험은 로봇 옆에서 합니다.** 팔 주변을 비우고, 비상 정지 버튼 위치를 확인하세요.

1. 바탕화면 **OMX Control Panel v3**를 더블클릭합니다. 처음에는 우클릭 → **실행 허용**이 필요할 수 있습니다.
2. 왼쪽 위 버전이 v3.14.4인지, 오른쪽 위 계정이 내 계정인지 확인합니다.
3. **① 데이터 수집 → 로봇 구성 → 구성 바꾸기**에서 왼팔·오른팔과 카메라 역할을 고릅니다.
4. **점검하기**: 포트, 모터 12개, 카메라가 모두 정상인지 봅니다.
5. **카메라 켜기**: 네 대가 모두 나오는지 봅니다.
6. **텔레옵**: 시작하면 팔이 리더 자세로 천천히 간 뒤 따라옵니다. **중지**하면 시작 자세로 돌아가 손목 카메라가 위를 향하고, 그리퍼를 닫은 뒤 힘이 빠집니다.
7. **녹화**: 짧게 2개를 찍어 ② 학습 탭 목록에 나오는지 봅니다.
8. **AI 추론 통신(UDP)**: ③ 평가 탭 맨 위 **AI 추론 통신 · UDP**에 GPU 노트북의 공유기 주소와 포트(빈 칸에 보이는 예시 포트, 계정마다 다름)를 적고, **키 만들기·GPU에 복사**를 누른 뒤 **저장**합니다. 옆에 "연결됨 · 왕복 몇 ms"가 나오면 됩니다. GPU 노트북의 UDP 서버는 그쪽 제어판이 자동으로 켭니다. 안 되면 빨간 글씨로 어디서 막혔는지 나옵니다(방화벽이면 GPU 노트북 관리자에게 화면의 `sudo ufw allow …` 명령을 전달).
9. **평가**: 모델을 고르고 시작합니다. 먼저 모델의 처음 자세로 이동합니다. 두 번째 시도부터 첫 동작까지 천천히 다가갑니다.

이상하게 끝나면 위쪽에 빨간 **err_code_N**이 나옵니다. 누르면 원인과 조치를 볼 수 있습니다. 전체 목록은 **사용법 → 오류 코드**에 있습니다.

## 6. GPU 노트북이 다를 때

| 상황 | 달라지는 점 | 할 일 |
|---|---|---|
| **다른 GPU 노트북으로 바꿈** | 내 PC의 `omx-gpu` 별칭이 새 노트북을 가리켜야 합니다 | 새 노트북에 내 계정을 받은 뒤 `bash install_control.sh`를 다시 실행하고, 3/8에서 **다른 노트북으로 바꾼다**를 고릅니다 |
| **데이터와 모델** | GPU 노트북마다 따로 저장되고 **자동으로 옮겨지지 않습니다** | 필요한 데이터를 옮기거나 Hugging Face로 받습니다. 모델 평가에는 그 모델과 학습 데이터 정보가 있는 노트북을 씁니다 |
| **공유 폴더 없음** | 설치가 개인 폴더 `~/omx_data`로 진행됩니다 | 팀이 함께 쓰려면 [3-2](#3-2-기본-설정-gpu-노트북-관리자-sudo)로 만든 뒤 설치를 다시 실행합니다 |
| **다른 로봇 보드** | 저장소 규칙은 현재 팀 보드 일련번호만 압니다. 다른 보드에는 `/dev/omx_*_N`이 생기지 않습니다 | 일련번호를 `ls /dev/serial/by-id/`로 확인해 저장소 관리자에게 전달합니다. 규칙에 추가된 뒤 다시 설치합니다 |
| **GPU 메모리가 다름** | 16GB 기준으로 맞춘 설정입니다 | 메모리 부족이 나면 학습 배치를 줄입니다. 학습 탭의 예상값을 봅니다 |
| **로봇을 어느 쪽에 꽂나** | GPU 노트북에만 있으면 GPU 패널, 내 PC에도 있거나 양쪽 다 없으면 내 PC 패널이 열립니다 | 카메라 USB 위치는 PC마다 다르므로, 옮긴 뒤 **로봇 구성**을 다시 맞춥니다 |
| **USB 허브나 칸을 바꿈** | 카메라는 USB 위치로 구분합니다 | **로봇 구성**에서 카메라를 다시 고릅니다. 카메라 이름 `front`, `front_2`, `wrist_left`, `wrist_right`는 모델 입력과 같아야 합니다 |
| **여럿이 같은 노트북 사용** | 계정마다 패널과 포트가 따로입니다. 로봇 사용은 한 번에 한 사람, 학습은 팀 전체에서 한 번에 하나입니다 | 사용 중이면 화면에 누가 쓰는지 나옵니다 |
| **집이나 다른 Wi-Fi에서 접속** | Tailscale이 연결되면 어디서든 됩니다. GPU 노트북은 켜져 있어야 합니다 | 끊기면 `tailscale status`로 확인합니다. 로봇이 움직이는 작업은 현장에 사람이 있을 때만 합니다 |

## 7. 문제 해결

| 증상 | 확인할 것 | 해결할 사람 |
|---|---|---|
| `tailscale status`에 GPU 노트북이 없음 | 다른 계정으로 로그인했습니다. `sudo tailscale logout` 후 초대받은 계정으로 `sudo tailscale up` | 팀원 |
| GPU 노트북이 `offline` | 꺼짐, 절전, Wi-Fi 끊김, 키 만료 | GPU 또는 Tailscale 관리자 |
| `Unable to connect to coordination server` 경고 | 일부 Wi-Fi에서 나옵니다. 다른 기기로 접속되면 당장은 쓸 수 있습니다. 다른 Wi-Fi에서 다시 확인합니다 | 팀원 |
| `ssh omx-gpu`가 비밀번호를 물음 | `bash install_control.sh`를 다시 실행해 키를 등록합니다 | 팀원 |
| `Permission denied` (SSH) | 계정 이름과 비밀번호, GPU 노트북 SSH 서버 | GPU 관리자 |
| 점검에서 로봇 포트 권한 오류, `err_code_2` | `id`에 `dialout`이 있는지, 다시 로그인했는지. USB와 12V 전원 | GPU 관리자 또는 팀원 |
| 카메라가 안 열림, `err_code_7` | `id`에 `video`가 있는지. **로봇 구성**의 카메라 위치 | GPU 관리자 또는 팀원 |
| `err_code_1` | 모터 통신이 반복해 실패했습니다. 해당 팔 케이블, 전원, 허브 | 현장 담당 |
| 학습 데이터가 목록에 없음 | 학습 탭은 `/srv/il/lerobot_data/<계정>/<이름>/meta/info.json`만 봅니다. 데이터를 그 위치로 옮기고 패널 설정의 `data_home`을 확인합니다 | 팀원 |
| 패널 표시 버전이 예전 | 브라우저 새로고침. 그래도 같으면 패널을 돌리는 쪽에서 `bash ~/il_ws/omx_control_panel_v3/omx_tools/omx-panel-server.sh --restart` | 팀원 |
| 바탕화면 아이콘이 반응 없음 | 우클릭 → 실행 허용, 터미널에서 `omx-panel-v3` 실행 | 팀원 |

로그는 패널을 돌린 쪽의 `~/il_ws/omx_control_panel_v3/logs/`에 있습니다.

## 8. 업데이트

```bash
cd ~/omx-control-panel && git pull && bash install_control.sh
```
설정과 로봇 구성은 유지됩니다. GPU 노트북의 내 계정도 같은 버전으로 맞출지 묻습니다. **Y**를 고르세요.
녹화, 텔레옵, 평가 중이면 끝난 뒤 합니다. 학습은 재시작의 영향을 받지 않습니다.

## 9. 제거

내 PC와 GPU 노트북의 내 계정에서 각각 실행합니다.
```bash
pkill -u "$USER" -f omx_control_panel_v3/omx_tools/omx_web.py
rm -rf ~/il_ws/omx_control_panel_v3 ~/.config/omx/v3
rm -f ~/.local/bin/omx-panel-v3 ~/.local/share/omx/omx-control-v3.* ~/.local/share/applications/omx-control-v3.desktop "$(xdg-user-dir DESKTOP)/omx-control-v3.desktop"
```
녹화 데이터와 모델은 지워지지 않습니다. LeRobot 환경(`~/venv/il3`)은 필요 없을 때 따로 지웁니다.

## 부록: Windows, macOS에서 접속

설치 프로그램은 Ubuntu용입니다. 다른 OS는 브라우저로 접속만 합니다. 로봇은 GPU 노트북에 꽂습니다.
1. [Tailscale 앱](https://tailscale.com/download)을 설치하고 초대받은 계정으로 로그인합니다.
2. 터미널에서 터널을 엽니다. 포트는 GPU 노트북에서 설치가 끝날 때 알려 준 번호(8865부터)입니다.
   ```bash
   ssh -N -L 8865:127.0.0.1:8865 <내 GPU 계정>@<GPU 노트북 100.x 주소>
   ```
3. 브라우저에서 `http://127.0.0.1:8865`를 엽니다.

제어판은 GPU 노트북 안에서만(127.0.0.1) 열리므로, 같은 Wi-Fi의 다른 사람은 접속할 수 없습니다.
