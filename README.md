# OMX Control Panel

OMX-AI 로봇팔(리더·팔로워)로 모방학습을 하는 팀용 제어판입니다. 데이터 수집(텔레옵·녹화), 학습(ACT·Diffusion·SmolVLA), 평가(모델로 로봇 실행·분석)를 브라우저 한 화면에서 합니다.

- **제어 PC** (팀원 각자의 우분투 PC): 리더·팔로워·카메라를 연결하고 텔레옵·녹화·평가를 합니다. 단일 로봇과 협동 로봇(양팔)을 모두 지원합니다.
- **GPU 노트북**: 학습과 AI 추론 서버, 팀 공유 데이터를 맡습니다. 두 PC는 Tailscale + SSH로 연결합니다.

## 제어 PC 설치

```bash
git clone https://github.com/tjkoobro-netizen/omx-control-panel.git ~/omx-control-panel
cd ~/omx-control-panel && bash install_control.sh
```

단계별 안내(그림, 문제 해결 포함): [docs/install_control.html](docs/install_control.html) — 내려받은 뒤 브라우저로 여세요.

업데이트: `cd ~/omx-control-panel && git pull && bash install_control.sh` (설정과 로봇 구성은 유지됩니다)

v3.0.0부터는 v2 제어판 옆에 따로 설치됩니다: 바탕화면 아이콘 **OMX Control Panel v3**, 폴더 `~/il_ws/omx_control_panel_v3`, 명령 `omx-panel-v3`. 기존 **OMX Control Panel**(v2)은 그대로 남습니다. 처음 설치할 때 v2 설정·로봇 구성을 가져옵니다. 두 제어판을 동시에 켜고 로봇을 쓰지 마세요.

v2를 마지막 버전(v2.2.3)으로 맞추려면: `cd ~/omx-control-panel && git checkout v2.2.3 && bash install_control.sh && git checkout main`

## 그 밖의 문서

- GPU 노트북 설치(관리자용)와 설정 항목: [INSTALL.md](INSTALL.md)
- 버전별 변경 내용: [CHANGELOG.md](CHANGELOG.md)

요구 사항: Ubuntu 24.04(22.04는 Python 3.12 추가 설치), LeRobot 0.6.1·Python 3.12(설치 파일이 맞춰 줌), OMX-AI(OpenRB-150, 팔로워 모터 ID 11~16, 리더 1~6).
