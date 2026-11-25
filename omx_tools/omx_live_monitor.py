#!/usr/bin/env python3
"""관제 미션 중 과부하·데이터 손실 실시간 감시 (Guide, 11/24 TK 요청) — 제어 PC(tkoo-pc)에서 패널과 따로 돈다. 읽기만 한다.

1초마다 한 줄(JSONL)을 남기고, 문제가 보이면 WARN 줄을 함께 남긴다. 평가 프로그램(omx_run, warm 제외)이 돌 때는 매초,
쉬는 동안은 30초마다 기록한다.

보는 것
  카메라   : 패널이 평가 중 /dev/shm/omx_live_*stats.json 에 쓰는 카메라별 실제 fps (LeRobot 읽기 스레드 기준)
             커널 USB·UVC 오류 줄 (journalctl -k: uvcvideo, usb, xhci, "reset", "disconnect", "error")
  평가 로그: logs/*_eval.log 에서 원격 추론 왕복·GPU 추론 ms·보낸 영상 KB·'응답 없어 다시 보냄' 누적, 큐 비어 기다림,
             'Record loop is running slower', Corrupt JPEG, 카메라 읽기 실패
  제어 PC  : CPU %(전체·가장 바쁜 코어), load, 남은 메모리, omx_run CPU %
             네트워크 장치 RX/TX 오류·버림, UDP InErrors·RcvbufErrors·SndbufErrors, omx_run UDP 소켓 drops
  GPU 노트북(--gpu omx-gpu): ssh 한 개로 1초마다 UDP 카운터·5600 포트 소켓 drops·네트워크 장치 버림·load

사용
  python3 omx_live_monitor.py --gpu omx-gpu                # 계속 감시 (기록: live/YYYYMMDD.jsonl, 지금 상태: live/status.json)
  python3 omx_live_monitor.py --watch                      # 다른 창에서 지금 상태를 사람이 읽는 줄로 보기
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import threading
import time
from collections import deque
from pathlib import Path

# 기록 폴더는 프로그램 위치와 따로 둔다 (패널 저장소 omx_tools 로 옮겨도 배포 때 기록이 지워지지 않게)
OUT = Path(os.environ.get("OMX_LOSS_DIR", "~/il_ws/omx_loss_tracker/live")).expanduser()
PANEL_LOGS = Path(os.environ.get("OMX_PANEL_LOGS", "~/il_ws/omx_control_panel_v3/logs")).expanduser()
CAM_FPS_NOMINAL = 30.0
CAM_FPS_WARN = 24.0          # 이 밑이면 경고 (0 = 멈춤)
CPU_WARN = 90.0
GPU_PORT = 5600

# ---------------------------------------------------------------- /proc 읽기
def _cpu_times():
    out = {}
    for line in Path("/proc/stat").read_text().splitlines():
        if line.startswith("cpu"):
            p = line.split()
            v = list(map(int, p[1:]))
            idle = v[3] + v[4]
            out[p[0]] = (sum(v), idle)
    return out


def _cpu_pct(a, b):
    res = {}
    for k in b:
        if k in a:
            dt, di = b[k][0] - a[k][0], b[k][1] - a[k][1]
            res[k] = round(100.0 * (1 - di / dt), 1) if dt > 0 else 0.0
    return res


def _net_dev():
    out = {}
    for line in Path("/proc/net/dev").read_text().splitlines()[2:]:
        name, rest = line.split(":", 1)
        v = list(map(int, rest.split()))
        out[name.strip()] = {"rx_pkts": v[1], "rx_errs": v[2], "rx_drop": v[3], "tx_pkts": v[9], "tx_errs": v[10], "tx_drop": v[11]}
    return out


def _udp_snmp():
    lines = [l for l in Path("/proc/net/snmp").read_text().splitlines() if l.startswith("Udp:")]
    keys, vals = lines[0].split()[1:], list(map(int, lines[1].split()[1:]))
    return dict(zip(keys, vals))


def _udp_sockets():
    """inode -> (local port, drops)"""
    out = {}
    for f in ("/proc/net/udp", "/proc/net/udp6"):
        try:
            for line in Path(f).read_text().splitlines()[1:]:
                p = line.split()
                out[p[9]] = (int(p[1].split(":")[1], 16), int(p[-1]))
        except OSError:
            pass
    return out


def _pids(pattern):
    res = []
    for d in glob.glob("/proc/[0-9]*"):
        try:
            cmd = Path(d, "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            continue
        if pattern(cmd):
            res.append(int(d.rsplit("/", 1)[1]))
    return res


def _proc_ticks(pid):
    try:
        p = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return int(p[11]) + int(p[12])
    except (OSError, IndexError):
        return None


def _sock_inodes(pid):
    res = set()
    for fd in glob.glob(f"/proc/{pid}/fd/*"):
        try:
            t = os.readlink(fd)
        except OSError:
            continue
        if t.startswith("socket:["):
            res.add(t[8:-1])
    return res


def _mem_avail_mb():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) // 1024
    return None


def _is_eval(cmd):
    return "omx_run.py" in cmd and "--omx-warm" not in cmd and not cmd.startswith("/bin/bash")


# ---------------------------------------------------------------- 따라 읽기 (스레드)
class KernelUSB(threading.Thread):
    """journalctl -k -f 에서 USB·카메라 오류 줄만"""
    PAT = re.compile(r"uvcvideo|usb \d|xhci|USB disconnect|reset (high|full|super)-speed|cannot|error|Failed", re.I)
    SKIP = re.compile(r"apparmor|audit:", re.I)

    def __init__(self):
        super().__init__(daemon=True)
        self.count, self.recent = 0, deque(maxlen=20)

    def run(self):
        try:
            p = subprocess.Popen(["journalctl", "-k", "-f", "-n", "0", "-o", "short-iso", "--no-pager"],
                                 stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        except OSError:
            return
        for line in p.stdout:
            if self.PAT.search(line) and not self.SKIP.search(line):
                self.count += 1
                self.recent.append(line.strip()[:200])


class EvalLog(threading.Thread):
    """가장 최근 *_eval.log 를 따라 읽으며 통신·루프·카메라 줄을 센다"""
    ASYNC = re.compile(r"추론 (\d+)ms.*?큐 (\d+), 큐가 비어 기다린 횟수 (\d+) \(([\d.]+)s\).*?왕복 (\d+)ms \(GPU 추론 (\d+)ms, 보낸 영상 (\d+)KB\)(?:.*?다시 보냄 (\d+)회)?")
    SLOW = re.compile(r"running slower \(([\d.]+) Hz\)")
    # 패널 5초 감시 줄 (Admin 11/24 제안 형식; 'total' 이 붙은 줄은 에피소드 끝 요약)
    CAM = re.compile(r"([\w]+) ([\d.]+)fps age (\d+)/(\d+)ms dup (\d+) err (\d+)")
    UDP = re.compile(r"req (\d+) bytes ([\d.]+)MB frag (\d+) resp (\d+) timeout (\d+) stale (\d+) rtt (\d+)/(\d+)ms")
    LOOP = re.compile(r"p5 ([\d.]+)Hz min ([\d.]+)Hz over (\d+)")

    def __init__(self):
        super().__init__(daemon=True)
        self.file = None
        self.last_async = None
        self.slow_n, self.corrupt_n, self.camfail_n, self.err_n = 0, 0, 0, 0
        self.slow_min_hz = None
        self.recent = deque(maxlen=10)
        self.reset_5s()

    def reset_5s(self):
        self.cams5, self.udp5, self.loop5 = None, None, None
        self.sum5 = {"cam_dup": 0, "cam_err": 0, "udp_timeout": 0, "udp_stale": 0, "udp_req": 0, "udp_resp": 0, "loop_over": 0, "n5": 0}

    def run(self):
        f = None
        while True:
            logs = sorted(PANEL_LOGS.glob("*_eval.log"), key=lambda p: p.stat().st_mtime)
            newest = logs[-1] if logs else None
            if newest is not None and newest != self.file:
                f = open(newest, errors="replace")
                f.seek(0, 2) if self.file is None else None   # 처음 켤 때는 끝부터, 새 평가면 처음부터
                self.file = newest
                self.last_async, self.slow_n, self.corrupt_n, self.camfail_n, self.err_n, self.slow_min_hz = None, 0, 0, 0, 0, None
                self.reset_5s()
            line = f.readline() if f else ""
            if not line:
                time.sleep(0.2)
                continue
            total = " total" in line.split(":", 1)[0]
            if "[omx] cams" in line and not total:
                c = {g[0]: {"fps": float(g[1]), "age_p50": int(g[2]), "age_p95": int(g[3]), "dup": int(g[4]), "err": int(g[5])} for g in self.CAM.findall(line.split(":", 1)[1])}
                self.cams5 = c
                self.sum5["cam_dup"] += sum(v["dup"] for v in c.values())
                self.sum5["cam_err"] += sum(v["err"] for v in c.values())
                self.sum5["n5"] += 1
            u = self.UDP.search(line) if "[omx] udp" in line and not total else None
            if u:
                self.udp5 = {"req": int(u[1]), "mb": float(u[2]), "frag": int(u[3]), "resp": int(u[4]), "timeout": int(u[5]), "stale": int(u[6]), "rtt_p50": int(u[7]), "rtt_p95": int(u[8])}
                for k in ("timeout", "stale", "req", "resp"):
                    self.sum5[f"udp_{k}"] += self.udp5[k]
            lp = self.LOOP.search(line) if "[omx] loop5s" in line and not total else None
            if lp:
                self.loop5 = {"p5_hz": float(lp[1]), "min_hz": float(lp[2]), "over": int(lp[3])}
                self.sum5["loop_over"] += self.loop5["over"]
            m = self.ASYNC.search(line)
            if m:
                self.last_async = {"infer_ms": int(m[1]), "queue": int(m[2]), "queue_empty_n": int(m[3]), "queue_empty_s": float(m[4]),
                                   "rtt_ms": int(m[5]), "gpu_ms": int(m[6]), "sent_kb": int(m[7]), "resent_total": int(m[8] or 0)}
            s = self.SLOW.search(line)
            if s:
                self.slow_n += 1
                hz = float(s[1])
                self.slow_min_hz = hz if self.slow_min_hz is None else min(self.slow_min_hz, hz)
            if "Corrupt JPEG" in line or "Premature end of JPEG" in line:
                self.corrupt_n += 1
            if ("카메라" in line and ("못 읽" in line or "사진 없음" in line)) or "TimeoutError" in line:
                self.camfail_n += 1
                self.recent.append(line.strip()[:200])
            if "Traceback" in line or "Error" in line and "[omx]" not in line:
                self.err_n += 1


GPU_DIR = "~/omx_live_monitor"
UDP_SERVER_LOG = "~/il_ws/omx_control_panel_v3/logs/udp_server.log"
UDP_KEYS = ("completed_infer", "fragments_expected", "fragments_received", "fragments_missing", "timeout_infer", "superseded_infer",
            "late", "checksum", "bad_mac", "conflict", "duplicate", "malformed", "untrusted_peer", "capacity", "skipped_backlog",
            "refused", "failed", "reply_failed", "send_failed", "replied_actions")   # GPU 노트북 쪽 기록 폴더 (--gpu-local)


def gpu_local_start(host):
    """GPU 노트북에 감시 스크립트를 두고 그쪽 파일에만 1초마다 쓰게 띄운다 (미션 중 통신 없음). 이미 돌고 있으면 그대로"""
    code = GpuSide.REMOTE.replace("PORT", str(GPU_PORT)).replace("print(json.dumps(", "out.write(json.dumps(").replace("),flush=True)", ")+chr(10));out.flush()")
    code = "import os,time\nout=open(os.path.expanduser('%s/gpu_'+time.strftime('%%Y%%m%%d')+'.jsonl'),'a')\n" % GPU_DIR + code
    d = "$HOME/omx_live_monitor"   # pid 파일로 확인 (pgrep -f 는 이 명령줄 자신을 찾는다)
    sh = (f"mkdir -p {d} && cat > {d}/gpu_side.py && "
          f"if [ -f {d}/pid ] && kill -0 $(cat {d}/pid) 2>/dev/null; then echo 이미 실행 중 $(cat {d}/pid); "
          f"else setsid nohup nice -n 19 python3 -u {d}/gpu_side.py >/dev/null 2>&1 < /dev/null & echo $! > {d}/pid; sleep 1; "
          f"kill -0 $(cat {d}/pid) && echo 시작 $(cat {d}/pid); fi")
    r = subprocess.run(["ssh", "-o", "ConnectTimeout=8", host, sh], input=code, text=True, capture_output=True, timeout=30)
    return (r.stdout.strip() or r.stderr.strip())[:200]


def gpu_local_collect(host):
    """미션 뒤: GPU 쪽 기록을 live/gpu/ 로 가져온다 (GPU 쪽 기록은 계속 돈다; 멈추려면 --gpu-stop)"""
    dst = OUT / "gpu"
    dst.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["rsync", "-a", f"{host}:{GPU_DIR}/", str(dst) + "/"], capture_output=True, text=True, timeout=120)
    # AI 서버 손실 카운터 (Guide_3 11/24: udp_server.log 의 "[omx-udp] stats {json}" 10초마다, 서버 시작부터 누적)
    r2 = subprocess.run(["rsync", "-a", f"{host}:{UDP_SERVER_LOG}", str(dst / "udp_server.log")], capture_output=True, text=True, timeout=120)
    return r.returncode, (r.stderr or "").strip()[:200], "udp_server.log", r2.returncode, (r2.stderr or "").strip()[:160]


class GpuSide(threading.Thread):
    """ssh 한 개로 GPU 노트북의 UDP·네트워크 카운터를 1초마다 받는다"""
    REMOTE = r'''
import time,json
def snmp():
    l=[x for x in open("/proc/net/snmp").read().splitlines() if x.startswith("Udp:")]
    return dict(zip(l[0].split()[1:],map(int,l[1].split()[1:])))
def sock():
    d=0;q=0
    for f in ("/proc/net/udp","/proc/net/udp6"):
        try:
            for x in open(f).read().splitlines()[1:]:
                p=x.split()
                if int(p[1].split(":")[1],16)==PORT: d+=int(p[-1]); q+=int(p[4].split(":")[1],16)
        except OSError: pass
    return d,q
def dev():
    o={}
    for x in open("/proc/net/dev").read().splitlines()[2:]:
        n,r=x.split(":",1); v=list(map(int,r.split())); o[n.strip()]=[v[2],v[3],v[10],v[11]]
    return o
while True:
    d,q=sock()
    print(json.dumps({"t":time.time(),"udp":snmp(),"sock_drops":d,"sock_rxq":q,"dev":dev(),"load":float(open("/proc/loadavg").read().split()[0])}),flush=True)
    time.sleep(1)
'''

    def __init__(self, host):
        super().__init__(daemon=True)
        self.host, self.last, self.first, self.err = host, None, None, None

    def run(self):
        code = self.REMOTE.replace("PORT", str(GPU_PORT))
        while True:
            try:
                p = subprocess.Popen(["ssh", "-o", "ConnectTimeout=8", "-o", "ServerAliveInterval=5", self.host, "nice -n 19 python3 -u -"],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                p.stdin.write(code)
                p.stdin.close()
                for line in p.stdout:
                    try:
                        v = json.loads(line)
                    except ValueError:
                        continue
                    if self.first is None:
                        self.first = v
                    self.last, self.err = v, None
                self.err = (p.stderr.read() or "ssh 끊김").strip()[:160]
            except OSError as exc:
                self.err = str(exc)[:160]
            self.last = None
            time.sleep(5)


# ---------------------------------------------------------------- 본체
def camera_stats():
    files = sorted(glob.glob("/dev/shm/omx_live_*stats.json"), key=os.path.getmtime)
    if not files:
        return None, None
    f = files[-1]
    age = time.time() - os.path.getmtime(f)
    try:
        return json.loads(Path(f).read_text()), round(age, 1)
    except (OSError, ValueError):
        return None, round(age, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default=None, help="GPU 노트북 ssh 이름 (예: omx-gpu). 없으면 제어 PC만")
    ap.add_argument("--gpu-mode", choices=("local", "stream"), default="local",
                    help="local(기본): GPU 쪽은 그 노트북 파일에만 기록, 미션 중 통신 없음 → --collect 로 나중에 가져옴. stream: 1초마다 ssh 로 받아 실시간 경고(약 0.4 KB/s)")
    ap.add_argument("--collect", action="store_true", help="GPU 쪽 기록을 live/gpu/ 로 가져오고 끝")
    ap.add_argument("--gpu-stop", action="store_true", help="GPU 쪽 기록 프로세스를 멈추고 끝")
    ap.add_argument("--report", default=None, metavar="YYYYMMDD", help="그날 기록을 평가 회차별로 요약해 live/report_YYYYMMDD.md 에 쓰고 끝")
    ap.add_argument("--watch", action="store_true", help="live/status.json 을 사람이 읽는 줄로 1초마다 보여 준다")
    ap.add_argument("--idle-every", type=float, default=30.0)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.watch:
        return watch()
    if a.gpu_stop:
        r = subprocess.run(["ssh", a.gpu or "omx-gpu", "kill $(cat $HOME/omx_live_monitor/pid) && rm -f $HOME/omx_live_monitor/pid && echo 멈춤"], capture_output=True, text=True, timeout=20)
        print((r.stdout + r.stderr).strip())
        return
    if a.collect:
        print(gpu_local_collect(a.gpu or "omx-gpu"))
        return
    if a.report:
        return report(a.report)
    kern, elog = KernelUSB(), EvalLog()
    kern.start()
    elog.start()
    gpu = None
    if a.gpu and a.gpu_mode == "stream":
        gpu = GpuSide(a.gpu)
        gpu.start()
    elif a.gpu:
        print("GPU 쪽 로컬 기록:", gpu_local_start(a.gpu), flush=True)
    prev = {"cpu": _cpu_times(), "dev": _net_dev(), "udp": _udp_snmp(), "t": time.time(), "ticks": {}, "kern": 0, "resent": None,
            "gpu_udp": None, "gpu_drops": None, "gpu_dev": None, "sock_drops": {}}
    last_write = 0.0
    run_start = None
    while True:
        time.sleep(1.0)
        now = time.time()
        cpu, dev, udp = _cpu_times(), _net_dev(), _udp_snmp()
        pct = _cpu_pct(prev["cpu"], cpu)
        if now - prev.get("pid_t", 0) >= 5 or any(not os.path.exists(f"/proc/{q}") for q in prev.get("evals", [])):   # 프로세스 훑기는 5초마다
            prev["evals"], prev["pid_t"] = _pids(_is_eval), now
        evals = prev["evals"]
        active = bool(evals)
        if run_start is not None and not active:   # 평가 하나가 끝남 → 그 회차 요약
            try:
                summarize_run(run_start, now, elog.file.name if elog.file else None)
            except Exception as exc:  # noqa: BLE001
                print("요약 실패", exc, flush=True)
        run_start = (run_start or now) if active else None
        row = {"t": round(now, 2), "at": time.strftime("%Y-%m-%d %H:%M:%S"), "eval_active": active,
               "eval_s": None if run_start is None else round(now - run_start, 1),
               "cpu": pct.get("cpu"), "cpu_core_max": max((v for k, v in pct.items() if k != "cpu"), default=None),
               "load1": float(Path("/proc/loadavg").read_text().split()[0]), "mem_avail_mb": _mem_avail_mb()}
        warns = []
        # omx_run CPU %, UDP 소켓 drops
        hz = os.sysconf("SC_CLK_TCK")
        socks = _udp_sockets()
        run_cpu, sdrops = {}, 0
        for pid in evals:
            tk = _proc_ticks(pid)
            if tk is not None and pid in prev["ticks"]:
                run_cpu[pid] = round(100.0 * (tk - prev["ticks"][pid]) / hz / (now - prev["t"]), 1)
            prev["ticks"][pid] = tk
            for ino in _sock_inodes(pid):
                if ino in socks:
                    sdrops += socks[ino][1]
        row["omx_run_cpu"] = sum(run_cpu.values()) if run_cpu else None
        row["omx_run_udp_sock_drops"] = sdrops if evals else None
        # 네트워크 장치
        nd = {}
        for k, v in dev.items():
            if k == "lo" or k not in prev["dev"]:
                continue
            d = {x: v[x] - prev["dev"][k][x] for x in ("rx_errs", "rx_drop", "tx_errs", "tx_drop", "rx_pkts", "tx_pkts")}
            if d["rx_pkts"] or d["tx_pkts"] or any(d[x] for x in ("rx_errs", "rx_drop", "tx_errs", "tx_drop")):
                nd[k] = d
            if any(d[x] for x in ("rx_errs", "rx_drop", "tx_errs", "tx_drop")):
                warns.append(f"네트워크 {k} 오류·버림 {d}")
        row["net"] = nd
        du = {k: udp[k] - prev["udp"].get(k, udp[k]) for k in ("InDatagrams", "OutDatagrams", "InErrors", "RcvbufErrors", "SndbufErrors", "NoPorts")}
        row["udp"] = du
        if du["InErrors"] or du["RcvbufErrors"] or du["SndbufErrors"]:
            warns.append(f"제어 PC UDP 버림 InErrors {du['InErrors']} RcvbufErrors {du['RcvbufErrors']} SndbufErrors {du['SndbufErrors']}")
        if row["cpu_core_max"] is not None and row["cpu_core_max"] >= CPU_WARN and active:
            warns.append(f"CPU 코어 하나가 {row['cpu_core_max']}% (전체 {row['cpu']}%)")
        # 카메라
        cams, age = camera_stats()
        if active and cams:
            row["cam_stats_age_s"] = age
            row["cam_fps"] = {k: (v or {}).get("fps") for k, v in cams.items()}
            if age is not None and age < 3:
                for k, f_ in row["cam_fps"].items():
                    if f_ is not None and f_ < CAM_FPS_WARN:
                        warns.append(f"카메라 {k} {f_} fps" + (" (멈춤)" if f_ == 0 else f" (< {CAM_FPS_WARN:g})"))
        # 커널 USB
        kn = kern.count - prev["kern"]
        prev["kern"] = kern.count
        row["kernel_usb_new"] = kn
        if kn:
            warns.append(f"커널 USB·카메라 메시지 {kn}줄: {list(kern.recent)[-1]}")
        # 평가 로그
        if active and elog.file is not None:
            la = elog.last_async
            row["eval_log"] = elog.file.name
            row["infer"] = la
            row["loop_slow_lines"], row["loop_slow_min_hz"] = elog.slow_n, elog.slow_min_hz
            row["corrupt_jpeg"], row["cam_read_fail"] = elog.corrupt_n, elog.camfail_n
            row["cams5"], row["udp5"], row["loop5"], row["sum5"] = elog.cams5, elog.udp5, elog.loop5, dict(elog.sum5)
            key5 = (elog.sum5["n5"], elog.sum5["udp_req"])
            if key5 != prev.get("key5"):   # 새 5초 줄이 들어왔을 때만 경고
                prev["key5"] = key5
                for k, v in (elog.cams5 or {}).items():
                    if v["err"] or v["age_p95"] > 150 or v["fps"] < CAM_FPS_WARN:
                        warns.append(f"카메라 {k} 5초: {v['fps']}fps 나이 p95 {v['age_p95']}ms 같은 프레임 {v['dup']} 오류 {v['err']}")
                u5 = elog.udp5
                if u5 and (u5["timeout"] or u5["stale"] or u5["resp"] < u5["req"]):
                    warns.append(f"추론 통신 5초: 요청 {u5['req']} 응답 {u5['resp']} 타임아웃 {u5['timeout']} 늦은 응답 버림 {u5['stale']} 왕복 p95 {u5['rtt_p95']}ms")
                l5 = elog.loop5
                if l5 and l5["min_hz"] < 20:
                    warns.append(f"제어 루프 5초: 최저 {l5['min_hz']}Hz, 33ms 넘은 단계 {l5['over']}")
            if la is not None:
                if prev["resent"] is not None and la["resent_total"] > prev["resent"]:
                    warns.append(f"원격 추론 응답 없어 다시 보냄 +{la['resent_total'] - prev['resent']} (누적 {la['resent_total']}, 왕복 {la['rtt_ms']}ms)")
                prev["resent"] = la["resent_total"]
        else:
            prev["resent"] = None
        # GPU
        if gpu is not None:
            g = gpu.last
            if g is None:
                row["gpu"] = {"error": gpu.err}
                if active:
                    warns.append(f"GPU 노트북 감시 끊김 ({gpu.err})")
            else:
                gu = {k: g["udp"][k] - (prev["gpu_udp"] or g["udp"])[k] for k in ("InDatagrams", "OutDatagrams", "InErrors", "RcvbufErrors", "SndbufErrors")}
                gd = g["sock_drops"] - (prev["gpu_drops"] if prev["gpu_drops"] is not None else g["sock_drops"])
                gdev = {}
                for k, v in g["dev"].items():
                    pv = (prev["gpu_dev"] or g["dev"]).get(k, v)
                    dd = [v[i] - pv[i] for i in range(4)]
                    if any(dd):
                        gdev[k] = {"rx_errs": dd[0], "rx_drop": dd[1], "tx_errs": dd[2], "tx_drop": dd[3]}
                row["gpu"] = {"udp": gu, "port_sock_drops": gd, "port_rxq": g["sock_rxq"], "dev_err": gdev, "load1": g["load"], "age_s": round(now - g["t"], 1)}
                if gu["InErrors"] or gu["RcvbufErrors"] or gd:
                    warns.append(f"GPU UDP 버림 InErrors {gu['InErrors']} RcvbufErrors {gu['RcvbufErrors']} {GPU_PORT}포트 소켓 {gd}")
                if gdev:
                    warns.append(f"GPU 네트워크 장치 오류·버림 {gdev}")
                prev["gpu_udp"], prev["gpu_drops"], prev["gpu_dev"] = g["udp"], g["sock_drops"], g["dev"]
        row["warn"] = warns
        prev.update(cpu=cpu, dev=dev, udp=udp, t=now)
        # 기록
        (OUT / "status.tmp").write_text(json.dumps(row, ensure_ascii=False))
        os.replace(OUT / "status.tmp", OUT / "status.json")
        if active or warns or now - last_write >= a.idle_every:
            with open(OUT / f"{time.strftime('%Y%m%d')}.jsonl", "a") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if warns:
                with open(OUT / f"{time.strftime('%Y%m%d')}_warn.log", "a") as f:
                    for w in warns:
                        f.write(f"{row['at']} {'[평가 중] ' if active else ''}{w}\n")
            last_write = now


def _rows(day):
    f = OUT / f"{day}.jsonl"
    if not f.exists():
        return []
    out = []
    for line in f.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            pass
    return out


def _gpu_rows(day):
    f = OUT / "gpu" / f"gpu_{day}.jsonl"
    if not f.exists():
        return []
    out = []
    for line in f.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            pass
    return out


def _udp_stats():
    f = OUT / "gpu" / "udp_server.log"
    out = []
    if f.exists():
        for line in f.read_text(errors="replace").splitlines():
            i = line.find("[omx-udp] stats ")
            if i >= 0:
                try:
                    out.append(json.loads(line[i + 16:]))
                except ValueError:
                    pass
    return out


def _udp_window(stats, t0, t1):
    """서버 누적 카운터의 [t0, t1] 구간 증가분 (재시작 = started 바뀜이면 그 구간은 0부터). infer_ms 는 구간 값 중 최대 p95"""
    if not stats:
        return None
    tot, p95, n = {}, [], 0
    for started in {s_["started"] for s_ in stats}:
        seg = sorted((s_ for s_ in stats if s_["started"] == started), key=lambda s_: s_["ts"])
        inw = [s_ for s_ in seg if t0 - 2 <= s_["ts"] <= t1 + 12]     # 10초마다 남기므로 끝을 넉넉히
        if not inw:
            continue
        before = [s_ for s_ in seg if s_["ts"] < t0 - 2]
        base = before[-1] if before else {}
        for k in UDP_KEYS:
            tot[k] = tot.get(k, 0) + inw[-1].get(k, 0) - base.get(k, 0)
        p95 += [s_["infer_ms_p95"] for s_ in inw if s_.get("infer_ms_p95") is not None]
        n += 1
    if not n:
        return None
    tot = {k: v for k, v in tot.items() if v}
    tot["infer_ms_p95_max"] = max(p95) if p95 else None
    return tot


def _run_summary(rows, grows, t0, t1):
    R = [r for r in rows if t0 - 1 <= r["t"] <= t1 + 1 and r.get("eval_active")]
    if not R:
        return None
    s = {"start": time.strftime("%H:%M:%S", time.localtime(t0)), "dur_s": round(t1 - t0, 1), "eval_log": next((r.get("eval_log") for r in R if r.get("eval_log")), None)}
    fps = {}
    for r in R:
        for k, v in (r.get("cam_fps") or {}).items():
            if v is not None:
                fps.setdefault(k, []).append(v)
    s["cam_fps_min_med"] = {k: (min(v), sorted(v)[len(v) // 2]) for k, v in fps.items()}
    s["cam_stall_s"] = {k: sum(1 for x in v if x == 0) for k, v in fps.items() if any(x == 0 for x in v)}
    s["cpu_max"] = max((r.get("cpu") or 0) for r in R)
    s["cpu_core_max"] = max((r.get("cpu_core_max") or 0) for r in R)
    last_inf = [r["infer"] for r in R if r.get("infer")]
    s["infer_rtt_ms_last"] = last_inf[-1]["rtt_ms"] if last_inf else None
    s["resent_total"] = last_inf[-1]["resent_total"] if last_inf else None
    s["loop_slow_lines"] = max((r.get("loop_slow_lines") or 0) for r in R)
    s["corrupt_jpeg"] = max((r.get("corrupt_jpeg") or 0) for r in R)
    s["cam_read_fail"] = max((r.get("cam_read_fail") or 0) for r in R)
    s["pc_udp_drops"] = sum((r["udp"].get("InErrors", 0) + r["udp"].get("RcvbufErrors", 0) + r["udp"].get("SndbufErrors", 0)) for r in R)
    s["pc_net_drops"] = sum(sum(v[x] for x in ("rx_errs", "rx_drop", "tx_errs", "tx_drop")) for r in R for v in (r.get("net") or {}).values())
    s["kernel_usb_lines"] = sum(r.get("kernel_usb_new") or 0 for r in R)
    last5 = [r["sum5"] for r in R if r.get("sum5")]
    if last5 and last5[-1].get("n5"):
        s["panel_5s"] = last5[-1]   # 패널 5초 줄 합 (같은 프레임·카메라 오류·UDP 타임아웃·늦은 응답·루프 초과)
        ages = [v["age_p95"] for r in R for v in (r.get("cams5") or {}).values()]
        s["cam_age_p95_max_ms"] = max(ages) if ages else None
    s["warn_n"] = sum(len(r.get("warn") or []) for r in R)
    G = [g for g in grows if t0 - 1 <= g["t"] <= t1 + 1]
    if len(G) >= 2:
        a, b = G[0], G[-1]
        s["gpu_udp_drops"] = sum(b["udp"][k] - a["udp"][k] for k in ("InErrors", "RcvbufErrors"))
        s["gpu_port_sock_drops"] = b["sock_drops"] - a["sock_drops"]
        s["gpu_net_drops"] = sum(sum(b["dev"][k][i] - a["dev"].get(k, b["dev"][k])[i] for i in range(4)) for k in b["dev"])
    else:
        s["gpu"] = "기록 없음 (--collect 전이거나 GPU 쪽이 안 돌았음)"
    s["udp_server"] = _udp_window(_udp_stats(), t0, t1)
    return s


def summarize_run(t0, t1, eval_log):
    day = time.strftime("%Y%m%d", time.localtime(t0))
    s = _run_summary(_rows(day), _gpu_rows(day), t0, t1)
    if s:
        with open(OUT / f"{day}_runs.jsonl", "a") as f:
            f.write(json.dumps(dict(s, t0=t0, t1=t1), ensure_ascii=False) + "\n")


def report(day):
    """그날 평가 회차별 요약 (GPU 쪽은 --collect 뒤에 채워진다)"""
    rows, grows = _rows(day), _gpu_rows(day)
    runs = []
    rf = OUT / f"{day}_runs.jsonl"
    for line in (rf.read_text().splitlines() if rf.exists() else []):
        r = json.loads(line)
        runs.append(_run_summary(rows, grows, r["t0"], r["t1"]) or r)
    L = [f"# 관제 미션 손실·과부하 요약 {day}", "", f"평가 {len(runs)}회. 기준: 카메라 {CAM_FPS_WARN:g} fps 미만 경고, CPU 코어 {CPU_WARN:g}% 이상 경고.", ""]
    for r in runs:
        L.append(f"## {r['start']} ({r['dur_s']}초) {r.get('eval_log') or ''}")
        L.append(f"- 카메라 fps (최소, 중앙): " + ", ".join(f"{k} {v[0]}/{v[1]}" for k, v in (r.get('cam_fps_min_med') or {}).items()) + (f" · 멈춤(초) {r['cam_stall_s']}" if r.get('cam_stall_s') else ""))
        L.append(f"- CPU 최대 {r['cpu_max']}% · 코어 최대 {r['cpu_core_max']}% · 루프 느림 줄 {r['loop_slow_lines']} · Corrupt JPEG {r['corrupt_jpeg']} · 카메라 읽기 실패 {r['cam_read_fail']} · 커널 USB 줄 {r['kernel_usb_lines']}")
        L.append(f"- 원격 추론: 마지막 왕복 {r['infer_rtt_ms_last']}ms · 응답 없어 다시 보냄 {r['resent_total']}회")
        L.append(f"- 버림: 제어 PC UDP {r['pc_udp_drops']} · 네트워크 장치 {r['pc_net_drops']} · GPU UDP {r.get('gpu_udp_drops', r.get('gpu'))} · GPU {GPU_PORT}포트 소켓 {r.get('gpu_port_sock_drops', '-')} · GPU 네트워크 장치 {r.get('gpu_net_drops', '-')}")
        us = r.get("udp_server")
        if us:
            L.append(f"- AI 서버(재조립): 완성 추론 {us.get('completed_infer', 0)} · 조각 {us.get('fragments_received', 0)}/{us.get('fragments_expected', 0)} 받음, 끝내 안 옴 {us.get('fragments_missing', 0)}"
                     f" · 조각 모자라 시간 초과 {us.get('timeout_infer', 0)} · 새 요청에 밀림 {us.get('superseded_infer', 0)} · 늦은 조각 {us.get('late', 0)}"
                     f" · CRC/MAC 실패 {us.get('checksum', 0)}/{us.get('bad_mac', 0)} · 계산 건너뜀 {us.get('skipped_backlog', 0)} · 추론 실패 {us.get('failed', 0)}"
                     f" · 응답 보냄 {us.get('replied_actions', 0)} · 추론 p95 최대 {us.get('infer_ms_p95_max')}ms")
        elif r.get("udp_server") is None:
            L.append("- AI 서버(재조립): 기록 없음 (서버 카운터 배포 전이거나 --collect 전)")
        if r.get("panel_5s"):
            p5 = r["panel_5s"]
            L.append(f"- 패널 5초 줄 합: UDP 요청 {p5['udp_req']} 응답 {p5['udp_resp']} 타임아웃 {p5['udp_timeout']} 늦은 응답 {p5['udp_stale']} · 카메라 같은 프레임 {p5['cam_dup']} 오류 {p5['cam_err']} (프레임 나이 p95 최대 {r.get('cam_age_p95_max_ms')}ms) · 루프 33ms 초과 {p5['loop_over']}")
        L.append(f"- 경고 {r['warn_n']}줄 (live/{day}_warn.log)")
        L.append("")
    (OUT / f"report_{day}.md").write_text("\n".join(L))
    print("\n".join(L))


def watch():
    while True:
        try:
            r = json.loads((OUT / "status.json").read_text())
        except (OSError, ValueError):
            time.sleep(1)
            continue
        cams = " ".join(f"{k}:{v}" for k, v in (r.get("cam_fps") or {}).items())
        inf = r.get("infer") or {}
        g = r.get("gpu") or {}
        print(f"{r['at']} {'평가중 ' + str(r.get('eval_s')) + 's' if r['eval_active'] else '대기'} | CPU {r['cpu']}% (코어 최대 {r['cpu_core_max']}) "
              f"| 카메라 {cams or '-'} | 왕복 {inf.get('rtt_ms', '-')}ms 재전송 {inf.get('resent_total', '-')} "
              f"| UDP버림 PC {r['udp'].get('InErrors', 0) + r['udp'].get('RcvbufErrors', 0)} GPU {(g.get('udp') or {}).get('RcvbufErrors', '-')}/{g.get('port_sock_drops', '-')}"
              + ("".join(f"\n   ⚠ {w}" for w in r.get("warn") or [])), flush=True)
        time.sleep(1)


if __name__ == "__main__":
    main()
