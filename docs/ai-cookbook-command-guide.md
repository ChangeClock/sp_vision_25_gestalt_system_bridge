# Gestalt 外部视觉控制 AI-Cookbook

本文是 `sp_vision` 接管 Gestalt 哨兵的发布包调用契约。它覆盖
Development、Preview 和最终 Shipping；算法配置与三靶基线仍以仓库根目录
`readme.md` 为准。

## 1. Preview 与 Shipping 的关系

Gestalt 的 `-preview` 发布车道打出的仍是 **Shipping 二进制**：
`UE_BUILD_SHIPPING=1`，只是额外保留 Preview 诊断日志。因此 Preview 通过能证明
Shipping 编译分支可用，但最终发布包还要做一次“不依赖游戏日志”的 smoke。

外部视觉控制在 Shipping 中采用显式 opt-in，而不是无条件开放。完整视觉流程的
推荐命令行是：

```text
-externalvisioncontrol -visionbridge -wsbind=127.0.0.1
```

- `-externalvisioncontrol`：与 loopback WS 共同组成授权条件，只打开外部视觉所需的最小控制白名单；必须是裸参数，`-externalvisioncontrol=1` 不生效。
- `-visionbridge`：启动即创建同机共享内存帧流；它是数据面预开关，不是授权条件。虽然授权后可用白名单 `UEExec r.VisionBridge.Enable 1` 动态开启，发布包流程仍推荐启动时显式传入。
- `-wsbind=127.0.0.1`：把 AttributeMap/控制 WS 固定在回环地址。即使 Shipping
  当前默认也是回环，验收命令仍显式写出，避免配置漂移或误暴露到局域网。

`sp_vision` 和游戏必须运行在同一台 Windows 主机：帧像素与同帧相机位姿走按
游戏 PID 命名的本地共享内存，控制与遥测走回环 WebSocket。

不带 `-externalvisioncontrol` 时，Shipping 必须 fail closed：`ExtAimClaim` 和外部
视觉 UE 命令不生效；不带 `-visionbridge` 时启动阶段不会预建共享帧流，但若在授权后
明确发送白名单 `UEExec r.VisionBridge.Enable 1`，仍会按设计动态创建。这是安全预期，
不是回归。

权限判断不依赖日志或 Kismet fallback。UEcho 创建 V8 时会把真实
`FCommandLine::Get()` 与 `FApp::GetBuildConfiguration()` 装入受 GC 保护的
`StartupParams`，经 `puerts.argv` 传给 `StartGame.ts` 再注入命令管理器。缺失或
注入失败时 build 视为 `unknown` 并 fail closed；Preview 日志中的
`Failed to get StartupParams` / `Failed to inject startup params into echo` 是硬错误。

## 2. CI 发布包与阻塞终端 A

本回归不以本地 UE 编译产物作为发布结论。先让 Gestalt 主仓的 Preview/Shipping tag
CI 完成 `package-ue` 与 `publish-package-nas`，再使用该 tag 对应的不可变 NAS drop：

```text
\\nas\ue5\GamePackages\gestalt_system\ci\<完整-tag>\
```

包体较大，不作为 GitLab artifact 上传。目录必须同时存在 `PUBLISHED.ok` 与
`manifest.json`；核对 manifest 的 tag/commit/buildConfig 和 launcher SHA256 后，使用
其中的 `Build/Win64` 内容（或在 Gestalt 主仓调用
`scripts/fetch-package-drop.ps1 -DropId <完整-tag>` 做同一套完整性校验）。不要用可变的
`current.txt` 猜版本，也不要把本机偶然残留的 `Build/Win64` 当成该 tag 的产物。

**终端 A（保持阻塞，承载游戏进程）**：推荐通过
`gestalt_system/scripts/ai-match-selftest.ps1` 启动 CI 包。这个终端必须保持运行，直到
终端 B 的完整对局退出；不要在 `.wsport` 刚出现后关闭它：

```powershell
$gameRepo = 'C:\path\to\gestalt_system'
$gameExe  = "$gameRepo\Build\Win64\Gestalt_System.exe"

powershell -File "$gameRepo\scripts\ai-match-selftest.ps1" `
  -PackagedExe $gameExe `
  -SkipMatchStart 1 -MatchObserveSeconds 900 -MapId 4 `
  -RenderMode windowed -ResX 1280 -ResY 720 `
  -WsBind 127.0.0.1 `
  -ExtraArgs '-externalvisioncontrol -visionbridge'
```

等待 `Saved\ai-selftest\<run>.log.wsport` 出现，从文件读取端口；不要在 Shipping
里 grep `WebSocket server started`。同一时间只保留一个带 VisionBridge 的目标游戏
窗口。视觉进程还会把该 WS port 反查到监听 PID，并只接受同 PID 的窗口、共享内存
mapping 与 frame writer；即使意外并存多个 publisher，也不会把 A 的控制流配上 B 的
像素流。

## 3. WS 与命令矩阵

所有 TS 命令都通过 WS `console.exec` 发送；UE 原生命令必须以 `UEExec ` 开头。
`rgbCamera.applySettings` 是独立 JSON-RPC，不是控制台命令。

| 接口/命令 | Development | Preview / Shipping + 对应 opt-in | Shipping 默认 | 说明 |
|---|---:|---:|---:|---|
| `attribute.watchAttributeMaps` | 是 | 是 | 是 | 属性遥测；效果判定主通道 |
| `console.exec` 传输 | 是 | 是 | 是 | 是否执行仍取决于具体命令门控 |
| `rgbCamera.applySettings` | 是 | 是 | 是 | 设置 FOV/曝光/ISO/`armLength:0` |
| `Respawn <pid> <entity> <team>` | 是 | 是 | 是 | 核心编排命令；异步等待新 map 且 HP>0 |
| `SetAttribute <pid> <attr> <value>` | 是 | 是 | 是 | 精确写单车属性 |
| `SetMatchStatus 1` | 是 | 是 | 是 | 接管完成后再开赛 |
| `ExtAimClaim <pid> <0\|1>` | 是 | 是 | 否 | Shipping 需 `-externalvisioncontrol`；claim 是 5s 可续租 lease |
| `UEExec RBTakeOver ...` | 是 | 是 | 否 | 外控白名单；切视角/释放 |
| `UEExec RBExtAim ...` | 是 | 是 | 否 | 外控白名单；绝对角、速度前馈和开火 |
| `UEExec r.VisionBridge.Enable 1` | 是 | 是 | 否 | 白名单内，但发布包首选 `-visionbridge` |
| `UEExec r.MotionBlurQuality ...` | 是 | 是 | 否 | 外控渲染白名单 |
| `UEExec r.AntiAliasingMethod ...` | 是 | 是 | 否 | 外控渲染白名单 |
| `UEExec r.RobotNav.DebugDraw ...` | 是 | 是 | 否 | 外控渲染白名单 |
| `UEExec t.MaxFPS ...` | 是 | 是 | 否 | 外控性能白名单 |
| 其他任意 `UEExec` | 是 | 否 | 否 | Shipping 不因 opt-in 变成通用远程控制台 |
| `RBNavLab` | 是 | 是 | 否 | 自动化面；Shipping 需 loopback + `-automationcontrol` |
| `UEExec RBNavGoto ...` / `UEExec RBTeleport ...` | 是 | 是 | 否 | 仅导航/三靶自动化白名单；Shipping 需 loopback + `-automationcontrol` |

外部视觉 Shipping 白名单只有：`RBTakeOver`、`RBExtAim`、
`r.VisionBridge.Enable`、`r.MotionBlurQuality`、`r.AntiAliasingMethod`、
`r.RobotNav.DebugDraw`、`t.MaxFPS`。不要把 Preview 日志里“UEExec 已收到”当成
任意命令均获授权。

### 常用 JSON-RPC

```json
{"type":0,"id":1,"method":"console.exec","params":{"command":"ExtAimClaim 0 1"}}
{"type":0,"id":2,"method":"console.exec","params":{"command":"UEExec RBExtAim 0 45.0 -3.5 0 30 0"}}
{"type":0,"id":3,"method":"rgbCamera.applySettings","params":{"camera":{"enabled":1,"fovDegrees":25,"shutterSpeed":120,"iso":600,"armLength":0}}}
```

## 4. 阻塞终端 B：红方 HACHISEN 完整对局

终端 A 的游戏停在 prep 且 `.wsport` 已出现后，在本仓库另开终端 B 并保持阻塞运行：

```powershell
$port = [int](Get-Content '<run>.log.wsport' -Raw)
$env:PATH = '<opencv-vc16-bin>;<openvino-libs>;' + $env:PATH
.\build\Release\gestalt.exe $port configs\gestalt_match_sentry.yaml --timeout=600
```

`gestalt_match_sentry.yaml` 的顺序是：

1. 在发送任何 `SetMatchStatus` 前完成 OpenVINO 模型编译、一次黑帧 infer warmup 和
   solver/tracker/aimer/shooter 构造；失败就停在 prep。
2. `Respawn 0 66000005 0`，等待 pid 0 对应的新 combat map 且 HP>0，并同时验证
   `PlayerID=0`、`TeamID=0`、`Class=1004` 与玩家表
   `ConnectionEntityConfigId=66000005`。
3. `SetAttribute 0 50000088 1`；不要在 claim 前自行写 TargetMode 90。
4. `ExtAimClaim 0 1` 原子保存旧 TargetMode 并切到 90，再 `UEExec RBTakeOver 0`；
   独立 RAII 线程每 1 秒重复同一 claim（5 秒无续租会自动恢复），因此模型推理、
   共享内存等待或 UI 阻塞不会误丢 lease；死亡等待前先停续租并显式发送
   `ExtAimClaim 0 0`，复活后再重新 claim。
5. 应用相机（必须含 `armLength:0`）并初始化共享帧协议 v3；要求至少 3 个
   frame id/QPC 严格递增的新帧同时满足 1280x720、FOV 25°、实际 armLength 0，且
   identity flags bit0..2 全为 1、player/map 等于当前接管 pid/map、ViewActor unique id
   非零且等于 takeover target、三帧 takeover epoch 相同且非零。
6. 从 AttributeMap 确认 TargetMode=90。复活路径还要等待 `Defeated=0`、
   `CanOperate=1`、`IsChassisOnline=1`、`Weakened=0`，且死亡物理姿态已恢复到可用的
   炮口 pitch 角域；随后才以 fire=0 下发小角度 `RBExtAim`，并从
   `TurretYaw/TurretPitch` 严格证明双轴到达后恢复原姿态。瞬态未到达会在回位、短暂稳定后
   用反向 yaw 再探一次，仍不能证明则硬失败。只有全部效果门通过才发送
   `SetMatchStatus 1`；底盘移动与经济仍由内置 AI 管理。
7. 每次复活只接受 HP 已经从 0 回到正值、map id 等于尸体 map（原地复活）或严格更高
   （重建 pawn）且四重身份仍正确的存活 pawn，再按 **claim → takeover → apply camera**
   重挂；重复 mode90、连续帧、FOV/armLength、物理就绪和小角度响应效果门后才恢复控制循环。
   死亡等待沿用整场 `--timeout` 的总截止时间，不另设更短的 180 秒截断。
8. WS/采集一旦中断，恢复后必须重新执行 **claim → takeover → apply camera** 与
   mode90、frame/FOV/armLength、RBExtAim 响应全门；不能证明就安全失败。WS 与像素流
   必须属于同一监听/writer PID。
9. 对局结算先用 fire=0 清掉锁存扳机，再 `ExtAimClaim 0 0` 和
   `UEExec RBTakeOver release`；等待 AttributeMap 证明 TargetMode 已不为 90（恢复
   claim 保存的旧值）。

异常退出也有两层失效保护：游戏侧若 0.75 秒没有新的 fire=1 `RBExtAim`，会原生
强制发送 fire=0；若 5 秒没有新的幂等 `ExtAimClaim <pid> 1` heartbeat，TS 会移除
claim 状态并恢复 claim 前的 TargetMode。该 lease 是每 pid 独立的 JS 墙钟单次
定时器，每次独立线程的 1s heartbeat 都会清旧并重置。因而视觉进程崩溃、被 taskkill 或 WS 断开
不会留下持续射击或整场 mode90 占用。

丢锁超过 2 秒后，桥按真实 `steady_clock` 时间推进 yaw 60°/s 与 pitch
±15° 三角波；时间步有上限，卡帧不会补跳。扫描激活时若 pitch 在带外，会平滑
回带。只要 tracker 前的 `armors_ui` 原始检测快照非空，两轴立即冻结，不等待
tracker 成功建轨；冻结目标取检测当下的遥测 yaw/pitch，绝不沿用旧扫描目标。

## 5. 回归与效果验收

### 快速协议 smoke

按矩阵逐项发送，并从**状态变化**判断，而非只看命令返回：

- `.wsport` 可读、WS 能订阅 AttributeMap，`MatchStatus(80000005)` 初始为 0。
- `gestalt.exe` 能打开共享帧协议 v3，frame id/QPC 严格前进，分辨率/FOV 与配置一致；
  连续三帧的 ViewActor/target、pid/map、epoch 与 identity flags 证明这些像素确实来自
  当前接管 pawn 的 FSceneView。
- 帧原子元数据的 `camera_arm_length_cm` 为 0，画面为枪管第一人称，不是 Tower
  400 cm 追尾视角。
- mode90 + claim 后连续下发两个 yaw/pitch 目标，遥测
  `TurretYaw/TurretPitch(10000111/10000112)` 朝目标变化。
- allowance 非零且火力门允许时，`BulletFiredTotal(63000002)` 随 fire 请求增加。
- `ExtAimClaim 0 0` 后 TargetMode 从 90 恢复；无 opt-in 的 Shipping 对同一外控
  命令应无效果。不可用“先 SetAttribute 90，再 claim”测试恢复语义。
- 中断 heartbeat 后约 5 秒（另加少量 JS 事件循环调度延迟）TargetMode 自动恢复；fire=1 后中断 `RBExtAim`，
  0.75 秒 watchdog 后不再保持扳机输入。

### 三靶算法回归（Development/Preview 测试资产）

三靶 bench 依赖 `RBNavLab`，不属于最终 Shipping 外控白名单；在 Development，或
专用 Preview 自动化中同时带
`-externalvisioncontrol -automationcontrol -visionbridge -wsbind=127.0.0.1`，再运行根
README 的三份配置：

```powershell
.\build\Release\gestalt.exe $port configs\gestalt_sentry_outpost.yaml --timeout=300
.\build\Release\gestalt.exe $port configs\gestalt_sentry_fortress_infantry.yaml --timeout=300
.\build\Release\gestalt.exe $port configs\gestalt_sentry_base_hero.yaml --timeout=300
```

验收 `RESULT` 中 `frames>0`、`det_rate>0`、`bullets>0`，并与根 README 的同构型
基线比较 hit rate；不要把 `fire_cmds` 当实际出膛数。

### 完整对局验收

- AttributeMap 观察到 `MatchStatus: 0 → 1 → 2`，而不是仅因 600 秒超时退出。
- `frames>0`、`det_rate>0`、`bullets>0`、`damage_dealt>0`。
- 每次死亡后出现同 map 原地复活或更高 map 的重建复活，`lives_lost` 与复活次数一致；
  复活后的 FOV、`armLength:0`、mode90 与外控角度响应仍成立。
- 每次生命的 BulletFired/DamageApplied 只结转一次；若比赛恰在死亡等待中结束，
  RESULT 不得把最后一条命重复计入 `bullets`/`damage_dealt`。此时 combat map 可能已被
  回收并把 TeamID 改为休眠哨兵值；验收沿用该生命存活期已通过的 HACHISEN 身份，
  死亡时立即停 heartbeat 并显式 release；teardown 再次幂等 release，并在 5 秒验证窗内
  以不存在存活控制对象作为安全交还证明，不读取回收 map 中可能冻结的 TargetMode=90
  作为失败依据。
- 扫描期间 yaw 速率按真实时间接近 60°/s；原始检测出现时不继续扫；长卡帧后
  单帧步进不超过 `60 * 0.20 = 12°`。
- `RESULT=MATCH_COMPLETE` 且进程退出码为 0；该裁决同时要求
  曾自然观察到 `match_status>=2`（锁存为 `match_end_seen=1`）、`frames>0`、
  `det_rate>0`、`bullets>0`、`damage_dealt>0`、所有
  takeover/revive/recovery 效果门、`pid0_hachisen=1`、`frame_writer_gate=1`、
  `view_identity_gate=1`、非零 `takeover_epoch`、`ws_continuity_gate=1`、
  `capture_recovery_gate=1`，以及 release 后的二选一安全门：活体 pid current map 的
  TargetMode 不再为 90；或自然结算恰逢死亡时，显式 release 已发送、无存活控制对象且
  heartbeat 在 5 秒验证窗内保持停止的 terminal-inactive 门。
  任一条件不满足均为 `MATCH_FAILED`/退出码 2。

2026-07-14 使用 CI 包体 `v0.1.9.65-preview` 的实测基线：红方 HACHISEN 自然完整对局
约 7.1 分钟，144 发、伤害 324、命中率 11.2%、`det_rate=0.260`、阵亡 4 次；前三次
同 map 原地复活均通过 FOV25/armLength0/身份/外控响应重挂，末次在自然结算时通过
terminal-inactive release，最终上述全部 RESULT 门为 true 且游戏日志无 claim lease error。

## 6. 最终 Shipping 的无日志判定

最终 Shipping 可能没有 Preview 的 `[AutoStart]`、`[UEExec]`、
`WebSocket server started` 等日志。发布验收必须只依赖：

1. `<abslog>.wsport` sidecar 发现实际随机端口；
2. WS 属性变化确认生成、mode90、云台响应、开火、伤害、复活和结算；
3. 共享帧 packet 的 frame id/QPC/相机元数据持续前进；
4. `gestalt.exe` 自身的 `RESULT` 与采集统计作为视觉侧证据；
5. 游戏进程正常退出/仍存活的进程状态，而不是某条 UE 日志是否出现。

先跑带日志 Preview 定位问题，再以同一参数跑无日志 Shipping smoke；只有两者都
满足效果判据，才能确认发布包外部接管链路可用。
