# Nexovara Pose

**Webcam-based full-body tracking for SteamVR and VRChat.**  
Version `0.8.9.1 beta` · MIT-licensed source code

[English](#english) · [Русский](#русский) · [Releases](https://github.com/FrizeTV/Nexovara-Pose/releases) · [Issues](https://github.com/FrizeTV/Nexovara-Pose/issues)

## English

Nexovara Pose estimates your 3D pose from one or more cameras connected to your PC and uses it to drive virtual body trackers. Use OSC to send trackers directly to VRChat, or use the bundled SteamVR driver to make them available to SteamVR games. The driver can also create a virtual headset and controllers for playing in a window on a monitor.

The app interface is currently in Russian. This guide uses English translations of the controls. See the [Russian guide](#русский) for the exact on-screen labels.

[Install](#install-and-run) · [Calibrate](#first-run-and-calibration) · [VRChat OSC](#connect-to-vrchat-over-osc) · [SteamVR](#connect-to-steamvr-with-a-physical-headset) · [Virtual headset](#run-steamvr-with-a-virtual-headset) · [Tracking options](#cameras-models-and-tracking-options) · [Build and test](#build-test-and-diagnose)

### Choose a tracking mode

| Mode | Use it for | Requirements |
| --- | --- | --- |
| **Auto (detect on startup)** | Automatically choose an output based on the running VR apps and the SteamVR driver state | Requirements depend on the selected output |
| **VRChat (OSC)** | Send body trackers directly to VRChat over local OSC | VRChat in VR mode, with OSC enabled |
| **SteamVR (Nexovara Pose driver)** | Use virtual body trackers with a physical headset in SteamVR games that support FBT | SteamVR and the Nexovara Pose driver |
| **SteamVR without a headset** | Play on a monitor with a virtual headset, body trackers and camera-driven controllers | SteamVR and the Nexovara Pose driver. Keyboard and mouse controls require Windows |
| **Test without a game** | Check tracking in the preview and inspect outgoing OSC packets | Nexovara Pose, with the OSC monitor optional |

The default trackers are the hip and both feet. **Settings → Trackers** can also enable the chest, both knees and both elbows. Tracker slots stay fixed when you enable or disable individual trackers. Each game decides which roles it can use.

### Install and run

To run from source, you need 64-bit Python. For tracking, use a webcam, a phone configured as a webcam, or a local video file. The setup scripts accept Python 3.10–3.14. Packages need compatible wheels for your Python version and platform. Python 3.12 is a practical starting point if installation fails on a newer version.

If a Windows binary is available under [Releases](https://github.com/FrizeTV/Nexovara-Pose/releases), extract the full release archive and follow its release notes. To run from source, download the source ZIP using **Code → Download ZIP**, extract it, and run the commands below from the folder containing `main.py`.

**Windows (PowerShell)**

```powershell
.\setup.bat
.\run.bat
```

**Linux or macOS (terminal)**

```bash
bash setup.sh
bash run.sh
```

Setup creates `.venv`, installs the base dependencies, offers the optional ONNX Runtime and OSCQuery dependencies, and runs a self-check. It enables portable storage and normally prepares the base models. On Linux, the Qt platform plugin may require system libraries such as `libxcb-cursor0` and `libegl1` on Debian-based distributions. The macOS script is for the desktop app. The SteamVR driver targets Windows and Linux.

### First run and calibration

1. Run **Auto setup** if it does not open on first launch. It checks available camera modes and benchmarks pose estimation. Review the camera and output mode under **Settings**. Auto setup keeps the settings you changed manually.
2. Position the camera so your whole body is visible from head to feet. Use even lighting. In **Settings → Body and calibration**, enter your height in metres.
3. Press **Start** (`F5`). Stand upright facing the camera, look straight ahead and keep your arms down. Press **Calibrate** (`F6`) and stay in position until it finishes. If using a physical headset with the driver, wear it during calibration.
4. Check the camera preview, the 3D skeleton and the tracker-confidence panel. Then run the game's own FBT calibration. If you move the main camera or change the tracking setup, repeat app calibration and game calibration.

In the app, `F5` starts or stops tracking, `F7` pauses the tracker pose, and `Ctrl+1` through `Ctrl+9` switch between saved profiles. Click a person in the preview to select them when several people are visible.

### Connect to VRChat over OSC

1. Select **Settings → Game and output → Game / application → VRChat (OSC)**. Start VRChat in VR mode on the same PC and enable **Action Menu → Options → OSC → Enabled**.
2. Start tracking and calibrate in Nexovara Pose. In VRChat, open **Quick Menu → Tracking & IK**. Enable **Toggle FBT** if needed and choose **Calibrate FBT**.
3. If the avatar faces the wrong way, face the camera, look straight ahead and press **Align heading in VRChat** while tracking is running.

The default OSC destination is `127.0.0.1:9000`. To anchor the body to the headset pose reported by VRChat, enable **Use the VRChat headset pose (OSCQuery)** and accept VRChat's prompt if it appears. This feature uses `zeroconf`. If headset data is unavailable, tracking continues using the camera. Use VRChat's VR mode for body trackers and the SteamVR driver for virtual finger controllers.

See VRChat's [OSC tracker documentation](https://docs.vrchat.com/docs/osc-trackers) and [FBT guide](https://docs.vrchat.com/docs/full-body-tracking).

### Connect to SteamVR with a physical headset

1. Build the driver using the instructions below, or use the complete driver folder supplied with a release. The app expects `driver/dist/nexovarapose/` relative to its project folder or executable.
2. Run SteamVR once if this is a new SteamVR installation, then close it. In Nexovara Pose, select **Settings → Game and output → Game / application → SteamVR (Nexovara Pose driver)**. Open **SteamVR diagnostics and setup… → Fix**, or use **Install or update in SteamVR**.
3. While SteamVR is closed, use **Assign SteamVR tracker roles…** to assign roles. Start SteamVR, then start and calibrate tracking in Nexovara Pose. Complete FBT calibration in the game.
4. After rebuilding the driver, close SteamVR and run **Install or update in SteamVR** again.

The driver sends the physical headset pose back to the app to align the body with it. The app matches the headset pose to the time the camera frame was captured. It can also send tracker velocities so SteamVR can predict motion between updates. The default driver port is `39580`. If you already use physical controllers, set finger tracking to **Preview only** to avoid registering extra controllers.

### Run SteamVR with a virtual headset

1. With SteamVR closed and your physical headset disconnected, select **Settings → Game and output → Game / application → SteamVR without a headset** and run **SteamVR diagnostics and setup… → Fix**. This registers the driver and configures SteamVR to start with the virtual headset. The same settings page has window size, vertical field of view and head rotation options.
2. Start SteamVR. Use its **Display VR View** option to view the scene on your monitor. Start and calibrate tracking in Nexovara Pose, then launch the game in VR mode.
3. On Windows, with keyboard and mouse controls enabled, use `F8` to open the VRChat Quick Menu, `F10` to aim the right hand with the mouse, and left-click **Calibrate FBT**. Stand in the calibration pose and press `F9` to simulate both triggers. Press `F10` again to return to camera-driven hand aiming.
4. Before returning to a physical headset, close SteamVR and disable virtual-headset mode using **Toggle headset-free mode**.

### Cameras, models and tracking options

| Feature | Where to find it and how it works |
| --- | --- |
| Camera and video input | **Camera** lists devices by name where available and lets you pick a local video file. Adjust capture resolution, requested FPS, pixel format, rotation, mirroring and vertical field of view. The app can retry a busy or disconnected camera and look it up again by name. |
| Multiple cameras | **Camera → Add camera** supports up to four cameras total. Put a second camera about 45–90° from the first with overlapping views. While tracking, move with your whole body visible and keep other people out of frame. Automatic camera alignment runs in the background. For manual alignment use **Calibrate cameras (8 s)**. The camera panel shows how many additional camera feeds contribute to the 3D pose. A phone can act as a webcam through a tool such as DroidCam, Iriun or Camo. |
| Body models | **Performance** offers MediaPipe Pose Lite, Full and Heavy. In **Single model / RTMPose** mode, you can use RTMPose-m or RTMPose-x to refine the MediaPipe pose. The ensemble modes can combine MediaPipe with up to four ONNX models chosen from RTMPose-m, RTMPose-x, MoveNet Lightning, MoveNet Thunder, RTMO-m and ViTPose-S WholeBody. |
| Model downloads and performance | Open **Performance → Download models in advance**, tick the models you want and click **Download selected models**. Then select automatic or manual mode under **Model ensemble**. Automatic mode selects from the downloaded models based on available memory and frame processing time. Running more models in manual mode can reduce tracking FPS. MediaPipe supplies depth. The additional models refine 2D joint positions when their estimates agree. The **Pose model** card reports active models and timing. MediaPipe uses the CPU on Windows. ONNX Runtime can use DirectML on Windows or CUDA where a compatible runtime is installed. Install only one ONNX Runtime package in the environment. |
| Person selection | **Person tracking** lets you lock tracking to one person, limit how many people the app considers and set when it switches to the only person in view. Selection uses pose, position and appearance cues in memory. The preview and confidence indicators help spot occlusion or a lost subject. |
| Smoothing and movement | **Smoothing and realism** controls interpolation, configurable output rate (90 Hz by default), prediction, per-body-part smoothing, rotation smoothing, correction for left and right leg swaps, joint limits, estimation for legs hidden from view, and transitions between standing, sitting and lying down. **Game and output** also has camera-based movement estimation for setups without a physical headset anchor. |
| Accuracy and body fit | **Accuracy** offers bone-length constraints, foot locking, floor clamping, visibility thresholds, One Euro filter controls and a configurable period for holding the last tracker poses if tracking is lost. **Body and calibration** lets you set your height, scale, yaw and position offsets. |
| Hands and fingers | In **Fingers**, enable the separate MediaPipe hand model and choose preview or SteamVR output. The driver can expose Index-style controllers with finger skeletons and gesture-based trigger and grip inputs. Finger inference frequency and smoothing are configurable. |
| Profiles | Use **Profiles** to save settings for different games, avatars or people, including calibration, tracker selection and smoothing. Switch in the toolbar or with `Ctrl+1` through `Ctrl+9`. |
| Recording and replay | **Record** records numerical pose data to a local file. In **Recording and replay**, you can replay a recording through the tracking pipeline, use it to find better smoothing settings or delete it. |

### Build, test and diagnose

**Windows driver.** Install Visual Studio Build Tools with the **Desktop development with C++** workload, then run `.\build_driver.bat`. The script builds the driver and runs its C++ tests.

**Linux driver.** Install CMake 3.16 or later and a C++17 compiler. The current `build_driver.sh` can exit at the first progress update. Run CMake directly from the project root.

```bash
cmake -S driver -B driver/build -DCMAKE_BUILD_TYPE=Release
cmake --build driver/build --parallel
(cd driver/build && ctest --output-on-failure)
```

CMake writes the binary and resources to `driver/dist/nexovarapose/`. The C++ tests cover the UDP protocol and driver behavior in a mock SteamVR host. For the Windows desktop build, run `.\build_exe.bat` after setup. It produces `dist\Nexovara Pose\Nexovara Pose.exe` in a folder that should be kept together. The driver is built separately.

**Python tests.** Run from the project root after setup. ONNX and OSCQuery tests can use the optional runtime dependencies. The ONNX fixture tests also need `onnx`.

```powershell
.\.venv\Scripts\python.exe -m pip install pytest python-osc onnx
.\.venv\Scripts\python.exe -m pytest tests
```

```bash
.venv/bin/python -m pip install pytest python-osc onnx
.venv/bin/python -m pytest tests
```

| Symptom | What to check |
| --- | --- |
| The camera opens but shows no frames | Close other apps using it, select the device again in **Camera**, and try a lower resolution or frame rate. |
| Tracking lags or feet drift | Check camera FPS, lighting, framing and model timings. Try MediaPipe Lite or fewer ONNX models. Start with hip and feet, recalibrate, and consider a second camera if depth is the issue. |
| SteamVR shows no trackers | Use **SteamVR diagnostics and setup…** to check that the driver is built, registered and loaded and that it receives data. Then check roles and the game's calibration. |
| VRChat shows no OSC trackers | Verify VR mode, OSC and the port. To inspect local traffic, close VRChat, select **Test without a game** and use `.\osc_monitor.bat` or `bash osc_monitor.sh`. |
| Setup or startup fails | On Windows use `.\selfcheck.bat` and `.\run_debug.bat`. On Linux/macOS run `.venv/bin/python -I tools/selfcheck.py` and check the terminal. In the app, **Privacy → Open folder** opens the data and logs folder. |

### Source layout

The Python app captures frames, estimates pose, combines camera views and applies calibration and filters. A separate sender publishes tracker poses to local OSC or to the C++ OpenVR driver. The driver registers the virtual devices with SteamVR.

| Path | Responsibility |
| --- | --- |
| [`main.py`](main.py), [`fbt/ui/`](fbt/ui/) | Qt startup, previews, setup wizard, settings, model downloads and SteamVR diagnostics |
| [`fbt/camera.py`](fbt/camera.py), [`fbt/multicam.py`](fbt/multicam.py), [`fbt/engine.py`](fbt/engine.py) | Camera capture, camera alignment and fusion, and the tracking pipeline |
| [`fbt/model_catalog.py`](fbt/model_catalog.py), [`fbt/pose.py`](fbt/pose.py), [`fbt/onnx_pose.py`](fbt/onnx_pose.py), [`fbt/rtmpose.py`](fbt/rtmpose.py), [`fbt/ensemble.py`](fbt/ensemble.py), [`fbt/hands.py`](fbt/hands.py) | Model registry, MediaPipe and ONNX inference, joint fusion and finger tracking |
| [`fbt/person.py`](fbt/person.py), [`fbt/skeleton.py`](fbt/skeleton.py), [`fbt/filters.py`](fbt/filters.py), [`fbt/realism.py`](fbt/realism.py), [`fbt/guidance.py`](fbt/guidance.py) | Person selection, skeleton calibration, filtering, pose constraints and on-screen guidance |
| [`fbt/sender.py`](fbt/sender.py), [`fbt/outputs.py`](fbt/outputs.py), [`fbt/hmd.py`](fbt/hmd.py), [`fbt/oscquery.py`](fbt/oscquery.py), [`fbt/steamvr.py`](fbt/steamvr.py), [`fbt/controls.py`](fbt/controls.py) | Timed output, OSC and driver packets, headset anchoring, SteamVR registration and simulated controls |
| [`fbt/config.py`](fbt/config.py), [`fbt/profiles.py`](fbt/profiles.py), [`fbt/recording.py`](fbt/recording.py), [`fbt/autotune.py`](fbt/autotune.py) | Local settings, profiles, pose recordings and tuning |
| [`fbt/presets.py`](fbt/presets.py), [`fbt/system.py`](fbt/system.py), [`fbt/mathutil.py`](fbt/mathutil.py), [`fbt/net.py`](fbt/net.py), [`fbt/osc.py`](fbt/osc.py) | Output presets, timing, 3D math, model-download networking and OSC encoding |
| [`driver/src/`](driver/src/), [`driver/nexovarapose/`](driver/nexovarapose/), [`driver/third_party/`](driver/third_party/) | C++ OpenVR driver, manifest and resources, and vendored Valve code |
| [`tests/`](tests/), [`driver/tests/`](driver/tests/), [`tools/`](tools/), [`models/README.txt`](models/README.txt) | Python and C++ tests, setup and diagnostic tools, and manual base-model downloads |

### Local data, limitations and license

Camera frames are processed locally. The recording feature stores numerical poses, and tracker packets go to `127.0.0.1`. The optional OSCQuery feature advertises its service over mDNS for discovery. Setup can fetch Python packages, and the app downloads model files over HTTPS when needed. SteamVR configuration changes are initiated from the UI with SteamVR closed and use backups.

With `portable.txt` present, settings, recordings and logs are stored in `data/` alongside the project or executable. Without the marker, the app uses `%APPDATA%\Nexovara Pose` on Windows, `${XDG_CONFIG_HOME:-~/.config}/Nexovara Pose` on Linux, or `~/Library/Application Support/Nexovara Pose` on macOS. Models may be in the project `models/` folder or in the app's data folder. Use **Privacy** to open the data folder, clear logs, or remove personal settings and recordings while keeping the model files.

This is a beta. A single camera estimates depth and can lose accuracy during occlusion, sideways movement or poor lighting. Additional cameras and models cost processing time, and the configured output rate is separate from camera and inference FPS. Hardware, SteamVR runtime and game compatibility can vary. Report problems in [Issues](https://github.com/FrizeTV/Nexovara-Pose/issues) with the OS, camera setup, output mode, steps to reproduce and relevant log lines.

The project's own code is under the [MIT License](LICENSE). See [Third-party notices](THIRD_PARTY_NOTICES.md) for dependencies, model terms and the Valve [OpenVR](driver/third_party/openvr/LICENSE) and [sample-code](driver/third_party/valve/LICENSE) licenses.

---

## Русский

Nexovara Pose оценивает трёхмерную позу человека по видео с одной или нескольких камер и превращает её в виртуальные трекеры тела. Программа может передавать трекеры напрямую в VRChat через OSC или предоставлять их играм SteamVR через собственный драйвер. Драйвер также умеет создавать виртуальный шлем и контроллеры для игры на мониторе.

[Установка](#установка-и-запуск) · [Калибровка](#первый-запуск-и-калибровка) · [VRChat OSC](#vrchat-через-osc) · [SteamVR](#steamvr-с-настоящим-шлемом) · [Виртуальный шлем](#steamvr-с-виртуальным-шлемом) · [Настройки трекинга](#камеры-модели-и-настройки-трекинга) · [Сборка и тесты](#сборка-тесты-и-диагностика)

### Какой режим выбрать

| Режим в приложении | Для чего нужен | Что потребуется |
| --- | --- | --- |
| **Авто (определять при запуске)** | Выбор вывода по запущенным VR-программам и состоянию драйвера SteamVR | Требования зависят от выбранного режима вывода |
| **VRChat (OSC)** | Передача трекеров напрямую в VRChat по локальному OSC | VRChat в режиме VR и включённый OSC |
| **SteamVR — драйвер Nexovara Pose** | Виртуальные трекеры с настоящим шлемом в играх SteamVR с поддержкой FBT | SteamVR и драйвер Nexovara Pose |
| **SteamVR без шлема** | Виртуальный шлем, трекеры и контроллеры с камеры для игры на мониторе | SteamVR и драйвер Nexovara Pose. Управление с клавиатуры и мыши работает в Windows |
| **Тест без игр** | Проверка предпросмотра и отправляемых координат | Приложение и, при необходимости, OSC-монитор |

По умолчанию включены три трекера: таз и обе стопы. В **Настройки → Трекеры** можно добавить грудь, оба колена и оба локтя. Номера трекеров закреплены за частями тела. Поддержка отдельных ролей зависит от игры.

### Установка и запуск

Для запуска из исходников нужен 64-битный Python. Для трекинга подойдёт веб-камера, телефон в режиме веб-камеры или локальный видеофайл. Скрипты установки принимают Python 3.10–3.14 при наличии готовых пакетов для вашей версии Python и системы. Если установка на более новой версии не проходит, попробуйте Python 3.12.

Если в [Releases](https://github.com/FrizeTV/Nexovara-Pose/releases) есть готовая сборка Windows, распакуйте архив целиком и следуйте инструкции релиза. Для запуска из исходников скачайте ZIP через **Code → Download ZIP**, распакуйте его и выполняйте команды в папке, где лежит `main.py`.

**Windows (PowerShell)**

```powershell
.\setup.bat
.\run.bat
```

**Linux или macOS (терминал)**

```bash
bash setup.sh
bash run.sh
```

Установщик создаёт `.venv`, ставит основные зависимости, предлагает установить ONNX Runtime и компоненты OSCQuery и запускает самопроверку. Он включает портативный режим и по умолчанию подготавливает базовые модели. В Linux плагину Qt могут понадобиться системные библиотеки, например `libxcb-cursor0` и `libegl1` в дистрибутивах на базе Debian. Скрипт для macOS предназначен для запуска приложения. Драйвер SteamVR собирается для Windows и Linux.

### Первый запуск и калибровка

1. Нажмите **Автонастройка**, если она не открылась при первом запуске. Программа проверит режимы камеры и измерит скорость обработки. Откройте **Настройки**, чтобы проверить камеру и режим вывода. Автонастройка сохраняет параметры, которые вы выставили вручную.
2. Поставьте камеру так, чтобы было видно всё тело от головы до стоп. Обеспечьте равномерное освещение. В **Настройки → Тело и калибровка** укажите рост в метрах.
3. Нажмите **Старт** (`F5`). Встаньте прямо лицом к камере, опустите руки и смотрите вперёд. Нажмите **Калибровка** (`F6`) и дождитесь завершения. Если используете драйвер с настоящим шлемом, наденьте шлем перед калибровкой.
4. Проверьте предпросмотр, 3D-скелет и уверенность трекеров. Затем выполните отдельную FBT-калибровку в игре. Если переставили основную камеру или поменяли схему трекинга, повторите обе калибровки.

В окне программы `F5` запускает и останавливает трекинг, `F7` удерживает позу трекеров, а `Ctrl+1`–`Ctrl+9` переключают сохранённые профили. Если в кадре несколько людей, нажмите на нужного человека в предпросмотре.

### VRChat через OSC

1. Выберите **Настройки → Игра и вывод → Игра / приложение → VRChat (OSC)**. Запустите VRChat в режиме VR на том же ПК и включите **Action Menu → Options → OSC → Enabled**.
2. Запустите трекинг и откалибруйтесь в Nexovara Pose. В VRChat откройте **Quick Menu → Tracking & IK**, при необходимости включите **Toggle FBT** и выберите **Calibrate FBT**.
3. Если аватар смотрит в другую сторону, встаньте лицом к камере, смотрите прямо и нажмите **Выровнять направление в VRChat** при запущенном трекинге.

Стандартный адрес OSC — `127.0.0.1:9000`. Чтобы привязать тело к позе шлема из VRChat, включите **Брать позу шлема из VRChat (OSCQuery) — точнее** и подтвердите запрос в игре, если он появится. Функции нужен пакет `zeroconf`. Если поза шлема недоступна, трекинг продолжится по камере. Для трекеров тела нужен режим VR, а для виртуальных контроллеров с пальцами — драйвер SteamVR.

См. документацию VRChat по [OSC-трекерам](https://docs.vrchat.com/docs/osc-trackers) и [полному отслеживанию тела](https://docs.vrchat.com/docs/full-body-tracking).

### SteamVR с настоящим шлемом

1. Соберите драйвер по инструкции ниже или возьмите полную папку драйвера из релиза. Приложение ищет её по пути `driver/dist/nexovarapose/` относительно папки проекта или исполняемого файла.
2. Если SteamVR установлен только что, запустите его один раз, затем закройте. В Nexovara Pose выберите **Настройки → Игра и вывод → Игра / приложение → SteamVR — драйвер Nexovara Pose**. Нажмите **Проверка и настройка SteamVR…**, затем **Исправить**. Для установки или обновления драйвера также есть кнопка **Установить / обновить в SteamVR**.
3. Пока SteamVR закрыт, нажмите **Назначить роли трекерам в SteamVR…**. Затем запустите SteamVR, включите трекинг и откалибруйтесь в Nexovara Pose. После этого выполните FBT-калибровку в игре.
4. После пересборки драйвера закройте SteamVR и снова нажмите **Установить / обновить в SteamVR**.

Драйвер передаёт приложению позу настоящего шлема, чтобы привязать к нему тело. Приложение сопоставляет её со временем съёмки кадра и может передавать скорости трекеров для упреждения в SteamVR. Стандартный порт драйвера — `39580`. При использовании физических контроллеров для трекинга пальцев выбирайте **Только показывать в программе**, чтобы SteamVR не зарегистрировал дополнительные контроллеры.

### SteamVR с виртуальным шлемом

1. Закройте SteamVR и отключите настоящий шлем. Выберите **Настройки → Игра и вывод → Игра / приложение → SteamVR без шлема** и нажмите **Проверка и настройка SteamVR…**, затем **Исправить**. Приложение зарегистрирует драйвер и настроит SteamVR на запуск с виртуальным шлемом. На той же странице можно изменить размер окна, вертикальный угол обзора и поворот виртуального шлема.
2. Запустите SteamVR. Чтобы видеть сцену на мониторе, откройте **Показать VR-вид**. Запустите трекинг и откалибруйтесь в Nexovara Pose, затем запустите игру в режиме VR.
3. В Windows при включённом управлении с клавиатуры и мыши нажмите `F8`, чтобы открыть Quick Menu VRChat. Нажмите `F10`, наведите правую руку мышью и щёлкните левой кнопкой по **Calibrate FBT**. Примите калибровочную позу и нажмите `F9` для имитации обоих курков. Повторное нажатие `F10` вернёт управление направлением руки с камеры.
4. Чтобы вернуться к настоящему шлему, закройте SteamVR и выключите виртуальный режим кнопкой **Включить/выключить режим без шлема**.

### Камеры, модели и настройки трекинга

| Возможность | Где найти и как работает |
| --- | --- |
| Камера и видеофайл | В разделе **Камера** можно выбрать устройство по имени, если оно доступно, или открыть локальный видеофайл. Настраиваются разрешение, запрашиваемый FPS, формат кадров, поворот, отражение и вертикальный угол обзора. Программа повторяет попытки захвата, если камера занята или отключена, и может найти её заново по имени. |
| Несколько камер | **Камера → Добавить камеру** поддерживает до четырёх камер суммарно. Поставьте вторую под углом примерно 45–90° к первой, чтобы зоны обзора пересекались. Во время трекинга двигайтесь так, чтобы камеры видели вас целиком, без посторонних в кадре. Программа автоматически уточняет взаимный поворот камер. Для ручного режима есть **Калибровка камер (8 с)**. Карточка камеры показывает число дополнительных потоков, участвующих в 3D-слиянии. Телефон можно подключить как веб-камеру через DroidCam, Iriun или Camo. |
| Модели тела | В разделе **Производительность** доступны MediaPipe Pose Lite, Full и Heavy. Режим **Одна модель / RTMPose** позволяет подключить RTMPose-m или RTMPose-x. В ансамбль можно добавить к MediaPipe до четырёх ONNX-моделей из каталога RTMPose-m, RTMPose-x, MoveNet Lightning, MoveNet Thunder, RTMO-m и ViTPose-S WholeBody. |
| Загрузка моделей и скорость | Откройте **Производительность → Модели: скачать заранее**, отметьте нужные модели и нажмите **Скачать выбранные**. Затем в блоке **Ансамбль моделей** выберите автоматический или ручной режим. Авто подбирает скачанные модели по запасу памяти и времени обработки кадра. Ручной выбор может снизить FPS обработки. MediaPipe даёт глубину, дополнительные модели уточняют 2D-координаты суставов, если их оценки согласуются. Активные модели и время обработки видны в карточке **Нейросеть**. В Windows MediaPipe работает на процессоре, а ONNX Runtime может использовать DirectML. CUDA доступна с совместимым пакетом ONNX Runtime. В одном окружении нужен один пакет ONNX Runtime. |
| Выбор человека | **Человек в кадре** позволяет закрепить трекинг за одним человеком, ограничить число кандидатов и настроить переключение на единственного человека в кадре. Признаки положения, позы и внешнего вида обрабатываются в памяти. Предпросмотр и уверенность трекеров показывают, когда часть тела перекрыта или человек потерян. |
| Плавность и движение | В разделе **Плавность и реализм** настраиваются интерполяция, частота отправки (по умолчанию 90 Гц), предсказание, сглаживание по частям тела, плавность поворота, коррекция перепутанных ног, ограничения суставов, оценка скрытых ног и переходы между позами стоя, сидя и лёжа. В разделе **Игра и вывод** есть оценка перемещения по кадру для режимов без привязки к настоящему шлему. |
| Точность и размеры тела | В разделе **Точность** можно зафиксировать длину костей и положение стоп, ограничить движение ниже пола и настроить пороги видимости, фильтр One Euro и время удержания позы при потере трекинга. В разделе **Тело и калибровка** настраиваются рост, масштаб, поворот и смещения. |
| Руки и пальцы | В разделе **Пальцы** включается отдельная модель MediaPipe для рук. Результат можно показывать в приложении или передавать в SteamVR. Драйвер может создавать контроллеры в стиле Index со скелетом пальцев и жестами курка и хвата. Настраиваются частота распознавания и сглаживание. |
| Профили | В разделе **Профили** можно сохранить настройки под игру, аватар или человека, включая калибровку, трекеры и сглаживание. Профили переключаются на верхней панели или клавишами `Ctrl+1`–`Ctrl+9`. |
| Запись и повтор | **Запись** сохраняет числовые координаты позы в локальный файл. В разделе **Запись и повтор** можно воспроизвести запись вместо видео с камеры, подобрать по ней настройки сглаживания или удалить её. |

### Сборка, тесты и диагностика

**Драйвер для Windows.** Установите Visual Studio Build Tools с компонентом **Разработка классических приложений на C++** и запустите `.\build_driver.bat`. Скрипт собирает драйвер и запускает тесты C++.

**Драйвер для Linux.** Нужны CMake 3.16 или новее и компилятор C++17. Текущий `build_driver.sh` может завершиться на первом обновлении индикатора прогресса. Соберите драйвер напрямую из корня проекта:

```bash
cmake -S driver -B driver/build -DCMAKE_BUILD_TYPE=Release
cmake --build driver/build --parallel
(cd driver/build && ctest --output-on-failure)
```

CMake помещает библиотеку и ресурсы в `driver/dist/nexovarapose/`. Тесты C++ проверяют UDP-протокол и работу драйвера в имитации SteamVR. Для сборки Windows-приложения запустите `.\build_exe.bat` после установки зависимостей. Результат — `dist\Nexovara Pose\Nexovara Pose.exe` и соседние файлы, которые нужно хранить вместе. Драйвер собирается отдельно.

**Тесты Python.** Запускайте из корня проекта после установки. Для части тестов нужны дополнительные зависимости ONNX Runtime и OSCQuery. Тестам с искусственной ONNX-моделью также нужен `onnx`.

```powershell
.\.venv\Scripts\python.exe -m pip install pytest python-osc onnx
.\.venv\Scripts\python.exe -m pytest tests
```

```bash
.venv/bin/python -m pip install pytest python-osc onnx
.venv/bin/python -m pytest tests
```

| Проблема | Что проверить |
| --- | --- |
| Камера открывается, но кадров нет | Закройте другие программы, которые используют камеру, выберите её заново в разделе **Камера** и попробуйте снизить разрешение или частоту кадров. |
| Трекинг запаздывает или стопы смещаются | Проверьте FPS камеры, свет, видимость тела и время обработки моделей. Попробуйте MediaPipe Lite или меньше ONNX-моделей. Начните с таза и стоп, откалибруйтесь заново. Если проблема с глубиной, попробуйте вторую камеру. |
| SteamVR не показывает трекеры | Откройте **Проверка и настройка SteamVR…** и проверьте сборку, регистрацию, загрузку драйвера и приём данных. Затем проверьте роли и калибровку в игре. |
| VRChat не видит OSC-трекеры | Проверьте режим VR, OSC и порт. Чтобы посмотреть локальные пакеты, закройте VRChat, выберите **Тест без игр** и запустите `.\osc_monitor.bat` или `bash osc_monitor.sh`. |
| Установка или запуск завершаются ошибкой | В Windows используйте `.\selfcheck.bat` и `.\run_debug.bat`. В Linux/macOS запустите `.venv/bin/python -I tools/selfcheck.py` и проверьте вывод терминала. В приложении **Конфиденциальность → Открыть папку** открывает данные и журналы. |

### Структура проекта

Приложение на Python получает кадры, оценивает позу, объединяет данные камер и применяет калибровку и фильтры. Отдельный поток отправляет позы трекеров через локальный OSC или в драйвер OpenVR на C++. Драйвер регистрирует виртуальные устройства в SteamVR.

| Путь | Назначение |
| --- | --- |
| [`main.py`](main.py), [`fbt/ui/`](fbt/ui/) | Запуск Qt, предпросмотр, мастер настройки, параметры, загрузка моделей и диагностика SteamVR |
| [`fbt/camera.py`](fbt/camera.py), [`fbt/multicam.py`](fbt/multicam.py), [`fbt/engine.py`](fbt/engine.py) | Захват, калибровка и слияние камер, конвейер трекинга |
| [`fbt/model_catalog.py`](fbt/model_catalog.py), [`fbt/pose.py`](fbt/pose.py), [`fbt/onnx_pose.py`](fbt/onnx_pose.py), [`fbt/rtmpose.py`](fbt/rtmpose.py), [`fbt/ensemble.py`](fbt/ensemble.py), [`fbt/hands.py`](fbt/hands.py) | Каталог, инференс MediaPipe и ONNX, слияние координат и трекинг пальцев |
| [`fbt/person.py`](fbt/person.py), [`fbt/skeleton.py`](fbt/skeleton.py), [`fbt/filters.py`](fbt/filters.py), [`fbt/realism.py`](fbt/realism.py), [`fbt/guidance.py`](fbt/guidance.py) | Выбор человека, калибровка скелета, фильтры, ограничения поз и подсказки |
| [`fbt/sender.py`](fbt/sender.py), [`fbt/outputs.py`](fbt/outputs.py), [`fbt/hmd.py`](fbt/hmd.py), [`fbt/oscquery.py`](fbt/oscquery.py), [`fbt/steamvr.py`](fbt/steamvr.py), [`fbt/controls.py`](fbt/controls.py) | Частота отправки, пакеты OSC и драйвера, привязка к шлему, регистрация SteamVR и управление |
| [`fbt/config.py`](fbt/config.py), [`fbt/profiles.py`](fbt/profiles.py), [`fbt/recording.py`](fbt/recording.py), [`fbt/autotune.py`](fbt/autotune.py) | Локальные настройки, профили, записи позы и подбор параметров |
| [`fbt/presets.py`](fbt/presets.py), [`fbt/system.py`](fbt/system.py), [`fbt/mathutil.py`](fbt/mathutil.py), [`fbt/net.py`](fbt/net.py), [`fbt/osc.py`](fbt/osc.py) | Режимы вывода, системный таймер, 3D-математика, загрузка моделей и кодирование OSC |
| [`driver/src/`](driver/src/), [`driver/nexovarapose/`](driver/nexovarapose/), [`driver/third_party/`](driver/third_party/) | Драйвер OpenVR на C++, его манифест и ресурсы, включённый код Valve |
| [`tests/`](tests/), [`driver/tests/`](driver/tests/), [`tools/`](tools/), [`models/README.txt`](models/README.txt) | Тесты Python и C++, инструменты установки и диагностики, ручная загрузка базовых моделей |

### Локальные данные, ограничения и лицензия

Кадры камеры обрабатываются локально. Запись сохраняет числовые данные позы, а пакеты трекеров отправляются на `127.0.0.1`. Если включён OSCQuery, он публикует объявление службы через mDNS для обнаружения. При установке скачиваются пакеты Python, а приложение при необходимости загружает файлы моделей по HTTPS. Настройки SteamVR меняются по команде пользователя при закрытом SteamVR и с резервным копированием.

Если рядом с проектом или исполняемым файлом лежит `portable.txt`, настройки, записи и журналы хранятся в `data/`. Без этого файла используются `%APPDATA%\Nexovara Pose` в Windows, `${XDG_CONFIG_HOME:-~/.config}/Nexovara Pose` в Linux или `~/Library/Application Support/Nexovara Pose` в macOS. Модели могут лежать в `models/` проекта или в папке данных. В разделе **Конфиденциальность** можно открыть папку, очистить журналы или сбросить личные настройки и записи с сохранением весов моделей.

Это бета-версия. Одна камера оценивает глубину приблизительно. Точность падает, когда тело частично перекрыто, вы поворачиваетесь боком или в комнате мало света. Дополнительные камеры и модели требуют больше ресурсов. Частота отправки трекеров и фактический FPS камеры и инференса измеряются отдельно. Совместимость зависит от оборудования, версии SteamVR и игры. Если нашли проблему, откройте [Issue](https://github.com/FrizeTV/Nexovara-Pose/issues) и приложите версию ОС, схему камер, режим вывода, шаги воспроизведения и относящиеся к проблеме строки журнала.

Собственный код проекта распространяется по [лицензии MIT](LICENSE). Лицензии зависимостей и условия использования моделей описаны в [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Лицензии Valve лежат отдельно для [OpenVR](driver/third_party/openvr/LICENSE) и [примеров кода](driver/third_party/valve/LICENSE).
