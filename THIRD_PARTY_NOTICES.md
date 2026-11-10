# Nexovara Pose — third-party notices

Nexovara Pose is distributed under the MIT License in `LICENSE`. The following
components are used by the source project or its optional build/test paths.
Their licenses remain applicable to the component and are not replaced by the
Nexovara Pose license.

| Component | Use | License / notice |
|---|---|---|
| Valve OpenVR headers | SteamVR driver ABI | BSD-3-Clause; see `driver/third_party/openvr/LICENSE` |
| Valve hand simulation sample | Index-style finger skeleton | BSD-3-Clause; see `driver/third_party/valve/LICENSE` |
| MediaPipe Tasks | Pose and hand landmarks | Apache-2.0; Google LLC |
| MediaPipe `.task` models | Downloaded pose/hand models | Keep the model's upstream license and attribution with the downloaded model |
| OpenCV / opencv-contrib-python | Camera and image processing | Apache-2.0 and bundled third-party notices from the wheel |
| NumPy | Numerical processing | BSD-3-Clause; retain NumPy notices from the installed distribution |
| PySide6-Essentials / Qt | User interface | LGPL-3.0/GPL/commercial Qt terms; see the PySide6 and Qt distribution notices |
| ONNX Runtime | RTMPose and ONNX inference | MIT; retain the ONNX Runtime notice |
| zeroconf | OSCQuery/mDNS discovery | LGPL-2.1 |
| python-osc | Optional tests/tools | MIT |
| pytest | Tests only | MIT |
| PyInstaller | Optional executable packaging | GPL-2.0-or-later with the PyInstaller exception |
| RTMPose / MMPose exported models | Optional refinement models | Apache-2.0; retain upstream model attribution and license |
| [RTMO](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmo) | Optional ONNX model from OpenMMLab | See the upstream MMPose license and model terms |
| [MoveNet Lightning](https://huggingface.co/Xenova/movenet-singlepose-lightning) / [Thunder](https://huggingface.co/Xenova/movenet-singlepose-thunder) | Optional ONNX exports | Apache-2.0 per the model cards |
| [ViTPose-S WholeBody](https://huggingface.co/JunkyByte/easy_ViTPose) | Optional ONNX export | See the source project and upstream model terms |

## Distribution rules

Do not redistribute SteamVR itself or Valve runtime binaries. Nexovara Pose only
ships its own driver, the OpenVR interface header, and the required notices.

When publishing a binary package, include this file, `LICENSE`, both Valve
license files, and the notices/licenses shipped by the dependency wheels or
installer. Model files downloaded after installation must retain their upstream
license and attribution; a model hash in `data/models.json` is an integrity
record, not a license grant.

License summaries are provided for packaging guidance. Before commercial
distribution, verify the exact versions installed in the release environment
and include their original license texts and NOTICE files.
