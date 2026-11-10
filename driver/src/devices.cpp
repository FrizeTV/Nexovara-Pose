#include "devices.h"

#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>

#include "hand_simulation.h"  // Valve sample (BSD-3): finger curls -> 31-bone hand skeleton

namespace nexovarapose {

bool g_predict_velocity = true;

void Log(const char* fmt, ...) {
    char buf[1024];
    va_list args;
    va_start(args, fmt);
    std::vsnprintf(buf, sizeof buf, fmt, args);
    va_end(args);
    if (vr::VRDriverLog()) vr::VRDriverLog()->Log(buf);
}

// --------------------------------------------------------------- base
vr::DriverPose_t DeviceBase::MakePose(const DeviceSample& s, bool valid) {
    vr::DriverPose_t pose{};
    pose.qWorldFromDriverRotation.w = 1.0;
    pose.qDriverFromHeadRotation.w = 1.0;
    pose.vecPosition[0] = s.pos[0];
    pose.vecPosition[1] = s.pos[1];
    pose.vecPosition[2] = s.pos[2];
    pose.qRotation.x = s.rot[0];
    pose.qRotation.y = s.rot[1];
    pose.qRotation.z = s.rot[2];
    pose.qRotation.w = s.rot[3];
    pose.poseIsValid = valid;
    pose.deviceIsConnected = true;
    pose.result = valid ? vr::TrackingResult_Running_OK : vr::TrackingResult_Running_OutOfRange;
    pose.willDriftInYaw = false;
    pose.shouldApplyHeadModel = false;  // the app already sends eye/hand positions
    return pose;
}

vr::EVRInitError DeviceBase::Activate(uint32_t object_id) {
    index_ = object_id;
    vr::PropertyContainerHandle_t c = vr::VRProperties()->TrackedDeviceToPropertyContainer(index_);
    vr::VRProperties()->SetStringProperty(c, vr::Prop_TrackingSystemName_String, "nexovarapose");
    vr::VRProperties()->SetStringProperty(c, vr::Prop_ManufacturerName_String, "Nexovara Pose");
    vr::VRProperties()->SetStringProperty(c, vr::Prop_SerialNumber_String, serial_.c_str());
    SetupProperties(c);
    last_pose_ = MakePose(DeviceSample{}, false);
    return vr::VRInitError_None;
}

void DeviceBase::Update(const SampleStore& store, Clock::time_point now) {
    if (index_ == vr::k_unTrackedDeviceIndexInvalid) return;
    DeviceSample s;
    double age = 1e9;
    if (!store.Get(id_, s, age, now)) return;
    const bool valid = age < kStaleSeconds;
    if (valid) OnSample(s);
    last_pose_ = MakePose(s, valid);
    if (valid && g_predict_velocity && (s.flags & kFlagVelocity)) {
        // The app's velocity (computed from its smooth output stream). The sample
        // is `age` old: tell SteamVR, so between two packets it moves the tracker on
        // along the velocity instead of drawing it in steps (headsets run at 90-144 Hz).
        for (int k = 0; k < 3; ++k) last_pose_.vecVelocity[k] = s.vel[k];
        last_pose_.poseTimeOffset = -(age < kMaxSampleAge ? age : kMaxSampleAge);
    }
    vr::VRServerDriverHost()->TrackedDevicePoseUpdated(index_, last_pose_, sizeof(vr::DriverPose_t));
}

// --------------------------------------------------------------- tracker
void TrackerDevice::SetupProperties(vr::PropertyContainerHandle_t c) {
    // looks and behaves like a VIVE Tracker, so every FBT game accepts it
    vr::VRProperties()->SetStringProperty(c, vr::Prop_ModelNumber_String, "Nexovara Pose Tracker");
    vr::VRProperties()->SetStringProperty(c, vr::Prop_RenderModelName_String, "{htc}vr_tracker_vive_1_0");
    vr::VRProperties()->SetStringProperty(c, vr::Prop_InputProfilePath_String, "{htc}/input/vive_tracker_profile.json");
    vr::VRProperties()->SetStringProperty(c, vr::Prop_ControllerType_String, "vive_tracker");
    vr::VRProperties()->SetInt32Property(c, vr::Prop_ControllerRoleHint_Int32, vr::TrackedControllerRole_OptOut);
}

// --------------------------------------------------------------- hand
HandDevice::HandDevice(uint8_t id, std::string serial, bool left)
    : DeviceBase(id, std::move(serial)), left_(left), sim_(std::make_unique<MyHandSimulation>()) {}

HandDevice::~HandDevice() = default;

void HandDevice::SetupProperties(vr::PropertyContainerHandle_t c) {
    // Index-controller compatible: games get skeletal fingers and Index bindings
    auto* p = vr::VRProperties();
    p->SetStringProperty(c, vr::Prop_ModelNumber_String, "Nexovara Pose Hand");
    p->SetInt32Property(c, vr::Prop_ControllerRoleHint_Int32,
                        left_ ? vr::TrackedControllerRole_LeftHand : vr::TrackedControllerRole_RightHand);
    p->SetStringProperty(c, vr::Prop_InputProfilePath_String, "{indexcontroller}/input/index_controller_profile.json");
    p->SetStringProperty(c, vr::Prop_ControllerType_String, "knuckles");
    p->SetStringProperty(c, vr::Prop_RenderModelName_String,
                         left_ ? "{indexcontroller}valve_controller_knu_1_0_left"
                               : "{indexcontroller}valve_controller_knu_1_0_right");
    auto* in = vr::VRDriverInput();
    const auto abs1 = vr::VRScalarType_Absolute;
    const auto one = vr::VRScalarUnits_NormalizedOneSided;
    in->CreateScalarComponent(c, "/input/trigger/value", &h_[kTriggerValue], abs1, one);
    in->CreateBooleanComponent(c, "/input/trigger/click", &h_[kTriggerClick]);
    in->CreateBooleanComponent(c, "/input/trigger/touch", &h_[kTriggerTouch]);
    in->CreateScalarComponent(c, "/input/grip/value", &h_[kGripValue], abs1, one);
    in->CreateScalarComponent(c, "/input/grip/force", &h_[kGripForce], abs1, one);
    in->CreateBooleanComponent(c, "/input/grip/touch", &h_[kGripTouch]);
    in->CreateScalarComponent(c, "/input/finger/index", &h_[kFingerIndex], abs1, one);
    in->CreateScalarComponent(c, "/input/finger/middle", &h_[kFingerMiddle], abs1, one);
    in->CreateScalarComponent(c, "/input/finger/ring", &h_[kFingerRing], abs1, one);
    in->CreateScalarComponent(c, "/input/finger/pinky", &h_[kFingerPinky], abs1, one);
    in->CreateSkeletonComponent(c, left_ ? "/input/skeleton/left" : "/input/skeleton/right",
                                left_ ? "/skeleton/hand/left" : "/skeleton/hand/right", "/pose/raw",
                                vr::VRSkeletalTracking_Partial, nullptr, 0, &h_[kSkeleton]);
    // buttons of the Index profile: pressed from the keyboard/mouse of the app
    // ("SteamVR without a headset": VRChat's Quick Menu is B, calibration = triggers)
    in->CreateBooleanComponent(c, "/input/a/click", &h_[kAClick]);
    in->CreateBooleanComponent(c, "/input/a/touch", &h_[kATouch]);
    in->CreateBooleanComponent(c, "/input/b/click", &h_[kBClick]);
    in->CreateBooleanComponent(c, "/input/b/touch", &h_[kBTouch]);
    in->CreateBooleanComponent(c, "/input/system/click", &h_[kSystemClick]);
    in->CreateBooleanComponent(c, "/input/system/touch", &h_[kSystemTouch]);
    in->CreateBooleanComponent(c, "/input/thumbstick/click", &h_[kStickClick]);
    in->CreateBooleanComponent(c, "/input/thumbstick/touch", &h_[kStickTouch]);
    const auto two = vr::VRScalarUnits_NormalizedTwoSided;
    in->CreateScalarComponent(c, "/input/thumbstick/x", &h_[kStickX], abs1, two);
    in->CreateScalarComponent(c, "/input/thumbstick/y", &h_[kStickY], abs1, two);
    in->CreateHapticComponent(c, "/output/haptic", &h_[kHaptic]);  // games may vibrate: accept it
}

void HandDevice::OnSample(const DeviceSample& s) {
    auto* in = vr::VRDriverInput();
    float c[5];  // thumb, index, middle, ring, little
    for (int k = 0; k < 5; ++k) c[k] = (s.flags & kFlagCurls) ? s.curls[k] : 0.0f;
    const uint16_t b = (s.flags & kFlagButtons) ? s.buttons : 0;
    // keys of the app act like pulling the trigger / squeezing the controller
    if (b & kBtnTrigger) c[1] = 1.0f;
    if (b & kBtnGrip) c[2] = c[3] = c[4] = 1.0f;
    const float grip = (c[2] + c[3] + c[4]) / 3.0f;
    // gestures: bend the index finger = trigger, make a fist = grip
    in->UpdateScalarComponent(h_[kTriggerValue], c[1], 0);
    in->UpdateBooleanComponent(h_[kTriggerClick], c[1] > 0.85f, 0);
    in->UpdateBooleanComponent(h_[kTriggerTouch], c[1] > 0.20f, 0);
    in->UpdateBooleanComponent(h_[kAClick], (b & kBtnA) != 0, 0);
    in->UpdateBooleanComponent(h_[kATouch], (b & kBtnA) != 0, 0);
    in->UpdateBooleanComponent(h_[kBClick], (b & kBtnB) != 0, 0);
    in->UpdateBooleanComponent(h_[kBTouch], (b & kBtnB) != 0, 0);
    in->UpdateBooleanComponent(h_[kSystemClick], (b & kBtnSystem) != 0, 0);
    in->UpdateBooleanComponent(h_[kSystemTouch], (b & kBtnSystem) != 0, 0);
    in->UpdateBooleanComponent(h_[kStickClick], (b & kBtnStick) != 0, 0);
    in->UpdateBooleanComponent(h_[kStickTouch], (b & kBtnStick) != 0, 0);
    in->UpdateScalarComponent(h_[kGripValue], grip, 0);
    in->UpdateScalarComponent(h_[kGripForce], grip > 0.8f ? (grip - 0.8f) / 0.2f : 0.0f, 0);
    in->UpdateBooleanComponent(h_[kGripTouch], grip > 0.20f, 0);
    in->UpdateScalarComponent(h_[kFingerIndex], c[1], 0);
    in->UpdateScalarComponent(h_[kFingerMiddle], c[2], 0);
    in->UpdateScalarComponent(h_[kFingerRing], c[3], 0);
    in->UpdateScalarComponent(h_[kFingerPinky], c[4], 0);
    vr::VRBoneTransform_t bones[31];
    sim_->ComputeSkeletonTransforms(left_ ? vr::TrackedControllerRole_LeftHand : vr::TrackedControllerRole_RightHand,
                                    MyFingerCurls{c[0], c[1], c[2], c[3], c[4]},
                                    MyFingerSplays{0, 0, 0, 0, 0}, bones);
    in->UpdateSkeletonComponent(h_[kSkeleton], vr::VRSkeletalMotionRange_WithController, bones, 31);
    in->UpdateSkeletonComponent(h_[kSkeleton], vr::VRSkeletalMotionRange_WithoutController, bones, 31);
}

// --------------------------------------------------------------- display (virtual HMD window)
void DisplayComponent::GetWindowBounds(int32_t* x, int32_t* y, uint32_t* w, uint32_t* h) {
    *x = c_.window_x;
    *y = c_.window_y;
    *w = c_.window_width;
    *h = c_.window_height;
}

void DisplayComponent::GetRecommendedRenderTargetSize(uint32_t* w, uint32_t* h) {
    *w = c_.render_width;
    *h = c_.render_height;
}

void DisplayComponent::GetEyeOutputViewport(vr::EVREye eye, uint32_t* x, uint32_t* y, uint32_t* w, uint32_t* h) {
    *y = 0;
    *w = c_.window_width / 2;
    *h = c_.window_height;
    *x = eye == vr::Eye_Left ? 0 : c_.window_width / 2;
}

void DisplayComponent::GetProjectionRaw(vr::EVREye, float* l, float* r, float* t, float* b) {
    // tangents of the half angles; the horizontal one follows the eye viewport's
    // aspect ratio, so the picture in the window is not stretched
    const float tv = std::tan(c_.vertical_fov_deg * 0.5f * 3.14159265f / 180.0f);
    const float aspect = float(c_.window_width / 2) / float(c_.window_height ? c_.window_height : 1);
    *l = -tv * aspect;
    *r = tv * aspect;
    *t = -tv;
    *b = tv;
}

vr::DistortionCoordinates_t DisplayComponent::ComputeDistortion(vr::EVREye, float u, float v) {
    vr::DistortionCoordinates_t d{};  // a flat monitor: no lens distortion
    d.rfRed[0] = d.rfGreen[0] = d.rfBlue[0] = u;
    d.rfRed[1] = d.rfGreen[1] = d.rfBlue[1] = v;
    return d;
}

// --------------------------------------------------------------- HMD
const char* const kChaperoneJson = R"({
  "json_id": "chaperone_info",
  "version": 5,
  "time": "2026-10-26T00:00:00",
  "universes": [
    {
      "universeID": "1128678996",
      "play_area": [3.0, 3.0],
      "collision_bounds": [
        [[-1.5, 0.0, -1.5], [-1.5, 2.4, -1.5], [1.5, 2.4, -1.5], [1.5, 0.0, -1.5]],
        [[1.5, 0.0, -1.5], [1.5, 2.4, -1.5], [1.5, 2.4, 1.5], [1.5, 0.0, 1.5]],
        [[1.5, 0.0, 1.5], [1.5, 2.4, 1.5], [-1.5, 2.4, 1.5], [-1.5, 0.0, 1.5]],
        [[-1.5, 0.0, 1.5], [-1.5, 2.4, 1.5], [-1.5, 2.4, -1.5], [-1.5, 0.0, -1.5]]
      ],
      "seated": {"translation": [0.0, 1.6, 0.0], "yaw": 0.0},
      "standing": {"translation": [0.0, 0.0, 0.0], "yaw": 0.0}
    }
  ]
})";

HmdDevice::HmdDevice(uint8_t id, std::string serial, const DisplayConfig& cfg)
    : DeviceBase(id, std::move(serial)), display_(std::make_unique<DisplayComponent>(cfg)) {}

void* HmdDevice::GetComponent(const char* name) {
    if (std::strcmp(name, vr::IVRDisplayComponent_Version) == 0) return display_.get();
    return nullptr;
}

void HmdDevice::SetupProperties(vr::PropertyContainerHandle_t c) {
    auto* p = vr::VRProperties();
    p->SetStringProperty(c, vr::Prop_ModelNumber_String, "Nexovara Pose Virtual HMD");
    const float ipd = vr::VRSettings()->GetFloat(vr::k_pch_SteamVR_Section, vr::k_pch_SteamVR_IPD_Float);
    p->SetFloatProperty(c, vr::Prop_UserIpdMeters_Float, ipd);
    p->SetFloatProperty(c, vr::Prop_DisplayFrequency_Float, 0.0f);
    p->SetFloatProperty(c, vr::Prop_UserHeadToEyeDepthMeters_Float, 0.0f);
    p->SetFloatProperty(c, vr::Prop_SecondsFromVsyncToPhotons_Float, 0.011f);
    p->SetBoolProperty(c, vr::Prop_IsOnDesktop_Bool, false);
    p->SetBoolProperty(c, vr::Prop_DisplayDebugMode_Bool, true);
    p->SetStringProperty(c, vr::Prop_InputProfilePath_String, "{nexovarapose}/input/nexovarapose_hmd_profile.json");
    // Our own "room": without it SteamVR logs "[Chaperone] The universe id is invalid",
    // keeps asking for Room Setup and games may not get a valid standing space.
    // Standing = the app's space (floor at y = 0), a 3x3 m area, bounds hidden.
    p->SetUint64Property(c, vr::Prop_CurrentUniverseId_Uint64, kUniverseId);
    p->SetStringProperty(c, vr::Prop_DriverProvidedChaperoneJson_String, kChaperoneJson);
    p->SetBoolProperty(c, vr::Prop_DriverProvidedChaperoneVisibility_Bool, false);
}

void HmdDevice::Update(const SampleStore& store, Clock::time_point now) {
    if (index_ == vr::k_unTrackedDeviceIndexInvalid) return;
    DeviceSample s;
    double age = 1e9;
    if (!store.Get(id_, s, age, now)) {
        // no data yet: stand at eye height at the origin so SteamVR can start
        s.pos[1] = 1.6f;
        last_pose_ = MakePose(s, true);
    } else {
        last_pose_ = MakePose(s, true);  // an HMD must stay valid; it just stops moving
    }
    vr::VRServerDriverHost()->TrackedDevicePoseUpdated(index_, last_pose_, sizeof(vr::DriverPose_t));
}

}  // namespace nexovarapose
