// SteamVR devices fed by the Nexovara Pose app: body trackers, hands (Index-style
// controllers with finger skeleton), and an optional virtual HMD.
#pragma once

#include <atomic>
#include <memory>
#include <string>

#include "openvr_driver.h"
#include "protocol.h"
#include "receiver.h"

class MyHandSimulation;

namespace nexovarapose {

void Log(const char* fmt, ...);

constexpr double kStaleSeconds = 0.5;  // no data for this long -> "out of range"
// pass the app's velocity on to SteamVR, so it predicts trackers like the headset
// and moves them smoothly between packets (setting driver_nexovarapose.predictVelocity)
extern bool g_predict_velocity;
constexpr double kMaxSampleAge = 0.05;   // s: extrapolate at most this far
constexpr uint64_t kUniverseId = 1128678996;  // "CFBT"; 0 is invalid, 1 is Oculus
extern const char* const kChaperoneJson;     // our room for "without a headset"

// Shared pose plumbing.
class DeviceBase : public vr::ITrackedDeviceServerDriver {
public:
    DeviceBase(uint8_t id, std::string serial) : id_(id), serial_(std::move(serial)) {}
    // Devices are owned and destroyed through DeviceBase pointers. The virtual
    // destructor goes AFTER the OpenVR interface methods in the vtable, so the
    // layout SteamVR relies on is unchanged.
    virtual ~DeviceBase() = default;
    const std::string& serial() const { return serial_; }
    uint8_t id() const { return id_; }
    // Called from IServerTrackedDeviceProvider::RunFrame.
    virtual void Update(const SampleStore& store, Clock::time_point now);

    // ITrackedDeviceServerDriver
    vr::EVRInitError Activate(uint32_t object_id) override;
    void Deactivate() override { index_ = vr::k_unTrackedDeviceIndexInvalid; }
    void EnterStandby() override {}
    void* GetComponent(const char*) override { return nullptr; }
    void DebugRequest(const char*, char* response, uint32_t size) override {
        if (size) response[0] = 0;
    }
    vr::DriverPose_t GetPose() override { return last_pose_; }

protected:
    virtual void SetupProperties(vr::PropertyContainerHandle_t container) = 0;
    virtual void OnSample(const DeviceSample&) {}
    static vr::DriverPose_t MakePose(const DeviceSample& s, bool valid);

    uint8_t id_;
    std::string serial_;
    uint32_t index_ = vr::k_unTrackedDeviceIndexInvalid;
    vr::DriverPose_t last_pose_{};
};

class TrackerDevice final : public DeviceBase {
public:
    using DeviceBase::DeviceBase;

protected:
    void SetupProperties(vr::PropertyContainerHandle_t c) override;
};

class HandDevice final : public DeviceBase {
public:
    HandDevice(uint8_t id, std::string serial, bool left);
    ~HandDevice();

protected:
    void SetupProperties(vr::PropertyContainerHandle_t c) override;
    void OnSample(const DeviceSample& s) override;

private:
    enum {
        kTriggerValue, kTriggerClick, kTriggerTouch, kGripValue, kGripForce, kGripTouch,
        kFingerIndex, kFingerMiddle, kFingerRing, kFingerPinky, kSkeleton,
        kAClick, kATouch, kBClick, kBTouch, kSystemClick, kSystemTouch,
        kStickClick, kStickTouch, kStickX, kStickY, kHaptic, kCount
    };
    bool left_;
    vr::VRInputComponentHandle_t h_[kCount] = {};
    std::unique_ptr<MyHandSimulation> sim_;
};

struct DisplayConfig {
    int32_t window_x = 0, window_y = 0;
    uint32_t window_width = 1600, window_height = 900;
    uint32_t render_width = 800, render_height = 900;  // per eye = half the window
    float vertical_fov_deg = 90.0f;
};

class DisplayComponent final : public vr::IVRDisplayComponent {
public:
    explicit DisplayComponent(const DisplayConfig& c) : c_(c) {}
    void GetWindowBounds(int32_t* x, int32_t* y, uint32_t* w, uint32_t* h) override;
    bool IsDisplayOnDesktop() override { return true; }
    bool IsDisplayRealDisplay() override { return false; }
    void GetRecommendedRenderTargetSize(uint32_t* w, uint32_t* h) override;
    void GetEyeOutputViewport(vr::EVREye eye, uint32_t* x, uint32_t* y, uint32_t* w, uint32_t* h) override;
    void GetProjectionRaw(vr::EVREye eye, float* l, float* r, float* t, float* b) override;
    vr::DistortionCoordinates_t ComputeDistortion(vr::EVREye eye, float u, float v) override;
    bool ComputeInverseDistortion(vr::HmdVector2_t*, vr::EVREye, uint32_t, float, float) override {
        return false;
    }

private:
    DisplayConfig c_;
};

class HmdDevice final : public DeviceBase {
public:
    HmdDevice(uint8_t id, std::string serial, const DisplayConfig& cfg);
    void* GetComponent(const char* name) override;
    void Update(const SampleStore& store, Clock::time_point now) override;

protected:
    void SetupProperties(vr::PropertyContainerHandle_t c) override;

private:
    std::unique_ptr<DisplayComponent> display_;
};

}  // namespace nexovarapose
