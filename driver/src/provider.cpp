// IServerTrackedDeviceProvider + HmdDriverFactory entry point of driver_nexovarapose.
//
// Devices appear lazily: a body tracker / hand is added to SteamVR the first
// time the app sends data for it, so unused trackers never clutter SteamVR.
// The virtual HMD exists only when "driver_nexovarapose.enableHmd" is true (the
// app's "SteamVR without a headset" mode sets it).
#include <array>
#include <cmath>
#include <cstring>
#include <memory>
#include <vector>

#include "devices.h"
#include "receiver.h"

#if defined(_WIN32)
#define NEXOVARAPOSE_EXPORT extern "C" __declspec(dllexport)
#else
#define NEXOVARAPOSE_EXPORT extern "C" __attribute__((visibility("default")))
#endif

namespace nexovarapose {

static const char* kSection = "driver_nexovarapose";
// Same order as the app's tracker list (fixed slots).
static const char* kTrackerNames[kTrackerCount] = {"hip",       "left_foot",  "right_foot", "left_knee",
                                                   "right_knee", "chest",     "left_elbow", "right_elbow"};

static int32_t SettingInt(const char* key, int32_t def) {
    vr::EVRSettingsError err = vr::VRSettingsError_None;
    int32_t v = vr::VRSettings()->GetInt32(kSection, key, &err);
    return err == vr::VRSettingsError_None ? v : def;
}

static bool SettingBool(const char* key, bool def) {
    vr::EVRSettingsError err = vr::VRSettingsError_None;
    bool v = vr::VRSettings()->GetBool(kSection, key, &err);
    return err == vr::VRSettingsError_None ? v : def;
}

class Provider final : public vr::IServerTrackedDeviceProvider {
public:
    vr::EVRInitError Init(vr::IVRDriverContext* ctx) override {
        VR_INIT_SERVER_DRIVER_CONTEXT(ctx);
        g_predict_velocity = SettingBool("predictVelocity", true);
        int32_t port = SettingInt("port", 39580);
        if (port < 1024 || port > 65535) port = 39580;
        receiver_ = std::make_unique<UdpReceiver>(store_, [](const std::string& m) { Log("nexovarapose: %s", m.c_str()); });
        if (!receiver_->Start(static_cast<uint16_t>(port))) {
            Log("nexovarapose: could not open UDP port %d", port);
        }
        if (SettingBool("enableHmd", false)) {
            DisplayConfig d;
            d.window_x = SettingInt("window_x", 0);
            d.window_y = SettingInt("window_y", 0);
            d.window_width = static_cast<uint32_t>(Clamp(SettingInt("window_width", 1600), 320, 7680));
            d.window_height = static_cast<uint32_t>(Clamp(SettingInt("window_height", 900), 240, 4320));
            // per-eye render size: 0 = automatic (exactly what the window shows: half
            // its width, full height) - rendering more is wasted work for the GPU
            const int32_t rw = SettingInt("render_width", 0), rh = SettingInt("render_height", 0);
            d.render_width = static_cast<uint32_t>(rw > 0 ? Clamp(rw, 160, 4096) : int32_t(d.window_width / 2));
            d.render_height = static_cast<uint32_t>(rh > 0 ? Clamp(rh, 120, 4096) : int32_t(d.window_height));
            d.vertical_fov_deg = static_cast<float>(Clamp(SettingInt("vertical_fov", 90), 40, 130));
            hmd_ = std::make_unique<HmdDevice>(kHmd, "Nexovara Pose-HMD", d);
            if (!vr::VRServerDriverHost()->TrackedDeviceAdded(hmd_->serial().c_str(), vr::TrackedDeviceClass_HMD,
                                                              hmd_.get())) {
                Log("nexovarapose: failed to add the virtual HMD");
            }
        }
        Log("nexovarapose: driver initialised (virtual HMD %s, protocol %d, velocity %s)",
            hmd_ ? "on" : "off", int(kProtocolVersion), g_predict_velocity ? "on" : "off");
        return vr::VRInitError_None;
    }

    void Cleanup() override {
        if (receiver_) receiver_->Stop();
        receiver_.reset();
        // SteamVR deactivates the devices before Cleanup; now they can go
        devices_.clear();
        hmd_.reset();
        VR_CLEANUP_SERVER_DRIVER_CONTEXT();
    }

    const char* const* GetInterfaceVersions() override { return vr::k_InterfaceVersions; }

    void RunFrame() override {
        const auto now = Clock::now();
        ReportDataFlow(now);
        SendHeadsetPose(now);
        AddNewDevices();
        if (hmd_) hmd_->Update(store_, now);
        for (auto& d : devices_) d->Update(store_, now);
        vr::VREvent_t ev{};
        while (vr::VRServerDriverHost()->PollNextEvent(&ev, sizeof ev)) {
        }
    }

    bool ShouldBlockStandbyMode() override { return false; }
    void EnterStandby() override {}
    void LeaveStandby() override {}

private:
    static int32_t Clamp(int32_t v, int32_t lo, int32_t hi) { return v < lo ? lo : (v > hi ? hi : v); }

    // The headset pose (SteamVR raw space) back to the app, so it can glue the
    // body to the REAL headset without any extra package on the app side.
    void SendHeadsetPose(Clock::time_point now) {
        if (!receiver_ || now - last_feedback_ < std::chrono::milliseconds(8)) return;
        last_feedback_ = now;
        vr::TrackedDevicePose_t poses[1] = {};
        vr::VRServerDriverHost()->GetRawTrackedDevicePoses(0.0f, poses, 1);
        const auto& m = poses[0].mDeviceToAbsoluteTracking.m;
        float pos[3] = {m[0][3], m[1][3], m[2][3]};
        float rot[4] = {0, 0, 0, 1};
        bool valid = poses[0].bPoseIsValid;
        if (valid) {
            MatrixToQuat(m, rot);
            for (float v : pos) valid = valid && std::isfinite(v);
            for (float v : rot) valid = valid && std::isfinite(v);
        }
        uint16_t flags = (valid ? kFbHmdValid : 0) | (hmd_ ? kFbHmdVirtual : 0);
        if (!valid) pos[0] = pos[1] = pos[2] = 0.0f, rot[0] = rot[1] = rot[2] = 0.0f, rot[3] = 1.0f;
        uint8_t buf[kFeedbackSize];
        EncodeFeedback(++feedback_seq_, flags, pos, rot, buf);
        receiver_->SendToApp(buf, sizeof buf, now);
    }

    // One line in SteamVR's log (vrserver.txt) when data starts / stops / is
    // rejected: the Nexovara Pose app reads these lines in its "SteamVR check".
    void ReportDataFlow(Clock::time_point now) {
        if (!receiver_) return;
        const uint64_t packets = store_.packets();
        const uint64_t rejected = receiver_->rejected();
        if (packets != last_packets_) {
            if (!receiving_) Log("nexovarapose: receiving data from the Nexovara Pose app");
            receiving_ = true;
            last_packets_ = packets;
            last_data_ = now;
        } else if (receiving_ && now - last_data_ > std::chrono::seconds(3)) {
            Log("nexovarapose: no data from the Nexovara Pose app for 3 s (tracking stopped?)");
            receiving_ = false;
        }
        if (rejected != last_rejected_ && now - last_reject_log_ > std::chrono::seconds(10)) {
            Log("nexovarapose: rejected %llu bad packet(s) (app and driver versions differ? rebuild the driver)",
                static_cast<unsigned long long>(rejected - last_rejected_));
            last_rejected_ = rejected;
            last_reject_log_ = now;
        }
    }

    void AddNewDevices() {
        for (uint8_t id = 0; id < kTrackerCount; ++id) {
            if (added_[id] || !store_.Seen(id)) continue;
            Add(std::make_unique<TrackerDevice>(id, std::string("Nexovara Pose-") + kTrackerNames[id]),
                vr::TrackedDeviceClass_GenericTracker);
        }
        for (uint8_t id : {uint8_t(kLeftHand), uint8_t(kRightHand)}) {
            if (added_[id] || !store_.Seen(id)) continue;
            const bool left = id == kLeftHand;
            Add(std::make_unique<HandDevice>(id, left ? "Nexovara Pose-left-hand" : "Nexovara Pose-right-hand", left),
                vr::TrackedDeviceClass_Controller);
        }
    }

    void Add(std::unique_ptr<DeviceBase> dev, vr::ETrackedDeviceClass cls) {
        added_[dev->id()] = true;  // never retry: one attempt per device per session
        if (vr::VRServerDriverHost()->TrackedDeviceAdded(dev->serial().c_str(), cls, dev.get())) {
            Log("nexovarapose: added %s", dev->serial().c_str());
            devices_.push_back(std::move(dev));
        } else {
            Log("nexovarapose: SteamVR refused %s", dev->serial().c_str());
            refused_.push_back(std::move(dev));  // keep alive: SteamVR may still hold the pointer
        }
    }

    SampleStore store_;
    std::unique_ptr<UdpReceiver> receiver_;
    std::unique_ptr<HmdDevice> hmd_;
    std::vector<std::unique_ptr<DeviceBase>> devices_;
    std::vector<std::unique_ptr<DeviceBase>> refused_;
    std::array<bool, 256> added_{};
    uint64_t last_packets_ = 0, last_rejected_ = 0;
    uint32_t feedback_seq_ = 0;
    Clock::time_point last_feedback_{};
    bool receiving_ = false;
    Clock::time_point last_data_{}, last_reject_log_{};
};

static Provider g_provider;

}  // namespace nexovarapose

NEXOVARAPOSE_EXPORT void* HmdDriverFactory(const char* interface_name, int* return_code) {
    if (std::strcmp(vr::IServerTrackedDeviceProvider_Version, interface_name) == 0) {
        return &nexovarapose::g_provider;
    }
    if (return_code) *return_code = vr::VRInitError_Init_InterfaceNotFound;
    return nullptr;
}
