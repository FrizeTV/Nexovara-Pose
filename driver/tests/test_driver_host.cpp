// End-to-end test of the built driver WITHOUT SteamVR: a mock vrserver loads
// driver_nexovarapose (.so/.dll), sends it a Nexovara Pose packet over UDP and checks that
// devices are created with the right classes, poses arrive, inputs and the
// hand skeleton are updated, stale data is reported, and Cleanup works.
//   nexovarapose_driver_host_test <path to driver_nexovarapose.so|.dll>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <thread>
#include <vector>

#include "openvr_driver.h"

#ifdef _WIN32
#include <winsock2.h>
#include <windows.h>
#else
#include <arpa/inet.h>
#include <dlfcn.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

static int g_failed = 0;
#define CHECK(c)                                                     \
    do {                                                             \
        if (!(c)) {                                                  \
            std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); \
            ++g_failed;                                              \
        }                                                            \
    } while (0)

// ------------------------------------------------------------- mock vrserver
struct Settings : vr::IVRSettings {
    std::map<std::string, int32_t> ints;
    std::map<std::string, bool> bools;
    static std::string K(const char* s, const char* k) { return std::string(s) + "." + k; }
    const char* GetSettingsErrorNameFromEnum(vr::EVRSettingsError) override { return ""; }
    void SetBool(const char*, const char*, bool, vr::EVRSettingsError*) override {}
    void SetInt32(const char*, const char*, int32_t, vr::EVRSettingsError*) override {}
    void SetFloat(const char*, const char*, float, vr::EVRSettingsError*) override {}
    void SetString(const char*, const char*, const char*, vr::EVRSettingsError*) override {}
    bool GetBool(const char* s, const char* k, vr::EVRSettingsError* e) override {
        auto it = bools.find(K(s, k));
        if (e) *e = it == bools.end() ? vr::VRSettingsError_ReadFailed : vr::VRSettingsError_None;
        return it != bools.end() && it->second;
    }
    int32_t GetInt32(const char* s, const char* k, vr::EVRSettingsError* e) override {
        auto it = ints.find(K(s, k));
        if (e) *e = it == ints.end() ? vr::VRSettingsError_ReadFailed : vr::VRSettingsError_None;
        return it == ints.end() ? 0 : it->second;
    }
    float GetFloat(const char*, const char*, vr::EVRSettingsError* e) override {
        if (e) *e = vr::VRSettingsError_None;
        return 0.064f;
    }
    void GetString(const char*, const char*, char* v, uint32_t n, vr::EVRSettingsError*) override {
        if (n) v[0] = 0;
    }
    void RemoveSection(const char*, vr::EVRSettingsError*) override {}
    void RemoveKeyInSection(const char*, const char*, vr::EVRSettingsError*) override {}
};

struct Properties : vr::IVRProperties {
    std::map<uint64_t, std::map<int, std::string>> strings;
    std::map<uint64_t, std::map<int, int32_t>> ints;
    vr::ETrackedPropertyError ReadPropertyBatch(vr::PropertyContainerHandle_t, vr::PropertyRead_t*, uint32_t) override {
        return vr::TrackedProp_Success;
    }
    vr::ETrackedPropertyError WritePropertyBatch(vr::PropertyContainerHandle_t c, vr::PropertyWrite_t* b,
                                                 uint32_t n) override {
        for (uint32_t i = 0; i < n; ++i) {
            if (b[i].writeType != vr::PropertyWrite_Set) continue;
            if (b[i].unTag == vr::k_unStringPropertyTag)
                strings[c][b[i].prop] = std::string(static_cast<const char*>(b[i].pvBuffer));
            else if (b[i].unTag == vr::k_unInt32PropertyTag)
                ints[c][b[i].prop] = *static_cast<const int32_t*>(b[i].pvBuffer);
            b[i].eError = vr::TrackedProp_Success;
        }
        return vr::TrackedProp_Success;
    }
    const char* GetPropErrorNameFromEnum(vr::ETrackedPropertyError) override { return ""; }
    vr::PropertyContainerHandle_t TrackedDeviceToPropertyContainer(vr::TrackedDeviceIndex_t d) override {
        return 1000 + d;
    }
};

struct LogSink : vr::IVRDriverLog {
    std::string all;
    void Log(const char* m) override {
        std::printf("  [driver] %s\n", m);
        all += m;
        all += "\n";
    }
};

struct Device {
    std::string serial;
    vr::ETrackedDeviceClass cls;
    vr::ITrackedDeviceServerDriver* drv;
    vr::DriverPose_t pose{};
    int pose_updates = 0;
};

struct Host : vr::IVRServerDriverHost {
    std::vector<Device> devices;
    bool TrackedDeviceAdded(const char* s, vr::ETrackedDeviceClass c, vr::ITrackedDeviceServerDriver* d) override {
        devices.push_back({s, c, d});
        d->Activate(uint32_t(devices.size() - 1));  // vrserver activates soon after
        return true;
    }
    void TrackedDevicePoseUpdated(uint32_t i, const vr::DriverPose_t& p, uint32_t sz) override {
        CHECK(sz == sizeof(vr::DriverPose_t));
        if (i < devices.size()) {
            devices[i].pose = p;
            devices[i].pose_updates++;
        }
    }
    void VsyncEvent(double) override {}
    void VendorSpecificEvent(uint32_t, vr::EVREventType, const vr::VREvent_Data_t&, double) override {}
    bool IsExiting() override { return false; }
    bool PollNextEvent(vr::VREvent_t*, uint32_t) override { return false; }
    void GetRawTrackedDevicePoses(float, vr::TrackedDevicePose_t* p, uint32_t n) override {
        if (n < 1) return;
        p[0] = {};
        p[0].bPoseIsValid = true;   // the "real" headset: 1.75 m high, looking along -Z
        p[0].mDeviceToAbsoluteTracking.m[0][0] = p[0].mDeviceToAbsoluteTracking.m[1][1] =
            p[0].mDeviceToAbsoluteTracking.m[2][2] = 1.0f;
        p[0].mDeviceToAbsoluteTracking.m[0][3] = 0.2f;
        p[0].mDeviceToAbsoluteTracking.m[1][3] = 1.75f;
        p[0].mDeviceToAbsoluteTracking.m[2][3] = -0.4f;
    }
    void RequestRestart(const char*, const char*, const char*, const char*) override {}
    uint32_t GetFrameTimings(vr::Compositor_FrameTiming*, uint32_t) override { return 0; }
    void SetDisplayEyeToHead(uint32_t, const vr::HmdMatrix34_t&, const vr::HmdMatrix34_t&) override {}
    void SetDisplayProjectionRaw(uint32_t, const vr::HmdRect2_t&, const vr::HmdRect2_t&) override {}
    void SetRecommendedRenderTargetSize(uint32_t, uint32_t, uint32_t) override {}
};

struct Input : vr::IVRDriverInput {
    uint64_t next = 1;
    std::map<std::string, uint64_t> by_name;
    std::map<uint64_t, float> scalars;
    std::map<uint64_t, bool> bools;
    int skeleton_updates = 0;
    vr::EVRInputError CreateBooleanComponent(vr::PropertyContainerHandle_t c, const char* n,
                                             vr::VRInputComponentHandle_t* h) override {
        *h = by_name[std::to_string(c) + n] = next++;
        return vr::VRInputError_None;
    }
    vr::EVRInputError UpdateBooleanComponent(vr::VRInputComponentHandle_t h, bool v, double) override {
        bools[h] = v;
        return vr::VRInputError_None;
    }
    vr::EVRInputError CreateScalarComponent(vr::PropertyContainerHandle_t c, const char* n,
                                            vr::VRInputComponentHandle_t* h, vr::EVRScalarType,
                                            vr::EVRScalarUnits) override {
        *h = by_name[std::to_string(c) + n] = next++;
        return vr::VRInputError_None;
    }
    vr::EVRInputError UpdateScalarComponent(vr::VRInputComponentHandle_t h, float v, double) override {
        scalars[h] = v;
        return vr::VRInputError_None;
    }
    vr::EVRInputError CreateHapticComponent(vr::PropertyContainerHandle_t, const char*,
                                            vr::VRInputComponentHandle_t* h) override {
        *h = next++;
        return vr::VRInputError_None;
    }
    vr::EVRInputError CreateSkeletonComponent(vr::PropertyContainerHandle_t c, const char* n, const char*, const char*,
                                              vr::EVRSkeletalTrackingLevel, const vr::VRBoneTransform_t*, uint32_t,
                                              vr::VRInputComponentHandle_t* h) override {
        *h = by_name[std::to_string(c) + n] = next++;
        return vr::VRInputError_None;
    }
    vr::EVRInputError UpdateSkeletonComponent(vr::VRInputComponentHandle_t, vr::EVRSkeletalMotionRange,
                                              const vr::VRBoneTransform_t* t, uint32_t n) override {
        CHECK(n == 31);
        for (uint32_t i = 0; i < n; ++i) CHECK(std::isfinite(t[i].orientation.w));
        skeleton_updates++;
        return vr::VRInputError_None;
    }
    vr::EVRInputError CreatePoseComponent(vr::PropertyContainerHandle_t, const char*,
                                          vr::VRInputComponentHandle_t* h) override {
        *h = next++;
        return vr::VRInputError_None;
    }
    vr::EVRInputError UpdatePoseComponent(vr::VRInputComponentHandle_t, const vr::HmdMatrix34_t*, double) override {
        return vr::VRInputError_None;
    }
    vr::EVRInputError CreateEyeTrackingComponent(vr::PropertyContainerHandle_t, const char*,
                                                 vr::VRInputComponentHandle_t* h) override {
        *h = next++;
        return vr::VRInputError_None;
    }
    vr::EVRInputError UpdateEyeTrackingComponent(vr::VRInputComponentHandle_t, const vr::VREyeTrackingData_t*,
                                                 double) override {
        return vr::VRInputError_None;
    }
};

struct Dummy {
    char pad[64] = {};
};

struct Context : vr::IVRDriverContext {
    Settings settings;
    Properties props;
    LogSink log;
    Host host;
    Input input;
    Dummy manager, resources;
    void* GetGenericInterface(const char* v, vr::EVRInitError* e) override {
        if (e) *e = vr::VRInitError_None;
        if (!std::strcmp(v, vr::IVRSettings_Version)) return static_cast<vr::IVRSettings*>(&settings);
        if (!std::strcmp(v, vr::IVRProperties_Version)) return static_cast<vr::IVRProperties*>(&props);
        if (!std::strcmp(v, vr::IVRDriverLog_Version)) return static_cast<vr::IVRDriverLog*>(&log);
        if (!std::strcmp(v, vr::IVRServerDriverHost_Version)) return static_cast<vr::IVRServerDriverHost*>(&host);
        if (!std::strcmp(v, vr::IVRDriverInput_Version)) return static_cast<vr::IVRDriverInput*>(&input);
        if (!std::strcmp(v, vr::IVRDriverManager_Version)) return &manager;   // never called
        if (!std::strcmp(v, vr::IVRResources_Version)) return &resources;    // never called
        if (e) *e = vr::VRInitError_Init_InterfaceNotFound;
        return nullptr;
    }
    vr::DriverHandle_t GetDriverHandle() override { return 1; }
};

// ------------------------------------------------------------- packet helpers
static void PutF(std::vector<uint8_t>& b, float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    for (int i = 0; i < 4; ++i) b.push_back(uint8_t(u >> (8 * i)));
}

static std::vector<uint8_t> Packet(const std::vector<std::pair<uint8_t, float>>& devs, float curl,
                                   uint16_t buttons = 0) {
    std::vector<uint8_t> b = {'C', 'F', 'B', 'T', 1, 0, 0, 0, 1, 0, 0, 0, uint8_t(devs.size()), 0, 0, 0};
    for (auto& d : devs) {
        b.push_back(d.first);
        b.push_back(buttons ? 7 : 3);  // valid + curls (+ buttons)
        b.push_back(uint8_t(buttons)), b.push_back(uint8_t(buttons >> 8));
        PutF(b, 0.1f), PutF(b, d.second), PutF(b, -0.3f);   // position
        PutF(b, 0), PutF(b, 0), PutF(b, 0), PutF(b, 1);       // rotation
        for (int k = 0; k < 5; ++k) PutF(b, curl);           // curls
    }
    return b;
}

// one tracker with the app's velocity (flag bit3, velocity in the curls' place)
static std::vector<uint8_t> VelPacket(uint8_t id, float y, float vy) {
    std::vector<uint8_t> b = {'C', 'F', 'B', 'T', 1, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0};
    b.push_back(id);
    b.push_back(1 | 8);
    b.push_back(0), b.push_back(0);
    PutF(b, 0.1f), PutF(b, y), PutF(b, -0.3f);
    PutF(b, 0), PutF(b, 0), PutF(b, 0), PutF(b, 1);
    PutF(b, 0.0f), PutF(b, vy), PutF(b, 0.0f), PutF(b, 0.0f), PutF(b, 0.0f);
    return b;
}

#ifdef _WIN32
static SOCKET g_app = INVALID_SOCKET;  // the "Nexovara Pose app" (created in main, after WSAStartup)
#else
static int g_app = -1;
#endif

static void Send(const std::vector<uint8_t>& b, uint16_t port) {
    auto s = g_app;
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_port = htons(port);
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    sendto(s, reinterpret_cast<const char*>(b.data()), int(b.size()), 0, reinterpret_cast<sockaddr*>(&a), sizeof a);
}

// newest feedback datagram from the driver (empty if none)
static std::vector<uint8_t> ReadFeedback() {
    std::vector<uint8_t> last;
    uint8_t buf[256];
    for (;;) {
#ifdef _WIN32
        u_long nb = 1;
        ioctlsocket(g_app, FIONBIO, &nb);
        const int n = recv(g_app, reinterpret_cast<char*>(buf), sizeof buf, 0);
#else
        const int n = static_cast<int>(recv(g_app, buf, sizeof buf, MSG_DONTWAIT));
#endif
        if (n <= 0) break;
        last.assign(buf, buf + n);
    }
    return last;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("usage: %s <driver library>\n", argv[0]);
        return 2;
    }
#ifdef _WIN32
    WSADATA wsa;
    WSAStartup(MAKEWORD(2, 2), &wsa);
    HMODULE lib = LoadLibraryA(argv[1]);
    auto factory = lib ? reinterpret_cast<void* (*)(const char*, int*)>(GetProcAddress(lib, "HmdDriverFactory")) : nullptr;
#else
    void* lib = dlopen(argv[1], RTLD_NOW);
    auto factory = lib ? reinterpret_cast<void* (*)(const char*, int*)>(dlsym(lib, "HmdDriverFactory")) : nullptr;
#endif
    CHECK(factory != nullptr);
    if (!factory) return 1;
    int rc = 0;
    CHECK(factory("IVRFoo_999", &rc) == nullptr && rc == vr::VRInitError_Init_InterfaceNotFound);
    auto* provider = static_cast<vr::IServerTrackedDeviceProvider*>(factory(vr::IServerTrackedDeviceProvider_Version, &rc));
    CHECK(provider != nullptr);

    {   // the "app" socket lives on 127.0.0.1 (feedback is sent back to it)
        g_app = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
        sockaddr_in me{};
        me.sin_family = AF_INET;
        me.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        bind(g_app, reinterpret_cast<sockaddr*>(&me), sizeof me);
    }
    Context ctx;
    const uint16_t port = 39877;
    ctx.settings.ints["driver_nexovarapose.port"] = port;
    ctx.settings.bools["driver_nexovarapose.enableHmd"] = true;
    CHECK(provider->Init(&ctx) == vr::VRInitError_None);
    CHECK(ctx.host.devices.size() == 1 && ctx.host.devices[0].cls == vr::TrackedDeviceClass_HMD);
    void* display = ctx.host.devices[0].drv->GetComponent(vr::IVRDisplayComponent_Version);
    CHECK(display != nullptr);
    if (display) {
        int32_t x, y;
        uint32_t w, h;
        auto* dc = static_cast<vr::IVRDisplayComponent*>(display);
        dc->GetWindowBounds(&x, &y, &w, &h);
        CHECK(w == 1600 && h == 900);
        dc->GetRecommendedRenderTargetSize(&w, &h);
        CHECK(w == 800 && h == 900);  // per eye = what the window shows
        float l, r, t, b;
        dc->GetProjectionRaw(vr::Eye_Left, &l, &r, &t, &b);
        CHECK(std::fabs(t + 1.0f) < 1e-4f && std::fabs(b - 1.0f) < 1e-4f);
        CHECK(std::fabs(r - 800.0f / 900.0f) < 1e-4f && std::fabs(l + r) < 1e-6f);
    }

    provider->RunFrame();
    CHECK(ctx.host.devices[0].pose.poseIsValid);  // HMD valid even before data
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    Send(Packet({{0, 0.95f}, {1, 0.1f}, {2, 0.1f}, {16, 1.62f}, {17, 1.0f}, {18, 1.0f}}, 0.9f), port);
    for (int i = 0; i < 50 && ctx.host.devices.size() < 6; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        provider->RunFrame();
    }
    provider->RunFrame();
    CHECK(ctx.host.devices.size() == 6);
    int trackers = 0, controllers = 0;
    for (auto& d : ctx.host.devices) {
        trackers += d.cls == vr::TrackedDeviceClass_GenericTracker;
        controllers += d.cls == vr::TrackedDeviceClass_Controller;
        CHECK(d.pose.poseIsValid && d.pose.result == vr::TrackingResult_Running_OK);
    }
    CHECK(trackers == 3 && controllers == 2);
    CHECK(std::fabs(ctx.host.devices[0].pose.vecPosition[1] - 1.62) < 1e-5);  // HMD moved
    bool hip_ok = false;
    for (auto& d : ctx.host.devices)
        if (d.serial == "Nexovara Pose-hip") hip_ok = std::fabs(d.pose.vecPosition[1] - 0.95) < 1e-5;
    CHECK(hip_ok);
    {   // the app's velocity reaches SteamVR (prediction + smooth motion between packets);
        // packets without it (older apps) give exactly 0, as before
        auto hip = [&]() -> Device* {
            for (auto& d : ctx.host.devices)
                if (d.serial == "Nexovara Pose-hip") return &d;
            return nullptr;
        };
        CHECK(hip() && hip()->pose.vecVelocity[1] == 0.0 && hip()->pose.poseTimeOffset == 0.0);
        Send(VelPacket(0, 0.97f, 0.8f), port);
        for (int i = 0; i < 20 && !(hip() && hip()->pose.vecVelocity[1] != 0.0); ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            provider->RunFrame();
        }
        CHECK(hip() && std::fabs(hip()->pose.vecVelocity[1] - 0.8) < 1e-6);
        CHECK(hip() && hip()->pose.poseTimeOffset <= 0.0 && hip()->pose.poseTimeOffset >= -0.05);
        Send(Packet({{0, 0.95f}}, 0.0f), port);
        for (int i = 0; i < 20 && !(hip() && hip()->pose.vecVelocity[1] == 0.0); ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            provider->RunFrame();
        }
        CHECK(hip() && hip()->pose.vecVelocity[1] == 0.0 && hip()->pose.poseTimeOffset == 0.0);
    }
    CHECK(ctx.input.skeleton_updates > 0);
    uint64_t trig = 0;
    for (auto& kv : ctx.input.by_name)
        if (kv.first.find("/input/trigger/value") != std::string::npos) trig = kv.second;
    CHECK(trig && std::fabs(ctx.input.scalars[trig] - 0.9f) < 1e-6f);
    CHECK(ctx.log.all.find("receiving data from the Nexovara Pose app") != std::string::npos);
    {   // the driver answered with the headset pose (raw space)
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        provider->RunFrame();
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        auto fb = ReadFeedback();
        CHECK(fb.size() == 44 && fb.size() >= 4 && std::memcmp(fb.data(), "CFBH", 4) == 0);
        if (fb.size() == 44) {
            float y, qw;
            std::memcpy(&y, fb.data() + 20, 4);
            std::memcpy(&qw, fb.data() + 40, 4);
            CHECK(std::fabs(y - 1.75f) < 1e-6f && std::fabs(qw - 1.0f) < 1e-6f);
            CHECK((fb[6] & 1) && (fb[6] & 2));   // valid, and the virtual HMD is ours here
        }
    }

    // keyboard buttons of the app: B (VRChat Quick Menu) + trigger, fingers open
    auto find_bool = [&](const char* name, bool want_right) {
        for (auto& kv : ctx.input.by_name)
            if (kv.first.find(name) != std::string::npos) {
                // container ids: right hand was added after the left one
                bool right = false;
                for (size_t i = 0; i < ctx.host.devices.size(); ++i)
                    if (ctx.host.devices[i].serial == "Nexovara Pose-right-hand" &&
                        kv.first.rfind(std::to_string(1000 + i), 0) == 0)
                        right = true;
                if (right == want_right) return kv.second;
            }
        return uint64_t(0);
    };
    Send(Packet({{18, 1.0f}}, 0.0f, 0x2 | 0x8), port);
    for (int i = 0; i < 20; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        provider->RunFrame();
    }
    const uint64_t b_click = find_bool("/input/b/click", true);
    const uint64_t a_click = find_bool("/input/a/click", true);
    const uint64_t trig_click = find_bool("/input/trigger/click", true);
    CHECK(b_click && ctx.input.bools[b_click]);
    CHECK(a_click && !ctx.input.bools[a_click]);
    CHECK(trig_click && ctx.input.bools[trig_click]);
    Send(Packet({{18, 1.0f}}, 0.0f), port);  // released
    for (int i = 0; i < 20; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        provider->RunFrame();
    }
    CHECK(!ctx.input.bools[b_click] && !ctx.input.bools[trig_click]);
    // the virtual HMD brings its own room (no "universe id is invalid" in SteamVR)
    {
        const auto& hp = ctx.props.strings[1000];
        auto it = hp.find(vr::Prop_DriverProvidedChaperoneJson_String);
        CHECK(it != hp.end() && it->second.find("\"universeID\": \"1128678996\"") != std::string::npos);
        CHECK(it != hp.end() && it->second.find("chaperone_info") != std::string::npos);
    }
    // tracker type properties (looks like a VIVE Tracker to games)
    bool tracker_model = false;
    for (auto& kv : ctx.props.strings)
        for (auto& p : kv.second)
            if (p.first == vr::Prop_ControllerType_String && p.second == "vive_tracker") tracker_model = true;
    CHECK(tracker_model);

    // stale data -> trackers report "out of range"
    std::this_thread::sleep_for(std::chrono::milliseconds(650));
    provider->RunFrame();
    for (auto& d : ctx.host.devices)
        if (d.cls == vr::TrackedDeviceClass_GenericTracker) CHECK(!d.pose.poseIsValid);
    // garbage never adds devices
    Send(std::vector<uint8_t>(40, 0x41), port);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    provider->RunFrame();
    CHECK(ctx.host.devices.size() == 6);

    for (auto& d : ctx.host.devices) d.drv->Deactivate();
    provider->Cleanup();
    if (g_failed) {
        std::printf("%d check(s) failed\n", g_failed);
        return 1;
    }
    CHECK(ctx.log.all.find("rejected") != std::string::npos);  // garbage is reported in the log
    std::printf("driver host test passed: HMD + 3 trackers + 2 hands, poses, inputs, buttons, skeleton, "
                "timeout, log\n");
    return 0;
}
