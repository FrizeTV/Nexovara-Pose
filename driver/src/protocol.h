// Nexovara Pose driver <-> app protocol. Header-only, no OpenVR dependency (unit-testable).
//
// One UDP datagram per frame, sent by the Nexovara Pose app to 127.0.0.1:<port>.
// Little-endian.
//
//   header (16 bytes)
//     char[4]  magic   = "CFBT"
//     uint16   version = 1
//     uint16   flags   = 0 (reserved)
//     uint32   seq       (wraps)
//     uint8    count     number of device records (<= 32)
//     uint8[3] reserved
//   record (52 bytes) x count
//     uint8    id        0..7 body trackers (hip, left_foot, right_foot, left_knee,
//                        right_knee, chest, left_elbow, right_elbow), 16 HMD,
//                        17 left hand, 18 right hand
//     uint8    flags     bit0 valid, bit1 curls present, bit2 buttons present,
//                        bit3 velocity present (trackers / HMD only, not with bit1):
//                        then curls[0..2] carry the linear velocity, m/s (|v| <= 20)
//     uint16   buttons   hands only (bit2 set): bit0 A, bit1 B, bit2 system,
//                        bit3 trigger, bit4 grip, bit5 thumbstick click
//                        (keyboard/mouse controls of the app for playing
//                        without real controllers); otherwise 0
//     float[3] position  metres, OpenVR standing space (right-handed, +Y up)
//     float[4] rotation  quaternion x, y, z, w
//     float[5] curls     thumb, index, middle, ring, little (0 = open, 1 = fist)
//
// Parsing is strict: wrong size, magic, version, id, non-finite numbers or
// absurd values reject the whole packet (fail closed).
//
// Feedback, driver -> app (44 bytes), sent back to the address the app's packets
// come from (always 127.0.0.1), ~90 times per second while the app is sending:
//     char[4]  magic   = "CFBH"
//     uint16   version = 1
//     uint16   flags     bit0 headset pose valid, bit1 the headset is Nexovara Pose's
//                        own virtual one ("without a headset" mode)
//     uint32   seq
//     uint32   reserved
//     float[3] headset position, SteamVR raw space (where drivers report poses)
//     float[4] headset rotation x, y, z, w
// The app uses it to glue the body to the REAL headset (no extra packages).
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace nexovarapose {

constexpr uint16_t kProtocolVersion = 1;
constexpr size_t kHeaderSize = 16;
constexpr size_t kRecordSize = 52;
constexpr size_t kMaxRecords = 32;
constexpr size_t kMaxPacket = kHeaderSize + kMaxRecords * kRecordSize;
constexpr float kMaxPosition = 100.0f;  // metres

enum DeviceId : uint8_t {
    kTrackerFirst = 0,
    kTrackerCount = 8,
    kHmd = 16,
    kLeftHand = 17,
    kRightHand = 18,
};

enum RecordFlags : uint8_t { kFlagValid = 1, kFlagCurls = 2, kFlagButtons = 4, kFlagVelocity = 8 };
constexpr float kMaxVelocity = 20.0f;  // m/s

enum Buttons : uint16_t {
    kBtnA = 1 << 0,
    kBtnB = 1 << 1,
    kBtnSystem = 1 << 2,
    kBtnTrigger = 1 << 3,
    kBtnGrip = 1 << 4,
    kBtnStick = 1 << 5,
    kBtnAll = 0x3F,
};

struct DeviceSample {
    uint8_t id = 0;
    uint8_t flags = 0;
    uint16_t buttons = 0;  // valid only with kFlagButtons
    float pos[3] = {0, 0, 0};
    float rot[4] = {0, 0, 0, 1};  // x y z w
    float curls[5] = {0, 0, 0, 0, 0};
    float vel[3] = {0, 0, 0};     // valid only with kFlagVelocity
};

struct Packet {
    uint32_t seq = 0;
    std::vector<DeviceSample> devices;
};

constexpr size_t kFeedbackSize = 44;
enum FeedbackFlags : uint16_t { kFbHmdValid = 1, kFbHmdVirtual = 2 };

namespace detail {
inline void PutU16(uint8_t* p, uint16_t v) { p[0] = uint8_t(v), p[1] = uint8_t(v >> 8); }
inline void PutU32(uint8_t* p, uint32_t v) {
    for (int i = 0; i < 4; ++i) p[i] = uint8_t(v >> (8 * i));
}
inline void PutF32(uint8_t* p, float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof u);
    PutU32(p, u);
}
}  // namespace detail

// Fills out[kFeedbackSize].
inline void EncodeFeedback(uint32_t seq, uint16_t flags, const float pos[3], const float rot[4],
                           uint8_t* out) {
    std::memcpy(out, "CFBH", 4);
    detail::PutU16(out + 4, 1);
    detail::PutU16(out + 6, flags);
    detail::PutU32(out + 8, seq);
    detail::PutU32(out + 12, 0);
    for (int k = 0; k < 3; ++k) detail::PutF32(out + 16 + 4 * k, pos[k]);
    for (int k = 0; k < 4; ++k) detail::PutF32(out + 28 + 4 * k, rot[k]);
}

// Rotation part of a row-major 3x4 pose matrix -> quaternion (x, y, z, w).
inline void MatrixToQuat(const float m[3][4], float q[4]) {
    const double tr = double(m[0][0]) + m[1][1] + m[2][2];
    double x, y, z, w;
    if (tr > 0) {
        const double s = std::sqrt(tr + 1.0) * 2;
        w = 0.25 * s;
        x = (m[2][1] - m[1][2]) / s;
        y = (m[0][2] - m[2][0]) / s;
        z = (m[1][0] - m[0][1]) / s;
    } else if (m[0][0] > m[1][1] && m[0][0] > m[2][2]) {
        const double s = std::sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2;
        w = (m[2][1] - m[1][2]) / s;
        x = 0.25 * s;
        y = (m[0][1] + m[1][0]) / s;
        z = (m[0][2] + m[2][0]) / s;
    } else if (m[1][1] > m[2][2]) {
        const double s = std::sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2;
        w = (m[0][2] - m[2][0]) / s;
        x = (m[0][1] + m[1][0]) / s;
        y = 0.25 * s;
        z = (m[1][2] + m[2][1]) / s;
    } else {
        const double s = std::sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2;
        w = (m[1][0] - m[0][1]) / s;
        x = (m[0][2] + m[2][0]) / s;
        y = (m[1][2] + m[2][1]) / s;
        z = 0.25 * s;
    }
    q[0] = float(x), q[1] = float(y), q[2] = float(z), q[3] = float(w);
}

inline bool IsKnownDevice(uint8_t id) {
    return id < kTrackerCount || id == kHmd || id == kLeftHand || id == kRightHand;
}

namespace detail {
inline uint16_t U16(const uint8_t* p) { return uint16_t(p[0] | (p[1] << 8)); }
inline uint32_t U32(const uint8_t* p) {
    return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
}
inline float F32(const uint8_t* p) {
    uint32_t u = U32(p);
    float f;
    std::memcpy(&f, &u, sizeof f);
    return f;
}
}  // namespace detail

// Returns true and fills `out` only for a fully valid packet.
inline bool ParsePacket(const uint8_t* data, size_t len, Packet& out) {
    using namespace detail;
    if (data == nullptr || len < kHeaderSize || len > kMaxPacket) return false;
    if (std::memcmp(data, "CFBT", 4) != 0) return false;
    if (U16(data + 4) != kProtocolVersion) return false;
    const size_t count = data[12];
    if (count > kMaxRecords || len != kHeaderSize + count * kRecordSize) return false;

    Packet pkt;
    pkt.seq = U32(data + 8);
    pkt.devices.reserve(count);
    bool seen[256] = {false};
    for (size_t i = 0; i < count; ++i) {
        const uint8_t* r = data + kHeaderSize + i * kRecordSize;
        DeviceSample s;
        s.id = r[0];
        s.flags = r[1];
        if (!IsKnownDevice(s.id) || seen[s.id]) return false;
        seen[s.id] = true;
        // older apps send 0 here; unknown bits are ignored, not an error
        s.buttons = (s.flags & kFlagButtons) ? uint16_t(U16(r + 2) & kBtnAll) : uint16_t(0);
        for (int k = 0; k < 3; ++k) s.pos[k] = F32(r + 4 + 4 * k);
        for (int k = 0; k < 4; ++k) s.rot[k] = F32(r + 16 + 4 * k);
        for (int k = 0; k < 5; ++k) s.curls[k] = F32(r + 32 + 4 * k);
        for (float v : s.pos)
            if (!std::isfinite(v) || std::fabs(v) > kMaxPosition) return false;
        double n2 = 0;
        for (float v : s.rot) {
            if (!std::isfinite(v)) return false;
            n2 += double(v) * v;
        }
        const double n = std::sqrt(n2);
        if (n < 0.5 || n > 1.5) return false;  // not a rotation
        for (float& v : s.rot) v = float(v / n);
        for (float v : s.curls)
            if (!std::isfinite(v)) return false;
        if (s.flags & kFlagVelocity) {
            // the app's own (smooth) velocity of this device, in the curls' place
            if ((s.flags & kFlagCurls) || s.id == kLeftHand || s.id == kRightHand) return false;
            for (int k = 0; k < 3; ++k) {
                if (std::fabs(s.curls[k]) > kMaxVelocity) return false;
                s.vel[k] = s.curls[k];
                s.curls[k] = 0.f;
            }
        }
        for (float& v : s.curls) v = v < 0.f ? 0.f : (v > 1.f ? 1.f : v);
        pkt.devices.push_back(s);
    }
    out = std::move(pkt);
    return true;
}

}  // namespace nexovarapose
