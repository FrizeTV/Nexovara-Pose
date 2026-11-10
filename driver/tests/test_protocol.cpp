// Unit tests for the Nexovara Pose driver protocol + UDP receiver (no SteamVR needed).
//   nexovarapose_protocol_test                -> runs all tests
//   nexovarapose_protocol_test --parse FILE   -> parses a packet file, prints devices (used by
//                                          the Python test suite for a cross-language check)
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <thread>
#include <vector>

#include "protocol.h"
#include "receiver.h"

#ifdef _WIN32
#include <winsock2.h>
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

using namespace nexovarapose;

static int g_failed = 0;
#define CHECK(cond)                                                     \
    do {                                                                \
        if (!(cond)) {                                                  \
            std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); \
            ++g_failed;                                                 \
        }                                                               \
    } while (0)

static void PutU16(std::vector<uint8_t>& b, uint16_t v) {
    b.push_back(uint8_t(v));
    b.push_back(uint8_t(v >> 8));
}
static void PutU32(std::vector<uint8_t>& b, uint32_t v) {
    for (int i = 0; i < 4; ++i) b.push_back(uint8_t(v >> (8 * i)));
}
static void PutF(std::vector<uint8_t>& b, float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    PutU32(b, u);
}

static std::vector<uint8_t> MakePacket(const std::vector<DeviceSample>& devs, uint32_t seq = 7,
                                       int count_override = -1) {
    std::vector<uint8_t> b = {'C', 'F', 'B', 'T'};
    PutU16(b, kProtocolVersion);
    PutU16(b, 0);
    PutU32(b, seq);
    b.push_back(uint8_t(count_override >= 0 ? count_override : int(devs.size())));
    b.push_back(0), b.push_back(0), b.push_back(0);
    for (const auto& d : devs) {
        b.push_back(d.id);
        b.push_back(d.flags);
        PutU16(b, d.buttons);
        for (float v : d.pos) PutF(b, v);
        for (float v : d.rot) PutF(b, v);
        for (float v : d.curls) PutF(b, v);
    }
    return b;
}

static DeviceSample Dev(uint8_t id, float y = 1.0f) {
    DeviceSample d;
    d.id = id;
    d.flags = kFlagValid;
    d.pos[0] = 0.1f, d.pos[1] = y, d.pos[2] = -0.2f;
    d.rot[0] = 0, d.rot[1] = 0, d.rot[2] = 0, d.rot[3] = 1.2f;  // not unit: must be normalised
    return d;
}

static void TestValid() {
    DeviceSample h = Dev(kLeftHand);
    h.flags |= kFlagCurls;
    h.curls[1] = 0.7f;
    h.curls[4] = 3.0f;  // clamped to 1
    auto b = MakePacket({Dev(0), Dev(2, 0.1f), Dev(kHmd, 1.6f), h});
    CHECK(b.size() == kHeaderSize + 4 * kRecordSize);
    Packet p;
    const bool ok = ParsePacket(b.data(), b.size(), p);
    CHECK(ok);
    if (!ok || p.devices.size() != 4) {
        ++g_failed;
        return;
    }
    CHECK(p.seq == 7);
    CHECK(std::fabs(p.devices[0].rot[3] - 1.0f) < 1e-6f);  // normalised
    CHECK(std::fabs(p.devices[1].pos[1] - 0.1f) < 1e-6f);
    CHECK(p.devices[3].curls[1] == 0.7f && p.devices[3].curls[4] == 1.0f);
}

static void TestRejects() {
    Packet p;
    auto good = MakePacket({Dev(0)});
    auto b = good;
    b[0] = 'X';
    CHECK(!ParsePacket(b.data(), b.size(), p));  // magic
    b = good;
    b[4] = 9;
    CHECK(!ParsePacket(b.data(), b.size(), p));  // version
    b = good;
    b.pop_back();
    CHECK(!ParsePacket(b.data(), b.size(), p));  // truncated
    b = MakePacket({Dev(0)}, 1, 2);
    CHECK(!ParsePacket(b.data(), b.size(), p));  // count does not match size
    b = MakePacket({Dev(9)});
    CHECK(!ParsePacket(b.data(), b.size(), p));  // unknown device id
    b = MakePacket({Dev(0), Dev(0)});
    CHECK(!ParsePacket(b.data(), b.size(), p));  // duplicate id
    DeviceSample nan = Dev(1);
    nan.pos[0] = std::nanf("");
    b = MakePacket({nan});
    CHECK(!ParsePacket(b.data(), b.size(), p));  // NaN
    DeviceSample distant = Dev(1);
    distant.pos[2] = 1e6f;
    b = MakePacket({distant});
    CHECK(!ParsePacket(b.data(), b.size(), p));  // absurd position
    DeviceSample zero = Dev(1);
    zero.rot[3] = 0;
    b = MakePacket({zero});
    CHECK(!ParsePacket(b.data(), b.size(), p));  // not a rotation
    CHECK(!ParsePacket(nullptr, 100, p));
    std::vector<uint8_t> huge(kMaxPacket + 1, 0);
    CHECK(!ParsePacket(huge.data(), huge.size(), p));
    // random garbage never crashes and is rejected
    uint32_t x = 12345;
    for (int i = 0; i < 20000; ++i) {
        std::vector<uint8_t> r(size_t(x % 200));
        for (auto& c : r) c = uint8_t((x = x * 1664525u + 1013904223u) >> 24);
        if (r.size() >= 4) std::memcpy(r.data(), "CFBT", 4);
        ParsePacket(r.data(), r.size(), p);
    }
}

static void TestReceiverLoopback() {
#ifdef _WIN32
    WSADATA wsa;
    WSAStartup(MAKEWORD(2, 2), &wsa);
#endif
    SampleStore store;
    UdpReceiver rx(store, [](const std::string&) {});
    const uint16_t port = 39871;
    CHECK(rx.Start(port));
    auto pkt = MakePacket({Dev(0), Dev(kRightHand, 1.2f)});
    auto s = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_port = htons(port);
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    std::vector<uint8_t> junk = {1, 2, 3};
    sendto(s, reinterpret_cast<const char*>(junk.data()), int(junk.size()), 0, reinterpret_cast<sockaddr*>(&a), sizeof a);
    sendto(s, reinterpret_cast<const char*>(pkt.data()), int(pkt.size()), 0, reinterpret_cast<sockaddr*>(&a), sizeof a);
    for (int i = 0; i < 100 && store.packets() == 0; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(10));
    DeviceSample out;
    double age = 0;
    const bool got = store.Get(kRightHand, out, age, Clock::now());
    CHECK(got);
    CHECK(!got || (std::fabs(out.pos[1] - 1.2f) < 1e-6f && age < 1.0));
    CHECK(store.Seen(0) && !store.Seen(1));
    CHECK(rx.rejected() == 1);
    rx.Stop();
#ifdef _WIN32
    closesocket(s);
#else
    close(s);
#endif
}

static void TestButtons() {
    DeviceSample h = Dev(kRightHand);
    h.flags |= kFlagButtons;
    h.buttons = kBtnB | kBtnTrigger | 0x8000;  // unknown high bit: ignored, not an error
    DeviceSample t = Dev(0);
    t.buttons = kBtnA;                          // no kFlagButtons -> ignored
    auto b = MakePacket({h, t});
    Packet p;
    CHECK(ParsePacket(b.data(), b.size(), p));
    CHECK(p.devices.size() == 2);
    CHECK(p.devices[0].buttons == (kBtnB | kBtnTrigger));
    CHECK(p.devices[1].buttons == 0);
}

static void TestFeedback() {
    // 90 degrees about Y
    const float c = std::cos(1.5707963f), sn = std::sin(1.5707963f);
    const float m[3][4] = {{c, 0, sn, 0.1f}, {0, 1, 0, 1.7f}, {-sn, 0, c, -0.3f}};
    float q[4];
    MatrixToQuat(m, q);
    CHECK(std::fabs(q[0]) < 1e-5f && std::fabs(q[2]) < 1e-5f);
    CHECK(std::fabs(q[1] - std::sin(0.7853982f)) < 1e-5f && std::fabs(q[3] - std::cos(0.7853982f)) < 1e-5f);
    const float pos[3] = {0.1f, 1.7f, -0.3f};
    uint8_t buf[kFeedbackSize];
    EncodeFeedback(42, kFbHmdValid, pos, q, buf);
    CHECK(std::memcmp(buf, "CFBH", 4) == 0 && buf[4] == 1 && buf[6] == kFbHmdValid && buf[8] == 42);
    float y;
    std::memcpy(&y, buf + 20, 4);
    CHECK(std::fabs(y - 1.7f) < 1e-6f);
}

static int ParseFile(const char* path) {
    std::ifstream f(path, std::ios::binary);
    std::vector<uint8_t> b((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    Packet p;
    if (!ParsePacket(b.data(), b.size(), p)) {
        std::printf("INVALID\n");
        return 1;
    }
    std::printf("seq %u\n", p.seq);
    for (const auto& d : p.devices) {
        std::printf("%u %u %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.4f %.4f %.4f %.4f %.4f %u\n", d.id, d.flags,
                    d.pos[0], d.pos[1], d.pos[2], d.rot[0], d.rot[1], d.rot[2], d.rot[3], d.curls[0], d.curls[1],
                    d.curls[2], d.curls[3], d.curls[4], unsigned(d.buttons));
    }
    return 0;
}

static void TestVelocity() {
    // velocity rides in the curls' place of tracker records (flag bit3)
    DeviceSample d = Dev(0);
    d.flags = kFlagValid | kFlagVelocity;
    d.curls[0] = 1.5f, d.curls[1] = -0.25f, d.curls[2] = 0.0f;
    Packet p;
    auto b = MakePacket({d});
    CHECK(ParsePacket(b.data(), b.size(), p));
    CHECK(p.devices.size() == 1 && p.devices[0].vel[0] == 1.5f && p.devices[0].vel[1] == -0.25f);
    CHECK(p.devices[0].curls[0] == 0.0f);                 // not mistaken for curls
    d.curls[0] = 50.0f;                                   // absurd speed
    b = MakePacket({d});
    CHECK(!ParsePacket(b.data(), b.size(), p));
    DeviceSample h = Dev(kLeftHand);                      // hands carry curls, not velocity
    h.flags = kFlagValid | kFlagVelocity;
    b = MakePacket({h});
    CHECK(!ParsePacket(b.data(), b.size(), p));
    DeviceSample c = Dev(0);                              // both at once: refused
    c.flags = kFlagValid | kFlagCurls | kFlagVelocity;
    b = MakePacket({c});
    CHECK(!ParsePacket(b.data(), b.size(), p));
    DeviceSample o = Dev(0);                              // old app: no bit3 -> no velocity
    o.curls[0] = 0.5f;
    b = MakePacket({o});
    CHECK(ParsePacket(b.data(), b.size(), p) && p.devices[0].vel[0] == 0.0f);
}

int main(int argc, char** argv) {
    if (argc == 3 && std::strcmp(argv[1], "--parse") == 0) return ParseFile(argv[2]);
    TestValid();
    TestRejects();
    TestButtons();
    TestFeedback();
    TestReceiverLoopback();
    TestVelocity();
    if (g_failed) {
        std::printf("%d check(s) failed\n", g_failed);
        return 1;
    }
    std::printf("all protocol tests passed\n");
    return 0;
}
