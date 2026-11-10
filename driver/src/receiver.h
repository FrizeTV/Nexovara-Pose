// UDP receiver: listens ONLY on 127.0.0.1, keeps the newest sample per device.
// No OpenVR dependency (unit-testable).
#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <thread>

#include "protocol.h"

namespace nexovarapose {

using Clock = std::chrono::steady_clock;

class SampleStore {
public:
    void Apply(const Packet& p, Clock::time_point now);
    // Newest sample of a device; false if it was never received.
    bool Get(uint8_t id, DeviceSample& out, double& age_seconds, Clock::time_point now) const;
    bool Seen(uint8_t id) const;
    uint64_t packets() const { return packets_.load(); }

private:
    struct Slot {
        bool seen = false;
        DeviceSample sample;
        Clock::time_point time;
    };
    mutable std::mutex mx_;
    std::array<Slot, 256> slots_{};
    std::atomic<uint64_t> packets_{0};
};

class UdpReceiver {
public:
    using LogFn = std::function<void(const std::string&)>;
    UdpReceiver(SampleStore& store, LogFn log) : store_(store), log_(std::move(log)) {}
    ~UdpReceiver() { Stop(); }
    bool Start(uint16_t port);  // binds 127.0.0.1:port
    void Stop();
    uint64_t rejected() const { return rejected_.load(); }
    // Send a datagram back to the app (the loopback address its last valid
    // packet came from), only if it sent something within `max_age`.
    bool SendToApp(const uint8_t* data, size_t len, Clock::time_point now,
                   std::chrono::milliseconds max_age = std::chrono::milliseconds(2000));

private:
    void Loop();
    SampleStore& store_;
    LogFn log_;
    std::thread thread_;
    std::atomic<bool> running_{false};
    std::atomic<uint64_t> rejected_{0};
    intptr_t sock_ = -1;
    std::mutex app_mx_;
    uint32_t app_addr_ = 0;   // network byte order; 0 = unknown
    uint16_t app_port_ = 0;   // network byte order
    Clock::time_point app_time_{};
};

}  // namespace nexovarapose
