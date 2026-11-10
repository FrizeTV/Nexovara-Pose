#include "receiver.h"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
using socklen_t = int;
using sock_t = SOCKET;
static const sock_t kBadSocket = INVALID_SOCKET;
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>
using sock_t = int;
static const sock_t kBadSocket = -1;
#endif

#include <cstring>

namespace nexovarapose {

void SampleStore::Apply(const Packet& p, Clock::time_point now) {
    std::lock_guard<std::mutex> lk(mx_);
    for (const auto& s : p.devices) {
        if (!(s.flags & kFlagValid)) continue;
        Slot& slot = slots_[s.id];
        slot.seen = true;
        slot.sample = s;
        slot.time = now;
    }
    packets_.fetch_add(1);
}

bool SampleStore::Get(uint8_t id, DeviceSample& out, double& age, Clock::time_point now) const {
    std::lock_guard<std::mutex> lk(mx_);
    const Slot& slot = slots_[id];
    if (!slot.seen) return false;
    out = slot.sample;
    age = std::chrono::duration<double>(now - slot.time).count();
    return true;
}

bool SampleStore::Seen(uint8_t id) const {
    std::lock_guard<std::mutex> lk(mx_);
    return slots_[id].seen;
}

static void CloseSocket(intptr_t s) {
#ifdef _WIN32
    closesocket(static_cast<SOCKET>(s));
#else
    close(static_cast<int>(s));
#endif
}

bool UdpReceiver::Start(uint16_t port) {
#ifdef _WIN32
    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) {
        log_("WSAStartup failed");
        return false;
    }
#endif
    const sock_t s = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (s == kBadSocket) {
        log_("socket() failed");
#ifdef _WIN32
        WSACleanup();
#endif
        return false;
    }
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);  // this computer only
    if (bind(s, reinterpret_cast<sockaddr*>(&addr), sizeof addr) != 0) {
        log_("bind(127.0.0.1:" + std::to_string(port) + ") failed - port busy?");
        CloseSocket(static_cast<intptr_t>(s));
#ifdef _WIN32
        WSACleanup();
#endif
        return false;
    }
#ifdef _WIN32
    DWORD tv = 100;  // ms
#else
    timeval tv{0, 100000};
#endif
    setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<const char*>(&tv), sizeof tv);
    sock_ = static_cast<intptr_t>(s);
    running_ = true;
    thread_ = std::thread(&UdpReceiver::Loop, this);
    log_("listening on 127.0.0.1:" + std::to_string(port));
    return true;
}

void UdpReceiver::Stop() {
    if (running_.exchange(false)) {
        if (thread_.joinable()) thread_.join();
    }
    if (sock_ != -1) {
        CloseSocket(sock_);
        sock_ = -1;
#ifdef _WIN32
        WSACleanup();
#endif
    }
}

bool UdpReceiver::SendToApp(const uint8_t* data, size_t len, Clock::time_point now,
                            std::chrono::milliseconds max_age) {
    sockaddr_in to{};
    {
        std::lock_guard<std::mutex> lk(app_mx_);
        if (sock_ == -1 || app_port_ == 0 || app_addr_ != htonl(INADDR_LOOPBACK) || now - app_time_ > max_age)
            return false;
        to.sin_family = AF_INET;
        to.sin_addr.s_addr = app_addr_;
        to.sin_port = app_port_;
    }
    const int n = static_cast<int>(sendto(static_cast<sock_t>(sock_), reinterpret_cast<const char*>(data),
                                          static_cast<int>(len), 0, reinterpret_cast<sockaddr*>(&to), sizeof to));
    return n == static_cast<int>(len);
}

void UdpReceiver::Loop() {
    uint8_t buf[kMaxPacket + 1];
    while (running_) {
        sockaddr_in from{};
        socklen_t fromlen = sizeof from;
        const int n = static_cast<int>(recvfrom(static_cast<sock_t>(sock_), reinterpret_cast<char*>(buf),
                                                sizeof buf, 0, reinterpret_cast<sockaddr*>(&from), &fromlen));
        if (n <= 0) continue;  // timeout or error: just poll the running flag again
        if (from.sin_addr.s_addr != htonl(INADDR_LOOPBACK)) {  // defence in depth
            rejected_.fetch_add(1);
            continue;
        }
        Packet p;
        if (ParsePacket(buf, static_cast<size_t>(n), p)) {
            const auto now = Clock::now();
            store_.Apply(p, now);
            std::lock_guard<std::mutex> lk(app_mx_);
            app_addr_ = from.sin_addr.s_addr;   // checked above: 127.0.0.1
            app_port_ = from.sin_port;
            app_time_ = now;
        } else {
            rejected_.fetch_add(1);
        }
    }
}

}  // namespace nexovarapose
