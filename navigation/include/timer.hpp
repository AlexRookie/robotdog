#pragma once

#include "single_thread.hpp"

class Timer : protected single_thread {
public:
  typedef std::function<void()> CallbackType;

  Timer(): single_thread(thread_fun_t([this](const bool& t){this->worker(t);}), "Timer")
  {}

  ~Timer() {
    stop();
  }

  void registerCallback(CallbackType callback) {
    this->callback = callback;
  }

  void start(int period) {
	  if (single_thread::isAlive()) {
		  return;
	  }
    this->period = period;
	  single_thread::start();
  }

  bool stop() {
	  if (!single_thread::isAlive()) {
		  return true;
	  }
	  return single_thread::stop();
  }

private:
  CallbackType callback;
  int period;

  void worker(const bool& terminating) {
    auto t0 = std::chrono::steady_clock::now();
	  while (!terminating) {
      auto t1 = t0 + std::chrono::milliseconds(period);
      callback();
      std::this_thread::sleep_until(t1);
      t0 = t1;
    }
  }
};