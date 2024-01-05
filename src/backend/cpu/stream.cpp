#include "cuda/stream.hpp"

#include <tl/expected.hpp>

#include <atomic>
#include <mutex>
#include <string>
#include <tuple>
#include <unordered_map>

// Global atomic counter for stream IDs
std::atomic<uint64_t> gStreamIdCounter(0);

// Global map to store stream ID to settings
std::unordered_map<uint64_t, std::pair<std::string, StreamPriority>> gStreamIdMap;

// Mutex to protect access to the gStreamIdToNameMap
std::mutex gMapMutex;

tl::expected<cudaStream_t, StreamError> createStream(
  const std::string_view name, const StreamPriority priority) {
  // Generate a unique stream ID
  uint64_t streamId = gStreamIdCounter.fetch_add(1);

  {
    // Store the name associated with the stream ID
    std::lock_guard<std::mutex> lock(gMapMutex);
    gStreamIdMap.emplace(streamId, std::make_pair(name, priority));
  }

  // In a CPU-only context, the stream can be represented by its unique ID
  // Cast the streamId to cudaStream_t (which is just a placeholder in this context)
  return reinterpret_cast<cudaStream_t>(uintptr_t(streamId));
}

void destroyStream(cudaStream_t stream) {
  // In a CPU-only context, the stream can be represented by its unique ID
  // Cast the stream to uint64_t to get the unique ID
  const uint64_t streamId = reinterpret_cast<uint64_t>(stream);

  {
    // Remove the stream ID from the map
    std::lock_guard<std::mutex> lock(gMapMutex);
    gStreamIdMap.erase(streamId);
  }
}
