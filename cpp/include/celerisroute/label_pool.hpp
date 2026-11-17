// Reusable fixed-size block pool for RCSP labels.
//
// boost::r_c_shortest_paths default-constructs its label allocator (the
// allocator instance passed in is ignored), so the allocator type cannot carry
// a pointer to a workspace. PoolAllocator is therefore stateless and resolves
// the pool through a thread-local pointer that LabelPoolScope sets for the
// duration of one solver call. Each thread sees only its own pointer, so
// concurrent solves on different workspaces never share a pool.
#pragma once

#include <cstddef>
#include <memory>
#include <new>
#include <utility>
#include <vector>

namespace celerisroute {

class LabelPool {
public:
    LabelPool() = default;
    LabelPool(const LabelPool&) = delete;
    LabelPool& operator=(const LabelPool&) = delete;
    ~LabelPool() {
        for (const Chunk& chunk : chunks_) ::operator delete(chunk.data);
    }

    void* allocate(std::size_t bytes) {
        bytes = round_up(bytes);
        if (bytes != block_size_) {
            // A new label type: rebuild the free list for its size once every
            // block of the previous size has been returned.
            if (live_ != 0 || foreign_ != 0) {
                ++foreign_;
                return ::operator new(bytes);
            }
            block_size_ = bytes;
            free_ = nullptr;
            chunk_index_ = 0;
            cursor_ = chunks_.empty() ? nullptr : chunks_[0].data;
            end_ = chunks_.empty() ? nullptr : chunks_[0].data + chunks_[0].size;
        }
        ++live_;
        if (free_ != nullptr) {
            void* block = free_;
            free_ = free_->next;
            return block;
        }
        while (cursor_ == nullptr || cursor_ + block_size_ > end_) next_chunk();
        void* block = cursor_;
        cursor_ += block_size_;
        return block;
    }

    void deallocate(void* block, std::size_t bytes) noexcept {
        if (round_up(bytes) != block_size_) {
            --foreign_;
            ::operator delete(block);
            return;
        }
        --live_;
        auto* node = static_cast<FreeNode*>(block);
        node->next = free_;
        free_ = node;
    }

    // Bytes held in chunks, whether or not they are handed out.
    std::size_t bytes_reserved() const {
        std::size_t total = 0;
        for (const Chunk& chunk : chunks_) total += chunk.size;
        return total;
    }

private:
    struct FreeNode {
        FreeNode* next;
    };
    struct Chunk {
        std::byte* data;
        std::size_t size;
    };

    static std::size_t round_up(std::size_t bytes) {
        const std::size_t align = alignof(std::max_align_t);
        if (bytes < sizeof(FreeNode)) bytes = sizeof(FreeNode);
        return (bytes + align - 1) / align * align;
    }

    void next_chunk() {
        if (cursor_ != nullptr) ++chunk_index_;
        if (chunk_index_ == chunks_.size()) {
            const std::size_t previous = chunks_.empty() ? 0 : chunks_.back().size;
            std::size_t size = previous == 0 ? kFirstChunk : previous * 2;
            if (size > kMaxChunk) size = kMaxChunk;
            if (size < block_size_) size = block_size_;
            chunks_.push_back(Chunk{static_cast<std::byte*>(::operator new(size)), size});
        }
        cursor_ = chunks_[chunk_index_].data;
        end_ = cursor_ + chunks_[chunk_index_].size;
    }

    static constexpr std::size_t kFirstChunk = std::size_t{1} << 16;
    static constexpr std::size_t kMaxChunk = std::size_t{1} << 24;

    std::size_t block_size_ = 0;
    std::size_t live_ = 0;     // pool blocks currently handed out
    std::size_t foreign_ = 0;  // blocks of another size served by operator new
    FreeNode* free_ = nullptr;
    std::vector<Chunk> chunks_;
    std::size_t chunk_index_ = 0;
    std::byte* cursor_ = nullptr;
    std::byte* end_ = nullptr;
};

namespace detail {
inline thread_local LabelPool* tls_label_pool = nullptr;
}

// Routes PoolAllocator on the current thread to `pool` until destroyed.
class LabelPoolScope {
public:
    explicit LabelPoolScope(LabelPool* pool)
        : previous_(std::exchange(detail::tls_label_pool, pool)) {}
    ~LabelPoolScope() { detail::tls_label_pool = previous_; }
    LabelPoolScope(const LabelPoolScope&) = delete;
    LabelPoolScope& operator=(const LabelPoolScope&) = delete;

private:
    LabelPool* previous_;
};

template <class T>
struct PoolAllocator {
    using value_type = T;

    PoolAllocator() noexcept = default;
    template <class U>
    PoolAllocator(const PoolAllocator<U>&) noexcept {}

    T* allocate(std::size_t count) {
        LabelPool* pool = detail::tls_label_pool;
        if (pool != nullptr && count == 1 && alignof(T) <= alignof(std::max_align_t)) {
            return static_cast<T*>(pool->allocate(sizeof(T)));
        }
        return std::allocator<T>().allocate(count);
    }

    void deallocate(T* pointer, std::size_t count) noexcept {
        LabelPool* pool = detail::tls_label_pool;
        if (pool != nullptr && count == 1 && alignof(T) <= alignof(std::max_align_t)) {
            pool->deallocate(pointer, sizeof(T));
            return;
        }
        std::allocator<T>().deallocate(pointer, count);
    }

    template <class U>
    friend bool operator==(const PoolAllocator&, const PoolAllocator<U>&) noexcept {
        return true;
    }
    template <class U>
    friend bool operator!=(const PoolAllocator&, const PoolAllocator<U>&) noexcept {
        return false;
    }
};

}  // namespace celerisroute
