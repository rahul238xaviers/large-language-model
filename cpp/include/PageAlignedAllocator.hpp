#pragma once

#include <cstdlib>
#include <cstddef>
#include <new>
#include <stdexcept>

template <typename T>
class PageAlignedAllocator {
public:
    using value_type = T;

    PageAlignedAllocator() = default;

    template <typename U>
    PageAlignedAllocator(const PageAlignedAllocator<U>&) {}

    T* allocate(size_t n) {
        if (n == 0) return nullptr;
        if (n > size_t(-1) / sizeof(T))
            throw std::bad_array_new_length();
        void* ptr = nullptr;
        if (posix_memalign(&ptr, 16384, n * sizeof(T)) != 0)
            throw std::bad_alloc();
        return static_cast<T*>(ptr);
    }

    void deallocate(T* ptr, size_t) noexcept {
        free(ptr);
    }
};

template <typename T, typename U>
bool operator==(const PageAlignedAllocator<T>&, const PageAlignedAllocator<U>&) {
    return true;
}
template <typename T, typename U>
bool operator!=(const PageAlignedAllocator<T>&, const PageAlignedAllocator<U>&) {
    return false;
}
