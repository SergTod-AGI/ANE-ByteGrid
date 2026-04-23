#import "config.h"
#import "iosurface_io.h"

#import <CoreFoundation/CoreFoundation.h>

IOSurfaceRef ane_bg_create_surface(uint32_t channels, uint32_t seq) {
    const uint32_t bytes = ane_bg_tensor_bytes(channels, seq);
    const uint32_t width = seq;
    const uint32_t height = channels;
    const uint32_t bpr = seq * 2u;
    const uint32_t alloc_size = bytes;
    const void *keys[] = {
        kIOSurfaceWidth,
        kIOSurfaceHeight,
        kIOSurfaceBytesPerRow,
        kIOSurfaceBytesPerElement,
        kIOSurfaceAllocSize,
    };
    const uint32_t one = 2u;
    const void *values[] = {
        CFNumberCreate(NULL, kCFNumberSInt32Type, &width),
        CFNumberCreate(NULL, kCFNumberSInt32Type, &height),
        CFNumberCreate(NULL, kCFNumberSInt32Type, &bpr),
        CFNumberCreate(NULL, kCFNumberSInt32Type, &one),
        CFNumberCreate(NULL, kCFNumberSInt32Type, &alloc_size),
    };
    CFDictionaryRef dict = CFDictionaryCreate(NULL, keys, values, 5,
                                              &kCFTypeDictionaryKeyCallBacks,
                                              &kCFTypeDictionaryValueCallBacks);
    IOSurfaceRef surface = IOSurfaceCreate(dict);
    CFRelease(dict);
    for (size_t i = 0; i < 5; ++i) {
        CFRelease(values[i]);
    }
    if (surface && IOSurfaceGetAllocSize(surface) < bytes) {
        CFRelease(surface);
        return NULL;
    }
    return surface;
}

uint16_t *ane_bg_lock_surface_fp16(IOSurfaceRef surface, size_t *element_count) {
    if (!surface) {
        return NULL;
    }
    IOSurfaceLock(surface, 0, NULL);
    const size_t bytes = IOSurfaceGetAllocSize(surface);
    if (element_count) {
        *element_count = bytes / sizeof(uint16_t);
    }
    return (uint16_t *)IOSurfaceGetBaseAddress(surface);
}

void ane_bg_unlock_surface(IOSurfaceRef surface) {
    if (surface) {
        IOSurfaceUnlock(surface, 0, NULL);
    }
}

void ane_bg_pack_tc_to_cf_fp16(const uint16_t *src, uint16_t *dst, uint32_t channels, uint32_t seq) {
    for (uint32_t t = 0; t < seq; ++t) {
        for (uint32_t c = 0; c < channels; ++c) {
            dst[c * seq + t] = src[t * channels + c];
        }
    }
}

void ane_bg_unpack_cf_to_tc_fp16(const uint16_t *src, uint16_t *dst, uint32_t channels, uint32_t seq) {
    for (uint32_t t = 0; t < seq; ++t) {
        for (uint32_t c = 0; c < channels; ++c) {
            dst[t * channels + c] = src[c * seq + t];
        }
    }
}
