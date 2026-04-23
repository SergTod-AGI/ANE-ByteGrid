#ifndef ANE_BG_IOSURFACE_IO_H
#define ANE_BG_IOSURFACE_IO_H

#include <IOSurface/IOSurface.h>
#include <stdint.h>

IOSurfaceRef ane_bg_create_surface(uint32_t channels, uint32_t seq);
uint16_t *ane_bg_lock_surface_fp16(IOSurfaceRef surface, size_t *element_count);
void ane_bg_unlock_surface(IOSurfaceRef surface);
void ane_bg_pack_tc_to_cf_fp16(const uint16_t *src, uint16_t *dst, uint32_t channels, uint32_t seq);
void ane_bg_unpack_cf_to_tc_fp16(const uint16_t *src, uint16_t *dst, uint32_t channels, uint32_t seq);

#endif
