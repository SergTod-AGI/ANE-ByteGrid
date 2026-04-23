#import "../training/config.h"
#import "../training/iosurface_io.h"

#import <Foundation/Foundation.h>

static BOOL test_tensor_sizes(void) {
    return ane_bg_tensor_bytes(ANE_BG_HIDDEN, ANE_BG_SEQ) == 262144u &&
           ane_bg_tensor_bytes(ANE_BG_PACKED_CHANNELS, ANE_BG_BLOCK) == 262144u &&
           ane_bg_tensor_bytes(ANE_BG_GLU, ANE_BG_SEQ) == 524288u;
}

static BOOL test_pack_roundtrip(void) {
    const uint32_t channels = 4;
    const uint32_t seq = 8;
    uint16_t src[32];
    uint16_t packed[32];
    uint16_t dst[32];
    for (uint32_t i = 0; i < 32; ++i) {
        src[i] = (uint16_t)i;
    }
    ane_bg_pack_tc_to_cf_fp16(src, packed, channels, seq);
    ane_bg_unpack_cf_to_tc_fp16(packed, dst, channels, seq);
    for (uint32_t i = 0; i < 32; ++i) {
        if (src[i] != dst[i]) {
            return NO;
        }
    }
    return YES;
}

int main(void) {
    @autoreleasepool {
        if (!test_tensor_sizes()) {
            NSLog(@"shape test failed");
            return 1;
        }
        if (!test_pack_roundtrip()) {
            NSLog(@"pack roundtrip failed");
            return 1;
        }
        NSLog(@"shape tests passed");
        return 0;
    }
}
