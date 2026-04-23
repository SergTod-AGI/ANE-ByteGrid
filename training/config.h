#ifndef ANE_BG_CONFIG_H
#define ANE_BG_CONFIG_H

#include <stdint.h>

enum {
    ANE_BG_SEQ = 256,
    ANE_BG_INPUT_CHANNELS = 320,
    ANE_BG_BYTE_CHANNELS = 256,
    ANE_BG_CLASS_CHANNELS = 16,
    ANE_BG_POS_CHANNELS = 32,
    ANE_BG_CTRL_CHANNELS = 16,
    ANE_BG_HIDDEN = 512,
    ANE_BG_LAYERS = 24,
    ANE_BG_BLOCK = 16,
    ANE_BG_GROUPS = 16,
    ANE_BG_PACKED_CHANNELS = 8192,
    ANE_BG_GLU = 1024,
    ANE_BG_VOCAB = 256
};

static inline uint32_t ane_bg_tensor_elements(uint32_t channels, uint32_t seq) {
    return channels * seq;
}

static inline uint32_t ane_bg_tensor_bytes(uint32_t channels, uint32_t seq) {
    return ane_bg_tensor_elements(channels, seq) * 2u;
}

#endif
