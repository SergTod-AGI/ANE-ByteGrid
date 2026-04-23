#import "model.h"
#import "config.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const uint32_t kANEBlobMagic = 0xDEADBEEF;
static const uint32_t kANEBlobVersion = 2;
static const size_t kANEBlobAlignment = 64;
static const size_t kANEBlobFileHeaderSize = 64;
static const size_t kANEBlobChunkMetaSize = 64;

static size_t ane_bg_align_up(size_t n, size_t align) {
    return (n + align - 1u) & ~(align - 1u);
}

static float ane_bg_env_float(NSString *name, float fallback) {
    NSString *value = NSProcessInfo.processInfo.environment[name];
    if (value.length == 0) {
        return fallback;
    }
    return value.floatValue;
}

static float ane_bg_alpha_depth_factor(NSInteger layer) {
    const float depthPower = ane_bg_env_float(@"ANE_BG_ALPHA_DEPTH_POWER", 0.0f);
    const float depth = (float)(layer + 1) / (float)ANE_BG_LAYERS;
    return powf(depth, depthPower);
}

static uint32_t ane_bg_next_u32(uint32_t *state) {
    *state = (*state * 1664525u) + 1013904223u;
    return *state;
}

@interface ANEBlobWriter ()
@property (nonatomic, strong) NSMutableArray<NSDictionary *> *entries;
@end

@implementation ANEBlobWriter

- (instancetype)init {
    self = [super init];
    if (!self) {
        return nil;
    }
    _entries = [NSMutableArray array];
    return self;
}

- (NSInteger)addFloat16:(const float *)data count:(size_t)count {
    NSMutableData *raw = [NSMutableData dataWithLength:count * sizeof(uint16_t)];
    uint16_t *dst = raw.mutableBytes;
    for (size_t i = 0; i < count; ++i) {
        dst[i] = ane_bg_fp16_from_float(data[i]);
    }
    return [self addRawWithType:ANEBlobDataTypeFloat16 bytes:raw];
}

- (NSInteger)addFloat32:(const float *)data count:(size_t)count {
    NSMutableData *raw = [NSMutableData dataWithLength:count * sizeof(float)];
    memcpy(raw.mutableBytes, data, raw.length);
    return [self addRawWithType:ANEBlobDataTypeFloat32 bytes:raw];
}

- (NSInteger)addRawWithType:(ANEBlobDataType)type bytes:(NSData *)bytes {
    NSDictionary *entry = @{
        @"type": @(type),
        @"data": bytes ?: [NSData data]
    };
    [self.entries addObject:entry];
    return self.entries.count - 1;
}

- (uint64_t)offsetForBlobAtIndex:(NSInteger)index {
    if (index < 0 || index >= self.entries.count) {
        return 0;
    }
    size_t metaSize = ane_bg_align_up(kANEBlobFileHeaderSize + (kANEBlobChunkMetaSize * self.entries.count), kANEBlobAlignment);
    size_t offset = metaSize;
    for (NSInteger i = 0; i < index; ++i) {
        NSData *data = self.entries[(NSUInteger)i][@"data"];
        offset += ane_bg_align_up(data.length, kANEBlobAlignment);
    }
    return (uint64_t)offset;
}

- (NSInteger)count {
    return self.entries.count;
}

- (NSData *)build:(NSError **)error {
    if (self.entries.count == 0) {
        if (error) {
            *error = [NSError errorWithDomain:@"ANEByteGrid"
                                         code:13
                                     userInfo:@{NSLocalizedDescriptionKey: @"BlobWriter has no entries."}];
        }
        return nil;
    }

    size_t metaSize = ane_bg_align_up(kANEBlobFileHeaderSize + (kANEBlobChunkMetaSize * self.entries.count), kANEBlobAlignment);
    size_t totalSize = metaSize;
    for (NSDictionary *entry in self.entries) {
        NSData *data = entry[@"data"];
        totalSize += ane_bg_align_up(data.length, kANEBlobAlignment);
    }

    NSMutableData *blob = [NSMutableData dataWithLength:totalSize];
    uint8_t *buf = blob.mutableBytes;
    uint64_t count = (uint64_t)self.entries.count;
    memcpy(buf, &count, sizeof(count));
    memcpy(buf + 8, &(uint32_t){kANEBlobVersion}, sizeof(uint32_t));

    size_t dataOffset = metaSize;
    for (NSUInteger i = 0; i < self.entries.count; ++i) {
        NSDictionary *entry = self.entries[i];
        NSData *data = entry[@"data"];
        const uint32_t type = [entry[@"type"] unsignedIntValue];
        const size_t metaOffset = kANEBlobFileHeaderSize + (kANEBlobChunkMetaSize * i);
        memcpy(buf + metaOffset, &(uint32_t){kANEBlobMagic}, sizeof(uint32_t));
        memcpy(buf + metaOffset + 4, &type, sizeof(uint32_t));
        memcpy(buf + metaOffset + 8, &(uint64_t){data.length}, sizeof(uint64_t));
        memcpy(buf + metaOffset + 16, &(uint64_t){dataOffset}, sizeof(uint64_t));
        memcpy(buf + dataOffset, data.bytes, data.length);
        dataOffset += ane_bg_align_up(data.length, kANEBlobAlignment);
    }
    return blob;
}

@end

uint16_t ane_bg_fp16_from_float(float value) {
    union {
        float f;
        uint32_t u;
    } in = { .f = value };
    const uint32_t sign = (in.u >> 16) & 0x8000u;
    int32_t exp = ((in.u >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = in.u & 0x7fffffu;
    if (exp <= 0) {
        if (exp < -10) {
            return (uint16_t)sign;
        }
        mant = (mant | 0x800000u) >> (1 - exp);
        return (uint16_t)(sign | ((mant + 0x1000u) >> 13));
    }
    if (exp >= 31) {
        return (uint16_t)(sign | 0x7c00u);
    }
    return (uint16_t)(sign | ((uint32_t)exp << 10) | ((mant + 0x1000u) >> 13));
}

void ane_bg_fill_xavier_uniform(uint16_t *dst, size_t rows, size_t cols, uint32_t seed) {
    float scale = sqrtf(6.0f / (float)(rows + cols));
    uint32_t state = seed ? seed : 1u;
    for (size_t i = 0; i < rows * cols; ++i) {
        float u = (float)(ane_bg_next_u32(&state) & 0x00ffffffu) / (float)0x01000000u;
        float v = (u * 2.0f - 1.0f) * scale;
        dst[i] = ane_bg_fp16_from_float(v);
    }
}

BOOL ane_bg_write_weight_blob(NSString *path, const uint16_t *data, size_t element_count, NSError **error) {
    const uint32_t payloadBytes = (uint32_t)(element_count * sizeof(uint16_t));
    NSMutableData *blob = [NSMutableData dataWithLength:128];
    uint8_t *header = blob.mutableBytes;
    header[0] = 0x01;
    header[4] = 0x02;
    memcpy(header + 64, &(uint32_t){0xDEADBEEF}, sizeof(uint32_t));
    header[68] = 0x01;
    memcpy(header + 72, &payloadBytes, sizeof(payloadBytes));
    memcpy(header + 80, &(uint32_t){128}, sizeof(uint32_t));
    [blob appendBytes:data length:element_count * sizeof(uint16_t)];
    return [blob writeToFile:path options:NSDataWritingAtomic error:error];
}

static BOOL ane_bg_generate_blob(NSString *path, size_t rows, size_t cols, uint32_t seed, NSError **error) {
    const size_t count = rows * cols;
    uint16_t *buffer = calloc(count, sizeof(uint16_t));
    if (!buffer) {
        if (error) {
            *error = [NSError errorWithDomain:@"ANEByteGrid" code:12 userInfo:@{NSLocalizedDescriptionKey: @"calloc failed"}];
        }
        return NO;
    }
    ane_bg_fill_xavier_uniform(buffer, rows, cols, seed);
    BOOL ok = ane_bg_write_weight_blob(path, buffer, count, error);
    free(buffer);
    return ok;
}

static BOOL ane_bg_generate_blob_scaled(NSString *path, size_t rows, size_t cols, uint32_t seed, float scale, NSError **error) {
    const size_t count = rows * cols;
    uint16_t *buffer = calloc(count, sizeof(uint16_t));
    if (!buffer) {
        if (error) {
            *error = [NSError errorWithDomain:@"ANEByteGrid" code:12 userInfo:@{NSLocalizedDescriptionKey: @"calloc failed"}];
        }
        return NO;
    }
    ane_bg_fill_xavier_uniform(buffer, rows, cols, seed);
    if (fabsf(scale - 1.0f) > 1e-6f) {
        for (size_t i = 0; i < count; ++i) {
            // Decode->scale->re-encode to keep blob format unchanged while allowing head-only tuning.
            uint16_t h = buffer[i];
            const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
            uint32_t exp = (h >> 10) & 0x1fu;
            uint32_t frac = h & 0x03ffu;
            uint32_t bits = 0;
            if (exp == 0) {
                if (frac == 0) {
                    bits = sign;
                } else {
                    exp = 1;
                    while ((frac & 0x0400u) == 0) {
                        frac <<= 1;
                        exp--;
                    }
                    frac &= 0x03ffu;
                    bits = sign | ((exp + 127 - 15) << 23) | (frac << 13);
                }
            } else if (exp == 31) {
                bits = sign | 0x7f800000u | (frac << 13);
            } else {
                bits = sign | ((exp + 127 - 15) << 23) | (frac << 13);
            }
            union { uint32_t u; float f; } v = { .u = bits };
            v.f *= scale;
            buffer[i] = ane_bg_fp16_from_float(v.f);
        }
    }
    BOOL ok = ane_bg_write_weight_blob(path, buffer, count, error);
    free(buffer);
    return ok;
}

static BOOL ane_bg_generate_scalar_blob(NSString *path, float value, NSError **error) {
    uint16_t scalar = ane_bg_fp16_from_float(value);
    return ane_bg_write_weight_blob(path, &scalar, 1, error);
}

static BOOL ane_bg_generate_constant_vector_blob(NSString *path, size_t count, float value, NSError **error) {
    uint16_t *buffer = calloc(count, sizeof(uint16_t));
    if (!buffer) {
        if (error) {
            *error = [NSError errorWithDomain:@"ANEByteGrid" code:12 userInfo:@{NSLocalizedDescriptionKey: @"calloc failed"}];
        }
        return NO;
    }
    const uint16_t encoded = ane_bg_fp16_from_float(value);
    for (size_t i = 0; i < count; ++i) {
        buffer[i] = encoded;
    }
    BOOL ok = ane_bg_write_weight_blob(path, buffer, count, error);
    free(buffer);
    return ok;
}

BOOL ane_bg_generate_all_weight_blobs(NSString *weight_dir, NSError **error) {
    NSFileManager *fm = NSFileManager.defaultManager;
    const float rmsInit = ane_bg_env_float(@"ANE_BG_RMS_INIT", 1.0f);
    const float alphaBase = ane_bg_env_float(@"ANE_BG_ALPHA_INIT", 1.0f / sqrtf((float)ANE_BG_LAYERS));
    const float alphaLocalMul = ane_bg_env_float(@"ANE_BG_ALPHA_LOCAL_MUL", 1.15f);
    const float alphaGlobalMul = ane_bg_env_float(@"ANE_BG_ALPHA_GLOBAL_MUL", 1.40f);
    const float alphaMLPMul = ane_bg_env_float(@"ANE_BG_ALPHA_MLP_MUL", 0.05f);
    const float headWeightScale = ane_bg_env_float(@"ANE_BG_HEAD_WEIGHT_SCALE", 0.98f);
    const float headLogitScale = ane_bg_env_float(@"ANE_BG_HEAD_LOGIT_SCALE", 0.97f);
    if (![fm createDirectoryAtPath:weight_dir withIntermediateDirectories:YES attributes:nil error:error]) {
        return NO;
    }
    if (!ane_bg_generate_blob([weight_dir stringByAppendingPathComponent:@"stem.bin"], 512, 320, 1u, error)) {
        return NO;
    }
    for (NSInteger layer = 0; layer < 24; ++layer) {
        const float depthFactor = ane_bg_alpha_depth_factor(layer);
        const float alphaLocal = alphaBase * alphaLocalMul * depthFactor;
        const float alphaGlobal = alphaBase * alphaGlobalMul * depthFactor;
        const float alphaMLP = alphaBase * alphaMLPMul * depthFactor;
        if (!ane_bg_generate_blob([weight_dir stringByAppendingPathComponent:[NSString stringWithFormat:@"block_%02ld_local.bin", (long)layer]], 8192, 16, (uint32_t)(10 + layer), error)) {
            return NO;
        }
        if (!ane_bg_generate_blob([weight_dir stringByAppendingPathComponent:[NSString stringWithFormat:@"block_%02ld_global.bin", (long)layer]], 8192, 16, (uint32_t)(100 + layer), error)) {
            return NO;
        }
        if (!ane_bg_generate_blob([weight_dir stringByAppendingPathComponent:[NSString stringWithFormat:@"block_%02ld_wv.bin", (long)layer]], 1024, 512, (uint32_t)(200 + layer), error)) {
            return NO;
        }
        if (!ane_bg_generate_blob([weight_dir stringByAppendingPathComponent:[NSString stringWithFormat:@"block_%02ld_wg.bin", (long)layer]], 1024, 512, (uint32_t)(300 + layer), error)) {
            return NO;
        }
        if (!ane_bg_generate_blob([weight_dir stringByAppendingPathComponent:[NSString stringWithFormat:@"block_%02ld_wo.bin", (long)layer]], 512, 1024, (uint32_t)(400 + layer), error)) {
            return NO;
        }
        if (!ane_bg_generate_constant_vector_blob([weight_dir stringByAppendingPathComponent:[NSString stringWithFormat:@"block_%02ld_rms_local.bin", (long)layer]], ANE_BG_HIDDEN, rmsInit, error)) {
            return NO;
        }
        if (!ane_bg_generate_constant_vector_blob([weight_dir stringByAppendingPathComponent:[NSString stringWithFormat:@"block_%02ld_rms_global.bin", (long)layer]], ANE_BG_HIDDEN, rmsInit, error)) {
            return NO;
        }
        if (!ane_bg_generate_constant_vector_blob([weight_dir stringByAppendingPathComponent:[NSString stringWithFormat:@"block_%02ld_rms_ffn.bin", (long)layer]], ANE_BG_HIDDEN, rmsInit, error)) {
            return NO;
        }
        if (!ane_bg_generate_scalar_blob([weight_dir stringByAppendingPathComponent:[NSString stringWithFormat:@"block_%02ld_alpha_local.bin", (long)layer]], alphaLocal, error)) {
            return NO;
        }
        if (!ane_bg_generate_scalar_blob([weight_dir stringByAppendingPathComponent:[NSString stringWithFormat:@"block_%02ld_alpha_global.bin", (long)layer]], alphaGlobal, error)) {
            return NO;
        }
        if (!ane_bg_generate_scalar_blob([weight_dir stringByAppendingPathComponent:[NSString stringWithFormat:@"block_%02ld_alpha_mlp.bin", (long)layer]], alphaMLP, error)) {
            return NO;
        }
    }
    if (!ane_bg_generate_constant_vector_blob([weight_dir stringByAppendingPathComponent:@"head_rms.bin"], ANE_BG_HIDDEN, rmsInit, error)) {
        return NO;
    }
    if (!ane_bg_generate_blob_scaled([weight_dir stringByAppendingPathComponent:@"head.bin"], 256, 512, 999u, headWeightScale, error)) {
        return NO;
    }
    return ane_bg_generate_scalar_blob([weight_dir stringByAppendingPathComponent:@"head_logit_scale.bin"], headLogitScale, error);
}
