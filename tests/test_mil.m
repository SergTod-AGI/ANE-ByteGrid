#import "../training/ane_mil_gen.h"
#import "../training/model.h"

#import <Foundation/Foundation.h>
#import <math.h>

static BOOL contains(NSString *haystack, NSString *needle) {
    return [haystack rangeOfString:needle].location != NSNotFound;
}

static BOOL file_exists(NSString *path) {
    return [[NSFileManager defaultManager] fileExistsAtPath:path];
}

static float fp16_to_float(uint16_t h) {
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
    union {
        uint32_t u;
        float f;
    } out = { .u = bits };
    return out.f;
}

static BOOL read_blob_scalar(NSString *path, float *valueOut) {
    NSData *data = [NSData dataWithContentsOfFile:path];
    if (data.length < 130) {
        return NO;
    }
    const uint8_t *bytes = data.bytes;
    uint16_t encoded = 0;
    memcpy(&encoded, bytes + 128, sizeof(encoded));
    if (valueOut) {
        *valueOut = fp16_to_float(encoded);
    }
    return YES;
}

static BOOL expect_contains(NSString *label, NSString *haystack, NSString *needle) {
    if (contains(haystack, needle)) {
        return YES;
    }
    NSLog(@"MIL test failed: %@ missing %@", label, needle);
    return NO;
}

static BOOL expect_file_exists(NSString *path) {
    if (file_exists(path)) {
        return YES;
    }
    NSLog(@"MIL test failed: missing file %@", path);
    return NO;
}

static BOOL expect_near(NSString *label, float actual, float expected, float tolerance) {
    if (fabsf(actual - expected) <= tolerance) {
        return YES;
    }
    NSLog(@"MIL test failed: %@ expected %.5f got %.5f", label, expected, actual);
    return NO;
}

int main(void) {
    @autoreleasepool {
        NSString *weights = @"/tmp/ane-bytegrid-test-weights";
        NSError *error = nil;
        if (!ane_bg_generate_all_weight_blobs(weights, &error)) {
            NSLog(@"blob generation failed: %@", error);
            return 1;
        }

        NSString *stem = ane_bg_gen_stem_mil([weights stringByAppendingPathComponent:@"stem.bin"]);
        NSString *local = ane_bg_gen_local_mixer_mil([weights stringByAppendingPathComponent:@"block_00_local.bin"]);
        NSString *global = ane_bg_gen_global_mixer_mil([weights stringByAppendingPathComponent:@"block_00_global.bin"]);
        NSString *glu = ane_bg_gen_channel_glu_mil(weights);
        NSString *block = ane_bg_gen_block_mil(weights, 0);
        NSString *head = ane_bg_gen_head_mil([weights stringByAppendingPathComponent:@"head.bin"]);
        NSString *full = ane_bg_gen_full_model_mil(weights);
        NSString *exact = ane_bg_gen_conv_fp16_exact(320, 512, 256);

        if (!expect_contains(@"stem", stem, @"tensor<fp16, [512, 320, 1, 1]>")) return 1;
        if (!expect_contains(@"stem", stem, @"[buildInfo = dict<string, string>(")) return 1;
        if (!expect_contains(@"stem", stem, @"@model_path/weights/stem.bin")) return 1;
        if (!expect_contains(@"stem", stem, @"offset = uint64(64)")) return 1;
        if (!expect_contains(@"local", local, @"groups=gr")) return 1;
        if (!expect_contains(@"local", local, @"transpose")) return 1;
        if (!expect_contains(@"global", global, @"tensor<fp16, [1,8192,1,16]>")) return 1;
        if (!expect_contains(@"glu", glu, @"sigmoid")) return 1;
        if (!expect_contains(@"block", block, @"reduce_sum")) return 1;
        if (!expect_contains(@"block", block, @"pow(x=local_var, y=local_nhalf)")) return 1;
        if (!expect_contains(@"block", block, @"@model_path/weights/block_00_rms_local.bin")) return 1;
        if (!expect_contains(@"block", block, @"@model_path/weights/block_00_rms_global.bin")) return 1;
        if (!expect_contains(@"block", block, @"@model_path/weights/block_00_rms_ffn.bin")) return 1;
        if (!expect_contains(@"block", block, @"@model_path/weights/block_00_alpha_local.bin")) return 1;
        if (!expect_contains(@"block", block, @"@model_path/weights/block_00_alpha_global.bin")) return 1;
        if (!expect_contains(@"block", block, @"@model_path/weights/block_00_alpha_mlp.bin")) return 1;
        if (!expect_contains(@"block", block, @"localScaled = mul(x=localOut, y=alphaLocal)")) return 1;
        if (!expect_contains(@"block", block, @"globalScaled = mul(x=globalOut, y=alphaGlobal)")) return 1;
        if (!expect_contains(@"block", block, @"mlpScaled = mul(x=mlp, y=alphaMLP)")) return 1;
        if (!expect_contains(@"block", block, @"tensor<fp16, [1,1024,1,256]> h1 = conv")) return 1;
        if (!expect_contains(@"head", head, @"@model_path/weights/head_rms.bin")) return 1;
        if (!expect_contains(@"head", head, @"x = xL")) return 1;
        if (!expect_contains(@"full", full, @"block_23_wo.bin")) return 1;
        if (!expect_contains(@"full", full, @"block_23_rms_local.bin")) return 1;
        if (!expect_contains(@"full", full, @"block_23_rms_global.bin")) return 1;
        if (!expect_contains(@"full", full, @"block_23_rms_ffn.bin")) return 1;
        if (!expect_contains(@"full", full, @"block_23_alpha_local.bin")) return 1;
        if (!expect_contains(@"full", full, @"block_23_alpha_global.bin")) return 1;
        if (!expect_contains(@"full", full, @"block_23_alpha_mlp.bin")) return 1;
        if (!expect_contains(@"full", full, @"head_rms.bin")) return 1;
        if (!expect_contains(@"exact", exact, @"tensor<fp16, [512, 320, 1, 1]> W = const()")) return 1;
        if (!expect_file_exists([weights stringByAppendingPathComponent:@"block_00_rms_local.bin"])) return 1;
        if (!expect_file_exists([weights stringByAppendingPathComponent:@"block_00_rms_global.bin"])) return 1;
        if (!expect_file_exists([weights stringByAppendingPathComponent:@"block_00_rms_ffn.bin"])) return 1;
        if (!expect_file_exists([weights stringByAppendingPathComponent:@"block_00_alpha_local.bin"])) return 1;
        if (!expect_file_exists([weights stringByAppendingPathComponent:@"block_00_alpha_global.bin"])) return 1;
        if (!expect_file_exists([weights stringByAppendingPathComponent:@"block_00_alpha_mlp.bin"])) return 1;
        if (!expect_file_exists([weights stringByAppendingPathComponent:@"head_rms.bin"])) return 1;

        float rmsValue = 0.0f;
        float alphaLocal0 = 0.0f;
        float alphaGlobal0 = 0.0f;
        float alphaMLP0 = 0.0f;
        float alphaLocalLast = 0.0f;
        if (!read_blob_scalar([weights stringByAppendingPathComponent:@"block_00_rms_local.bin"], &rmsValue)) return 1;
        if (!read_blob_scalar([weights stringByAppendingPathComponent:@"block_00_alpha_local.bin"], &alphaLocal0)) return 1;
        if (!read_blob_scalar([weights stringByAppendingPathComponent:@"block_00_alpha_global.bin"], &alphaGlobal0)) return 1;
        if (!read_blob_scalar([weights stringByAppendingPathComponent:@"block_00_alpha_mlp.bin"], &alphaMLP0)) return 1;
        if (!read_blob_scalar([weights stringByAppendingPathComponent:@"block_23_alpha_local.bin"], &alphaLocalLast)) return 1;
        if (!expect_near(@"rms init", rmsValue, 1.0f, 0.01f)) return 1;
        const float alphaBase = 1.0f / sqrtf(24.0f);
        const float depthFactor0 = powf(1.0f / 24.0f, 0.0f);
        if (!expect_near(@"alpha local layer0", alphaLocal0, alphaBase * 1.15f * depthFactor0, 0.01f)) return 1;
        if (!expect_near(@"alpha global layer0", alphaGlobal0, alphaBase * 1.40f * depthFactor0, 0.01f)) return 1;
        if (!expect_near(@"alpha mlp layer0", alphaMLP0, alphaBase * 0.05f * depthFactor0, 0.01f)) return 1;
        if (!(alphaMLP0 < alphaLocal0 && alphaLocal0 < alphaGlobal0)) {
            NSLog(@"MIL test failed: alpha branch ordering unexpected mlp=%.5f local=%.5f global=%.5f",
                  alphaMLP0, alphaLocal0, alphaGlobal0);
            return 1;
        }
        if (!(alphaLocalLast >= alphaLocal0)) {
            NSLog(@"MIL test failed: alpha depth profile invalid first=%.5f last=%.5f",
                  alphaLocal0, alphaLocalLast);
            return 1;
        }

        NSLog(@"MIL tests passed");
        return 0;
    }
}
