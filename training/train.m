#import "ane_mil_gen.h"
#import "ane_runtime.h"
#import "config.h"
#import "iosurface_io.h"
#import "model.h"

#import <Foundation/Foundation.h>
#import <math.h>
#import <mach/mach_time.h>

typedef struct {
    size_t nanInfCount;
    double saturationRate;
    double mean;
    double variance;
} TrainTensorStats;

typedef struct {
    double loss;
    double perplexity;
    double meanMargin;
    TrainTensorStats hidden;
    TrainTensorStats output;
    BOOL hasHidden;
    BOOL hasOutput;
    BOOL stable;
} TrainSplitMetrics;

typedef struct {
    TrainSplitMetrics train;
    TrainSplitMetrics eval;
    BOOL stable;
    double compileMs;
    double loadMs;
    double avgMs;
    double throughputMBps;
    double passesPerSecond;
    double tokensPerSecond;
    double estConvGMACPerPass;
    double estConvTFLOPS;
} TrainRunMetrics;

static double train_estimated_conv_macs_per_pass(void) {
    const double stemMACs = (double)ANE_BG_HIDDEN * (double)ANE_BG_INPUT_CHANNELS * (double)ANE_BG_SEQ;
    const double localMACs = 2.0 * (double)ANE_BG_PACKED_CHANNELS * (double)ANE_BG_BLOCK;
    const double globalMACs = 2.0 * (double)ANE_BG_PACKED_CHANNELS * (double)ANE_BG_BLOCK;
    const double wvMACs = (double)ANE_BG_GLU * (double)ANE_BG_HIDDEN * (double)ANE_BG_SEQ;
    const double wgMACs = (double)ANE_BG_GLU * (double)ANE_BG_HIDDEN * (double)ANE_BG_SEQ;
    const double woMACs = (double)ANE_BG_HIDDEN * (double)ANE_BG_GLU * (double)ANE_BG_SEQ;
    const double blockMACs = localMACs + globalMACs + wvMACs + wgMACs + woMACs;
    const double headMACs = (double)ANE_BG_VOCAB * (double)ANE_BG_HIDDEN * (double)ANE_BG_SEQ;
    return stemMACs + ((double)ANE_BG_LAYERS * blockMACs) + headMACs;
}

static double train_ticks_to_ms(uint64_t ticks) {
    static mach_timebase_info_data_t timebase;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
      mach_timebase_info(&timebase);
    });
    return ((double)ticks * (double)timebase.numer / (double)timebase.denom) / 1e6;
}

static int train_env_int(NSString *name, int fallback) {
    NSString *value = [NSProcessInfo.processInfo.environment objectForKey:name];
    if (value.length == 0) {
        return fallback;
    }
    NSInteger parsed = value.integerValue;
    return parsed > 0 ? (int)parsed : fallback;
}

static float train_env_float(NSString *name, float fallback) {
    NSString *value = [NSProcessInfo.processInfo.environment objectForKey:name];
    if (value.length == 0) {
        return fallback;
    }
    return value.floatValue;
}

static NSString *train_env_string(NSString *name, NSString *fallback) {
    NSString *value = [NSProcessInfo.processInfo.environment objectForKey:name];
    if (value.length == 0) {
        return fallback;
    }
    return value;
}

static float train_alpha_depth_factor(NSInteger layer) {
    const float depthPower = train_env_float(@"ANE_BG_ALPHA_DEPTH_POWER", 0.0f);
    const float depth = (float)(layer + 1) / (float)ANE_BG_LAYERS;
    return powf(depth, depthPower);
}

static uint8_t train_pattern_byte(NSUInteger token, uint32_t seed);

static BOOL train_env_bool(NSString *name, BOOL fallback) {
    NSString *value = [NSProcessInfo.processInfo.environment objectForKey:name];
    if (value.length == 0) {
        return fallback;
    }
    NSString *lower = value.lowercaseString;
    if ([lower isEqualToString:@"0"] || [lower isEqualToString:@"false"] || [lower isEqualToString:@"no"]) {
        return NO;
    }
    if ([lower isEqualToString:@"1"] || [lower isEqualToString:@"true"] || [lower isEqualToString:@"yes"]) {
        return YES;
    }
    return fallback;
}

static float train_fp16_to_float(uint16_t h) {
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

static BOOL train_compute_tensor_stats(IOSurfaceRef surface, float saturationThreshold, TrainTensorStats *outStats) {
    if (!outStats) {
        return NO;
    }
    size_t count = 0;
    uint16_t *buf = ane_bg_lock_surface_fp16(surface, &count);
    if (!buf || count == 0) {
        ane_bg_unlock_surface(surface);
        return NO;
    }

    size_t nanInfCount = 0;
    size_t saturatedCount = 0;
    double sum = 0.0;
    double sqSum = 0.0;
    for (size_t i = 0; i < count; ++i) {
        float value = train_fp16_to_float(buf[i]);
        if (!isfinite(value)) {
            nanInfCount++;
            continue;
        }
        if (fabsf(value) >= saturationThreshold) {
            saturatedCount++;
        }
        sum += (double)value;
        sqSum += (double)value * (double)value;
    }
    ane_bg_unlock_surface(surface);

    double finiteCount = (double)(count - nanInfCount);
    double mean = finiteCount > 0.0 ? (sum / finiteCount) : 0.0;
    double variance = finiteCount > 0.0 ? (sqSum / finiteCount) - (mean * mean) : 0.0;
    if (variance < 0.0) {
        variance = 0.0;
    }

    outStats->nanInfCount = nanInfCount;
    outStats->saturationRate = count > 0 ? ((double)saturatedCount / (double)count) : 0.0;
    outStats->mean = mean;
    outStats->variance = variance;
    return YES;
}

static void train_log_surface_stats(NSString *label, IOSurfaceRef surface, uint32_t channels, uint32_t seq) {
    size_t count = 0;
    uint16_t *buf = ane_bg_lock_surface_fp16(surface, &count);
    if (!buf || count == 0) {
        NSLog(@"%@ stats unavailable", label);
        ane_bg_unlock_surface(surface);
        return;
    }

    float minValue = INFINITY;
    float maxValue = -INFINITY;
    double sum = 0.0;
    size_t nonZero = 0;
    for (size_t i = 0; i < count; ++i) {
        float value = train_fp16_to_float(buf[i]);
        if (value < minValue) minValue = value;
        if (value > maxValue) maxValue = value;
        sum += value;
        if (fabsf(value) > 1e-6f) {
            nonZero++;
        }
    }
    double mean = sum / (double)count;

    NSMutableString *sample = [NSMutableString string];
    const uint32_t sampleCount = MIN((uint32_t)8, channels);
    for (uint32_t c = 0; c < sampleCount; ++c) {
        float value = train_fp16_to_float(buf[c * seq]);
        [sample appendFormat:@"%s%.4f", c == 0 ? "" : ", ", value];
    }
    ane_bg_unlock_surface(surface);

    NSLog(@"%@ stats: elems=%zu nz=%zu min=%.4f max=%.4f mean=%.4f token0[0:%u]=[%@]",
          label,
          count,
          nonZero,
          minValue,
          maxValue,
          mean,
          sampleCount,
          sample);
}

static void train_log_output_sanity(NSString *label, IOSurfaceRef output, uint32_t targetSeed) {
    size_t count = 0;
    uint16_t *buf = ane_bg_lock_surface_fp16(output, &count);
    if (!buf || count < (size_t)(ANE_BG_VOCAB * ANE_BG_SEQ)) {
        NSLog(@"output sanity unavailable");
        ane_bg_unlock_surface(output);
        return;
    }

    NSUInteger correct = 0;
    double targetLogitSum = 0.0;
    double bestLogitSum = 0.0;
    NSMutableArray<NSString *> *samples = [NSMutableArray array];
    const uint32_t samplePositions = MIN((uint32_t)8, (uint32_t)(ANE_BG_SEQ - 1));
    for (uint32_t t = 0; t + 1 < ANE_BG_SEQ; ++t) {
        const uint8_t target = train_pattern_byte(t + 1, targetSeed);
        float bestValue = -INFINITY;
        uint32_t bestIndex = 0;
        for (uint32_t c = 0; c < ANE_BG_VOCAB; ++c) {
            float value = train_fp16_to_float(buf[c * ANE_BG_SEQ + t]);
            if (value > bestValue) {
                bestValue = value;
                bestIndex = c;
            }
        }
        float targetValue = train_fp16_to_float(buf[target * ANE_BG_SEQ + t]);
        targetLogitSum += targetValue;
        bestLogitSum += bestValue;
        if (bestIndex == target) {
            correct++;
        }
        if (t < samplePositions) {
            [samples addObject:[NSString stringWithFormat:@"t%u target=%u pred=%u target_logit=%.3f best_logit=%.3f",
                                t, (unsigned)target, (unsigned)bestIndex, targetValue, bestValue]];
        }
    }
    ane_bg_unlock_surface(output);

    const double positions = (double)(ANE_BG_SEQ - 1);
    NSLog(@"%@ sanity: top1=%.2f%% mean_target_logit=%.4f mean_best_logit=%.4f",
          label,
          positions > 0.0 ? (100.0 * (double)correct / positions) : 0.0,
          positions > 0.0 ? targetLogitSum / positions : 0.0,
          positions > 0.0 ? bestLogitSum / positions : 0.0);
    NSLog(@"%@ sample: %@", label, [samples componentsJoinedByString:@" | "]);
}

static double train_compute_output_loss(IOSurfaceRef output, uint32_t targetSeed, double *meanMarginOut) {
    size_t count = 0;
    uint16_t *buf = ane_bg_lock_surface_fp16(output, &count);
    if (!buf || count < (size_t)(ANE_BG_VOCAB * ANE_BG_SEQ)) {
        ane_bg_unlock_surface(output);
        if (meanMarginOut) {
            *meanMarginOut = 0.0;
        }
        return NAN;
    }

    double lossSum = 0.0;
    double marginSum = 0.0;
    const uint32_t positions = ANE_BG_SEQ - 1u;
    for (uint32_t t = 0; t < positions; ++t) {
        const uint8_t target = train_pattern_byte(t + 1u, targetSeed);
        float maxLogit = -INFINITY;
        for (uint32_t c = 0; c < ANE_BG_VOCAB; ++c) {
            float value = train_fp16_to_float(buf[c * ANE_BG_SEQ + t]);
            if (value > maxLogit) {
                maxLogit = value;
            }
        }

        double expSum = 0.0;
        float targetLogit = 0.0f;
        for (uint32_t c = 0; c < ANE_BG_VOCAB; ++c) {
            float value = train_fp16_to_float(buf[c * ANE_BG_SEQ + t]);
            if (c == target) {
                targetLogit = value;
            }
            expSum += exp((double)value - (double)maxLogit);
        }

        const double logSumExp = (double)maxLogit + log(expSum);
        lossSum += logSumExp - (double)targetLogit;
        marginSum += (double)maxLogit - (double)targetLogit;
    }
    ane_bg_unlock_surface(output);

    const double meanLoss = positions > 0u ? lossSum / (double)positions : 0.0;
    const double meanMargin = positions > 0u ? marginSum / (double)positions : 0.0;
    if (meanMarginOut) {
        *meanMarginOut = meanMargin;
    }
    return meanLoss;
}

static void train_log_output_loss(NSString *label, IOSurfaceRef output, uint32_t targetSeed, double meanLoss) {
    if (!isfinite(meanLoss)) {
        NSLog(@"output loss unavailable");
        return;
    }
    const double perplexity = exp(meanLoss);
    double meanMargin = 0.0;
    (void)train_compute_output_loss(output, targetSeed, &meanMargin);
    const double uniformLoss = log((double)ANE_BG_VOCAB);
    const double uniformPerplexity = (double)ANE_BG_VOCAB;
    const double lossDelta = meanLoss - uniformLoss;
    const double perplexityRatio = uniformPerplexity > 0.0 ? (perplexity / uniformPerplexity) : 0.0;
    NSLog(@"%@ loss: cross_entropy=%.4f (uniform=%.4f delta=%+.4f) perplexity=%.2f (uniform=%.2f ratio=%.3f) mean_margin=%.4f",
          label,
          meanLoss,
          uniformLoss,
          lossDelta,
          perplexity,
          uniformPerplexity,
          perplexityRatio,
          meanMargin);
}

static uint8_t train_pattern_byte(NSUInteger token, uint32_t seed) {
    static const char *kAlphabet = "ane-bytegrid-44m|";
    const size_t alphabetLen = strlen(kAlphabet);
    uint8_t base = (uint8_t)kAlphabet[(token + (NSUInteger)(seed % alphabetLen)) % alphabetLen];
    uint32_t mix = (uint32_t)token * 29u;
    mix ^= (seed * 131u) + (seed >> 1);
    return (uint8_t)(base ^ (uint8_t)(mix & 0x3fu));
}

static uint32_t train_byte_class(uint8_t byte) {
    if (byte >= 'a' && byte <= 'z') return 0;
    if (byte >= 'A' && byte <= 'Z') return 1;
    if (byte >= '0' && byte <= '9') return 2;
    if (byte == ' ') return 3;
    if (byte == '\n') return 4;
    if (byte == '\t') return 5;
    if (byte == '-' || byte == '_') return 6;
    if (byte == '|' || byte == '/' || byte == '\\') return 7;
    if (byte == '.' || byte == ',' || byte == ';' || byte == ':') return 8;
    if (byte == '(' || byte == ')' || byte == '[' || byte == ']' || byte == '{' || byte == '}') return 9;
    if (byte == '"' || byte == '\'') return 10;
    if (byte < 0x20) return 11;
    if (byte < 0x80) return 12;
    if ((byte & 0xe0u) == 0xc0u) return 13;
    if ((byte & 0xc0u) == 0x80u) return 14;
    return 15;
}

static void train_fill_input_surface(IOSurfaceRef surface, uint32_t seed) {
    size_t count = 0;
    uint16_t *buf = ane_bg_lock_surface_fp16(surface, &count);
    if (!buf) {
        return;
    }
    memset(buf, 0, count * sizeof(uint16_t));

    const uint16_t one = ane_bg_fp16_from_float(1.0f);
    const uint16_t negOne = ane_bg_fp16_from_float(-1.0f);

    for (uint32_t t = 0; t < ANE_BG_SEQ; ++t) {
        const uint8_t byte = train_pattern_byte(t, seed);
        const uint32_t byteOffset = 0;
        const uint32_t classOffset = ANE_BG_BYTE_CHANNELS;
        const uint32_t posOffset = classOffset + ANE_BG_CLASS_CHANNELS;
        const uint32_t ctrlOffset = posOffset + ANE_BG_POS_CHANNELS;

        buf[(byteOffset + byte) * ANE_BG_SEQ + t] = one;

        const uint32_t cls = train_byte_class(byte);
        buf[(classOffset + cls) * ANE_BG_SEQ + t] = one;

        for (uint32_t i = 0; i < ANE_BG_POS_CHANNELS / 2; ++i) {
            const uint32_t period = 1u << (i + 1u);
            const BOOL bit = ((t / period) & 1u) != 0;
            buf[(posOffset + (2u * i)) * ANE_BG_SEQ + t] = bit ? one : negOne;
            buf[(posOffset + (2u * i) + 1u) * ANE_BG_SEQ + t] = bit ? negOne : one;
        }

        if (t == 0) {
            buf[(ctrlOffset + 0u) * ANE_BG_SEQ + t] = one;
        }
        if (t == ANE_BG_SEQ - 1u) {
            buf[(ctrlOffset + 1u) * ANE_BG_SEQ + t] = one;
        }
        if ((t % 16u) == 0u) {
            buf[(ctrlOffset + 2u) * ANE_BG_SEQ + t] = one;
        }
        if ((t % 32u) < 16u) {
            buf[(ctrlOffset + 3u) * ANE_BG_SEQ + t] = one;
        }
        if ((byte & 1u) != 0u) {
            buf[(ctrlOffset + 4u) * ANE_BG_SEQ + t] = one;
        }
        if (byte >= '0' && byte <= '9') {
            buf[(ctrlOffset + 5u) * ANE_BG_SEQ + t] = one;
        }
        if (byte >= 'a' && byte <= 'z') {
            buf[(ctrlOffset + 6u) * ANE_BG_SEQ + t] = one;
        }
        if (byte >= 'A' && byte <= 'Z') {
            buf[(ctrlOffset + 7u) * ANE_BG_SEQ + t] = one;
        }
        if (byte == ' ' || byte == '\n' || byte == '\t') {
            buf[(ctrlOffset + 8u) * ANE_BG_SEQ + t] = one;
        }
        if (byte == '|' || byte == '-' || byte == '_') {
            buf[(ctrlOffset + 9u) * ANE_BG_SEQ + t] = one;
        }
    }
    ane_bg_unlock_surface(surface);
}

static ANEByteGridRuntime *compile_runtime_or_log(NSString *label,
                                                  NSString *mil,
                                                  NSString *weightDir,
                                                  double *compileMs,
                                                  double *loadMs,
                                                  NSError **error) {
    ANEByteGridRuntime *runtime = [[ANEByteGridRuntime alloc] init];
    if (![runtime compileMIL:mil weightDirectory:weightDir error:error]) {
        NSLog(@"%@ compile failed: %@", label, (*error).localizedDescription);
        return nil;
    }
    *compileMs += runtime.lastCompileDurationMs;
    *loadMs += runtime.lastLoadDurationMs;
    return runtime;
}

static BOOL train_append_line(NSString *path, NSString *line, NSError **error) {
    NSData *data = [line dataUsingEncoding:NSUTF8StringEncoding];
    if (!data) {
        return NO;
    }
    NSFileManager *fm = NSFileManager.defaultManager;
    if (![fm fileExistsAtPath:path]) {
        return [data writeToFile:path options:NSDataWritingAtomic error:error];
    }
    NSFileHandle *handle = [NSFileHandle fileHandleForWritingAtPath:path];
    if (!handle) {
        if (error) {
            *error = [NSError errorWithDomain:@"ANEByteGrid"
                                         code:122
                                     userInfo:@{NSLocalizedDescriptionKey: [NSString stringWithFormat:@"Failed opening log file %@", path]}];
        }
        return NO;
    }
    @try {
        [handle seekToEndOfFile];
        [handle writeData:data];
    } @catch (NSException *exception) {
        if (error) {
            *error = [NSError errorWithDomain:@"ANEByteGrid"
                                         code:123
                                     userInfo:@{NSLocalizedDescriptionKey: [NSString stringWithFormat:@"Failed appending to %@", path]}];
        }
        [handle closeFile];
        return NO;
    }
    [handle closeFile];
    return YES;
}

static void train_append_experiment_logs(NSString *root,
                                         uint32_t trainSeed,
                                         uint32_t evalSeed,
                                         double compileMs,
                                         double loadMs,
                                         double avgMs,
                                         double throughputMBps,
                                         double passesPerSecond,
                                         double tokensPerSecond,
                                         double estConvGMACPerPass,
                                         double estConvTFLOPS,
                                         TrainSplitMetrics trainMetrics,
                                         TrainSplitMetrics evalMetrics,
                                         BOOL overallStable) {
    NSString *jsonlPath = train_env_string(@"ANE_BG_EXPERIMENT_JSONL", [root stringByAppendingPathComponent:@"build/experiments.jsonl"]);
    NSString *tsvPath = train_env_string(@"ANE_BG_EXPERIMENT_TSV", [root stringByAppendingPathComponent:@"build/experiments.tsv"]);
    NSFileManager *fm = NSFileManager.defaultManager;
    [fm createDirectoryAtPath:[jsonlPath stringByDeletingLastPathComponent] withIntermediateDirectories:YES attributes:nil error:nil];
    [fm createDirectoryAtPath:[tsvPath stringByDeletingLastPathComponent] withIntermediateDirectories:YES attributes:nil error:nil];

    NSDate *now = [NSDate date];
    NSString *timestamp = [NSString stringWithFormat:@"%.3f", now.timeIntervalSince1970];
    NSDictionary *record = @{
        @"timestamp": timestamp,
        @"train_seed": @(trainSeed),
        @"eval_seed": @(evalSeed),
        @"local_mul": @(train_env_float(@"ANE_BG_ALPHA_LOCAL_MUL", 1.15f)),
        @"global_mul": @(train_env_float(@"ANE_BG_ALPHA_GLOBAL_MUL", 1.40f)),
        @"mlp_mul": @(train_env_float(@"ANE_BG_ALPHA_MLP_MUL", 0.05f)),
        @"depth_power": @(train_env_float(@"ANE_BG_ALPHA_DEPTH_POWER", 0.0f)),
        @"head_weight_scale": @(train_env_float(@"ANE_BG_HEAD_WEIGHT_SCALE", 0.98f)),
        @"head_logit_scale": @(train_env_float(@"ANE_BG_HEAD_LOGIT_SCALE", 0.97f)),
        @"compile_ms": @(compileMs),
        @"load_ms": @(loadMs),
        @"avg_ms": @(avgMs),
        @"throughput_mb_s": @(throughputMBps),
        @"passes_per_s": @(passesPerSecond),
        @"tokens_per_s": @(tokensPerSecond),
        @"est_conv_gmac_per_pass": @(estConvGMACPerPass),
        @"est_conv_tflops": @(estConvTFLOPS),
        @"train_ce": @(trainMetrics.loss),
        @"eval_ce": @(evalMetrics.loss),
        @"train_ppx": @(trainMetrics.perplexity),
        @"eval_ppx": @(evalMetrics.perplexity),
        @"ce_gap_eval_minus_train": @(evalMetrics.loss - trainMetrics.loss),
        @"train_margin": @(trainMetrics.meanMargin),
        @"eval_margin": @(evalMetrics.meanMargin),
        @"stable": @(overallStable)
    };
    NSError *jsonError = nil;
    NSData *jsonData = [NSJSONSerialization dataWithJSONObject:record options:0 error:&jsonError];
    if (!jsonData || jsonError) {
        NSLog(@"experiment log json serialization failed: %@", jsonError.localizedDescription);
    } else {
        NSString *jsonLine = [[NSString alloc] initWithData:jsonData encoding:NSUTF8StringEncoding];
        NSError *appendError = nil;
        if (!train_append_line(jsonlPath, [jsonLine stringByAppendingString:@"\n"], &appendError)) {
            NSLog(@"experiment jsonl append failed: %@", appendError.localizedDescription);
        }
    }

    NSString *header = @"timestamp\ttrain_seed\teval_seed\tlocal_mul\tglobal_mul\tmlp_mul\tdepth_power\thead_weight_scale\thead_logit_scale\tcompile_ms\tload_ms\tavg_ms\tthroughput_mb_s\tpasses_per_s\ttokens_per_s\test_conv_gmac_per_pass\test_conv_tflops\ttrain_ce\teval_ce\ttrain_ppx\teval_ppx\tce_gap_eval_minus_train\ttrain_margin\teval_margin\tstable\n";
    if (![fm fileExistsAtPath:tsvPath]) {
        train_append_line(tsvPath, header, nil);
    }
    NSString *row = [NSString stringWithFormat:
                     @"%@\t%u\t%u\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.3f\t%.3f\t%.3f\t%.2f\t%.2f\t%.2f\t%.3f\t%.3f\t%.4f\t%.4f\t%.2f\t%.2f\t%+.4f\t%.4f\t%.4f\t%@\n",
                     timestamp,
                     trainSeed,
                     evalSeed,
                     train_env_float(@"ANE_BG_ALPHA_LOCAL_MUL", 1.15f),
                     train_env_float(@"ANE_BG_ALPHA_GLOBAL_MUL", 1.40f),
                     train_env_float(@"ANE_BG_ALPHA_MLP_MUL", 0.05f),
                     train_env_float(@"ANE_BG_ALPHA_DEPTH_POWER", 0.0f),
                     train_env_float(@"ANE_BG_HEAD_WEIGHT_SCALE", 0.98f),
                     train_env_float(@"ANE_BG_HEAD_LOGIT_SCALE", 0.97f),
                     compileMs,
                     loadMs,
                     avgMs,
                     throughputMBps,
                     passesPerSecond,
                     tokensPerSecond,
                     estConvGMACPerPass,
                     estConvTFLOPS,
                     trainMetrics.loss,
                     evalMetrics.loss,
                     trainMetrics.perplexity,
                     evalMetrics.perplexity,
                     evalMetrics.loss - trainMetrics.loss,
                     trainMetrics.meanMargin,
                     evalMetrics.meanMargin,
                     overallStable ? @"YES" : @"NO"];
    NSError *tsvError = nil;
    if (!train_append_line(tsvPath, row, &tsvError)) {
        NSLog(@"experiment tsv append failed: %@", tsvError.localizedDescription);
    }
}

static BOOL run_staged_pipeline(NSArray<ANEByteGridRuntime *> *pipeline,
                                IOSurfaceRef input,
                                IOSurfaceRef hiddenA,
                                IOSurfaceRef hiddenB,
                                IOSurfaceRef output,
                                NSError **error) {
    if (pipeline.count < 2) {
        if (error) {
            *error = [NSError errorWithDomain:@"ANEByteGrid"
                                         code:100
                                     userInfo:@{NSLocalizedDescriptionKey: @"Pipeline is incomplete."}];
        }
        return NO;
    }

    ANEByteGridRuntime *stem = pipeline.firstObject;
    if (![stem evaluateInputSurface:input outputSurface:hiddenA error:error]) {
        return NO;
    }

    IOSurfaceRef current = hiddenA;
    IOSurfaceRef next = hiddenB;
    for (NSUInteger i = 1; i + 1 < pipeline.count; ++i) {
        if (![pipeline[i] evaluateInputSurface:current outputSurface:next error:error]) {
            return NO;
        }
        IOSurfaceRef tmp = current;
        current = next;
        next = tmp;
    }

    return [[pipeline lastObject] evaluateInputSurface:current outputSurface:output error:error];
}

static int train_run_once(NSString *root, BOOL logSurfaceStats, double *lossOut, TrainRunMetrics *metricsOut) {
    NSString *weightDir = [root stringByAppendingPathComponent:@"weights"];
    const float rmsInit = train_env_float(@"ANE_BG_RMS_INIT", 1.0f);
    const float alphaBase = train_env_float(@"ANE_BG_ALPHA_INIT", 1.0f / sqrtf((float)ANE_BG_LAYERS));
    const float alphaLocalMul = train_env_float(@"ANE_BG_ALPHA_LOCAL_MUL", 1.15f);
    const float alphaGlobalMul = train_env_float(@"ANE_BG_ALPHA_GLOBAL_MUL", 1.40f);
    const float alphaMLPMul = train_env_float(@"ANE_BG_ALPHA_MLP_MUL", 0.05f);
    const float alphaDepthPower = train_env_float(@"ANE_BG_ALPHA_DEPTH_POWER", 0.0f);
    const float headWeightScale = train_env_float(@"ANE_BG_HEAD_WEIGHT_SCALE", 0.98f);
    const float headLogitScale = train_env_float(@"ANE_BG_HEAD_LOGIT_SCALE", 0.97f);
    const float firstDepth = train_alpha_depth_factor(0);
    const float lastDepth = train_alpha_depth_factor(ANE_BG_LAYERS - 1);
    NSError *error = nil;
    if (!ane_bg_generate_all_weight_blobs(weightDir, &error)) {
        NSLog(@"weight generation failed: %@", error);
        return 1;
    }
    NSLog(@"ANE init profile: rms=%.5f alpha_base=%.5f local_mul=%.2f global_mul=%.2f mlp_mul=%.2f depth_power=%.2f head_weight_scale=%.3f head_logit_scale=%.3f first_depth=%.5f last_depth=%.5f",
          rmsInit,
          alphaBase,
          alphaLocalMul,
          alphaGlobalMul,
          alphaMLPMul,
          alphaDepthPower,
          headWeightScale,
          headLogitScale,
          firstDepth,
          lastDepth);

    NSString *mil = ane_bg_gen_full_model_mil(weightDir);
    NSString *milPath = [root stringByAppendingPathComponent:@"build/bytegrid_44m.mil"];
    [mil writeToFile:milPath atomically:YES encoding:NSUTF8StringEncoding error:&error];
    if (error) {
        NSLog(@"failed to write MIL: %@", error);
        return 1;
    }

    NSMutableArray<ANEByteGridRuntime *> *pipeline = [NSMutableArray arrayWithCapacity:26];
    double compileMs = 0.0;
    double loadMs = 0.0;

    ANEByteGridRuntime *stem = compile_runtime_or_log(@"stem",
                                                      ane_bg_gen_stem_mil([weightDir stringByAppendingPathComponent:@"stem.bin"]),
                                                      weightDir,
                                                      &compileMs,
                                                      &loadMs,
                                                      &error);
    if (!stem) {
        return 1;
    }
    [pipeline addObject:stem];

    for (NSInteger layer = 0; layer < ANE_BG_LAYERS; ++layer) {
        NSString *label = [NSString stringWithFormat:@"block_%02ld", (long)layer];
        ANEByteGridRuntime *block = compile_runtime_or_log(label,
                                                           ane_bg_gen_block_mil(weightDir, layer),
                                                           weightDir,
                                                           &compileMs,
                                                           &loadMs,
                                                           &error);
        if (!block) {
            return 1;
        }
        [pipeline addObject:block];
    }

    ANEByteGridRuntime *head = compile_runtime_or_log(@"head",
                                                      ane_bg_gen_head_mil([weightDir stringByAppendingPathComponent:@"head.bin"]),
                                                      weightDir,
                                                      &compileMs,
                                                      &loadMs,
                                                      &error);
    if (!head) {
        return 1;
    }
    [pipeline addObject:head];
    NSLog(@"ANE pipeline ready: kernels=%lu compile=%.2f ms load=%.2f ms",
          (unsigned long)pipeline.count, compileMs, loadMs);

    IOSurfaceRef input = ane_bg_create_surface(ANE_BG_INPUT_CHANNELS, ANE_BG_SEQ);
    IOSurfaceRef hiddenA = ane_bg_create_surface(ANE_BG_HIDDEN, ANE_BG_SEQ);
    IOSurfaceRef hiddenB = ane_bg_create_surface(ANE_BG_HIDDEN, ANE_BG_SEQ);
    IOSurfaceRef output = ane_bg_create_surface(ANE_BG_VOCAB, ANE_BG_SEQ);
    if (!input || !hiddenA || !hiddenB || !output) {
        NSLog(@"failed to allocate IOSurface tensors");
        if (input) CFRelease(input);
        if (hiddenA) CFRelease(hiddenA);
        if (hiddenB) CFRelease(hiddenB);
        if (output) CFRelease(output);
        return 1;
    }

    const int trainSeed = train_env_int(@"ANE_BG_TRAIN_SEED", 0);
    const int evalSeed = train_env_int(@"ANE_BG_EVAL_SEED", 1);

    train_fill_input_surface(input, (uint32_t)trainSeed);
    if (logSurfaceStats) {
        train_log_surface_stats(@"train input", input, ANE_BG_INPUT_CHANNELS, ANE_BG_SEQ);
    }

    if (!run_staged_pipeline(pipeline, input, hiddenA, hiddenB, output, &error)) {
        NSLog(@"ANE staged evaluate failed: %@", error.localizedDescription);
        CFRelease(input);
        CFRelease(hiddenA);
        CFRelease(hiddenB);
        CFRelease(output);
        return 1;
    }

    TrainSplitMetrics trainMetrics = {0};
    trainMetrics.hasHidden = train_compute_tensor_stats(hiddenA, 6.0f, &trainMetrics.hidden);
    trainMetrics.hasOutput = train_compute_tensor_stats(output, 6.0f, &trainMetrics.output);
    trainMetrics.stable = trainMetrics.hasHidden &&
                          trainMetrics.hasOutput &&
                          trainMetrics.hidden.nanInfCount == 0 &&
                          trainMetrics.output.nanInfCount == 0 &&
                          trainMetrics.hidden.saturationRate < 0.02 &&
                          trainMetrics.output.saturationRate < 0.02 &&
                          trainMetrics.hidden.variance > 1e-5 &&
                          trainMetrics.output.variance > 1e-5;
    trainMetrics.loss = train_compute_output_loss(output, (uint32_t)trainSeed, &trainMetrics.meanMargin);
    trainMetrics.perplexity = isfinite(trainMetrics.loss) ? exp(trainMetrics.loss) : NAN;
    if (logSurfaceStats) {
        train_log_surface_stats(@"train hidden", hiddenA, ANE_BG_HIDDEN, ANE_BG_SEQ);
        train_log_surface_stats(@"train output", output, ANE_BG_VOCAB, ANE_BG_SEQ);
        train_log_output_sanity(@"train output", output, (uint32_t)trainSeed);
        train_log_output_loss(@"output", output, (uint32_t)trainSeed, trainMetrics.loss);
    }

    train_fill_input_surface(input, (uint32_t)evalSeed);
    if (logSurfaceStats) {
        train_log_surface_stats(@"eval input", input, ANE_BG_INPUT_CHANNELS, ANE_BG_SEQ);
    }
    if (!run_staged_pipeline(pipeline, input, hiddenA, hiddenB, output, &error)) {
        NSLog(@"ANE eval-split staged evaluate failed: %@", error.localizedDescription);
        CFRelease(input);
        CFRelease(hiddenA);
        CFRelease(hiddenB);
        CFRelease(output);
        return 1;
    }
    TrainSplitMetrics evalMetrics = {0};
    evalMetrics.hasHidden = train_compute_tensor_stats(hiddenA, 6.0f, &evalMetrics.hidden);
    evalMetrics.hasOutput = train_compute_tensor_stats(output, 6.0f, &evalMetrics.output);
    evalMetrics.stable = evalMetrics.hasHidden &&
                         evalMetrics.hasOutput &&
                         evalMetrics.hidden.nanInfCount == 0 &&
                         evalMetrics.output.nanInfCount == 0 &&
                         evalMetrics.hidden.saturationRate < 0.02 &&
                         evalMetrics.output.saturationRate < 0.02 &&
                         evalMetrics.hidden.variance > 1e-5 &&
                         evalMetrics.output.variance > 1e-5;
    evalMetrics.loss = train_compute_output_loss(output, (uint32_t)evalSeed, &evalMetrics.meanMargin);
    evalMetrics.perplexity = isfinite(evalMetrics.loss) ? exp(evalMetrics.loss) : NAN;
    if (logSurfaceStats) {
        train_log_surface_stats(@"eval hidden", hiddenA, ANE_BG_HIDDEN, ANE_BG_SEQ);
        train_log_surface_stats(@"eval output", output, ANE_BG_VOCAB, ANE_BG_SEQ);
        train_log_output_sanity(@"eval output", output, (uint32_t)evalSeed);
        train_log_output_loss(@"eval", output, (uint32_t)evalSeed, evalMetrics.loss);
    }

    TrainTensorStats combinedHidden = {
        .nanInfCount = trainMetrics.hidden.nanInfCount + evalMetrics.hidden.nanInfCount,
        .saturationRate = MAX(trainMetrics.hidden.saturationRate, evalMetrics.hidden.saturationRate),
        .mean = 0.5 * (trainMetrics.hidden.mean + evalMetrics.hidden.mean),
        .variance = MIN(trainMetrics.hidden.variance, evalMetrics.hidden.variance)
    };
    TrainTensorStats combinedOutput = {
        .nanInfCount = trainMetrics.output.nanInfCount + evalMetrics.output.nanInfCount,
        .saturationRate = MAX(trainMetrics.output.saturationRate, evalMetrics.output.saturationRate),
        .mean = 0.5 * (trainMetrics.output.mean + evalMetrics.output.mean),
        .variance = MIN(trainMetrics.output.variance, evalMetrics.output.variance)
    };
    const BOOL guardrailsStable = trainMetrics.stable && evalMetrics.stable;
    NSLog(@"guardrails: hidden_naninf=%zu hidden_sat=%.6f hidden_var=%.6f output_naninf=%zu output_sat=%.6f output_var=%.6f stable=%@",
          combinedHidden.nanInfCount,
          combinedHidden.saturationRate,
          combinedHidden.variance,
          combinedOutput.nanInfCount,
          combinedOutput.saturationRate,
          combinedOutput.variance,
          guardrailsStable ? @"YES" : @"NO");
    NSLog(@"split loss: train_ce=%.4f eval_ce=%.4f gap_eval_minus_train=%+.4f train_ppx=%.2f eval_ppx=%.2f",
          trainMetrics.loss,
          evalMetrics.loss,
          evalMetrics.loss - trainMetrics.loss,
          trainMetrics.perplexity,
          evalMetrics.perplexity);

    if (lossOut) {
        *lossOut = trainMetrics.loss;
    }

    const int warmup = train_env_int(@"ANE_BG_WARMUP", 1);
    const int iters = train_env_int(@"ANE_BG_ITERS", 3);
    train_fill_input_surface(input, (uint32_t)trainSeed);
    for (int i = 0; i < warmup; ++i) {
        if (!run_staged_pipeline(pipeline, input, hiddenA, hiddenB, output, &error)) {
            NSLog(@"ANE warmup failed: %@", error.localizedDescription);
            CFRelease(input);
            CFRelease(hiddenA);
            CFRelease(hiddenB);
            CFRelease(output);
            return 1;
        }
    }

    uint64_t t0 = mach_absolute_time();
    for (int i = 0; i < iters; ++i) {
        if (!run_staged_pipeline(pipeline, input, hiddenA, hiddenB, output, &error)) {
            NSLog(@"ANE timed run failed: %@", error.localizedDescription);
            CFRelease(input);
            CFRelease(hiddenA);
            CFRelease(hiddenB);
            CFRelease(output);
            return 1;
        }
    }
    double avgMs = train_ticks_to_ms(mach_absolute_time() - t0) / (double)iters;
    double passesPerSecond = avgMs > 0.0 ? 1000.0 / avgMs : 0.0;
    double tokensPerSecond = passesPerSecond * (double)ANE_BG_SEQ;
    double hiddenBytes = (double)ane_bg_tensor_bytes(ANE_BG_HIDDEN, ANE_BG_SEQ);
    double totalBytes = (double)ane_bg_tensor_bytes(ANE_BG_INPUT_CHANNELS, ANE_BG_SEQ) + hiddenBytes;
    totalBytes += (double)ANE_BG_LAYERS * hiddenBytes * 2.0;
    totalBytes += hiddenBytes + (double)ane_bg_tensor_bytes(ANE_BG_VOCAB, ANE_BG_SEQ);
    double throughputMBps = avgMs > 0.0 ? (totalBytes / (1024.0 * 1024.0)) / (avgMs / 1000.0) : 0.0;
    const double estConvMACsPerPass = train_estimated_conv_macs_per_pass();
    const double estConvGMACPerPass = estConvMACsPerPass / 1e9;
    const double estConvTFLOPS = passesPerSecond > 0.0 ? ((estConvMACsPerPass * 2.0 * passesPerSecond) / 1e12) : 0.0;
    NSLog(@"ANE staged evaluate succeeded: warmup=%d iters=%d avg=%.3f ms/pass throughput=%.2f MB/s passes/s=%.2f tokens/s=%.0f est_conv=%.3f GMAC/pass est_tflops=%.3f kernels=%lu",
          warmup,
          iters,
          avgMs,
          throughputMBps,
          passesPerSecond,
          tokensPerSecond,
          estConvGMACPerPass,
          estConvTFLOPS,
          (unsigned long)pipeline.count);
    if (train_env_bool(@"ANE_BG_EXPERIMENT_LOG", YES)) {
        train_append_experiment_logs(root,
                                     (uint32_t)trainSeed,
                                     (uint32_t)evalSeed,
                                     compileMs,
                                     loadMs,
                                     avgMs,
                                     throughputMBps,
                                     passesPerSecond,
                                     tokensPerSecond,
                                     estConvGMACPerPass,
                                     estConvTFLOPS,
                                     trainMetrics,
                                     evalMetrics,
                                     guardrailsStable);
    }
    if (metricsOut) {
        metricsOut->train = trainMetrics;
        metricsOut->eval = evalMetrics;
        metricsOut->stable = guardrailsStable;
        metricsOut->compileMs = compileMs;
        metricsOut->loadMs = loadMs;
        metricsOut->avgMs = avgMs;
        metricsOut->throughputMBps = throughputMBps;
        metricsOut->passesPerSecond = passesPerSecond;
        metricsOut->tokensPerSecond = tokensPerSecond;
        metricsOut->estConvGMACPerPass = estConvGMACPerPass;
        metricsOut->estConvTFLOPS = estConvTFLOPS;
    }

    CFRelease(input);
    CFRelease(hiddenA);
    CFRelease(hiddenB);
    CFRelease(output);
    NSLog(@"training skeleton completed on ANE private runtime with staged kernels");
    return 0;
}

static double train_alpha_objective(TrainRunMetrics metrics,
                                    float localMul,
                                    float globalMul,
                                    float mlpMul,
                                    double gapWeight,
                                    double speedWeight,
                                    double regWeight,
                                    float regLocalCenter,
                                    float regGlobalCenter,
                                    float regMLPCenter) {
    if (!metrics.stable || !isfinite(metrics.train.loss) || !isfinite(metrics.eval.loss) || !isfinite(metrics.avgMs)) {
        return INFINITY;
    }
    const double gap = fabs(metrics.eval.loss - metrics.train.loss);
    const double reg =
        ((double)localMul - (double)regLocalCenter) * ((double)localMul - (double)regLocalCenter) +
        ((double)globalMul - (double)regGlobalCenter) * ((double)globalMul - (double)regGlobalCenter) +
        ((double)mlpMul - (double)regMLPCenter) * ((double)mlpMul - (double)regMLPCenter);
    return metrics.eval.loss + (gapWeight * gap) + (speedWeight * metrics.avgMs) + (regWeight * reg);
}

static void train_set_env_float(const char *name, float value) {
    NSString *formatted = [NSString stringWithFormat:@"%.6f", value];
    setenv(name, formatted.UTF8String, 1);
}

static void train_append_alpha_opt_row(NSString *path,
                                       NSInteger step,
                                       float localMul,
                                       float globalMul,
                                       float mlpMul,
                                       float headWeightScale,
                                       float headLogitScale,
                                       TrainRunMetrics metrics,
                                       double objective,
                                       NSString *note) {
    NSFileManager *fm = NSFileManager.defaultManager;
    if (![fm fileExistsAtPath:path]) {
        NSString *header = @"step\tlocal_mul\tglobal_mul\tmlp_mul\thead_weight_scale\thead_logit_scale\ttrain_ce\teval_ce\tgap\ttrain_ppx\teval_ppx\tavg_ms\tthroughput_mb_s\tstable\tobjective\tnote\n";
        train_append_line(path, header, nil);
    }
    NSString *row = [NSString stringWithFormat:
                     @"%ld\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%+.4f\t%.2f\t%.2f\t%.3f\t%.2f\t%@\t%.6f\t%@\n",
                     (long)step,
                     localMul,
                     globalMul,
                     mlpMul,
                     headWeightScale,
                     headLogitScale,
                     metrics.train.loss,
                     metrics.eval.loss,
                     metrics.eval.loss - metrics.train.loss,
                     metrics.train.perplexity,
                     metrics.eval.perplexity,
                     metrics.avgMs,
                     metrics.throughputMBps,
                     metrics.stable ? @"YES" : @"NO",
                     objective,
                     note ?: @""];
    train_append_line(path, row, nil);
}

typedef struct {
    NSString *label;
    const char *envName;
    float value;
    float minValue;
    float maxValue;
} TrainUpdateParam;

static float train_clamp_value(float value, float minValue, float maxValue) {
    return MIN(maxValue, MAX(minValue, value));
}

static void train_append_update_row(NSString *path,
                                    NSInteger step,
                                    NSString *target,
                                    NSString *note,
                                    NSString *accepted,
                                    double objectiveBefore,
                                    double objectiveAfter,
                                    TrainRunMetrics metricsBefore,
                                    TrainRunMetrics metricsAfter,
                                    const TrainUpdateParam *paramsBefore,
                                    const TrainUpdateParam *paramsAfter,
                                    size_t paramCount) {
    NSFileManager *fm = NSFileManager.defaultManager;
    if (![fm fileExistsAtPath:path]) {
        NSString *header = @"step\ttarget\taccepted\tnote\tobj_before\tobj_after\ttrain_ce_before\teval_ce_before\tgap_before\tavg_ms_before\ttrain_ce_after\teval_ce_after\tgap_after\tavg_ms_after\tparams_before\tparams_after\n";
        train_append_line(path, header, nil);
    }
    NSMutableArray<NSString *> *beforePairs = [NSMutableArray array];
    NSMutableArray<NSString *> *afterPairs = [NSMutableArray array];
    for (size_t i = 0; i < paramCount; ++i) {
        [beforePairs addObject:[NSString stringWithFormat:@"%@=%.4f", paramsBefore[i].label, paramsBefore[i].value]];
    }
    for (size_t i = 0; i < paramCount; ++i) {
        [afterPairs addObject:[NSString stringWithFormat:@"%@=%.4f", paramsAfter[i].label, paramsAfter[i].value]];
    }
    NSString *row = [NSString stringWithFormat:
                     @"%ld\t%@\t%@\t%@\t%.6f\t%.6f\t%.4f\t%.4f\t%+.4f\t%.3f\t%.4f\t%.4f\t%+.4f\t%.3f\t%@\t%@\n",
                     (long)step,
                     target ?: @"",
                     accepted ?: @"NO",
                     note ?: @"",
                     objectiveBefore,
                     objectiveAfter,
                     metricsBefore.train.loss,
                     metricsBefore.eval.loss,
                     metricsBefore.eval.loss - metricsBefore.train.loss,
                     metricsBefore.avgMs,
                     metricsAfter.train.loss,
                     metricsAfter.eval.loss,
                     metricsAfter.eval.loss - metricsAfter.train.loss,
                     metricsAfter.avgMs,
                     [beforePairs componentsJoinedByString:@","],
                     [afterPairs componentsJoinedByString:@","]];
    train_append_line(path, row, nil);
}

static double train_objective_for_metrics(TrainRunMetrics metrics,
                                          double gapWeight,
                                          double speedWeight,
                                          double regWeight,
                                          float regLocalCenter,
                                          float regGlobalCenter,
                                          float regMLPCenter) {
    const float localMul = train_env_float(@"ANE_BG_ALPHA_LOCAL_MUL", 1.15f);
    const float globalMul = train_env_float(@"ANE_BG_ALPHA_GLOBAL_MUL", 1.40f);
    const float mlpMul = train_env_float(@"ANE_BG_ALPHA_MLP_MUL", 0.05f);
    return train_alpha_objective(metrics,
                                 localMul,
                                 globalMul,
                                 mlpMul,
                                 gapWeight,
                                 speedWeight,
                                 regWeight,
                                 regLocalCenter,
                                 regGlobalCenter,
                                 regMLPCenter);
}

static double train_eval_objective_current_env(NSString *root,
                                               int baseEvalSeed,
                                               int extraEvalSeed,
                                               double gapWeight,
                                               double speedWeight,
                                               double regWeight,
                                               float regLocalCenter,
                                               float regGlobalCenter,
                                               float regMLPCenter,
                                               TrainRunMetrics *metricsOut,
                                               BOOL *okOut) {
    double lossPrimary = NAN;
    TrainRunMetrics metricsPrimary = {0};
    int rc = train_run_once(root, NO, &lossPrimary, &metricsPrimary);
    if (rc != 0) {
        if (okOut) *okOut = NO;
        return INFINITY;
    }
    double objectivePrimary = train_objective_for_metrics(metricsPrimary,
                                                          gapWeight,
                                                          speedWeight,
                                                          regWeight,
                                                          regLocalCenter,
                                                          regGlobalCenter,
                                                          regMLPCenter);
    double objective = objectivePrimary;
    if (extraEvalSeed >= 0 && extraEvalSeed != baseEvalSeed) {
        NSString *oldEval = [NSProcessInfo.processInfo.environment objectForKey:@"ANE_BG_EVAL_SEED"];
        train_set_env_float("ANE_BG_EVAL_SEED", (float)extraEvalSeed);
        double lossExtra = NAN;
        TrainRunMetrics metricsExtra = {0};
        int extraRC = train_run_once(root, NO, &lossExtra, &metricsExtra);
        if (extraRC == 0) {
            double objectiveExtra = train_objective_for_metrics(metricsExtra,
                                                                gapWeight,
                                                                speedWeight,
                                                                regWeight,
                                                                regLocalCenter,
                                                                regGlobalCenter,
                                                                regMLPCenter);
            objective = 0.5 * (objectivePrimary + objectiveExtra);
        } else {
            objective = INFINITY;
        }
        if (oldEval.length > 0) {
            setenv("ANE_BG_EVAL_SEED", oldEval.UTF8String, 1);
        } else {
            unsetenv("ANE_BG_EVAL_SEED");
        }
    }
    if (metricsOut) {
        *metricsOut = metricsPrimary;
    }
    if (okOut) *okOut = isfinite(objective);
    return objective;
}

static int train_run_update_loop(NSString *root, TrainRunMetrics baselineMetrics) {
    const int steps = train_env_int(@"ANE_BG_UPDATE_STEPS", 4);
    const float probeDelta = train_env_float(@"ANE_BG_UPDATE_PROBE_DELTA", 0.02f);
    const float updateStep = train_env_float(@"ANE_BG_UPDATE_STEP", 0.01f);
    const double gapWeight = (double)train_env_float(@"ANE_BG_UPDATE_GAP_WEIGHT", 0.25f);
    const double speedWeight = (double)train_env_float(@"ANE_BG_UPDATE_SPEED_WEIGHT", 0.0f);
    const double regWeight = (double)train_env_float(@"ANE_BG_UPDATE_REG_WEIGHT", 0.01f);
    const int baseEvalSeed = train_env_int(@"ANE_BG_EVAL_SEED", 1);
    const int extraEvalSeed = train_env_int(@"ANE_BG_UPDATE_EXTRA_EVAL_SEED", 2);
    const double minImprove = (double)train_env_float(@"ANE_BG_UPDATE_MIN_IMPROVE", 0.0001f);
    const double maxSlowdownFrac = (double)train_env_float(@"ANE_BG_UPDATE_MAX_SLOWDOWN_FRAC", 0.15f);
    NSString *target = train_env_string(@"ANE_BG_UPDATE_TARGET", @"head_weight");
    NSString *tsvPath = train_env_string(@"ANE_BG_UPDATE_TSV", [root stringByAppendingPathComponent:@"build/update_loop.tsv"]);
    [[NSFileManager defaultManager] createDirectoryAtPath:[tsvPath stringByDeletingLastPathComponent]
                              withIntermediateDirectories:YES
                                               attributes:nil
                                                    error:nil];

    const float regLocalCenter = train_env_float(@"ANE_BG_ALPHA_LOCAL_MUL", 1.15f);
    const float regGlobalCenter = train_env_float(@"ANE_BG_ALPHA_GLOBAL_MUL", 1.40f);
    const float regMLPCenter = train_env_float(@"ANE_BG_ALPHA_MLP_MUL", 0.05f);
    const double baselineAvgMs = baselineMetrics.avgMs;

    TrainUpdateParam params[3] = {0};
    size_t paramCount = 0;
    if ([target isEqualToString:@"residuals"]) {
        params[paramCount++] = (TrainUpdateParam){ @"local", "ANE_BG_ALPHA_LOCAL_MUL",
            train_env_float(@"ANE_BG_ALPHA_LOCAL_MUL", 1.15f),
            train_env_float(@"ANE_BG_ALPHA_OPT_LOCAL_MIN", 0.80f),
            train_env_float(@"ANE_BG_ALPHA_OPT_LOCAL_MAX", 1.40f) };
        params[paramCount++] = (TrainUpdateParam){ @"global", "ANE_BG_ALPHA_GLOBAL_MUL",
            train_env_float(@"ANE_BG_ALPHA_GLOBAL_MUL", 1.40f),
            train_env_float(@"ANE_BG_ALPHA_OPT_GLOBAL_MIN", 1.00f),
            train_env_float(@"ANE_BG_ALPHA_OPT_GLOBAL_MAX", 1.60f) };
        params[paramCount++] = (TrainUpdateParam){ @"mlp", "ANE_BG_ALPHA_MLP_MUL",
            train_env_float(@"ANE_BG_ALPHA_MLP_MUL", 0.05f),
            train_env_float(@"ANE_BG_ALPHA_OPT_MLP_MIN", 0.01f),
            train_env_float(@"ANE_BG_ALPHA_OPT_MLP_MAX", 0.30f) };
    } else if ([target isEqualToString:@"head_hybrid"]) {
        params[paramCount++] = (TrainUpdateParam){ @"head_weight", "ANE_BG_HEAD_WEIGHT_SCALE",
            train_env_float(@"ANE_BG_HEAD_WEIGHT_SCALE", 0.98f),
            train_env_float(@"ANE_BG_ALPHA_OPT_HEAD_WEIGHT_MIN", 0.60f),
            train_env_float(@"ANE_BG_ALPHA_OPT_HEAD_WEIGHT_MAX", 1.60f) };
        params[paramCount++] = (TrainUpdateParam){ @"head_logit", "ANE_BG_HEAD_LOGIT_SCALE",
            train_env_float(@"ANE_BG_HEAD_LOGIT_SCALE", 0.97f),
            train_env_float(@"ANE_BG_ALPHA_OPT_HEAD_LOGIT_MIN", 0.60f),
            train_env_float(@"ANE_BG_ALPHA_OPT_HEAD_LOGIT_MAX", 1.60f) };
    } else if ([target isEqualToString:@"head_logit"]) {
        params[paramCount++] = (TrainUpdateParam){ @"head_logit", "ANE_BG_HEAD_LOGIT_SCALE",
            train_env_float(@"ANE_BG_HEAD_LOGIT_SCALE", 0.97f),
            train_env_float(@"ANE_BG_ALPHA_OPT_HEAD_LOGIT_MIN", 0.60f),
            train_env_float(@"ANE_BG_ALPHA_OPT_HEAD_LOGIT_MAX", 1.60f) };
    } else {
        target = @"head_weight";
        params[paramCount++] = (TrainUpdateParam){ @"head_weight", "ANE_BG_HEAD_WEIGHT_SCALE",
            train_env_float(@"ANE_BG_HEAD_WEIGHT_SCALE", 0.98f),
            train_env_float(@"ANE_BG_ALPHA_OPT_HEAD_WEIGHT_MIN", 0.60f),
            train_env_float(@"ANE_BG_ALPHA_OPT_HEAD_WEIGHT_MAX", 1.60f) };
    }

    NSString *oldExperimentLog = [NSProcessInfo.processInfo.environment objectForKey:@"ANE_BG_EXPERIMENT_LOG"];
    setenv("ANE_BG_EXPERIMENT_LOG", "0", 1);
    for (size_t i = 0; i < paramCount; ++i) {
        train_set_env_float(params[i].envName, params[i].value);
    }

    TrainRunMetrics currentMetrics = baselineMetrics;
    BOOL okCurrent = NO;
    double currentObjective = train_eval_objective_current_env(root,
                                                               baseEvalSeed,
                                                               extraEvalSeed,
                                                               gapWeight,
                                                               speedWeight,
                                                               regWeight,
                                                               regLocalCenter,
                                                               regGlobalCenter,
                                                               regMLPCenter,
                                                               &currentMetrics,
                                                               &okCurrent);
    if (!okCurrent) {
        if (oldExperimentLog.length > 0) {
            setenv("ANE_BG_EXPERIMENT_LOG", oldExperimentLog.UTF8String, 1);
        } else {
            unsetenv("ANE_BG_EXPERIMENT_LOG");
        }
        return 1;
    }

    for (int step = 1; step <= steps; ++step) {
        TrainUpdateParam before[3] = {0};
        TrainUpdateParam proposal[3] = {0};
        for (size_t i = 0; i < paramCount; ++i) {
            before[i] = params[i];
            float original = params[i].value;
            float minusValue = train_clamp_value(original - probeDelta, params[i].minValue, params[i].maxValue);
            float plusValue = train_clamp_value(original + probeDelta, params[i].minValue, params[i].maxValue);

            train_set_env_float(params[i].envName, minusValue);
            BOOL okMinus = NO;
            double objMinus = train_eval_objective_current_env(root, baseEvalSeed, extraEvalSeed,
                                                               gapWeight, speedWeight, regWeight,
                                                               regLocalCenter, regGlobalCenter, regMLPCenter,
                                                               NULL, &okMinus);
            train_set_env_float(params[i].envName, plusValue);
            BOOL okPlus = NO;
            double objPlus = train_eval_objective_current_env(root, baseEvalSeed, extraEvalSeed,
                                                              gapWeight, speedWeight, regWeight,
                                                              regLocalCenter, regGlobalCenter, regMLPCenter,
                                                              NULL, &okPlus);
            train_set_env_float(params[i].envName, original);

            float direction = 0.0f;
            if (okMinus && okPlus) {
                direction = (objPlus < objMinus) ? 1.0f : -1.0f;
            } else if (okPlus) {
                direction = 1.0f;
            } else if (okMinus) {
                direction = -1.0f;
            }
            float proposedValue = train_clamp_value(original + (direction * updateStep), params[i].minValue, params[i].maxValue);
            proposal[i] = (TrainUpdateParam){ params[i].label, params[i].envName, proposedValue, params[i].minValue, params[i].maxValue };
        }

        for (size_t i = 0; i < paramCount; ++i) {
            train_set_env_float(proposal[i].envName, proposal[i].value);
        }
        TrainRunMetrics proposalMetrics = {0};
        BOOL okProposal = NO;
        double proposalObjective = train_eval_objective_current_env(root,
                                                                    baseEvalSeed,
                                                                    extraEvalSeed,
                                                                    gapWeight,
                                                                    speedWeight,
                                                                    regWeight,
                                                                    regLocalCenter,
                                                                    regGlobalCenter,
                                                                    regMLPCenter,
                                                                    &proposalMetrics,
                                                                    &okProposal);
        BOOL accepted = okProposal &&
                        isfinite(proposalObjective) &&
                        (proposalObjective + minImprove < currentObjective) &&
                        proposalMetrics.stable &&
                        (baselineAvgMs <= 0.0 || proposalMetrics.avgMs <= baselineAvgMs * (1.0 + maxSlowdownFrac));
        NSString *note = accepted ? @"accepted" : @"rejected";
        train_append_update_row(tsvPath,
                                step,
                                target,
                                note,
                                accepted ? @"YES" : @"NO",
                                currentObjective,
                                proposalObjective,
                                currentMetrics,
                                proposalMetrics,
                                before,
                                proposal,
                                paramCount);
        if (!accepted) {
            for (size_t i = 0; i < paramCount; ++i) {
                train_set_env_float(before[i].envName, before[i].value);
            }
            NSLog(@"update-loop step %d rejected: target=%@ objective_before=%.6f objective_after=%.6f", step, target, currentObjective, proposalObjective);
            break;
        }

        for (size_t i = 0; i < paramCount; ++i) {
            params[i].value = proposal[i].value;
        }
        currentMetrics = proposalMetrics;
        currentObjective = proposalObjective;
        NSLog(@"update-loop step %d accepted: target=%@ objective=%.6f eval_ce=%.4f gap=%+.4f",
              step,
              target,
              currentObjective,
              currentMetrics.eval.loss,
              currentMetrics.eval.loss - currentMetrics.train.loss);
    }

    if (oldExperimentLog.length > 0) {
        setenv("ANE_BG_EXPERIMENT_LOG", oldExperimentLog.UTF8String, 1);
    } else {
        unsetenv("ANE_BG_EXPERIMENT_LOG");
    }
    train_set_env_float("ANE_BG_EVAL_SEED", (float)baseEvalSeed);
    return 0;
}

static int train_run_alpha_optimization(NSString *root, TrainRunMetrics baselineMetrics) {
    const int steps = train_env_int(@"ANE_BG_ALPHA_OPT_STEPS", 3);
    const float delta = train_env_float(@"ANE_BG_ALPHA_OPT_DELTA", 0.05f);
    const double gapWeight = (double)train_env_float(@"ANE_BG_ALPHA_OPT_GAP_WEIGHT", 0.25f);
    const double speedWeight = (double)train_env_float(@"ANE_BG_ALPHA_OPT_SPEED_WEIGHT", 0.0f);
    const double regWeight = (double)train_env_float(@"ANE_BG_ALPHA_OPT_REG_WEIGHT", 0.01f);
    const float localMin = train_env_float(@"ANE_BG_ALPHA_OPT_LOCAL_MIN", 0.80f);
    const float localMax = train_env_float(@"ANE_BG_ALPHA_OPT_LOCAL_MAX", 1.40f);
    const float globalMin = train_env_float(@"ANE_BG_ALPHA_OPT_GLOBAL_MIN", 1.00f);
    const float globalMax = train_env_float(@"ANE_BG_ALPHA_OPT_GLOBAL_MAX", 1.60f);
    const float mlpMin = train_env_float(@"ANE_BG_ALPHA_OPT_MLP_MIN", 0.01f);
    const float mlpMax = train_env_float(@"ANE_BG_ALPHA_OPT_MLP_MAX", 0.30f);
    const float headWeightMin = train_env_float(@"ANE_BG_ALPHA_OPT_HEAD_WEIGHT_MIN", 0.60f);
    const float headWeightMax = train_env_float(@"ANE_BG_ALPHA_OPT_HEAD_WEIGHT_MAX", 1.60f);
    const float headLogitMin = train_env_float(@"ANE_BG_ALPHA_OPT_HEAD_LOGIT_MIN", 0.60f);
    const float headLogitMax = train_env_float(@"ANE_BG_ALPHA_OPT_HEAD_LOGIT_MAX", 1.60f);
    const int extraEvalSeed = train_env_int(@"ANE_BG_ALPHA_OPT_EXTRA_EVAL_SEED", -1);
    const int baseEvalSeed = train_env_int(@"ANE_BG_EVAL_SEED", 1);

    float localMul = train_env_float(@"ANE_BG_ALPHA_LOCAL_MUL", 1.15f);
    float globalMul = train_env_float(@"ANE_BG_ALPHA_GLOBAL_MUL", 1.40f);
    float mlpMul = train_env_float(@"ANE_BG_ALPHA_MLP_MUL", 0.05f);
    float headWeightScale = train_env_float(@"ANE_BG_HEAD_WEIGHT_SCALE", 0.98f);
    float headLogitScale = train_env_float(@"ANE_BG_HEAD_LOGIT_SCALE", 0.97f);
    const float regLocalCenter = localMul;
    const float regGlobalCenter = globalMul;
    const float regMLPCenter = mlpMul;

    NSString *optPath = train_env_string(@"ANE_BG_ALPHA_OPT_TSV", [root stringByAppendingPathComponent:@"build/alpha_opt.tsv"]);
    [[NSFileManager defaultManager] createDirectoryAtPath:[optPath stringByDeletingLastPathComponent]
                              withIntermediateDirectories:YES
                                               attributes:nil
                                                    error:nil];

    TrainRunMetrics currentMetrics = baselineMetrics;
    double currentObjective = train_alpha_objective(currentMetrics,
                                                    localMul,
                                                    globalMul,
                                                    mlpMul,
                                                    gapWeight,
                                                    speedWeight,
                                                    regWeight,
                                                    regLocalCenter,
                                                    regGlobalCenter,
                                                    regMLPCenter);
    train_append_alpha_opt_row(optPath, 0, localMul, globalMul, mlpMul, headWeightScale, headLogitScale, currentMetrics, currentObjective, @"baseline");

    for (int step = 1; step <= steps; ++step) {
        BOOL improved = NO;
        float bestLocal = localMul;
        float bestGlobal = globalMul;
        float bestMLP = mlpMul;
        float bestHeadWeight = headWeightScale;
        float bestHeadLogit = headLogitScale;
        TrainRunMetrics bestMetrics = currentMetrics;
        double bestObjective = currentObjective;
        NSString *bestNote = @"none";

        const char *paramNames[] = { "ANE_BG_ALPHA_LOCAL_MUL", "ANE_BG_ALPHA_GLOBAL_MUL", "ANE_BG_ALPHA_MLP_MUL", "ANE_BG_HEAD_WEIGHT_SCALE", "ANE_BG_HEAD_LOGIT_SCALE" };
        float *paramValues[] = { &localMul, &globalMul, &mlpMul, &headWeightScale, &headLogitScale };
        NSString *paramLabels[] = { @"local", @"global", @"mlp", @"head_weight", @"head_logit" };

        for (int p = 0; p < 5; ++p) {
            for (int d = -1; d <= 1; d += 2) {
                float candidateValue = *paramValues[p] + ((float)d * delta);
                if ((p == 0 && (candidateValue < localMin || candidateValue > localMax)) ||
                    (p == 1 && (candidateValue < globalMin || candidateValue > globalMax)) ||
                    (p == 2 && (candidateValue < mlpMin || candidateValue > mlpMax)) ||
                    (p == 3 && (candidateValue < headWeightMin || candidateValue > headWeightMax)) ||
                    (p == 4 && (candidateValue < headLogitMin || candidateValue > headLogitMax))) {
                    continue;
                }

                train_set_env_float("ANE_BG_ALPHA_LOCAL_MUL", localMul);
                train_set_env_float("ANE_BG_ALPHA_GLOBAL_MUL", globalMul);
                train_set_env_float("ANE_BG_ALPHA_MLP_MUL", mlpMul);
                train_set_env_float("ANE_BG_HEAD_WEIGHT_SCALE", headWeightScale);
                train_set_env_float("ANE_BG_HEAD_LOGIT_SCALE", headLogitScale);
                train_set_env_float(paramNames[p], candidateValue);

                setenv("ANE_BG_EXPERIMENT_LOG", "0", 1);
                double candidateLoss = NAN;
                TrainRunMetrics candidateMetrics = {0};
                int rc = train_run_once(root, NO, &candidateLoss, &candidateMetrics);
                if (rc != 0) {
                    NSLog(@"alpha-opt candidate failed: %@=%0.4f", paramLabels[p], candidateValue);
                    setenv("ANE_BG_EXPERIMENT_LOG", "1", 1);
                    continue;
                }
                double objectivePrimary = train_alpha_objective(candidateMetrics,
                                                                (p == 0) ? candidateValue : localMul,
                                                                (p == 1) ? candidateValue : globalMul,
                                                                (p == 2) ? candidateValue : mlpMul,
                                                                gapWeight,
                                                                speedWeight,
                                                                regWeight,
                                                                regLocalCenter,
                                                                regGlobalCenter,
                                                                regMLPCenter);
                double candidateObjective = objectivePrimary;
                if (extraEvalSeed >= 0 && extraEvalSeed != baseEvalSeed) {
                    NSString *oldEval = [NSProcessInfo.processInfo.environment objectForKey:@"ANE_BG_EVAL_SEED"];
                    train_set_env_float("ANE_BG_EVAL_SEED", (float)extraEvalSeed);
                    double extraLoss = NAN;
                    TrainRunMetrics extraMetrics = {0};
                    int extraRC = train_run_once(root, NO, &extraLoss, &extraMetrics);
                    if (extraRC == 0) {
                        double objectiveExtra = train_alpha_objective(extraMetrics,
                                                                      (p == 0) ? candidateValue : localMul,
                                                                      (p == 1) ? candidateValue : globalMul,
                                                                      (p == 2) ? candidateValue : mlpMul,
                                                                      gapWeight,
                                                                      speedWeight,
                                                                      regWeight,
                                                                      regLocalCenter,
                                                                      regGlobalCenter,
                                                                      regMLPCenter);
                        candidateObjective = 0.5 * (objectivePrimary + objectiveExtra);
                    }
                    if (oldEval.length > 0) {
                        setenv("ANE_BG_EVAL_SEED", oldEval.UTF8String, 1);
                    } else {
                        unsetenv("ANE_BG_EVAL_SEED");
                    }
                }
                setenv("ANE_BG_EXPERIMENT_LOG", "1", 1);
                if (candidateObjective + 1e-6 < bestObjective) {
                    improved = YES;
                    bestObjective = candidateObjective;
                    bestMetrics = candidateMetrics;
                    bestLocal = localMul;
                    bestGlobal = globalMul;
                    bestMLP = mlpMul;
                    bestHeadWeight = headWeightScale;
                    bestHeadLogit = headLogitScale;
                    if (p == 0) bestLocal = candidateValue;
                    if (p == 1) bestGlobal = candidateValue;
                    if (p == 2) bestMLP = candidateValue;
                    if (p == 3) bestHeadWeight = candidateValue;
                    if (p == 4) bestHeadLogit = candidateValue;
                    bestNote = [NSString stringWithFormat:@"%@=%0.4f", paramLabels[p], candidateValue];
                }
            }
        }

        if (!improved) {
            train_append_alpha_opt_row(optPath, step, localMul, globalMul, mlpMul, headWeightScale, headLogitScale, currentMetrics, currentObjective, @"no_improvement_stop");
            NSLog(@"alpha-opt: no improvement at step %d (objective=%.6f), stopping.", step, currentObjective);
            break;
        }

        localMul = bestLocal;
        globalMul = bestGlobal;
        mlpMul = bestMLP;
        headWeightScale = bestHeadWeight;
        headLogitScale = bestHeadLogit;
        currentMetrics = bestMetrics;
        currentObjective = bestObjective;

        train_set_env_float("ANE_BG_ALPHA_LOCAL_MUL", localMul);
        train_set_env_float("ANE_BG_ALPHA_GLOBAL_MUL", globalMul);
        train_set_env_float("ANE_BG_ALPHA_MLP_MUL", mlpMul);
        train_set_env_float("ANE_BG_HEAD_WEIGHT_SCALE", headWeightScale);
        train_set_env_float("ANE_BG_HEAD_LOGIT_SCALE", headLogitScale);
        train_append_alpha_opt_row(optPath, step, localMul, globalMul, mlpMul, headWeightScale, headLogitScale, currentMetrics, currentObjective, bestNote);
        NSLog(@"alpha-opt step %d accepted: local=%.4f global=%.4f mlp=%.4f head_weight=%.4f head_logit=%.4f objective=%.6f train_ce=%.4f eval_ce=%.4f gap=%+.4f",
              step,
              localMul,
              globalMul,
              mlpMul,
              headWeightScale,
              headLogitScale,
              currentObjective,
              currentMetrics.train.loss,
              currentMetrics.eval.loss,
              currentMetrics.eval.loss - currentMetrics.train.loss);
    }

    train_set_env_float("ANE_BG_EVAL_SEED", (float)baseEvalSeed);
    return 0;
}

static int run_training_skeleton(NSString *root) {
    const BOOL logSurfaceStats = train_env_bool(@"ANE_BG_LOG_SURFACE_STATS", YES);
    double baselineLoss = NAN;
    TrainRunMetrics baselineMetrics = {0};
    int rc = train_run_once(root, logSurfaceStats, &baselineLoss, &baselineMetrics);
    if (rc != 0) {
        return rc;
    }

    if (train_env_bool(@"ANE_BG_UPDATE_LOOP", NO)) {
        return train_run_update_loop(root, baselineMetrics);
    }

    if (train_env_bool(@"ANE_BG_ALPHA_OPTIMIZE", NO)) {
        return train_run_alpha_optimization(root, baselineMetrics);
    }

    if (!train_env_bool(@"ANE_BG_PERTURB_EXPERIMENT", NO) || !isfinite(baselineLoss)) {
        return 0;
    }

    NSString *singleBranch = [NSProcessInfo.processInfo.environment objectForKey:@"ANE_BG_PERTURB_BRANCH"];
    float delta = train_env_float(@"ANE_BG_PERTURB_DELTA", 0.05f);
    NSArray<NSString *> *branches = singleBranch.length > 0 ? @[singleBranch] : @[ @"local", @"global", @"mlp", @"depth" ];
    NSArray<NSNumber *> *deltas = @[@(-delta), @(delta)];
    NSString *originalLogValue = [NSProcessInfo.processInfo.environment objectForKey:@"ANE_BG_LOG_SURFACE_STATS"];
    NSMutableArray<NSDictionary *> *sweepResults = [NSMutableArray array];
    setenv("ANE_BG_LOG_SURFACE_STATS", "0", 1);

    for (NSString *requestedBranch in branches) {
        NSString *branch = requestedBranch;
        NSString *envName = nil;
        float fallback = 0.65f;
        if ([branch isEqualToString:@"local"]) {
            envName = @"ANE_BG_ALPHA_LOCAL_MUL";
            fallback = 1.15f;
        } else if ([branch isEqualToString:@"global"]) {
            envName = @"ANE_BG_ALPHA_GLOBAL_MUL";
            fallback = 1.40f;
        } else if ([branch isEqualToString:@"depth"]) {
            envName = @"ANE_BG_ALPHA_DEPTH_POWER";
            fallback = 0.0f;
        } else {
            envName = @"ANE_BG_ALPHA_MLP_MUL";
            branch = @"mlp";
            fallback = 0.05f;
        }

        NSString *originalValue = [NSProcessInfo.processInfo.environment objectForKey:envName];
        const float currentValue = originalValue.length > 0 ? originalValue.floatValue : fallback;
        for (NSNumber *deltaNumber in deltas) {
            const float sweepDelta = deltaNumber.floatValue;
            const float perturbedValue = currentValue + sweepDelta;
            setenv(envName.UTF8String, [[NSString stringWithFormat:@"%.6f", perturbedValue] UTF8String], 1);

            double perturbedLoss = NAN;
            int perturbedRC = train_run_once(root, NO, &perturbedLoss, NULL);
            if (perturbedRC != 0 || !isfinite(perturbedLoss)) {
                if (originalValue.length > 0) {
                    setenv(envName.UTF8String, originalValue.UTF8String, 1);
                } else {
                    unsetenv(envName.UTF8String);
                }
                if (originalLogValue.length > 0) {
                    setenv("ANE_BG_LOG_SURFACE_STATS", originalLogValue.UTF8String, 1);
                } else {
                    unsetenv("ANE_BG_LOG_SURFACE_STATS");
                }
                NSLog(@"perturbation experiment failed for %@ branch", branch);
                return perturbedRC != 0 ? perturbedRC : 0;
            }

            NSLog(@"perturbation sweep: branch=%@ base_value=%.4f perturbed_value=%.4f delta=%+.4f baseline_loss=%.4f perturbed_loss=%.4f loss_change=%+.4f",
                  branch,
                  currentValue,
                  perturbedValue,
                  sweepDelta,
                  baselineLoss,
                  perturbedLoss,
                  perturbedLoss - baselineLoss);
            [sweepResults addObject:@{
                @"branch": branch,
                @"base": @(currentValue),
                @"perturbed": @(perturbedValue),
                @"delta": @(sweepDelta),
                @"loss_change": @(perturbedLoss - baselineLoss)
            }];
        }

        if (originalValue.length > 0) {
            setenv(envName.UTF8String, originalValue.UTF8String, 1);
        } else {
            unsetenv(envName.UTF8String);
        }
    }

    if (originalLogValue.length > 0) {
        setenv("ANE_BG_LOG_SURFACE_STATS", originalLogValue.UTF8String, 1);
    } else {
        unsetenv("ANE_BG_LOG_SURFACE_STATS");
    }

    if (sweepResults.count > 0) {
        NSArray<NSDictionary *> *ranked = [sweepResults sortedArrayUsingComparator:^NSComparisonResult(NSDictionary *lhs, NSDictionary *rhs) {
            double lhsChange = [lhs[@"loss_change"] doubleValue];
            double rhsChange = [rhs[@"loss_change"] doubleValue];
            if (lhsChange < rhsChange) return NSOrderedAscending;
            if (lhsChange > rhsChange) return NSOrderedDescending;
            return NSOrderedSame;
        }];
        NSMutableArray<NSString *> *summary = [NSMutableArray arrayWithCapacity:ranked.count];
        for (NSDictionary *entry in ranked) {
            [summary addObject:[NSString stringWithFormat:@"%@:%+.4f->%.4f (%+.4f)",
                                entry[@"branch"],
                                [entry[@"delta"] doubleValue],
                                [entry[@"perturbed"] doubleValue],
                                [entry[@"loss_change"] doubleValue]]];
        }
        NSLog(@"perturbation summary (best to worst): %@", [summary componentsJoinedByString:@" | "]);
    }

    if (!singleBranch.length) {
        NSString *origLocal = [NSProcessInfo.processInfo.environment objectForKey:@"ANE_BG_ALPHA_LOCAL_MUL"];
        NSString *origGlobal = [NSProcessInfo.processInfo.environment objectForKey:@"ANE_BG_ALPHA_GLOBAL_MUL"];
        const float baseLocal = origLocal.length > 0 ? origLocal.floatValue : 1.15f;
        const float baseGlobal = origGlobal.length > 0 ? origGlobal.floatValue : 1.40f;
        NSArray<NSNumber *> *jointOffsets = @[@(-0.05f), @(0.0f), @(0.05f)];
        NSMutableArray<NSDictionary *> *jointResults = [NSMutableArray array];

        for (NSNumber *localOffset in jointOffsets) {
            for (NSNumber *globalOffset in jointOffsets) {
                const float localValue = baseLocal + localOffset.floatValue;
                const float globalValue = baseGlobal + globalOffset.floatValue;
                if (fabsf(localValue - baseLocal) < 1e-6f && fabsf(globalValue - baseGlobal) < 1e-6f) {
                    continue;
                }
                setenv("ANE_BG_ALPHA_LOCAL_MUL", [[NSString stringWithFormat:@"%.6f", localValue] UTF8String], 1);
                setenv("ANE_BG_ALPHA_GLOBAL_MUL", [[NSString stringWithFormat:@"%.6f", globalValue] UTF8String], 1);

                double jointLoss = NAN;
                int jointRC = train_run_once(root, NO, &jointLoss, NULL);
                if (jointRC != 0 || !isfinite(jointLoss)) {
                    if (origLocal.length > 0) {
                        setenv("ANE_BG_ALPHA_LOCAL_MUL", origLocal.UTF8String, 1);
                    } else {
                        unsetenv("ANE_BG_ALPHA_LOCAL_MUL");
                    }
                    if (origGlobal.length > 0) {
                        setenv("ANE_BG_ALPHA_GLOBAL_MUL", origGlobal.UTF8String, 1);
                    } else {
                        unsetenv("ANE_BG_ALPHA_GLOBAL_MUL");
                    }
                    NSLog(@"joint perturbation experiment failed for local/global grid");
                    return jointRC != 0 ? jointRC : 0;
                }

                const double lossChange = jointLoss - baselineLoss;
                [jointResults addObject:@{
                    @"local": @(localValue),
                    @"global": @(globalValue),
                    @"loss_change": @(lossChange)
                }];
                NSLog(@"joint perturbation: local=%.4f global=%.4f baseline_loss=%.4f joint_loss=%.4f loss_change=%+.4f",
                      localValue,
                      globalValue,
                      baselineLoss,
                      jointLoss,
                      lossChange);
            }
        }

        if (origLocal.length > 0) {
            setenv("ANE_BG_ALPHA_LOCAL_MUL", origLocal.UTF8String, 1);
        } else {
            unsetenv("ANE_BG_ALPHA_LOCAL_MUL");
        }
        if (origGlobal.length > 0) {
            setenv("ANE_BG_ALPHA_GLOBAL_MUL", origGlobal.UTF8String, 1);
        } else {
            unsetenv("ANE_BG_ALPHA_GLOBAL_MUL");
        }

        if (jointResults.count > 0) {
            NSArray<NSDictionary *> *jointRanked = [jointResults sortedArrayUsingComparator:^NSComparisonResult(NSDictionary *lhs, NSDictionary *rhs) {
                double lhsChange = [lhs[@"loss_change"] doubleValue];
                double rhsChange = [rhs[@"loss_change"] doubleValue];
                if (lhsChange < rhsChange) return NSOrderedAscending;
                if (lhsChange > rhsChange) return NSOrderedDescending;
                return NSOrderedSame;
            }];
            NSMutableArray<NSString *> *jointSummary = [NSMutableArray arrayWithCapacity:jointRanked.count];
            for (NSDictionary *entry in jointRanked) {
                [jointSummary addObject:[NSString stringWithFormat:@"(local=%.2f,global=%.2f => %+.4f)",
                                         [entry[@"local"] doubleValue],
                                         [entry[@"global"] doubleValue],
                                         [entry[@"loss_change"] doubleValue]]];
            }
            NSLog(@"joint perturbation summary (best to worst): %@", [jointSummary componentsJoinedByString:@" | "]);
        }
    }
    return 0;
}

int main(void) {
    @autoreleasepool {
        NSString *root = NSFileManager.defaultManager.currentDirectoryPath;
        return run_training_skeleton(root);
    }
}
