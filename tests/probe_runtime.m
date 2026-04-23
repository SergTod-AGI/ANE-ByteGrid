#import "../training/ane_mil_gen.h"
#import "../training/ane_runtime.h"
#import "../training/model.h"

#import <Foundation/Foundation.h>
#import <stdlib.h>

static NSString *probe_control_mil(void) {
    return [NSString stringWithFormat:
            @"%@"
            "{\n"
            "    func main<ios18>(tensor<fp16, [1, 1, 1, 1]> x) {\n"
            "        tensor<fp16, [1, 1, 1, 1]> y = sigmoid(x = x)[name = string(\"sigmoid\")];\n"
            "    } -> (y);\n"
            "}\n",
            ANE_BG_MIL_BUILD_INFO_HEADER];
}

static BOOL ensure_probe_weights(NSString *weightDir, NSError **error) {
    if (!ane_bg_generate_all_weight_blobs(weightDir, error)) {
        return NO;
    }
    NSString *src = [weightDir stringByAppendingPathComponent:@"stem.bin"];
    NSString *dst = [weightDir stringByAppendingPathComponent:@"weight.bin"];
    NSData *srcData = [NSData dataWithContentsOfFile:src options:0 error:error];
    if (!srcData) {
        return NO;
    }
    return [srcData writeToFile:dst options:NSDataWritingAtomic error:error];
}

static void run_probe_case(NSString *label,
                           NSString *mil,
                           NSString *weightDir,
                           BOOL *okOut) {
    NSError *error = nil;
    ANEByteGridRuntime *runtime = [[ANEByteGridRuntime alloc] init];
    BOOL ok = [runtime compileMIL:mil weightDirectory:weightDir error:&error];
    if (okOut) {
        *okOut = ok;
    }
    if (ok) {
        NSLog(@"PROBE %@: COMPILED compile=%.2fms load=%.2fms weights=%lu",
              label,
              runtime.lastCompileDurationMs,
              runtime.lastLoadDurationMs,
              (unsigned long)runtime.compiledWeightCount);
        return;
    }
    NSLog(@"PROBE %@: FAILED %@", label, error.localizedDescription);
}

static BOOL should_run_case(NSString *selected, NSString *caseName) {
    if (selected.length == 0 || [selected isEqualToString:@"all"]) {
        return YES;
    }
    return [selected isEqualToString:caseName];
}

int main(void) {
    @autoreleasepool {
        NSString *root = NSFileManager.defaultManager.currentDirectoryPath;
        NSString *weightDir = [root stringByAppendingPathComponent:@"weights_probe"];
        NSError *error = nil;
        if (!ensure_probe_weights(weightDir, &error)) {
            NSLog(@"probe weight generation failed: %@", error.localizedDescription);
            return 1;
        }

        const char *probeCase = getenv("ANE_PROBE_CASE");
        NSString *selected = [NSString stringWithUTF8String:(probeCase ? probeCase : "all")];
        selected = selected.lowercaseString;
        BOOL allOK = YES;
        BOOL ranAny = NO;

        if (should_run_case(selected, @"control")) {
            BOOL ok = NO;
            run_probe_case(@"control_sigmoid", probe_control_mil(), weightDir, &ok);
            allOK = allOK && ok;
            ranAny = YES;
        }
        if (should_run_case(selected, @"stem")) {
            BOOL ok = NO;
            run_probe_case(@"stem_only", ane_bg_gen_stem_mil(weightDir), weightDir, &ok);
            allOK = allOK && ok;
            ranAny = YES;
        }
        if (should_run_case(selected, @"conv")) {
            BOOL ok = NO;
            run_probe_case(@"exact_conv_fp16", ane_bg_gen_conv_fp16_exact(320, 512, 256), weightDir, &ok);
            allOK = allOK && ok;
            ranAny = YES;
        }
        if (should_run_case(selected, @"dynamic")) {
            BOOL ok = NO;
            run_probe_case(@"exact_conv_dynamic_fp16", ane_bg_gen_conv_dynamic_fp16_exact(320, 512, 256), weightDir, &ok);
            allOK = allOK && ok;
            ranAny = YES;
        }

        if (!ranAny) {
            NSLog(@"Unknown ANE_PROBE_CASE='%@' (expected: all|control|stem|conv|dynamic).", selected);
            return 2;
        }
        return allOK ? 0 : 1;
    }
}
