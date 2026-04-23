#import "../training/ane_runtime.h"

#import <Foundation/Foundation.h>

static BOOL equal_arrays(NSArray<NSString *> *lhs, NSArray<NSString *> *rhs) {
    return [lhs isEqualToArray:rhs];
}

int main(void) {
    @autoreleasepool {
        NSString *mixedMIL =
            @"program(1.3)\n"
            "{\n"
            "  func main<ios18>(tensor<fp16, [1,1,1,1]> x) {\n"
            "    tensor<fp16, [1,1,1,1]> W0 = const()[val=tensor<fp16, [1,1,1,1]>(BLOBFILE(path=string(\"@model_path/weights/a.bin\"), offset=uint64(64)))];\n"
            "    tensor<fp16, [1,1,1,1]> W1 = const()[val=tensor<fp16, [1,1,1,1]>(BLOBFILE(path = string(\"@model_path/weights/b.bin\"), offset = uint64(64)))];\n"
            "    tensor<fp16, [1,1,1,1]> W2 = const()[val=tensor<fp16, [1,1,1,1]>(BLOBFILE( path = string( \"/tmp/c.bin\" ), offset = uint64(64)))];\n"
            "    tensor<fp16, [1,1,1,1]> W3 = const()[val=tensor<fp16, [1,1,1,1]>(BLOBFILE(path = string(\"@model_path/weights/b.bin\"), offset = uint64(64)))];\n"
            "    tensor<fp16, [1,1,1,1]> y = add(x=x, y=W0);\n"
            "  } -> (y);\n"
            "}\n";

        NSArray<NSString *> *expected = @[
            @"@model_path/weights/a.bin",
            @"@model_path/weights/b.bin",
            @"/tmp/c.bin",
        ];
        NSArray<NSString *> *actual = ane_bg_extract_blobfile_paths(mixedMIL);
        if (!equal_arrays(actual, expected)) {
            NSLog(@"parse test failed: expected %@ got %@", expected, actual);
            return 1;
        }

        if (![ane_bg_extract_blobfile_paths(nil) isEqual:@[]]) {
            NSLog(@"parse test failed for nil input");
            return 1;
        }

        if (![ane_bg_extract_blobfile_paths(@"program(1.3) { }") isEqual:@[]]) {
            NSLog(@"parse test failed for weightless input");
            return 1;
        }

        NSLog(@"runtime parse tests passed");
        return 0;
    }
}
