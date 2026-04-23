#import "ane_runtime.h"

#import <dlfcn.h>
#import <mach/mach_time.h>
#import <objc/message.h>
#import <objc/runtime.h>
#import <regex.h>
#import <strings.h>

static NSString *const ANEByteGridErrorDomain = @"ANEByteGrid";
static const unsigned int kANEQoSUserInitiated = 21;

typedef NS_ENUM(NSInteger, ANEByteGridErrorCode) {
    ANEByteGridErrorUnavailable = 1,
    ANEByteGridErrorInvalidMIL = 2,
    ANEByteGridErrorWeightLoadFailed = 3,
    ANEByteGridErrorCompileFailed = 4,
    ANEByteGridErrorLoadFailed = 5,
    ANEByteGridErrorEvaluateFailed = 6,
    ANEByteGridErrorInvalidSurface = 7,
    ANEByteGridErrorRequestFailed = 8,
};

static mach_timebase_info_data_t ANETimebase(void) {
    static dispatch_once_t onceToken;
    static mach_timebase_info_data_t timebase;
    dispatch_once(&onceToken, ^{
      mach_timebase_info(&timebase);
    });
    return timebase;
}

static double ANEElapsedMs(uint64_t start, uint64_t end) {
    mach_timebase_info_data_t tb = ANETimebase();
    uint64_t delta = end - start;
    return ((double)delta * (double)tb.numer / (double)tb.denom) / 1e6;
}

static NSError *ANEError(ANEByteGridErrorCode code, NSString *message, NSError *underlying) {
    NSString *fullMessage = underlying.localizedDescription.length > 0
        ? [NSString stringWithFormat:@"%@ %@", message, underlying.localizedDescription]
        : message;
    NSMutableDictionary *info = [NSMutableDictionary dictionaryWithObject:fullMessage forKey:NSLocalizedDescriptionKey];
    if (underlying) {
        info[NSUnderlyingErrorKey] = underlying;
    }
    return [NSError errorWithDomain:ANEByteGridErrorDomain code:code userInfo:info];
}

static id ANECallObjC(id target, SEL selector) {
    return ((id (*)(id, SEL))objc_msgSend)(target, selector);
}

static id ANECallObjC1(id target, SEL selector, id arg1) {
    return ((id (*)(id, SEL, id))objc_msgSend)(target, selector, arg1);
}

static id ANECallObjC2(id target, SEL selector, id arg1, id arg2) {
    return ((id (*)(id, SEL, id, id))objc_msgSend)(target, selector, arg1, arg2);
}

static id ANECallObjC3(id target, SEL selector, id arg1, id arg2, id arg3) {
    return ((id (*)(id, SEL, id, id, id))objc_msgSend)(target, selector, arg1, arg2, arg3);
}

static id ANECallObjC7(id target, SEL selector, id arg1, id arg2, id arg3, id arg4, id arg5, id arg6, id arg7) {
    return ((id (*)(id, SEL, id, id, id, id, id, id, id))objc_msgSend)(target, selector, arg1, arg2, arg3, arg4, arg5, arg6, arg7);
}

static BOOL ANECallBoolRequest(id target, SEL selector, unsigned int qos, id options, id request, NSError **error) {
    return ((BOOL (*)(id, SEL, unsigned int, id, id, NSError **))objc_msgSend)(target, selector, qos, options, request, error);
}

static BOOL ANECallBoolOptions(id target, SEL selector, unsigned int qos, id options, NSError **error) {
    return ((BOOL (*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(target, selector, qos, options, error);
}

static BOOL ANECallBoolOptionsOnly(id target, SEL selector, id options, NSError **error) {
    return ((BOOL (*)(id, SEL, id, NSError **))objc_msgSend)(target, selector, options, error);
}

static BOOL ANECallBoolError(id target, SEL selector, unsigned int qos, NSError **error) {
    return ((BOOL (*)(id, SEL, unsigned int, NSError **))objc_msgSend)(target, selector, qos, error);
}

static BOOL ANEEnvFlagEnabled(const char *name) {
    const char *value = getenv(name);
    if (!value || value[0] == '\0') {
        return NO;
    }
    if (strcmp(value, "0") == 0) {
        return NO;
    }
    if (strcasecmp(value, "false") == 0 || strcasecmp(value, "no") == 0) {
        return NO;
    }
    return YES;
}

static NSArray<NSString *> *ANESelectorNamesForClass(Class cls, BOOL classMethods) {
    if (!cls) {
        return @[];
    }
    Class methodHost = classMethods ? object_getClass((id)cls) : cls;
    if (!methodHost) {
        return @[];
    }
    unsigned int count = 0;
    Method *methods = class_copyMethodList(methodHost, &count);
    NSMutableArray<NSString *> *names = [NSMutableArray arrayWithCapacity:count];
    for (unsigned int i = 0; i < count; ++i) {
        SEL sel = method_getName(methods[i]);
        if (sel) {
            [names addObject:NSStringFromSelector(sel)];
        }
    }
    free(methods);
    [names sortUsingSelector:@selector(compare:)];
    return names;
}

NSArray<NSString *> *ane_bg_extract_blobfile_paths(NSString *mil) {
    if (mil.length == 0) {
        return @[];
    }
    NSError *error = nil;
    NSRegularExpression *regex =
        [NSRegularExpression regularExpressionWithPattern:@"BLOBFILE\\(\\s*path\\s*=\\s*string\\(\\s*\"([^\"]+)\"\\s*\\)"
                                                  options:0
                                                    error:&error];
    if (!regex) {
        return @[];
    }
    NSArray<NSTextCheckingResult *> *matches =
        [regex matchesInString:mil options:0 range:NSMakeRange(0, mil.length)];
    NSMutableOrderedSet<NSString *> *paths = [NSMutableOrderedSet orderedSet];
    for (NSTextCheckingResult *match in matches) {
        if (match.numberOfRanges < 2) {
            continue;
        }
        NSString *path = [mil substringWithRange:[match rangeAtIndex:1]];
        if (path.length > 0) {
            [paths addObject:path];
        }
    }
    return paths.array;
}

@interface ANEByteGridRuntime ()
@property (nonatomic, assign) BOOL privateAPIAvailable;
@property (nonatomic, assign) BOOL modelLoaded;
@property (nonatomic, assign) double lastCompileDurationMs;
@property (nonatomic, assign) double lastLoadDurationMs;
@property (nonatomic, assign) double lastEvaluationDurationMs;
@property (nonatomic, assign) double lastThroughputMBPerSecond;
@property (nonatomic, assign) NSUInteger compiledWeightCount;
@property (nonatomic, strong) NSBundle *aneBundle;
@property (nonatomic, strong) id compiledModel;
@property (nonatomic, strong) id modelDescriptor;
@property (nonatomic, strong) NSString *materializedModelDirectory;
@property (nonatomic, strong) NSString *normalizedMIL;
@property (nonatomic, strong) NSDictionary<NSString *, NSDictionary *> *descriptorWeights;
@property (nonatomic, assign) Class descriptorClass;
@property (nonatomic, assign) Class inMemoryModelClass;
@property (nonatomic, assign) Class requestClass;
@property (nonatomic, assign) Class ioSurfaceClass;
@end

@implementation ANEByteGridRuntime

- (instancetype)init {
    self = [super init];
    if (!self) {
        return nil;
    }
    NSError *error = nil;
    self.privateAPIAvailable = [self loadPrivateAPI:&error];
    if (!self.privateAPIAvailable) {
        NSLog(@"ANE runtime unavailable: %@", error.localizedDescription);
    }
    return self;
}

- (void)dealloc {
    [self unloadCompiledModel];
}

- (void)maybeLogRuntimeSelectorIntrospection {
    if (!ANEEnvFlagEnabled("ANE_BG_RUNTIME_SELECTOR_DUMP")) {
        return;
    }
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        NSArray<NSString *> *descriptorClassMethods = ANESelectorNamesForClass(self.descriptorClass, YES);
        NSArray<NSString *> *descriptorInstanceMethods = ANESelectorNamesForClass(self.descriptorClass, NO);
        NSArray<NSString *> *modelClassMethods = ANESelectorNamesForClass(self.inMemoryModelClass, YES);
        NSArray<NSString *> *modelInstanceMethods = ANESelectorNamesForClass(self.inMemoryModelClass, NO);
        NSLog(@"ANE selector dump _ANEInMemoryModelDescriptor class methods (%lu): %@",
              (unsigned long)descriptorClassMethods.count,
              descriptorClassMethods);
        NSLog(@"ANE selector dump _ANEInMemoryModelDescriptor instance methods (%lu): %@",
              (unsigned long)descriptorInstanceMethods.count,
              descriptorInstanceMethods);
        NSLog(@"ANE selector dump _ANEInMemoryModel class methods (%lu): %@",
              (unsigned long)modelClassMethods.count,
              modelClassMethods);
        NSLog(@"ANE selector dump _ANEInMemoryModel instance methods (%lu): %@",
              (unsigned long)modelInstanceMethods.count,
              modelInstanceMethods);
    });
}

- (id)createDescriptorForMILData:(NSData *)milData
                         milText:(NSString *)milText
                         weights:(NSDictionary *)weights
                    usedSelector:(NSString **)usedSelector
                           error:(NSError **)error {
    NSArray<NSDictionary<NSString *, id> *> *candidates = @[
        @{@"sel": @"modelWithMILText:weights:optionsPlist:", @"argc": @3, @"payload": @"data"},
        @{@"sel": @"modelWithMILText:weights:optionsPlist:", @"argc": @3, @"payload": @"text"},
        @{@"sel": @"modelWithNetworkDescription:weights:optionsPlist:", @"argc": @3, @"payload": @"text"},
        @{@"sel": @"modelWithMILText:weights:options:", @"argc": @3, @"payload": @"data"},
        @{@"sel": @"modelWithMILText:weights:options:", @"argc": @3, @"payload": @"text"},
        @{@"sel": @"modelWithMIL:weights:optionsPlist:", @"argc": @3, @"payload": @"data"},
        @{@"sel": @"modelWithMIL:weights:options:", @"argc": @3, @"payload": @"data"},
        @{@"sel": @"modelWithMILText:weights:", @"argc": @2, @"payload": @"data"},
        @{@"sel": @"modelWithMILText:weights:", @"argc": @2, @"payload": @"text"},
        @{@"sel": @"modelWithMIL:weights:", @"argc": @2, @"payload": @"data"},
        @{@"sel": @"descriptorWithMILText:weights:optionsPlist:", @"argc": @3, @"payload": @"data"},
        @{@"sel": @"descriptorWithMILText:weights:optionsPlist:", @"argc": @3, @"payload": @"text"},
    ];
    NSMutableArray<NSString *> *tried = [NSMutableArray array];

    for (NSDictionary<NSString *, id> *candidate in candidates) {
        NSString *name = candidate[@"sel"];
        NSInteger argc = [candidate[@"argc"] integerValue];
        NSString *payloadKind = candidate[@"payload"] ?: @"data";
        id payload = [payloadKind isEqualToString:@"text"] ? milText : milData;
        if (!payload) {
            continue;
        }
        SEL sel = NSSelectorFromString(name);
        if (![self.descriptorClass respondsToSelector:sel]) {
            continue;
        }
        [tried addObject:[NSString stringWithFormat:@"%@[%@]", name, payloadKind]];
        @try {
            id descriptor = nil;
            if (argc == 3) {
                descriptor = ANECallObjC3(self.descriptorClass, sel, payload, weights, nil);
            } else if (argc == 2) {
                descriptor = ANECallObjC2(self.descriptorClass, sel, payload, weights);
            }
            if (descriptor) {
                if (usedSelector) {
                    *usedSelector = [NSString stringWithFormat:@"%@[%@]", name, payloadKind];
                }
                return descriptor;
            }
        } @catch (NSException *exception) {
            if (ANEEnvFlagEnabled("ANE_BG_RUNTIME_SELECTOR_DUMP")) {
                NSLog(@"ANE descriptor selector exception %@[%@]: %@", name, payloadKind, exception.reason ?: @"<no reason>");
            }
        }
    }

    if (error) {
        NSArray<NSString *> *available = ANESelectorNamesForClass(self.descriptorClass, YES);
        NSString *message = [NSString stringWithFormat:@"No descriptor constructor accepted the payload. tried=%@ available=%@",
                             tried.count > 0 ? [tried componentsJoinedByString:@","] : @"<none>",
                             [available componentsJoinedByString:@","]];
        *error = ANEError(ANEByteGridErrorCompileFailed, message, nil);
    }
    return nil;
}

- (id)createInMemoryModelFromDescriptor:(id)descriptor
                           usedSelector:(NSString **)usedSelector {
    NSArray<NSString *> *classConstructors = @[
        @"inMemoryModelWithDescriptor:",
        @"modelWithDescriptor:",
        @"inMemoryModelFromDescriptor:",
    ];
    for (NSString *name in classConstructors) {
        SEL sel = NSSelectorFromString(name);
        if (![self.inMemoryModelClass respondsToSelector:sel]) {
            continue;
        }
        id model = ANECallObjC1(self.inMemoryModelClass, sel, descriptor);
        if (model) {
            if (usedSelector) {
                *usedSelector = [@"+" stringByAppendingString:name];
            }
            return model;
        }
    }

    SEL allocSel = @selector(alloc);
    if (![self.inMemoryModelClass respondsToSelector:allocSel]) {
        return nil;
    }
    id instance = ANECallObjC(self.inMemoryModelClass, allocSel);
    NSArray<NSString *> *initializers = @[
        @"initWithDescriptor:",
        @"initWithModelDescriptor:",
        @"initWithDesctiptor:",
    ];
    for (NSString *name in initializers) {
        SEL sel = NSSelectorFromString(name);
        if (![instance respondsToSelector:sel]) {
            continue;
        }
        id model = ANECallObjC1(instance, sel, descriptor);
        if (model) {
            if (usedSelector) {
                *usedSelector = [@"-" stringByAppendingString:name];
            }
            return model;
        }
    }
    return nil;
}

- (BOOL)invokeCompileOnModel:(id)model
                usedSelector:(NSString **)usedSelector
                       error:(NSError **)error {
    NSArray<NSDictionary<NSString *, id> *> *candidates = @[
        @{@"sel": @"compileWithQoS:options:error:", @"kind": @"qos_options", @"optionsMode": @"dict"},
        @{@"sel": @"compileWithQoS:options:error:", @"kind": @"qos_options", @"optionsMode": @"nil"},
        @{@"sel": @"compileWithQoS:error:", @"kind": @"qos_error"},
        @{@"sel": @"compileWithOptions:error:", @"kind": @"options_error", @"optionsMode": @"dict"},
        @{@"sel": @"compileWithOptions:error:", @"kind": @"options_error", @"optionsMode": @"nil"},
    ];
    NSError *lastError = nil;
    NSMutableArray<NSString *> *tried = [NSMutableArray array];
    for (NSDictionary<NSString *, id> *candidate in candidates) {
        NSString *name = candidate[@"sel"];
        NSString *kind = candidate[@"kind"];
        NSString *optionsMode = candidate[@"optionsMode"] ?: @"n/a";
        SEL sel = NSSelectorFromString(name);
        if (![model respondsToSelector:sel]) {
            continue;
        }
        [tried addObject:[NSString stringWithFormat:@"%@[%@]", name, optionsMode]];
        NSError *callError = nil;
        BOOL ok = NO;
        if ([kind isEqualToString:@"qos_options"]) {
            id options = [optionsMode isEqualToString:@"nil"] ? nil : @{};
            ok = ANECallBoolOptions(model, sel, kANEQoSUserInitiated, options, &callError);
        } else if ([kind isEqualToString:@"qos_error"]) {
            ok = ANECallBoolError(model, sel, kANEQoSUserInitiated, &callError);
        } else if ([kind isEqualToString:@"options_error"]) {
            id options = [optionsMode isEqualToString:@"nil"] ? nil : @{};
            ok = ANECallBoolOptionsOnly(model, sel, options, &callError);
        }
        if (ok) {
            if (usedSelector) {
                *usedSelector = [NSString stringWithFormat:@"%@[%@]", name, optionsMode];
            }
            return YES;
        }
        lastError = callError ?: lastError;
    }
    if (usedSelector) {
        *usedSelector = tried.count > 0 ? [@"tried=" stringByAppendingString:[tried componentsJoinedByString:@","]] : @"<no matching selectors>";
    }
    if (error) {
        *error = lastError;
    }
    return NO;
}

- (BOOL)invokeLoadOnModel:(id)model
             usedSelector:(NSString **)usedSelector
                    error:(NSError **)error {
    NSArray<NSDictionary<NSString *, id> *> *candidates = @[
        @{@"sel": @"loadWithQoS:options:error:", @"kind": @"qos_options", @"optionsMode": @"dict"},
        @{@"sel": @"loadWithQoS:options:error:", @"kind": @"qos_options", @"optionsMode": @"nil"},
        @{@"sel": @"loadWithQoS:error:", @"kind": @"qos_error"},
        @{@"sel": @"loadWithOptions:error:", @"kind": @"options_error", @"optionsMode": @"dict"},
        @{@"sel": @"loadWithOptions:error:", @"kind": @"options_error", @"optionsMode": @"nil"},
    ];
    NSError *lastError = nil;
    NSMutableArray<NSString *> *tried = [NSMutableArray array];
    for (NSDictionary<NSString *, id> *candidate in candidates) {
        NSString *name = candidate[@"sel"];
        NSString *kind = candidate[@"kind"];
        NSString *optionsMode = candidate[@"optionsMode"] ?: @"n/a";
        SEL sel = NSSelectorFromString(name);
        if (![model respondsToSelector:sel]) {
            continue;
        }
        [tried addObject:[NSString stringWithFormat:@"%@[%@]", name, optionsMode]];
        NSError *callError = nil;
        BOOL ok = NO;
        if ([kind isEqualToString:@"qos_options"]) {
            id options = [optionsMode isEqualToString:@"nil"] ? nil : @{};
            ok = ANECallBoolOptions(model, sel, kANEQoSUserInitiated, options, &callError);
        } else if ([kind isEqualToString:@"qos_error"]) {
            ok = ANECallBoolError(model, sel, kANEQoSUserInitiated, &callError);
        } else if ([kind isEqualToString:@"options_error"]) {
            id options = [optionsMode isEqualToString:@"nil"] ? nil : @{};
            ok = ANECallBoolOptionsOnly(model, sel, options, &callError);
        }
        if (ok) {
            if (usedSelector) {
                *usedSelector = [NSString stringWithFormat:@"%@[%@]", name, optionsMode];
            }
            return YES;
        }
        lastError = callError ?: lastError;
    }
    if (usedSelector) {
        *usedSelector = tried.count > 0 ? [@"tried=" stringByAppendingString:[tried componentsJoinedByString:@","]] : @"<no matching selectors>";
    }
    if (error) {
        *error = lastError;
    }
    return NO;
}

- (void)emitCompileFailureDiagnosticsForStage:(NSString *)stage
                                normalizedMIL:(NSString *)normalizedMIL
                               materializedDir:(NSString *)materializedDir
                               underlyingError:(NSError *)underlyingError {
    NSString *cwd = NSFileManager.defaultManager.currentDirectoryPath ?: @"";
    NSString *dumpRoot = [[cwd stringByAppendingPathComponent:@"build"] stringByAppendingPathComponent:@"compile_dumps"];
    NSString *stamp = [NSString stringWithFormat:@"%.0f", [NSDate.date timeIntervalSince1970]];
    NSString *token = [NSString stringWithFormat:@"%@_%@_%@", stamp, stage ?: @"unknown", NSUUID.UUID.UUIDString];
    NSString *dumpDir = [dumpRoot stringByAppendingPathComponent:token];
    NSError *fsError = nil;
    [NSFileManager.defaultManager createDirectoryAtPath:dumpDir
                            withIntermediateDirectories:YES
                                             attributes:nil
                                                  error:&fsError];
    if (fsError) {
        NSLog(@"ANE compile diagnostics warning: failed to create dump dir: %@", fsError.localizedDescription);
        return;
    }

    if (normalizedMIL.length > 0) {
        NSString *milPath = [dumpDir stringByAppendingPathComponent:@"normalized.mil"];
        [normalizedMIL writeToFile:milPath atomically:YES encoding:NSUTF8StringEncoding error:nil];
    }

    NSString *materializedModelMILPath = nil;
    if (materializedDir.length > 0) {
        materializedModelMILPath = [materializedDir stringByAppendingPathComponent:@"model.mil"];
    }
    if (materializedModelMILPath.length > 0 &&
        [NSFileManager.defaultManager fileExistsAtPath:materializedModelMILPath]) {
        NSString *copyPath = [dumpDir stringByAppendingPathComponent:@"materialized_model.mil"];
        [NSFileManager.defaultManager removeItemAtPath:copyPath error:nil];
        [NSFileManager.defaultManager copyItemAtPath:materializedModelMILPath toPath:copyPath error:nil];
    }

    NSMutableArray<NSString *> *weightLines = [NSMutableArray array];
    NSString *weightsRoot = [materializedDir stringByAppendingPathComponent:@"weights"];
    if (weightsRoot.length > 0 && [NSFileManager.defaultManager fileExistsAtPath:weightsRoot]) {
        NSDirectoryEnumerator *enumerator = [NSFileManager.defaultManager enumeratorAtPath:weightsRoot];
        for (NSString *relativePath in enumerator) {
            NSString *fullPath = [weightsRoot stringByAppendingPathComponent:relativePath];
            BOOL isDir = NO;
            if ([NSFileManager.defaultManager fileExistsAtPath:fullPath isDirectory:&isDir] && !isDir) {
                NSDictionary<NSFileAttributeKey, id> *attrs = [NSFileManager.defaultManager attributesOfItemAtPath:fullPath error:nil];
                unsigned long long size = [attrs[NSFileSize] unsignedLongLongValue];
                [weightLines addObject:[NSString stringWithFormat:@"%@\t%llu", relativePath, size]];
            }
        }
    }
    if (weightLines.count > 0) {
        NSString *manifest = [weightLines componentsJoinedByString:@"\n"];
        NSString *manifestPath = [dumpDir stringByAppendingPathComponent:@"weights_manifest.tsv"];
        [manifest writeToFile:manifestPath atomically:YES encoding:NSUTF8StringEncoding error:nil];
    }

    NSMutableString *summary = [NSMutableString string];
    [summary appendFormat:@"stage=%@\n", stage ?: @"unknown"];
    [summary appendFormat:@"underlying=%@\n", underlyingError.localizedDescription ?: @"<none>"];
    [summary appendFormat:@"materialized_dir=%@\n", materializedDir ?: @"<none>"];
    [summary appendFormat:@"keep_failed_model_dir=%@\n", ANEEnvFlagEnabled("ANE_BG_KEEP_FAILED_MODEL_DIR") ? @"YES" : @"NO"];
    NSString *summaryPath = [dumpDir stringByAppendingPathComponent:@"summary.txt"];
    [summary writeToFile:summaryPath atomically:YES encoding:NSUTF8StringEncoding error:nil];

    if (ANEEnvFlagEnabled("ANE_BG_RUNTIME_SELECTOR_DUMP")) {
        NSMutableString *selectorDump = [NSMutableString string];
        [selectorDump appendString:@"_ANEInMemoryModelDescriptor class methods\n"];
        [selectorDump appendFormat:@"%@\n\n", [ANESelectorNamesForClass(self.descriptorClass, YES) componentsJoinedByString:@"\n"]];
        [selectorDump appendString:@"_ANEInMemoryModelDescriptor instance methods\n"];
        [selectorDump appendFormat:@"%@\n\n", [ANESelectorNamesForClass(self.descriptorClass, NO) componentsJoinedByString:@"\n"]];
        [selectorDump appendString:@"_ANEInMemoryModel class methods\n"];
        [selectorDump appendFormat:@"%@\n\n", [ANESelectorNamesForClass(self.inMemoryModelClass, YES) componentsJoinedByString:@"\n"]];
        [selectorDump appendString:@"_ANEInMemoryModel instance methods\n"];
        [selectorDump appendFormat:@"%@\n", [ANESelectorNamesForClass(self.inMemoryModelClass, NO) componentsJoinedByString:@"\n"]];
        NSString *selectorPath = [dumpDir stringByAppendingPathComponent:@"selector_dump.txt"];
        [selectorDump writeToFile:selectorPath atomically:YES encoding:NSUTF8StringEncoding error:nil];
    }

    NSLog(@"ANE compile diagnostics: stage=%@ dump=%@", stage, dumpDir);
}

- (BOOL)compileMIL:(NSString *)mil weightDirectory:(NSString *)weightDirectory error:(NSError **)error {
    [self unloadCompiledModel];

    if (![self loadPrivateAPI:error]) {
        return NO;
    }
    [self maybeLogRuntimeSelectorIntrospection];
    if (mil.length == 0) {
        if (error) {
            *error = ANEError(ANEByteGridErrorInvalidMIL, @"MIL source is empty.", nil);
        }
        return NO;
    }

    NSMutableDictionary<NSString *, NSDictionary *> *weights = [NSMutableDictionary dictionary];
    NSMutableDictionary<NSString *, NSData *> *materializedFiles = [NSMutableDictionary dictionary];
    NSString *normalizedMIL = [self normalizedMILFromSource:mil
                                            weightDirectory:weightDirectory
                                              weightsOut:weights
                                        materializedFilesOut:materializedFiles
                                                      error:error];
    if (!normalizedMIL) {
        return NO;
    }

    NSData *milData = [normalizedMIL dataUsingEncoding:NSUTF8StringEncoding];
    if (!milData) {
        if (error) {
            *error = ANEError(ANEByteGridErrorInvalidMIL, @"Failed to encode MIL as UTF-8.", nil);
        }
        return NO;
    }

    NSString *materializedDir = [self materializeModelArtifactsForModelID:nil
                                                                  milData:milData
                                                        materializedFiles:materializedFiles
                                                                     error:error];
    if (!materializedDir) {
        return NO;
    }

    SEL perfStatsMaskSelector = @selector(setPerfStatsMask:);

    NSString *descriptorSelector = nil;
    NSError *descriptorError = nil;
    uint64_t start = mach_absolute_time();
    id descriptor = [self createDescriptorForMILData:milData
                                             milText:normalizedMIL
                                             weights:weights
                                        usedSelector:&descriptorSelector
                                               error:&descriptorError];
    if (!descriptor) {
        [self emitCompileFailureDiagnosticsForStage:@"descriptor_rejected"
                                      normalizedMIL:normalizedMIL
                                     materializedDir:materializedDir
                                     underlyingError:descriptorError];
        if (!ANEEnvFlagEnabled("ANE_BG_KEEP_FAILED_MODEL_DIR")) {
            [self cleanupMaterializedModelDirectory:materializedDir];
        }
        if (error) {
            *error = descriptorError ?: ANEError(ANEByteGridErrorCompileFailed,
                                                 @"_ANEInMemoryModelDescriptor rejected the MIL/weight payload.",
                                                 nil);
        }
        return NO;
    }
    if (ANEEnvFlagEnabled("ANE_BG_RUNTIME_SELECTOR_DUMP")) {
        NSLog(@"ANE selector match: descriptor constructor=%@", descriptorSelector ?: @"<unknown>");
    }

    NSString *modelConstructor = nil;
    id model = [self createInMemoryModelFromDescriptor:descriptor usedSelector:&modelConstructor];
    if (!model) {
        [self emitCompileFailureDiagnosticsForStage:@"model_create_failed"
                                      normalizedMIL:normalizedMIL
                                     materializedDir:materializedDir
                                     underlyingError:nil];
        if (!ANEEnvFlagEnabled("ANE_BG_KEEP_FAILED_MODEL_DIR")) {
            [self cleanupMaterializedModelDirectory:materializedDir];
        }
        if (error) {
            *error = ANEError(ANEByteGridErrorCompileFailed, @"_ANEInMemoryModel could not be created from the in-memory descriptor.", nil);
        }
        return NO;
    }
    if (ANEEnvFlagEnabled("ANE_BG_RUNTIME_SELECTOR_DUMP")) {
        NSLog(@"ANE selector match: model constructor=%@", modelConstructor ?: @"<unknown>");
    }

    if ([model respondsToSelector:perfStatsMaskSelector]) {
        ((void (*)(id, SEL, unsigned int))objc_msgSend)(model, perfStatsMaskSelector, UINT32_MAX);
    }

    if ([model respondsToSelector:@selector(setModelURL:)]) {
        NSURL *modelURL = [NSURL fileURLWithPath:materializedDir isDirectory:YES];
        ANECallObjC1(model, @selector(setModelURL:), modelURL);
    }

    NSError *compileError = nil;
    NSString *compileSelectorName = nil;
    BOOL compiled = [self invokeCompileOnModel:model usedSelector:&compileSelectorName error:&compileError];
    self.lastCompileDurationMs = ANEElapsedMs(start, mach_absolute_time());
    if (ANEEnvFlagEnabled("ANE_BG_RUNTIME_SELECTOR_DUMP")) {
        NSLog(@"ANE selector match: compile selector=%@", compileSelectorName ?: @"<none>");
    }
    if (!compiled) {
        NSLog(@"ANE compile underlying error: %@", compileError);
        [self emitCompileFailureDiagnosticsForStage:@"compile_failed"
                                      normalizedMIL:normalizedMIL
                                     materializedDir:materializedDir
                                     underlyingError:compileError];
        if (!ANEEnvFlagEnabled("ANE_BG_KEEP_FAILED_MODEL_DIR")) {
            [self cleanupMaterializedModelDirectory:materializedDir];
        }
        if (error) {
            *error = ANEError(ANEByteGridErrorCompileFailed, @"ANE compilation failed.", compileError);
        }
        return NO;
    }

    start = mach_absolute_time();
    NSError *loadError = nil;
    NSString *loadSelectorName = nil;
    BOOL loaded = [self invokeLoadOnModel:model usedSelector:&loadSelectorName error:&loadError];
    self.lastLoadDurationMs = ANEElapsedMs(start, mach_absolute_time());
    if (ANEEnvFlagEnabled("ANE_BG_RUNTIME_SELECTOR_DUMP")) {
        NSLog(@"ANE selector match: load selector=%@", loadSelectorName ?: @"<none>");
    }
    if (!loaded) {
        NSLog(@"ANE load underlying error: %@", loadError);
        [self emitCompileFailureDiagnosticsForStage:@"load_failed"
                                      normalizedMIL:normalizedMIL
                                     materializedDir:materializedDir
                                     underlyingError:loadError];
        if (!ANEEnvFlagEnabled("ANE_BG_KEEP_FAILED_MODEL_DIR")) {
            [self cleanupMaterializedModelDirectory:materializedDir];
        }
        if (error) {
            *error = ANEError(ANEByteGridErrorLoadFailed, @"ANE model load failed.", loadError);
        }
        return NO;
    }

    self.compiledModel = model;
    self.modelDescriptor = descriptor;
    self.materializedModelDirectory = materializedDir;
    self.normalizedMIL = normalizedMIL;
    self.descriptorWeights = weights;
    self.compiledWeightCount = weights.count;
    self.modelLoaded = YES;
    return YES;
}

- (BOOL)evaluateInputSurface:(IOSurfaceRef)input outputSurface:(IOSurfaceRef)output error:(NSError **)error {
    if (!self.modelLoaded || !self.compiledModel) {
        if (error) {
            *error = ANEError(ANEByteGridErrorEvaluateFailed, @"Model must be compiled and loaded before evaluation.", nil);
        }
        return NO;
    }
    if (!input || !output) {
        if (error) {
            *error = ANEError(ANEByteGridErrorInvalidSurface, @"Input and output IOSurfaces are required.", nil);
        }
        return NO;
    }
    if (!self.privateAPIAvailable) {
        if (error) {
            *error = ANEError(ANEByteGridErrorUnavailable, @"AppleNeuralEngine private APIs are unavailable on this OS build.", nil);
        }
        return NO;
    }

    SEL wrapSurfaceSelector = @selector(objectWithIOSurface:);
    SEL requestSelector = @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:);
    SEL validateSelector = @selector(validate);
    SEL evaluateSelector = @selector(evaluateWithQoS:options:request:error:);

    id inputObject = ((id (*)(id, SEL, IOSurfaceRef))objc_msgSend)(self.ioSurfaceClass, wrapSurfaceSelector, input);
    id outputObject = ((id (*)(id, SEL, IOSurfaceRef))objc_msgSend)(self.ioSurfaceClass, wrapSurfaceSelector, output);
    if (!inputObject || !outputObject) {
        if (error) {
            *error = ANEError(ANEByteGridErrorRequestFailed, @"Failed to wrap IOSurfaces for ANE submission.", nil);
        }
        return NO;
    }

    id requestObject = ANECallObjC7(self.requestClass,
                                    requestSelector,
                                    @[ inputObject ],
                                    @[ @0 ],
                                    @[ outputObject ],
                                    @[ @0 ],
                                    nil,
                                    nil,
                                    @0);
    if (!requestObject) {
        if (error) {
            *error = ANEError(ANEByteGridErrorRequestFailed, @"_ANERequest creation failed.", nil);
        }
        return NO;
    }
    if ([requestObject respondsToSelector:validateSelector] &&
        !((BOOL (*)(id, SEL))objc_msgSend)(requestObject, validateSelector)) {
        if (error) {
            *error = ANEError(ANEByteGridErrorRequestFailed, @"_ANERequest validation failed.", nil);
        }
        return NO;
    }

    NSError *evaluateError = nil;
    uint64_t start = mach_absolute_time();
    BOOL ok = ANECallBoolRequest(self.compiledModel, evaluateSelector, kANEQoSUserInitiated, @{}, requestObject, &evaluateError);
    self.lastEvaluationDurationMs = ANEElapsedMs(start, mach_absolute_time());

    double totalBytes = (double)IOSurfaceGetAllocSize(input) + (double)IOSurfaceGetAllocSize(output);
    if (self.lastEvaluationDurationMs > 0.0) {
        self.lastThroughputMBPerSecond = (totalBytes / (1024.0 * 1024.0)) / (self.lastEvaluationDurationMs / 1000.0);
    } else {
        self.lastThroughputMBPerSecond = 0.0;
    }

    if (!ok) {
        NSLog(@"ANE evaluate underlying error: %@", evaluateError);
        if (error) {
            *error = ANEError(ANEByteGridErrorEvaluateFailed, @"ANE evaluation failed.", evaluateError);
        }
        return NO;
    }

    return YES;
}

- (BOOL)loadPrivateAPI:(NSError **)error {
    if (self.privateAPIAvailable && self.descriptorClass && self.inMemoryModelClass && self.requestClass && self.ioSurfaceClass) {
        return YES;
    }

    NSArray<NSString *> *candidatePaths = @[
        @"/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",
        @"/System/Library/PrivateFrameworks/AppleNeuralEngine.framework",
    ];
    void *handle = NULL;
    for (NSString *path in candidatePaths) {
        handle = dlopen(path.fileSystemRepresentation, RTLD_NOW | RTLD_LOCAL);
        if (handle) {
            break;
        }
    }

    if (!handle) {
        NSBundle *bundle = [NSBundle bundleWithPath:@"/System/Library/PrivateFrameworks/AppleNeuralEngine.framework"];
        NSError *bundleError = nil;
        if (![bundle loadAndReturnError:&bundleError]) {
            if (error) {
                NSString *message = @"AppleNeuralEngine.framework could not be loaded. This macOS build does not expose the required private runtime.";
                *error = ANEError(ANEByteGridErrorUnavailable, message, bundleError);
            }
            self.privateAPIAvailable = NO;
            return NO;
        }
        self.aneBundle = bundle;
    }

    self.descriptorClass = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    self.inMemoryModelClass = NSClassFromString(@"_ANEInMemoryModel");
    self.requestClass = NSClassFromString(@"_ANERequest");
    self.ioSurfaceClass = NSClassFromString(@"_ANEIOSurfaceObject");
    self.privateAPIAvailable = (self.descriptorClass &&
                                self.inMemoryModelClass &&
                                self.requestClass &&
                                self.ioSurfaceClass);

    if (!self.privateAPIAvailable && error) {
        *error = ANEError(ANEByteGridErrorUnavailable,
                          @"AppleNeuralEngine loaded, but one or more required private classes are missing on this OS version.",
                          nil);
    }
    return self.privateAPIAvailable;
}

- (NSString *)normalizedMILFromSource:(NSString *)mil
                      weightDirectory:(NSString *)weightDirectory
                            weightsOut:(NSMutableDictionary<NSString *, NSDictionary *> *)weightsOut
                  materializedFilesOut:(NSMutableDictionary<NSString *, NSData *> *)materializedFilesOut
                                error:(NSError **)error {
    NSString *normalized = [mil copy];
    NSFileManager *fm = NSFileManager.defaultManager;
    NSString *baseDir = weightDirectory.stringByStandardizingPath;
    BOOL isDir = NO;
    if (baseDir.length == 0 || ![fm fileExistsAtPath:baseDir isDirectory:&isDir] || !isDir) {
        if (error) {
            *error = ANEError(ANEByteGridErrorWeightLoadFailed,
                              [NSString stringWithFormat:@"Weight directory not found: %@", weightDirectory ?: @"<nil>"],
                              nil);
        }
        return nil;
    }

    NSArray<NSString *> *blobPaths = ane_bg_extract_blobfile_paths(mil);
    for (NSString *path in blobPaths) {
        NSString *relativePath = nil;
        NSString *absolutePath = nil;
        NSString *descriptorPath = nil;

        if ([path hasPrefix:@"@model_path/weights/"]) {
            descriptorPath = path;
            relativePath = [path substringFromIndex:[@"@model_path/weights/" length]];
            absolutePath = [baseDir stringByAppendingPathComponent:relativePath];
        } else {
            absolutePath = path.stringByStandardizingPath;
            if (![absolutePath hasPrefix:baseDir]) {
                if (error) {
                    *error = ANEError(ANEByteGridErrorWeightLoadFailed,
                                      [NSString stringWithFormat:@"BLOBFILE path %@ is outside weight directory %@", absolutePath, baseDir],
                                      nil);
                }
                return nil;
            }

            relativePath = [absolutePath substringFromIndex:baseDir.length];
            if ([relativePath hasPrefix:@"/"]) {
                relativePath = [relativePath substringFromIndex:1];
            }
            descriptorPath = [@"@model_path/weights" stringByAppendingPathComponent:relativePath];
        }

        if (relativePath.length == 0) {
            if (error) {
                *error = ANEError(ANEByteGridErrorWeightLoadFailed,
                                  [NSString stringWithFormat:@"Failed to resolve relative BLOBFILE path for %@", path],
                                  nil);
            }
            return nil;
        }

        BOOL isSubdir = NO;
        if (![fm fileExistsAtPath:absolutePath isDirectory:&isSubdir] || isSubdir) {
            if (error) {
                *error = ANEError(ANEByteGridErrorWeightLoadFailed,
                                  [NSString stringWithFormat:@"Referenced weight blob not found: %@", absolutePath],
                                  nil);
            }
            return nil;
        }

        NSData *data = [NSData dataWithContentsOfFile:absolutePath options:NSDataReadingMappedIfSafe error:nil];
        if (!data) {
            if (error) {
                *error = ANEError(ANEByteGridErrorWeightLoadFailed,
                                  [NSString stringWithFormat:@"Failed to read weight blob %@", absolutePath],
                                  nil);
            }
            return nil;
        }

        weightsOut[descriptorPath] = @{@"offset": @0, @"data": data};
        materializedFilesOut[relativePath] = data;

        normalized = [normalized stringByReplacingOccurrencesOfString:absolutePath withString:descriptorPath];
        if (![descriptorPath isEqualToString:path]) {
            normalized = [normalized stringByReplacingOccurrencesOfString:[baseDir stringByAppendingPathComponent:relativePath]
                                                               withString:descriptorPath];
        }
    }

    return normalized;
}

- (NSString *)materializeModelArtifactsForModelID:(NSString *)modelID
                                          milData:(NSData *)milData
                                materializedFiles:(NSDictionary<NSString *, NSData *> *)materializedFiles
                                             error:(NSError **)error {
    NSString *token = modelID.length > 0 ? modelID : NSUUID.UUID.UUIDString;
    NSString *root = [NSTemporaryDirectory() stringByAppendingPathComponent:token];
    NSFileManager *fm = NSFileManager.defaultManager;
    NSString *weightsRoot = [root stringByAppendingPathComponent:@"weights"];
    NSError *fsError = nil;
    if (![fm createDirectoryAtPath:weightsRoot withIntermediateDirectories:YES attributes:nil error:&fsError]) {
        if (error) {
            *error = ANEError(ANEByteGridErrorCompileFailed, @"Failed to materialize temporary ANE model directory.", fsError);
        }
        return nil;
    }
    if (![milData writeToFile:[root stringByAppendingPathComponent:@"model.mil"] options:NSDataWritingAtomic error:&fsError]) {
        [self cleanupMaterializedModelDirectory:root];
        if (error) {
            *error = ANEError(ANEByteGridErrorCompileFailed, @"Failed to write temporary model.mil for ANE compilation.", fsError);
        }
        return nil;
    }

    for (NSString *relativePath in materializedFiles) {
        NSString *fullPath = [weightsRoot stringByAppendingPathComponent:relativePath];
        NSString *dir = [fullPath stringByDeletingLastPathComponent];
        if (![fm createDirectoryAtPath:dir withIntermediateDirectories:YES attributes:nil error:&fsError]) {
            [self cleanupMaterializedModelDirectory:root];
            if (error) {
                *error = ANEError(ANEByteGridErrorCompileFailed, @"Failed to create temporary weight directory.", fsError);
            }
            return nil;
        }
        if (![materializedFiles[relativePath] writeToFile:fullPath options:NSDataWritingAtomic error:&fsError]) {
            [self cleanupMaterializedModelDirectory:root];
            if (error) {
                *error = ANEError(ANEByteGridErrorCompileFailed, @"Failed to write temporary weight blob for ANE compilation.", fsError);
            }
            return nil;
        }
    }

    return root;
}

- (void)cleanupMaterializedModelDirectory:(NSString *)path {
    if (path.length == 0) {
        return;
    }
    [NSFileManager.defaultManager removeItemAtPath:path error:nil];
}

- (void)unloadCompiledModel {
    if (self.compiledModel && [self.compiledModel respondsToSelector:@selector(unloadWithQoS:error:)]) {
        NSError *error = nil;
        ANECallBoolError(self.compiledModel, @selector(unloadWithQoS:error:), kANEQoSUserInitiated, &error);
        if (error) {
            NSLog(@"ANE unload warning: %@", error.localizedDescription);
        }
    }
    [self cleanupMaterializedModelDirectory:self.materializedModelDirectory];
    self.compiledModel = nil;
    self.modelDescriptor = nil;
    self.materializedModelDirectory = nil;
    self.normalizedMIL = nil;
    self.descriptorWeights = nil;
    self.modelLoaded = NO;
    self.lastCompileDurationMs = 0.0;
    self.lastLoadDurationMs = 0.0;
    self.lastEvaluationDurationMs = 0.0;
    self.lastThroughputMBPerSecond = 0.0;
    self.compiledWeightCount = 0;
}

@end
