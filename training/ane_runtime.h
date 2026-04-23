#ifndef ANE_BG_RUNTIME_H
#define ANE_BG_RUNTIME_H

#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>

FOUNDATION_EXPORT NSArray<NSString *> *ane_bg_extract_blobfile_paths(NSString *mil);

@interface ANEByteGridRuntime : NSObject
@property (nonatomic, readonly) BOOL privateAPIAvailable;
@property (nonatomic, readonly) BOOL modelLoaded;
@property (nonatomic, readonly) double lastCompileDurationMs;
@property (nonatomic, readonly) double lastLoadDurationMs;
@property (nonatomic, readonly) double lastEvaluationDurationMs;
@property (nonatomic, readonly) double lastThroughputMBPerSecond;
@property (nonatomic, readonly) NSUInteger compiledWeightCount;
- (instancetype)init;
- (BOOL)compileMIL:(NSString *)mil weightDirectory:(NSString *)weightDirectory error:(NSError **)error;
- (BOOL)evaluateInputSurface:(IOSurfaceRef)input outputSurface:(IOSurfaceRef)output error:(NSError **)error;
@end

#endif
