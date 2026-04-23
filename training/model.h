#ifndef ANE_BG_MODEL_H
#define ANE_BG_MODEL_H

#import <Foundation/Foundation.h>

typedef NS_ENUM(uint32_t, ANEBlobDataType) {
    ANEBlobDataTypeFloat16 = 1,
    ANEBlobDataTypeFloat32 = 2,
    ANEBlobDataTypeUInt8 = 3,
    ANEBlobDataTypeInt8 = 8,
};

@interface ANEBlobWriter : NSObject
- (NSInteger)addFloat16:(const float *)data count:(size_t)count;
- (NSInteger)addFloat32:(const float *)data count:(size_t)count;
- (NSInteger)addRawWithType:(ANEBlobDataType)type bytes:(NSData *)bytes;
- (uint64_t)offsetForBlobAtIndex:(NSInteger)index;
- (NSInteger)count;
- (NSData *)build:(NSError **)error;
@end

BOOL ane_bg_write_weight_blob(NSString *path, const uint16_t *data, size_t element_count, NSError **error);
uint16_t ane_bg_fp16_from_float(float value);
void ane_bg_fill_xavier_uniform(uint16_t *dst, size_t rows, size_t cols, uint32_t seed);
BOOL ane_bg_generate_all_weight_blobs(NSString *weight_dir, NSError **error);

#endif
