#ifndef ANE_BG_MIL_GEN_H
#define ANE_BG_MIL_GEN_H

#import <Foundation/Foundation.h>

FOUNDATION_EXPORT NSString *const ANE_BG_MIL_BUILD_INFO_HEADER;

NSString *ane_bg_gen_conv_fp16_exact(NSUInteger inCh, NSUInteger outCh, NSUInteger spatial);
NSString *ane_bg_gen_conv_dynamic_fp16_exact(NSUInteger inCh, NSUInteger outCh, NSUInteger spatial);
NSString *ane_bg_gen_stem_mil(NSString *weight_path);
NSString *ane_bg_gen_local_mixer_mil(NSString *weight_path);
NSString *ane_bg_gen_global_mixer_mil(NSString *weight_path);
NSString *ane_bg_gen_channel_glu_mil(NSString *weight_dir);
NSString *ane_bg_gen_block_mil(NSString *weight_dir, NSInteger layer);
NSString *ane_bg_gen_head_mil(NSString *weight_path);
NSString *ane_bg_gen_full_model_mil(NSString *weight_dir);

#endif
