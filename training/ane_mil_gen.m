#import "ane_mil_gen.h"

NSString *const ANE_BG_MIL_BUILD_INFO_HEADER =
    @"program(1.3)\n"
    "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
    "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
    "{\"coremltools-version\", \"9.0\"}})]\n";

static const unsigned long long kANEWeightBlobMILOffset = 64ULL;

static NSString *ane_bg_conv_prelude(void) {
    return
    @"        string pt = const()[name=string(\"pt\"), val=string(\"valid\")];\n"
    "        tensor<int32, [2]> st = const()[name=string(\"st\"), val=tensor<int32, [2]>([1,1])];\n"
    "        tensor<int32, [4]> pd = const()[name=string(\"pd\"), val=tensor<int32, [4]>([0,0,0,0])];\n"
    "        tensor<int32, [2]> dl = const()[name=string(\"dl\"), val=tensor<int32, [2]>([1,1])];\n";
}

static NSMutableString *ane_bg_program_with_main_header(NSString *signature) {
    NSMutableString *mil = [NSMutableString stringWithString:ANE_BG_MIL_BUILD_INFO_HEADER];
    [mil appendString:@"{\n"];
    [mil appendFormat:@"    func main<ios18>(%@) {\n", signature];
    return mil;
}

static void ane_bg_append_rmsnorm(NSMutableString *mil,
                                  NSString *prefix,
                                  NSString *inputVar,
                                  NSString *weightPath,
                                  NSString *outputVar) {
    [mil appendFormat:@"        tensor<fp16, [1,512,1,256]> %@_sq = mul(x=%@, y=%@)[name=string(\"%@_sq\")];\n",
                      prefix, inputVar, inputVar, prefix];
    [mil appendFormat:@"        tensor<int32, [1]> %@_rax = const()[name=string(\"%@_rax\"), val=tensor<int32, [1]>([1])];\n",
                      prefix, prefix];
    [mil appendFormat:@"        bool %@_kd = const()[name=string(\"%@_kd\"), val=bool(true)];\n",
                      prefix, prefix];
    [mil appendFormat:@"        tensor<fp16, [1,1,1,256]> %@_ss = reduce_sum(x=%@_sq, axes=%@_rax, keep_dims=%@_kd)[name=string(\"%@_ss\")];\n",
                      prefix, prefix, prefix, prefix, prefix];
    [mil appendFormat:@"        fp16 %@_invd = const()[name=string(\"%@_invd\"), val=fp16(0.001953125)];\n",
                      prefix, prefix];
    [mil appendFormat:@"        tensor<fp16, [1,1,1,256]> %@_mean = mul(x=%@_ss, y=%@_invd)[name=string(\"%@_mean\")];\n",
                      prefix, prefix, prefix, prefix];
    [mil appendFormat:@"        fp16 %@_eps = const()[name=string(\"%@_eps\"), val=fp16(0.00001)];\n",
                      prefix, prefix];
    [mil appendFormat:@"        tensor<fp16, [1,1,1,256]> %@_var = add(x=%@_mean, y=%@_eps)[name=string(\"%@_var\")];\n",
                      prefix, prefix, prefix, prefix];
    [mil appendFormat:@"        fp16 %@_nhalf = const()[name=string(\"%@_nhalf\"), val=fp16(-0.5)];\n",
                      prefix, prefix];
    [mil appendFormat:@"        tensor<fp16, [1,1,1,256]> %@_rrms = pow(x=%@_var, y=%@_nhalf)[name=string(\"%@_rrms\")];\n",
                      prefix, prefix, prefix, prefix];
    [mil appendFormat:@"        tensor<fp16, [1,512,1,256]> %@_scaled = mul(x=%@, y=%@_rrms)[name=string(\"%@_scaled\")];\n",
                      prefix, inputVar, prefix, prefix];
    [mil appendFormat:@"        tensor<fp16, [1,512,1,1]> %@_rw = const()[name=string(\"%@_rw\"), val=tensor<fp16, [1,512,1,1]>(BLOBFILE(path=string(\"@model_path/weights/%@\"), offset=uint64(64)))];\n",
                      prefix, prefix, weightPath];
    [mil appendFormat:@"        tensor<fp16, [1,512,1,256]> %@ = mul(x=%@_scaled, y=%@_rw)[name=string(\"%@\")];\n",
                      outputVar, prefix, prefix, outputVar];
}

NSString *ane_bg_gen_conv_fp16_exact(NSUInteger inCh, NSUInteger outCh, NSUInteger spatial) {
    return [NSString stringWithFormat:
            @"%@"
            "{\n"
            "    func main<ios18>(tensor<fp16, [1, %lu, 1, %lu]> x) {\n"
            "        string c_pad_type = const()[name = string(\"c_pad_type\"), val = string(\"valid\")];\n"
            "        tensor<int32, [2]> c_strides = const()[name = string(\"c_strides\"), val = tensor<int32, [2]>([1, 1])];\n"
            "        tensor<int32, [4]> c_pad = const()[name = string(\"c_pad\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
            "        tensor<int32, [2]> c_dilations = const()[name = string(\"c_dilations\"), val = tensor<int32, [2]>([1, 1])];\n"
            "        int32 c_groups = const()[name = string(\"c_groups\"), val = int32(1)];\n"
            "        tensor<fp16, [%lu, %lu, 1, 1]> W = const()[name = string(\"W\"), val = tensor<fp16, [%lu, %lu, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(64)))];\n"
            "        tensor<fp16, [1, %lu, 1, %lu]> y = conv(dilations = c_dilations, groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = W, x = x)[name = string(\"conv\")];\n"
            "    } -> (y);\n"
            "}\n",
            ANE_BG_MIL_BUILD_INFO_HEADER,
            (unsigned long)inCh, (unsigned long)spatial,
            (unsigned long)outCh, (unsigned long)inCh, (unsigned long)outCh, (unsigned long)inCh,
            (unsigned long)outCh, (unsigned long)spatial];
}

NSString *ane_bg_gen_conv_dynamic_fp16_exact(NSUInteger inCh, NSUInteger outCh, NSUInteger spatial) {
    return [NSString stringWithFormat:
            @"%@"
            "{\n"
            "    func main<ios18>(tensor<fp16, [1, %lu, 1, %lu]> x, tensor<fp16, [%lu, %lu, 1, 1]> W) {\n"
            "        string c_pad_type = const()[name = string(\"c_pad_type\"), val = string(\"valid\")];\n"
            "        tensor<int32, [2]> c_strides = const()[name = string(\"c_strides\"), val = tensor<int32, [2]>([1, 1])];\n"
            "        tensor<int32, [4]> c_pad = const()[name = string(\"c_pad\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
            "        tensor<int32, [2]> c_dilations = const()[name = string(\"c_dilations\"), val = tensor<int32, [2]>([1, 1])];\n"
            "        int32 c_groups = const()[name = string(\"c_groups\"), val = int32(1)];\n"
            "        tensor<fp16, [1, %lu, 1, %lu]> y = conv(dilations = c_dilations, groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = W, x = x)[name = string(\"conv\")];\n"
            "    } -> (y);\n"
            "}\n",
            ANE_BG_MIL_BUILD_INFO_HEADER,
            (unsigned long)inCh, (unsigned long)spatial, (unsigned long)outCh, (unsigned long)inCh,
            (unsigned long)outCh, (unsigned long)spatial];
}

NSString *ane_bg_gen_stem_mil(NSString *weight_path) {
    (void)weight_path;
    return [NSString stringWithFormat:
            @"%@"
            "{\n"
            "    func main<ios18>(tensor<fp16, [1, 320, 1, 256]> x) {\n"
            "        string c_pad_type = const()[name = string(\"c_pad_type\"), val = string(\"valid\")];\n"
            "        tensor<int32, [2]> c_strides = const()[name = string(\"c_strides\"), val = tensor<int32, [2]>([1, 1])];\n"
            "        tensor<int32, [4]> c_pad = const()[name = string(\"c_pad\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
            "        tensor<int32, [2]> c_dilations = const()[name = string(\"c_dilations\"), val = tensor<int32, [2]>([1, 1])];\n"
            "        int32 c_groups = const()[name = string(\"c_groups\"), val = int32(1)];\n"
            "        tensor<fp16, [512, 320, 1, 1]> W = const()[name = string(\"W\"), val = tensor<fp16, [512, 320, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/stem.bin\"), offset = uint64(64)))];\n"
            "        tensor<fp16, [1, 512, 1, 256]> y = conv(dilations = c_dilations, groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = W, x = x)[name = string(\"stem\")];\n"
            "    } -> (y);\n"
            "}\n",
            ANE_BG_MIL_BUILD_INFO_HEADER];
}

NSString *ane_bg_gen_local_mixer_mil(NSString *weight_path) {
    (void)weight_path;
    NSMutableString *mil = ane_bg_program_with_main_header(@"tensor<fp16, [1, 512, 1, 256]> x");
    [mil appendString:@"        tensor<int32, [4]> sh0 = const()[name=string(\"sh0\"), val=tensor<int32, [4]>([1,512,16,16])];\n"];
    [mil appendString:@"        tensor<fp16, [1,512,16,16]> x4 = reshape(shape=sh0, x=x)[name=string(\"x4\")];\n"];
    [mil appendString:@"        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];\n"];
    [mil appendString:@"        tensor<fp16, [1,512,16,16]> xt = transpose(perm=pm, x=x4)[name=string(\"xt\")];\n"];
    [mil appendString:@"        tensor<int32, [4]> sh1 = const()[name=string(\"sh1\"), val=tensor<int32, [4]>([1,8192,1,16])];\n"];
    [mil appendString:@"        tensor<fp16, [1,8192,1,16]> xp = reshape(shape=sh1, x=xt)[name=string(\"xp\")];\n"];
    [mil appendString:ane_bg_conv_prelude()];
    [mil appendString:@"        int32 gr = const()[name=string(\"gr\"), val=int32(512)];\n"];
    [mil appendString:@"        tensor<fp16, [8192,16,1,1]> W = const()[name=string(\"W\"), val=tensor<fp16, [8192,16,1,1]>(BLOBFILE(path=string(\"@model_path/weights/block_00_local.bin\"), offset=uint64(64)))];\n"];
    [mil appendString:@"        tensor<fp16, [1,8192,1,16]> ym = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=W, x=xp)[name=string(\"mix\")];\n"];
    [mil appendString:@"        tensor<fp16, [1,512,16,16]> y4 = reshape(shape=sh0, x=ym)[name=string(\"y4\")];\n"];
    [mil appendString:@"        tensor<fp16, [1,512,16,16]> yi = transpose(perm=pm, x=y4)[name=string(\"yi\")];\n"];
    [mil appendString:@"        tensor<int32, [4]> sho = const()[name=string(\"sho\"), val=tensor<int32, [4]>([1,512,1,256])];\n"];
    [mil appendString:@"        tensor<fp16, [1,512,1,256]> y = reshape(shape=sho, x=yi)[name=string(\"out\")];\n"];
    [mil appendString:@"    } -> (y);\n}\n"];
    return mil;
}

NSString *ane_bg_gen_global_mixer_mil(NSString *weight_path) {
    (void)weight_path;
    NSMutableString *mil = ane_bg_program_with_main_header(@"tensor<fp16, [1, 512, 1, 256]> x");
    [mil appendString:@"        tensor<int32, [4]> sh0 = const()[name=string(\"sh0\"), val=tensor<int32, [4]>([1,512,16,16])];\n"];
    [mil appendString:@"        tensor<fp16, [1,512,16,16]> x4 = reshape(shape=sh0, x=x)[name=string(\"x4\")];\n"];
    [mil appendString:@"        tensor<int32, [4]> sh1 = const()[name=string(\"sh1\"), val=tensor<int32, [4]>([1,8192,1,16])];\n"];
    [mil appendString:@"        tensor<fp16, [1,8192,1,16]> xp = reshape(shape=sh1, x=x4)[name=string(\"xp\")];\n"];
    [mil appendString:ane_bg_conv_prelude()];
    [mil appendString:@"        int32 gr = const()[name=string(\"gr\"), val=int32(512)];\n"];
    [mil appendString:@"        tensor<fp16, [8192,16,1,1]> W = const()[name=string(\"W\"), val=tensor<fp16, [8192,16,1,1]>(BLOBFILE(path=string(\"@model_path/weights/block_00_global.bin\"), offset=uint64(64)))];\n"];
    [mil appendString:@"        tensor<fp16, [1,8192,1,16]> ym = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=W, x=xp)[name=string(\"mix\")];\n"];
    [mil appendString:@"        tensor<int32, [4]> sho = const()[name=string(\"sho\"), val=tensor<int32, [4]>([1,512,1,256])];\n"];
    [mil appendString:@"        tensor<fp16, [1,512,1,256]> y = reshape(shape=sho, x=ym)[name=string(\"out\")];\n"];
    [mil appendString:@"    } -> (y);\n}\n"];
    return mil;
}

NSString *ane_bg_gen_channel_glu_mil(NSString *weight_dir) {
    (void)weight_dir;
    NSMutableString *mil = ane_bg_program_with_main_header(@"tensor<fp16, [1, 512, 1, 256]> x");
    [mil appendString:ane_bg_conv_prelude()];
    [mil appendString:@"        int32 gr = const()[name=string(\"gr\"), val=int32(1)];\n"];
    [mil appendString:@"        tensor<fp16, [1024,512,1,1]> Wv = const()[name=string(\"Wv\"), val=tensor<fp16, [1024,512,1,1]>(BLOBFILE(path=string(\"@model_path/weights/wv.bin\"), offset=uint64(64)))];\n"];
    [mil appendString:@"        tensor<fp16, [1024,512,1,1]> Wg = const()[name=string(\"Wg\"), val=tensor<fp16, [1024,512,1,1]>(BLOBFILE(path=string(\"@model_path/weights/wg.bin\"), offset=uint64(64)))];\n"];
    [mil appendString:@"        tensor<fp16, [512,1024,1,1]> Wo = const()[name=string(\"Wo\"), val=tensor<fp16, [512,1024,1,1]>(BLOBFILE(path=string(\"@model_path/weights/wo.bin\"), offset=uint64(64)))];\n"];
    [mil appendString:@"        tensor<fp16, [1,1024,1,256]> h1 = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=Wv, x=x)[name=string(\"c1\")];\n"];
    [mil appendString:@"        tensor<fp16, [1,1024,1,256]> h3 = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=Wg, x=x)[name=string(\"c3\")];\n"];
    [mil appendString:@"        tensor<fp16, [1,1024,1,256]> sig = sigmoid(x=h1)[name=string(\"sg\")];\n"];
    [mil appendString:@"        tensor<fp16, [1,1024,1,256]> silu = mul(x=h1, y=sig)[name=string(\"si\")];\n"];
    [mil appendString:@"        tensor<fp16, [1,1024,1,256]> gate = mul(x=silu, y=h3)[name=string(\"gt\")];\n"];
    [mil appendString:@"        tensor<fp16, [1,512,1,256]> y = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=Wo, x=gate)[name=string(\"out\")];\n"];
    [mil appendString:@"    } -> (y);\n}\n"];
    return mil;
}

NSString *ane_bg_gen_block_mil(NSString *weight_dir, NSInteger layer) {
    (void)weight_dir;
    NSString *local = [NSString stringWithFormat:@"block_%02ld_local.bin", (long)layer];
    NSString *global = [NSString stringWithFormat:@"block_%02ld_global.bin", (long)layer];
    NSString *wv = [NSString stringWithFormat:@"block_%02ld_wv.bin", (long)layer];
    NSString *wg = [NSString stringWithFormat:@"block_%02ld_wg.bin", (long)layer];
    NSString *wo = [NSString stringWithFormat:@"block_%02ld_wo.bin", (long)layer];
    NSString *rmsLocal = [NSString stringWithFormat:@"block_%02ld_rms_local.bin", (long)layer];
    NSString *rmsGlobal = [NSString stringWithFormat:@"block_%02ld_rms_global.bin", (long)layer];
    NSString *rmsFFN = [NSString stringWithFormat:@"block_%02ld_rms_ffn.bin", (long)layer];
    NSString *alphaLocal = [NSString stringWithFormat:@"block_%02ld_alpha_local.bin", (long)layer];
    NSString *alphaGlobal = [NSString stringWithFormat:@"block_%02ld_alpha_global.bin", (long)layer];
    NSString *alphaMLP = [NSString stringWithFormat:@"block_%02ld_alpha_mlp.bin", (long)layer];

    NSMutableString *mil = ane_bg_program_with_main_header(@"tensor<fp16, [1, 512, 1, 256]> x");
    [mil appendString:ane_bg_conv_prelude()];
    [mil appendString:@"        int32 one = const()[name=string(\"one\"), val=int32(1)];\n"];
    [mil appendString:@"        int32 groups = const()[name=string(\"groups\"), val=int32(512)];\n"];
    [mil appendString:@"        tensor<int32, [4]> sh0 = const()[name=string(\"sh0\"), val=tensor<int32, [4]>([1,512,16,16])];\n"];
    [mil appendString:@"        tensor<int32, [4]> sh1 = const()[name=string(\"sh1\"), val=tensor<int32, [4]>([1,8192,1,16])];\n"];
    [mil appendString:@"        tensor<int32, [4]> sho = const()[name=string(\"sho\"), val=tensor<int32, [4]>([1,512,1,256])];\n"];
    [mil appendString:@"        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];\n"];
    [mil appendFormat:@"        tensor<fp16, [8192,16,1,1]> localW = const()[name=string(\"localW\"), val=tensor<fp16, [8192,16,1,1]>(BLOBFILE(path=string(\"@model_path/weights/%@\"), offset=uint64(64)))];\n", local];
    [mil appendFormat:@"        tensor<fp16, [8192,16,1,1]> globalW = const()[name=string(\"globalW\"), val=tensor<fp16, [8192,16,1,1]>(BLOBFILE(path=string(\"@model_path/weights/%@\"), offset=uint64(64)))];\n", global];
    [mil appendFormat:@"        tensor<fp16, [1024,512,1,1]> Wv = const()[name=string(\"Wv\"), val=tensor<fp16, [1024,512,1,1]>(BLOBFILE(path=string(\"@model_path/weights/%@\"), offset=uint64(64)))];\n", wv];
    [mil appendFormat:@"        tensor<fp16, [1024,512,1,1]> Wg = const()[name=string(\"Wg\"), val=tensor<fp16, [1024,512,1,1]>(BLOBFILE(path=string(\"@model_path/weights/%@\"), offset=uint64(64)))];\n", wg];
    [mil appendFormat:@"        tensor<fp16, [512,1024,1,1]> Wo = const()[name=string(\"Wo\"), val=tensor<fp16, [512,1024,1,1]>(BLOBFILE(path=string(\"@model_path/weights/%@\"), offset=uint64(64)))];\n", wo];
    [mil appendFormat:@"        tensor<fp16, [1,1,1,1]> alphaLocal = const()[name=string(\"alphaLocal\"), val=tensor<fp16, [1,1,1,1]>(BLOBFILE(path=string(\"@model_path/weights/%@\"), offset=uint64(64)))];\n", alphaLocal];
    [mil appendFormat:@"        tensor<fp16, [1,1,1,1]> alphaGlobal = const()[name=string(\"alphaGlobal\"), val=tensor<fp16, [1,1,1,1]>(BLOBFILE(path=string(\"@model_path/weights/%@\"), offset=uint64(64)))];\n", alphaGlobal];
    [mil appendFormat:@"        tensor<fp16, [1,1,1,1]> alphaMLP = const()[name=string(\"alphaMLP\"), val=tensor<fp16, [1,1,1,1]>(BLOBFILE(path=string(\"@model_path/weights/%@\"), offset=uint64(64)))];\n", alphaMLP];
    ane_bg_append_rmsnorm(mil, @"local", @"x", rmsLocal, @"u");
    [mil appendString:@"        tensor<fp16, [1,512,16,16]> x4 = reshape(shape=sh0, x=u)[name=string(\"x4\")];\n"];
    [mil appendString:@"        tensor<fp16, [1,512,16,16]> xt = transpose(perm=pm, x=x4)[name=string(\"xt\")];\n"];
    [mil appendString:@"        tensor<fp16, [1,8192,1,16]> xp = reshape(shape=sh1, x=xt)[name=string(\"xp\")];\n"];
    [mil appendString:@"        tensor<fp16, [1,8192,1,16]> yl = conv(dilations=dl, groups=groups, pad=pd, pad_type=pt, strides=st, weight=localW, x=xp)[name=string(\"yl\")];\n"];
    [mil appendString:@"        tensor<fp16, [1,512,16,16]> y4 = reshape(shape=sh0, x=yl)[name=string(\"y4\")];\n"];
    [mil appendString:@"        tensor<fp16, [1,512,16,16]> yi = transpose(perm=pm, x=y4)[name=string(\"yi\")];\n"];
    [mil appendString:@"        tensor<fp16, [1,512,1,256]> localOut = reshape(shape=sho, x=yi)[name=string(\"local\")];\n"];
    [mil appendString:@"        tensor<fp16, [1,512,1,256]> localScaled = mul(x=localOut, y=alphaLocal)[name=string(\"local_scaled\")];\n"];
    [mil appendString:@"        tensor<fp16, [1,512,1,256]> hLocal = add(x=x, y=localScaled)[name=string(\"local_res\")];\n"];
    ane_bg_append_rmsnorm(mil, @"global", @"hLocal", rmsGlobal, @"v");
    [mil appendString:@"        tensor<fp16, [1,512,16,16]> gx4 = reshape(shape=sh0, x=v)[name=string(\"gx4\")];\n"];
    [mil appendString:@"        tensor<fp16, [1,8192,1,16]> gxp = reshape(shape=sh1, x=gx4)[name=string(\"gxp\")];\n"];
    [mil appendString:@"        tensor<fp16, [1,8192,1,16]> yg = conv(dilations=dl, groups=groups, pad=pd, pad_type=pt, strides=st, weight=globalW, x=gxp)[name=string(\"yg\")];\n"];
    [mil appendString:@"        tensor<fp16, [1,512,1,256]> globalOut = reshape(shape=sho, x=yg)[name=string(\"global\")];\n"];
    [mil appendString:@"        tensor<fp16, [1,512,1,256]> globalScaled = mul(x=globalOut, y=alphaGlobal)[name=string(\"global_scaled\")];\n"];
    [mil appendString:@"        tensor<fp16, [1,512,1,256]> hGlobal = add(x=hLocal, y=globalScaled)[name=string(\"global_res\")];\n"];
    ane_bg_append_rmsnorm(mil, @"ffn", @"hGlobal", rmsFFN, @"w");
    [mil appendString:@"        tensor<fp16, [1,1024,1,256]> h1 = conv(dilations=dl, groups=one, pad=pd, pad_type=pt, strides=st, weight=Wv, x=w)[name=string(\"c1\")];\n"];
    [mil appendString:@"        tensor<fp16, [1,1024,1,256]> h3 = conv(dilations=dl, groups=one, pad=pd, pad_type=pt, strides=st, weight=Wg, x=w)[name=string(\"c3\")];\n"];
    [mil appendString:@"        tensor<fp16, [1,1024,1,256]> sig = sigmoid(x=h1)[name=string(\"sg\")];\n"];
    [mil appendString:@"        tensor<fp16, [1,1024,1,256]> silu = mul(x=h1, y=sig)[name=string(\"si\")];\n"];
    [mil appendString:@"        tensor<fp16, [1,1024,1,256]> gate = mul(x=silu, y=h3)[name=string(\"gt\")];\n"];
    [mil appendString:@"        tensor<fp16, [1,512,1,256]> mlp = conv(dilations=dl, groups=one, pad=pd, pad_type=pt, strides=st, weight=Wo, x=gate)[name=string(\"mlp\")];\n"];
    [mil appendString:@"        tensor<fp16, [1,512,1,256]> mlpScaled = mul(x=mlp, y=alphaMLP)[name=string(\"mlp_scaled\")];\n"];
    [mil appendString:@"        tensor<fp16, [1,512,1,256]> y = add(x=hGlobal, y=mlpScaled)[name=string(\"block_out\")];\n"];
    [mil appendString:@"    } -> (y);\n}\n"];
    return mil;
}

NSString *ane_bg_gen_head_mil(NSString *weight_path) {
    (void)weight_path;
    NSMutableString *mil = ane_bg_program_with_main_header(@"tensor<fp16, [1, 512, 1, 256]> x");
    [mil appendString:ane_bg_conv_prelude()];
    [mil appendString:@"        int32 c_groups = const()[name = string(\"c_groups\"), val = int32(1)];\n"];
    ane_bg_append_rmsnorm(mil, @"head", @"x", @"head_rms.bin", @"xL");
    [mil appendString:@"        tensor<fp16, [256, 512, 1, 1]> W = const()[name = string(\"W\"), val = tensor<fp16, [256, 512, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/head.bin\"), offset = uint64(64)))];\n"];
    [mil appendString:@"        tensor<fp16, [1, 1, 1, 1]> hs = const()[name = string(\"hs\"), val = tensor<fp16, [1, 1, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/head_logit_scale.bin\"), offset = uint64(64)))];\n"];
    [mil appendString:@"        tensor<fp16, [1, 256, 1, 256]> y0 = conv(dilations = dl, groups = c_groups, pad = pd, pad_type = pt, strides = st, weight = W, x = xL)[name = string(\"head\")];\n"];
    [mil appendString:@"        tensor<fp16, [1, 256, 1, 256]> y = mul(x = y0, y = hs)[name = string(\"head_scale\")];\n"];
    [mil appendString:@"    } -> (y);\n}\n"];
    return mil;
}

NSString *ane_bg_gen_full_model_mil(NSString *weight_dir) {
    (void)weight_dir;
    NSMutableString *mil = ane_bg_program_with_main_header(@"tensor<fp16, [1, 320, 1, 256]> x");
    [mil appendString:ane_bg_conv_prelude()];
    [mil appendString:@"        tensor<fp16, [512,320,1,1]> stem = const()[name=string(\"stemW\"), val=tensor<fp16, [512,320,1,1]>(BLOBFILE(path=string(\"@model_path/weights/stem.bin\"), offset=uint64(64)))];\n"];
    [mil appendString:@"        int32 one = const()[name=string(\"one\"), val=int32(1)];\n"];
    [mil appendString:@"        tensor<fp16, [1,512,1,256]> h_00 = conv(dilations=dl, groups=one, pad=pd, pad_type=pt, strides=st, weight=stem, x=x)[name=string(\"stem\")];\n"];
    for (NSInteger layer = 0; layer < 24; ++layer) {
        NSString *local = [NSString stringWithFormat:@"block_%02ld_local.bin", (long)layer];
        NSString *global = [NSString stringWithFormat:@"block_%02ld_global.bin", (long)layer];
        NSString *wv = [NSString stringWithFormat:@"block_%02ld_wv.bin", (long)layer];
        NSString *wg = [NSString stringWithFormat:@"block_%02ld_wg.bin", (long)layer];
        NSString *wo = [NSString stringWithFormat:@"block_%02ld_wo.bin", (long)layer];
        NSString *rmsLocal = [NSString stringWithFormat:@"block_%02ld_rms_local.bin", (long)layer];
        NSString *rmsGlobal = [NSString stringWithFormat:@"block_%02ld_rms_global.bin", (long)layer];
        NSString *rmsFFN = [NSString stringWithFormat:@"block_%02ld_rms_ffn.bin", (long)layer];
        NSString *alphaLocal = [NSString stringWithFormat:@"block_%02ld_alpha_local.bin", (long)layer];
        NSString *alphaGlobal = [NSString stringWithFormat:@"block_%02ld_alpha_global.bin", (long)layer];
        NSString *alphaMLP = [NSString stringWithFormat:@"block_%02ld_alpha_mlp.bin", (long)layer];
        NSString *inputVar = [NSString stringWithFormat:@"h_%02ld", (long)layer];
        NSString *localResVar = [NSString stringWithFormat:@"h_local_%02ld", (long)layer];
        NSString *globalResVar = [NSString stringWithFormat:@"h_global_%02ld", (long)layer];
        NSString *outputVar = [NSString stringWithFormat:@"h_%02ld", (long)(layer + 1)];
        NSString *localNormVar = [NSString stringWithFormat:@"u_%02ld", (long)layer];
        NSString *globalNormVar = [NSString stringWithFormat:@"v_%02ld", (long)layer];
        NSString *ffnNormVar = [NSString stringWithFormat:@"w_%02ld", (long)layer];
        NSString *localScaledVar = [NSString stringWithFormat:@"local_scaled_%02ld", (long)layer];
        NSString *globalScaledVar = [NSString stringWithFormat:@"global_scaled_%02ld", (long)layer];
        NSString *mlpScaledVar = [NSString stringWithFormat:@"mlp_scaled_%02ld", (long)layer];
        [mil appendFormat:@"        tensor<fp16, [8192,16,1,1]> localW_%02ld = const()[name=string(\"localW_%02ld\"), val=tensor<fp16, [8192,16,1,1]>(BLOBFILE(path=string(\"@model_path/weights/%@\"), offset=uint64(64)))];\n", (long)layer, (long)layer, local];
        [mil appendFormat:@"        tensor<fp16, [8192,16,1,1]> globalW_%02ld = const()[name=string(\"globalW_%02ld\"), val=tensor<fp16, [8192,16,1,1]>(BLOBFILE(path=string(\"@model_path/weights/%@\"), offset=uint64(64)))];\n", (long)layer, (long)layer, global];
        [mil appendFormat:@"        tensor<fp16, [1024,512,1,1]> wv_%02ld = const()[name=string(\"wv_%02ld\"), val=tensor<fp16, [1024,512,1,1]>(BLOBFILE(path=string(\"@model_path/weights/%@\"), offset=uint64(64)))];\n", (long)layer, (long)layer, wv];
        [mil appendFormat:@"        tensor<fp16, [1024,512,1,1]> wg_%02ld = const()[name=string(\"wg_%02ld\"), val=tensor<fp16, [1024,512,1,1]>(BLOBFILE(path=string(\"@model_path/weights/%@\"), offset=uint64(64)))];\n", (long)layer, (long)layer, wg];
        [mil appendFormat:@"        tensor<fp16, [512,1024,1,1]> wo_%02ld = const()[name=string(\"wo_%02ld\"), val=tensor<fp16, [512,1024,1,1]>(BLOBFILE(path=string(\"@model_path/weights/%@\"), offset=uint64(64)))];\n", (long)layer, (long)layer, wo];
        [mil appendFormat:@"        tensor<fp16, [1,1,1,1]> alphaLocal_%02ld = const()[name=string(\"alphaLocal_%02ld\"), val=tensor<fp16, [1,1,1,1]>(BLOBFILE(path=string(\"@model_path/weights/%@\"), offset=uint64(64)))];\n", (long)layer, (long)layer, alphaLocal];
        [mil appendFormat:@"        tensor<fp16, [1,1,1,1]> alphaGlobal_%02ld = const()[name=string(\"alphaGlobal_%02ld\"), val=tensor<fp16, [1,1,1,1]>(BLOBFILE(path=string(\"@model_path/weights/%@\"), offset=uint64(64)))];\n", (long)layer, (long)layer, alphaGlobal];
        [mil appendFormat:@"        tensor<fp16, [1,1,1,1]> alphaMLP_%02ld = const()[name=string(\"alphaMLP_%02ld\"), val=tensor<fp16, [1,1,1,1]>(BLOBFILE(path=string(\"@model_path/weights/%@\"), offset=uint64(64)))];\n", (long)layer, (long)layer, alphaMLP];
        [mil appendFormat:@"        tensor<int32, [4]> sh0_%02ld = const()[name=string(\"sh0_%02ld\"), val=tensor<int32, [4]>([1,512,16,16])];\n", (long)layer, (long)layer];
        [mil appendFormat:@"        tensor<int32, [4]> sh1_%02ld = const()[name=string(\"sh1_%02ld\"), val=tensor<int32, [4]>([1,8192,1,16])];\n", (long)layer, (long)layer];
        [mil appendFormat:@"        tensor<int32, [4]> sho_%02ld = const()[name=string(\"sho_%02ld\"), val=tensor<int32, [4]>([1,512,1,256])];\n", (long)layer, (long)layer];
        [mil appendFormat:@"        tensor<int32, [4]> pm_%02ld = const()[name=string(\"pm_%02ld\"), val=tensor<int32, [4]>([0,1,3,2])];\n", (long)layer, (long)layer];
        [mil appendFormat:@"        int32 groups_%02ld = const()[name=string(\"groups_%02ld\"), val=int32(512)];\n", (long)layer, (long)layer];
        ane_bg_append_rmsnorm(mil, [NSString stringWithFormat:@"local_%02ld", (long)layer], inputVar, rmsLocal, localNormVar);
        [mil appendFormat:@"        tensor<fp16, [1,512,16,16]> x4_%02ld = reshape(shape=sh0_%02ld, x=%@);\n", (long)layer, (long)layer, localNormVar];
        [mil appendFormat:@"        tensor<fp16, [1,512,16,16]> xt_%02ld = transpose(perm=pm_%02ld, x=x4_%02ld);\n", (long)layer, (long)layer, (long)layer];
        [mil appendFormat:@"        tensor<fp16, [1,8192,1,16]> xp_%02ld = reshape(shape=sh1_%02ld, x=xt_%02ld);\n", (long)layer, (long)layer, (long)layer];
        [mil appendFormat:@"        tensor<fp16, [1,8192,1,16]> yl_%02ld = conv(dilations=dl, groups=groups_%02ld, pad=pd, pad_type=pt, strides=st, weight=localW_%02ld, x=xp_%02ld);\n", (long)layer, (long)layer, (long)layer, (long)layer];
        [mil appendFormat:@"        tensor<fp16, [1,512,16,16]> y4_%02ld = reshape(shape=sh0_%02ld, x=yl_%02ld);\n", (long)layer, (long)layer, (long)layer];
        [mil appendFormat:@"        tensor<fp16, [1,512,16,16]> yi_%02ld = transpose(perm=pm_%02ld, x=y4_%02ld);\n", (long)layer, (long)layer, (long)layer];
        [mil appendFormat:@"        tensor<fp16, [1,512,1,256]> local_%02ld = reshape(shape=sho_%02ld, x=yi_%02ld);\n", (long)layer, (long)layer, (long)layer];
        [mil appendFormat:@"        tensor<fp16, [1,512,1,256]> %@ = mul(x=local_%02ld, y=alphaLocal_%02ld)[name=string(\"local_scaled_%02ld\")];\n", localScaledVar, (long)layer, (long)layer, (long)layer];
        [mil appendFormat:@"        tensor<fp16, [1,512,1,256]> %@ = add(x=%@, y=%@)[name=string(\"local_res_%02ld\")];\n", localResVar, inputVar, localScaledVar, (long)layer];
        ane_bg_append_rmsnorm(mil, [NSString stringWithFormat:@"global_%02ld", (long)layer], localResVar, rmsGlobal, globalNormVar);
        [mil appendFormat:@"        tensor<fp16, [1,512,16,16]> gx4_%02ld = reshape(shape=sh0_%02ld, x=%@);\n", (long)layer, (long)layer, globalNormVar];
        [mil appendFormat:@"        tensor<fp16, [1,8192,1,16]> gxp_%02ld = reshape(shape=sh1_%02ld, x=gx4_%02ld);\n", (long)layer, (long)layer, (long)layer];
        [mil appendFormat:@"        tensor<fp16, [1,8192,1,16]> yg_%02ld = conv(dilations=dl, groups=groups_%02ld, pad=pd, pad_type=pt, strides=st, weight=globalW_%02ld, x=gxp_%02ld);\n", (long)layer, (long)layer, (long)layer, (long)layer];
        [mil appendFormat:@"        tensor<fp16, [1,512,1,256]> global_%02ld = reshape(shape=sho_%02ld, x=yg_%02ld);\n", (long)layer, (long)layer, (long)layer];
        [mil appendFormat:@"        tensor<fp16, [1,512,1,256]> %@ = mul(x=global_%02ld, y=alphaGlobal_%02ld)[name=string(\"global_scaled_%02ld\")];\n", globalScaledVar, (long)layer, (long)layer, (long)layer];
        [mil appendFormat:@"        tensor<fp16, [1,512,1,256]> %@ = add(x=%@, y=%@)[name=string(\"global_res_%02ld\")];\n", globalResVar, localResVar, globalScaledVar, (long)layer];
        ane_bg_append_rmsnorm(mil, [NSString stringWithFormat:@"ffn_%02ld", (long)layer], globalResVar, rmsFFN, ffnNormVar);
        [mil appendFormat:@"        tensor<fp16, [1,1024,1,256]> v_%02ld = conv(dilations=dl, groups=one, pad=pd, pad_type=pt, strides=st, weight=wv_%02ld, x=%@);\n", (long)layer, (long)layer, ffnNormVar];
        [mil appendFormat:@"        tensor<fp16, [1,1024,1,256]> g_%02ld = conv(dilations=dl, groups=one, pad=pd, pad_type=pt, strides=st, weight=wg_%02ld, x=%@);\n", (long)layer, (long)layer, ffnNormVar];
        [mil appendFormat:@"        tensor<fp16, [1,1024,1,256]> s_%02ld = sigmoid(x=g_%02ld);\n", (long)layer, (long)layer];
        [mil appendFormat:@"        tensor<fp16, [1,1024,1,256]> m_%02ld = mul(x=v_%02ld, y=s_%02ld);\n", (long)layer, (long)layer, (long)layer];
        [mil appendFormat:@"        tensor<fp16, [1,512,1,256]> o_%02ld = conv(dilations=dl, groups=one, pad=pd, pad_type=pt, strides=st, weight=wo_%02ld, x=m_%02ld);\n", (long)layer, (long)layer, (long)layer];
        [mil appendFormat:@"        tensor<fp16, [1,512,1,256]> %@ = mul(x=o_%02ld, y=alphaMLP_%02ld)[name=string(\"mlp_scaled_%02ld\")];\n", mlpScaledVar, (long)layer, (long)layer, (long)layer];
        [mil appendFormat:@"        tensor<fp16, [1,512,1,256]> %@ = add(x=%@, y=%@)[name=string(\"mlp_res_%02ld\")];\n", outputVar, globalResVar, mlpScaledVar, (long)layer];
    }
    ane_bg_append_rmsnorm(mil, @"head", @"h_24", @"head_rms.bin", @"xL");
    [mil appendString:@"        tensor<fp16, [256,512,1,1]> head = const()[name=string(\"head\"), val=tensor<fp16, [256,512,1,1]>(BLOBFILE(path=string(\"@model_path/weights/head.bin\"), offset=uint64(64)))];\n"];
    [mil appendString:@"        tensor<fp16, [1,1,1,1]> headScale = const()[name=string(\"headScale\"), val=tensor<fp16, [1,1,1,1]>(BLOBFILE(path=string(\"@model_path/weights/head_logit_scale.bin\"), offset=uint64(64)))];\n"];
    [mil appendString:@"        tensor<fp16, [1,256,1,256]> logits0 = conv(dilations=dl, groups=one, pad=pd, pad_type=pt, strides=st, weight=head, x=xL)[name=string(\"logits\")];\n"];
    [mil appendString:@"        tensor<fp16, [1,256,1,256]> logits = mul(x=logits0, y=headScale)[name=string(\"logits_scaled\")];\n"];
    [mil appendString:@"    } -> (logits);\n}\n"];
    return mil;
}
