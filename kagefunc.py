"""
kageru’s collection of vapoursynth functions.
"""
from functools import partial
from vsutil import *
import vapoursynth as vs
import mvsfunc as mvf
import fvsfunc as fvf

core = vs.core


def inverse_scale(source: vs.VideoNode, width: int = None, height: int = 0, kernel: str = 'bilinear', taps: int = 4,
                  a1: float = 1 / 3, a2: float = 1 / 3, mask_detail: bool = False, masking_areas: list = None,
                  mask_threshold: float = 0.05, show_mask: bool = False, denoise: bool = False, bm3d_sigma: float = 1,
                  knl_strength: float = 0.4, use_gpu: bool = True) -> vs.VideoNode:
    """
    source = input clip
    width, height, kernel, taps, a1, a2 are parameters for resizing.
    mask_detail, masking_areas, mask_threshold are parameters for masking; mask_detail = False to disable.
    masking_areas takes frame tuples to define areas which will be masked (e.g. opening and ending)
    masking_areas = [[1000, 2500], [30000, 32000]]. Start and end frame are inclusive.
    mask_threshold is the binarization threshold. Value must be normalized for floats (0-1) or an 8-bit integer.
    denoise, bm3d_sigma, knl_strength, use_gpu are parameters for denoising; denoise = False to disable
    use_gpu = True -> chroma will be denoised with KNLMeansCL (faster)
    """
    if get_depth(source) != 32:
        source = source.resize.Point(format=source.format.replace(bits_per_sample=32, sample_type=vs.FLOAT))
    luma = get_y(source)
    width = fallback(width, getw(height, source.width / source.height))
    if mask_threshold > 1:
        mask_threshold /= 256
    if denoise and use_gpu:
        source = core.knlm.KNLMeansCL(source, a=2, h=knl_strength, d=3, device_type='gpu', device_id=0, channels='UV')
    planes = split(source)
    planes = _descale_planes(planes, width, height, kernel, taps, a1, a2)

    if mask_detail:
        mask = _generate_detail_mask(luma, planes[0], kernel, taps, a1, a2, mask_threshold)
        if show_mask:
            return mask
        if masking_areas is None:
            planes[0] = _apply_mask(luma, planes[0], mask)
        else:
            planes[0] = _apply_mask_to_area(luma, planes[0], mask, masking_areas)
    scaled = join(planes)
    if denoise:
        scaled = mvf.BM3D(scaled, radius1=1, sigma=[bm3d_sigma, 0] if use_gpu else bm3d_sigma)
    return scaled


# the following 4 functions are mostly called from inside inverse_scale
def _descale_planes(planes, width, height, kernel, taps, b, c):
    planes[0] = get_descale_filter(kernel, b=b, c=c, taps=taps)(planes[0], width, height)
    planes[1], planes[2] = [core.resize.Bicubic(plane, width, height, src_left=0.25) for plane in planes[1:]]
    return planes


def _generate_detail_mask(source, downscaled, kernel='bicubic', taps=4, b=1 / 3, c=1 / 3, threshold=0.05):
    upscaled = fvf.Resize(downscaled, source.width, source.height, kernel=kernel, taps=taps, a1=b, a2=c)
    mask = core.std.Expr([source, upscaled], 'x y - abs') \
        .resize.Bicubic(downscaled.width, downscaled.height).std.Binarize(threshold)
    mask = iterate(mask, core.std.Maximum, 2)
    return iterate(mask, core.std.Inflate, 2)


def _apply_mask(source, scaled, mask):
    noalias = core.fmtc.resample(source, scaled.width, scaled.height, kernel='blackmanminlobe', taps=5)
    return core.std.MaskedMerge(scaled, noalias, mask)


def _apply_mask_to_area(source, scaled, mask, area):
    if len(area) == 2 and isinstance(area[0], int):
        area = [[area[0], area[1]]]
    noalias = core.fmtc.resample(source, scaled.width, scaled.height, kernel='blackmanminlobe', taps=5)
    for a in area:  # TODO: use ReplaceFrames
        source_cut = core.std.Trim(noalias, a[0], a[1])
        scaled_cut = core.std.Trim(scaled, a[0], a[1])
        mask_cut = core.std.Trim(mask, a[0], a[1])
        masked = _apply_mask(source_cut, scaled_cut, mask_cut)
        scaled = insert_clip(scaled, masked, a[0])
    return scaled


# Currently, this should fail for non mod4 subsampled input.
# Not really relevant, though, as 480p, 576p, 720p, and 1080p are all mod32
def generate_keyframes(clip: vs.VideoNode, out_path=None) -> None:
    """
    probably only useful for fansubbing
    generates qp-filename for keyframes to simplify timing
    """
    import os
    clip = core.resize.Bilinear(clip, 640, 360)  # speed up the analysis by resizing first
    clip = core.wwxd.WWXD(clip)
    out_txt = "# WWXD log file, using qpfile format\n\n"
    for i in range(clip.num_frames):
        if clip.get_frame(i).props.Scenechange == 1:
            out_txt += "%d I -1\n" % i
        if i % 1000 == 0:
            print(i)
    out_path = fallback(out_path, os.path.expanduser("~") + "/Desktop/keyframes.txt")
    text_file = open(out_path, "w")
    text_file.write(out_txt)
    text_file.close()


def adaptive_grain(clip: vs.VideoNode, strength=0.25, static=True, luma_scaling=12, show_mask=False) -> vs.VideoNode:
    """
    Generates grain based on frame and pixel brightness. Details can be found here:
    https://kageru.moe/blog/article/adaptivegrain
    Strength is the strength of the grain generated by AddGrain, static=True for static grain, luma_scaling
    manipulates the grain alpha curve. Higher values will generate less grain (especially in brighter scenes),
    while lower values will generate more grain, even in brighter scenes.
    """
    import numpy as np

    def fill_lut(y):
        """
        Using horner's method to compute this polynomial:
        (1 - (1.124 * x - 9.466 * x² + 36.624 * x³ - 45.47 * x⁴ + 18.188 * x⁵)) ** ((y²) * luma_scaling) * 255
        Using the normal polynomial is about 2.5x slower during the initial generation.
        I know it doesn't matter as it only saves a few ms (or seconds at most), but god damn, just let me have
        some fun here, will ya? Just truncating (rather than rounding) the array would also half the processing
        time, but that would decrease the precision and is also just unnecessary.
        """
        x = np.arange(0, 1, 1 / (1 << 8))
        z = (1 - (x * (1.124 + x * (-9.466 + x * (36.624 + x * (-45.47 + x * 18.188)))))) ** ((y ** 2) * luma_scaling)
        if clip.format.sample_type == vs.INTEGER:
            z = z * 255
            z = np.rint(z).astype(int)
        return z.tolist()

    def generate_mask(n, f, clip):
        frameluma = round(f.props.PlaneStatsAverage * 999)
        table = lut[int(frameluma)]
        return core.std.Lut(clip, lut=table)

    lut = [None] * 1000
    for y in np.arange(0, 1, 0.001):
        lut[int(round(y * 1000))] = fill_lut(y)

    luma = get_y(fvf.Depth(clip, 8)).std.PlaneStats()
    grained = core.grain.Add(clip, var=strength, constant=static)

    mask = core.std.FrameEval(luma, partial(generate_mask, clip=luma), prop_src=luma)
    mask = core.resize.Spline36(mask, clip.width, clip.height)

    if get_depth(clip) != 8:
        mask = fvf.Depth(mask, bits=get_depth(clip))
    if show_mask:
        return mask

    return core.std.MaskedMerge(clip, grained, mask)


# TODO: implement blending zone in which both clips are merged to avoid abrupt and visible kernel changes.
def conditional_resize(src: vs.VideoNode, kernel='bilinear', width=1280, height=720, thr=0.00015,
                       debug=False) -> vs.VideoNode:
    """
    Fix oversharpened upscales by comparing a regular downscale with a blurry bicubic kernel downscale.
    Similar to the avisynth function. thr is lower in vapoursynth because it's normalized (between 0 and 1)
    """

    def compare(n, down, oversharpened, diff_default, diff_os):
        error_default = diff_default.get_frame(n).props.PlaneStatsDiff
        error_os = diff_os.get_frame(n).props.PlaneStatsDiff
        if debug:
            debugstring = "error when scaling with {:s}: {:.5f}\nerror when scaling with bicubic (b=0, c=1): " \
                          "{:.5f}\nUsing sharp debicubic: {:s}".format(kernel, error_default, error_os,
                                                                       str(error_default - thr > error_os))
            oversharpened = oversharpened.sub.Subtitle(debugstring)
            down = down.sub.Subtitle(debugstring)
        if error_default - thr > error_os:
            return oversharpened
        return down

    if hasattr(core, 'descale'):
        down = get_descale_filter(kernel)(width, height)
        oversharpened = core.descale.Debicubic(width, height, b=0, c=1)
    else:
        down = src.fmtc.resample(width, height, kernel=kernel, invks=True)
        oversharpened = src.fmtc.resample(width, height, kernel='bicubic', a1=0, a2=1, invks=True)

    # we only need luma for the comparison
    rescaled = get_y(down).fmtc.resample(src.width, src.height, kernel=kernel)
    oversharpened_up = get_y(oversharpened).fmtc.resample(src.width, src.height, kernel='bicubic', a1=0, a2=1)

    src_luma = get_y(src)
    diff_default = core.std.PlaneStats(rescaled, src_luma)
    diff_os = core.std.PlaneStats(oversharpened_up, src_luma)

    return core.std.FrameEval(
        down, partial(compare, down=down, oversharpened=oversharpened, diff_os=diff_os, diff_default=diff_default))


def squaremask(clip: vs.VideoNode, width: int, height: int, offset_x: int, offset_y: int) -> vs.VideoNode:
    """
    “There must be a better way!”
    Basically a small script that draws white rectangles on a black background.
    Python-only replacement for manual paint/photoshop/gimp masks, as long as these don't go beyond a simple rectangle.
    Can be merged with an edgemask to only mask certain edge areas.
    TL;DR: Unless you're scenefiltering, this is useless.
    """
    bits = get_depth(clip)
    src_w = clip.width
    src_h = clip.height
    mask_format = clip.format.replace(color_family=vs.GRAY, subsampling_w=0, subsampling_h=0)

    if mask_format.sample_type == vs.FLOAT:
        white = 1
    else:
        white = (1 << bits) - 1

    center = core.std.BlankClip(width=width, height=height, _format=mask_format, color=white,
                                length=clip.num_frames, fpsnum=clip.fps.numerator, fpsden=clip.fps.denominator)

    if offset_x:
        left = core.std.BlankClip(center, width=offset_x, height=height, color=0)
        center = core.std.StackHorizontal([left, center])

    if center.width < src_w:
        right = core.std.BlankClip(center, width=src_w - center.width, height=height, color=0)
        center = core.std.StackHorizontal([center, right])

    if offset_y:
        top = core.std.BlankClip(center, width=src_w, height=offset_y, color=0)
        center = core.std.StackVertical([top, center])

    if center.height < src_h:
        bottom = core.std.BlankClip(center, width=src_w, height=src_h - center.height, color=0)
        center = core.std.StackVertical([center, bottom])

    return center


def retinex_edgemask(src: vs.VideoNode, sigma=1) -> vs.VideoNode:
    """
    Use retinex to greatly improve the accuracy of the edge detection in dark scenes.
    sigma is the sigma of tcanny
    """
    luma = get_y(src)
    ret = core.retinex.MSRCP(luma, sigma=[50, 200, 350], upper_thr=0.005)
    mask = core.std.Expr([kirsch(luma), ret.tcanny.TCanny(mode=1, sigma=sigma).std.Minimum(
        coordinates=[1, 0, 1, 0, 0, 1, 0, 1])], 'x y +')
    return mask


def kirsch(src: vs.VideoNode) -> vs.VideoNode:
    """
    Kirsch edge detection. This uses 8 directions, so it's slower but better than Sobel (4 directions).
    more information: https://ddl.kageru.moe/konOJ.pdf
    """
    weights = [5] * 3 + [-3] * 5
    weights = [weights[-i:] + weights[:-i] for i in range(4)]
    clip = [core.std.Convolution(src, (w[:4] + [0] + w[4:]), saturate=False) for w in weights]
    return core.std.Expr(clip, 'x y max z max a max')


def fast_sobel(src: vs.VideoNode) -> vs.VideoNode:
    """
    Should behave similar to std.Sobel() but faster since it has no additional high-/lowpass, gain, or the sqrt.
    The internal filter is also a little brighter.
    """
    sobel_x = src.std.Convolution([-1, -2, -1, 0, 0, 0, 1, 2, 1], saturate=False)
    sobel_y = src.std.Convolution([-1, 0, 1, -2, 0, 2, -1, 0, 1], saturate=False)
    return core.std.Expr([sobel_x, sobel_y], 'x y max')


def get_descale_filter(kernel: str, **kwargs):
    """
    Stolen from a declined pull request.
    Originally written by @stuxcrystal on Github.
    """
    filters = {
        'bilinear': (lambda **kwargs: core.descale.Debilinear),
        'spline16': (lambda **kwargs: core.descale.Despline16),
        'spline36': (lambda **kwargs: core.descale.Despline36),
        'bicubic': (lambda b, c, **kwargs: partial(core.descale.Debicubic, b=b, c=c)),
        'lanczos': (lambda taps, **kwargs: partial(core.descale.Delanczos, taps=taps)),
    }
    return filters[kernel](**kwargs)


def hardsubmask(clip: vs.VideoNode, ref: vs.VideoNode, expand_n=None) -> vs.VideoNode:
    """
    Uses multiple techniques to mask the hardsubs in video streams like Anime on Demand or Wakanim.
    Might (should) work for other hardsubs, too, as long as the subs are somewhat close to black/white.
    It's kinda experimental, but I wanted to try something like this.
    It works by finding the edge of the subtitle (where the black border and the white fill color touch),
    and it grows these areas into a regular brightness + difference mask via hysteresis.
    This should (in theory) reliably find all hardsubs in the image with barely any false positives (or none at all).
    Output depth and processing precision are the same as the input
    It is not necessary for 'clip' and 'ref' to have the same bit depth, as 'ref' will be dithered to match 'clip'
    Most of this code was written by Zastin (https://github.com/Z4ST1N)
    Clean code soon(tm)
    """
    clp_f = clip.format
    bits = clp_f.bits_per_sample
    stype = clp_f.sample_type

    expand_n = fallback(expand_n, clip.width // 200)

    yuv_fmt = core.register_format(clp_f.color_family, vs.INTEGER, 8, clp_f.subsampling_w, clp_f.subsampling_h)

    y_range = 219 << (bits - 8) if stype == vs.INTEGER else 1
    uv_range = 224 << (bits - 8) if stype == vs.INTEGER else 1
    offset = 16 << (bits - 8) if stype == vs.INTEGER else 0

    uv_abs = ' abs ' if stype == vs.FLOAT else ' {} - abs '.format((1 << bits) // 2)
    yexpr = 'x y - abs {thr} > 255 0 ?'.format(thr=y_range * 0.7)
    uvexpr = 'x {uv_abs} {thr} < y {uv_abs} {thr} < and 255 0 ?'.format(uv_abs=uv_abs, thr=uv_range * 0.1)

    difexpr = 'x {upper} > x {lower} < or x y - abs {mindiff} > and 255 0 ?'.format(upper=y_range * 0.8 + offset,
                                                                                    lower=y_range * 0.2 + offset,
                                                                                    mindiff=y_range * 0.1)

    # right shift by 4 pixels.
    # fmtc uses at least 16 bit internally, so it's slower for 8 bit,
    # but its behaviour when shifting/replicating edge pixels makes it faster otherwise
    if bits < 16:
        right = core.resize.Point(clip, src_left=4)
    else:
        right = core.fmtc.resample(clip, sx=4, flt=False)
    subedge = core.std.Expr([clip, right], [yexpr, uvexpr], yuv_fmt.id)
    c444 = split(subedge.resize.Bicubic(format=vs.YUV444P8, filter_param_a=0, filter_param_b=0.5))
    subedge = core.std.Expr(c444, 'x y z min min')

    clip, ref = get_y(clip), get_y(ref)
    ref = ref if clip.format == ref.format else fvf.Depth(ref, bits)

    clips = [clip.std.Convolution([1] * 9), ref.std.Convolution([1] * 9)]
    diff = core.std.Expr(clips, difexpr, vs.GRAY8).std.Maximum().std.Maximum()

    mask = core.misc.Hysteresis(subedge, diff)
    mask = iterate(mask, core.std.Maximum, expand_n)
    mask = mask.std.Inflate().std.Inflate().std.Convolution([1] * 9)
    mask = fvf.Depth(mask, bits, range=1, range_in=1)
    return mask


def hardsubmask_fades(clip, ref, expand_n=8, highpass=5000):
    """
    Uses Sobel edge detection to find edges that are only present in the main clip.
    These should (theoretically) be the subtitles.
    The video is blurred beforehand to prevent compression artifacts from being recognized as subtitles.
    This may create more false positives than the other hardsubmask,
    but it is capable of finding subtitles of any color and subtitles during fadein/fadeout.
    Setting highpass to a lower value may catch very slight changes (e.g. the last frame of a low-contrast fade),
    but it will make the mask more susceptible to artifacts.
    """
    clip = core.fmtc.bitdepth(clip, bits=16).std.Convolution([1] * 9)
    ref = core.fmtc.bitdepth(ref, bits=16).std.Convolution([1] * 9)
    clipedge = get_y(clip).std.Sobel()
    refedge = get_y(ref).std.Sobel()
    mask = core.std.Expr([clipedge, refedge], 'x y - {} < 0 65535 ?'.format(highpass)).std.Median()
    mask = iterate(mask, core.std.Maximum, expand_n)
    mask = iterate(mask, core.std.Inflate, 4)
    return mask


def crossfade(clipa, clipb, duration):
    """
    Crossfade clipa into clipb. Duration is the length of the blending zone.
    For example, crossfade(a, b, 100) will fade the last 100 frames of a into b.
    """

    def fade_image(frame_number, clipa, clipb):
        return core.std.Merge(clipa, clipb, weight=frame_number / clipa.num_frames)

    if clipa.format.id != clipb.format.id or clipa.height != clipb.height or clipa.width != clipb.width:
        raise ValueError('Crossfade: Both clips must have the same dimensions and format.')
    fade = core.std.FrameEval(clipa[-duration:], partial(fade_image, clipa=clipa[-duration:], clipb=clipb[:duration]))
    return clipa[:-duration] + fade + clipb[duration:]


def hybriddenoise(src, knl=0.5, sigma=2, radius1=1):
    """
    denoise luma with BM3D (CPU-based) and chroma with KNLMeansCL (GPU-based)
    sigma = luma denoise strength
    knl = chroma denoise strength. The algorithm is different, so this value is different from sigma
    BM3D's sigma default is 5, KNL's is 1.2, to give you an idea of the order of magnitude
    radius1 = temporal radius of luma denoising, 0 for purely spatial denoising
    """
    y = get_y(src)
    y = mvf.BM3D(y, radius1=radius1, sigma=sigma)
    denoised = core.knlm.KNLMeansCL(src, a=2, h=knl, d=3, device_type='gpu', device_id=0, channels='UV')
    return core.std.ShufflePlanes([y, denoised], planes=[0, 1, 2], colorfamily=vs.YUV)


# helpers

def getw(height, aspect_ratio=16 / 9, only_even=True):
    """
    Returns width for image.
    """
    width = height * aspect_ratio
    width = int(round(width))
    if only_even:
        width = width // 2 * 2
    return width


def getY(c: vs.VideoNode) -> vs.VideoNode:
    """
    Deprecated alias, use get_y instead
    """
    raise DeprecationWarning('getY is deprecated. Please use vsutils.get_y instead')
