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
                  b: float = 1 / 3, c: float = 1 / 3, mask_detail: bool = False, descale_mask_zones: str = '',
                  denoise: bool = False, bm3d_sigma: float = 1, knl_strength: float = 0.4, use_gpu: bool = True) \
        -> vs.VideoNode:
    """
    Use descale to reverse the scaling on a given input clip.
    width, height, kernel, taps, a1, a2 are parameters for resizing.
    descale_mask_zones can be used to only mask certain zones to improve performance; uses rfs syntax.
    denoise, bm3d_sigma, knl_strength, use_gpu are parameters for denoising; denoise = False to disable
    use_gpu = True -> chroma will be denoised with KNLMeansCL (faster)
    """
    if not height:
        raise ValueError('inverse_scale: you need to specify a value for the output height')

    only_luma = source.format.num_planes == 1

    if get_depth(source) != 32:
        source = source.resize.Point(format=source.format.replace(bits_per_sample=32, sample_type=vs.FLOAT))
    width = fallback(width, getw(height, source.width / source.height))

    # if we denoise luma and chroma separately, do the chroma here while it’s still 540p
    if denoise and use_gpu and not only_luma:
        source = core.knlm.KNLMeansCL(source, a=2, h=knl_strength, d=3, device_type='gpu', device_id=0, channels='UV')

    planes = split(source)
    planes[0] = _descale_luma(planes[0], width, height, kernel, taps, b, c)
    if only_luma:
        return planes[0]
    planes = _descale_chroma(planes, width, height)

    if mask_detail:
        upscaled = fvf.Resize(planes[0], source.width, source.height, kernel=kernel, taps=taps, a1=b, a2=c)
        planes[0] = mask_descale(get_y(source), planes[0], upscaled, zones=descale_mask_zones)
    scaled = join(planes)
    return mvf.BM3D(scaled, radius1=1, sigma=[bm3d_sigma, 0] if use_gpu else bm3d_sigma) if denoise else scaled


def _descale_luma(luma, width, height, kernel, taps, b, c):
    return get_descale_filter(kernel, b=b, c=c, taps=taps)(luma, width, height)


def _descale_chroma(planes, width, height):
    planes[1], planes[2] = [core.resize.Bicubic(plane, width, height, src_left=0.25) for plane in planes[1:]]
    return planes


def mask_descale(original: vs.VideoNode, descaled: vs.VideoNode, upscaled: vs.VideoNode,
                 threshold: float = 0.05, zones: str = '', debug: bool = False):
    downscaled = core.resize.Spline36(original, descaled.width, descaled.height)
    assert get_depth(original) == get_depth(descaled), "Source and descaled clip need to have the same bitdepth"
    detail_mask = _generate_descale_mask(original, descaled, upscaled)
    if debug:
        return detail_mask
    merged = core.std.MaskedMerge(descaled, downscaled, detail_mask)
    return fvf.ReplaceFrames(descaled, merged, zones) if zones else merged


def _generate_descale_mask(source, downscaled, upscaled, threshold=0.05):
    mask = core.std.Expr([source, upscaled], 'x y - abs') \
        .resize.Bicubic(downscaled.width, downscaled.height).std.Binarize(threshold)
    mask = iterate(mask, core.std.Maximum, 2)
    return iterate(mask, core.std.Inflate, 2)


# Currently, this should fail for non mod4 subsampled input.
# Not really relevant though, as 480p, 576p, 720p, and 1080p are all mod32
def generate_keyframes(clip: vs.VideoNode, out_path: str = None, header: bool = True) -> None:
    """
    probably only useful for fansubbing
    generates qp-filename for keyframes to simplify timing
    """
    import os
    # Speed up the analysis by resizing first. Converting to 8 bit also seems to improve the accuracy of wwxd.
    clip = core.resize.Bilinear(clip, 640, 360, format=vs.YUV420P8)
    clip = core.wwxd.WWXD(clip)
    out_txt = "# WWXD log file, using qpfile format\n\n" if header else ""
    for i in range(clip.num_frames):
        if clip.get_frame(i).props.Scenechange == 1:
            out_txt += f"{i} I -1\n" if i != 0 else ""
        if i % 1 == 0:
            print(f"Progress: {i}/{clip.num_frames} frames", end="\r")
    out_path = fallback(out_path, os.path.expanduser("~") + "/Desktop/keyframes.txt")
    with open(out_path, "w") as text_file:
        text_file.write(out_txt)


def adaptive_grain(clip: vs.VideoNode, strength=0.25, static=True, luma_scaling=12, show_mask=False) -> vs.VideoNode:
    """
    Generates grain based on frame and pixel brightness. Details can be found here:
    https://kageru.moe/blog/article/adaptivegrain
    Strength is the strength of the grain generated by AddGrain, static=True for static grain, luma_scaling
    manipulates the grain alpha curve. Higher values will generate less grain (especially in brighter scenes),
    while lower values will generate more grain, even in brighter scenes.
    """
    mask = core.adg.Mask(clip.std.PlaneStats(), luma_scaling)
    grained = core.grain.Add(clip, var=strength, constant=static)
    if get_depth(clip) != 8:
        mask = depth(mask, get_depth(clip))
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
    white = 1 if mask_format.sample_type == vs.FLOAT else (1 << bits) - 1

    center = core.std.BlankClip(clip, width=width, height=height, _format=mask_format, color=white, length=1)

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

    return center * clip.num_frames


def retinex_edgemask(src: vs.VideoNode, sigma=1) -> vs.VideoNode:
    """
    Use retinex to greatly improve the accuracy of the edge detection in dark scenes.
    sigma is the sigma of tcanny
    """
    luma = get_y(src)
    max_value = 1 if src.format.sample_type == vs.FLOAT else (1 << get_depth(src)) - 1
    ret = core.retinex.MSRCP(luma, sigma=[50, 200, 350], upper_thr=0.005)
    tcanny = ret.tcanny.TCanny(mode=1, sigma=sigma).std.Minimum(coordinates=[1, 0, 1, 0, 0, 1, 0, 1])
    return core.std.Expr([kirsch(luma), tcanny], f'x y + {max_value} min')


def kirsch(src: vs.VideoNode) -> vs.VideoNode:
    """
    Kirsch edge detection. This uses 8 directions, so it's slower but better than Sobel (4 directions).
    more information: https://ddl.kageru.moe/konOJ.pdf
    """
    weights = [5] * 3 + [-3] * 5
    weights = [weights[-i:] + weights[:-i] for i in range(4)]
    clip = [core.std.Convolution(src, (w[:4] + [0] + w[4:]), saturate=False) for w in weights]
    return core.std.Expr(clip, 'x y max z max a max')


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
    ref = ref if clip.format == ref.format else depth(ref, bits)

    clips = [clip.std.Convolution([1] * 9), ref.std.Convolution([1] * 9)]
    diff = core.std.Expr(clips, difexpr, vs.GRAY8).std.Maximum().std.Maximum()

    mask = core.misc.Hysteresis(subedge, diff)
    mask = iterate(mask, core.std.Maximum, expand_n)
    mask = mask.std.Inflate().std.Inflate().std.Convolution([1] * 9)
    return depth(mask, bits, range=1, range_in=1)


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
    return iterate(mask, core.std.Inflate, 4)


def crossfade(clipa, clipb, duration):
    """
    Crossfade clipa into clipb. Duration is the length of the blending zone.
    For example, crossfade(a, b, 100) will fade the last 100 frames of a into b.
    """

    def fade_image(n, clipa, clipb):
        return core.std.Merge(clipa, clipb, weight=n / clipa.num_frames)

    if clipa.format.id != clipb.format.id or clipa.height != clipb.height or clipa.width != clipb.width:
        raise ValueError('Crossfade: Both clips must have the same dimensions and format.')
    fade = core.std.FrameEval(core.std.BlankClip(clipa, length=duration+1), partial(fade_image, clipa=clipa[-duration-1:], clipb=clipb[:duration]))
    return clipa[:-duration] + fade[1:] + clipb[duration:]


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
    return width // 2 * 2 if only_even else width
