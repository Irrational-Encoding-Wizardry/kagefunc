import vapoursynth as vs

core = vs.core

# tbh, this is the kind of script where I should think about the implementation before writing code.
# As it is, this is just a mess

def framematch(clipa: vs.VideoNode, clipb: vs.VideoNode, threshold=0.05, hardsubbed=False, use_chroma=False,
               max_shift=100) -> vs.VideoNode:
    """
    Remove additional frames to match two input videos.
    Returns clipb that has been modified to match clipa.
    """
    # def prepare_hardsubs(clip):

    offset = 0

    if clipa.format != clipb.format:
        raise ValueError('Framematch: the input clip\'s formats don\' match')

    # Use a gauss kernel to blur away the artifacts
    clipa, clipb = [c.fmtc.resample(getw(360, clipa.width / clipa.height), 360, kernel='gauss') for c in [clipa, clipb]]
    if hardsubbed:
        # remove the lower 4th of the image, but make the crop value mod 2
        clipa, clipb = [c.std.Crop(bottom=(c.width // 8 * 2)) for c in [clipa, clipb]]
    clipa, clipb = [c.std.PlaneStats() for c in [clipa, clipb]]

    def pick(n, a, b):
        global offset
        if a.get_frame(n).props.PlaneStatsAverage - b.get_frame(n + offset).props.PlaneStatsAverage > 0.1:
            offset += 1
            return core.std.FrameEval()  # recursively call this? there must be a better way
        else:
            # might be correct, might not be
            diff = core.std.Expr([a[n], b[n + offset]], 'x y - abs').std.Binarize(5000)
            if diff.std.PlaneStats().get_frame(0).props.PlaneStatsAverage < threshold:
                return # well, return what?


def getw(h, ar=16 / 9, only_even=True):
    w = h * ar
    w = int(round(w))
    if only_even:
        w = w // 2 * 2
    return w
