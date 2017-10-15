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

    # Initialize arrays for the PlaneStatsAverage of each frame.
    values_a = [-1.]*len(clipa)
    values_b = [-1.]*len(clipb)

    offset = 0

    if clipa.format != clipb.format:
        raise ValueError('Framematch: the input clip\'s formats don\' match')

    # Use a gauss kernel to blur away the artifacts
    clipa, clipb = [c.fmtc.resample(getw(360, clipa.width / clipa.height), 360, kernel='gauss') for c in [clipa, clipb]]
    if hardsubbed:
        # remove the lower 4th of the image, but make the crop value mod 2
        # this is also phenominally stupid because we have to return that clip later :FailFish:
        clipa, clipb = [c.std.Crop(bottom=(c.width // 8 * 2)) for c in [clipa, clipb]]
    clipa, clipb = [c.std.PlaneStats() for c in [clipa, clipb]]

    def pick(n, a, b):
        global offset
        # Some kind of lazy evaluation of the list
        if values_a[n] == -1:
            values_a[n] = a.get_frame(n).props.PlaneStatsAverage
        if values_b[n] == -1:
            values_b[n] = b.get_frame(n+offset).props.PlaneStatsAverage
         
        while abs(values_a[n] - values_b[n+offset]) > 0.1:
            offset += 1
        diff = core.std.Expr([a[n], b[n + offset]], 'x y - abs')#.std.Binarize(5000)
        if diff.std.PlaneStats().get_frame(0).props.PlaneStatsAverage < threshold:
            return clipb[n+offset]


def getw(h, ar=16 / 9, only_even=True):
    w = h * ar
    w = int(round(w))
    if only_even:
        w = w // 2 * 2
    return w
