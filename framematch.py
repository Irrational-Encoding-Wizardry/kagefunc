import vapoursynth as vs
core = vs.core

def framematch(clipa, clipb, hardsubbed=False, use_chroma=False, max_shift=100):
    """
    Remove additional frames to match two input videos.
    Returns clipb that has been modified to match clipa.
    """
    # def prepare_hardsubs(clip):
    if clipa.format != clipb.format:
        raise ValueError('Framematch: the input clip\'s formats don\' match')
    if clipa.width != clipb.width or clipa.height != clipb.height:
        # Since we return b later, it makes sense to change a here
        clipa = clipa.resize.Bilinear(clipb.width, clipb.height)
    if hardsubbed:
        clipa, clipb = [c.std.Crop(bottom=(c.width//8*2)) for c in [clipa, clipb]]
    
