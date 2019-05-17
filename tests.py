"""
Many of these don’t actually test the logic and just make some 
basic assertions as well as a call to check if frames are produced.
"""
import unittest
import vapoursynth as vs
import kagefunc as kgf


class KagefuncTests(unittest.TestCase):
    BLACK_SAMPLE_CLIP = vs.core.std.BlankClip(_format=vs.YUV420P8, width=160, height=120, color=[0, 128, 128],
                                              length=100)
    WHITE_SAMPLE_CLIP = vs.core.std.BlankClip(_format=vs.YUV420P8, width=160, height=120, color=[255, 128, 128],
                                              length=100)
    GREYSCALE_SAMPLE_CLIP = vs.core.std.BlankClip(_format=vs.GRAY8, width=160, height=120, color=[255])

    def test_retinex_edgemask(self):
        mask = kgf.retinex_edgemask(self.BLACK_SAMPLE_CLIP)
        self.assert_same_bitdepth(mask, self.BLACK_SAMPLE_CLIP)
        self.assert_same_length(mask, self.BLACK_SAMPLE_CLIP)
        self.assertEqual(mask.format.color_family, vs.GRAY)
        # request a frame to see if that errors
        mask.get_frame(0)

    def test_inverse_scale(self):
        src = self.BLACK_SAMPLE_CLIP
        resized = kgf.inverse_scale(self.GREYSCALE_SAMPLE_CLIP, height=90)
        self.assertEqual(resized.format.id, vs.GRAYS)
        self.assertEqual(resized.height, 90)
        self.assertEqual(resized.width, 120)
        resized = kgf.inverse_scale(src, height=90)
        self.assertEqual(resized.format.id, vs.YUV444PS)
        self.assertEqual(resized.height, 90)
        self.assertEqual(resized.width, 120)
        self.assert_runs(kgf.inverse_scale(src, height=90, denoise=True))
        self.assert_runs(kgf.inverse_scale(src, height=90, mask_detail=True))
        self.assert_runs(kgf.inverse_scale(src, height=90, descale_mask_zones='[0, 4]', mask_detail=True))
        resized.get_frame(0)

    def test_squaremask(self):
        mask = kgf.squaremask(self.BLACK_SAMPLE_CLIP, 30, 30, 20, 0)
        self.assert_same_length(mask, self.BLACK_SAMPLE_CLIP)
        self.assert_same_bitdepth(mask, self.BLACK_SAMPLE_CLIP)
        self.assert_same_dimensions(mask, self.BLACK_SAMPLE_CLIP)
        mask.get_frame(0)

    def test_adaptive_grain(self):
        grained = kgf.adaptive_grain(self.BLACK_SAMPLE_CLIP)
        self.assert_same_metadata(grained, self.BLACK_SAMPLE_CLIP)
        grained.get_frame(0)

    @staticmethod
    def assert_runs(clip: vs.VideoNode):
        clip.get_frame(0)

    def assert_same_dimensions(self, clip_a: vs.VideoNode, clip_b: vs.VideoNode):
        """
        Assert that two clips have the same width and height.
        """
        self.assertEqual(clip_a.height, clip_b.height, 'Same height expected, was {clip_a.height} and {clip_b.height}')
        self.assertEqual(clip_a.width, clip_b.width, 'Same width expected, was {clip_a.width} and {clip_b.width}')

    def assert_same_format(self, clip_a: vs.VideoNode, clip_b: vs.VideoNode):
        """
        Assert that two clips have the same format (but not necessarily size).
        """
        self.assertEqual(clip_a.format.id, clip_b.format.id, 'Same format expected')

    def assert_same_bitdepth(self, clip_a: vs.VideoNode, clip_b: vs.VideoNode):
        """
        Assert that two clips have the same number of bits per sample.
        """
        self.assertEqual(clip_a.format.bits_per_sample, clip_b.format.bits_per_sample,
                         'Same depth expected, was {clip_a.format.bits_per_sample} and {clip_b.format.bits_per_sample}')

    def assert_same_length(self, clip_a: vs.VideoNode, clip_b: vs.VideoNode):
        self.assertEqual(len(clip_a), len(clip_b),
                         'Same number of frames expected, was {len(clip_a)} and {len(clip_b)}.')

    def assert_same_metadata(self, clip_a: vs.VideoNode, clip_b: vs.VideoNode):
        """
        Assert that two clips have the same height and width, format, depth, and length.
        """
        self.assert_same_format(clip_a, clip_b)
        self.assert_same_dimensions(clip_a, clip_b)
        self.assert_same_length(clip_a, clip_b)

    def assert_same_frame(self, clip_a: vs.VideoNode, clip_b: vs.VideoNode):
        """
        Assert that two frames are identical. Only the first frame of the arguments is used.
        """
        diff = vs.core.std.PlaneStats(clip_a, clip_b)
        frame = diff.get_frame(0)
        self.assertEqual(frame.props.PlaneStatsDiff, 0)


if __name__ == '__main__':
    unittest.main()
