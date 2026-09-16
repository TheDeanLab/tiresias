import unittest


class PublicApiTests(unittest.TestCase):
    def test_public_api_exports_core_functions(self):
        import tiresias

        for name in [
            "estimate_blind_psf_scipy",
            "estimate_blind_psf_cupy",
            "estimate_psf_array_cupy",
            "deconvolve_with_cupy",
            "generate_theoretical_psf",
            "generate_psf_seed",
            "load_psf_seed",
            "estimate_psf_from_chunks",
        ]:
            self.assertTrue(hasattr(tiresias, name), name)
        self.assertFalse(hasattr(tiresias, "deconvolve_with_cucim"))
        self.assertFalse(hasattr(tiresias, "measure_beam_width_profile"))


if __name__ == "__main__":
    unittest.main()
