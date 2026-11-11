import hashlib
from pathlib import Path
import tempfile
import unittest
import time
from unittest.mock import patch
from preparation_cache import Deferred, HashCache


class PreparationCacheTests(unittest.TestCase):
    def test_cached_success_cannot_hide_same_size_mutation(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'asset'; p.write_bytes(b'old')
            expected = hashlib.sha256(b'old').hexdigest()
            cache = HashCache(); cache.check(p, expected)
            p.write_bytes(b'new')
            with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
                cache.check(p, expected)

    def test_expected_digest_change_is_checked_on_cache_hit(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'asset'; p.write_bytes(b'x')
            cache = HashCache(); cache.check(p)
            with self.assertRaises(ValueError): cache.check(p, '0'*64)

    def test_resume_with_no_reread_and_budget_deferral(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d)/'a', Path(d)/'b'
            a.write_bytes(b'1234'); b.write_bytes(b'5678')
            cache = HashCache(byte_budget=4); cache.check(a)
            with self.assertRaises(Deferred): cache.check(b)
            resumed = HashCache(cache.entries, byte_budget=4)
            with patch('preparation_cache.time.time_ns', return_value=time.time_ns()+3_000_000_000):
                resumed.check(a); resumed.check(b)
            self.assertEqual(resumed.bytes_read, 4)
            self.assertEqual(resumed.hits, 1)

    def test_oversized_first_file_is_not_starved(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'a'; p.write_bytes(b'12345678')
            cache = HashCache(byte_budget=4)
            cache.check(p); self.assertEqual(cache.bytes_read, 8)

    def test_disappeared_asset_is_not_reused(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'a'; p.write_bytes(b'x'); cache = HashCache(); cache.check(p)
            p.unlink()
            with self.assertRaises(FileNotFoundError): cache.check(p)

    def test_symlink_retarget_invalidates_asset_lookup(self):
        with tempfile.TemporaryDirectory() as d:
            a, b, link = [Path(d)/name for name in ['a','b','link']]
            a.write_bytes(b'a'); b.write_bytes(b'b'); link.symlink_to(a)
            cache = HashCache(); expected = cache.check(link)
            link.unlink(); link.symlink_to(b)
            with self.assertRaises(ValueError): cache.check(link, expected)


if __name__ == '__main__': unittest.main()
