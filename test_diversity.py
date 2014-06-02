from numpy import zeros, ones_like, uint64
from numpy.random import random
from numpy.testing import assert_allclose

from diversity import q_score

def test_qscore():
    all_zeros = zeros([10000], uint64)
    all_ones = ones_like(all_zeros, uint64)

    assert q_score(all_ones, all_zeros)     == -1
    assert q_score(all_zeros, all_ones)     == -1
    assert q_score(all_ones, all_ones)      == 1
    assert q_score(all_zeros, all_zeros)    == 1

    for i in range(1000):
        random_rater_1 = (random(all_zeros.shape) > 0.5).astype(uint64)
        random_rater_2 = (random(all_zeros.shape) > 0.5).astype(uint64)
        assert_allclose(q_score(random_rater_1, random_rater_2), 0, atol = 0.1)
        assert q_score(random_rater_1, random_rater_1) == 1
        assert q_score(random_rater_1, (random_rater_1 + 1) % 2) == -1


if __name__ == '__main__':
    test_qscore()
