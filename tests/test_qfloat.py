"""
Testing the QFloats in python
"""

import sys, os, time

import unittest
import numpy as np
from concrete import fhe

sys.path.append(os.getcwd())

from matrix_inversion.qfloat import QFloat, SignedBinary

BASE = 2
SIZE = 32


class TestQFloat(unittest.TestCase):
    ##################################################  NON FHE TESTS ##################################################

    def test_conversion_np(self):
        # test conversion from float to QFloat and vice-versa
        for i in range(100):
            base = BASE or np.random.randint(2, 10)
            size = SIZE or np.random.randint(20, 30)
            ints = np.random.randint(8, 12)
            f = (np.random.randint(0, 20000) - 10000) / 100  # float of type (+/-)xx.xx
            qf = QFloat.from_float(f, size, ints, base)
            if not ((qf.to_float() - f) < 0.1):
                raise ValueError(
                    "Wrong QFloat: "
                    + str(qf)
                    + " for float: "
                    + str(f)
                    + " and conversion: "
                    + str(qf.to_float())
                )

    def test_str(self):
        # positive value:
        qf = QFloat.from_float(13.75, 10, 5, 2)
        assert str(qf) == "01101.11000"

        # negative value
        qf = QFloat.from_float(-13.75, 10, 5, 2)
        assert str(qf) == "-01101.11000"

        # zero by value or sign
        qf = QFloat.from_float(0, 10, 5, 2)
        assert str(qf) == "00000.00000"

        qf = QFloat.from_float(1, 10, 5, 2)
        qf._sign = 0
        assert str(qf) == "00000.00000"

    def test_sign_np(self):
        # zero
        f = 0
        qf = QFloat.from_float(f, 10, 5, 2)
        if not (qf.get_sign() == 1):  # sign of 0 is 1
            print(qf.get_sign())
            raise ValueError(
                "Wrong sign for QFloat: " + str(qf) + " for float: " + str(f)
            )

        # non zero
        for i in range(100):
            base = BASE or np.random.randint(2, 10)
            size = SIZE or np.random.randint(20, 30)
            ints = np.random.randint(8, 12)
            f = (np.random.randint(0, 20000) - 10000) / 100  # float of type (+/-)xx.xx
            if f == 0:
                f += 1
            qf = QFloat.from_float(f, size, ints, base)
            if not (qf.get_sign() == np.sign(f)):
                raise ValueError(
                    "Wrong sign for QFloat: " + str(qf) + " for float: " + str(f)
                )

    def test_add_sub_np(self):
        # test add and sub
        for i in range(100):
            base = BASE or np.random.randint(2, 10)
            size = SIZE or np.random.randint(20, 30)
            ints = np.random.randint(8, 12)
            f1 = (np.random.randint(0, 20000) - 10000) / 100  # float of type (+/-)xx.xx
            f2 = (np.random.randint(0, 20000) - 10000) / 100  # float of type (+/-)xx.xx
            qf1 = QFloat.from_float(f1, size, ints, base)

            assert (SignedBinary(1) + qf1).to_float() - (1 + f1) < 0.1
            assert (SignedBinary(1) - qf1).to_float() - (1 - f1) < 0.1

            qf2 = QFloat.from_float(f2, size, ints, base)
            assert (qf1 + qf2).to_float() - (f1 + f2) < 0.1
            assert (qf1 - qf2).to_float() - (f1 - f2) < 0.1
            qf1 += qf2
            assert qf1.to_float() - (f1 + f2) < 0.1  # iadd

            #qf1._sign = 0  # sign is 0 must behave like a 0
            #assert (qf1 + qf2).to_float() - (f2) < 0.1

    def test_mul_np(self):
        # test multiplication by QFloat and integer
        for i in range(100):
            base = BASE or np.random.randint(2, 3)
            size = SIZE or np.random.randint(30, 40)
            ints = np.random.randint(10, 13)
            f1 = (np.random.randint(0, 200) - 100) / 10  # float of type (+/-)x.x
            f2 = (np.random.randint(0, 200) - 100) / 10  # float of type (+/-)x.x
            integer = np.random.randint(-2, 3)
            qf1 = QFloat.from_float(f1, size, ints, base)
            assert (SignedBinary(1) * qf1).to_float() - f1 < 0.1
            qf2 = QFloat.from_float(f2, size, ints, base)
            prod = qf1 * qf2

            assert prod.to_float() - (f1 * f2) < 0.1  # mul

            # from mul
            assert QFloat.from_mul(qf1, qf2).to_float() - (f1 * f2) < 0.1

            qf1 *= qf2
            assert qf1.to_float() - (f1 * f2) < 0.1  # imul

            # qf1._sign = 0  # sign is 0 must behave like a 0
            # assert (qf1 * qf2).to_float() == 0

            # from mul with specific values
            f1 = np.random.randint(1, 100) / 1.0  # float of type (+/-)xx.
            f2 = (np.random.randint(1, 10000)) / 10000000  # float of type (+/-)0.000xxx
            qf1 = QFloat.from_float(f1, 18, 18, 2)
            qf2 = QFloat.from_float(f2, 25, 0, 2)
            # from mul
            assert QFloat.from_mul(qf1, qf2, 18, 1).to_float() - (f1 * f2) < 0.1

    def test_div_np(self):
        # test division
        for i in range(100):
            base = BASE or np.random.randint(2, 3)
            size = SIZE or np.random.randint(30, 40)
            ints = np.random.randint(10, 13)
            f1 = (np.random.randint(0, 200) - 100) / 10  # float of type (+/-)x.x
            f2 = (np.random.randint(0, 200) - 100) / 10  # float of type (+/-)x.x
            if f2 == 0:
                f2 += 1.0
            if f1 == 0:
                f1 += 1.0
            qf1 = QFloat.from_float(f1, size, ints, base)

            assert (SignedBinary(1) / qf1).to_float() - 1.0 / f1 < 0.1

            assert (SignedBinary(-1) / qf1).to_float() - (-1.0 / f1) < 0.1
            assert np.abs((qf1 / SignedBinary(0)).to_float()) > 1000  # overflow

            newlen = np.random.randint(30, 40)
            newints = np.random.randint(10, 13)
            assert qf1.invert(1, newlen, newints).to_float() - 1.0 / f1 < 0.1
            assert (SignedBinary(-1) / qf1).to_float() - (-1.0 / f1) < 0.1

            qf2 = QFloat.from_float(f2, size, ints, base)
            div = qf1 / qf2
            if not (div.to_float() - (f1 / f2) < 0.1):
                raise Exception(
                    "Wrong division for f1:"
                    + str(f1)
                    + " f2: "
                    + str(f2)
                    + " f1/f2: "
                    + str(f1 / f2)
                    + " and div : "
                    + str(div.to_float())
                )

    def test_abs_np(self):
        for i in range(100):
            base = BASE or np.random.randint(2, 3)
            size = SIZE or np.random.randint(30, 40)
            ints = np.random.randint(10, 13)
            f1 = (np.random.randint(0, 200) - 100) / 10  # float of type (+/-)x.x
            qf1 = QFloat.from_float(f1, size, ints, base)
            assert abs(qf1).to_float() - (abs(f1)) < 0.1

    def test_ge(self):
        for i in range(100):
            base = BASE or np.random.randint(2, 3)
            size = SIZE or np.random.randint(30, 40)
            ints = np.random.randint(10, 13)
            f1 = (np.random.randint(0, 20) - 10) / 10  # float of type (+/-)x.x
            f2 = (np.random.randint(0, 20) - 10) / 10  # float of type (+/-)x.x
            qf1 = QFloat.from_float(f1, size, ints, base)
            qf2 = QFloat.from_float(f2, size, ints, base)
            assert 1 * (qf1 >= qf2) == 1 * (f1 >= f2)


unittest.main()

# suite = unittest.TestLoader().loadTestsFromName('test_qfloat.TestQFloat.test_div_np')
# unittest.TextTestRunner(verbosity=1).run(suite)
