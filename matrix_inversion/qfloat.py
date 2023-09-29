"""
This module implements the QFloat class that allows to quantize floating point values in FHE
author : Lcressot
"""

import numpy as np
import numbers
from concrete import fhe
import base_p_arrays as bpa

Tracer = fhe.tracing.tracer.Tracer


class Zero:
    """
    A simple class to differentiate values that we know are zero from others
    It is usefull to save computations for operations with QFloats
    """

    def copy(self):
        """
        Copy the object
        This is is the same as returning itself
        """
        return self

    def to_float(self):
        """
        Convert to an unencrypted float
        """
        return float(0)

    def __add__(self, other):
        """
        Add a Zero to an object
        This does not affect the object
        """
        if isinstance(other, Zero):
            return self

        return other

    def __radd__(self, other):
        """
        Add a Zero to an object (from the right)
        This does not affect the object
        """
        if isinstance(other, Zero):
            return self

        return other

    def __sub__(self, other):
        """
        Subtract an object from a Zero
        This returns the negative of the object
        """
        if isinstance(other, Zero):
            return self

        return -other

    def __rsub__(self, other):
        """
        Subtract a Zero from an object
        This does not affect the object
        """
        return other

    def __mul__(self, other):
        """
        Multiply with a Zero
        This gives a Zero
        """
        return self

    def __rmul__(self, other):
        """
        Multiply with a Zero (from the right)
        This gives a Zero
        """
        return self

    def __truediv__(self, other):
        """
        Division of a Zero by another object
        This gives a Zero, or an error if the other object is a Zero as well
        """
        if isinstance(other, Zero):
            raise ValueError("division by Zero")

        return self

    def __rtruediv__(self, other):
        """
        Division of an object by a Zero
        This raises an error
        """
        raise ValueError("division by Zero")

    def __neg__(self):
        """
        Return the negative of a Zero which is itself
        """
        return self

    def neg(self):
        """
        Change the sign in place which makes no change
        """
        return self

    def __abs__(self):
        """
        Return a copy of the object with positive sign, which does not change it
        """
        return self


class SignedBinary:
    """
    A simple class to differentiate values that we know are binary (+1, -1 or 0) from others
    It is usefull to save computations for operations with QFloats
    The value can be encrypted or not, as long as it is binary
    """

    def __init__(self, value):
        """
        Initialize with a signed binary value
        """
        # init with a signed binary value (+1, -1 or 0)
        self._value = value
        self._encrypted = isinstance(value, Tracer)

    @property
    def value(self):
        """
        Getter of value
        """
        return self._value

    @value.setter
    def value(self, newvalue):
        """
        Setter of value
        """
        self._value = newvalue
        self._encrypted = isinstance(newvalue, Tracer)

    @property
    def encrypted(self):
        """
        Getter of encrypted
        """
        return self._encrypted

    @encrypted.setter
    def encrypted(self, newencrypted):
        """
        Setter of encrypted
        """
        raise ValueError("Setting this variables from outside is forbidden")

    def copy(self):
        """
        Copy the object
        """
        return SignedBinary(self._value)

    def to_float(self):
        """
        Convert object to an unencrypted float
        Cannot be called in FHE with an encrypted value
        """
        return float(self._value)

    def __add__(self, other):
        """
        Add an object (Zero, SignedBinary or QFloat)
        """
        if isinstance(other, SignedBinary):
            # potentially no more a binary
            return self._value + other._value
        if isinstance(other, QFloat):
            return other.__add__(self)

        return self._value + other

    def __sub__(self, other):
        """
        Subtract other object
        """
        res = self.copy() + (-other)
        return res

    def __mul__(self, other):
        """
        Multiply with an object (Zero, SignedBinary or QFloat)
        """
        if isinstance(other, SignedBinary):
            # stays binary
            return SignedBinary(self._value * other._value)
        if isinstance(other, QFloat):
            return other.__mul__(self)

        return self._value * other

    def __truediv__(self, other):
        """
        Divide by an object (Zero, SignedBinary or QFloat)
        """
        if isinstance(other, SignedBinary):
            # stays binary
            return SignedBinary(self._value // other._value)
        if isinstance(other, QFloat):
            return other.__rtruediv__(self)

        return self._value / other

    def __neg__(self):
        """
        Return an new object with opposite sign
        """
        return SignedBinary(-1 * self._value)

    def neg(self):
        """
        Change the sign in place
        """
        self._value *= -1
        return self

    def __abs__(self):
        """
        Return a copy of the object with positive sign
        """
        return SignedBinary(np.abs(self._value))


class QFloat:
    """
    A class for quantizing floats
    Floats are encoded as encrypted or unencrypted arrays of integers in a specified base
    Encrypted QFloat can be summed, multiplied etc. with unencrypted QFloats, and vice-versa

    TODO:
    - allow operations (see from_mul) between QFloats of different lengths to optimize algorithms
    - replace length + ints with length + dot position where dot position can be
    negative to encode efficiently for very low numbers.
    For isntance: 0.000000001110 encoded as -.--------1110 with dot position = -8
    - keep track of overflow (in tidy, set QFloat.overflow = dividend),
    then decrypt this value to know if the computation failed
    - do not compute tidying on the floating part when adding an integer or SignedBinary in iadd

    """

    # keepTidy: wether to keep arrays tidy at all time. Setting to False can save time in FHE.
    KEEP_TIDY=False

    # Statistics for counting operations between QFloats within a circuit
    ADDITIONS = 0
    MULTIPLICATION = 0
    DIVISION = 0

    def __init__(self, array, ints=None, base=2, is_tidy=True, sign=None):
        """
        - array: an encrypted or unencrypted array representing a number in base p (little endian)
        - ints: gives the number of digits before the dot,
            so ints = 1 will encode a number like x.xxxx...
        - is_tidy: wether the array is already tidy
        - sign: provide the sign if it is already known (usefull in FHE to save computations)
        """
        if not (isinstance(array, np.ndarray) or isinstance(array, Tracer)):
            raise ValueError("array must be np.ndarray or Tracer")

        if isinstance(array, np.ndarray):
            array = array.astype("int")

        self._encrypted = isinstance(array, Tracer)

        if len(array.shape) > 1:
            raise ValueError("array must be one dimension")
        self._array = array[:]

        if not self._encrypted:
            self._array = self._array.astype(int)

        if not (isinstance(base, int) and base > 1):
            raise ValueError("base must be a int >1")
        self._base = base
        if ints is None:
            ints = array.size // 2
        else:
            if not (isinstance(ints, int) and 0 <= ints and array.size >= ints):
                raise ValueError("ints must be in range [0,array.size]")
        self._ints = int(ints)
        self._sign = sign  # a sign of 0 makes the QFloat null
        if isinstance(self._sign, float):
            self._sign = int(self._sign)

        self._is_base_tidy = is_tidy
        self._is_tidy = is_tidy

        if not self._is_tidy and QFloat.KEEP_TIDY:
            self.tidy()

    @classmethod
    def reset_stats(cls):
        """
        Reset the class statitics
        """
        cls.ADDITIONS = 0
        cls.MULTIPLICATION = 0
        cls.DIVISION = 0

    @classmethod
    def show_stats(cls):
        """
        Display the class statitics
        """
        print("\nQFloat statistics :")
        print("======================")
        print("Additions       : " + str(cls.ADDITIONS))
        print("Multiplications : " + str(cls.MULTIPLICATION))
        print("Divisions       : " + str(cls.DIVISION))
        print("\n")

    def _check_unencrypted(self):
        if self._encrypted:
            raise ValueError("This function does not work on encrypted QFloats")

    # ======================================================================================
    #                        Functions for unencrypted QFloats only
    # ======================================================================================

    def toStr(self, tidy=True):
        """
        Convert the QFloat to a string

        WARNING : does not work on encrypted QFLoat
        """
        self._check_unencrypted()

        if tidy: # tidy before return the representation
            self.tidy()
            sgn = self.get_sign()

            integerPart = (sgn*self._array[0:self._ints]).astype('int')
            floatPart = (sgn*self._array[self._ints:]).astype('int')

            if self._base <= 10:
                integerPart = ''.join([str(i) for i in integerPart])
                floatPart = ''.join([str(i) for i in floatPart])
            else:
                integerPart = str(integerPart)
                floatPart = str(floatPart)

            sgnstr = '' if sgn >=0 else '-'

            return sgnstr+integerPart+'.'+floatPart

        else: # return the untidy representation

            integerPart = self._array[0:self._ints].astype('int')
            floatPart = self._array[self._ints:].astype('int')

            integerPart = ''.join([str(i) for i in integerPart])
            floatPart = ''.join([str(i) for i in floatPart])

            return integerPart+'.'+floatPart

    def __str__(self):
        """
        Convert the QFloat to a tidy string

        WARNING : does not work on encrypted QFLoat
        """
        return self.toStr(True)   

    @classmethod
    def from_float(cls, f, length=10, ints=None, base=2):
        """
        Create a QFloat from an unencrypted float

        WARNING : will return an unencrypted QFLoat
        """
        if ints is None:
            ints = length // 2

        integer_part = int(f)
        float_part = f - integer_part

        int_array = bpa.int_to_base_p(integer_part, ints, base)
        float_array = bpa.float_to_base_p(float_part, length - ints, base)

        array = fhe.zeros(length)
        array[0:ints] = int_array
        array[ints:] = float_array
        sign = np.sign(f) or 1  # zero has sign 1

        # set with abs value of array
        return cls(array, ints, base, True, sign)

    def to_float(self):
        """
        Create an unencrypted float from the QFloat

        WARNING : does not work on encrypted QFLoat
        """
        self._check_unencrypted()

        integerPart = bpa.base_p_to_int(self._array[0:self._ints], self._base)
        floatPart = bpa.base_p_to_float(self._array[self._ints:], self._base)

        return integerPart + floatPart        

    # ======================================================================================
    #                      Functions for both encrypted or unencrypted QFloats
    #
    #       Functions with (self,other) can work for mixed encrypted and unencrypted arrays
    # ======================================================================================

    @property
    def ints(self):
        """
        Getter of ints
        """
        return self._ints

    @ints.setter
    def ints(self, newints):
        """
        Setter of ints
        """
        raise ValueError("Setting this variables from outside is forbidden")

    @property
    def base(self):
        """
        Getter of base
        """
        return self._base

    @base.setter
    def base(self, newbase):
        """
        Setter of base
        """
        raise ValueError("Setting this variables from outside is forbidden")

    @property
    def is_base_tidy(self):
        """
        Getter of is_base_tidy
        """
        return self._is_base_tidy

    @is_base_tidy.setter
    def is_base_tidy(self, newis_base_tidy):
        """
        Setter of is_base_tidy
        """
        raise ValueError("Setting this variables from outside is forbidden")

    @property
    def array(self):
        """
        Getter of array
        """
        return self._array

    @array.setter
    def array(self, newarray):
        """
        Setter of array
        """
        raise ValueError("Setting this variables from outside is forbidden")

    @property
    def encrypted(self):
        """
        Getter of encrypted
        """
        return self._encrypted

    @encrypted.setter
    def encrypted(self, newencrypted):
        """
        Setter of encrypted
        """
        raise ValueError("Setting this variables from outside is forbidden")

    @property
    def sign(self):
        """
        Getter of sign
        """
        return self._sign

    @sign.setter
    def sign(self, newsign):
        """
        Setter of sign
        """
        raise ValueError("Setting this variables from outside is forbidden")

    @classmethod
    def zero(cls, length, ints, base, encrypted=True):
        """
        Create a QFloat of 0
        """
        if not (isinstance(length, int) and length > 0):
            raise ValueError("length must be a positive int")
        if encrypted:
            return cls(fhe.zeros(length), ints, base, True, fhe.ones(1)[0])

        return cls(np.zeros(length), ints, base, True, 1)

    @classmethod
    def zero_like(cls, other):
        """
        Create a QFloat of 0 with same shape as other
        """
        if not isinstance(other, cls):
            raise ValueError("Object must be a QFloat")

        return cls.zero(len(other), other.ints, other.base, other.encrypted)

    @classmethod
    def one(cls, length, ints, base, encrypted=True):
        """
        Create a QFloat of 1
        """
        if encrypted:
            array = fhe.zeros(length)
            array[ints - 1] = fhe.ones(1)[0]
            return cls(array, ints, base, True, fhe.ones(1)[0])
        else:
            array = np.zeros(length)
            array[ints - 1] = 1
            return cls(array, ints, base, True, 1)

    @classmethod
    def one_like(cls, other):
        """
        Create a QFloat of 1 with same shape as other
        """
        if not isinstance(other, cls):
            raise ValueError("Object must be a QFloat")

        return cls.one(len(other), other.ints, other.base, other.encrypted)

    def copy(self):
        """
        Create a QFloat copy
        """
        copy = QFloat(self.to_array(), self._ints, self._base, self._is_tidy, self._sign)
        copy._is_base_tidy = self._is_base_tidy
        return copy

    def to_array(self):
        """
        Rerturn copy of array
        """
        if not self._encrypted:
            return np.copy(self._array)
        else:
            return self._array[:]

    def set_len_ints(self, newlen, newints):
        """
        Set new length and new ints
        WARNING : This operation may troncate the int or float part
        """
        zeros = fhe.zeros if self._encrypted else lambda x: np.zeros(x, dtype="int")
        # set ints
        if self._ints != newints:
            if newints > self._ints:
                self._array = np.concatenate(
                    (zeros(int(newints - self._ints)), self._array), axis=0
                )
            else:
                self._array = self._array[self._ints - newints :]
            self._ints = int(newints)

        # set length
        difflen = int(newlen - len(self))
        if difflen != 0:
            if difflen > 0:
                self._array = np.concatenate((self._array, zeros(difflen)), axis=0)
            else:
                self._array = self._array[:-difflen]

        return self

    def check_compatibility(self, other):
        """
        Check wether other has equal encoding
        """
        if not isinstance(other, QFloat):
            raise ValueError("Object must also be a " + str(QFloat))

        if self._base != other.base:
            raise ValueError(str(QFloat) + "s bases are different")

        if len(self) != len(other):
            raise ValueError(str(QFloat) + "s have different length")

        if self._ints != other.ints:
            raise ValueError(str(QFloat) + "s have different dot index")

    def get_sign(self):
        """
        Return the sign of the QFloat
        WARNING: 0 is considered positive for faster computations
        but it can happen than a sign is set to zero in other function (see __mul__)
        """
        if self._sign is None:
            # base tidy so that sign can be computed from the first non zero number:
            self.base_tidy()

            # most efficient way to compute the sign of the first non zero number:
            borrow = 0
            for i in range(len(self)):
                borrow = self._array[-i-1] - borrow < 0

            self._sign = 2*(1-borrow)-1

        return self._sign

    def base_tidy(self):
        """
        Tidy array so that values are in range [-(base-1), base-1]
        as they should be, but signs can be mixed
        Keeping arrays untidy when possible saves computation time
        """
        if self._is_base_tidy:
            return

        dividend = 0
        for i in reversed(range(len(self))):
            curr = self._array[i] + dividend
            dividend = (np.abs(curr) // self._base) * np.sign(curr)
            curr -= dividend * self._base
            self._array[i] = curr

        # TODO: keep track of overflow
        # QFloat.OVERFLOW = dividend

        self._is_base_tidy = True

    @classmethod
    def multi_base_tidy(cls, arrays, base):
        """
        Tidy arrays so that values are in range [-(base-1), base-1]
        as they should be, but signs can be mixed
        Keeping arrays untidy when possible saves computation time
        """

        dividends = fhe.zeros(arrays.shape[0])
        for i in reversed(range(arrays.shape[1])):
            curr = arrays[:, i] + dividends
            dividends = (np.abs(curr) // base) * np.sign(curr)
            curr -= dividends * base
            arrays[:, i] = curr[:]

        # TODO: keep track of overflow
        # QFloat.OVERFLOW = np.any(dividends)

        return arrays

    def tidy(self):
        """
        Tidy array so that values are in range [-base, base] and signs are all the same
        This gives the standard reprensentation of the number
        Keeping arrays untidy when possible saves computation time
        """
        if self._is_tidy:
            return

        # first of all, make all values (negative or positive) fall under the base:
        if not self._is_base_tidy:
            self.base_tidy()

        # then, make all value the same sign. To do this, we consider that: 
        # - a Qfloat F is the sum of its positive and negative parts: F = P + N
        # - as N is negative, we also have F = P - |N|, where both P and |N| are positive,
        #   so we can use the regular subtraction algorithm
        # - if P < |N|, we can use F = -1 * (|N| - P) so the subtraction algorithm works

        P =  self._array*(self._array >= 0)
        absN = -1*(self._array*(self._array < 0))

        P_minus_absN, is_negative = bpa.base_p_subtraction(P, absN, self._base, True)
        is_positive_or_zero = (1-is_negative)
        self._array = is_positive_or_zero*P_minus_absN - is_negative*bpa.base_p_subtraction(absN,P,self._base)

        self._sign = 2*is_positive_or_zero-1
        self._is_tidy=True

    def __len__(self):
        """
        Return length of array
        """
        return self._array.size

    def __eq__(self, other):
        """
        Return wether self is identical to other
        """
        self.check_compatibility(other)

        if not (self._is_tidy and other._is_tidy):
            raise Exception('cannot compare QFloats that are not tidy')        

        return (len(self) - np.sum(self.array == other.array))==0

    def __lt__(self, other):
        """ Computes wether an array is lower than another

        An array A is lower than a array B if and only if:
        there exist an index i such that: A[i] < B[i]  AND  for all k<i, A[k] <= B[k]

        This is the mathematical converse of A >= B (see self.__ge__) which is easier to do in fhe

        If array have different length, the shorter one is considered to have extra -np.inf values
        """
        return 1-(self >= other)

    def __le__(self, other):
        """ Computes wether an array is lower or equal than another

        The proposition "A <= B" is equivalent to "B >= A" (see self.__ge__)

        If array have different length, the shorter one is considered to have extra extra -np.inf values
        """       
        return other >= self

    def __gt__(self, other):
        """ Computes wether an array is greater than another

        The proposition "A > B" is the mathematical converse of "B >= A" (see self.__ge__)

        If array have different length, the shorter one is considered to have extra extra -np.inf values
        """
        return 1-(other >= self)

    def __ge__(self, other):
        """ Computes wether an array is greater or equal than another, in alphabetical order

        An array A is greater or equal than an array B if and only if:
        for all index i:  either  A[i] >= B[i]  or  there is an index k<i where A[k] > B[k]
        """     
        self.check_compatibility(other)

        self.tidy()
        other.tidy()

        return bpa.is_greater_or_equal(self.array, other.array)

    def __abs__(self):
        """
        Returns the absolute value
        """
        absval = self.copy()
        absval.tidy() # need to tidy before computing abs
        absval._array = np.abs(absval._array)
        absval._sign = 1
        return absval

    def abs(self):
        """
        In place absolute value
        """
        self.tidy() # need to tidy before computing abs
        self._array = np.abs(self._array)
        self._sign = 1
        return self

    def __add__(self, other):
        """
        Sum with another QFLoat or single integer
        """
        addition = self.copy()
        addition += other
        return addition

    def __radd__(self, other):
        """
        Add other object on the right
        """
        return self.__add__(other)

    @classmethod
    def check_convert_fhe(cls, qfloat, condition):
        """
        If qfloat is not encrypted and condition is verified,
        convert qfloat array to fhe array
        """
        if (not qfloat.encrypted) and condition:
            fhe_array = fhe.ones(len(qfloat)) * qfloat.array
            qfloat.array = fhe_array
            qfloat.encrypted = True

    def self_check_convert_fhe(self, condition):
        """
        If self is not encrypted and condition is verified,
        convert self to fhe array
        """
        QFloat.check_convert_fhe(self, condition)

    def __iadd__(self, other):
        """
        Sum with another QFLoat or single integer, in place
        """

        if isinstance(other, Zero):
            return  # no change if adding a Zero

        QFloat.ADDITIONS += (
            1  # count addition in all other cases, cause we have to tidy
        )

        if isinstance(other, SignedBinary):
            self.self_check_convert_fhe(other.encrypted)
            self._array[self._ints - 1] += other.value
            # TODO : in this case, we don't need to base_tidy the floating part of the QFloat
        else:
            self.self_check_convert_fhe(other.encrypted)

            self.check_compatibility(other)
            self._array += other._array

        self._is_tidy=False
        self._is_base_tidy=False
        self._sign=None

        # base tidy to keep bitwidth low
        self.base_tidy()

        if QFloat.KEEP_TIDY:
            self.tidy()

        return self

    def __sub__(self, other):
        """
        Subtract other object
        """
        res = self.copy()
        res += -other
        return res

    def __rsub__(self, other):
        """
        Subtract other object on the right
        """
        res = other.copy()
        res += -self
        return res

    def __imul__(self, other):
        """
        Multiply with another QFLoat or integer, in place

        WARNING: precision of multiplication does not increase, so it may overflow if not enough
        """
        if isinstance(other, SignedBinary):
            self.self_check_convert_fhe(other.encrypted)
            # multiply everything by a binary value, which keeps the array tidy
            self._array *= other.value
            #self._is_tidy and self._is_base_tidy are not impacted here
            if self._sign is not None:
                self._sign = self._sign*np.sign(other.value)  
        else:
            QFloat.MULTIPLICATION += 1  # count only multiplications with other Qfloat
            # multiply with another compatible QFloat

            # always base tidy before a multiplication between two
            # QFloats prevent multiplying big bitwidths
            self.base_tidy()
            other.base_tidy()

            self.self_check_convert_fhe(other.encrypted)
            self.check_compatibility(other)

            # A QFloat array is made of 2 parts, integer part and float part
            # The multiplication array will be the sum of integer * other + float * other
            n = len(self)
            mularray = fhe.zeros((n, n))

            # integer part, shift  to the left
            for i in range(0, self._ints):
                mularray[i, 0 : n - (self._ints - 1 - i)] = (
                    self._array[i] * other._array[self._ints - 1 - i :]
                )
            # float part, shift to the right
            for i in range(self._ints, n):
                mularray[i, 1 + i - self._ints :] = (
                    self._array[i] * other._array[0 : n - (i - self._ints) - 1]
                )

            # the multiplication array is made from the sum of the muarray rows
            self._array = np.sum(mularray, axis=0)

            if self._sign is not None and other.sign is not None: # avoid computing sign of the sign if we already know it
                self._sign = self._sign*other.sign
            else:
                self._sign=None

            self._is_tidy=False
            self._is_base_tidy=False

            # base tidy to keep bitwidth low
            self.base_tidy()

        if QFloat.KEEP_TIDY:
            self.tidy()

        return self

    def __mul__(self, other):
        """
        Multiply with another QFLoat or number, see __imul__
        """
        # special case when multiplying by unencrypted 0, the result is an unencrypted 0
        if isinstance(other, Zero):
            return Zero()

        multiplication = self.copy()
        if isinstance(other, Tracer) or (
            isinstance(other, SignedBinary) and other.encrypted
        ):
            multiplication.self_check_convert_fhe(
                True
            )  # convert to encrypted array if needed

        multiplication *= other

        return multiplication

    def __rmul__(self, other):
        """
        Multiply with other object on the right
        """
        return self.__mul__(other)

    def __neg__(self):
        """
        Negative
        """
        neg = self.copy()
        neg._array *= -1
        if neg.sign is not None:
            neg._sign *= -1        
        return neg

    def neg(self):
        """
        In place negative
        Use it for faster subtraction if you don't need
        to keep the second QFloat: a - b = a + b.neg()
        """
        self._array *= -1
        if self._sign is not None:
            self._sign *= -1
        return self

    @classmethod
    def from_mul(cls, a, b, newlength=None, newints=None):
        """
        Compute the multiplication of QFloats a and b,
        inside a new QFloat of given length and ints

        Warning: if the new length and ints are too low,
        the result might be cropped
        """
        if newlength is None:
            newlength = len(a)
        if newints is None:
            newints = a.ints

        # special case when multiplying by unencrypted 0,
        # the result is an unencrypted 0
        if isinstance(a, Zero) or isinstance(b, Zero):
            return Zero()

        if isinstance(a, SignedBinary) or isinstance(b, SignedBinary):
            if isinstance(a, SignedBinary) and isinstance(b, SignedBinary):
                return a * b
            # simple multiplication and size setting
            multiplication = a * b
            multiplication.set_len_ints(newlength, newints)

        else:
            cls.MULTIPLICATION += 1  # count only multiplications with other Qfloat

            # QFloats should always be base tidy before a multiplication
            assert a.is_base_tidy
            assert b.is_base_tidy

            # convert a to encrypted if needed
            cls.check_convert_fhe(a, b.encrypted)

            # check base compatibility
            if not a.base == b.base:
                raise ValueError("bases are different")

            # A QFloat array is made of 2 parts, integer part and float part
            # The multiplication array will be the sum of a.integer * b + a.float * b
            mularray = fhe.zeros((len(a), newlength))
            for i in range(0, len(a)):
                # index from b where the array mul should be inserted in mularray
                indb = newints - a.ints + i + 1 - b.ints
                # compute only needed multiplication of b._array*a._array[i],
                # accounting for crops
                ind1 = 0 if indb >= 0 else -indb
                ind2 = min(len(b), newlength - indb)
                if ind2 > ind1:
                    mul = b.array[ind1:ind2] * a.array[i]

                    if ind2 - ind1 == 1:
                        mul = mul.reshape(1)
                    bpa.insert_array_at_index(mul, mularray, i, indb + ind1)

            # the multiplication array is made from the sum
            # of the mularray rows with product of signs
            multiplication = QFloat(
                np.sum(mularray, axis=0), newints, a.base, False, None
            )

            if a.get_sign() is not None and b.get_sign() is not None: # avoid computing sign of the sign if we already know it
                multiplication._sign = a.get_sign()*b.get_sign()
            else:
                multiplication._sign=None

            multiplication._is_tidy=False
            multiplication._is_base_tidy=False

            # base tidy to keep bitwidth low
            multiplication.base_tidy()

        if QFloat.KEEP_TIDY:
            multiplication.tidy()

        return multiplication

    @classmethod
    def multi_from_mul(cls, list_a, list_b, newlength=None, newints=None):
        """
        Compute the multiplication of QFloats elements of list_a and list_b,
        inside new QFloats of given length and ints

        The objective is to gain speed by grouping the multiplication operations

        Warning: if the new length and ints are too low,
        the result might be cropped
        """

        # Objects can be either QFloats, SignedBinaries or Zeros
        # We will tensorize only pairs of QFloats

        a0 = None
        b0 = None
        for a in list_a:
            if isinstance(a, cls):
                a0 = a
                break
        for b in list_b:
            if isinstance(b, cls):
                b0 = b
                break

        if newlength is None:
            if a0 is not None:
                newlength = len(a0)
            elif b0 is not None:
                newlength = len(b0)

        if newints is None:
            if a0 is not None:
                newints = a0.ints
            elif b0 is not None:
                newints = b0.ints

        # make sure both the lists and arrays have all the same sizes and bases:
        assert len(list_a) == len(list_b)

        if a0 is not None and b0 is not None:
            for i in range(len(list_a)):
                if isinstance(list_a[i], cls):
                    assert len(list_a[i]) == len(a0)
                if isinstance(list_b[i], cls):
                    assert len(list_b[i]) == len(a0)
                if isinstance(list_a[i], cls):
                    assert list_a[i].base == a0.base
                if isinstance(list_b[i], cls):
                    assert list_b[i].base == a0.base
                # and check that ints are the same within a and b
                if isinstance(list_a[i], cls):
                    assert list_a[i].ints == a0.ints
                if isinstance(list_b[i], cls):
                    assert list_b[i].ints == b0.ints

            # QFloats should always be base tidy before a multiplication
            for a in list_a:
                if isinstance(a, cls):
                    assert a.is_base_tidy
            for b in list_b:
                if isinstance(b, cls):
                    assert b.is_base_tidy

        # prepate the list of multiplications with None values for now
        list_ab = [None] * len(list_a)

        # skip the easy multiplication (Zero or SignedBinary)
        indices_qfloat_mul = (
            []
        )  # list the qfloat x qfloat mul that will need to be done
        for i in range(len(list_a)):
            a = list_a[i]
            b = list_b[i]

            if isinstance(a, Zero) or isinstance(b, Zero):
                list_ab[i] = Zero()

            elif isinstance(a, SignedBinary) or isinstance(b, SignedBinary):
                if isinstance(a, SignedBinary) and isinstance(b, SignedBinary):
                    list_ab[i] = a * b
                # simple multiplication and size setting
                list_ab[i] = a * b
                list_ab[i].set_len_ints(newlength, newints)

            else:
                indices_qfloat_mul.append(i)

        n_qfloat_mul = len(indices_qfloat_mul)
        cls.MULTIPLICATION += (
            n_qfloat_mul  # count only multiplications with other Qfloat
        )

        if n_qfloat_mul == 0:
            return list_ab

        if n_qfloat_mul == 1:
            index = indices_qfloat_mul[0]
            list_ab[index] = cls.from_mul(
                list_a[index], list_b[index], newlength, newints
            )
            return list_ab

        # n_qfloat_mul > 1 means we can make tensorization to gain speed :

        mularray = fhe.zeros((n_qfloat_mul, len(a), newlength))

        # concat arrays of QFloats that will be multiplied
        a_arrays = np.concatenate(
            tuple(np.reshape(list_a[i].array, (1, -1)) for i in indices_qfloat_mul),
            axis=0,
        )
        b_arrays = np.concatenate(
            tuple(np.reshape(list_b[i].array, (1, -1)) for i in indices_qfloat_mul),
            axis=0,
        )

        for i in range(0, len(a0)):
            # index from b where the array mul should be inserted in mularray
            indb = newints - a0.ints + i + 1 - b0.ints
            # compute only needed multiplication of b._array*a._array[i],
            # accounting for crops
            ind1 = 0 if indb >= 0 else -indb
            ind2 = min(len(b0), newlength - indb)
            if ind2 > ind1:
                mul = b_arrays[:, ind1:ind2] * a_arrays[:, i].reshape((n_qfloat_mul, 1))

                # if ind2 - ind1 == 1:
                #     mul = mul.reshape((n_qfloat_mul,1))
                bpa.insert_array_at_index_3D(mul, mularray, i, indb + ind1)

        # the multiplication array is made from the sum
        # of the mularray rows with product of signs
        sum_array = np.sum(mularray, axis=1)

        # multi tidy
        sum_array = cls.multi_base_tidy(sum_array, a0.base)

        for i in range(n_qfloat_mul):
            index = indices_qfloat_mul[i]
            # fill a QFloat from the results (which are tidy)
            multiplication = QFloat(
                sum_array[i],
                newints,
                a0.base,
                True,
                list_a[index].sign * list_b[index].sign,
            )
            # multiplication = QFloat(
            #     sum_array[i], newints, a0.base, True, list_a[index].sign * list_b[index].sign
            # )
            # multiplication.base_tidy()

            # put result in the list
            assert list_ab[indices_qfloat_mul[i]] is None  # just to be sure
            list_ab[index] = multiplication

        return list_ab

    def __itruediv__(self, other):
        """
        Divide by another QFLoat, in place
        Dividing requires arrays to be tidy and will return a tidy array

        Consider two integers a and b, that we want to divide with float precision fp:
        We have: (a / b) = (a * fp) / b / fp
        Where a * fp / b is an integer division, and ((a * fp) / b / fp)
        is a float number with fp precision

        WARNING: dividing by zero will give zero
        WARNING: precision of division does not increase
        """
        if isinstance(other, Zero):
            raise ValueError("division by Zero")

        if isinstance(other, SignedBinary):
            self.self_check_convert_fhe(other.encrypted)

            # In this case, the array is either unchanged (just signed),
            # or overflowed (dividing by 0 causes overflow)
            is_zero = other.value == 0
            sign = other.value  # the value is also its sign
            self._array = (1 - is_zero) * self._array + is_zero * fhe.ones(
                len(self)
            ) * (self._base - 1)
            if self._sign is not None:
                self._sign = (1 - is_zero) * sign + is_zero * self._sign
            return self

        # other must be tidy before division
        other.tidy()

        QFloat.DIVISION += 1  # count only divisions with other Qfloat
        self.check_compatibility(other)
        # must be tidy before division
        self.tidy()

        # get signs and make arrays positive
        signa = self.get_sign()
        a = signa*(self._array)

        signb = other.get_sign()
        b = signb*(other._array)           

        # The float precision is the number of digits after the dot:
        fp = len(self) - self._ints

        # We consider each array as representing integers a and b here
        # Let's left shit the first array which corresponds by multiplying a by 2^fp:
        shift_arr = np.concatenate((a, fhe.zeros(fp)), axis=0)
        # Make the integer division (a*fp)/b with our long division algorithm:
        div_array = bpa.base_p_division(shift_arr, b, self._base)
        # The result array encodes for a QFloat with fp precision, which is equivalent to divide the result by fp
        # giving as expected the number (a * fp) / b / fp :
        self._sign = signa*signb
        self._array = div_array[fp:]*self._sign  # result is tidy and signed

        return self

    def __truediv__(self, other):
        """
        Divide by another QFLoat, see __itruediv__
        """
        division = self.copy()
        division /= other
        return division

    def __rtruediv__(self, other):
        """
        Return other divided by self
        """
        if isinstance(other, Zero):
            # special case when other is unencrypted 0, the result is an unencrypted 0
            return Zero()

        if isinstance(other, SignedBinary):
            # get sign, then invert ith this sign (can be 0)
            return self.invert(
                other.value, len(self), self._ints
            )  # the value is also its sign

        if isinstance(other, QFloat):
            return other / self

        raise ValueError("Unknown class for other")

    def invert(self, sign=1, newlength=None, newints=None):
        """
        Compute the signed invert of the QFloat, with different length and ints values if requested
        """
        if not (
            isinstance(sign, SignedBinary)
            or (isinstance(sign, numbers.Integral) and abs(sign) == 1)
        ):
            raise ValueError("sign must be a SignedBinary or a signed binary scalar")

        QFloat.DIVISION += 1  # this division is counted because it is heavy

        # must be base tidy before dividing
        self.tidy()

        if newlength is None:
            newlength = len(self)
        if newints is None:
            newints = self._ints

        a = fhe.ones(1)  # an array with one value to save computations
        
        signb = self.get_sign()
        b = self._array*signb

        # The float precision is the number of digits after the dot:
        fp = newlength - newints  # new float precision
        fpself = len(self) - self._ints  # self float precision

        # We consider each array as representing integers a and b here
        # Let's left shit the first array which corresponds by multiplying
        # a by 2^(fpself + fp) (decimals of old+new precision):
        shift_arr = np.concatenate((a, fhe.zeros(fpself + fp)), axis=0)
        # Make the integer division (a*fp)/b with our long division algorithm:
        div_array = bpa.base_p_division(shift_arr, b, self._base)

        # Correct size of div_array
        diff = newlength - div_array.size
        if diff > 0:
            div_array = np.concatenate((fhe.zeros(diff), div_array), axis=0)
        else:
            div_array = div_array[-diff:]
        # The result array encodes for a QFloat with fp precision,
        # which is equivalent to divide the result by fp,
        # giving as expected the number (a * fp) / b / fp :
        newsign = sign * signb
        invert_div = QFloat(div_array*newsign, newints, self._base, True, newsign)

        return invert_div

    @classmethod
    def multi_invert(cls, list_qfloats, sign=1, newlength=None, newints=None):
        """
        Compute the signed invert of the QFloat, with different length and ints values if requested
        """
        if not (
            isinstance(sign, SignedBinary)
            or (isinstance(sign, numbers.Integral) and abs(sign) == 1)
        ):
            raise ValueError("sign must be a SignedBinary or a signed binary scalar")

        qf0 = list_qfloats[0]
        for qfloat in list_qfloats:
            assert isinstance(qfloat, cls)
            assert qfloat.is_base_tidy
            assert len(qfloat) == len(qf0)
            assert qfloat.base == qf0.base
            assert qfloat.ints == qf0.ints

        n_qfloats = len(list_qfloats)

        QFloat.DIVISION += n_qfloats  # this division is counted because it is heavy

        if newlength is None:
            newlength = len(qf0)
        if newints is None:
            newints = qf0.ints

        a_arrays = fhe.ones(
            (n_qfloats, 1)
        )  # arrays with one value to save computations
        b_arrays = np.concatenate(
            tuple(np.reshape(list_qfloats[i].array, (1, -1)) for i in range(n_qfloats)),
            axis=0,
        )

        # The float precision is the number of digits after the dot:
        fp = newlength - newints  # new float precision
        fpself = len(qf0) - qf0.ints  # current float precision

        # We consider each array as representing integers a and b here
        # Let's left shit the first array which corresponds by multiplying
        # a by 2^(fpself + fp) (decimals of old+new precision):
        shift_arr = np.concatenate(
            (a_arrays, fhe.zeros((n_qfloats, fpself + fp))), axis=1
        )
        # Make the integer division (a*fp)/b with our long division algorithm:
        div_array = bpa.multi_base_p_division(shift_arr, b_arrays, qf0.base)

        # Correct size of div_array
        diff = newlength - div_array.shape[1]
        if diff > 0:
            div_array = np.concatenate(
                (fhe.zeros((n_qfloats, diff)), div_array), axis=1
            )
        else:
            div_array = div_array[:, -diff:]
        # The result array encodes for a QFloat with fp precision,
        # which is equivalent to divide the result by fp,
        # giving as expected the number (a * fp) / b / fp :
        results = []
        for i in range(n_qfloats):
            newsign = sign * list_qfloats[i].sign
            results.append(QFloat(div_array[i, :], newints, qf0.base, True, newsign))

        return results
