import sys, os, time

import unittest
import numpy as np
from concrete import fhe

sys.path.append(os.getcwd())

from matrix_inversion.QFloat import QFloat, SignedBinary


def print_red(text):
    # ANSI escape sequence for red color
    red_color = "\033[91m"
    # ANSI escape sequence to reset color back to default
    reset_color = "\033[0m"
    # Print text in red
    print(red_color + text + reset_color)    

def measure_time(function, descripton, *inputs):
    #Compute a function on inputs and return output along with duration
    print(descripton+' ...', end="", flush=True)
    print("\r", end="")
    start = time.time()
    output = function(*inputs)
    end = time.time()
    print(f"|  {descripton} : {end-start:.2f} s  |")
    return output

def float_array_to_qfloat_arrays_fhe(arr, qf_len, qf_ints, qf_base):
    """
    converts a float list to arrays representing qfloats
    """
    qf_array = [ QFloat.fromFloat(f, qf_len, qf_ints, qf_base) for f in arr]
    n=len(qf_array)
    qf_arrays = fhe.zeros((n, qf_len))
    qf_signs = fhe.zeros(n)
    for i in range(n):
        qf_arrays[i,:] = qf_array[i].toArray()
        qf_signs[i] = qf_array[i].getSign()

    return qf_arrays, qf_signs

def float_array_to_qfloat_arrays_python(arr, qf_len, qf_ints, qf_base):
    """
    converts a float list to arrays representing qfloats
    """
    qf_array = [ QFloat.fromFloat(f, qf_len, qf_ints, qf_base) for f in arr]
    n=len(qf_array)
    qf_arrays = np.zeros((n, qf_len), dtype='int')
    qf_signs = np.zeros(n, dtype='int')
    for i in range(n):
        qf_arrays[i,:] = qf_array[i].toArray()
        qf_signs[i] = qf_array[i].getSign()

    return qf_arrays, qf_signs


def qfloat_arrays_to_qfloat_list(qf_arrays, qf_signs, qf_ints, qf_base):
    """
    converts qfloats arrays to a QFloat matrix
    """
    n = int(qf_arrays.shape[0])
    qf_L = []
    for i in range(n):
        qf = QFloat(qf_arrays[i,:], qf_ints, qf_base, True, qf_signs[i])
        qf_L.append(qf)

    return qf_L

def qfloat_list_to_qfloat_arrays(L, qf_len, qf_ints, qf_base):
    """
    converts a QFloat 2D-list matrix to integer arrays 
    """
    if not isinstance(L, list):
        raise TypeError('L must be list')
    n=len(L)
    qf_arrays = fhe.zeros((n, qf_len))
    for i in range(n):
        if isinstance(L[i], QFloat):
            qf_arrays[i,:] = L[i].toArray()
        elif isinstance(L[i], SignedBinary):
            qf_arrays[i,qf_ints-1] = L[i].value
        elif isinstance(L[i], Zero):
            qf_arrays[i,qf_ints-1] = 0
        else:
            qf_arrays[i,qf_ints-1] = L[i]

    return qf_arrays

def qfloat_arrays_to_float_array(qf_arrays, qf_ints, qf_base):
    """
    converts qfloats arrays to a float matrix
    """
    n = int(qf_arrays.shape[0])
    arr=np.zeros(n)
    for i in range(n):
        arr[i] = QFloat(qf_arrays[i,:], qf_ints, qf_base).toFloat()

    return arr



class QFloatCircuit:
   
    """
    Circuit factory class for testing FheSeq on 2 sequences input
    """
    def __init__(self, n, circuit_function, qf_len, qf_ints, qf_base, verbose=False):
        inputset=[]
        for i in range(100):
            floatList = [np.random.uniform(0,100,1)[0] for i in range(n)]
            qf_arrays, qf_signs = float_array_to_qfloat_arrays_python(floatList, qf_len, qf_ints, qf_base)
            inputset.append((qf_arrays, qf_signs))

        params = [qf_len, qf_ints, qf_base]
        compiler = fhe.Compiler(lambda x,y: circuit_function(x,y,params), {"x": "encrypted", "y": "encrypted"})
        make_circuit = lambda : compiler.compile(
            inputset=inputset,
            configuration=fhe.Configuration(
                enable_unsafe_features=True,
                use_insecure_key_cache=True,
                insecure_key_cache_location=".keys",
                #dataflow_parallelize=True,
            ),
            verbose=False,
        )
        self.qf_len = qf_len
        self.qf_ints = qf_ints
        self.qf_base = qf_base

        self.circuit = measure_time( make_circuit, 'Compiling')

    
    def run(self, floatList, simulate=False, raw_output=False):
        if not self.circuit: raise Error('circuit was not set')
        qf_arrays, qf_signs = float_array_to_qfloat_arrays_fhe(floatList, self.qf_len, self.qf_ints, self.qf_base)

        # Run FHE
        if not simulate:
            encrypted = measure_time(self.circuit.encrypt, 'Encrypting', qf_arrays, qf_signs)
            run = measure_time(self.circuit.run,'Running', encrypted)
            decrypted = self.circuit.decrypt(run)
        else:
            decrypted = measure_time(self.circuit.simulate,'Simulating', qf_arrays, qf_signs)

        if not raw_output:
            qf_Res = qfloat_arrays_to_float_array(decrypted, self.qf_ints, self.qf_base)
        else:
            qf_Res = decrypted

        return qf_Res



class TestQFloat(unittest.TestCase):

##################################################  FHE TESTS ##################################################

    def test_add_sub_fhe(self):
        # test add and sub

        def add_qfloats(qf_arrays, qf_signs, params):
            qf_len, qf_ints, qf_base = params
            a,b = qfloat_arrays_to_qfloat_list(qf_arrays, qf_signs, qf_ints, qf_base)
            res = [a+b]
            return qfloat_list_to_qfloat_arrays(res, qf_len, qf_ints, qf_base)

        for i in range(10):
            #base = np.random.randint(2,10)
            base = 2
            size = np.random.randint(20,30)
            ints = np.random.randint(12, 16)
            f1 = np.random.uniform(0,100,1)[0]
            f2 = np.random.uniform(0,100,1)[0]

            circuit = QFloatCircuit(2, add_qfloats, size, ints, base)
            addition = circuit.run(np.array([f1,f2]), False)[0]
            assert(addition  - (f1+f2) < 0.01)


    def test_mul_fhe(self):
        # test add and sub

        def mul_qfloats(qf_arrays, qf_signs, params):
            qf_len, qf_ints, qf_base = params
            a,b = qfloat_arrays_to_qfloat_list(qf_arrays, qf_signs, qf_ints, qf_base)
            res = [a*b]
            return qfloat_list_to_qfloat_arrays(res, qf_len, qf_ints, qf_base)

        for i in range(10):
            #base = np.random.randint(2,10)
            base = 2
            size = np.random.randint(20,30)
            ints = np.random.randint(12, 16)
            f1 = np.random.uniform(0,100,1)[0]
            f2 = np.random.uniform(0,100,1)[0]

            circuit = QFloatCircuit(2, mul_qfloats, size, ints, base)
            multiplication = circuit.run(np.array([f1,f2]), False)[0]
            assert(multiplication  - (f1*f2) < 0.01)

    def test_div_fhe(self):
        # test add and sub

        def div_qfloats(qf_arrays, qf_signs, params):
            qf_len, qf_ints, qf_base = params
            a,b = qfloat_arrays_to_qfloat_list(qf_arrays, qf_signs, qf_ints, qf_base)
            res = [a/b]
            return qfloat_list_to_qfloat_arrays(res, qf_len, qf_ints, qf_base)

        for i in range(10):
            #base = np.random.randint(2,10)
            base = 2
            size = np.random.randint(20,30)
            ints = np.random.randint(12, 16)
            f1 = np.random.uniform(0,100,1)[0]
            f2 = np.random.uniform(0,100,1)[0]

            circuit = QFloatCircuit(2, div_qfloats, size, ints, base)
            division = circuit.run(np.array([f1,f2]), False)[0]
            assert(division  - (f1/f2) < 0.01)  


    # def test_multi_fhe(self):
    #     # test multi operations to count time

    #     def multi_qfloats(qf_arrays, qf_signs, params):
    #         qf_len, qf_ints, qf_base = params
    #         a,b = qfloat_arrays_to_qfloat_list(qf_arrays, qf_signs, qf_ints, qf_base)
    #         res = a+a+a+a-b
    #         res = res*a*a
            
    #         return qfloat_list_to_qfloat_arrays([res], qf_len, qf_ints, qf_base)

    #     for i in range(10):
    #         #base = np.random.randint(2,10)
    #         base = 2
    #         size = np.random.randint(20,30)
    #         ints = np.random.randint(12, 16)
    #         f1 = np.random.uniform(0,100,1)[0]
    #         f2 = np.random.uniform(0,100,1)[0]

    #         QFloat.KEEP_TIDY=False
    #         circuit = QFloatCircuit(2, multi_qfloats, size, ints, base)
    #         multi = circuit.run(np.array([f1,f2]), False)[0]
    #         QFloat.KEEP_TIDY=True
    #         #assert(division  - (f1/f2) < 0.01)  


unittest.main()

# suite = unittest.TestLoader().loadTestsFromName('test_QFloat.TestQFloat.test_multi_fhe')
# unittest.TextTestRunner(verbosity=1).run(suite)