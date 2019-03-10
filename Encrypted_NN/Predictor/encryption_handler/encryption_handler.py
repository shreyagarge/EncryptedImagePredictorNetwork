import seal
import numpy as np
from seal import ChooserEvaluator,     \
                 Ciphertext,           \
                 Decryptor,            \
                 Encryptor,            \
                 EncryptionParameters, \
                 Evaluator,            \
                 IntegerEncoder,       \
                 FractionalEncoder,    \
                 KeyGenerator,         \
                 MemoryPoolHandle,     \
                 Plaintext,            \
                 SEALContext,          \
                 EvaluationKeys,       \
                 GaloisKeys,           \
                 PolyCRTBuilder,       \
                 ChooserEncoder,       \
                 ChooserEvaluator,     \
                 ChooserPoly
                 
class EncryptionHandler:

    def __init__(self, config):

        # pass
        self._context = self._build_context(config)

        self._setup_members(self._context, config)
        

    def _build_context(self, config):

        #set up encryption parameters and context
        parms = EncryptionParameters()
        parms.set_poly_modulus(config['poly_modulus'])
        parms.set_coeff_modulus(seal.coeff_modulus_128(config['coeff_modulus']))
        parms.set_plain_modulus(1 << 18)
        context = SEALContext(parms)

        return context

    def _create_keys(self, context):

        keygen = KeyGenerator(context)
        public_key = keygen.public_key()
        secret_key = keygen.secret_key()

        ev_keys16 = EvaluationKeys()
        keygen.generate_evaluation_keys(16, ev_keys16)

        return public_key, secret_key, ev_keys16

    def _setup_members(self, context, config):

        self._encoder = FractionalEncoder(context.plain_modulus(),context.poly_modulus(),config['int_coeff'],config['fract_coeff'],config['base'])
        self._evaluator = Evaluator(context)

        pk, sk, self._ev_key = self._create_keys(context)
        self._encryptor = Encryptor(context, pk)
        self._decryptor = Decryptor(context, sk)

    def get_matrix(self, mat):

        shape = mat.shape

        assert(len(shape) == 1 or len(shape) == 4)

        if len(shape)==1:
            p_c = np.zeros((shape[0]))
            for i in range(shape[0]):
                p_c[i] = self.decrypt_ciphertext(mat[i])
            
        

        else:
            p_c = np.zeros((shape[0],shape[1],shape[2],shape[3]))
            for i in range(shape[1]):
                for j in range(shape[2]):
                    for k in range(shape[3]):
                        p_c[0,i,j,k] = self.decrypt_ciphertext(mat[0,i,j,k])
            
        return p_c




    def decrypt_ciphertext(self, cipher):

        plain = Plaintext()
        self.decryptor.decrypt(cipher, plain)
        pl =  self.encoder.decode(plain)
        
        return pl


    def re_encrypt(self, x):
        
        shape = x.shape
        prod = 1
        for s in shape:
            prod = prod*s

        x = np.reshape(x,prod)

        temp_x = self.get_matrix(x)

        x = []
        for i in range(prod):
            x.append(Ciphertext())
            self.encryptor.encrypt(self.encoder.encode(temp_x[i]), x[i])
        x = np.reshape(x,shape)
        
        return x

    @property
    def encryptor(self):
        return self._encryptor

    @property
    def evaluator(self):
        return self._evaluator

    @property
    def encoder(self):
        return self._encoder

    @property
    def decryptor(self):
        return self._decryptor

    @property
    def package(self):

        pack = self._Package(self.encoder, self.evaluator, self.encryptor, self._ev_key, self.re_encrypt)
        return pack


    class _Package:

        def __init__(self, encoder, evaluator, encryptor, ev_key, re_encrypt_function):

            self.encoder = encoder
            self.evaluator = evaluator
            self.encryptor = encryptor
            self.Ciphertext = self._new_ciphertext
            self.Plaintext = self._new_plaintext #remove in the end
            self.evaluation_key = ev_key
            self.re_encrypt = re_encrypt_function

        def _new_ciphertext(self, arg = None):

            if arg is None:
                return Ciphertext()
            else:
                return Ciphertext(arg)

        def _new_plaintext(self):

            return Plaintext()




