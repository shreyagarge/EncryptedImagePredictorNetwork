import numpy as np 
import logging
import time
# give path relative to docker root (Encrypted_NN)

path = {'weights':'Predictor/server_net/models/model_5/weights/',
        'biases':'Predictor/server_net/models/model_5/biases/'}


class Predictor:

    def __init__(self, enc_operators, debug = False, handler = None):

        self._encoder = enc_operators.encoder
        self._encryptor = enc_operators.encryptor
        self._evaluator = enc_operators.evaluator
        self._Ciphertext = enc_operators.Ciphertext
        self._ev_key = enc_operators.evaluation_key
        self._denoise = enc_operators.re_encrypt
        self.debug = debug

        self._model = self._load_model()

        logging.basicConfig(format='\n[%(levelname)s]: %(asctime)s {%(name)s} \n\t\t-- %(message)s\n',datefmt='%I:%M:%S %p')
        self._logger = logging.getLogger(__name__)

        if self.debug:
            self._logger.setLevel(logging.DEBUG)
            
            if handler is not None:
                self._debughandler = handler
                self._Plaintext = enc_operators.Plaintext
        else: self._logger.setLevel(logging.INFO)


    def _load_model(self):

        model = {}
        model['wc1'] = np.load(path['weights']+'weights1.npy')
        model['wc2'] = np.load(path['weights']+'weights2.npy')
        model['wf1'] = np.load(path['weights']+'weights3.npy')
        model['otw'] = np.load(path['weights']+'weights4.npy')

        model['bc1'] = np.load(path['biases']+'bc1.npy')
        model['bc2']= np.load(path['biases']+'bc2.npy')
        model['bf1']= np.load(path['biases']+'bd1.npy')
        model['otb'] = np.load(path['biases']+'bout.npy')

        return model

    def predict_image(self, image):

        # image is 724 long Ciphertext object array
        self._logger.debug('Reshaping Image to 1 x 28 x 28 x 1')
        image = np.reshape(image,(1,28,28,1))

        

        # layer 1 - convolution, bias addition 
        self._logger.info('First convolution')
        start = time.clock()
        con = self._conv2d(image,self._model['wc1'],debug=False)
        con = self._add_bias(con,self._model['bc1'])
        print("time taken for first convolution:  " + (str)(time.clock() - start)+"s")

        # self._logger.debug('con1b shape: %s', str(con.shape))
        if self.debug:
            print("Noise budget after first convolution: " + (str)(self._debughandler.decryptor.invariant_noise_budget(con[0,15,15,1])) + " bits")



        # layer 2 - activation 
        self._logger.info('activation')
        start = time.clock()
        con = self._activ_fun(con) 
        print("time taken for first activation function:  " + (str)(time.clock() - start)+"s")
    
        # self._logger.debug('con1af shape: %s', str(con.shape))
        if self.debug:
            print("Noise budget after first activation: " + (str)(self._debughandler.decryptor.invariant_noise_budget(con[0,15,15,1])) + " bits")

        con = self._denoise(con)
        if self.debug:
            print("Noise budget after denoise: " + (str)(self._debughandler.decryptor.invariant_noise_budget(con[0,15,15,1])) + " bits")

        # layer 3 - mean pooling
        self._logger.info('meanpooling')
        start = time.clock()
        con = self._meanpool2(con)
        print("time taken for first meanpool:  " + (str)(time.clock() - start)+"s")
    
        # self._logger.debug('con1mean shape: %s', str(con.shape))
        if self.debug:
            print("Noise budget after first meanpool: " + (str)(self._debughandler.decryptor.invariant_noise_budget(con[0,8,8,1])) + " bits")



        # layer 4 - convolution, bias addition 
        self._logger.info('Second convolution')
        start = time.clock()
        con = self._conv2d(con,self._model['wc2'],debug=False)
        con = self._add_bias(con,self._model['bc2'])
        print("time taken for second convolution:  " + (str)(time.clock() - start)+"s")
        # self._logger.debug('con2b shape: %s', str(con.shape))
        
        if self.debug:
            print("Noise budget after second convolution: " + (str)(self._debughandler.decryptor.invariant_noise_budget(con[0,8,8,1])) + " bits")
        con = self._denoise(con)
        if self.debug:
            print("Noise budget after denoise: " + (str)(self._debughandler.decryptor.invariant_noise_budget(con[0,8,8,1])) + " bits")


        # layer 5 - mean pooling 
        self._logger.info('meanpooling')
        start = time.clock()
        con = self._meanpool2(con)
        print("time taken for second meanpooling:  " + (str)(time.clock() - start)+"s")
        # self._logger.debug('mean2 shape: %s', str(con.shape))

        if self.debug:
            print("Noise budget after second meanpool: " + (str)(self._debughandler.decryptor.invariant_noise_budget(con[0,5,5,1])) + " bits")



        # layer 6 - fully connected 
        self._logger.info('Fully connected layer 1')
        newshape = con.shape[1]*con.shape[2]*con.shape[3]
        fc = np.reshape(con,(newshape))
        start =  time.clock()
        fc = self._fully_connect(fc,self._model['wf1'])
        print("time taken for first fully connected layer:  " + (str)(time.clock() - start)+"s")
        if self.debug:
            print("Noise budget after first fcl: " + (str)(self._debughandler.decryptor.invariant_noise_budget(fc[99])) + " bits")
        fc = self._denoise(fc)
        if self.debug:
            print("Noise budget after denoise: " + (str)(self._debughandler.decryptor.invariant_noise_budget(fc[99])) + " bits")
        fc = self._add_bias1(fc,self._model['bf1'])
        


        # layer 7 - activation
        self._logger.info('activation')
        start =  time.clock()
        fc = self._activ_fun(fc)
        print("time taken for activation 2:  " + (str)(time.clock() - start)+"s")
        
        
        if self.debug:
            print("Noise budget after first fcl act: " + (str)(self._debughandler.decryptor.invariant_noise_budget(fc[99])) + " bits")
        fc = self._denoise(fc)
        if self.debug:
            print("Noise budget after denoise: " + (str)(self._debughandler.decryptor.invariant_noise_budget(fc[99])) + " bits")

        # layer 8 - fully connected
        self._logger.info('Fully connected layer 2')
        start =  time.clock()
        fc2 = self._fully_connect(fc,self._model['otw'])
        print("time taken for second fcl:  " + (str)(time.clock() - start)+"s")
        if self.debug:
            print("Noise budget second fcl: " + (str)(self._debughandler.decryptor.invariant_noise_budget(fc2[4])) + " bits")
        fc = self._denoise(fc)
        if self.debug:
            print("Noise budget after denoise " + (str)(self._debughandler.decryptor.invariant_noise_budget(fc2[4])) + " bits")
        logits = self._add_bias1(fc2,self._model['otb'])
        
        

        return logits


    def _dotProduct(self,enc_x, plain_b, debug = False):
        
        dt = np.dtype('O')
        dotProduct = np.zeros((len(enc_x), len(plain_b[0])), dtype = dt)

        if len(enc_x[0,:]) != len(plain_b[:,0]):
            return 0    
            self._logger.error("dimension mismatch for dot product")
        
        for j in range(len(dotProduct[:,0])):
            if debug:
                self._logger.debug("%d / %d",j,len(dotProduct[:,0]))
            for i in range(len(dotProduct[0, :])):
                sumt = self._Ciphertext()
                self._encryptor.encrypt(self._encoder.encode(0),sumt)
                
                for column in range(len(enc_x[0, :])):
                   
                    temp_x = self._Ciphertext(enc_x[j,column])
                    self._evaluator.multiply_plain(temp_x, self._encoder.encode(plain_b[column,i]))
                    
                    self._evaluator.add(sumt,temp_x)
                    
                dotProduct[j, i] = sumt
                # if debug:
                #     self._logger.debug("idx: %d,%d: %s",j,i,str(self._debughandler.decrypt_ciphertext(dotProduct[j,i])))
        
        return dotProduct


    def _calc_pad(self, pad, in_siz, out_siz, stride, ksize):
        """Calculate padding width.

        Args:
            pad: padding method
            ksize: kernel size [I, J].

        Returns:
            pad_: Actual padding width.
        """
        if pad == 'SAME':
            return (out_siz - 1) * stride + ksize - in_siz
        elif pad == 'VALID':
            return 0
        else:
            return pad


    def _calc_size(self, h, kh, pad, sh):
        """Calculate output image size on one dimension.

        Args:
            h: input image size.
            kh: kernel size.
            pad: padding strategy.
            sh: stride.

        Returns:
            s: output size.
        """

        if pad == 'VALID':
            return np.ceil((h - kh + 1) / sh)
        elif pad == 'SAME':
            return np.ceil(h / sh)
        else:
            return int(np.ceil((h - kh + pad + 1) / sh))



    def _extract_sliding_windows(self,x, ksize, pad, stride, floor_first=True):
        """Converts a matrix to sliding windows.

        Args:
            x: [N, H, W, C]
            k: [KH, KW]
            pad: [PH, PW]
            stride: [SH, SW]

        Returns:
            y: [N, (H-KH+PH+1)/SH, (W-KW+PW+1)/SW, KH * KW, C]
        """
        n = x.shape[0]
        h = x.shape[1]
        w = x.shape[2]
        c = x.shape[3]
        kh = ksize[0]
        kw = ksize[1]
        sh = stride[0]
        sw = stride[1]

        h2 = int(self._calc_size(h, kh, pad, sh))
        w2 = int(self._calc_size(w, kw, pad, sw))
        ph = int(self._calc_pad(pad, h, h2, sh, kh))
        pw = int(self._calc_pad(pad, w, w2, sw, kw))

        ph0 = int(np.floor(ph / 2))
        ph1 = int(np.ceil(ph / 2))
        pw0 = int(np.floor(pw / 2))
        pw1 = int(np.ceil(pw / 2))

        if floor_first:
            pph = (ph0, ph1)
            ppw = (pw0, pw1)
        else:
            pph = (ph1, ph0)
            ppw = (pw1, pw0)
        x = np.pad(
            x, ((0, 0), pph, ppw, (0, 0)),
            mode='constant',
            constant_values=(0.0, ))
        dt = np.dtype('O')
        y = np.zeros([n, h2, w2, kh, kw, c], dtype = dt)
        #y = np.zeros([n, h2, w2, kh, kw, c])
        for ii in range(h2):
            for jj in range(w2):
                xx = ii * sh
                yy = jj * sw
                y[:, ii, jj, :, :, :] = x[:, xx:xx + kh, yy:yy + kw, :]
        return y




    def _conv2d(self,x, w, pad='SAME', stride=(1, 1), debug = False):
        """2D convolution 

        Args:
            x: [N, H, W, C]
            w: [I, J, C, K]
            pad: [PH, PW]
            stride: [SH, SW]

        Returns:
            y: [N, H', W', K]
        """

        ksize = w.shape[:2]
        x = self._extract_sliding_windows(x, ksize, pad, stride)
        ws = w.shape
        w = w.reshape([ws[0] * ws[1] * ws[2], ws[3]])
        xs = x.shape
        x = x.reshape([xs[0] * xs[1] * xs[2], -1])
       
        # the matrix was padded with zeros. so we replace those zeros with encrypted value of 0,
        # in order to maintain uniformity of object types in the matrix 
        enc_0 = self._Ciphertext()
        self._encryptor.encrypt(self._encoder.encode(0),enc_0)

        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                if x[i,j] == 0:
                    x[i,j] = enc_0

        y = self._dotProduct(x,w, debug=False)
        y = y.reshape([xs[0], xs[1], xs[2], -1])
        return y


    def _add_bias(self,a,b):
        """Bias addition after convolution -  bias to be added to the entire matrix in each channel
    
        Args:
            a: [1,height,width,channels] 
            b: [channels]

        Returns:
            a: [1,height,width,channels]
        """
        
        for i in range(a.shape[3]):
            for j in range(a.shape[2]):
                for k in range(a.shape[1]):
                    self._evaluator.add_plain(a[0,j,k,i],self._encoder.encode(b[i]))
                    
        return a

    def _add_bias1(self,x,y):
        """bias addition after fc - only one channel but different bias added to each node(element in the array)

        Args:
            x: [#nodes]
            y: [#nodes]

        Returns:
            x: [#nodes]
        """
        
        for i in range(len(x)):
            self._evaluator.add_plain(x[i],self._encoder.encode(y[i]))
            
        return x

    def _activ_fun(self,x):
        """activation function - applies the activation (square) function to every element in the input array/ matrix

        Args:
            x

        Returns:
            x
        """
        if len(x.shape) == 1:
            s1 = x.shape[0]
            squared = np.zeros((s1))
            for i in range(s1):
                self._evaluator.square(x[i])
                self._evaluator.relinearize(x[i],self._ev_key)
                
        else:
            s1 = x.shape[1]
            s2 = x.shape[2]
            s3 = x.shape[3]
            
            for i in range(s1):
                for j in range(s2):
                    for k in range(s3):
                        self._evaluator.square(x[0,i,j,k])
                        self._evaluator.relinearize(x[0,i,j,k],self._ev_key)
        return x


    def _meanpool2(self,x):
        """meanpool2 takes (1,height,width,channels) input and performs meanpooling on each of the #channels
           matrices seperately and gives a (1,height/2,width/2,channels) output

        Args:
            x: [1,n,h,c]

        Returns:
            y: [1,n/2,h/2,c]
        """
        dt = np.dtype('O')
        retval = np.zeros((1,int(x.shape[1]/2),int(x.shape[2]/2),x.shape[3]),dtype = dt)
        for chan in range(x.shape[3]):
            ii,jj,i,j=0,0,0,0
            while i < x.shape[1]:
                j,jj=0,0
                while j < x.shape[2]:
                    res = self._Ciphertext()
                    advals = [x[0,i,j,chan],x[0,i+1,j,chan],x[0,i,j+1,chan],x[0,i+1,j+1,chan]]
                    self._evaluator.add_many(advals,res)
                    self._evaluator.multiply_plain(res,self._encoder.encode(0.25))
                    retval[0,ii,jj,chan] = res
                    jj+=1
                    j+=2
                ii+=1
                i+=2

        return retval

        """fully_connect takes an array of length n input and multiplies with an (n x m) matrix to give an array of length m output

        Args:
            x: [n]
            y: [n,m]

        Returns:
            z: [m]
        """
    def _fully_connect(self,x,y):
        retval = np.zeros((y.shape[1]),dtype=np.dtype('O'))

        for i in range(y.shape[1]):
            res = self._Ciphertext()
            self._encryptor.encrypt(self._encoder.encode(0),res)
            for j in range(y.shape[0]):
                temp_x = self._Ciphertext(x[j])
                self._evaluator.multiply_plain(temp_x, self._encoder.encode(y[j,i]))
                self._evaluator.add(res,temp_x)
            retval[i] = res
        return retval
