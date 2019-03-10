from encryption_config import config
from encryption_handler.encryption_handler import EncryptionHandler
from server_net.predictor_net import Predictor
# import PyQt4
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import time
import sys

#print(matplotlib.get_backend())


def main():

    debug = False
    if len(sys.argv)>2:
        if sys.argv[2] == 'debug':
            debug = True


    imfile = "Predictor/images/"+sys.argv[1]+".npy"
    test_im = np.load(imfile)

    plt.imshow(np.reshape(test_im,(28,28)))
    # plt.plot([1,2,2,3,4,5])
    plt.show()
    # plt.savefig("newfigtest")
    # raw_input()
    start1=time.clock()
    handler = EncryptionHandler(config)
    op = handler.package

    #encrypt image
    ln = test_im.shape[0]
    start = time.clock()
    encrypted_image = []
    for i in range(ln):
        encrypted_image.append((op.Ciphertext()))
        handler.encryptor.encrypt(handler.encoder.encode(test_im[i]), encrypted_image[i])
    print("time taken for encrypting image:  " + (str)(time.clock() - start)+"s")
    print("Noise budget in fresh encryption: " + (str)(handler.decryptor.invariant_noise_budget(encrypted_image[100])) + " bits")
  
    #call the predictor 
    # predictor =  Predictor(op, debug = True, decryptor = handler.decryptor)
    predictor = Predictor(op,debug= debug, handler= handler)
    logits = predictor.predict_image(encrypted_image)

    start = time.clock()
    dec_logits=(handler.get_matrix(logits))
    print(dec_logits)
    print("Prediction : "+(str)(np.argmax(dec_logits)))
    print("time taken for decrypting image:  " + (str)(time.clock() - start)+"s")
    print("total time taken for the network: " + (str)(time.clock() - start1)+"s")
    # count = 0
    # for i in a:
    #     print(count+1)
    #     print(handler.decrypt_mat(i))
    #     count+=1
if __name__ == '__main__':
    main()