# Importing Libraries

from foolbox.criteria import TargetClass
from foolbox.criteria import Misclassification

from numpy import linalg as LA
import matplotlib.pyplot as plt

from foolbox.attacks import CarliniWagnerL2Attack
from foolbox.attacks import SaliencyMapAttack
from foolbox.attacks import GradientSignAttack

from foolbox.v1.attacks import FGSM
from foolbox.v1.attacks import MomentumIterativeAttack
#from foolbox.v1.attacks import GradientSignAttack

from skimage.measure import compare_ssim


from keras import layers, models

import numpy as np

from keras.utils import np_utils

from keras import backend as K
from keras.applications import vgg16

import tensorflow as tf


import pickle

import foolbox

import json

import timeit
start = timeit.default_timer()

import cvxpy as cp
from numpy import linalg as LA
from ISMAIL_big_picture_journal_lib import sup_lbl_from_lbl,get_S_T_S_T_comp_from_lbl,Imperceptibility,ADMM_,Attack_performance,cvxPy_pert_gen

########################################################################
###############################################  Fashion MNIST dataset import
############################################################################

#tf.keras.backend.set_learning_phase(False)
# Keras Parameters
batch_size = 28
nb_classes = 10
nb_epoch = 2
img_rows, img_col = 28, 28
img_channels = 1
# download mnist data and split into train and test sets
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
# reshape data to fit model
X_train = train_images.reshape(train_images.shape[0], 28, 28, 1)
X_test = test_images.reshape(test_images.shape[0], 28, 28, 1)
X_train, X_test = X_train/255, X_test/255
# normalization:
train_images = train_images / 255
test_images = test_images / 255
print("")

y_train = np_utils.to_categorical(train_labels,10)
y_test = np_utils.to_categorical(test_labels,10)

X_train_1d = X_train.reshape(60000,784,1)
X_test_1d  = X_test.reshape(10000,784,1)

################################################################################
############## Loading the model and preprocessing #####################
######################################################################################

########### load the propoer model here


model1 = tf.keras.models.load_model('my_model_1d_last_dense_activation_seperate')
model1.summary()
####################################################################





####################################################################################
############RE-LABEL TRAIN_LABELS AND TEST_LABELS (Using a dictonary) #########################
######################################################################################
dic5 = {2:0, 4:0, 6:0, 5:2, 7:2, 9:2, 8:4}
train_labels_5 = [dic5[x] if x in dic5.keys() else x for x in train_labels]
test_labels_5 = [dic5[x] if x in dic5.keys() else x for x in test_labels]

'''
your mapping is different than mine. Here is the mapping from the paper you gave me.
0 ==> {0,2,4,6} top
1 ==> {1} bottom
2 ==> {5,7,9} shoes
3 ==> {3} dress
4 ==> {8}
'''
######################################################################################
# #####################################################################
################### loading Grads and testing the vectorization
#####################################################################

Grad_MNIST_model1      = pickle.load(open("/home/user/.PyCharmCE2019.1/config/scratches/saved_models_variables/Grad_MNIST_model1_1d_before_SM.p","rb"))

disc_values            = pickle.load(open("/home/user/.PyCharmCE2019.1/config/scratches/saved_models_variables/disc_values_before_SM.p","rb"))


################################################################################
##################################### BUILDING THE ALG - 1 PROBLEM WITH CVXPY ######
################################################################################

######## to save eta, ceate a vectorized empty np array of size 10000,28*28,1
number_of_observations = 10000

### tensors to save and to calc CApert, CApert_sup, ELA, RLA, and sigmas
eta_vec                     = np.zeros(shape=(number_of_observations,28*28,1))
imperceptibility_rho_2_save = np.nan*np.ones(shape=(number_of_observations,1))
imperceptibility_rho_i_save = np.nan*np.ones(shape=(number_of_observations,1))
imperceptibility_sssim_save = np.nan*np.ones(shape=(number_of_observations,1))
pred_pert_lbls              = np.zeros(shape=(number_of_observations))
pred_pert_sup_lbls          = np.zeros(shape=(number_of_observations))
pred_lbls                   = np.zeros(shape=(number_of_observations))

cnt = 0

Q = 3
epsilon_D = 0.18

######################### loading perturbations from MIFGSM
MIFGSM_perturbed_images              = pickle.load(open("/home/user/.PyCharmCE2019.1/config/scratches/saved_models_variables/MIFGSM_perturbed_images.p","rb"))

MIFGSM_perturbations                 = pickle.load(open("/home/user/.PyCharmCE2019.1/config/scratches/saved_models_variables/MIFGSM_perturbations.p","rb"))

MIFGSM_pred_label_w_pert             = pickle.load(open("/home/user/.PyCharmCE2019.1/config/scratches/saved_models_variables/MIFGSM_pred_label_w_pert.p","rb"))

MIFGSM_pred_label_w_pert_super_label = pickle.load(open("/home/user/.PyCharmCE2019.1/config/scratches/saved_models_variables/MIFGSM_pred_super_label_w_pert.p","rb"))






for id in range(number_of_observations):

    ######## LET THE INPUT IMAGE be:
    id = id
    input_image = X_test_1d[id]

    input_image_reshaped = input_image.reshape(784)

    ######## get tru_lbl
    tru_lbl = test_labels[id]

    ######## get tru_sup_lbl
    tru_sup_lbl = sup_lbl_from_lbl(tru_lbl)

    ######## get pred_lbl
    pred_lbl = np.argmax(model1(input_image.reshape(1, 784, 1)))
    pred_lbls[id] = pred_lbl

    ######## get_pred_sup_lbl
    pred_sup_lbl = sup_lbl_from_lbl(pred_lbl)

    ######## get S_T and S_T_comp: this is based on the tru lbl not the predicted lbl
    [S_T,S_T_comp] = get_S_T_S_T_comp_from_lbl(tru_lbl)

    ######## get vectozied gradients and disc values of of the disgnated lbl

    Grad_MNIST_model1_vec_disgnated = Grad_MNIST_model1[id,:,:]

    #print('Grad_MNIST_model1_vec_disgnated = ' , Grad_MNIST_model1_vec_disgnated.shape)

    disc_values_disgnated = disc_values[id,:]

    ####### get S_T_comp_star as the reduced/sorted set with cardinality = Q
    # get the indicies of the highest Q values from the f(input image), where f is the discriminant vector before the softmax
    # vector before softmax is:
    disc_values = pickle.load(
        open("/home/user/.PyCharmCE2019.1/config/scratches/saved_models_variables/disc_values_before_SM.p", "rb"))
    disc_values_disgnated = disc_values[id, :]

    # remove S_T values and place them with -100.0
    temp = disc_values[id, :]
    disc_values_disgnated_excluding_S_T = temp
    disc_values_disgnated_excluding_S_T[S_T] = -100.0
    S_T_comp_star = (-disc_values_disgnated_excluding_S_T).argsort()[0:Q]

    # # keep this to restart above variables in the case of using j_star from the NOC methid
    disc_values = pickle.load(
        open("/home/user/.PyCharmCE2019.1/config/scratches/saved_models_variables/disc_values_before_SM.p", "rb"))
    disc_values_disgnated = disc_values[id, :]


    ###### SAVE eta[id] of each j \in S_T_comp
    # initial
    eta_vec_j = np.zeros(shape=(10,28*28,1))
    # distance initial
    D_j       = 1000000*np.ones(shape=(10,      1))



    ####################################### Alg .II

    ## try MIFGSM; if good, then exit the program and we found eta^*

    if MIFGSM_pred_label_w_pert_super_label[id] != tru_sup_lbl:
        eta_cvx = MIFGSM_perturbations[id,:,:,:].reshape(784,1)
        eta_vec[id, :, :] = eta_cvx.reshape(n, 1)
        eta_source = 'MIFGSM'
        cnt = cnt + 1
        rho_2 = Imperceptibility(input_image, eta_cvx)[0]
        rho_inf = Imperceptibility(input_image, eta_cvx)[1]
        D_ssim = Imperceptibility(input_image, eta_cvx)[2]
        imperceptibility_rho_2_save[id] = rho_2
        imperceptibility_rho_i_save[id] = rho_inf
        imperceptibility_sssim_save[id] = D_ssim
        image_pert = eta_cvx + input_image
        #pred_pert_sup_lbls[id] = sup_lbl_from_lbl(np.argmax(model1(image_pert.reshape(1, 784, 1))))
        pred_pert_lbls[id] = MIFGSM_pred_label_w_pert[id]
        pred_pert_sup_lbls[id] = MIFGSM_pred_label_w_pert_super_label[id]
        print('id = ', id, "eta_source = " ,  'MIFGSM' , ' ; winning_label = ', 'Nadaaaaaa',  'pred_sup_lbl = ', pred_sup_lbl, 'predecited_perturbed_super_lbl = ',
              MIFGSM_pred_label_w_pert_super_label[id], ' (rho_2,rho_inf, ssim) = ', Imperceptibility(input_image,eta_cvx)[0:2], ' ; count = ', cnt)



    ## ELSE
    else:
        flag = 0
        eta_source = 'not MIFGSM'
        for jj in S_T_comp_star:
            j_star = jj
            # find eta_jj

            ########
            epsilon = 10

            ####### get matrix G \in N \times |S_T| and b \in |S_T|, where G_columns = [grad_j_star - grad_l], for all l \in S_T
            n = 28*28
            card_S_T = len(S_T) # cardinality of the set S_T

            mat_G = np.zeros(shape=(n,card_S_T)) # init mat_G

            vec_b_wout = np.zeros(shape=(card_S_T,1) )

            temp_jstar = Grad_MNIST_model1_vec_disgnated[j_star , : ,:]
            temp_jstar = temp_jstar.reshape(n,)
            b_jstar    =  disc_values_disgnated[j_star]
            #b_jstar    = b_jstar.reshape(1,)

            for i in range(card_S_T):
                temp1 = Grad_MNIST_model1_vec_disgnated[S_T[i] , : ,:]
                temp1 = temp1.reshape(n,)

                b_l   = disc_values_disgnated[S_T[i]]
            #    b_l   = b_l.reshape(1,)

                mat_G[:,i] = temp_jstar - temp1
                vec_b_wout[  i] = b_l - b_jstar

            vec_b = vec_b_wout + epsilon

            ###############################################################################################
            ##### ADMM
            #### algorithm parameters
            r_penalty_factor = 0.0075
            number_of_iterations_tau = 10

            # eADMM stopping criteria
            epsilon_A = 0.15

            admm_type = "ADMM"

            eta_cvx = ADMM_(input_image,model1,pred_sup_lbl,r_penalty_factor,number_of_iterations_tau,epsilon_A,mat_G, vec_b,admm_type)
            ################################################################################################




        ################# calculate the distance
            image_pert_temp = input_image + eta_cvx
            #D_j[jj] = LA.norm(eta_cvx, 2)
            D_j[jj] = Imperceptibility(input_image,eta_cvx)[0]

            if sup_lbl_from_lbl(np.argmax(model1(image_pert_temp.reshape(1, 784, 1)))) != pred_sup_lbl and D_j[jj] <= epsilon_D:

                #print('break for is used')
                flag = 1
                eta_cvx = eta_cvx
                eta_vec[id, :, :] = eta_cvx.reshape(n, 1)
                cnt = cnt + 1
                rho_2 = Imperceptibility(input_image, eta_cvx)[0]
                rho_inf = Imperceptibility(input_image, eta_cvx)[1]
                D_ssim = Imperceptibility(input_image, eta_cvx)[2]
                imperceptibility_rho_2_save[id] = rho_2
                imperceptibility_rho_i_save[id] = rho_inf
                imperceptibility_sssim_save[id] = D_ssim
                image_pert = eta_cvx + input_image
                pred_pert_lbls[id] = np.argmax(model1(image_pert.reshape(1, 784, 1)))
                pred_pert_sup_lbls[id] = sup_lbl_from_lbl(np.argmax(model1(image_pert.reshape(1, 784, 1))))
                print('id = ', id, "eta_source = ", 'not MIFGSM and break is used', ' ; winning_label = ', jj, 'pred_sup_lbl = ',
                      pred_sup_lbl, 'predecited_perturbed_super_lbl = ',
                      pred_pert_sup_lbls[id], ' (rho_2,rho_inf, ssim) = ', Imperceptibility(input_image, eta_cvx)[0:2],
                      ' ; count = ', cnt)
                break


            else:
                # save the  mother fucking eta_cvx to choose from in the future
                # save eta for each j \in S_T_comp
                eta_vec_j[jj,:,:] = eta_cvx.reshape(n,1)


        if flag != 1:
            winning_label = np.argmin(D_j)
            eta_cvx = eta_vec_j[winning_label, :, :]
            eta_cvx = eta_cvx
            rho_2 = Imperceptibility(input_image, eta_cvx)[0]
            rho_inf = Imperceptibility(input_image, eta_cvx)[1]
            D_ssim = Imperceptibility(input_image, eta_cvx)[2]


            # cnt is increased iff T(k(x+eta)) != T(k(x))
            if sup_lbl_from_lbl(np.argmax(model1((input_image+eta_cvx).reshape(1, 784, 1)))) != pred_sup_lbl:
                cnt = cnt + 1
                imperceptibility_rho_2_save[id] = rho_2
                imperceptibility_rho_i_save[id] = rho_inf
                imperceptibility_sssim_save[id] = D_ssim


            image_pert = eta_cvx + input_image
            pred_pert_lbls[id] = np.argmax(model1(image_pert.reshape(1, 784, 1)))
            pred_pert_sup_lbls[id] = sup_lbl_from_lbl(np.argmax(model1(image_pert.reshape(1, 784, 1))))
            print('id = ', id, "eta_source = ", 'not MIFGSM and no break', ' ; winning_label = ', winning_label,
                  'pred_sup_lbl = ',
                  pred_sup_lbl, 'predecited_perturbed_super_lbl = ',
                  pred_pert_sup_lbls[id], ' (rho_2,rho_inf, ssim) = ', Imperceptibility(input_image, eta_cvx)[0:2],
                  ' ; count = ', cnt)





attack_success = cnt / number_of_observations

print('ATTACK SUCCESS = ' , attack_success*100 , '%')


CA_pert, CA_pert_sup,  RLA, ELA,RLA_sup, ELA_sup , sigma_2, sigma_inf, sigma_s = \
    Attack_performance(test_labels[0:number_of_observations] ,
                       pred_lbls,
                       pred_pert_lbls ,
                       imperceptibility_rho_2_save,
                       imperceptibility_rho_i_save,
                       imperceptibility_sssim_save)

# attack performace
print('Number of observations = ', number_of_observations ,
    '\n CA_pert = ' , CA_pert,
      "\n CA_pert_sup = " , CA_pert_sup ,
      "\n RLA = " , RLA ,
      "\n ELA = " , ELA,
      '\n RLA_sup = ' , RLA_sup,
      '\n ELA_sup = ' , ELA_sup,
      "\n sigma_2 = " , sigma_2 ,
      "\n sigma_inf = " , sigma_inf ,
      '\n ssim = ' , sigma_s)





# # #####################################################################
# # ################### Plotting images
# # #####################################################################
# print("")
#
# plt.figure()
# plt.subplot(1,3,1)
# plt.title('Original')
# plt.imshow(input_image.reshape(28,28))
# plt.axis('off')
#
#
# plt.subplot(1,3,2)
# plt.title('pertubations')
# plt.imshow(eta_cvx.reshape(28,28))
# plt.axis('off')
#
#
# plt.subplot(1,3,3)
# plt.title('perturbed image')
# plt.imshow(image_pert.reshape(28,28))
# plt.axis('off')
#
#
# plt.show()
# # ########################################################################


stop = timeit.default_timer()

print('Time: ', stop - start)

#pickle.dump(eta_vec, open("eta_vec_alg2_samples.p", "wb"))


print('break here')




