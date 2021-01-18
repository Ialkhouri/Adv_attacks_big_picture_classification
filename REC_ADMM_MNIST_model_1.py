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



##### get test_labels from y_test, since y_test is the one hot encoder



######################################################################
#### Initiating runtime for foolbox - not required here#
#########################################################################
#fmodel = foolbox.models.TensorFlowEagerModel(model1, bounds=(0, 1))
#####################################################################






#####################################################################
################### loading Grads and testing the vectorization
#####################################################################


# this is of size shape=(10000,10,28,28,1)

#### load different here

Grad_MNIST_model1      = pickle.load(open("/home/user/.PyCharmCE2019.1/config/scratches/saved_models_variables/Grad_MNIST_model1_1d_before_SM.p","rb"))


# this is of size shape=(10000,10,28*28,1)
#Grad_MNIST_model1_vec = Grad_MNIST_model1.reshape(10000, 10, 28*28 , 1)

# this is of size shape=(10000,10)
disc_values            = pickle.load(open("/home/user/.PyCharmCE2019.1/config/scratches/saved_models_variables/disc_values_before_SM.p","rb"))


## testing the vectorization of the grad: PASS
# grad_test = Grad_MNIST_model1[0,9,:,:,:]
#
# vec_grad_test  = grad_test.reshape(28*28 , 1)
#
# mat_vec_grad_test = vec_grad_test.reshape(28,28,1)





################################################################################
##################################### BUILDING THE NOC PROBLEM WITH CVXPY ######
################################################################################

######## to save eta, ceate a vectorized empty np array of size 10000,28*28,1
number_of_observations = 1000

### tensors to save and to calc CApert, CApert_sup, ELA, RLA, and sigmas
eta_vec                     = np.zeros(shape=(number_of_observations,28*28,1))
imperceptibility_rho_2_save = np.nan*np.ones(shape=(number_of_observations,1))
imperceptibility_rho_i_save = np.nan*np.ones(shape=(number_of_observations,1))
imperceptibility_sssim_save = np.nan*np.ones(shape=(number_of_observations,1))
pred_pert_lbls              = np.zeros(shape=(number_of_observations))
pred_pert_sup_lbls          = np.zeros(shape=(number_of_observations))
pred_lbls                   = np.zeros(shape=(number_of_observations))

cnt = 0

S_T_comp_star_save=[]

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


    #### let Q =
    Q = 3



    ####### get S_T_comp_star as the reduced/sorted set with cardinality = Q
    # get the indicies of the highest Q values from the f(input image), where f is the discriminant vector before the softmax
    # vector before softmax is:
    disc_values            = pickle.load(open("/home/user/.PyCharmCE2019.1/config/scratches/saved_models_variables/disc_values_before_SM.p","rb"))
    disc_values_disgnated = disc_values[id,:]

    # remove S_T values and place them with -100.0
    temp = disc_values[id, :]
    disc_values_disgnated_excluding_S_T = temp
    disc_values_disgnated_excluding_S_T[S_T] = -100.0
    S_T_comp_star = (-disc_values_disgnated_excluding_S_T).argsort()[0:Q]

    # # keep this to restart above variables in the case of using j_star from the NOC methid
    disc_values            = pickle.load(open("/home/user/.PyCharmCE2019.1/config/scratches/saved_models_variables/disc_values_before_SM.p","rb"))
    disc_values_disgnated = disc_values[id,:]

    ########
    epsilon = 10



    ####### get matrix G \in N \times |S_T| and b \in |S_T|, where G_columns = [grad_j_star - grad_l], for all l \in S_T
    n = 28*28
    card_S_T = len(S_T) # cardinality of the set S_T

    mat_G = np.zeros(shape=(n,card_S_T*Q)) # init mat_G

    vec_b_wout = np.zeros(shape=(card_S_T*Q,1) )


    # get the combination vector of S_t and S_T_comp_star
    combo = np.transpose([np.tile(S_T, len(S_T_comp_star)), np.repeat(S_T_comp_star, len(S_T))])

    #temp_jstar = Grad_MNIST_model1_vec_disgnated[j_star , : ,:]
    #temp_jstar = temp_jstar.reshape(n,)
    #b_jstar    =  disc_values_disgnated[j_star]
    #b_jstar    = b_jstar.reshape(1,)

    for v in range(len(combo)):
        temp_l = Grad_MNIST_model1_vec_disgnated[combo[v][0] , : ,:].reshape(n,)
        temp_j = Grad_MNIST_model1_vec_disgnated[combo[v][1] , : ,:].reshape(n,)

        mat_G[:, v] = temp_j - temp_l

        b_l   = disc_values_disgnated[combo[v][0]]
        b_j   = disc_values_disgnated[combo[v][1]]
    #    b_l   = b_l.reshape(1,)
        vec_b_wout[  v] = b_l - b_j
    #print('mat_G = ' , mat_G.shape)


    vec_b = vec_b_wout + epsilon

    ###############################################################################################
    ##### ADMM
    #### algorithm parameters
    r_penalty_factor         = 0.0075
    number_of_iterations_tau = 10

    # eADMM stopping criteria
    epsilon_A = 0.15

    admm_type = "ADMM"

    eta_cvx = ADMM_(input_image,model1,pred_sup_lbl,r_penalty_factor, number_of_iterations_tau, epsilon_A, mat_G, vec_b, admm_type)
    ################################################################################################




    ########## save, then reshape eta to be a matrix:

    eta_vec[id,:,:] = eta_cvx.reshape(n,1)
    eta_cvx = eta_cvx.reshape(784,1)

    #### add to image to  check the pret pred lbl

    # pert image

    image_pert = eta_cvx + input_image

    # pred_pert_lbl
    pred_pert_lbl = np.argmax(model1(image_pert.reshape(1, 784, 1)))
    pred_pert_lbls[id] = pred_pert_lbl
    # pred_pert_sup_lbl
    pred_pert_sup_lbl = sup_lbl_from_lbl(pred_pert_lbl)
    pred_pert_sup_lbls[id] = pred_pert_sup_lbl

    # calculate the imperceptibility:
    #rho_2 = LA.norm(eta_cvx.reshape(784)) / LA.norm(input_image.reshape(784))
    rho_2   = Imperceptibility(input_image,eta_cvx)[0]
    rho_inf = Imperceptibility(input_image, eta_cvx)[1]
    D_ssim  = Imperceptibility(input_image,eta_cvx)[2]


    imperceptibility_rho_2_save[id] = rho_2

    #if pred_sup_lbl != pred_pert_sup_lbl and rho_2 >= 0.000001 and rho_2 <= 0.35:
    if pred_sup_lbl != pred_pert_sup_lbl:
        cnt = cnt+1
        imperceptibility_rho_2_save[id] = rho_2
        imperceptibility_rho_i_save[id] = rho_inf
        imperceptibility_sssim_save[id] = D_ssim

    ##### logger:
    S_T_comp_star_save.append(S_T_comp_star)

    print('id = ', id, ' ; S_T_comp_star = ' ,S_T_comp_star ,  ' ; tru sup lbl = '  , tru_sup_lbl , ' ; pred_sup_lbl = ', pred_sup_lbl, ' ; predecited_perturbed_super_lbl = ',
          pred_pert_sup_lbl, ' imperceptibility = ', rho_2, 'count = ', cnt)
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


#pickle.dump(S_T_comp_star_save, open("S_T_comp_star_save.p", "wb"))

# # #####################################################################
# # ################### Plotting images
# # #####################################################################
# print("")
#
# iidd = 9
#
# plt.figure()
# plt.subplot(1,3,1)
# plt.title('Original')
# plt.imshow(X_test_1d[iidd].reshape(28,28))
# plt.axis('off')
#
#
# plt.subplot(1,3,2)
# plt.title('pertubations')
# plt.imshow(eta_vec[iidd,:,:].reshape(28,28))
# plt.axis('off')
#
#
# plt.subplot(1,3,3)
# plt.title('perturbed image')
# perturbed_image = X_test_1d[iidd] + eta_vec[iidd,:,:]
# plt.imshow(perturbed_image.reshape(28,28))
# plt.axis('off')
#
#
# plt.show()
# # ########################################################################

stop = timeit.default_timer()

print('Time: ', stop - start)



print('break here')





