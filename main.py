import tensorflow as tf
from train import *
from get_data import *
from Config import *

EPOCHS = train_config.EPOCHS

import matplotlib.pyplot as plt
from keras.utils import Progbar


for epoch in range(EPOCHS):

  tf.print("{}/{}".format(epoch+1, EPOCHS))
  pbar = Progbar(target=60000, unit_name="CGAN")

  for x,y in train_images_dataset:
    dis_loss, gen_loss = train_step(x,y)
    values=[("Critic Loss", np.round(dis_loss.numpy(),4)), ("Generator Loss", np.round(gen_loss.numpy(),4))]
    pbar.add(x.shape[0],values=values)
  if epoch % 5 == 0:
    generate_and_save_images(cgan.generator,epoch + 1)