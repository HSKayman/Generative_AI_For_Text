# Written by: Hasan Suca Kayman
# City College of New York, CUNY
# April 2024
# chatbot_RNN_runner

import os
import warnings
import time
#warnings.filterwarnings('ignore')

for first_n_elements in [400,600]:
    for batch_size in [16,64]:
        for number_of_hidden_layer in [256,512]:
            for number_of_epochs in [100,300]:
                start_time = time.time()
                os.system("python chat_bot_RNN.py {} {} {} {} ".format(first_n_elements,batch_size,number_of_hidden_layer,number_of_epochs))
                end_time = time.time()
                print("first_n_elements:{}\nbatch_size: {}\nnumber_of_hidden_layer: {}\n number_of_epochs:{} is completed.".format(first_n_elements,batch_size,number_of_hidden_layer,number_of_epochs))
                duration = end_time - start_time  # Duration in seconds
                print("Duration:{:.4f} seconds\n".format(duration))