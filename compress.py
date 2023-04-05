import h5py
from tensorflow import keras
import gzip

model = keras.models.load_model('combinedData(5)-modelv5-55epoch-1010steps-softmax-imgnet-trainableTrue-topless-avgpooling.h5')

with open('combinedData(5)-modelv5-55epoch-1010steps-softmax-imgnet-trainableTrue-topless-avgpooling.h5', 'rb') as f_in:
    with gzip.open('combinedData(5)-modelv5-55epoch-1010steps-softmax-imgnet-trainableTrue-topless-avgpooling.h5.gz', 'wb') as f_out:
        f_out.writelines(f_in)



# with gzip.open('model.h5.gz', 'rb') as f_in:
#     with open('model.h5', 'wb') as f_out:
#         f_out.writelines(f_in)

# with h5py.File('model.h5', 'r') as f:
#     model = load_model(f)