import h5py
import lzma
from tensorflow import keras
import gzip

model = keras.models.load_model('combinedData(5)-modelv5-55epoch-1010steps-softmax-imgnet-trainableTrue-topless-avgpooling-optimizerless.h5')

    
    
with h5py.File('combinedData(5)-modelv5-55epoch-1010steps-softmax-imgnet-trainableTrue-topless-avgpooling-optimizerless.h5', 'r') as f:
    # Get the model object from the h5 file
    model_weights = f["model_weights"][()]

# Compress the model using LZMA
compressed_data = lzma.compress(model_weights)

# Write the compressed data to a file
with open('model.h5.xz', 'wb') as f:
    f.write(compressed_data)


# with open('combinedData(5)-modelv5-55epoch-1010steps-softmax-imgnet-trainableTrue-topless-avgpooling.h5', 'rb') as f_in:
#     with gzip.open('combinedData(5)-modelv5-55epoch-1010steps-softmax-imgnet-trainableTrue-topless-avgpooling-lv9.h5.gz', 'wb', gzip_compression_level=9) as f_out:
#         f_out.writelines(f_in)



# with gzip.open('model.h5.gz', 'rb') as f_in:
#     with open('model.h5', 'wb') as f_out:
#         f_out.writelines(f_in)

# with h5py.File('model.h5', 'r') as f:
#     model = load_model(f)