import sys

import os
import os.path as op
import time

import click
import PIL
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from PIL import Image
sys.path.insert(0,'/home/nagarajudasari/automap/fastmri-reproducible-benchmark/') 

from fastmri_recon.config import *
from fastmri_recon.data.datasets.fastmri_pyfunc import train_masked_kspace_dataset_from_indexable
from fastmri_recon.data.datasets.fastmri_pyfunc_non_cartesian import train_nc_kspace_dataset_from_indexable
from fastmri_recon.models.subclassed_models.pdnet import PDNet
from fastmri_recon.models.subclassed_models.ncpdnet import NCPDNet
from fastmri_recon.models.functional_models.pdnet import pdnet
from fastmri_recon.models.training.compile import default_model_compile


gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
session = tf.compat.v1.InteractiveSession(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

# paths
train_path = f'{FASTMRI_DATA_DIR}singlecoil_train/'
val_path = f'{FASTMRI_DATA_DIR}singlecoil_val/'
test_path = f'{FASTMRI_DATA_DIR}singlecoil_test/'

n_volumes_train = 973

@click.command()
@click.option(
    'subclassed',
    '-s',
    is_flag=True,
    default=True,
    help='Flag to use the subclassed API.',
)
@click.option(
    'af',
    '-a',
    default='4',
    type=click.Choice(['4', '8']),
    help='The acceleration factor chosen for this fine tuning. Defaults to 4.',
)
@click.option(
    'contrast',
    '-c',
    default=None,
    type=click.Choice(['CORPDFS_FBK', 'CORPD_FBK', None], case_sensitive=False),
    help='The contrast chosen for this fine-tuning. Defaults to None.',
)
@click.option(
    'cuda_visible_devices',
    '-gpus',
    '--cuda-visible-devices',
    default='0123',
    type=str,
    help='The visible GPU devices. Defaults to 0123',
)
@click.option(
    'n_samples',
    '-ns',
    default=350,
    type=int,
    help='The number of samples to use for this training. Default to None, which means all samples are used.',
)
@click.option(
    'n_epochs',
    '-e',
    #default=300,
    default=5,
    type=int,
    help='The number of epochs to train the model. Default to 300.',
)
@click.option(
    'n_iter',
    '-i',
    default=5,
    type=int,
    help='The number of epochs to train the model. Default to 300.',
)
def train_pdnet(subclassed, af, contrast, cuda_visible_devices, n_samples, n_epochs, n_iter):
    print("Subclassed :"+ str(subclassed))
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(cuda_visible_devices)
    af = int(af)
    # generators
    train_set = train_masked_kspace_dataset_from_indexable(
        train_path,
        AF=af,
        contrast=contrast,
        inner_slices=8,
        rand=True,
        scale_factor=1e6,
        n_samples=n_samples,
    )

    train_set_nc = train_nc_kspace_dataset_from_indexable(
        path=train_path,
        image_size=(640,320),
        inner_slices=8,
        rand=True,
        scale_factor=1e6,
        contrast=contrast,
        n_samples=n_samples,
        acq_type = 'radial',
        af=af
    )
    
    a = next(iter(train_set_nc))
    #npa = np.array(a[0][0])
    #print(np.shape(a))
    print(a[1])
    #print("------------------------")
    print(a[0][0])
    #print("------------------------")
    print(a[0][1])
    #print(a[0].shape, a[1].shape)
    # result = Image.fromarray(np.abs(np.fft.ifft2(a[0][0][0,:,:,0])))
    # result.save("figure1.bmp")
    plt.imsave('filename_ifft_nc.jpeg', np.abs(np.fft.ifftshift(np.fft.ifft2(a[0][0][0,:,:,0]))), cmap='gray')
    #plt.imsave('filename_fft.jpeg', np.log10(np.abs((a[0][0][0,:,:,0]))), cmap='gray')
    #plt.imsave('filename_gr.jpeg', a[1][0,:,:,0], cmap='gray')
    #result2 = Image.fromarray(np.log(np.abs(a[0][0][0,:,:,0])))
    #result2.save("figure2.png")
    #plt.savefig('filename1.png', np.log(np.abs(a[0][0][0,:,:,0])))

    val_set = train_masked_kspace_dataset_from_indexable(
        val_path,
        AF=af,
        contrast=contrast,
        scale_factor=1e6,
    )

    val_set_nc = train_nc_kspace_dataset_from_indexable(
        path=val_path,
        image_size=(640,320),
        scale_factor=1e6,
        contrast=contrast,
        acq_type = 'radial',
        af=af
    )

    run_params = {
        'n_primal': 2,
        #'n_dual': 2,
        'n_iter': n_iter,
        'n_filters': 32,
    }
    additional_info = f'af{af}'
    if contrast is not None:
        additional_info += f'_{contrast}'
    if n_samples is not None:
        additional_info += f'_{n_samples}'
    if n_iter != 10:
        additional_info += f'_i{n_iter}'

    run_id = f'pdnet_{additional_info}_{int(time.time())}'
    chkpt_path = f'{CHECKPOINTS_DIR}checkpoints/{run_id}' + '-{epoch:02d}.hdf5'

    chkpt_cback = ModelCheckpoint(chkpt_path, period=n_epochs, save_weights_only=True)
    log_dir = op.join(f'{LOGS_DIR}logs', run_id)
    tboard_cback = TensorBoard(
        profile_batch=0,
        log_dir=log_dir,
        histogram_freq=0,
        write_graph=False,
        write_images=False,
    )
    if subclassed:
        print("inside if!#######")
        model = NCPDNet(**run_params)
        #model = PDNet(**run_params)
        default_model_compile(model, lr=1e-3)
    else:
        print("In else!########")
        def adapt_dataset_tensors_to_functional_model(model_input, model_output):
            mask = model_input[1]
            new_mask = tf.tile(mask, [1, 640, 1])
            return (model_input[0], new_mask), model_output
        train_set_nc = train_set_nc.map(adapt_dataset_tensors_to_functional_model, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        val_set_nc = val_set_nc.map(adapt_dataset_tensors_to_functional_model, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        #train_set = train_set.map(adapt_dataset_tensors_to_functional_model, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        #val_set = val_set.map(adapt_dataset_tensors_to_functional_model, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        model = pdnet(lr=1e-3, **run_params)
    print(run_id)
    model.fit(
        train_set_nc,
        #train_set,
        steps_per_epoch=n_volumes_train,
        epochs=n_epochs,
        validation_data=val_set_nc,
        #validation_data=val_set,
        validation_steps=5,
        verbose=1,
        callbacks=[tboard_cback, chkpt_cback,],
    )

if __name__ == '__main__':
    train_pdnet()
