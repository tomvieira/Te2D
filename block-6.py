import os
import cv2
import argparse
import numpy as np
import data_handler as dh

parser = argparse.ArgumentParser(
    description='Convert genomic data to images.')
parser.add_argument('-din', type=str, help='Input directory')
parser.add_argument('-dout', type=str, help='Output directory')
parser.add_argument('-size', type=str, help='Image size', default='224')
args = parser.parse_args()


def reshape_to_square(image):

    h, w = image.shape[:2]
    if h != 6:
        raise ValueError("The height of the image must be 6 pixels")

    block_size = 6  # Blocks 6x6
    num_blocks = w // block_size  # Number of complete 6x6 blocks

    # Create the list of blocks
    blocks = [image[:, i * block_size:(i + 1) * block_size]
              for i in range(num_blocks)]

    # Check for remainder and pad with zeros
    if w % block_size != 0:
        remainder = w % block_size
        pad_block = np.zeros((6, block_size), dtype=image.dtype)
        # Copy the remainder to the new zeroed matrix
        pad_block[:, :remainder] = image[:, -remainder:]
        blocks.append(pad_block)

    # Calculate the side of the new square image
    total_blocks = len(blocks)
    square_size = int(np.ceil(np.sqrt(total_blocks)))

    # Fill with empty blocks if necessary
    while len(blocks) < square_size**2:
        blocks.append(np.zeros((6, 6), dtype=image.dtype))

    # Stack the blocks to form the square image
    rows = [np.hstack(blocks[i * square_size:(i + 1) * square_size])
            for i in range(square_size)]
    square_image = np.vstack(rows)

    return square_image


def get_files(root):
    train_files = [root+'/Train/' +
                   train_file for train_file in os.listdir(root+'/Train')]
    test_files = [root+'/Test/' +
                  test_file for test_file in os.listdir(root+'/Test')]
    train_files.sort()
    test_files.sort()

    return train_files, test_files


def create_directories(numClasses, base_directory):

    for i in range(0, numClasses):
        folder_name = f"class_{i}"
        caminho_diretorio = os.path.join(base_directory, folder_name)

        if not os.path.exists(caminho_diretorio):
            os.makedirs(caminho_diretorio, exist_ok=True)
            print(f"Folder '{folder_name}' created in {base_directory}")
        else:
            print(
                f"Folder '{folder_name}' already exists in {base_directory}")


def vector_to_matrix(vector, n_rows, n_cols):
    matrix = np.zeros((n_rows, n_cols), dtype=int)
    matrix[vector, np.arange(n_cols)] = 1
    return matrix


def expand_greyscale_image_channels(grey_image):
    grey_image_arr = np.expand_dims(grey_image, -1)
    grey_image_arr_3_channel = grey_image_arr.repeat(3, axis=-1)
    return grey_image_arr_3_channel


def convert_to_img(data, y, dir_out, folder, size_img):
    img_num = 0
    cl_old = y[0]
    create_directories(len(np.unique(y)), dir_out+"/"+folder)

    for i, vetor in enumerate(data):
        matriz_resultante = vector_to_matrix(vetor, db.vocab_size, db.max_len)
        binary_image_gray = np.uint8(matriz_resultante * 255)
        binary_image_gray = remove_unnecessary_padding(binary_image_gray)
        # determine the size of the image
        n = binary_image_gray.flatten().shape[0] // 6
        # thresholding
        threshold = ((size_img) // 6) * size_img
        if n <= threshold:
            resized_image = reshape_directly(binary_image_gray, size_img)
        else:
            resized_image = reshape_to_square(binary_image_gray)
        resized_image = expand_greyscale_image_channels(resized_image)
        if (resized_image.shape != (size_img, size_img, 3)):
            resized_image = cv2.resize(
                resized_image, (size_img, size_img), interpolation=cv2.INTER_NEAREST)

        if y[i] != cl_old:
            img_num = 1
        else:
            img_num += 1
        cv2.imwrite(dir_out+"/"+folder+'/class_'+str(y[i])+'/image_'+str(
            img_num)+'.png', resized_image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        cl_old = y[i]


def vector_to_square_matrix(vector):
    vector = np.array(vector)
    length = len(vector)

    # Find the smallest n such that n*n >= length
    n = int(np.ceil(np.sqrt(length)))

    # Calculate the necessary padding
    padding_size = n**2 - length

    # Add zeros to the vector
    padded_vector = np.pad(vector, (0, padding_size), mode='constant')

    # Transform into square matrix
    square_matrix = padded_vector.reshape((n, n))

    return square_matrix


def remove_unnecessary_padding(matrix):
    # Create a mask to identify which columns contain at least one "1"
    col_mask = np.any(matrix[1:] == 255, axis=0)

    # Find the last column that contains a "1"
    last_col_with_one = np.max(np.where(col_mask)) if np.any(col_mask) else -1

    # Keep all columns up to the last one that contains a "1"
    return matrix[:, :last_col_with_one + 1] if last_col_with_one != -1 else np.array([[]])


def flatten_and_expand(matrix, img_size):
    # Get the number of elements in the matrix
    current_size = matrix.size
    target_size = img_size * img_size

    # If the matrix is smaller than or equal to target_size, flatten it
    if current_size <= target_size:
        # Flatten the matrix
        flattened = matrix.flatten()

        # Calculate the number of zeros needed to reach target_size elements
        zeros_needed = target_size - current_size

        # Add zeros, if necessary
        expanded = np.pad(flattened, (0, zeros_needed),
                          mode='constant', constant_values=0)

        # Convert back to img_size x img_size shape
        return expanded.reshape((6, target_size // 6))

    # If the matrix is larger than target_size, return it exactly as it came
    return matrix


def reshape_directly(image, img_size):
    h, w = image.shape[:2]
    if h != 6:
        raise ValueError("The height of the image must be 6 pixels")

    block_width = img_size  # Fixed block width
    num_blocks = w // block_width  # Number of complete 6 x img_size blocks

    # Create the list of blocks
    blocks = [image[:, i * block_width:(i + 1) * block_width]
              for i in range(num_blocks)]

    # Check for remainder and fill with zeros
    if w % block_width != 0:
        remainder = w % block_width
        pad_block = np.zeros((6, block_width), dtype=image.dtype)
        # Copy the remainder to the new zeroed matrix
        pad_block[:, :remainder] = image[:, -remainder:]
        blocks.append(pad_block)

    # Calculate the exact height needed without extrapolating
    total_blocks = len(blocks)
    stacked_height = total_blocks * 6
    # If the height exceeds img_size, cut off the excess
    if stacked_height > img_size:
        excess_blocks = (stacked_height - img_size) // 6
        blocks = blocks[:-excess_blocks]
        stacked_height = len(blocks) * 6

    # Adjust the last block if necessary to exactly img_size height
    if stacked_height < img_size:
        last_block_height = img_size - stacked_height
        blocks.append(
            np.zeros((last_block_height, block_width), dtype=image.dtype))

    # Stack the blocks to form the final image
    square_image = np.vstack(blocks)

    return square_image


dir_ = args.din
dir_out = args.dout
size_img = int(args.size)

if not os.path.exists(dir_out):
    os.makedirs(dir_out)

train_files, test_files = get_files(dir_)
db = dh.DataHandler(train_files, test_files)
print("Processing Test data...")
convert_to_img(db.x_test, db.y_test, dir_out, 'Test', size_img)
print("Processing Train data...")
convert_to_img(db.x_train, db.y_train, dir_out, 'Train', size_img)
print("Finished!")
