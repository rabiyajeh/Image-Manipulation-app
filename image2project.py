import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt

# Function to apply filters to the image
def apply_filter(image, filter_name, filter_params):
    if filter_name == 'Grayscale':
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif filter_name == 'Blur':
        kernel_size = filter_params.get('kernel_size', 5)
        return cv2.blur(image, (kernel_size, kernel_size))
    elif filter_name == 'Edge Detection':
        threshold1 = filter_params.get('threshold1', 100)
        threshold2 = filter_params.get('threshold2', 200)
        return cv2.Canny(image, threshold1, threshold2)
    elif filter_name == 'Invert':
        return cv2.bitwise_not(image)
    elif filter_name == 'Brightness':
        brightness = filter_params.get('brightness', 0)
        return cv2.convertScaleAbs(image, alpha=1, beta=brightness)
    elif filter_name == 'Contrast':
        contrast = filter_params.get('contrast', 1.0)
        return cv2.convertScaleAbs(image, alpha=contrast, beta=0)
    elif filter_name == 'Saturation':
        saturation = filter_params.get('saturation', 1.0)
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv_image[..., 1] = np.clip(hsv_image[..., 1] * saturation, 0, 255)
        return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    elif filter_name == 'Sharpen':
        kernel_sharpening = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        return cv2.filter2D(image, -1, kernel_sharpening)
    elif filter_name == 'Emboss':
        kernel_emboss = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
        return cv2.filter2D(image, -1, kernel_emboss)
    elif filter_name == 'Histogram Equalization':
        if len(image.shape) < 3:
            return cv2.equalizeHist(image)
        else:
            hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hsv_image[..., 2] = cv2.equalizeHist(hsv_image[..., 2])
            return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    elif filter_name == 'Blend':
        alpha = filter_params.get('alpha', 0.5)
        blend_image = filter_params.get('blend_image', None)
        if blend_image is not None:
            blend_image = cv2.resize(blend_image, (image.shape[1], image.shape[0]))
            return cv2.addWeighted(image, alpha, blend_image, 1 - alpha, 0)
    elif filter_name == 'Channel Manipulation':
        channel = filter_params.get('channel', 'R')
        if channel == 'R':
            image[:, :, 1] = 0
            image[:, :, 2] = 0
        elif channel == 'G':
            image[:, :, 0] = 0
            image[:, :, 2] = 0
        elif channel == 'B':
            image[:, :, 0] = 0
            image[:, :, 1] = 0
        return image
    elif filter_name == 'Segmentation':
        if len(image.shape) < 3:
            st.warning("Segmentation requires a color image.")
            return image
        else:
            return segment_image(image)
    elif filter_name == 'Color Space Conversion':
        return convert_color_space(image, filter_params.get('conversion', 'RGB'))
    elif filter_name == 'Thresholding':
        threshold_value = filter_params.get('threshold_value', 128)
        max_value = filter_params.get('max_value', 255)
        return cv2.threshold(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), threshold_value, max_value, cv2.THRESH_BINARY)[1]
    else:
        return image

# Function to segment the image using K-means clustering
def segment_image(image):
    flattened_image = image.reshape((-1, 3))
    flattened_image = np.float32(flattened_image)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    k = 8  # Number of clusters
    _, labels, centers = cv2.kmeans(flattened_image, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(image.shape)
    return segmented_image

# Function to convert color space of the image
def convert_color_space(image, conversion):
    if conversion == 'RGB':
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif conversion == 'HSV':
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    elif conversion == 'LAB':
        return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    else:
        return image

def main():
    st.title("Advanced Image Filters App")
    st.write("Upload one or more images and apply multiple filters!")

    uploaded_files = st.file_uploader("Upload Images", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_bytes = uploaded_file.read()
            image = np.array(Image.open(io.BytesIO(file_bytes)))
            st.image(image, caption=f'Uploaded Image ({uploaded_file.name})', use_column_width=True)

            # Image transformation options
            rotate_angle = st.slider('Rotate Image (degrees)', -180, 180, 0, key=f'{uploaded_file.name}_rotate')
            flip_horizontal = st.checkbox('Flip Horizontal', key=f'{uploaded_file.name}_flip_horizontal')
            flip_vertical = st.checkbox('Flip Vertical', key=f'{uploaded_file.name}_flip_vertical')
            crop_option = st.checkbox('Crop Image', key=f'{uploaded_file.name}_crop')
            if crop_option:
                crop_x = st.slider('Crop X (left)', 0, image.shape[1], 0, key=f'{uploaded_file.name}_crop_x')
                crop_y = st.slider('Crop Y (top)', 0, image.shape[0], 0, key=f'{uploaded_file.name}_crop_y')
                crop_width = st.slider('Crop Width', 1, image.shape[1], image.shape[1], key=f'{uploaded_file.name}_crop_width')
                crop_height = st.slider('Crop Height', 1, image.shape[0], image.shape[0], key=f'{uploaded_file.name}_crop_height')

            resize_option = st.checkbox('Resize Image', key=f'{uploaded_file.name}_resize')
            if resize_option:
                new_width = st.number_input('New Width', min_value=1, max_value=10000, value=image.shape[1], key=f'{uploaded_file.name}_new_width')
                new_height = st.number_input('New Height', min_value=1, max_value=10000, value=image.shape[0], key=f'{uploaded_file.name}_new_height')

            filter_sequence = st.multiselect('Select filters (hold Shift/Ctrl for multiple)', ['Grayscale', 'Blur', 'Edge Detection', 'Invert', 'Brightness', 'Contrast', 'Saturation', 'Sharpen', 'Emboss', 'Histogram Equalization', 'Blend', 'Channel Manipulation', 'Segmentation', 'Color Space Conversion', 'Thresholding'])

            filtered_image = image.copy()
            for filter_name in filter_sequence:
                if filter_name != 'Original':
                    if filter_name == 'Brightness':
                        brightness = st.slider('Brightness', -100, 100, 0, key=f'{uploaded_file.name}_brightness')
                        filter_params = {'brightness': brightness}
                    elif filter_name == 'Contrast':
                        contrast = st.slider('Contrast', 0.1, 5.0, 1.0, key=f'{uploaded_file.name}_contrast')
                        filter_params = {'contrast': contrast}
                    elif filter_name == 'Saturation':
                        saturation = st.slider('Saturation', 0.1, 5.0, 1.0, key=f'{uploaded_file.name}_saturation')
                        filter_params = {'saturation': saturation}
                    elif filter_name == 'Blur':
                        kernel_size = st.slider('Kernel Size', 1, 11, 5, key=f'{uploaded_file.name}_blur')
                        filter_params = {'kernel_size': kernel_size}
                    elif filter_name == 'Edge Detection':
                        threshold1 = st.slider('Threshold1', 0, 255, 100, key=f'{uploaded_file.name}_threshold1')
                        threshold2 = st.slider('Threshold2', 0, 255, 200, key=f'{uploaded_file.name}_threshold2')
                        filter_params = {'threshold1': threshold1, 'threshold2': threshold2}
                    elif filter_name == 'Blend':
                        blend_image = st.file_uploader("Upload Blend Image", type=['png', 'jpg', 'jpeg'], key=f'{uploaded_file.name}_blend_image')
                        if blend_image:
                            blend_image = np.array(Image.open(io.BytesIO(blend_image.read())))
                            alpha = st.slider('Blend Alpha', 0.0, 1.0, 0.5, key=f'{uploaded_file.name}_blend_alpha')
                            filter_params = {'alpha': alpha, 'blend_image': blend_image}
                        else:
                            filter_params = None
                    elif filter_name == 'Channel Manipulation':
                        channel = st.selectbox('Select Channel', ['R', 'G', 'B'], key=f'{uploaded_file.name}_channel')
                        filter_params = {'channel': channel}
                    elif filter_name == 'Color Space Conversion':
                        conversion = st.selectbox('Select Color Space', ['RGB', 'HSV', 'LAB'], key=f'{uploaded_file.name}_conversion')
                        filter_params = {'conversion': conversion}
                    elif filter_name == 'Thresholding':
                        threshold_value = st.slider('Threshold Value', 0, 255, 128, key=f'{uploaded_file.name}_threshold_value')
                        max_value = st.slider('Max Value', 0, 255, 255, key=f'{uploaded_file.name}_max_value')
                        filter_params = {'threshold_value': threshold_value, 'max_value': max_value}
                    else:
                        filter_params = {}

                    filtered_image = apply_filter(filtered_image, filter_name, filter_params)

            st.image(filtered_image, caption=f'Filtered Image ({uploaded_file.name})', use_column_width=True)

            # Option to plot image histogram
            if st.button('Show Image Histogram', key=f'{uploaded_file.name}_histogram'):
                plot_histogram(filtered_image)

            # Option to download the filtered image
            if st.button('Download Filtered Image', key=f'{uploaded_file.name}_download'):
                filtered_image_pil = Image.fromarray(filtered_image)
                buffered = io.BytesIO()
                filtered_image_pil.save(buffered, format="JPEG")
                st.download_button(label='Download', data=buffered.getvalue(), file_name=f'filtered_{uploaded_file.name}')
                st.success(f"Filtered image ({uploaded_file.name}) downloaded successfully!")

def plot_histogram(image):
    if len(image.shape) < 3:
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        plt.plot(hist, label='Intensity')
    else:
        color = ('b', 'g', 'r')
        for i, col in enumerate(color):
            hist = cv2.calcHist([image], [i], None, [256], [0, 256])
            plt.plot(hist, color=col, label=f'Channel {i}')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.title('Image Histogram')
    plt.legend()
    st.pyplot()

if __name__ == "__main__":
    main()
