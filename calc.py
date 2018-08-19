from skimage.measure import *
import numpy as np

class IntensityProps:
    def __init__(self, signal, median, total, mean, maximum, cropped):
        self.signal_intensity = signal
        self.median_intensity = median
        self.total_intensity = total
        self.mean_intensity = mean
        self.max_intensity = maximum
        
        self.is_cropped = cropped

def getOverlap(props1, props2, threshold):
    
    props2_bbox = props2.bbox
    props2_img = np.zeros((312, 312), dtype=int)
    props2_img[props2_bbox[0]:props2_bbox[2], props2_bbox[1]:props2_bbox[3]] = props2.image
    
    props1_bbox = props1.bbox
    props1_img = np.zeros((312, 312), dtype=int)
    props1_img[props1_bbox[0]:props1_bbox[2], props1_bbox[1]:props1_bbox[3]] = props1.image
    
    numOfIntersections = np.count_nonzero(props2_img & props1_img)
    overlap = numOfIntersections / props2.area
    recipOverlap = numOfIntersections / props1.area
    
    return (overlap >= threshold) & (recipOverlap >= threshold)

def region_intensity(props, sq_size, median_pixel, y_limit):
    
    intensity_img = props.intensity_image.copy() - median_pixel
    
    gfp_img = intensity_img[intensity_img != 0]
    
    cropped = props.bbox[2] >= y_limit-3
    
    if len(gfp_img) == 0:
        return IntensityProps(median_pixel, median_pixel, median_pixel, median_pixel, median_pixel, cropped)

    half_length = int(sq_size / 2)

    xsize=intensity_img.shape[1]
    ysize=intensity_img.shape[0]
    means = np.array([[np.mean(intensity_img[y-half_length:y+half_length+1, x-half_length:x+half_length+1]) for y in range(half_length, ysize-half_length)] for x in range(half_length, xsize-half_length)])
    
    try:
        maxcenterx = np.unravel_index(means.argmax(), means.shape)[0]+half_length
        maxcentery = np.unravel_index(means.argmax(), means.shape)[1]+half_length

        intensity = intensity_img[maxcentery-half_length:maxcentery+half_length+1, maxcenterx-half_length:maxcenterx+half_length+1].mean()
        
    except ValueError:
        intensity = gfp_img.mean()
    
    return IntensityProps(intensity, np.median(gfp_img), gfp_img.sum(), gfp_img.mean(), gfp_img.max(), cropped)

