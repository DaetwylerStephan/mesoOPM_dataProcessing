
from tifffile import imread, imwrite
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy
import scipy.ndimage as ndimage
import skimage



class mesoOPM_stitcher():
    """
    This class stitches images from the mesoscopic OPM system
    """
    def __init__(self):
        """
        init stitcher
        """

        self.rawimages = []
        self.weighted_images = []
        self.sigmoid_shapes=[]

    def load_image(self, pathlist):
        """
        load images specified in path list and save the raw data in self.rawdata
        :param pathlist:
        :return:
        """
        self.rawimages = []
        for imagepath_name in pathlist:
            self.rawimages.append(imread(imagepath_name))
            print("image loaded" + imagepath_name)

    def calculate_sigmoid_curves(self, ranges):
        """
        generate weighing arrays using sigmoidal functions and save them in self.sigmoid_shapes
        :param ranges: ranges of contribution of different images (in ascending order)
        """
        end_ranges = np.max(ranges)
        self.sigmoid_shapes = np.ones(shape=(len(ranges), end_ranges))

        for iter in range(len(ranges) - 1):
            print(iter)

            endposition = ranges[iter][1]
            startposition_next = ranges[iter + 1][0]
            overlapsize = endposition - startposition_next

            # define sigmoidal blending curves
            n1sigmoidrange = np.arange(-6, 6, 12. / overlapsize)
            n1sigmoid = 1 / (1 + np.exp(n1sigmoidrange))
            inverse_1sigmoid = 1 - n1sigmoid
            min(n1sigmoid + inverse_1sigmoid)

            self.sigmoid_shapes[iter][startposition_next:endposition] = n1sigmoid
            self.sigmoid_shapes[iter][endposition:] = 0
            self.sigmoid_shapes[iter + 1][0:startposition_next] = 0
            self.sigmoid_shapes[iter + 1][startposition_next:endposition] = inverse_1sigmoid

    def update_weighted_images(self):
        """
        use the sigmoid functions to weigh the different images and save them in self.weighted_images
        :return:
        """
        # make sigmoidal arrays for multiplication
        i_image = 0
        self.weighted_images = []
        for i_image in range(len(self.rawimages)):
            multiplication_matrix = np.zeros(shape=self.rawimages[i_image].shape)
            for i_plane in range(self.rawimages[i_image].shape[0]):
                multiplication_matrix[i_plane, :, :] = self.sigmoid_shapes[i_image][i_plane]
            self.weighted_images.append(multiplication_matrix * self.rawimages[i_image])

    def sum_up_weightedimages(self):
        """
        sums up weighted image
        :return: final image in uint16 format
        """
        finalimage = np.zeros(shape=self.rawimages[0].shape, dtype='uint16')
        for i_image in range(len(self.rawimages)):
            finalimage = finalimage + self.weighted_images[i_image]

        finalimage_uint16 = finalimage.astype('uint16')

        return finalimage_uint16


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    #############################################################
    # perform tile fusion
    #############################################################
    mesoStitcher = mesoOPM_stitcher()
    experimentfolder = "/archive/bioinformatics/Danuser_lab/Fiolka/MicroscopeDevelopment/mesoOPM/Zebrafish/vasculature/230526/Cell15_deskewed"
    experimentfolder_result = "/archive/bioinformatics/Danuser_lab/Fiolka/MicroscopeDevelopment/mesoOPM/Beads500nm/Agarose/SD_stitching/"

    imagelist = [os.path.join(experimentfolder, "Deskew_Rot_001_1_CH00_000000.tiff"),
                 os.path.join(experimentfolder, "Deskew_Rot_000_1_CH00_000000.tiff"),
                 ]
    #ranges = [(0, 225), (203, 365), (345, 497), (478,615), (580,764), (734, 906) ]
    ranges = [(0, 372), (352, 614)]

    mesoStitcher.load_image(imagelist)
    print("images loaded")
    mesoStitcher.calculate_sigmoid_curves(ranges)
    print("sigmoidal curves calculated")
    mesoStitcher.update_weighted_images()
    print("weights updated")
    finalimage_uint16 = mesoStitcher.sum_up_weightedimages()

    imwrite(experimentfolder_result + "fused_weighted_zebrafish_cell15_2tiles_230526.tif", finalimage_uint16)
    print("image saved")


