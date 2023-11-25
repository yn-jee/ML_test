"""import numpy as np


class Conv3x3:  # 파이썬에서 클래스 생성하는 방법
    def __init__(self, num_filters):  # __init__함수는 생성자 함수로, 객체 생성할 때 실행되는 함수입니다.
        self.num_filters = num_filters

        self.filters = np.random.randn(num_filters, 3, 3) / 9  # num_filters x 3 x 3

    def iterate_regions(self, image):
        h, w, _ = image.shape  # Updated to unpack 3 dimensions

        for i in range(h - 2):
            for j in range(w - 2):
                im_region = image[i:(i + 3), j:(j + 3)]
                yield im_region, i, j

    def forward(self, input):
        self.last_input = input
        h, w, _ = input.shape  # Get the dimensions of the input
        output = np.zeros((h - 2, w - 2, self.num_filters))

        for im_region, i, j in self.iterate_regions(input):
            for f in range(self.num_filters):
                output[i, j, f] = np.sum(im_region * self.filters[f])

        return output

"""

import numpy as np

class Conv3x3:
    def __init__(self, num_filters):
        self.num_filters = num_filters
        self.filters = np.random.randn(num_filters, 3, 3) * np.sqrt(2. / num_filters) Z

    def iterate_regions(self, image):
        h, w, _ = image.shape
        for i in range(h - 2):
            for j in range(w - 2):
                im_region = image[i:(i + 3), j:(j + 3)]
                yield im_region, i, j

    def forward(self, input):
        self.last_input = input  # Store the input for backpropagation
        h, w, _ = input.shape
        output = np.zeros((h - 2, w - 2, self.num_filters))

        for im_region, i, j in self.iterate_regions(input):
            for f in range(self.num_filters):
                output[i, j, f] = np.sum(im_region * self.filters[f])

        return output

    def backprop(self, d_L_d_out, learn_rate):
        d_L_d_filters = np.zeros(self.filters.shape)

        for im_region, i, j in self.iterate_regions(self.last_input):
            for f in range(self.num_filters):
                d_L_d_filters[f] += d_L_d_out[i, j, f] * im_region

        self.filters -= learn_rate * d_L_d_filters

        return None  # As specified, this function returns None
