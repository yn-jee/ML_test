import numpy as np


class MaxPool2:
    def iterate_regions(self, image):
        h, w, _ = image.shape
        new_h = h // 2  # // 연산은 나눗셈의 몫을 가져옴, 즉 나눗셈 결과의 정수만 가져옴
        new_w = w // 2

        for i in range(new_h):
            for j in range(new_w):
                im_region = image[(i * 2):(i * 2 + 2), (j * 2):(j * 2 + 2)]
                yield im_region, i, j

    def forward(self, input):
        self.last_input = input  # Back propagation에서 사용되므로, 지금 몰라도 됨

        h, w, num_filters = input.shape
        output = np.zeros((h // 2, w // 2, num_filters))

        for im_region, i, j in self.iterate_regions(input):
            output[i, j] = np.amax(im_region, axis=(0, 1))  # array에서 max값 찾기

        return output

    def backprop(self, d_L_d_out):  # 13 x 13 x 8
        d_L_d_input = np.zeros(self.last_input.shape)  # 26 x 26 x 8 크기

        for im_region, i, j in self.iterate_regions(self.last_input):
            h, w, f = im_region.shape  # 2 x 2 x 8
            amax = np.amax(im_region, axis=(0, 1))

            for i2 in range(h):  # range(h): 0~1
                for j2 in range(w):  # range(w): 0~1
                    for f2 in range(f):  # range(f): 0~7
                        # If this pixel was the max value, copy the gradient to it.
                        if im_region[i2, j2, f2] == amax[f2]:
                            d_L_d_input[i * 2 + i2, j * 2 + j2, f2] = d_L_d_out[i, j, f2]

        return d_L_d_input