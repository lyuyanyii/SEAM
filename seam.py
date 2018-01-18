import numpy as np
import cv2
import tqdm

class Seam:

    def __init__(self, file_name, output_name, info_name, rate, mask = None, protect = False, removal = False):
        self.input = cv2.imread(file_name).astype(np.float32)
        self.output_shape = (int(self.input.shape[0] * rate), int(self.input.shape[1] * rate))

        self.constant = 999

        self.output = np.array(self.input)
        n, m = self.input.shape[:2]
        self.trace = np.arange(n * m).reshape(n, m)
        mask = np.zeros((n, m))

        if removal:
            self.mask = mask
            self.object_removing()
            cv2.imwrite(output_name, self.output)
        else:
            if not protect:
                mask = np.zeros(self.input.shape[:2])
            self.mask = mask
            self.seam_carving()
            cv2.imwrite(output_name, self.output)
            
            if rate < 1:
                n, m = self.input.shape[:2]
                mask = np.zeros((n, m), dtype = np.int32)
                for idx in self.trace.flatten():
                    x, y = int(idx) // m, int(idx) % m
                    mask[x, y] = 1
                self.input = self.input.transpose(2, 0, 1)
                self.input = self.input * mask
                self.input = self.input.transpose(1, 2, 0)
                cv2.imwrite(info_name, self.input)
            else:
                mask = (self.trace > 0)
                self.output = self.output.transpose(2, 0, 1)
                self.output *= mask
                self.output = self.output.transpose(1, 2, 0)
                cv2.imwrite(info_name, self.output)

    def object_removing(self):
        while (self.mask > 0).any():
            ener_map = self.calc_ener_map(self.output)
            ener_map[ np.where(self.mask > 0) ] *= -self.constant
            f = self.DP(ener_map)
            seam = self.get_seam(f)
            self.output = self.seam_deleting(self.output, seam)
            self.mask = self.seam_deleting(self.mask, seam)

        n = self.input.shape[1] - self.output.shape[1]

        self.seam_inserting(n)

    def seam_carving(self):
        dx, dy = self.input.shape[0] - self.output_shape[0], self.input.shape[1] - self.output_shape[1]

        print("Row Processing")
        if dy > 0:
            self.seam_removing(dy)
        elif dy < 0:
            self.seam_inserting(-dy)

        print("Column Processing")
        self.output = self.rotate(self.output)
        self.mask = self.rotate(self.mask)
        self.trace = self.rotate(self.trace)
        if dx > 0:
            self.seam_removing(dx)
        elif dx < 0:
            self.seam_inserting(-dx)
        self.output = self.rotate(self.output)
        self.trace = self.rotate(self.trace)

    def rotate(self, img):
        if len(img.shape) == 3:
            img = img.transpose(1, 0, 2)
        else:
            img = img.transpose(1, 0)
        return img

    def seam_removing(self, n):
        for i in tqdm.tqdm(range(n)):
            ener_map = self.calc_ener_map(self.output)
            ener_map[ np.where(self.mask > 0) ] *= self.constant
            f = self.DP(ener_map)
            seam = self.get_seam(f)
            self.output = self.seam_deleting(self.output, seam)
            self.mask = self.seam_deleting(self.mask, seam)
            self.trace = self.seam_deleting(self.trace, seam)

    def seam_inserting(self, n):
        seams = []
        ori_mask = np.array(self.mask)
        ori_input = np.array(self.output)
        for i in tqdm.tqdm(range(n)):
            ener_map = self.calc_ener_map(self.output)
            ener_map[ np.where(self.mask > 0) ] *= self.constant
            f = self.DP(ener_map)
            seam = self.get_seam(f)
            seams.append(seam)
            self.output = self.seam_deleting(self.output, seam)
            self.mask = self.seam_deleting(self.mask, seam)

        self.output = ori_input
        self.mask = ori_mask
        for i in tqdm.tqdm(range(n)):
            seam = seams.pop(0)
            self.output = self.seam_adding(self.output, seam, 1)
            self.trace = self.seam_adding(self.trace, seam, 0)
            seams = self.update_seams(seams, seam)

    def calc_ener_map(self, img):
        b, g, r = cv2.split(img)
        b_ener = np.absolute(cv2.Scharr(b, -1, 1, 0)) + np.absolute(cv2.Scharr(b, -1, 0, 1))
        g_ener = np.absolute(cv2.Scharr(g, -1, 1, 0)) + np.absolute(cv2.Scharr(g, -1, 0, 1))
        r_ener = np.absolute(cv2.Scharr(r, -1, 1, 0)) + np.absolute(cv2.Scharr(r, -1, 0, 1))
        return b_ener + g_ener + r_ener

    def DP(self, ener_map):
        n, m = ener_map.shape
        f = np.zeros((n, m))
        for i in range(1, n):
            for j in range(m):
                f[i, j] = ener_map[i, j] + np.min(f[i - 1, max(0, j - 1):min(j + 1, m - 1)])
        return f

    def get_seam(self, f):
        n, m = f.shape
        seam = np.zeros((n, ), dtype = np.int32)
        seam[-1] = np.argmin(f[-1])
        for i in reversed(range(n - 1)):
            seam[i] = np.argmin(f[i, max(0, seam[i + 1] - 1):min(seam[i + 1] + 1, m - 1)]) + max(0, seam[i + 1] - 1)
        assert (seam >= 0).all() and (seam < m).all()
        return seam
    
    def seam_deleting(self, img, seam):
        n, m = img.shape[:2]
        img = img.reshape(n, m, -1)
        ans = np.zeros((n, m - 1, img.shape[2]))
        for i in range(n):
            ans[i, :, :] = np.delete(img[i, :, :], seam[i], axis = 0)
        if img.shape[2] == 1:
            ans = ans.reshape(n, m - 1)
        return ans

    def seam_adding(self, img, seam, padding):
        n, m = img.shape[:2]
        img = img.reshape(n, m, -1)
        ans = np.zeros((n, m + 1, img.shape[2]))
        for i in range(n):
            p = np.average(img[i, max(0, seam[i]-1):min(m-1,seam[i]+1), :], axis = 0) * padding
            ans[i, :, :] = np.concatenate([img[i, :seam[i], :], [p], img[i, seam[i]:, :]], axis = 0)
        if img.shape[2] == 1:
            ans = ans.reshape(n, m + 1)
        return ans

    def update_seams(self, seams, seam):
        new_seams = []
        for old_seam in seams:
            old_seam[ np.where(old_seam >= seam) ] += 2
            new_seams.append(np.array(old_seam))
        return new_seams


if __name__ == '__main__':
    #Seam("img/3.jpg", "img/3_sc.jpg", "img/3_seam.jpg", 0.8)
    suf = [".jpg", ".png", ".jpg", ".jpg", ".jpg", ".jpg", ".bmp"]

    for i in range(1, 8):
        print(i)
        name = "img/{}".format(i) + suf[i - 1]
        out_name = "img/{}_sc_0.8".format(i) + suf[i - 1]
        info_name = "img/{}_seam_0.8".format(i) + suf[i - 1]
        Seam(name, out_name, info_name, 0.8)

    for i in range(1, 8):
        print(i)
        name = "img/{}".format(i) + suf[i - 1]
        out_name = "img/{}_sc_1.2".format(i) + suf[i - 1]
        info_name = "img/{}_seam_1.2".format(i) + suf[i - 1]
        Seam(name, out_name, info_name, 1.2)
