import numpy as np
import maxflow
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

def display_img_arr(img_arr, r, c, dim,titles_arr):
    fl = 0
    fig = plt.figure(figsize = dim)
    for i in range(r):
        for j in range(c):
            if len(img_arr) == fl:
                break
            ax1 = fig.add_subplot(r, c, fl + 1)
            ax1.set_title(titles_arr[fl], fontsize = 20)
            ax1.imshow(img_arr[fl], cmap = 'gray')
            fl = fl + 1
    plt.show()
    
class GrabCut:
    def __init__(self, img, gamma, k, max_iter, bgRect, mask, iters):
        self.height = img.shape[1]
        self.width = img.shape[0]

        self.k = k
        self.gamma = gamma
        self.img = img.astype(np.float64)
        
        self.graph = maxflow.GraphFloat()
        self.pixels = self.graph.add_grid_nodes(self.img.shape[:2])
        
        self.set_beta()
        print("self.beta = ", self.beta)

        self.set_graph_weights()
        
        self.BG = 0 # Sure Background
        self.FG = 1 # Sure Foreground
        self.PR_BG = 2  #Probable background
        self.PR_FG = 3 #Probable foreground

        self.init_trimap(mask, bgRect)
        self.add_terminal_edges()
        self.iterative_step(iters, max_iter)
        

    def set_beta(self):
        beta = 0
        # calculates average over an image sample for (z_m-z_n)^2 for 
        self._left_diff = self.img[:, 1:] - self.img[:, :-1] # Left-difference
        self._upleft_diff = self.img[1:, 1:] - self.img[:-1, :-1] # Up-Left difference
        self._up_diff = self.img[1:, :] - self.img[:-1, :] # Up-difference
        self._upright_diff = self.img[1:, :-1] - self.img[:-1, 1:] # Up-Right difference
        # beta is as described in the paper
        beta = (self._left_diff*self._left_diff).sum() + (self._upleft_diff*self._upleft_diff).sum() \
            + (self._up_diff*self._up_diff).sum() + (self._upright_diff*self._upright_diff).sum() # According to the formula
        self.beta = 1/(2*beta/(4*self.width*self.height - 3*self.width - 3*self.height + 2))
        # 4*self.width*self.height - 3*self.width - 3*self.height + 2 is the number of pairs of neighbouring pixels in the image

    def set_graph_weights(self):
        for i in range(self.height):
            for j in range(self.width):
                current_pixel = self.pixels[j, i]
                if j-1 >= 0: # if top neighbor exists
                    dest_node = self.pixels[j-1, i]
                    temp = np.sum((self.img[j, i] - self.img[j-1, i])**2)
                    wt = np.exp(-self.beta * temp)
                    
                    n_link = self.gamma/1 * wt
                    self.graph.add_edge(current_pixel, dest_node , n_link, n_link)

                if i-1 >= 0: # if left neighbor exists
                    dest_node = self.pixels[j, i-1]
                    temp = np.sum((self.img[j, i] - self.img[j, i-1])**2)
                    wt = np.exp(-self.beta * temp)
                    n_link = self.gamma/1 * wt
                    self.graph.add_edge(current_pixel, dest_node, n_link, n_link)
                
                if i-1 >= 0 and j-1 >= 0: # if top left neighbor exists
                    dest_node  = self.pixels[j-1, i-1]
                    temp = np.sum((self.img[j, i] - self.img[j-1, i-1])**2)
                    wt = np.exp(-self.beta * temp )
                    n_link = self.gamma/np.sqrt(2) * wt
                    self.graph.add_edge(current_pixel, dest_node, n_link, n_link)
                
                if i+1 < self.height and j-1 >= 0: # if top right neighbor exists
                    temp = np.sum((self.img[j, i] - self.img[j-1, i+1])**2)
                    wt = np.exp(-self.beta * temp)
                    n_link = self.gamma/np.sqrt(2) * wt
                    self.graph.add_edge(current_pixel, dest_node, n_link, n_link)


    def init_trimap(self, mask, bgRect):
        self.bgRect = bgRect
        x, y, w, h = bgRect
        temp  = np.ones(shape = self.img.shape[:2])

       
        self.trimap = self.BG * temp # Initially all trimap background
        self.trimap[np.where(mask == 0)] = self.BG # Sure background
        y_start = y
        y_end = y_start+h+1
        x_start = x
        x_end = x_start+w+1

        self.trimap[y_start:y_end, x_start:x_end] = self.PR_FG # trimap unknown set
        self.trimap[np.where(mask == 1)] = self.FG # Sure foreground

    def add_terminal_edges(self):
        x, y = np.where(self.trimap == self.FG)

        for i in range(len(x)):
            x_i = x[i]
            y_i = y[i]
            edge = self.pixels[x_i, y_i]
            self.graph.add_tedge(edge, np.inf, 0)
        
        x, y = np.where(self.trimap == self.BG)
        
        for i in range(len(x)):
            x_i = x[i]
            y_i = y[i]
            edge = self.pixels[x_i, y_i]
            self.graph.add_tedge(edge, 0, np.inf)

    def set_cov_inv(self):
        self.cov_inv_fg =  np.linalg.inv(self.fg_gmm.covariances_)
        self.cov_inv_bg =  np.linalg.inv(self.bg_gmm.covariances_)

    def set_cov_det(self):
        self.bg_cov_det = np.linalg.det(self.bg_gmm.covariances_)
        self.fg_cov_det = np.linalg.det(self.fg_gmm.covariances_)

    def set_kmeans(self, max_iter):
        self.bg_kmeans = KMeans(n_clusters=self.k, max_iter = max_iter)
        self.fg_kmeans = KMeans(n_clusters=self.k, max_iter = max_iter)

    def set_gmm(self):
        self.bg_gmm = GaussianMixture(n_components = self.k)
        self.fg_gmm = GaussianMixture(n_components = self.k)

    def iterative_step(self, iters, max_iter):
        for i in range(iters):
            print(f"Iteration {i+1}")
            bg_indices = np.where(np.logical_or(self.trimap == self.BG, self.trimap == self.PR_BG))
            fg_indices = np.where(np.logical_or(self.trimap == self.FG, self.trimap == self.PR_FG))

            bg_set = self.img[bg_indices]
            fg_set = self.img[fg_indices]


            self.set_kmeans(max_iter)
            BG_GMM = np.empty(shape = len(bg_set), dtype = int)
            FG_GMM = np.empty(shape = len(fg_set), dtype = int)

            BG_KM = self.fg_kmeans.fit(bg_set) # K Means for background pixels
            FG_KM = self.fg_kmeans.fit(fg_set) # K Means for foreground pixels
            
            self.set_gmm()

            self.bg_gmm.fit(bg_set, BG_KM.labels_)
            self.fg_gmm.fit(fg_set, FG_KM.labels_)
           
            BG_GMM = self.bg_gmm.predict(bg_set)
            FG_GMM = self.fg_gmm.predict(fg_set)

            self.bg_gmm.fit(bg_set, BG_GMM)
            self.fg_gmm.fit(fg_set, FG_GMM)

            self.set_cov_det()

            D_bg = self.bg_gmm.weights_ / np.sqrt(self.bg_cov_det)
            D_fg = self.fg_gmm.weights_ / np.sqrt(self.fg_cov_det)
            
            self.set_cov_inv()
            tedge_weights_bg = np.empty(shape = (self.img.shape[0],self.img.shape[1]),dtype = np.float64)
            tedge_weights_fg = np.empty(shape = (self.img.shape[0],self.img.shape[1]), dtype = np.float64)
            
            r_ind, c_ind = np.where(np.logical_or(self.trimap == self.PR_BG, self.trimap == self.PR_FG))
            
            for k in range(len(r_ind)):
                node = self.img[r_ind[k], c_ind[k]]
                D_BG = 0
                D_FG = 0
                for j in range(self.k):
                    bg_u = self.bg_gmm.means_[j]
                    fg_u = self.fg_gmm.means_[j]
                    D_BG += D_bg[j] * np.exp(-0.5 * (node - bg_u).reshape(1, 3) @ self.cov_inv_bg[j] @ (node - bg_u).reshape(3, 1))[0][0] 
                    D_FG += D_fg[j] * np.exp(-0.5 * (node - fg_u).reshape(1, 3) @ self.cov_inv_fg[j] @ (node - fg_u).reshape(3, 1))[0][0]

                tedge_weights_fg[r_ind[k], c_ind[k]] = -np.log(D_BG)
                tedge_weights_bg[r_ind[k], c_ind[k]] = -np.log(D_FG)

                self.graph.add_tedge(self.pixels[r_ind[k], c_ind[k]], tedge_weights_fg[r_ind[k], c_ind[k]], tedge_weights_bg[r_ind[k], c_ind[k]])
            
           
            self.graph.maxflow()
          
            for j in range(len(r_ind)):
                edge = self.pixels[r_ind[j], c_ind[j]]
                self.graph.add_tedge(edge, -tedge_weights_fg[r_ind[j], c_ind[j]], -tedge_weights_bg[r_ind[j], c_ind[j]])
                
                if self.graph.get_segment(edge) == 0:
                    self.trimap[r_ind[j], c_ind[j]] = self.PR_FG
                else:
                    self.trimap[r_ind[j], c_ind[j]] = self.PR_BG