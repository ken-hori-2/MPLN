from cProfile import label
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
from matplotlib import patches
from matplotlib import animation, gridspec
from sklearn import preprocessing

# エージェントの移動の様子を可視化します
# 参考URL http://louistiao.me/posts/notebooks/embedding-matplotlib-animations-in-jupyter-notebooks/

# animation_Edit.py の整理ver.
# ani_integrate_test3_main.py の整理ver.

class Anim():
    
    def __init__(self, STATE_HISTORY, stress, phi, standard):

        self.state_history = STATE_HISTORY
        self.stress = stress


        self.fig = plt.figure(figsize=(9, 6)) #(12, 8))
        # self.fig = plt.figure(figsize=(8, 8.5))
        self.ax = plt.gca()
        self.ims = []

        # # self.gs = gridspec.GridSpec(2, 2, height_ratios=(1, 1))
        # self.gs = gridspec.GridSpec(1, 3)
        # ss1 = self.gs.new_subplotspec((0, 0), rowspan=1,colspan=2)
        # ss2 = self.gs.new_subplotspec((0, 2), rowspan=1,colspan=1)
        # self.ax = [plt.subplot(ss1), plt.subplot(ss2)]

        self.gs = gridspec.GridSpec(3, 3) # , height_ratios=(1, 1))
        ss1 = self.gs.new_subplotspec((0, 0), rowspan=3,colspan=2) #1)
        ss2 = self.gs.new_subplotspec((1, 2), rowspan=1,colspan=1) #2)
        ss3 = self.gs.new_subplotspec((2, 2), rowspan=1,colspan=1) #2)
        ss4 = self.gs.new_subplotspec((0, 2), rowspan=1,colspan=1) #2)
        self.ax = [plt.subplot(ss1), plt.subplot(ss2), plt.subplot(ss3), plt.subplot(ss4)] # , plt.subplot(ss5)]
        "-----------------------------------------------------"
        # グラフデータの初期化
        self.T = []
        # Statas数推移
        self.Stress_List= []
        self.im = []
        self.TEST = []
        self.legend_flag = True

        self.phi = phi
        self.standard = standard
        self.standard_list = []




        self.grid = [

            # Environment (a)
            [0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 1],
            [0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0],
            [0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0],
            [0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0],
            [0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0],
            [0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0],
            [0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0],
            [0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0],
            [0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0],
            [0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0],
            [0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0],
            [0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0],
            [0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0],
        ]

        self.NODELIST = [
            # Environment (2d)
                ["", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""],
                ["", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""],
                ["", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""],
                ["", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""],
                ["x", "", "x", "", "x", "", "x", "", "x", "", "x", "", "D", "", "g", "", "", "", ""],
                ["", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""],
                ["", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""],
                ["", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""],
                ["", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""],
                ["x", "", "x", "", "x", "", "x", "", "x", "", "B", "", "C", "", "x", "", "", "", ""],
                ["", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""],
                ["", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""],
                ["", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""],
                ["", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""],
                ["x", "", "x", "", "x", "", "x", "", "O", "", "A", "", "x", "", "x", "", "", "", ""],
                ["", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""],
                ["", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""],
                ["", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""],
                ["", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""],
        ]

        
    def view_plot_text(self):


        import matplotlib.pyplot as plt

        
        import numpy as np
        from PIL import Image
        
        # self.ax[0].plot([20.5], [20.5], marker="s", color='black', markersize = 520, alpha = 0.8)

        # 描画範囲の設定と目盛りを消す設定
        size = -19
        x, y = np.mgrid[-0.5:-size+0.5:1, -0.5:-size:1]
        self.ax[0].set_xlim(-0.5, -size-0.5)
        self.ax[0].set_ylim(-0.5, -size-0.5)
        self.ax[0].tick_params(axis='both', which='both', bottom='off', top='off',
                        labelbottom='off', right='off', left='off', labelleft='off')
        # self.ax[0].legend()
        test = [[22, 8], [23, 8], [24, 8], [25, 8], [26, 8], [27, 8], [21, 8], [20, 8], [19, 8], [18, 8], [17, 8], [16, 8], [15, 8], [14, 8], [13, 8], [12, 8], [11, 8], [10, 8], [9, 8], [8, 8], [7, 8], [6, 8], [5, 8], [4, 8], [3, 8], [2, 8], [1, 8], [0, 8]]

        # 格子状
        # test = [[22, 8], [23, 8], [23, 9], [23, 7], [23, 6], [24, 6], [22, 6], [21, 6], [23, 5], [23, 4], [24, 4], [25, 4], [26, 4], [22, 4], [21, 4], [20, 4], [19, 4], [18, 4], [18, 5], [18, 6], [17, 4], [18, 3], [18, 2], [17, 2], [16, 2], [15, 2], [14, 2], [13, 2], [12, 2], [13, 3], [13, 1], [13, 0], [12, 0], [11, 0], [10, 0], [14, 0], [15, 0], [16, 0], [9, 0], [8, 0], [7, 0], [11, 2], [10, 2], [9, 2], [8, 2], [7, 2], [6, 2], [5, 2], [6, 1], [6, 3], [6, 4], [6, 5], [5, 4], [4, 4], [3, 4], [2, 4], [6, 6], [7, 6], [8, 6], [6, 7], [6, 8], [7, 8], [6, 9], [6, 10], [8, 8], [9, 8], [10, 8], [11, 8], [12, 8], [13, 8], [14, 8], [13, 9], [13, 10], [12, 10], [11, 10], [10, 10], [9, 10], [8, 10], [7, 10], [5, 8], [13, 7], [4, 8], [3, 8], [2, 8], [1, 8], [0, 8], [0, 7], [0, 9], [0, 10], [1, 10], [2, 10], [3, 10], [4, 10], [5, 10], [13, 6], [14, 6], [15, 6], [16, 6], [17, 6], [18, 7], [18, 8], [19, 8], [17, 8], [16, 8], [15, 8], [18, 9], [18, 10], [19, 10], [17, 10], [16, 10], [15, 10], [14, 10], [18, 11], [20, 10], [21, 10], [22, 10], [23, 10], [23, 11], [23, 12], [22, 12], [23, 13], [24, 12], [25, 12], [26, 12], [24, 10], [21, 12], [23, 14], [20, 12], [19, 12], [18, 12], [18, 13], [18, 14], [18, 15], [18, 16], [19, 16], [20, 16], [21, 16], [17, 16], [22, 16], [23, 16], [24, 16], [25, 16], [23, 15], [25, 10], [24, 8], [25, 8], [21, 8], [26, 10], [22, 14], [21, 14], [24, 14], [25, 14], [26, 14], [27, 14], [28, 14], [28, 15], [20, 14], [17, 12], [19, 14], [17, 14], [16, 14], [15, 14], [14, 14], [13, 14], [13, 15], [13, 13], [13, 12], [14, 12], [13, 11], [12, 12], [11, 12], [20, 8], [19, 6], [28, 13], [25, 6], [26, 6], [27, 6], [28, 6], [28, 7], [28, 5], [28, 4], [27, 4], [23, 3], [23, 2], [22, 2], [21, 2], [20, 2], [23, 1], [24, 2], [23, 0], [22, 0], [21, 0], [20, 0], [24, 0], [25, 0], [25, 2], [26, 2], [27, 2], [28, 2], [28, 1], [28, 3], [28, 8], [26, 8], [27, 8], [27, 10], [28, 10], [28, 11], [28, 12], [27, 12], [28, 16], [27, 16], [26, 16], [28, 9], [12, 6], [13, 5], [13, 4], [14, 4], [15, 4], [12, 4], [5, 6], [9, 6], [10, 6], [11, 6], [12, 14], [13, 16], [14, 16], [15, 16], [15, 12], [12, 16], [11, 16], [10, 16], [9, 16], [8, 16], [10, 12], [9, 12], [8, 12], [7, 12], [16, 4], [11, 4], [10, 4], [9, 4], [8, 4], [7, 4], [19, 2], [18, 1], [18, 0], [17, 0], [19, 0], [6, 0], [5, 0], [4, 0], [3, 0], [28, 0], [20, 6], [1, 4], [0, 4], [0, 3], [0, 2], [1, 2], [2, 2], [3, 2], [4, 2], [0, 1], [0, 5], [0, 6], [1, 6], [2, 6], [3, 6], [4, 6], [16, 12], [6, 12], [6, 13], [6, 14], [7, 14], [5, 14], [6, 15], [8, 14], [9, 14], [10, 14], [11, 14], [6, 11], [0, 11], [0, 12], [1, 12], [2, 12], [3, 12], [4, 12], [5, 12], [0, 13], [0, 0], [1, 0], [2, 0], [27, 0], [26, 0], [16, 16], [0, 14], [0, 15], [1, 14], [2, 14], [3, 14], [4, 14], [0, 16], [6, 16], [7, 16], [5, 16], [4, 16], [3, 16], [2, 16], [1, 16]]
        test = [[18, 6], [18, 5], [18, 4], [18, 3], [17, 4], [16, 4], [17, 6], [16, 6], [15, 6], [18, 7], [14, 6], [18, 8], [18, 9], [17, 8], [18, 10], [18, 11], [18, 12], [18, 13], [18, 14], [17, 14], [18, 15], [18, 16], [15, 4], [18, 2], [17, 16], [16, 16], [15, 16], [14, 16], [13, 16], [12, 16], [11, 16], [10, 16], [9, 16], [8, 16], [7, 16], [6, 16], [5, 16], [4, 16], [3, 16], [2, 16], [1, 16], [0, 16],  [16, 8], [15, 8], [14, 8], [13, 8], [13, 7], [13, 6], [13, 9], [13, 10], [12, 10], [11, 10], [10, 10], [14, 10], [15, 10], [16, 10], [17, 10], [17, 12], [16, 12], [15, 12], [14, 12], [13, 12], [13, 13], [13, 11], [12, 8], [9, 10], [8, 10], [7, 10], [6, 10], [6, 11], [6, 12], [6, 13], [6, 14], [7, 14], [8, 14], [5, 10], [4, 10], [6, 15], [5, 14], [9, 14], [10, 14], [11, 14], [12, 14], [13, 14], [14, 14], [15, 14], [16, 14],  [13, 15], [12, 12], [11, 12], [10, 12], [9, 12], [8, 12], [7, 12], [5, 12], [4, 12], [3, 12], [2, 12], [1, 12], [0, 12], [0, 13], [0, 14], [0, 15], [1, 14], [24, 10], [21, 10], [20, 10], [21, 14], [24, 16], [25, 16], [26, 16], [24, 14], [25, 14], [26, 14], [27, 14], [27, 13], [27, 2], [27, 1], [27, 0], [26, 0], [25, 0], [24, 0], [23, 0], [21, 0], [20, 0], [19, 0], [18, 0], [17, 0], [18, 1], [17, 2], [11, 8], [10, 8], [9, 8], [8, 8], [7, 8], [6, 8], [5, 8], [4, 8], [3, 8], [2, 8], [1, 8], [0, 8], [0, 7], [0, 6], [1, 6], [2, 6], [0, 5], [3, 6], [4, 6], [5, 6], [6, 6], [7, 6], [8, 6], [6, 5], [6, 4], [9, 6], [10, 6], [11, 6], [12, 6], [13, 5], [13, 4], [13, 3], [13, 2], [14, 4], [14, 2], [15, 2], [16, 2], [16, 0], [15, 0], [14, 0], [13, 0], [13, 1], [12, 2], [11, 2], [12, 4], [11, 4], [10, 4], [9, 4], [8, 4], [7, 4], [5, 4], [6, 7], [6, 9], [12, 0], [11, 0], [10, 0], [9, 0], [8, 0], [7, 0], [6, 0], [5, 0], [4, 0], [3, 0], [2, 0], [1, 0], [0, 0], [6, 1], [6, 2], [5, 2], [6, 3], [4, 2], [3, 2], [2, 2], [1, 2], [0, 2], [0, 1], [7, 2], [8, 2], [9, 2], [10, 2], [4, 4], [3, 4], [0, 9], [2, 4], [1, 4], [0, 4], [0, 3], [0, 10], [1, 10], [2, 10], [3, 10], [0, 11], [4, 14], [3, 14], [27, 15], [27, 16], [2, 14]]
        
        
        
        # for t in range(len(test)):


        #     state = test[t]
        #     y = 46-(state[0] + state[0] + 0.5) # 14.5)
        #     x = state[1] + state[1] + 0.5

        #     self.ax[0].plot([x], [y], marker="s", color='grey', markersize = 18)

        # LandMark = [[22, 8], [18, 8], [18, 10], [13, 10], [13, 12], [6, 12], [6, 14], [0, 14],  [0, 16]]


        # # self.ax[0].plot([20.5], [20.5], marker="s", color='black', markersize = 520, alpha = 0.8)
        # for t in range(len(LandMark)):
        #     state = LandMark[t]
        #     y = 46-(state[0] + state[0] + 0.5) # 14.5)
        #     x = state[1] + state[1] + 0.5

        #     if state == [0, 8]:
        #         # self.ax[0].plot([x], [y], marker="s", color='orange', markersize = 18, alpha = 0.5)
        #         # self.ax[0].plot([x], [y], marker="s", color='red', markersize = 18, alpha = 0.5)
        #         pass
        #     # elif state == [5, 8]: # or state == [9, 8]: # or state == [5, 8]:
        #     #     # self.ax[0].plot([x], [y], marker="s", color='orange', markersize = 18, alpha = 0.5)
        #     #     self.ax[0].plot([x], [y], marker="s", color='blue', markersize = 18, alpha = 0.5)
        #     #     # self.ax[0].plot([x], [y], marker="s", color='blue', markersize = 28, alpha = 0.5)
        #     #     # pass
        #     else:
        #         self.ax[0].plot([x], [y], marker="s", color='green', markersize = 18, alpha = 0.5)


        "-----"
        import matplotlib.cm as cm
        import matplotlib.colors as colors

        size = -19

        pi = np.pi
        cmap = cm.binary
        cmap_data = cmap(np.arange(cmap.N))
        cmap_data[0, 3] = 0 # 0 のときのα値を0(透明)にする
        customized_gray = colors.ListedColormap(cmap_data)

        dem = [[0.0 for i in range(-size)] for i in range(-size)] #known or unknown
        x, y = np.mgrid[-0.5:-size+0.5:1, -0.5:-size:1]
        # x, y = np.mgrid[-0.5:-size-0.5:1, 0.5:-size+0.5:1]

        demGrid = self.ax[0].pcolor(x, y, dem, vmax=1, cmap=plt.cm.BrBG, alpha=0.2)
        
        soil = [[1.0 for i in range(-size)] for i in range(-size)] #2Dgridmap(xw, yw)
        
        "Add test-LBM"
        Node = ["A", "B", "C", "D", "O", "E", "F", "G",     "g"]
        test = [[0.0 for i in range(-size)] for i in range(-size)] #2Dgridmap(xw, yw) # Node
        for ix in range(-size):
            for iy in range(-size):   
                if self.grid[ix][iy] == 9:
                    soil[ix][iy] = 1 #sandy terrain
                else:
                    soil[ix][iy] = 0 #hard ground
                    "Add test-LBM"
                    if self.NODELIST[ix][iy] in Node: # Node
                        test[ix][iy] = 0.5 # 1 #sandy terrain
                    # elif self.env.NODELIST[ix][iy] == "x":
                    #     test[ix][iy] = 0.1 #sandy terrain

        "Add"
        test = np.flip(test, 1)
        test = np.rot90(test, k=1)
        test = np.fliplr(test)
        # lm = self.ax[0].pcolor(x, y, test, vmax=1, cmap=plt.cm.BrBG, alpha = 0.2)
        # lm = self.ax[0].pcolor(x, y, test, vmax=1, cmap=plt.cm.Greens, alpha = 0.5)
        lm = self.ax[0].pcolor(x, y, test, vmax=1, cmap=plt.cm.BrBG, alpha = 1.0)
        
        soil = np.flip(soil, 1)
        soil = np.rot90(soil, k=1)
        terrain = self.ax[0].pcolor(x, y, soil, vmax=1, cmap=plt.cm.Greys, alpha = 0.5)

        # map = [[1.0 for i in range(-size)] for i in range(-size)] #known or unknown

        # map  = np.flip(map, 1)
        # map = np.rot90(map, k=1)
        # known = self.ax[0].pcolor(x, y, map, vmax=1, cmap=customized_gray)


        "----------"
        Node = ["A", "B", "C", "D", "O", "E", "F", "G",     "g"]
        # if self.env.grid[state.row][state.column] in Node:
        # self.to_arrows(A, V)
        "----------"
        
        # # plt.plot(state.column, -state.row, ".y", markersize=10)
        # if self.NODELIST[state.row][state.column] in Node:
        #     self.ax[0].plot(state.column, -state.row, ".r", markersize=10)
        # else:
        #     self.ax[0].plot(state.column, -state.row, ".y", markersize=10)

        # tx = (8, 10, 12, 12, 14, 14, 16, 18)
        # ty = (-14+0.3, -14+0.3, -14+0.3, -9+0.3, -9+0.3, -4+0.3, -4+0.3, -4+0.3)
        
        # tx = (8, 8, 8, 8, 8) # , 12, 14, 14, 16, 18)
        # ty = (15+0.3, 12+0.3, 9+0.3, 6+0.3, 3+0.3) # , -4+0.3, -4+0.3)
        "----- 2d -----"
        tx = (8, 10, 10, 12, 12) # , 12, 14, 14, 16, 18)
        ty = (18-14, 18-14, 18-9, 18-9,18-4) # , -4+0.3, -4+0.3)
        # plt.plot(tx, ty, "*m", markersize=5)
        self.ax[0].plot(tx, ty, "*g", markersize=10)

        # goal_x = (8)
        # goal_y = (0.3)
        "----- 2d -----"
        goal_x = (14)
        goal_y = (18-4)
        self.ax[0].plot(goal_x, goal_y, "*r", markersize=10)


        # png_path = os.path.join(result_dir, "{0}.png".format(ww))
        # plt.savefig(png_path)
        
        # plt.show()
        "-----"



    def show(self, map):
        import matplotlib.cm as cm
        import matplotlib.colors as colors

        size = -19
        pi = np.pi
        cmap = cm.binary
        cmap_data = cmap(np.arange(cmap.N))
        cmap_data[0, 3] = 0 # 0 のときのα値を0(透明)にする
        customized_gray = colors.ListedColormap(cmap_data)
        map  = np.flip(map, 1)
        map = np.rot90(map, k=1)
        x, y = np.mgrid[-0.5:-size+0.5:1, -0.5:-size:1]
        known = self.ax[0].pcolor(x, y, map, vmax=1, cmap=customized_gray)

        return known

    def obserb(self, init, size, map):
        
        init_x, init_y = init[0], init[1]

        Node = ["A", "B", "C", "D", "O", "E", "F", "G",     "g",     "x"]

        if self.NODELIST[init_x][init_y] in Node: #交差点のみ前後一マス観測
            for i in range(-1,2):
                if init_x+i < 0 or init_x+i >=size:
                # if init_x+i >= 0 or init_x+i <size:
                    continue
                for j in range(-1,2):
                
                    if init_y+j < 0 or init_y+j >=size:
                    # if init_y+j >= 0 or init_y+j <size:
                        continue
                    
                    map[init_x+i][init_y+j] = 0
        map[init_x][init_y] = 0 # 現在のマスのみ観測
                
        return map

                
    def move_history(self, Env_Anim):

        OVER_CAPACITY = 1.0
        x_vals = np.array([0.0, 1.0])

        # size = -19
        # map = [[1.0 for i in range(-size)] for i in range(-size)] #known or unknown

        for t in range(len(self.stress)): # state_history)): # フレームごとの描画内容
        # for t in range(len(self.state_history)):

            # map = test.obserb(state, size, map)

            self.T.append(t)
            self.im = []

            self.Stress_List.append(self.stress[t])
            # self.im += (self.ax[1].plot(self.T, self.Stress_List, color="orange", alpha=0.7))
            self.im += (self.ax[1].plot(self.T, self.Stress_List, color="red", alpha=0.5))


            "Add 割合+基準距離"
            try:
                cm = plt.get_cmap("Purples")
                self.probability = (np.array([self.phi[t], 1.0 - self.phi[t]]))
                color_maps = [cm(self.phi[t]), cm(1-self.phi[t])]
                self.im += self.ax[2].bar(x_vals, self.probability, color=color_maps)
            except:
                pass
            
            try:
                self.standard_list.append(self.standard[t])
                self.im += (self.ax[3].plot(self.T, self.standard_list, color="orange", alpha=0.7))
            except:
                pass
            "Add 割合+基準距離"

            state = self.state_history[t]  # 現在の場所を描く


            # map = Env_Anim.obserb(state, size, map)
            # known = Env_Anim.show(map) # 一つ前の状態を表示させる場合
            # self.im += known



            try:
                prev_state = self.state_history[t-1]
            except:
                pass
            

            # if state[1] != 0:
            #     # y = 19-(state[0] + state[0] + 0.5) # 14.5)
            #     # x = state[1] + state[1] + 0.5
            #     y = 19 - state[0] - 1
            #     x = state[1]
            # else:
            #     # x = 0.5
            #     # y = 19-(state[0] + state[0] + 0.5) # 14.5)
            y = 19 - state[0] - 1
            x = state[1]
                
            try:
                if state == prev_state:
                    
                    if state[0] == prev_state[0]:
                        # self.im += self.ax[0].plot(x, y, marker="s", color='y', markersize = 18, alpha = 0.5)
                        self.im += self.ax[0].plot(x, y, ".y", markersize = 8)
                    else:
                        # self.im += self.ax[0].plot(x, y, marker="o", color='r', markersize = 15, alpha = 0.5)
                        self.im += self.ax[0].plot(x, y, ".r", markersize = 8)

                   
                else:
                    # self.im += self.ax[0].plot(x, y, marker="o", color='r', markersize = 15, alpha = 0.5)
                    self.im += self.ax[0].plot(x, y, ".r", markersize = 8)
            except:
                print("エラー(初回)")
                # self.im += self.ax[0].plot(x, y, marker="s", color='r', markersize = 15, alpha = 0.5)
                self.im += self.ax[0].plot(x, y, ".r", markersize = 8)
            self.ims.append(self.im)
            
            if t == 0:
                self.ims.append(self.im)

            #描画設定
            if self.legend_flag:  # 一回のみ凡例を描画
                self.ax[0].set_title("Environment")
                self.ax[3].set_title("Visualization")
                self.ax[0].scatter(0, -20, marker="o", color='r', label = "Agent Pose")
                # self.ax[0].scatter(0, -20, marker="s", color='green', label="Node")
                self.ax[0].scatter(0, -20, marker="s", color='white', label="Match Node")

                # self.ax[0].scatter(0, -20, marker="s", color='grey', label="Path", alpha=0.5)
                # self.ax[1].plot(self.T, self.Stress_List, color="red", alpha=0.5, label = "Accumulated Stress")
                self.ax[1].axhline(3, ls = "--", color = "red", label = "threshold")
                self.ax[1].plot(self.T, self.Stress_List, color="red", alpha=0.5, label = "Node's Stress")
                
                # self.ax[2].plot([-0.5, 1.5], [0.5, 0.5], color='r', linestyle='--', label='half in doubt(0.5)') # 平均
                
                self.ax[2].scatter(0, -20, marker="s", color='purple', label='Rate(%)', alpha=0.8)
                # self.ax[2].plot([-0.5, 1.5], [0.5, 0.5], color='r', linestyle='--', label='threshold')
                self.ax[2].axhline(0.5, ls = "--", color = "red", label = "threshold")

                self.ax[2].set_xticks(ticks=[0, 1]) # x軸目盛
                # self.ax[3].axhline(2, ls = "--", color = "red", label = "×2 & Thr") # here
                self.ax[3].axhline(2, ls = "--", color = "red", label = "threshold") # here
                # self.ax[3].plot(self.T, self.Stress_List, color="orange", alpha=0.7, label = "standard")
                self.ax[3].plot(self.T, self.Stress_List, color="orange", alpha=0.7, label = "Arc's Stress")
                # self.ax[2].set_xticks(ticks=[0, 1]) # x軸目盛
                # self.ax[3].axhline(2, ls = "--", color = "red", label = "×2 & Thr")
                # self.ax[3].plot(self.T, self.Stress_List, color="orange", alpha=0.7, label = "standard")
                self.legend_flag = False

            
    def view_anim(self): #　初期化関数とフレームごとの描画関数を用いて動画を作成する
        # self.anim = animation.ArtistAnimation(self.fig, self.ims, interval=450, repeat = True) # False)
        # self.anim = animation.ArtistAnimation(self.fig, self.ims, interval=455, repeat = True)
        self.anim = animation.ArtistAnimation(self.fig, self.ims, interval=250 + 450, repeat = True)


        # self.anim = animation.ArtistAnimation(self.fig, self.ims, interval=250, repeat = True)
        # self.ani = animation.ArtistAnimation(self.fig, self.ims, interval=250)
        # plt.legend(loc='lower right')
        # plt.legend(loc='center')
        self.ax[0].legend(loc='upper right')
        self.ax[1].legend(fontsize=7, loc='upper right')
        self.ax[2].legend(fontsize=7, loc='upper left')
        self.ax[3].legend(fontsize=7, loc='upper right')
        # self.ax[2].legend(loc='upper right')
        # self.ax[3].legend(loc='upper right')
        # self.ax[4].legend(loc='lower right')

        self.ax[2].set_ylim(0.0, 1+0.1)
        self.ax[2].set_xticklabels(["mismatch(x)", "match(o)"])
        plt.show()
        return True


if __name__ == "__main__":

    "修論"
    STATE_HISTORY = [[18, 8], [17, 8], [16, 8], [15, 8], [14, 8], [14, 8], [13, 8], [12, 8], [11, 8], [10, 8], [9, 8], [9, 8], [14, 8], [14, 8], [14, 7], [14, 6], [14, 8], [14, 8], [14, 9], [14, 10], [14, 10], [13, 10], [12, 10], [11, 10], [10, 10], [9, 10], [9, 10], [8, 10], [7, 10], [6, 10], [5, 10], [4, 10], [4, 10], [3, 10], [2, 10], [1, 10], [0, 10], [14, 10], [14, 10], [14, 11], [14, 12], [14, 12], [13, 12], [12, 12], [11, 12], [10, 12], [9, 12], [9, 12], [8, 12], [7, 12], [6, 12], [5, 12], [4, 12], [4, 12], [3, 12], [2, 12], [1, 12], [0, 12], [4, 12], [4, 12], [4, 13], [4, 14]]
    stress = [0, 0, 0, 0, 0, 0.13340000000000002, 0.13340000000000002, 0.13340000000000002, 0.13340000000000002, 0.13340000000000002, 0.8334, 0.8334, 0.8334, 0.8334, 0.8334, 1.3334000000000001, 1.3334000000000001, 1.3334000000000001, 1.3334000000000001, 1.3334000000000001, 1.4000000000000001, 1.4000000000000001, 1.4000000000000001, 1.4000000000000001, 1.4000000000000001, 0, 0.13340000000000002, 0.13340000000000002, 0.13340000000000002, 0.13340000000000002, 0.13340000000000002, 0.6667333333333334, 0.6667333333333334, 0.6667333333333334, 0.6667333333333334, 0.6667333333333334, 0.6667333333333334, 0.6667333333333334, 0.6667333333333334, 0.6667333333333334, 1.0667333333333335, 1.0667333333333335, 1.0667333333333335, 1.0667333333333335, 1.0667333333333335, 1.0667333333333335, 0, 0.2, 0.2, 0.2, 0.2, 0.2, 0, 0.13340000000000002, 0.13340000000000002, 0.13340000000000002, 0.13340000000000002, 0.13340000000000002, 0.13340000000000002, 0.13340000000000002, 0.13340000000000002, 0.13340000000000002]
    standard_list =  [0.0, 0.333, 0.666, 0.9990000000000001, 1.332, 0, 0.22200000000000003, 0.44400000000000006, 0.6660000000000001, 0.8880000000000001, 1.11, 0, 0, 0, 0.22200000000000003, 0.44400000000000006, 0, 0, 0.22200000000000003, 0.44400000000000006, 0, 0.1665, 0.333, 0.49950000000000006, 0.666, 0.8325, 0, 0.13319999999999999, 0.26639999999999997, 0.39959999999999996, 0.5327999999999999, 0.6659999999999999, 0, 0.22200000000000003, 0.44400000000000006, 0.6660000000000001, 0.8880000000000001, 0, 0, 0.1665, 0.333, 0, 0.2664, 0.5328, 0.7992000000000001, 1.0656, 1.332, 0, 0.13319999999999999, 0.26639999999999997, 0.39959999999999996, 0.5327999999999999, 0.6659999999999999, 0, 0.11099999999999999, 0.22199999999999998, 0.33299999999999996, 0.44399999999999995, 0, 0, 0.11099999999999999, 0.22199999999999998]
    rate_list =  [0.5, 0.5, 0.5, 0.5, 0.3333333333333333, 0.3333333333333333, 0.3333333333333333, 0.3333333333333333, 0.3333333333333333, 0.3333333333333333, 0.5, 0.5, 0.3333333333333333, 0.3333333333333333, 0.3333333333333333, 0.5, 0.5, 0.5, 0.3333333333333333, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.3333333333333333, 0.3333333333333333, 0.3333333333333333, 0.3333333333333333, 0.3333333333333333, 0.3333333333333333, 0.25, 0.25, 0.25, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.16666666666666666, 0.16666666666666666, 0.16666666666666666, 0.16666666666666666, 0.16666666666666666, 0.16666666666666666, 0.16666666666666666, 0.16666666666666666, 0.16666666666666666, 0.16666666666666666]





    Env_Anim = Anim(STATE_HISTORY, stress, rate_list, standard_list)

    print("STATE_HISTORY:{}".format(Env_Anim.state_history))
    print("length:{}".format(len(Env_Anim.state_history)))  
    print("length standard:{}".format(len(standard_list)))
    print("length rate:{}".format(len(rate_list)))

    Env_Anim.view_plot_text()
    Env_Anim.move_history(Env_Anim)
    Env_Anim.view_anim()

    