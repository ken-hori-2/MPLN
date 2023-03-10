import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from datetime import datetime


class DEMO():

    def __init__(self, env, i, result_dir):
        self.env = env

        self.BP = {}
        self.i = i
        self.result_dir = result_dir

    
    def show(self, state, map, Backed, DIR, trigar, VIZL, VIZD, STATE_HISTORY):

        viz = True # False
        
        if viz:
            size = -self.env.row_length
            
            "Normal - Large"
            # fig = plt.figure(figsize=(10, 5)) # here
            "修論"
            fig = plt.figure(figsize=(15, 8))

            #add_subplot()でグラフを描画する領域を追加する．引数は行，列，場所
            ax1 = fig.add_subplot(1, 2, 1)
            ax2 = fig.add_subplot(1, 2, 2)

            try:
                ax2.plot(VIZD, 'r-', VIZD, 'k.', alpha = 0.5, label = "Distance to goal")
                # ax2.plot(VIZD, 'r-', alpha = 0.5, label = "Distance to goal")
                ax3 = ax2.twinx()
                ax3.plot(VIZL, 'b-', VIZL, 'k.', alpha = 0.5, label = "Number of Landmarks found")
                # ax3.plot(VIZL, 'b-', alpha = 0.5, label = "Number of Landmarks found")
                h1, l1 = ax2.get_legend_handles_labels()
                h2, l2 = ax3.get_legend_handles_labels()
                ax2.legend(h1+h2, l1+l2, loc='lower right')
                ax2.set_xlabel('Steps')
                ax2.set_ylabel("Distance")
                ax2.grid(True)
                ax3.set_ylabel("Landmark")
                # plt.show()
            except:
                # print("viz エラー")
                pass

            # fig = plt.figure(figsize=(5, 5))

            ax1.set_xlabel('Environment')
            
            ax1.plot([-0.5, -0.5], [0.5, size+0.5], color='k')
            ax1.plot([-0.5, -size-0.5], [size+0.5, size+0.5], color='k')
            ax1.plot([-size-0.5, -0.5], [0.5, 0.5], color='k')
            ax1.plot([-size-0.5, -size-0.5], [size+0.5, 0.5], color='k')
            
            
            import matplotlib.cm as cm
            import matplotlib.colors as colors

            pi = np.pi
            cmap = cm.binary
            cmap_data = cmap(np.arange(cmap.N))
            cmap_data[0, 3] = 0 # 0 のときのα値を0(透明)にする
            customized_gray = colors.ListedColormap(cmap_data)

            dem = [[0.0 for i in range(-size)] for i in range(-size)] #known or unknown
            x, y = np.mgrid[-0.5:-size+0.5:1, -0.5:-size:1]
            # x, y = np.mgrid[-0.5:-size-0.5:1, 0.5:-size+0.5:1]

            demGrid = ax1.pcolor(x, -y, dem, vmax=1, cmap=plt.cm.BrBG, alpha=0.2)
            
            soil = [[0.0 for i in range(-size)] for i in range(-size)] #2Dgridmap(xw, yw)
            
            "Add test-LBM"
            # Node = ["A", "B", "C", "D", "O", "E", "F", "G",     "g"] # , "s"] # here
            Node = ["A", "B", "C", "D", "E", "F", "O", "H", "I", "J", "K", "g"]

            test = [[0.0 for i in range(-size)] for i in range(-size)] #2Dgridmap(xw, yw) # Node
            # test = [[1.0 for i in range(-size)] for i in range(-size)] #2Dgridmap(xw, yw) # Node # here

            if Backed:
                self.BP = Backed
                
            for ix in range(-size):
                for iy in range(-size):   
                    if self.env.grid[ix][iy] == 9:
                        soil[ix][iy] = 1 #sandy terrain
                    else:
                        soil[ix][iy] = 0 #hard ground
                        "Add test-LBM"
                        
                        if self.env.NODELIST[ix][iy] in Node: # Node
                            test[ix][iy] = 0.5 # 1 #sandy terrain
                        
                        if self.env.NODELIST[ix][iy] in self.BP:
                            test[ix][iy] = 1

            test = np.flip(test, 1)
            test = np.rot90(test, k=1)
            # lm = ax1.pcolor(x, -y, test, vmax=1, cmap=ax1.cm.Greens, alpha = 1.0) # マス目が消える
            lm = ax1.pcolor(x, -y, test, vmax=1, cmap=plt.cm.BrBG, alpha = 1.0)
            # lm = ax1.pcolor(x, -y, test, vmax=1, cmap=ax1.cm.RdGy, alpha = 1.0) # here
            
            soil = np.flip(soil, 1)
            soil = np.rot90(soil, k=1)
            # terrain = ax1.pcolor(x, -y, soil, vmax=1, cmap=ax1.cm.Greys, alpha = 0.2)
            terrain = ax1.pcolor(x, -y, soil, vmax=1, cmap=plt.cm.Greys, alpha = 0.5)
            # terrain = ax1.pcolor(x, -y, soil, vmax=1, cmap=ax1.cm.BuPu, alpha = 0.5) # here

            map  = np.flip(map, 1)
            map = np.rot90(map, k=1)
            known = ax1.pcolor(x, -y, map, vmax=1, cmap=customized_gray)
            Node = ["s", "A", "B", "C", "D", "E", "F", "O", "H", "I", "J", "K", "g"]
            if self.env.NODELIST[state.row][state.column] in Node:
                # ax1.plot(state.column, -state.row, ".r", markersize=10)
                # ax1.plot(state.column, -state.row, ".g", markersize=80, alpha = 0.2)
                
                ax1.plot(state.column, -state.row, ".r", markersize=10,     label = "Agent")
                
            elif self.env.NODELIST[state.row][state.column] == "x":
                ax1.plot(state.column, -state.row, "xb", markersize=5,     label = "Agent") # 10) # , alpha = 0.5)
            else:
                # ax1.plot(state.column, -state.row, ".g", markersize=30, alpha = 0.2)

                ax1.plot(state.column, -state.row, ".y", markersize=10,     label = "Agent")

            # Add
            # ax1.plot(state.column, -state.row, ".y", markersize=80, alpha = 0.2)

            try:
                max_dir = max(DIR)
                if state.row+(-DIR[0]+DIR[1])*2/max_dir < 0 or state.column+(-DIR[2]+DIR[3])*2/max_dir < 0 or state.row+(-DIR[0]+DIR[1])*2/max_dir >= -size or state.column+(-DIR[2]+DIR[3])*2/max_dir >= -size:
                    pass
                else:
                    if self.env.NODELIST[state.row][state.column] in Node or self.env.NODELIST[state.row][state.column] == "x":

                        if not trigar and not self.env.NODELIST[state.row][state.column] == "g":
                            # ax1.plot(state.column+(-DIR[2]+DIR[3])*2/max_dir, -state.row+(DIR[0]-DIR[1])*2/max_dir, "*y", markersize=10,     label = "Estimated") # , alpha = 0.5)
                            # ax1.plot([state.column, state.column+(-DIR[2]+DIR[3])*2/max_dir], [-state.row, -state.row+(DIR[0]-DIR[1])*2/max_dir], linestyle = "--", color='y', alpha = 0.5)
                            ax1.plot(state.column+(-DIR[2]+DIR[3])*2/max_dir, -state.row+(DIR[0]-DIR[1])*2/max_dir, marker = "*", color = "orange", markersize=10,     label = "Estimated") # , alpha = 0.5)
                            ax1.plot([state.column, state.column+(-DIR[2]+DIR[3])*2/max_dir], [-state.row, -state.row+(DIR[0]-DIR[1])*2/max_dir], linestyle = "--", color='orange', alpha = 0.5)
            except:
                pass

            
            # "Node, Goalの想定位置"
            # tx = (8, 8, 8, 8, 8)
            # ty = (-15+0.3, -12+0.3, -9+0.3, -6+0.3, -3+0.3)
            # "----- 2d -----"
            # tx = (8, 10, 10, 12, 12) # , 12, 14, 14, 16, 18)
            # ty = (-14+0.3, -14+0.3, -9+0.3, -9+0.3, -4+0.3) # , -4+0.3, -4+0.3)
            # ax1.plot(tx, ty, "*g", markersize=4+2,     label = "Node")

            # goal_x = (8)
            # goal_y = (0.3)
            # "----- 2d -----"
            # goal_x = (14)
            # # goal_x = (4)
            # goal_y = (-4+0.3)
            # ax1.plot(goal_x, goal_y, "*r", markersize=4+2,     label = "Goal")

            
            # ax1.legend(loc='upper right')
            ax1.legend(loc='upper left')


            # png_path = os.path.join(result_dir, "{0}.png".format(ww))
            # plt.savefig(png_path)




            # try:
            #     ax2.plot(VIZD, 'r-', VIZD, 'k.', alpha = 0.5, label = "Distance to goal")
            #     # a=np.arange(len(VIZL))
            #     ax3 = ax2.twinx()
            #     ax3.plot(VIZL, 'b-', VIZL, 'k.', alpha = 0.5, label = "Number of Landmarks found")
            #     # h1, l1 = ax2.get_legend_handles_labels()
            #     # h2, l2 = ax3.get_legend_handles_labels()
            #     # ax2.legend(h1+h2, l1+l2, loc='lower right')
            #     ax2.set_xlabel('Steps')
            #     ax2.set_ylabel("Distance")
            #     ax2.grid(True)
            #     ax3.set_ylabel("Landmark")
            #     # plt.show()
            # except:
            #     print("viz エラー")
            
            "修論"
            # plt.show()
            png_path = os.path.join(self.result_dir, "{}.png".format(len(STATE_HISTORY)))
            plt.savefig(png_path)
            plt.close()
            self.i+=1

    def obserb(self, init, size, map):
        
        init_x, init_y = init.row, init.column

        # Node = ["A", "B", "C", "D", "O", "E", "F", "G",     "g",     "x"]
        # Node = ["A", "B", "C", "D", "O", "E", "F", "G",     "g",     "x",          "s"] # "s"を追加

        # Node = ["A", "B", "C", "D", "E", "F", "O",   "g",     "x",          "s"] # "s"を追加
        Node = ["A", "B", "C", "D", "E", "F", "O", "H", "I", "J", "K", "g", "x"]

        if self.env.NODELIST[init_x][init_y] in Node: #交差点のみ前後一マス観測
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




    def viz(self, viz):
        # fig = ax1.figure(figsize=(3, 3))

        # viz.plot()
        # viz.plot.line(subplots=True, layout=(3, 1), grid=False, figsize=(5+2, 5), style=['-', '--', '-.', ':']) # , sharey=True) # 2, 2))
        
        
        
        
        
        # viz.plot.line(subplots=True, layout=(3, 1), grid=False, figsize=(5, 5), style=['-', '--', '-.', ':'])

        # plt.show()
        pass

    def bp_viz(self, Attribute):
        
        # Attribute[:5].plot.bar()
        # Attribute.plot.bar(subplots=True, layout=(1, 3), grid=False, figsize=(5+2, 5)) # , sharey=True) # 2, 2))
        
        
        
        
        
        # Attribute.plot.bar(subplots=True, layout=(1, 3), grid=False, figsize=(5, 5))
        # self.Attribute.plot.bar()
        
        # plt.show()
        pass