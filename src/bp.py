
import math
from refer import Property
import copy
import pprint
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import shortest_path, floyd_warshall, dijkstra, bellman_ford, johnson
import math

class Algorithm_bp():

    def __init__(self, *arg):
        
        self.state = arg[0] # state
        self.env = arg[1] # env
        self.agent = arg[2] # agent
        self.NODELIST = arg[3] # NODELIST
        self.Observation = arg[4]
        self.refer = Property()
        self.total_stress = 0
        self.stress = 0
        self.Stressfull = 8 # 10 # 4
        self.COUNT = 0
        self.done = False
        self.TRIGAR = False
        # self.TRIGAR_REVERSE = False
        self.BACK = False
        # self.BACK_REVERSE = False
        # self.on_the_way = False
        self.bf = True
        self.STATE_HISTORY = []
        # self.BPLIST = []
        # self.PROB = []
        # self.Arc = []
        self.OBS = []
        # self.Storage_Arc = []
        # self.SAVE = []
        self.goal = arg[7]
        # self.Node_l = ["s", "A", "B", "C", "D", "E", "F", "O", "g", "x"] # here
        self.Node_l = ["s", "A", "B", "C", "D", "E", "F", "O", "H", "I", "J", "K", "g", "x"]
        self.backed = []
        self.Unbacked = self.Node_l
        self.n_m = arg[5]
        self.test = arg[6]

    def move_cost_cal(self):
        
        self.move_cost_result_copy[self.move_cost_result_copy == np.inf] = np.nan
        self.demo = copy.copy(self.move_cost_result_copy)
        self.move_cost_result_copy = pd.Series(self.move_cost_result_copy, index=self.Node_l) # index=self.Unbacked)
        self.move_cost_result_copy.dropna(inplace=True)
        
        try:
            self.move_cost_result_copy.drop(index=["x"], inplace=True)
        except:
            pass


        self.move_cost_result_copy.drop(index=self.backed, inplace=True) # mv_copyはXの行成分抽出 = npの配列 -> 再度pandasでindex追加しているのでindexのみ削除で大丈夫
        self.test_bp_st.dropna(inplace=True)
        self.test_bp_st.drop(index=self.backed, inplace=True)

    def next_position_decision(self):
        
        self.next_attribute = self.agent.back_position(self.test_bp_st, self.Attribute)
        self.Backed_Node = self.next_attribute.index[0] # next_lm.index[0]
        # print("Backed Node : ", self.Backed_Node)

    def Finished_returning(self, Node):
        __a = self.n_m[self.state.row][self.state.column] # -> ここは戻る場所決定で決めた場所を代入というか戻った後はこの関数に入るので現在地を代入
        self.n = __a[0] # nを代入
        self.M = __a[1] # mを代入
        self.phi = [self.n, self.M]
        self.backed.append(self.Backed_Node)
        self.Unbacked = [i for i in Node if i not in self.backed]
        self.test_bp_st_copy = copy.copy(self.test_bp_st_pre)
        self.test_bp_st_copy.dropna(inplace=True)
        self.test_bp_st_copy.drop(index=self.backed, inplace=True)
        
        "Add 1116 一旦printして可視化するためだけのもの 今は下の方でやっているが、こっちで可視化用のcopy2でやってもいい"
        self.move_cost_result_copy2 = copy.copy(self.move_cost_result)
        self.move_cost_result_copy2 = pd.Series(self.move_cost_result_copy2, index=self.Node_l) # index=self.Unbacked)
        self.move_cost_result_copy2.drop(index=self.backed, columns=self.backed, inplace=True)
        self.move_cost_result_copy2[self.move_cost_result_copy2 == np.inf] = np.nan
        self.move_cost_result_copy2.dropna(inplace=True)
        self.move_cost_result_copy2.drop(index=["x"], inplace=True)
        # print("mv_copy2 : ", self.move_cost_result_copy2)
        # print("mv_copy : ", self.move_cost_result_copy)
        " ↑ or ↓ "
        # "----- これだとinplace=Trueでも、この後アルゴリズムを抜けるのでこの結果はリセットされてしまい反映されない -----"
        # self.move_cost_result_copy.drop(index=self.backed, inplace=True) # mv_copyはXの行成分抽出 = npの配列 -> 再度pandasでindex追加しているのでindexのみ削除で大丈夫
        # "-> エラー 上でindexのdropを既に削除しているのでエラーになる"
        # print("mv_copy drop backed: ", self.move_cost_result_copy)
        # print("mv_copy : ", self.move_cost_result_copy)
        "-----------------------------------------------------------------------------------------------"

        self.Attribute = self.agent.back_end(self.Attribute, self.next_attribute)
        self.BACK =True

        # self.total_stress = 0
        "⚠️ 要検討 ⚠️ 戻った時にどのくらい減少させるか test_s = 進んだ分だけ減少させるか = その場所までのストレスまで減少させるか"
        # print("⚠️ total : {}".format(self.total_stress))
        delta_s = self.Observation[self.state.row][self.state.column]
        delta_s = round(abs(1.0-delta_s), 3)
        if delta_s > 2:
            delta_s = 1.0
        
        # if self.total_stress - delta_s >= 0:
        #     self.total_stress -= delta_s
        # else:
        #     self.total_stress = 0
        
        self.STATE_HISTORY.append(self.state)
        self.TOTAL_STRESS_LIST.append(self.total_stress)

        "基準距離, 割合の可視化"
        self.test_s = 0
        self.standard_list.append(self.test_s)
        # self.rate_list.append(self.n/(self.M+self.n))    # ○
        self.rate_list.append(self.M/(self.M+self.n))      # ×


        self.VIZL.append(self.L_NUM)
        self.VIZD.append(self.D_NUM)
        self.TRIGAR = False
        # self.TRIGAR_REVERSE = False

    def BP(self, STATE_HISTORY, state, TRIGAR, OBS, BPLIST, action, Add_Advance, total_stress, SAVE_ARC, TOTAL_STRESS_LIST, move_cost_result, test_bp_st_pre, move_cost_result_X, standard_list, rate_list, map_viz_test, Attribute, VIZL, VIZD, LN, DN,     heatmap):
        self.STATE_HISTORY = STATE_HISTORY
        self.state = state
        self.TRIGAR = TRIGAR
        # self.OBS = OBS
        # self.BPLIST = BPLIST
        # self.Advance_action = action
        self.bf = True
        self.state_history_first = True
        self.Add_Advance = Add_Advance
        self.Backed_just_before = False
        self.total_stress = total_stress
        # self.SAVE_ARC = SAVE_ARC
        # self.first_pop = True
        self.BackPosition_finish = False
        pre, Node, Arc, Arc_sum, PERMISSION = self.refer.reference()
        self.TOTAL_STRESS_LIST = TOTAL_STRESS_LIST
        self.standard_list = standard_list
        self.rate_list = rate_list
        self.VIZL = VIZL
        self.VIZD = VIZD
        self.L_NUM = LN
        self.D_NUM = DN
        self.move_cost_result_pre = move_cost_result # self.l
        self.test_bp_st_pre = test_bp_st_pre
        self.test_bp_st = copy.copy(self.test_bp_st_pre)
        X = self.Node_l.index("x") # self.new)
        self.move_cost_result = move_cost_result_X # shortest_path(self.move_cost_result_pre, indices=X, directed=False) # bpで使う
        self.move_cost_result_copy = copy.deepcopy(self.move_cost_result)
        self.Attribute = Attribute
        self.map = map_viz_test
        size = self.env.row_length
        x, y = np.mgrid[-0.5:size+0.5:1,-0.5:size+0.5:1]
        states_known = set() #empty set
        for s in self.env.states:
            if self.map[s.row][s.column] == 0:
                states_known.add(s)
        
        while not self.done:

            self.start = self.state
            dist = math.sqrt((self.goal.row-self.start.row)**2+(self.goal.column-self.start.column)**2)
            self.D_NUM = dist
            self.map = self.test.obserb(self.state, size, self.map)
            
            try:
                states_known = set() #empty set
                for s in self.env.states:
                        if self.map[s.row][s.column] == 0:
                            states_known.add(s)
            except AttributeError:
            # except:
                break

            

            "============================================== Visualization ver. との違い =============================================="
            "戻る行動の可視化ver.の場合はここにReverseが入る"
            "============================================== Visualization ver. との違い =============================================="

            if self.BACK or self.bf:
                    try:
                        
                        if self.bf: # ストレスが溜まってから初回
                            
                            if self.Add_Advance:

                                # ユークリッド距離
                                # self.Arc = [math.sqrt((self.BPLIST[-1].row - self.BPLIST[x].row) ** 2 + (self.BPLIST[-1].column - self.BPLIST[x].column) ** 2) for x in range(len(self.BPLIST))]

                                self.move_cost_cal()

                            self.Attribute["move cost"] = self.move_cost_result_copy
                        else:
                            pass
                        self.bf = False
                        self.BACK = False
                        
                        self.next_position_decision()

                        try:
                            self.test.bp_viz(self.Attribute, STATE_HISTORY)
                        except:
                            pass
                        
                        NP = self.next_attribute["STATE"][0]
                        NP = self.next_attribute["STATE"]
                        # print(f"========Decision Next State=======\n⚠️  NEXT POSITION:\n{NP}\n==================================")
                        # self.on_the_way = True 
                    except:
                    # except Exception as e:
                    #     print('=== エラー内容 ===')
                    #     print('type:' + str(type(e)))
                    #     print('args:' + str(e.args))
                    #     print('message:' + e.message)
                    #     print('e自身:' + str(e))
                        # print("ERROR!")
                        # print("リトライ行動終了！")
                        # print(" = 戻り切った状態 🤖🔚")
                        self.BackPosition_finish = True
                        break
            try:

                if self.state == self.next_attribute["STATE"][0]:
                    # print("===== back end =====")
                    
                    self.Backed_just_before = True

                    self.Finished_returning(Node)
                    "----- Add -----"
                    self.test.show(self.state, self.map, self.backed, {},     self.TRIGAR,     self.VIZL, self.VIZD,     self.STATE_HISTORY)
                    
                    break

                else:
                    # if self.on_the_way:
                    #     self.on_the_way = False
                    # else:
                    #     # print("🔛 On the way BACK")
                    #     pass
                    pass
            except:
            # except Exception as e:
            #         print('=== エラー内容 ===')
            #         print('type:' + str(type(e)))
            #         print('args:' + str(e.args))
            #         print('message:' + e.message)
            #         print('e自身:' + str(e))
                    # print("state:{}".format(self.state))
                    # print("これ以上戻れません。 終了します。")
                    break # expansion 無しの場合は何回も繰り返さない
                
            # print(f"🤖 State:{self.state}")
            if not self.state_history_first:
                self.STATE_HISTORY.append(self.state)
                self.TOTAL_STRESS_LIST.append(self.total_stress)

                "基準距離, 割合の可視化"
                self.test_s = 0
                self.standard_list.append(self.test_s)
                # self.rate_list.append(self.n/(self.M+self.n))    # ○
                self.rate_list.append(self.M/(self.M+self.n))      # ×


                self.VIZL.append(self.L_NUM)
                self.VIZD.append(self.D_NUM)

                "----- Add -----"
                self.test.show(self.state, self.map, self.backed, {},     self.TRIGAR,     self.VIZL, self.VIZD,     self.STATE_HISTORY)

            
            self.state_history_first = False
            self.state = self.next_attribute["STATE"][0]

            heatmap[self.state.row][self.state.column] += 1
            self.COUNT += 1
        # self.COUNT = 0

        return self.total_stress, self.STATE_HISTORY, self.state, self.OBS, self.BackPosition_finish, self.TOTAL_STRESS_LIST, self.move_cost_result_pre, self.test_bp_st_pre, self.Backed_just_before, self.standard_list, self.rate_list, self.map, self.VIZL, self.VIZD, self.L_NUM, self.D_NUM, self.backed,     heatmap