from pprint import pprint
import numpy as np
import pprint
from refer import Property
import copy
import random
import pandas as pd
import math


class Algorithm_exp():

    def __init__(self, *arg):
        
        self.state = arg[0] # state
        self.env = arg[1] # env
        self.agent = arg[2] # agent
        self.NODELIST = arg[3] # NODELIST
        self.Observation = arg[4]
        self.refer = Property()
        self.total_stress = 0
        self.stress = 0
        self.Stressfull = 2.0
        self.COUNT = 0
        self.done = False
        self.TRIGAR = False
        self.STATE_HISTORY = []
        self.bp_end = False
        self.test_s = 0
        self.n_m = arg[5]
        self.RATE = arg[6]
        self.test = arg[7]
        self.goal = arg[8]

    def hierarchical_model_X(self): # 良い状態ではない時に「戻るタイミングは半信半疑」とした時のストレス値の蓄積の仕方

        self.End_of_O = True # ○の連続が途切れたのでTrue
        self.M += 1
        self.Σ = 1
        self.total_stress += self.Σ *1.0* (self.M/(self.M+self.n)) # n=5,0.2 # ここ main
        # self.total_stress += self.Σ # row
        self.STATE_HISTORY.append(self.state)
        self.TOTAL_STRESS_LIST.append(self.total_stress)

        "基準距離, 割合の可視化"
        self.standard_list.append(self.test_s)
        # self.rate_list.append(self.n/(self.M+self.n))    # ○
        self.rate_list.append(self.M/(self.M+self.n))      # ×


        self.VIZL.append(self.L_NUM)
        self.VIZD.append(self.D_NUM)
        self.test_s = 0
        
        return True
    
    def match(self):
        self.exp_find = True
        # print("\n============================\n🤖 🔛　アルゴリズム切り替え\n============================")

    def nomatch(self, test, DIR):
        # if self.grid[self.state.row][self.state.column] == 5:
        #     # print("===== 交差点! 🚥　🚙　✖️ =====")
        #     if self.state not in self.CrossRoad:
        #         # print("===== 未探索の交差点! 🚥　🚙　✖️ =====")
        #         self.CrossRoad.append(self.state)
        # print("CrossRoad : {}".format(self.CrossRoad))
        # print("事前情報にないNode!!!!!!!!!!!!")

        maru = ["x"] # , "O", "A", "B", "C", "D", "E"]

        if self.NODELIST[self.state.row][self.state.column] in maru: # == "x":
            true_or_false = self.hierarchical_model_X()
            # if self.M/(self.M+self.n) >= 0.5 + self.RATE: # 0.5: # here
            if self.M/(self.M+self.n) >= 0.5 + self.RATE:

                "----- Add -----"
                self.TRIGAR = True
                self.test.show(self.state, self.map, {}, DIR,     self.TRIGAR,     self.VIZL, self.VIZD,     self.STATE_HISTORY)

                self.threshold()

    def threshold(self):
        # self.TRIGAR = True # 上に移動
        # print("FULL ! MAX! 🔙⛔️", self.retry_num, self.rrr)

        self.env.mark(self.state, self.TRIGAR)

        viz = pd.DataFrame({"Arc's Stress":self.standard_list,
                    "Node's Stress":self.TOTAL_STRESS_LIST,
                    "RATE":self.rate_list,
                    })
        try:
            self.test.viz(viz)
        except:
            pass
        
        "----- Add 2D Back x-----"
        # リトライ一回以上で直近のxに戻る -> xに5回戻っても次のNodeが見つけられない時はoのNodeに戻る
        if self.retry_num >=1: # here リトライ一回以上
            if self.rrr < 5:
                # print("NODE POSITION x :", self.NODE_POSITION_x)
                self.state = self.NODE_POSITION_x
            else:
                self.state = self.NODE_POSITION
            self.rrr += 1
        else:
            self.state = self.NODE_POSITION # here
        "----- Add 2D Back x-----"
        # self.state = self.NODE_POSITION # here
        
        # print(f"🤖 State:{self.state}")
        # self.total_stress = 0
        self.STATE_HISTORY.append(self.state)
        self.TOTAL_STRESS_LIST.append(self.total_stress)
        
        "基準距離, 割合の可視化"
        self.standard_list.append(self.test_s)
        # self.rate_list.append(self.n/(self.M+self.n))    # ○
        self.rate_list.append(self.M/(self.M+self.n))      # ×

        self.VIZL.append(self.L_NUM)
        self.VIZD.append(self.D_NUM)

        self.test_s = 0
        self.move_step = 0
        self.Backed_just_before = True

    def lost_state(self):
        self.TRIGAR = True
        # print("LOST! 🔙⛔️", self.retry_num, self.rrr)

        self.env.mark(self.state, self.TRIGAR)
        
        "----- Add 2D Back x-----"
        if self.retry_num >=1:
            if self.rrr < 5:
                # print("NODE POSITION x :", self.NODE_POSITION_x)
                self.state = self.NODE_POSITION_x
            else:
                self.state = self.NODE_POSITION
            self.rrr += 1
        else:
            self.state = self.NODE_POSITION # here
        "----- Add 2D Back x-----"
        # self.state = self.NODE_POSITION # here

        self.STATE_HISTORY.append(self.state)
        self.TOTAL_STRESS_LIST.append(self.total_stress)
        # print(f"🤖 State:{self.state}")

        "基準距離, 割合の可視化"
        self.standard_list.append(self.test_s)
        # self.rate_list.append(self.n/(self.M+self.n))    # ○
        self.rate_list.append(self.M/(self.M+self.n))      # ×

        self.VIZL.append(self.L_NUM)
        self.VIZD.append(self.D_NUM)

        # self.total_stress = 0
        self.test_s = 0
        self.move_step = 0

    def all_explore(self, Returned_state):
        self.env.mark_all(Returned_state)
        self.All_explore = False
        # self.total_stress = 0
        self.move_step = 0
        # self.old_to_advance = self.NODELIST[self.state.row][self.state.column]
        
    def Explore(self, STATE_HISTORY, state, TRIGAR, total_stress, grid, CrossRoad, x, TOTAL_STRESS_LIST, Backed_just_before, standard_list, rate_list, map_viz_test, DIR, VIZL, VIZD, LN, DN,     heatmap): # , PERMISSION):

        self.STATE_HISTORY = STATE_HISTORY
        self.state = state
        self.TRIGAR = False # TRIGAR
        pre, Node, Arc, Arc_sum, PERMISSION = self.refer.reference()
        self.NODE_POSITION = state
        self.lost = False
        self.grid = grid
        self.CrossRoad = CrossRoad
        GOAL = False
        self.total_stress = total_stress
        self.stress = 0
        index = Node.index("s")
        self.TOTAL_STRESS_LIST = TOTAL_STRESS_LIST
        self.standard_list = standard_list
        self.rate_list = rate_list
        self.VIZL = VIZL
        self.VIZD = VIZD
        self.L_NUM = LN
        self.D_NUM = DN
        self.exp_find = False
        self.test_s = 0
        self.move_step = 0
        self.old_to_advance = self.NODELIST[self.state.row][self.state.column]
        self.Backed_just_before = Backed_just_before
        self.retry_num = x # here
        self.rrr = 0 # here
        self.NODE_POSITION_x = state # here
        self.index = Node.index(self.NODELIST[self.state.row][self.state.column])
        self.map = map_viz_test
        self.pre_action = None
        "->main.pyに移動"

        size = self.env.row_length
        x, y = np.mgrid[-0.5:size+0.5:1,-0.5:size+0.5:1]
        states_known = set() #empty set
        for s in self.env.states:
            if self.map[s.row][s.column] == 0:
                states_known.add(s)
        "----- Add -----"

        while not self.done:

            self.start = self.state
            dist = math.sqrt((self.goal.row-self.start.row)**2+(self.goal.column-self.start.column)**2)
            self.D_NUM = dist

            "----- Add -----"
            self.map = self.test.obserb(self.state, size, self.map)
            
            try:
                states_known = set() #empty set
                for s in self.env.states:
                        if self.map[s.row][s.column] == 0:
                            states_known.add(s)
            except AttributeError:
            # except:
                break

            if self.Backed_just_before: # 直前で戻っていた場合 これはbp.pyにてself.Backed_just_before = Trueを追加する
                __a = self.n_m[self.state.row][self.state.column] # -> ここは戻る場所決定で決めた場所を代入というか戻った後はこの関数に入るので現在地を代入
                # print(f"🤖 State:{self.state}")
                try:
                    self.n = __a[0] # nを代入
                    self.M = __a[1] # mを代入
                # except AttributeError: # here
                #     print("Error!")
                #     # break
                except:
                    pass
                self.phi = [self.n, self.M]
                self.Backed_just_before = False

            # if not self.crossroad:
            self.map_unexp_area = self.env.map_unexp_area(self.state)
            # if self.map_unexp_area:
            if self.map_unexp_area or self.NODELIST[self.state.row][self.state.column] == "g":
                # print("un explore area ! 🤖 ❓❓")
                
                # if self.total_stress + self.stress >= 0:
                if self.test_s + self.stress >= 0:
                    
                    # 蓄積量(傾き)
                    ex = (self.n/(self.n+self.M))
                    ex = -2*ex+2

                    "----- Add ----"
                    # ex = 1.0 # 蓄積量の階層化は一旦ナシ

                    try:
                        # self.test_s += round(self.stress/float(Arc[index-1]), 3) # 2)
                        self.test_s += round(self.stress/float(Arc[self.index-1]), 3) *ex # here 元々はこっち

                        if not self.NODELIST[self.state.row][self.state.column] in pre:
                            self.move_step += 1
                    # except:
                    except Exception as e:
                        # print('=== エラー内容 ===')
                        # print('type:' + str(type(e)))
                        # print('args:' + str(e.args))
                        # print('message:' + e.message)
                        # print('e自身:' + str(e))
                        self.test_s += 0
                        # self.move_step += 0

                    # print("Arc to the next node : {}".format(Arc[index-1]))

                if self.NODELIST[self.state.row][self.state.column] in pre:

                    rand = random.randint(0, 10)

                    # print("観測の不確実性 prob : {}".format(rand))

                    if rand > 1: # 0.8の確率で発見
                    # if rand >= 0:
                        # print("🪧 NODE : ⭕️")
                        
                        if self.NODELIST[self.state.row][self.state.column] == "g":
                            # print("🤖 GOALに到達しました。")
                            GOAL = True
                            self.STATE_HISTORY.append(self.state)
                            self.TOTAL_STRESS_LIST.append(self.total_stress)

                            "基準距離, 割合の可視化"
                            self.standard_list.append(self.test_s)
                            # self.rate_list.append(self.n/(self.M+self.n))    # ○
                            self.rate_list.append(self.M/(self.M+self.n))      # ×
                            self.VIZL.append(self.L_NUM)
                            self.VIZD.append(self.D_NUM)

                            "----- Add -----"
                            self.test.show(self.state, self.map, {}, DIR,     self.TRIGAR,     self.VIZL, self.VIZD,     self.STATE_HISTORY)
                            
                            break

                        self.match()

                        break # Advanceに移行
                    else:
                            # 0.2の確率で見落とした場合
                            # print(" ⚠️ 👀 見落としました!")
                            # print("🪧 NODE : ❌")
                            judge_node__x = self.nomatch(Node, Arc)
                            if judge_node__x:
                                # print("FULL ! MAX! 🔙⛔️")
                                self.threshold(pre)
                                self.test.show(self.state, self.map, {}, self.DIR,     self.TRIGAR,     self.VIZL, self.VIZD,     self.STATE_HISTORY)
                                break
                else:
                    
                    # print("🪧 NODE : ❌")
                    # print("no match!")

                    self.nomatch(self.test, DIR)

                    
            if self.NODELIST[self.state.row][self.state.column] == "x": # here
                self.NODE_POSITION_x = self.state

            # print("\n===== test_s[基準距離]:", self.test_s)
            # if self.total_stress >= self.Stressfull: # or self.M/(self.M+self.n) >= 0.5:
            if self.test_s >= 2.0:
                
                "----- Add -----"
                self.test.show(self.state, self.map, {}, DIR,     self.TRIGAR,     self.VIZL, self.VIZD,     self.STATE_HISTORY)
                
                self.threshold()


                "----- Add -----"
                self.test.show(self.state, self.map, {}, DIR,     self.TRIGAR,     self.VIZL, self.VIZD,     self.STATE_HISTORY)

                continue


            self.STATE_HISTORY.append(self.state)
            self.TOTAL_STRESS_LIST.append(self.total_stress)

            "基準距離, 割合の可視化"
            self.standard_list.append(self.test_s)
            # self.rate_list.append(self.n/(self.M+self.n))    # ○
            self.rate_list.append(self.M/(self.M+self.n))      # ×

            self.VIZL.append(self.L_NUM)
            self.VIZD.append(self.D_NUM)

            
            self.test.show(self.state, self.map, {}, DIR,     self.TRIGAR,     self.VIZL, self.VIZD,     self.STATE_HISTORY)
            if self.Backed_just_before:
                __a = self.n_m[self.state.row][self.state.column] # -> ここは戻る場所決定で決めた場所を代入というか戻った後はこの関数に入るので現在地を代入
                # print(f"🤖 State:{self.state}")
                try:
                    self.n = __a[0] # nを代入
                    self.M = __a[1] # mを代入
                # except AttributeError:
                #     print("Error!")
                #     # break
                except: # here
                    pass
                self.phi = [self.n, self.M]
                self.Backed_just_before = False
            
            # self.action, self.bp_end, self.All_explore, self.TRIGAR, self.Reverse, self.lost = self.agent.policy_exp(self.state, self.TRIGAR)
            self.action, self.bp_end, self.All_explore, self.TRIGAR, self.Reverse, self.lost = self.agent.mdp_exp(self.state, self.TRIGAR,     states_known, self.map, self.grid, DIR,     self.VIZL, self.VIZD, self.STATE_HISTORY)
            
            self.pre_action = self.action

            "----- Add -----"
            # self.test.show(self.state, self.map, {}, DIR,     self.TRIGAR,     self.VIZL, self.VIZD,     self.STATE_HISTORY)


            if self.lost:
                self.lost_state()
                "----- Add -----"
                self.test.show(self.state, self.map, {}, DIR,     self.TRIGAR,     self.VIZL, self.VIZD,     self.STATE_HISTORY)
                
            # print("All explore : {}".format(self.All_explore))
            if self.All_explore:
                self.all_explore(state)
                break
            
            if not self.lost:
                # self.next_state, self.stress, self.done = self.env._move(self.state, self.action, self.TRIGAR)
                self.next_state, self.stress, self.done = self.env.step(self.state, self.action, self.TRIGAR)
                self.prev_state = self.state # 1つ前のステップを保存 -> 後でストレスの減少に使う
                self.state = self.next_state
                heatmap[self.state.row][self.state.column] += 1
            else:
                self.lost = False

            self.COUNT += 1
            
        if self.done:
            pass

        return self.total_stress, self.STATE_HISTORY, self.state, self.TRIGAR, self.CrossRoad, GOAL, self.TOTAL_STRESS_LIST, self.move_step, self.old_to_advance, self.phi, self.standard_list, self.rate_list, self.test_s, self.map, self.pre_action, self.VIZL, self.VIZD, self.L_NUM, self.D_NUM, self.exp_find,     heatmap