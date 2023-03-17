from pprint import pprint
import numpy as np
from refer import Property
import pprint
import random
from scipy.sparse.csgraph import shortest_path, floyd_warshall, dijkstra, bellman_ford, johnson
from scipy.sparse import csr_matrix
import pandas as pd
import copy
from neural_relu import neural
import math

class Algorithm_advance():
    
    def __init__(self, *arg):
        
        self.state = arg[0] # state
        self.env = arg[1] # env
        self.agent = arg[2] # agent
        self.NODELIST = arg[3] # NODELIST
        self.Observation = arg[4]
        self.refer = Property() # arg[5]
        self.total_stress = 0
        self.stress = 0
        self.Stressfull = 2.0
        self.COUNT = 0
        self.done = False
        self.TRIGAR = False
        self.TRIGAR_REVERSE = False
        self.BACK = False
        self.BACK_REVERSE = False
        self.on_the_way = False
        self.bf = True
        self.STATE_HISTORY = []
        self.BPLIST = []
        self.test_bp_st_pre = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan     , np.nan, np.nan, np.nan, np.nan]
        self.PROB = []
        self.Arc = []
        self.OBS = []
        self.FIRST = True
        self.SAVE_ARC = []
        self.Storage = []
        self.Storage_Stress = []
        self.Storage_Arc = []
        self.DEMO_LIST = []
        self.SIGMA_LIST = []
        self.sigma = 0
        self.test_s = 0
        self.data_node = []
        self.XnWn_list = []
        self.save_s = []
        self.save_s_all = []
        self.End_of_O = False
        self.standard_list = []
        self.rate_list = []
        self.n_m = arg[5]
        self.RATE = arg[6]
        self.test = arg[7]
        self.VIZL = []
        self.VIZD = []
        self.goal = arg[8]
        # self.Node_l = ["s", "O", "A", "B", "C", "D", "E", "F", "g", "x"] # here
        # self.Node_l = ["s", "A", "B", "C", "D", "E", "F", "O", "g", "x"] # here
        self.Node_l = ["s", "A", "B", "C", "D", "E", "F", "O", "H", "I", "J", "K", "g", "x"]
        "-- init --"
        self.old = "s"
        # self.l = {"s":[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #           "A":[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #           "B":[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #           "C":[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #           "D":[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #           "E":[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #           "F":[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #           "O":[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #           "g":[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #           "x":[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}
        self.l = {"s":[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  "A":[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  "B":[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  "C":[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  "D":[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  "E":[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  "F":[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  "O":[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  "H":[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  "I":[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  "J":[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  "K":[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  "g":[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  "x":[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}
        # self.Node = ["s", "A", "B", "C", "D", "E", "F", "O", "g", "x"] # here
        self.Node = ["s", "A", "B", "C", "D", "E", "F", "O", "H", "I", "J", "K", "g", "x"]
        self.l = pd.DataFrame(self.l, index = pd.Index(self.Node))
        self.move_cost_result = []
        self.test_bp_st_pre = pd.Series(self.test_bp_st_pre, index=self.Node_l)
        self.weight = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan     , np.nan, np.nan, np.nan, np.nan]
        self.move_cost = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan     , np.nan, np.nan, np.nan, np.nan]
        self.Attribute = pd.DataFrame({"stress":self.weight,
                    'move cost':self.move_cost,
                    },
                    index=self.Node_l)
        self.Attribute.index.name = "Node"
        # test = self.Attribute.loc["A":"A"]
        "============================================== Visualization ver. との違い =============================================="

    def hierarchical_model_O(self, ΔS): # 良い状態では小さいずれは気にしない(でもそもそも距離のずれは気にする必要ないかも)

        "test-LBM 連続では無いとnを増やさないのは一旦ナシ -> End of O"
        if not self.Backed_just_before: # ここを追加 -> ただ進んだだけで途切れた場合はリセットするが、戻った場合はリセットしない

            "hierarchical_model_Xから移動"
            if self.End_of_O: # 直前までに○の連続が途切れていた場合は一旦リセット
                self.n=1      # resetで0ではなく、1 -> 1/(1+1)=0.5となる
                self.End_of_O = False

        if not self.RETRY:
            self.n += 1
        self.RETRY = False
        
        "----- 階層化-----"
        # "×の連続数は良い状態には用いないので、ここでリセットしても関係ないから大丈夫"
        # self.M=1      # resetで0ではなく、1 -> 1/(1+1)=0.5となる
        # # self.mmm=1    # resetで0ではなく、1 -> 1/(1+1)=0.5となる
        # Wn = np.array([1, -0.1])
        # print("重みWn [w1, w2] : ", Wn)
        # model = neural(Wn)
        # print(f"入力Xn[ΔS, n] : {ΔS}, {self.n}")

        # "===== 何連続から良い状態とするか -> n-?で決定 ====="
        # # neu_fire, XnWn = model.perceptron(np.array([ΔS, self.n-3]), B=0) # Relu関数 これがあるとないとではゴール到達率が違う defalt:n=0
        
        # "----- 0 ----- 今回は3連続で良い状態とした(n-1)"
        # neu_fire, XnWn = model.perceptron(np.array([ΔS, self.n-1]), B=0) # Relu関数 これがあるとないとではゴール到達率が違う defalt:n=0
        # "=============================================="
        # print(f"出力result [n={self.n} : {abs(neu_fire)}]")
        # if neu_fire > 0:
        #     print("🔥発火🔥")
        #     self.save_s.append(round(ΔS-neu_fire, 2))
        #     ΔS = neu_fire
        # else:
        #     print("💧発火しない💧")
        #     self.save_s.append(ΔS)
        #     ΔS = 0
        # self.data_node.append(abs(neu_fire))
        # self.XnWn_list.append(XnWn)
        # print("[result] : ", self.data_node)
        # print("[入力, 出力] : ", self.XnWn_list)
        "----- 階層化 -----"

        return ΔS

    def hierarchical_model_X(self): # 良い状態ではない時に「戻るタイミングは半信半疑」とした時のストレス値の蓄積の仕方

        self.End_of_O = True # ○の連続が途切れたのでTrue
        self.M += 1
        self.Σ = 1 # ×の時に蓄積する量は1.0とした
        self.n2 = copy.copy(self.n)
        
        self.total_stress += self.Σ *1.0* (self.M/(self.M+self.n2)) # n=5,0.2 # ここ main # 階層化 ver.
        "階層化なしver."
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

    def match(self, Node, Arc):

        self.index = Node.index(self.NODELIST[self.state.row][self.state.column]) # これがselfではなかったので更新されなかった
        
        # # print("<{}> match !".format(self.NODELIST[self.state.row][self.state.column]))
        # print("Pre_Arc (事前のArc) : {}".format(Arc[self.index]))
        # print("Act_Arc (実際のArc) : {}".format(self.move_step))
        # print("事前に対する実際のArc[基準距離] : {}".format(self.test_s))
        # # self.SAVE_ARC.append(self.test_s)
        # print(f"Total Stress:{self.total_stress}")

        "========================================================================================================"
        "-- min-cost-cal-edit --"
        self.new = self.NODELIST[self.state.row][self.state.column]
        "-- min-cost-cal-edit --"
        LastNode = self.old # self.Node_l.index(self.old)
        NextNode = self.new # self.Node_l.index(self.new)
        self.old = self.new
        if not self.NODELIST[self.state.row][self.state.column] == "s":
            Act_Arc_data = self.move_step
        else:
            Act_Arc_data = 0
        cost_row = LastNode
        cost_column = NextNode

        if self.l.loc[cost_row, cost_column] == 0 or Act_Arc_data < self.l.loc[cost_row, cost_column]:
            self.l.loc[cost_row, cost_column] = Act_Arc_data

        Landmark = self.NODELIST[self.state.row][self.state.column]
        self.test_bp_st_pre[f"{Landmark}"] = self.state

        try:
            kizyun_d = self.move_step/float(Arc[self.index])
        except:
            kizyun_d = 0

        if kizyun_d != 0:
            "-- これがいずれのΔSnodeの式 今はArc に対するΔSのみ --"
            if kizyun_d > 2:
                kizyun_d = 0.0
            kizyun_d = round(abs(1.0-kizyun_d), 3)
        else:
            # kizyun_d = 0.5 # 0.0 start 地点
            kizyun_d = 0.0 # start 地点
        # print("ΔS_Arc【基準ストレス】 : {}".format(kizyun_d))

        # if not self.NODELIST[self.state.row][self.state.column] == "s":
        #     self.SAVE_ARC.append(round(self.move_step, 2))
        self.move_step = 0

        
        # ΔS = 0.3 # これは蓄積分なので、戻る場所決定には使わない

        maru = ["s", "A", "B", "C", "D", "E", "F", "O", "g"] # この関数はmatchだから意味ない
        if not self.NODELIST[self.state.row][self.state.column] in maru:
            LM = 1.0
        else:
            LM = 0.0

        "----- 属性の追加 -----"
        # # maru = ["x", "O", "A", "B", "C", "D", "E"]
        # Similar = ["A2", "B2", "C2", "D2", "E2"]
        # # if self.NODELIST[self.state.row][self.state.column] in maru:
        # #     LM = 1.0
        # # el
        # if self.NODELIST[self.state.row][self.state.column] in Similar:
        #     LM = 0.5

        D = kizyun_d
        ΔS = 0.8*LM + 0.2*D # 0.5, 0.5だと距離のずれに敏感になり発生量が増える

        if self.n >= 3:
            self.total_stress = 0

        self.save_s_all.append(ΔS)
        ΔS = self.hierarchical_model_O(ΔS) # 関数 これがないとゴール到達率が下がる
        self.Σ = round(sum(self.save_s), 2)
        self.n_m[self.state.row][self.state.column] = (self.n, self.M) # 連続数(n, m)の追加
        self.phi = [self.n, self.M]
        
        if self.Observation[self.state.row][self.state.column] == -1: # 0だと0.0も含まれてしまう
            self.Observation[self.state.row][self.state.column] = round(abs(ΔS), 3)

        # "全部コメントアウトの時はsettingのobservationの数値をそのまま使う"
        # try:
        #     self.OBS.append(self.Observation[self.state.row][self.state.column])
        # except:
        #     self.OBS = self.OBS.tolist()
        #     self.OBS.append(self.Observation[self.state.row][self.state.column])
            
        self.Attribute.loc[f"{Landmark}", "stress"] = self.Observation[self.state.row][self.state.column]
        
        self.Add_Advance = True
        # self.BPLIST.append(self.state)
        # "BPLISTを保存"
        # for bp, stress in zip(self.BPLIST, self.OBS):
        #     if bp not in self.Storage:
        #         self.Storage.append(bp)
        #         self.Storage_Stress.append(stress)
                
        self.STATE_HISTORY.append(self.state)
        self.TOTAL_STRESS_LIST.append(self.total_stress)

        "基準距離, 割合の可視化"
        self.standard_list.append(self.test_s)
        # self.rate_list.append(self.n/(self.M+self.n))    # ○
        self.rate_list.append(self.M/(self.M+self.n))      # ×


        self.VIZL.append(self.L_NUM)
        self.VIZD.append(self.D_NUM)
        self.test_s = 0
        
        "基準距離を可視化に反映させないver.はコメントアウト"
        # self.total_stress = 0
        # self.total_stress += arc_s
        "基準距離を可視化に反映させないver.はコメントアウト +代わりに以下"
        # if not self.NODELIST[self.state.row][self.state.column] == "s": # これはスタート地点にノードを設定している場合、初期位置ではストレスを蓄積させないため
        maru = ["s"] #, "O", "A", "B", "C", "D"]
        if not self.NODELIST[self.state.row][self.state.column] in maru:
            self.total_stress += ΔS # 基準距離を可視化させないver.
        self.SIGMA_LIST.append(self.total_stress)
        
    
    def nomatch(self, Node, Arc):

        judge_node__x = False

        maru = ["O", "A", "B", "C", "D", "g"] # この関数はmatchだから意味ない
        if not self.NODELIST[self.state.row][self.state.column] in maru:
            LM = 1.0
        else:
            LM = 0.0

        # LM = 1.0
        # LM = 0.2
        
        # Node不一致なので基準距離は算出不可=0
        # kizyun_d = 1.0
        D = 0.0 # round(abs(1.0-kizyun_d), 3)

        "----- 属性の追加 -----"
        maru = ["x", "O", "A", "B", "C", "D", "E"]
        # Similar = ["A2", "B2", "C2", "D2", "E2"]
        if self.NODELIST[self.state.row][self.state.column] in maru:
            LM = 1.0
        # elif self.NODELIST[self.state.row][self.state.column] in Similar:
        #     LM = 0.5
        else:
            LM = 0.0
        ΔS = 0.8*LM + 0.2*D # 0.5, 0.5だと距離のずれに敏感になり発生量が増える
        "----- 属性の追加 -----"

        maru = ["x"] #, "O", "A", "B", "C", "D"]

        "Add 0214 観測の不確実性"
        # maru = ["x", "O", "A", "B", "C", "D", "E"]
        if self.NODELIST[self.state.row][self.state.column] in maru:
            self.total_stress += ΔS # 基準距離を可視化させないver.
            self.SIGMA_LIST.append(self.total_stress)

        # if self.grid[self.state.row][self.state.column] == 5:
        #     # print("\n\n\n交差点! 🚥　🚙　✖️")
        #     if self.state not in self.CrossRoad:
        #         # print("\n\n\n未探索の交差点! 🚥　🚙　✖️")
        #         self.CrossRoad.append(self.state)
        #     # print("CrossRoad : {}\n\n\n".format(self.CrossRoad))
        # print("事前情報にないNode!!!!!!!!!!!!")
        if self.NODELIST[self.state.row][self.state.column] in maru: # == "x":
            
            true_or_false = self.hierarchical_model_X()

            self.n_m[self.state.row][self.state.column] = (self.n, self.M) # 連続数(n, m)の追加
            self.phi = [self.n, self.M]
            
            # if self.M/(self.M+self.n) >= 0.5 + self.RATE: # 0.6: # 0.5: # 0.3: # 階層化 ver.
            if self.M/(self.M+self.n) >= 0.5 + self.RATE:
                self.TRIGAR = True
                self.COUNT += 1
                # self.BPLIST.append(self.state)
                self.Add_Advance = True
                judge_node__x = True

            # if self.M > 3: # 3連続で戻る(初期値でM=1だから>3)
            #     self.TRIGAR = True
            #     self.COUNT += 1
            #     self.BPLIST.append(self.state)
            #     self.Add_Advance = True
            #     judge_node__x = True

        
        return judge_node__x

    def threshold(self, pre):

        self.env.mark(self.state, self.TRIGAR) # mdp実装用
        self.TRIGAR = True
        self.COUNT += 1
        # self.BPLIST.append(self.state) # Arcを計算する為に、最初だけ必要
        self.Add_Advance = True
        "============================================== Visualization ver. との違い =============================================="
        # print(f"🤖 State:{self.state}")
        self.STATE_HISTORY.append(self.state)
        self.TOTAL_STRESS_LIST.append(self.total_stress)
        
        "基準距離, 割合の可視化"
        self.standard_list.append(self.test_s)
        # self.rate_list.append(self.n/(self.M+self.n))    # ○
        self.rate_list.append(self.M/(self.M+self.n))      # ×
        
        # self.SAVE_ARC.append(round(self.move_step, 2))
        self.VIZL.append(self.L_NUM)
        self.VIZD.append(self.D_NUM)
        self.new = "x"

        "----- Add 0203 -----" # 多分いらない->実際にこの関数に入るのはNode=xの時かArcでの基準距離のthrを超えた時のみ...つまり事前のNodeと一致していない時
        if self.NODELIST[self.state.row][self.state.column] in pre:
            self.new = self.NODELIST[self.state.row][self.state.column]

        viz = pd.DataFrame({"Arc's Stress":self.standard_list,
                    "Node's Stress":self.TOTAL_STRESS_LIST,
                    "RATE":self.rate_list,
                    })

        try:
            self.test.viz(viz)
        except:
            pass
        
        LastNode = self.Node_l.index(self.old)
        X = self.Node_l.index(self.new)

        Act_Arc_data = self.move_step
        cost_row = self.old # LastNode
        cost_column = self.new # X # NextNode -> "x"
        self.l.loc[cost_row, cost_column] = Act_Arc_data # 戻る場所からNodeまでの距離を一時的に最小値とか関係なく格納する
        self.move_cost_result_X = shortest_path(np.array(self.l), indices=X, directed=False)
        self.move_cost_result = self.l
        self.l.loc[cost_row, cost_column] = 0 # これが重要 戻り始める場所は毎回変わるのでリセットする

    def trigar(self, pre):

        self.env.mark(self.state, self.TRIGAR)
        # self.BPLIST.append(self.state) # Arcを計算する為に、最初だけ必要
        self.Add_Advance = True
        # self.SAVE_ARC.append(round(self.move_step, 2))
        self.new = "x"

        if self.NODELIST[self.state.row][self.state.column] in pre:
            self.new = self.NODELIST[self.state.row][self.state.column]

        LastNode = self.Node_l.index(self.old)
        X = self.Node_l.index(self.new)
        Act_Arc_data = self.move_step
        cost_row = self.old # LastNode
        cost_column = self.new # X # NextNode -> "x"
        self.l.loc[cost_row, cost_column] = Act_Arc_data # 戻る場所からNodeまでの距離を一時的に最小値とか関係なく格納する
        self.move_cost_result_X = shortest_path(np.array(self.l), indices=X, directed=False)
        self.move_cost_result = self.l
        self.l.loc[cost_row, cost_column] = 0 # これが重要 戻り始める場所は毎回変わるのでリセットする

    def Advance(self, STATE_HISTORY, state, TRIGAR, OBS, total_stress, grid, CrossRoad, x, TOTAL_STRESS_LIST, move_step, old_from_exp, move_cost_result, test_bp_st, Backed_just_before, phi, standard_list, rate_list, test_s, RETRY, map_viz_test, pre_action, DIR, VIZL, VIZD, LN, DN, backed, exp_find,     heatmap):
        self.STATE_HISTORY = STATE_HISTORY
        self.state = state
        self.TRIGAR = TRIGAR
        self.grid = grid
        self.total_stress = total_stress # 今はストレス値は共有していないのでいらない
        # self.OBS = OBS
        self.action = random.choice(self.env.actions) # コメントアウト 何も処理されない時はこれが prev action に入る
        self.Add_Advance = False
        self.Backed_just_before = Backed_just_before
        self.phi = phi
        GOAL = False
        self.CrossRoad = CrossRoad
        pre, Node, Arc, Arc_sum, PERMISSION = self.refer.reference()
        self.stress = 0
        self.index = Node.index("s")
        self.TOTAL_STRESS_LIST = TOTAL_STRESS_LIST
        self.standard_list = standard_list
        self.rate_list = rate_list
        self.test_s = test_s
        self.move_step = move_step
        self.old = old_from_exp
        self.RETRY = RETRY
        self.VIZL = VIZL
        self.VIZD = VIZD
        self.L_NUM = LN
        self.D_NUM = DN
        self.backed = backed
        self.exp_find = exp_find
        self.map = map_viz_test
        self.pre_action = None
        self.pre_action = pre_action
        self.DIR = DIR
        "-> main.pyに移動"
        size = self.env.row_length
        states_known = set() #empty set
        for s in self.env.states:
            if self.map[s.row][s.column] == 0:
                states_known.add(s)
        
        if self.Backed_just_before: # 直前で戻っていた場合 これはbp.pyにてself.Backed_just_before = Trueを追加する
            self.n = phi[0]
            self.M = phi[1]
        else: # 初期値
            self.n = phi[0] # 1
            self.M = phi[1] # 1

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

            self.move_step += 1 # here
            self.map_unexp_area = self.env.map_unexp_area(self.state)

            if self.map_unexp_area or self.FIRST     or self.NODELIST[self.state.row][self.state.column] == "g":

                    self.FIRST = False
                    # print("un explore area ! 🤖 ❓❓")
                    if self.test_s + self.stress >= 0:

                        # 蓄積量(傾き)
                        ex = (self.n/(self.n+self.M))
                        ex = -2*ex+2
                        "----- Add ----"
                        # ex = 1.0 # 蓄積量の階層化は一旦ナシ

                        # print("\n===== test_s[基準距離]:", self.test_s)
                        try:
                            self.test_s += round(self.stress/float(Arc[self.index-1]), 3) *ex # here 元々はこっち
                            # self.test_s = round(self.move_step/float(Arc[self.index-1]), 3) *ex # これでも同じ結果..."s"がないとおかしくなる -> やっぱりこれはダメ, Nodeでしかmove_step=0にならない
                            "基準距離を可視化に反映させないver.はコメントアウト"
                        except:
                            self.test_s += 0
                            "基準距離を可視化に反映させないver.はコメントアウト"

                        # print("Arc to the next node : {}".format(Arc[self.index-1]))

                    if self.NODELIST[self.state.row][self.state.column] in pre:

                        rand = random.randint(0, 10)
                        # print("観測の不確実性 prob : {}".format(rand))
                        # print("exp find : {}".format(self.exp_find))
                        if rand > 1 or self.exp_find:
                        # if rand >= 0 or self.exp_find:
                            # print("🪧 NODE : ⭕️")
                            # print("<{}> match !".format(self.NODELIST[self.state.row][self.state.column]))
                            if self.pre_action == self.env.actions[0]:
                                self.DIR[0] += 1
                            elif self.pre_action == self.env.actions[1]:
                                self.DIR[1] += 1
                            elif self.pre_action == self.env.actions[2]:
                                self.DIR[2] += 1
                            elif self.pre_action == self.env.actions[3]:
                                self.DIR[3] += 1

                            if not self.NODELIST[self.state.row][self.state.column] in self.backed:
                                self.L_NUM += 1
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
                                self.move_cost_result_X = None

                                "----- Add -----"
                                self.test.show(self.state, self.map, {}, self.DIR,     self.TRIGAR,     self.VIZL, self.VIZD,     self.STATE_HISTORY)


                                break

                            "----- 修論発表 -----"
                            if not self.NODELIST[self.state.row][self.state.column] == "s": # here
                                self.match(Node, Arc)
                                
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
                        judge_node__x = self.nomatch(Node, Arc)

                        if judge_node__x:
                            # print("FULL ! MAX! 🔙⛔️")
                            self.threshold(pre)
                            self.test.show(self.state, self.map, {}, self.DIR,     self.TRIGAR,     self.VIZL, self.VIZD,     self.STATE_HISTORY)
                            
                            break

                    
                    if self.test_s >= 2.0: # 基準距離で判断 階層化ver.
                    # if self.test_s >= 2.0 or self.total_stress >= 2.0: # row ver.
                        # print("FULL ! MAX! 🔙⛔️")
                        self.threshold(pre)
                        self.test.show(self.state, self.map, {}, self.DIR,     self.TRIGAR,     self.VIZL, self.VIZD,     self.STATE_HISTORY)
                        break
            else:
                # print("================\n🤖 何も処理しませんでした__2\n================")
                # print("マーキング = 1 の探索済みエリア")
                pass
                
            # print(f"🤖 State:{self.state}")
            self.STATE_HISTORY.append(self.state)
            self.TOTAL_STRESS_LIST.append(self.total_stress)
            self.test.show(self.state, self.map, {}, self.DIR,     self.TRIGAR,     self.VIZL, self.VIZD,     self.STATE_HISTORY)

            "基準距離, 割合の可視化"
            self.standard_list.append(self.test_s)
            # self.rate_list.append(self.n/(self.M+self.n)) # ○
            self.rate_list.append(self.M/(self.M+self.n))   # ×
            self.VIZL.append(self.L_NUM)
            self.VIZD.append(self.D_NUM)

            # self.action, self.Reverse, self.TRIGAR = self.agent.policy_advance(self.state, self.TRIGAR, self.action)
            self.action, self.Reverse, self.TRIGAR = self.agent.mdp(self.state, self.TRIGAR, self.action,     states_known, self.map, self.grid, self.DIR,     self.VIZL, self.VIZD, self.STATE_HISTORY)
            self.pre_action = self.action

            if self.TRIGAR:

                # print("Trigar")
                # print("ストレスが溜まり切る前にこれ以上進めない")
                self.trigar(pre)
                break


            self.next_state, self.stress, self.done = self.env.step(self.state, self.action, self.TRIGAR)
            self.prev_state = self.state # 1つ前のステップを保存 -> 後でストレスの減少に使う
            self.state = self.next_state

            heatmap[self.state.row][self.state.column] += 1

            self.COUNT += 1

        return self.total_stress, self.STATE_HISTORY, self.state, self.TRIGAR, self.OBS, self.BPLIST, self.action, self.Add_Advance, GOAL, self.SAVE_ARC, self.CrossRoad, self.Storage, self.Storage_Stress, self.TOTAL_STRESS_LIST, self.move_cost_result, self.test_bp_st_pre, self.move_cost_result_X, self.standard_list, self.rate_list, self.map, self.Attribute, self.Observation, self.DIR, self.VIZL, self.VIZD, self.L_NUM, self.D_NUM,     heatmap