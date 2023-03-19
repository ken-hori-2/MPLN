from cgi import test
from tkinter.messagebox import NO
import numpy as np
from sklearn import preprocessing
import random
from refer import Property
import math
# from Lost_Action import Agent_actions
import pandas as pd

class Agent():

    def __init__(self, env, marking_param, *arg):
        self.env = env
        self.actions = env.actions
        self.GOAL_REACH_EXP_VALUE = 50 # max_theta # 50
        self.lost = False
        self.test = False
        self.grid = arg[0]
        self.map = arg[1]
        self.NODELIST = arg[2]
        self.refer = Property()
        self.marking_param = marking_param
        # self.Node_l = ["s", "A", "B", "C", "D", "E", "F", "O", "g", "x"] # here
        self.Node_l = ["s", "A", "B", "C", "D", "E", "F", "O", "H", "I", "J", "K", "g", "x"]
        self.result_dir = arg[4]

    def policy_advance(self, state, TRIGAR, action):
        
        self.TRIGAR_advance = TRIGAR
        self.prev_action = action
        action = self.model_advance(state)
        self.Advance_action = action
        if action == None:
            return self.prev_action, self.Reverse, self.TRIGAR_advance # このprev action も仮
            
        return action, self.Reverse, self.TRIGAR_advance

    def mdp(self, state, TRIGAR, action,     states_known, map, grid, DIR,     VIZL, VIZD, STATE_HISTORY):
        self.TRIGAR_advance = TRIGAR
        self.VIZL = VIZL
        self.VIZD = VIZD
        y_n = False
        self.All = False
        self.Reverse = False
        # from mdp_virtual import ValueIterationPlanner
        from mdp import ValueIterationPlanner
        test = ValueIterationPlanner(self.env, self.result_dir)

        next_diretion = [(self.actions[0]), (self.actions[1]), (self.actions[2]), (self.actions[3])]
        y_n = False
        pre, Node, Arc, Arc_sum, PERMISSION = self.refer.reference()
        # 今はここに入ってbp_algorithmに遷移している
        if self.NODELIST[state.row][state.column] in pre:
                self.trigar = False
        elif self.NODELIST[state.row][state.column] == "x":
            # print("========\n交差点\n========")
            self.trigar = False
            self.TRIGAR_advance = False
        else:
            self.trigar = None
        # print("========\n探索開始\n========")
        exp_action = []
        for dir in next_diretion:
            y_n, action = self.env.expected_move(state, dir, self.trigar, self.All, self.marking_param)
            if y_n:
                y_n = False
                exp_action.append(action)

        if self.NODELIST[state.row][state.column] in pre or self.NODELIST[state.row][state.column] == "x":
            
            size = self.env.row_length
            m = [[1.0 for i in range(size)] for i in range(size)] #known or unknown
            for i in range(-1,2):
                if state.row+i < 0 or state.row+i >=size:
                    continue
                for j in range(-1,2):
                    if state.column+j < 0 or state.column+j >=size:
                        continue
                    
                    m[state.row+i][state.column+j] = 0
            states_known = set() #empty set
            for s in self.env.states:
                    if m[s.row][s.column] == 0:
                        states_known.add(s)

            if not exp_action:
                self.TRIGAR_advance = True
                exp_action = next_diretion # 全方向探索した場合は選択方向を初期化する

            # a, v, aaa = test.plan(states_known, map, state, DIR, exp_action) # here
            a, v, aaa = test.plan(states_known, m, state, DIR, exp_action) # here
            # a, v, aaa = test.plan(states_known2, m, state_init, DIR,     exp_action)
            
            new_pi = (aaa[state]) # here
            
            # new_pi = (aaa[state_init])

            
            # print("===== NEW PI =====\n", new_pi)
            action = new_pi
            test.show(aaa, v, state, m, DIR,     self.VIZL, self.VIZD, STATE_HISTORY)
            # test.show_values(a)
            
            # self.env.mark(state, None)
        
        else:
            "進める方向に継続して進む"
            if not exp_action:
                self.TRIGAR_advance = True
            else:
                action = exp_action[0]
                # print("===== NEW PI =====\n", action)

        return action, self.Reverse, self.TRIGAR_advance

    def policy_bp(self, state, TRIGAR, TRIGAR_REVERSE, COUNT):
        self.TRIGAR_bp = TRIGAR
        self.TRIGAR_REVERSE_bp = TRIGAR_REVERSE
        self.All = False
        self.Reverse = False
        self.COUNT = COUNT

        try:
            action, self.Reverse = self.model_bp(state)
        except:
        # except Exception as e:
        #     print('=== エラー内容 ===')
        #     print('type:' + str(type(e)))
        #     print('args:' + str(e.args))
        #     print('message:' + e.message)
        #     print('e自身:' + str(e))
            # print("agent / policy_bp ERROR")

            "動いていない時に迷ったとする場合"
            # if NOT_MOVE:
            #     self.All = True
            
            # これのおかげで沼でも少し動けている
            return random.choice(self.actions), self.Reverse, self.lost
            
        return action, self.Reverse , self.lost

    def policy_exp(self, state, TRIGAR):
        self.trigar = TRIGAR
        attribute = self.NODELIST[state.row][state.column]
        next_direction = random.choice(self.actions)
        self.All = False
        bp = False
        self.lost = False
        self.Reverse = False
        
        try:
            y_n, action, bp = self.model_exp(state)
        except:
            # self.All = True
            return self.actions[1], bp, self.All, self.trigar, self.Reverse, self.lost
        return action, bp, self.All, self.trigar, self.Reverse, self.lost

    def model_exp(self, state, TRIGAR):
        self.trigar = TRIGAR
        next_diretion = [(self.actions[0]), (self.actions[1]), (self.actions[2]), (self.actions[3])]
        y_n = False
        bp = False
        pre, Node, Arc, Arc_sum, PERMISSION = self.refer.reference()

        # 今はここに入ってbp_algorithmに遷移している
        if self.NODELIST[state.row][state.column] in pre:
                # print("========\n探索終了\n========")
                self.trigar = False
                bp = True
        elif self.NODELIST[state.row][state.column] == "x":
            # print("========\n交差点\n========")
            self.trigar = False

        exp_action = []
        for dir in next_diretion:
            y_n, action = self.env.expected_move(state, dir, self.trigar, self.All, self.marking_param)
            
            if y_n:
                y_n = False
                exp_action.append(action)
                
        if exp_action:
            
            # if self.NODELIST[state.row][state.column] in pre: # "x":
            "Edit"
            if self.NODELIST[state.row][state.column] in pre or self.NODELIST[state.row][state.column] == "x": # here
                # print("========\n交差点\n========")
                Average_Value = self.decision_action.value(exp_action)
                # print("\n===================\n🤖⚡️ Average_Value:{}".format(Average_Value))
                # print(" == 各行動後にストレスが減らせる確率:{}".format(Average_Value))
                # print(" == つまり、新しい情報が得られる確率:{} -----> これが一番重要・・・未探索かつこの数値が大きい方向の行動を選択\n===================\n".format(Average_Value))
                action_value = self.decision_action.policy(Average_Value)
                if action_value == self.env.actions[2]: #  LEFT:
                    NEXT = "LEFT  ⬅️"
                    # print("    At :-> {}".format(NEXT))
                if action_value == self.env.actions[3]: # RIGHT:
                    NEXT = "RIGHT ➡️"
                    # print("    At :-> {}".format(NEXT))  
                if action_value == self.env.actions[0]: #  UP:
                    NEXT = "UP    ⬆️"
                    # print("    At :-> {}".format(NEXT))
                if action_value == self.env.actions[1]: # DOWN:
                    NEXT = "DOWN  ⬇️"
                    # print("    At :-> {}".format(NEXT))

                # print("過去のエピソードから、現時点では、🤖⚠️ At == {}を選択する".format(action_value))
                Episode_0 = self.decision_action.save_episode(action_value)
            else:
                action_value = exp_action[0]
            y_n = True
            return y_n, action_value, bp
            
        if not bp:
            # print("==========\nこれ以上進めない状態\n or 次のマスは探索済み\n==========") # どの選択肢も y_n = False
            self.lost = True
        else:
            self.All = True

        # print("==========\n迷った状態\n==========") # どの選択肢も y_n = False
        # print("= 現在地からゴールに迎える選択肢はない\n")

    def mdp_exp(self, state, TRIGAR,     states_known, map, grid, DIR,     VIZL, VIZD, STATE_HISTORY):
        self.TRIGAR_advance = TRIGAR
        self.VIZL = VIZL
        self.VIZD = VIZD
        
        # from mdp_virtual import ValueIterationPlanner
        from mdp import ValueIterationPlanner

        
        test = ValueIterationPlanner(self.env, self.result_dir)

        next_diretion = [(self.actions[0]), (self.actions[1]), (self.actions[2]), (self.actions[3])]
        self.trigar = TRIGAR
        self.All = False
        self.lost = False
        self.Reverse = False
        y_n = False
        bp = False
        pre, Node, Arc, Arc_sum, PERMISSION = self.refer.reference()

        # 今はここに入ってbp_algorithmに遷移している
        if self.NODELIST[state.row][state.column] in pre:
                # print("========\n探索終了\n========")
                self.trigar = False
                bp = True
        elif self.NODELIST[state.row][state.column] == "x":
            # print("========\n交差点\n========")
            self.trigar = False
        # else:
        #     self.trigar = None
        # print("========\n探索開始\n========")
        exp_action = []
        for dir in next_diretion:
            y_n, action = self.env.expected_move(state, dir, self.trigar, self.All, self.marking_param)
            if y_n:
                y_n = False
                exp_action.append(action)

        if exp_action:

            # print("Unexplore:", exp_action)

            if self.NODELIST[state.row][state.column] in pre or self.NODELIST[state.row][state.column] == "x": # here
                size = self.env.row_length
                m = [[1.0 for i in range(size)] for i in range(size)] #known or unknown
                for i in range(-1,2):
                    if state.row+i < 0 or state.row+i >=size:
                        continue
                    for j in range(-1,2):
                        if state.column+j < 0 or state.column+j >=size:
                            continue
                        
                        m[state.row+i][state.column+j] = 0
                states_known = set() #empty set
                for s in self.env.states:
                        if m[s.row][s.column] == 0:
                            states_known.add(s)
                # print("Unexplore:", exp_action)
                # if not exp_action:
                #     self.TRIGAR_advance = True
                #     exp_action = next_diretion # 全方向探索した場合は選択方向を初期化する

                # a, v, aaa = test.plan(states_known, map, state, DIR, exp_action) # here
                a, v, aaa = test.plan(states_known, m, state, DIR, exp_action) # here

                # a, v, aaa = test.plan(states_known2, m, state_init, DIR,     exp_action)
                
                new_pi = (aaa[state]) # here
                
                # new_pi = (aaa[state_init])

                
                # print("===== NEW PI =====\n", new_pi)
                action = new_pi
                test.show(aaa, v, state, m, DIR, self.VIZL, self.VIZD, STATE_HISTORY)
                # test.show_values(a)

                
                # self.env.mark(state, None)

                
            else:
                "進める方向に継続して進む"
                action = exp_action[0]
            
            return action, bp, self.All, self.trigar, self.Reverse, self.lost
            

        if not bp:
            # print("==========\nこれ以上進めない状態\n or 次のマスは探索済み\n==========") # どの選択肢も y_n = False
            self.lost = True
        else:
            self.All = True

        # print("==========\n迷った状態\n==========") # どの選択肢も y_n = False
        # print("= 現在地からゴールに迎える選択肢はない\n")

        return action, bp, self.All, self.trigar, self.Reverse, self.lost

    def model_advance(self, state):

        next_diretion = [(self.actions[0]), (self.actions[1]), (self.actions[2]), (self.actions[3])]

        pre, Node, Arc, Arc_sum, PERMISSION = self.refer.reference()
        if self.NODELIST[state.row][state.column] in pre:
            # print("ランダムに決定")
            next_diretion = self.advance_direction_decision(next_diretion)
        else:
            next_diretion = [(self.actions[0]), (self.actions[1]), (self.actions[2]), (self.actions[3])]
        # advanceの行動の優先度をあらかじめ設定

        if self.NODELIST[state.row][state.column] == "x":
            # print("ランダムに決定")
            next_diretion = self.advance_direction_decision(next_diretion)
        y_n = False
        self.All = False
        self.Reverse = False

        if self.NODELIST[state.row][state.column] == "x":
            # print("========\n交差点\n========")
            self.TRIGAR_advance = False

        # print("========\nAdvance開始\n========")
        if not self.TRIGAR_advance:
            for dir in next_diretion:
                y_n, action = self.env.expected_move(state, dir, self.TRIGAR_advance, self.All, self.marking_param)

                if y_n:
                    self.prev_action = action
                    return action
                # print("y/n:{}".format(y_n))
        # print("==========\n迷った【許容を超える】状態\n==========") # どの選択肢も y_n = False
        # print("= これ以上先に現在地からゴールに迎える選択肢はない\n= 一旦体制を整える\n= 戻る")
        # print("\n というよりはストレスが溜まり切る前にこれ以上進めなくなってエラーが出る")
        self.TRIGAR_advance = True

    def model_bp(self, state):

        pre, Node, Arc, Arc_sum, PERMISSION = self.refer.reference()

        # print("========\nBACK 開始\n========")
        # print("TRIGAR : {}".format(self.TRIGAR_bp))
        # print("REVERSE : {}".format(self.TRIGAR_REVERSE_bp))
        
        if self.TRIGAR_REVERSE_bp:
            self.Reverse = True
            next_diretion = self.next_direction_decision("reverse")
            for dir in next_diretion:
                y_n, action = self.env.expected_move_return_reverse(state, dir, self.TRIGAR_REVERSE_bp, self.Reverse)

                if y_n:
                    self.lost = False
                    return action, self.Reverse
                # print("y/n:{}".format(y_n))
            # print("TRIGAR REVERSE ⚡️🏁")

        if self.TRIGAR_bp:
            next_diretion = self.next_direction_decision("trigar")

            for dir in next_diretion:
                y_n, action = self.env.expected_move_return(state, dir, self.TRIGAR_bp, self.All)

                if y_n:
                    self.lost = False
                    return action, self.Reverse

            if self.lost:
                # print("==========\nこれ以上戻れない状態\n or 次のマスは以前戻った場所\n==========") # どの選択肢も y_n = False
                for dir in next_diretion:
                    y_n, action = self.env.expected_not_move(state, dir, self.trigar, self.All)

                    if y_n:
                        return action, self.Reverse
                        
        # print("==========\n戻り終わった状態\n==========") # どの選択肢も y_n = False
        # print("= 現在地から次にゴールに迎える選択肢を選ぶ【未探索方向】\n")
        self.lost = True

    def back_position(self, test_bp_st, Attribute): # change

        max_num = Attribute["move cost"].max(skipna=True)
        Attribute["move cost"] = Attribute["move cost"]/max_num # 正規化
        max_s = Attribute["stress"].max(skipna=True)
        Attribute["stress"] = Attribute["stress"]/max_s # 正規化
        Attribute["PRODUCT"] = Attribute["stress"]*Attribute["move cost"]
        next_lm_index = Attribute["PRODUCT"].idxmin()
        min_Index = Attribute["PRODUCT"][Attribute["PRODUCT"] == Attribute["PRODUCT"].min()].index.values
        min_cost = Attribute["move cost"].idxmin()

        if len(min_Index) > 1:
            # print("min Index が複数個あります。")
            # print("min Index :", next_lm_index)
            # "random ver."
            # # min_Index = [random.choice(min_Index)] # 今はランダムだが move cost が小さい方にする
            # "min cost ver."
            min_Index = [x for x in min_Index if x == min_cost]
            # print("move cost の小さいインデックスを選択 ... min Index == min cost :", min_Index)
            next_lm_index = min_Index[0]
            # print(next_lm_index)

        Attribute["STATE"] = test_bp_st
        next_attribute = Attribute.loc[next_lm_index:next_lm_index]

        return next_attribute

    def back_end(self, Attribute, next_attribute):

        LM = next_attribute.index[0]

        "戻ったNode = NaNにする"
        Attribute.loc[LM:LM] = np.nan # here コメントアウトしてもretryする時は初期化するので意味ない
        
        return Attribute

    def next_direction_decision(self, trigar__or__reverse):
        if self.Advance_action == self.actions[0]: # Action.UP:
            self.BP_action = self.actions[1] # [0]
            next_diretion_trigar = [(self.actions[1]), (self.actions[0]), (self.actions[2]), (self.actions[3])]
            next_diretion_trigar_reverse = [(self.actions[0]), (self.actions[1]), (self.actions[2]), (self.actions[3])]
        elif self.Advance_action == self.actions[1]: # Action.DOWN:
            self.BP_action = self.actions[0] # [1]
            next_diretion_trigar = [(self.actions[0]), (self.actions[1]), (self.actions[2]), (self.actions[3])]
            next_diretion_trigar_reverse = [(self.actions[1]), (self.actions[0]), (self.actions[2]), (self.actions[3])]
        elif self.Advance_action == self.actions[2]: # Action.LEFT:
            self.BP_action = self.actions[3] # [2]
            next_diretion_trigar = [(self.actions[3]), (self.actions[2]), (self.actions[0]), (self.actions[1])]
            next_diretion_trigar_reverse = [(self.actions[2]), (self.actions[3]), (self.actions[0]), (self.actions[1])]
        elif self.Advance_action == self.actions[3]: # Action.RIGHT:
            self.BP_action = self.actions[2] # [3]
            next_diretion_trigar = [(self.actions[2]), (self.actions[3]), (self.actions[0]), (self.actions[1])]
            next_diretion_trigar_reverse = [(self.actions[3]), (self.actions[2]), (self.actions[0]), (self.actions[1])]
        else:
            next_diretion_trigar, next_diretion_trigar_reverse = self.next_direction_decision_prev_action()

        if trigar__or__reverse == "trigar":
            # print("tigar__or__reverse : {}".format(trigar__or__reverse))
            return next_diretion_trigar
        if trigar__or__reverse == "reverse":
            # print("tigar__or__reverse : {}".format(trigar__or__reverse))
            return next_diretion_trigar_reverse

    def next_direction_decision_prev_action(self):
        if self.prev_action == self.actions[0]: # Action.UP:
            self.BP_action = self.actions[1]
            next_diretion_trigar = [(self.actions[1]), (self.actions[0]), (self.actions[2]), (self.actions[3])]
            next_diretion_trigar_reverse = [(self.actions[0]), (self.actions[1]), (self.actions[2]), (self.actions[3])]
        elif self.prev_action == self.actions[1]: # Action.DOWN:
            self.BP_action = self.actions[0]
            next_diretion_trigar = [(self.actions[0]), (self.actions[1]), (self.actions[2]), (self.actions[3])]
            next_diretion_trigar_reverse = [(self.actions[1]), (self.actions[0]), (self.actions[2]), (self.actions[3])]
        elif self.prev_action == self.actions[2]: # Action.LEFT:
            self.BP_action = self.actions[3]
            next_diretion_trigar = [(self.actions[3]), (self.actions[2]), (self.actions[0]), (self.actions[1])]
            next_diretion_trigar_reverse = [(self.actions[2]), (self.actions[3]), (self.actions[0]), (self.actions[1])]
        elif self.prev_action == self.actions[3]: # Action.RIGHT:
            self.BP_action = self.actions[2]
            next_diretion_trigar = [(self.actions[2]), (self.actions[3]), (self.actions[0]), (self.actions[1])]
            next_diretion_trigar_reverse = [(self.actions[3]), (self.actions[2]), (self.actions[0]), (self.actions[1])]

        return next_diretion_trigar, next_diretion_trigar_reverse

    def advance_direction_decision(self, dir):

        test = random.sample(dir, len(dir))
        # print("test dir : {}, dir : {}".format(test, dir))
        return test # random.shuffle(dir)
        #  [<Action.RIGHT: -2>, <Action.DOWN: -1>, <Action.UP: 1>, <Action.LEFT: 2>]