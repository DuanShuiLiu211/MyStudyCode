class StateMachine:
    def __init__(self):
        self.states = {}
        self.current_state = None
        self.is_running = False

    def add_state(self, name, action):
        self.states[name] = action

    def set_start(self, name):
        self.current_state = name

    def run(self):
        self.is_running = True
        while self.is_running:
            if self.current_state not in self.states:
                print(f"错误: 未知状态 '{self.current_state}'")
                break
            action = self.states[self.current_state]
            self.current_state = action()

    def stop(self):
        self.is_running = False


# 示例状态机的具体实现
def start_state():
    print("这是开始状态")
    user_input = input("输入 'a' 进入状态1，输入 'b' 进入状态2: ")
    if user_input == "a":
        return "state1"
    elif user_input == "b":
        return "state2"
    else:
        print("无效输入，重新开始")
        return "start"


def state1():
    print("这是状态1")
    return "state2"


def state2():
    print("这是状态2")
    user_input = input("输入 'end' 结束状态机，输入 'back' 返回状态1: ")
    if user_input == "end":
        return "end"
    elif user_input == "back":
        return "state1"
    else:
        print("无效输入，重新进入状态2")
        return "state2"


def end_state():
    print("这是结束状态")
    return None


if __name__ == "__main__":
    sm = StateMachine()
    sm.add_state("start", start_state)
    sm.add_state("state1", state1)
    sm.add_state("state2", state2)
    sm.add_state("end", end_state)

    sm.set_start("start")
    sm.run()
