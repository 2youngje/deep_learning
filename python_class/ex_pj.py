import os
import numpy as np
import csv
import pandas as pd


def game_main(char):
    slime = Slime()
    while True:
        select_option = int(input("\n\n다음 중 어떤 것을 하시겠습니까?\n"
                                  "1. 몬스터 잡기\n"
                                  "2. 현재 상태 확인\n"
                                  "3. 물약사기 (30원)\n"
                                  "4. 게임 저장하기\n"
                                  "0. 게임 종료\n"
                                  ))

        if select_option == 1:
            slime = char.attack_monster(slime)

        elif select_option == 3:
            char.buy_portion()

        elif select_option == 2:
            char.print_state()

        elif select_option == 4:
            char.save_states()

        elif select_option == 0:
            print("===== 게임 종료 =====")
            break


class Character:
    # Step.2 캐릭터 클래스의 초깃값
    def __init__(self) -> None:
        self.lv = 1
        self.exp = 0
        self.HP = 100
        self.max_HP = 100
        self.damage = 10
        self.money = 100

    # Step.3 캐릭터의 정보출력 기능
    def print_state(self):
        print("-" * 10, "캐릭터 상태", "-" * 10)
        print("현재 레벨 :", self.lv)
        print("현재 경험치 :", self.exp)
        print("다음 레벨을 위한 경험치 :", self.lv * 100)
        print("HP :", self.HP)
        print("■" * int(self.HP / 5), "※" * int((self.max_HP - self.HP) / 5))
        print("HP 최대치 :", self.max_HP)
        print("공격력:", self.damage)
        print("돈 :", self.money)

    # Step.4 캐릭터 저장 기능
    def save_states(self):
        import pandas as pd

        data = {"lv": [self.lv],
                "exp": [self.exp],
                "HP": [self.HP],
                "max_HP": [self.max_HP],
                "damage": [self.damage],
                "money": [self.money]}

        file = pd.DataFrame(data)
        save_file = file.T

        save_file.to_csv("./save_file.csv")

    # Step.5 캐릭터 불러오기 기능
    def load_states(self):
        game_history = pd.read_csv("./save_file.csv", index_col=0)
        self.lv = game_history.iloc[0, 0]
        self.exp = game_history.iloc[1, 0]
        self.HP = game_history.iloc[2, 0]
        self.max_HP = game_history.iloc[3, 0]
        self.damage = game_history.iloc[4, 0]
        self.money = game_history.iloc[5, 0]

    # Step.10 HP getter, setter 만들기
    def get_HP(self):
        return self.HP

    def set_HP(self, after_ch_hp):
        self.HP = after_ch_hp

        # Step.11 몬스터 공격하는 기능 만들기

    def attack_monster(self, slime):
        after_hp = slime.get_HP() - 10
        slime.set_HP(after_hp)
        after_ch_hp = self.get_HP() - np.random.choice([0, slime.damage])
        self.set_HP(after_ch_hp)
        if slime.get_HP() <= 0:
            print("\n\n 경험치", slime.kill_exp, ", 돈", slime.kill_money, "획득!\n", "적을 잡았습니다.")
            self.exp += 50
            if self.exp == self.lv * 100:
                self.lv += 1
                self.max_HP += 10
                self.damage += 3
                self.money += slime.kill_money
                self.HP = self.max_HP
        return slime

    def buy_portion(self):
        self.money -= 30
        self.HP += 50
        print("\n물약을 섭취하여 HP가 상승하였습니다.")


# Step.9 슬라임 클래스 만들기
class Slime:
    def __init__(self) -> None:
        self.HP = 30
        self.damage = 5
        self.kill_exp = 50
        self.kill_money = 10

    # Step.10 HP getter, setter 만들기
    def get_HP(self):
        return self.HP

    def set_HP(self, after_hp):
        self.HP = after_hp

    # Step.1 게임 시작화면 만들기


while True:
    next_move = int(input("\nSeSac 온라인 게임에 오신 것을 환영합니다.\n\n"
                          "1. 새로운 게임 시작하기\n"
                          "2. 지난 게임 불러오기\n"
                          "3. 게임 종료하기\n"
                          "다음 중 어떤 것을 하시겠습니까?"))
    if next_move == 1:
        print("\n새로운 캐릭터를 생성합니다.")
        char = Character()
        char.print_state()
        game_main(char)

        break

    elif next_move == 2:
        if os.path.exists('save_file.csv'):
            print("\n저장된 파일을 불러옵니다.")
            char = Character()
            char.load_states()
            char.print_state()
            game_main(char)
            break

        else:
            print("\n저장된 파일이 없습니다. 메인 화면으로 돌아갑니다.")
            continue

    elif next_move == 3:
        print("\n게임을 종료합니다.")
        break