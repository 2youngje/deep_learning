import os
import pandas as pd

while True:
    next_move = int(input("\nSeSac 온라인에 오신 것을 환영합니다.\n\n"
                          "1. 새로운 게임 시작하기\n"
                          "2. 지난 게임 불러오기\n"
                          "3. 게임 종료하기\n"
                          "다음 중 어떤 것을 하시겠습니까? "))
    if next_move == 1 :
        print("\n새로운 캐릭터를 생성합니다.")
        break
    elif next_move == 2 :
        if os.path.exists('save_file.csv'):
            print("\n저장된 파일을 불러옵니다.")
            break
        else:
            print("\n저장된 파일이 없습니다. 메인 화면으로 돌아갑니다.")
            continue

    elif next_move == 3:
        print("\n게임을 종료합니다.")
        break

class Character:
    def __init__(self, lv=1, exp=0,lv_up_exp = 100, HP=100, max_HP=100, damage=10, money=100):
        self.lv = lv
        self.exp = exp
        self.lv_up_exp = lv_up_exp
        self.HP = HP
        self.max_HP = max_HP
        self.damage = damage
        self.money = money

    def calculate_next_level_exp(self):
        return self.lv * 100  # 다음 레벨을 위한 경험치 계산

    def print_states(self):
        next_level_exp = self.calculate_next_level_exp()
        print("----------")
        print(f"현재 레벨: {self.lv}")
        print(f"현재 경험치: {self.exp}")
        print(f"다음 레벨을 위한 경험치: {next_level_exp}")
        print(f"HP: {self.HP}")
        print(f"HP 최대치: {self.max_HP}")
        print(f"공격력: {self.damage}")
        print(f"돈: {self.money}")
        print("----------")

#캐릭터에 대한 다음의 save_ le.csv를 만드는 save_states 메서드를 만드세요.
    def save_states(self, filename):
        data = {
            "현재 레벨": [self.lv],
            "현재 경험치": [self.exp],
            "다음 레벨을 위한 경험치": [self.calculate_next_level_exp()],
            "HP": [self.HP],
            "HP 최대치": [self.max_HP],
            "공격력": [self.damage],
            "돈": [self.money]
        }
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)

    def load_states(self, filename):
        try:
            df = pd.read_csv(filename)
            if not df.empty:
                self.lv = df["현재 레벨"].iloc[0]
                self.exp = df["현재 경험치"].iloc[0]
                self.HP = df["HP"].iloc[0]
                self.max_HP = df["HP 최대치"].iloc[0]
                self.damage = df["공격력"].iloc[0]
                self.money = df["돈"].iloc[0]
            else:
                print("CSV 파일이 비어 있습니다.")
        except FileNotFoundError:
            print(f"파일 '{filename}'을 찾을 수 없습니다.")



    # Character 객체 생성
character = Character()

    # 캐릭터 정보를 CSV 파일에 저장
character.save_states('character_data.csv')

character.load_states('character_data.csv')

character.print_states()