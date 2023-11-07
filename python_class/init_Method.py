class Person:
    def __init__(self,name):
        self.name = name
        self.intro()

    def intro(self):
        intro = "hello I'm "+self.name
        return print(intro)

person1 = Person('lee')
person2 = Person('young')

person1
person2