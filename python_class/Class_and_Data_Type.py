class TestClass : # CamelCase
    pass # -> 나중에 구현을 완성하겠다.

object1 = TestClass()
object2 = TestClass()

print("object1 : ", type(object1))
print("object2 : ", type(object2))

class Person:
    def say_hello(self):
        print("Hello!")

    def say_bye(self):
        print("Goodbye!")

person1,person2 = Person(),Person()
print(person1.say_hello())
print(person1.say_bye())

print(person2.say_hello())
print(person2.say_bye())