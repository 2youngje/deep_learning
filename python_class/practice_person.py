class Person:
    def set_name(self,name):
        self.name =name
    def get_name(self):
        return self.name
    def get_family_name(self):
        return self.name[0]
    def get_personal_name(self):
        return self.name[1:]
person = Person()
person.name = "김철수"

print(person.get_name())
print(person.get_family_name())
print(person.get_personal_name())