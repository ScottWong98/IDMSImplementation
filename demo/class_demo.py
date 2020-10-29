class Greeter:

    def __init__(self, name):
        self.name = name

    @classmethod
    def greet(cls, loud=False):
        if loud:
            print("HELLO, %s!" % cls.name.upper())
        else:
            print("hello, %s!" % cls.name)


g = Greeter('Fred')
g.greet()
